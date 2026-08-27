import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import sobel
import os
import glob
import pandas as pd

def image_entropy_gray(image):
    """
    计算灰度图像的信息熵
    参数:
        image: 灰度图像 (numpy array, dtype=uint8)
    返回:
        熵值 (float)
    """
    # 计算直方图， bins=256，范围[0,256)
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist = hist.flatten()  # 变成1D数组
    # 计算概率分布
    total_pixels = image.size
    prob = hist / total_pixels
    # 过滤掉概率为0的值，避免log2(0)
    prob = prob[prob > 0]
    # 计算熵： -Σ p * log2(p)
    entropy = -np.sum(prob * np.log2(prob))
    return entropy

def keep_only_horizontal_long_texture(img_path, angle_threshold=30):
    """
    保留 水平 ±30° 以内的纹理，抑制其他所有方向纹理
    :param img_path: 灰度图 / 彩色图（会自动转灰度）
    :param angle_threshold: 角度阈值，默认 ±30°（接近水平）
    :return: 只保留水平方向纹理的二值图
    """
    # 1. 转灰度（兼容彩色图输入）
    if len(img_path.shape) == 3:
        gray = cv2.cvtColor(img_path, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_path
    
    # 2. 高斯降噪
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # ===================== 核心：梯度方向检测（计算角度）=====================
    # 计算 x、y 方向梯度
    grad_x = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)

    # 计算梯度强度 + 方向（角度）
    magnitude = cv2.magnitude(grad_x, grad_y)
    angle = cv2.phase(grad_x, grad_y, angleInDegrees=True)  # 角度 0~360°

    # ===================== 筛选：只保留 ±30° 内的水平纹理 =====================
    # 水平定义：0°/180° 左右 ±30°，即 0~30 和 150~210 和 330~360
    horizontal_mask = (
        ((angle >= 0) & (angle <= angle_threshold)) |
        ((angle >= 180 - angle_threshold) & (angle <= 180 + angle_threshold)) |
        ((angle >= 360 - angle_threshold) & (angle <= 360))
    )

    # 只保留水平方向的梯度
    horizontal_edges = np.zeros_like(magnitude)
    horizontal_edges[horizontal_mask] = magnitude[horizontal_mask]

    # 归一化到 0~255
    horizontal_edges = cv2.normalize(horizontal_edges, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # ===================== 二值化 + 形态学优化 =====================
    # OTSU 自动阈值
    _, binary = cv2.threshold(horizontal_edges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 水平形态学：强化长水平纹理
    kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    result = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_horizontal)
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel_horizontal)

    return result

def rotate_points(points, angle_deg):
    """将采样点列表绕原点旋转给定角度（度）"""
    angle_rad = np.deg2rad(angle_deg)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array([[c, -s], [s, c]])
    return [R @ p for p in points]

def compute_gradient_orientation(image):
    """计算每个像素的梯度方向（角度，单位：度，范围 -180 ~ 180）"""
    gx = sobel(image, axis=1)  # 水平方向导数
    gy = sobel(image, axis=0)  # 垂直方向导数
    orientation = np.rad2deg(np.arctan2(gy, gx))
    return orientation

def bilinear_interp(image, x, y):
    """双线性插值"""
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    x1, y1 = x0 + 1, y0 + 1
    if x1 >= image.shape[1] or y1 >= image.shape[0]:
        return image[y0, x0]
    # 四个角点像素值
    Ia = image[y0, x0]
    Ib = image[y0, x1]
    Ic = image[y1, x0]
    Id = image[y1, x1]
    # 权重
    wa = (x1 - x) * (y1 - y)
    wb = (x - x0) * (y1 - y)
    wc = (x1 - x) * (y - y0)
    wd = (x - x0) * (y - y0)           
    return wa*Ia + wb*Ib + wc*Ic + wd*Id

def orientation_adaptive_tlbp(image, P=8, R=1, use_local_orientation=True, fixed_angle=None, 
                              orientation_threshold=None, tlbp_threshold=5):
    """
    方向自适应阈值LBP（TLBP）
    - use_local_orientation: 若True，则使用每个像素的局部梯度方向；否则使用fixed_angle。
    - fixed_angle: 当use_local_orientation=False时，所有像素统一旋转到该角度（度）。
    - orientation_threshold: 若指定，则只对|局部方向 - 水平方向| < threshold的像素使用旋转采样，其余使用标准圆形LBP。
    - tlbp_threshold: TLBP核心阈值，仅当邻域像素与中心像素差值的绝对值超过该阈值时，编码为1
    """
    h, w = image.shape
    tlbp = np.zeros((h, w), dtype=np.uint8)
    
    # 预先计算标准圆形采样点（不旋转）
    std_points = []
    for i in range(P):
        theta = 2 * np.pi * i / P
        x = R * np.cos(theta)
        y = R * np.sin(theta)
        std_points.append((x, y))
    
    if use_local_orientation:
        orientations = compute_gradient_orientation(image)  # 每个像素的方向
        # 将方向映射到 [0, 180) 范围（无符号方向）
        orientations = orientations % 180
    else:
        orientations = np.full((h, w), fixed_angle % 180)
    
    # 遍历每个像素
    for i in range(1, h-1):
        for j in range(1, w-1):
            center = image[i, j]
            # 决定使用何种采样点
            if orientation_threshold is not None and use_local_orientation:
                # 只对接近水平方向（0°或180°）的区域旋转
                angle = orientations[i, j]
                if angle > 90: angle = 180 - angle  # 对称性，将钝角映射到锐角
                if angle < orientation_threshold:
                    points = rotate_points(std_points, angle)  # 旋转到局部方向
                else:
                    points = std_points
            elif use_local_orientation:
                points = rotate_points(std_points, orientations[i, j])
            else:
                points = std_points  # 无旋转
            
            # 编码TLBP（核心修改：引入阈值判断）
            binary = 0
            for k, (dx, dy) in enumerate(points):
                xi, yi = j + dx, i + dy
                if 0 <= xi < w and 0 <= yi < h:
                    val = bilinear_interp(image, xi, yi)
                else:
                    val = center
                # TLBP核心逻辑：差值绝对值超过阈值才编码为1
                if abs(val - center) > tlbp_threshold:
                    binary |= (1 << k)
            tlbp[i, j] = binary
    
    return tlbp

# ============ 批量处理主函数 ============
def batch_process_images(input_folder, output_folder, image_formats=['png', 'jpg', 'jpeg', 'bmp'],
                         tlbp_threshold=5):
    """
    批量处理文件夹中的图片
    :param input_folder: 输入图片文件夹路径
    :param output_folder: 输出结果文件夹路径
    :param image_formats: 支持的图片格式列表
    :param tlbp_threshold: TLBP阈值参数
    """
    # 创建输出文件夹（如果不存在）
    morph_save_folder = os.path.join(output_folder, 'morph_results')
    os.makedirs(morph_save_folder, exist_ok=True)
    
    # 收集所有图片路径
    image_paths = []
    for fmt in image_formats:
        image_paths.extend(glob.glob(os.path.join(input_folder, f'*.{fmt}')))

    
    if not image_paths:
        print(f"在输入文件夹 {input_folder} 中未找到任何图片")
        return
    
    # 初始化Excel数据列表
    excel_data = []
    
    # 批量处理每张图片
    for idx, img_path in enumerate(image_paths):
        # 获取图片名称
        img_name = os.path.basename(img_path)
        print(f"正在处理 [{idx+1}/{len(image_paths)}]: {img_name}")
        
        # 读取图像并转为灰度
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.medianBlur(img, 5)
        if img is None:
            print(f"⚠️  图像加载失败: {img_name}，跳过该图片")
            continue
        
        # 选择性自适应TLBP处理（替换原LBP）
        R = 1
        P = 8*R

        tlbp_selective = orientation_adaptive_tlbp(
            img, P=P, R=R, use_local_orientation=True, 
            orientation_threshold=30, tlbp_threshold=tlbp_threshold
        )
        _, tlbp = cv2.threshold(tlbp_selective, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)#临时添加
        kernel = np.ones((3, 3), dtype=np.uint8)#临时添加
        tlbp = cv2.morphologyEx(tlbp, cv2.MORPH_CLOSE, kernel)#临时添加

        
        morph = keep_only_horizontal_long_texture(tlbp_selective)

        entropy_tlbp = image_entropy_gray(tlbp_selective)
        entropy_morph = image_entropy_gray(morph)
        
        # 保存morph图
        morph_save_path = os.path.join(morph_save_folder, img_name)
        cv2.imwrite(morph_save_path, tlbp)#临时修改morph=》tlbp
        print(f"✅ Morph图已保存: {morph_save_path}")
        
        # 记录数据到Excel列表
        excel_data.append({
            '图片名称': img_name,
            'TLBP熵值(entropy)': round(entropy_tlbp, 6),  # 保留6位小数
            '最终图熵值(entropy)': round(entropy_morph, 6)  # 保留6位小数
        })
    
    # 生成Excel表格
    if excel_data:
        excel_path = os.path.join(output_folder, 'image_entropy_results.xlsx')
        df = pd.DataFrame(excel_data)
        df.to_excel(excel_path, index=False, engine='openpyxl')
        print(f"\n📊 Excel表格已生成: {excel_path}")
    else:
        print("\n❌ 没有成功处理的图片，未生成Excel表格")

# ============ 使用示例 ============
if __name__ == "__main__":

    process_model = input("是否使用多图片处理模式(0/1):")

    if process_model == "1":
        # 配置输入输出路径（请根据实际情况修改）
        INPUT_FOLDER = r"E:\VS code\data\new_data\water flow - X"  # 输入图片文件夹
        OUTPUT_ROOT_DIR = r"E:\VS code\SAM3\water_flow"  # 输出根文件夹（存放morph图+Excel）
        
        # 执行批量处理（可自定义TLBP阈值）
        batch_process_images(INPUT_FOLDER, OUTPUT_ROOT_DIR, tlbp_threshold=4)
    else:
        # 以下为可选的可视化代码（如需查看单张效果可保留）
        # 单张图片测试（可选）
        test_img_path = r"C:\Users\ShenYuLong\Desktop\Virtual_water_gauge_project\2665(1).png"#正常情况
        test_img_path = r"C:\Users\ShenYuLong\Desktop\Virtual_water_gauge_project\1256(1).png"#水流波纹
        test_img_path = r"C:\Users\ShenYuLong\Desktop\Virtual_water_gauge_project\2428(1).png"#耀斑


        img = cv2.imread(test_img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.medianBlur(img, 5)
        if img is not None:

            R = 1
            P = 8*R
            tlbp_threshold = 4  # 可调整TLBP阈值

            tlbp_selective = orientation_adaptive_tlbp(
                img, P=P, R=R, use_local_orientation=True, 
                orientation_threshold=30, tlbp_threshold=tlbp_threshold
            )

            _, tlbp = cv2.threshold(tlbp_selective, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)#临时添加
            kernel = np.ones((3, 3), dtype=np.uint8)#临时添加
            tlbp = cv2.morphologyEx(tlbp, cv2.MORPH_CLOSE, kernel)#临时添加

            morph = keep_only_horizontal_long_texture(tlbp_selective)

            entropy_tlbp = image_entropy_gray(tlbp_selective)
            entropy_morph = image_entropy_gray(morph)
            
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 4, 1)
            plt.imshow(img, cmap='gray')
            plt.title("Original")
            plt.axis('off')
            
            plt.subplot(1, 4, 2)
            plt.imshow(tlbp_selective, cmap='gray')
            plt.title(f"TLBP Selective (Entropy: {entropy_tlbp:.4f})")
            plt.axis('off')

            plt.subplot(1, 4, 3)
            plt.imshow(tlbp, cmap='gray')
            plt.title(f"TLBP MORPH_CLOSE ")
            plt.axis('off')
        
            plt.subplot(1, 4, 4)
            plt.imshow(morph, cmap='gray')
            plt.title(f"Morph (Entropy: {entropy_morph:.4f})")
            plt.axis('off')
            
            plt.tight_layout()
            plt.show()