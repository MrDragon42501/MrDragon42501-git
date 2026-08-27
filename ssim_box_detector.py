"""
SSIM差异图自动框检测器 v2
从SSIM差异图自动生成两个简单框：负向框(岸体) + 正向框(其余)

输入：image_comparison 的结果（二值化后的SSIM差异图）
输出：两个框 [(x1,y1,x2,y2,label), ...]

原理：
- 黑色区域 = 没变化 = 岸体/水面 → 负向框(排除)
- 白色区域 = 有变化 = 需要分割 → 正向框(包含)
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


class SSIMBoxDetector:
    def __init__(self):
        self.binary_image = None
        self.h, self.w = 0, 0
        self.positive_box = None   # 正向框
        self.negative_box = None   # 负向框

    def load_from_file(self, image_path):
        """
        从文件加载二值化后的SSIM差异图（image_comparison的结果）
        支持中文路径

        Args:
            image_path: 二值化SSIM差异图路径
        """
        # 使用 PIL 读取（支持中文路径）
        img = Image.open(image_path)
        self.binary_image = np.array(img)

        # 转换为灰度图
        if len(self.binary_image.shape) == 3:
            self.binary_image = cv2.cvtColor(self.binary_image, cv2.COLOR_RGB2GRAY)

        if self.binary_image is None:
            raise ValueError(f"无法读取图片: {image_path}")

        self.h, self.w = self.binary_image.shape
        print(f"加载SSIM差异图成功: {self.w} x {self.h}")

    def load_from_array(self, binary_array):
        """
        从numpy数组加载二值化SSIM差异图

        Args:
            binary_array: 二值化数组 (H, W)
        """
        self.binary_image = binary_array.copy()
        if len(self.binary_image.shape) == 3:
            self.binary_image = cv2.cvtColor(self.binary_image, cv2.COLOR_RGB2GRAY)

        self.h, self.w = self.binary_image.shape
        print(f"加载SSIM差异图成功: {self.w} x {self.h}")

    def detect_two_boxes(self):
        """
        自动检测两个框

        【正负框说明 - 用户确认版】
        ┌─────────────────────────────────────────────────────────────┐
        │ SSIM差异图分析：                                              │
        │   - 黑色区域 = 岸体(Bank) = 要分割！                          │
        │   - 白色区域 = 水面(Water) = 排除                             │
        │                                                              │
        │ 框分配：                                                      │
        │   - POS (positive_box) = 黑色 = 岸体 = label=True             │
        │   - NEG (negative_box) = 白色 = 水面 = label=False            │
        │                                                              │
        │ SAM3 label语义：                                              │
        │   - label=True  = 正向框 = 包含/分割该区域                     │
        │   - label=False = 负向框 = 排除该区域                         │
        └─────────────────────────────────────────────────────────────┘
        """
        if self.binary_image is None:
            raise ValueError("请先加载SSIM差异图")

        # 确保是灰度图
        if len(self.binary_image.shape) == 3:
            gray = cv2.cvtColor(self.binary_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.binary_image.copy()

        h, w = gray.shape
        
        # ============ 分析图像方向 ============
        # 判断是横向还是竖向
        is_horizontal = w > h
        
        if is_horizontal:
            # 横向图像：黑白在左右方向
            mid_y = h // 2
            col_means = np.mean(gray, axis=1)
            mid_row = col_means
            
            # 找左右边界
            black_cols = mid_row < 100
            if np.any(black_cols):
                roi_left = np.argmax(black_cols)
                roi_right = len(black_cols) - 1 - np.argmax(black_cols[::-1])
            else:
                roi_left, roi_right = 0, w - 1
            
            # 找黑白分界线
            mid_x = (roi_left + roi_right) // 2
            col = gray[:, mid_x]
            
            # 【关键】找黑色区域（岸体）和白色区域（水面）的边界
            black_mask = col < 100
            white_mask = col > 100
            
            if np.any(white_mask):
                boundary_y = np.argmax(white_mask)  # 第一个白色位置
            else:
                boundary_y = h // 2
            
            margin = 0
            # 【修正】黑色=岸体=POS，白色=水面=NEG
            self.positive_box = (roi_left + margin, boundary_y, roi_right - margin, h - margin)  # 右侧黑色(岸体)
            self.negative_box = (roi_left + margin, margin, roi_right - margin, boundary_y)  # 左侧白色(水面)
            
            print(f"[横向图像 {w}x{h}]")
            print(f"  黑色(岸体): X={boundary_y} - {w}")
            print(f"  白色(水面): X={margin} - {boundary_y}")
            
        else:
            # 竖向图像：黑白在上下方向
            # 【关键】黑色=岸体=POS，白色=水面=NEG
            # 按上方黑色部分和其他部分直接分为正框和负框
            
            # 计算行均值来分析
            row_means = np.mean(gray, axis=1)  # (h,) 每一行的均值
            
            # 找第一行白色出现的位置（黑白分界线）
            # 白色阈值: > 100
            boundary_y = h  # 默认全黑
            for y in range(h):
                if row_means[y] > 100:  # 第一行白色
                    boundary_y = y
                    break
            
            # 如果没找到白色，找灰度>50的位置
            if boundary_y == h:
                for y in range(h):
                    if row_means[y] > 50:
                        boundary_y = y
                        break
            
            margin = 0
            # 【按用户需求】直接按黑白分界线划分：
            # - 上方黑色部分 = 正向框(POS) = 岸体
            # - 下方其他部分 = 负向框(NEG) = 水面
            self.positive_box = (margin, margin, w - margin, boundary_y)  # 顶部黑色(岸体)
            self.negative_box = (margin, boundary_y, w - margin, h - margin)  # 下方其他(水面)
            
            print(f"[竖向图像 {w}x{h}]")
            print(f"  黑白分界线: Y={boundary_y}")
            print(f"  上方黑色(岸体): Y={margin} - {boundary_y} → POS")
            print(f"  下方其他(水面): Y={boundary_y} - {h} → NEG")
        
        print(f"  POS(黑色/岸体): {self.positive_box} [label=True]")
        print(f"  NEG(白色/水面): {self.negative_box} [label=False]")

        return self.get_boxes()

    def get_boxes(self):
        """
        获取两个框（用于SAM3分割）

        Returns:
            boxes: [(x1, y1, x2, y2, label), ...]
                - POS(岸体/黑色) = label=True  = 包含/分割
                - NEG(水面/白色) = label=False = 排除
        """
        boxes = []

        if self.positive_box:  # 黑色区域 = 岸体
            x1, y1, x2, y2 = self.positive_box
            boxes.append((x1, y1, x2, y2, True))   # 包含/分割岸体

        if self.negative_box:  # 白色区域 = 水面
            x1, y1, x2, y2 = self.negative_box
            boxes.append((x1, y1, x2, y2, False))  # 排除水面

        return boxes

    def visualize(self, save_path=None):
        """
        可视化检测结果 - 完整图像 + 红绿框标注

        【颜色说明】
        ┌──────────────────────────────────────┐
        │ POS (Bank/岸体/黑色) = 绿色框          │
        │ NEG (Water/水面/白色) = 红色框         │
        └──────────────────────────────────────┘

        Args:
            save_path: 保存路径
        """
        if self.binary_image is None:
            raise ValueError("请先加载SSIM差异图")

        result = cv2.cvtColor(self.binary_image, cv2.COLOR_GRAY2BGR)

        # 画负向框（红色）- NEG = 白色 = 水面
        if self.negative_box:
            x1, y1, x2, y2 = self.negative_box
            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(result, "NEG (Water)", (x1 + 5, y1 + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # 画正向框（绿色）- POS = 黑色 = 岸体
        if self.positive_box:
            x1, y1, x2, y2 = self.positive_box
            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(result, "POS (Bank)", (x1 + 5, y1 + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.title("SSIM Box Detection\n(Red=Negative/Bank, Green=Positive/Water)")
        plt.axis('off')

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"可视化已保存: {save_path}")

        plt.close()
        return result


# ================================================
# 示例使用
# ================================================
if __name__ == "__main__":
    # 使用 image_comparison 生成的二值化SSIM差异图
    SSIM_DIFF_PATH = r"C:\Users\71758\Desktop\flood_results\test_ssim.png"

    try:
        # 1. 创建检测器
        detector = SSIMBoxDetector()

        # 2. 加载SSIM差异图（image_comparison的结果）
        detector.load_from_file(SSIM_DIFF_PATH)

        # 3. 检测两个框（自动检测）
        boxes = detector.detect_two_boxes()

        # 4. 可视化
        detector.visualize(save_path=r"D:\sam3model\ssim_output\box_detection.png")

        # 5. 输出结果
        print("\n" + "=" * 50)
        print("自动检测结果：")
        print("=" * 50)
        for i, (x1, y1, x2, y2, label) in enumerate(boxes):
            box_type = "正向框" if label else "负向框"
            print(f"  {box_type}: [{x1}, {y1}, {x2}, {y2}]")

    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
