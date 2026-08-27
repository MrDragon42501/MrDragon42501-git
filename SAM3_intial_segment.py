import torch
import matplotlib

# 设置matplotlib后端为非交互式
matplotlib.use('Agg')  # 这行必须在导入pyplot之前

import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import numpy as np
import shutil
import os
import pandas as pd
from datetime import datetime
import cv2
import sys
sys.path.append(r"E:\VS code\SAM3")  # 添加模型路径
#################################### For Image ####################################
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import plot_results


def clear_output_folder(folder_path):
    """清空输出文件夹"""
    if os.path.exists(folder_path):
        response = input(f"是否清空输出文件夹 '{folder_path}' 中的所有文件？(y/n): ").lower().strip()
        if response == 'y' or response == 'yes':
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f'删除 {file_path} 失败: {e}')
            print("输出文件夹已清空")
        else:
            print("保留原有文件")
    else:
        print(f"输出文件夹 '{folder_path}' 不存在，将创建新文件夹")


def delete_existing_excel_files(folder_path, patterns=['水位线结果_*.xlsx', '失败图片_*.xlsx']):
    """删除文件夹中符合模式的Excel文件"""
    for pattern in patterns:
        for file_path in Path(folder_path).glob(pattern):
            try:
                os.remove(file_path)
                print(f"删除旧文件: {file_path.name}")
            except Exception as e:
                print(f"删除 {file_path} 失败: {e}")


def calculate_waterline_by_percentage(masks, image_height, threshold_percent=60, min_continuous_rows=3):
    """
    从上往下搜索，找到第一个满足连续行水体占比超过阈值的区域作为水位线

    Args:
        masks: 分割掩码 [num_masks, H, W] 或 [H, W]
        image_height: 图像高度
        threshold_percent: 阈值百分比（默认60%）
        min_continuous_rows: 最小连续行数（默认3行）

    Returns:
        waterline_y: 水位线的y坐标，如果没有找到则返回None
    """
    if masks is None:
        return None, None

    if isinstance(masks, torch.Tensor):
        masks = masks.cpu().numpy()

    if masks.dtype == np.float32 or masks.dtype == np.float64:
        masks = masks > 0.5

    if len(masks.shape) == 3:
        combined_mask = np.any(masks, axis=0)
    elif len(masks.shape) == 2:
        combined_mask = masks
    elif len(masks.shape) == 4:
        if masks.shape[1] == 1:
            masks = masks.squeeze(1)
            combined_mask = np.any(masks, axis=0)
        else:
            combined_mask = np.any(masks, axis=0)
    else:
                return None, None

    if len(combined_mask.shape) != 2:
        while len(combined_mask.shape) > 2:
            combined_mask = np.squeeze(combined_mask)
        if len(combined_mask.shape) != 2:
            return None

    if not np.any(combined_mask):
        return None

    h, w = combined_mask.shape
    total_pixels = w
    threshold_count = int(total_pixels * threshold_percent / 100)

    print(
        f"  搜索水位线: 阈值={threshold_percent}% ({threshold_count}/{total_pixels}像素), 最小连续行={min_continuous_rows}")

    found_y = None
    continuous_count = 0

    for y in range(h):
        row_mask_count = np.sum(combined_mask[y, :])
        if row_mask_count >= threshold_count:
            continuous_count += 1
            if continuous_count >= min_continuous_rows:
                found_y = y - continuous_count + 1
                print(f"  找到水位线: y={found_y}, 从y={found_y}开始连续{continuous_count}行达到阈值")
                break
        else:
            continuous_count = 0

    if found_y is None:
        print(f"  未找到连续{min_continuous_rows}行满足阈值{threshold_percent}%的区域")
        for y in range(h):
            row_mask_count = np.sum(combined_mask[y, :])
            if row_mask_count >= threshold_count:
                found_y = y
                percentage = (row_mask_count / total_pixels) * 100
                print(f"  找到第一行超过阈值: y={y}, 掩码占比={percentage:.1f}%")
                break

        if found_y is None:
            y_coords = np.where(combined_mask)[0]
            if len(y_coords) > 0:
                found_y = np.max(y_coords)
                print(f"  使用掩码最底部: y={found_y}")
            else:
                return None

    # 计算掩码最高点
    all_y = np.where(combined_mask)[0]
    mask_top_y = int(np.min(all_y)) if len(all_y) > 0 else 0
    return int(found_y), mask_top_y


def load_ground_truth_waterline(gt_file_path):
    """加载真实水位线数据"""
    try:
        df = pd.read_excel(gt_file_path)
        print(f"读取真实水位线文件: {gt_file_path}")
        print(f"文件列名: {df.columns.tolist()}")
        print(f"前5行数据预览:")
        print(df.head())

        gt_dict_exact = {}
        gt_dict_stem = {}
        gt_dict_numeric = {}
        gt_dict_with_zero = {}

        for _, row in df.iterrows():
            img_name = str(row.iloc[0])
            waterline = row.iloc[1]

            try:
                waterline = float(waterline)
            except:
                waterline = None
                continue

            gt_dict_exact[img_name] = waterline
            stem = Path(img_name).stem
            gt_dict_stem[stem] = waterline

            import re
            numbers = re.findall(r'\d+', img_name)
            if numbers:
                gt_dict_with_zero[numbers[0]] = waterline
                num_without_zero = str(int(numbers[0]))
                gt_dict_numeric[num_without_zero] = waterline

            gt_dict_exact[stem] = waterline

        print(f"\n成功加载真实水位线数据:")
        print(f"  精确匹配: {len(gt_dict_exact)} 条")
        print(f"  不带扩展名: {len(gt_dict_stem)} 条")
        print(f"  带前导零: {len(gt_dict_with_zero)} 条")
        print(f"  不带前导零: {len(gt_dict_numeric)} 条")

        return {
            'exact': gt_dict_exact,
            'stem': gt_dict_stem,
            'with_zero': gt_dict_with_zero,
            'numeric': gt_dict_numeric
        }
    except Exception as e:
        print(f"读取真实水位线文件失败: {e}")
        return {}


def find_ground_truth(filename, gt_dicts):
    """在多个字典中查找真实水位线"""
    filename_str = str(filename)

    for ext in ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']:
        if filename_str + ext in gt_dicts.get('exact', {}):
            return gt_dicts['exact'][filename_str + ext]

    if filename_str in gt_dicts.get('stem', {}):
        return gt_dicts['stem'][filename_str]

    import re
    numbers = re.findall(r'\d+', filename_str)
    if numbers:
        num = numbers[0]
        if num in gt_dicts.get('with_zero', {}):
            return gt_dicts['with_zero'][num]
        num_int = str(int(num))
        if num_int in gt_dicts.get('numeric', {}):
            return gt_dicts['numeric'][num_int]

    return None


def save_results_to_excel_with_comparison(success_results, failed_images, output_folder, gt_dicts):
    """将结果保存到Excel文件"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    delete_existing_excel_files(output_folder)

    if success_results:
        data = []
        matched_count = 0
        unmatched = []

        for filename, pred_waterline, mask_top_y, confidence, used_threshold in success_results:
            gt_waterline = find_ground_truth(filename, gt_dicts)

            if gt_waterline is not None:
                matched_count += 1
                diff = pred_waterline - gt_waterline
                diff_abs = abs(diff)
                gt_display = gt_waterline
                diff_display = diff
                diff_abs_display = diff_abs
                conf_display = f"{confidence:.4f}" if confidence is not None else "N/A"
                threshold_display = f"{used_threshold:.2f}"
                print(f"  {filename}: 测量={pred_waterline}, 真实={gt_waterline}, 差值={diff}")
            else:
                unmatched.append(filename)
                gt_display = "N/A"
                diff_display = "N/A"
                diff_abs_display = "N/A"
                conf_display = f"{confidence:.4f}" if confidence is not None else "N/A"
                threshold_display = f"{used_threshold:.2f}"
                print(f"  {filename}: 测量={pred_waterline}, 未找到真实值")

            mask_top_display = int(mask_top_y) if mask_top_y is not None else 'N/A'
            data.append(
                [filename, pred_waterline, mask_top_display, gt_display, diff_display, diff_abs_display, conf_display, threshold_display])

        success_df = pd.DataFrame(data, columns=['图片序号名', '测量水位线', '掩码最高位置', '真实水位线', '差值', '绝对差值', '置信度',
                                                 '使用阈值'])
        success_file = Path(gt_file)
        success_df.to_excel(success_file, index=False)
        print(f"\n对比结果已保存到: {success_file}")

    if failed_images:
        failed_df = pd.DataFrame(failed_images, columns=['图片序号名'])
        failed_file = Path(output_folder) / f"失败图片_{timestamp}.xlsx"
        failed_df.to_excel(failed_file, index=False)
        print(f"失败图片已保存到: {failed_file}")


def create_and_save_masks(masks, original_image_shape, output_path, file_name):
    """只保存合并后的掩码（不保存单个掩码）"""
    if masks is None:
        print(f"  无分割掩码可保存")
        return

    if isinstance(masks, torch.Tensor):
        masks_np = masks.cpu().numpy()
    else:
        masks_np = masks

    if masks_np.dtype == np.float32 or masks_np.dtype == np.float64:
        masks_np = masks_np > 0.5

    # 无论 masks_np 是什么形状，只生成合并掩码
    if len(masks_np.shape) == 3:
        combined_mask = np.any(masks_np, axis=0)
    elif len(masks_np.shape) == 2:
        combined_mask = masks_np
    elif len(masks_np.shape) == 4:
        if masks_np.shape[1] == 1:
            masks_np = masks_np.squeeze(1)
        combined_mask = np.any(masks_np, axis=0)
    else:
        print(f"  无法处理的掩码形状: {masks_np.shape}")
        return

    # 保存合并掩码
    combined_path = Path(output_path) / f"{file_name}_mask_combined.png"
    combined_img = (combined_mask * 255).astype(np.uint8)
    cv2.imwrite(str(combined_path), combined_img)
    print(f"  已保存合并掩码: {combined_path.name}")


def save_original_image(image_array, output_path, file_name, original_extension):
    """保存原始图片"""
    if len(image_array.shape) == 2:
        image_array = np.stack([image_array] * 3, axis=-1)

    original_path = Path(output_path) / f"{file_name}_original{original_extension}"
    cv2.imwrite(str(original_path), cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))
    print(f"  已保存原始图片: {original_path.name}")


def get_mask_bottom_y(mask):
    """获取掩码最低点的y坐标"""
    if mask is None:
        return -1

    if isinstance(mask, torch.Tensor):
        mask_np = mask.cpu().numpy()
    else:
        mask_np = mask

    if mask_np.dtype == np.float32 or mask_np.dtype == np.float64:
        mask_np = mask_np > 0.5

    mask_np = np.squeeze(mask_np)
    if mask_np.ndim != 2:
        return -1

    y_coords = np.where(mask_np)[0]
    if len(y_coords) == 0:
        return -1

    return int(np.max(y_coords))


def get_all_masks_sorted_by_confidence(masks, scores):
    """获取所有掩码并按置信度从高到低排序"""
    if masks is None or len(masks) == 0:
        return [], [], []

    if scores is not None and len(scores) > 0:
        if isinstance(scores, torch.Tensor):
            scores_np = scores.cpu().numpy()
        else:
            scores_np = np.array(scores)

        sorted_indices = np.argsort(scores_np)[::-1]
        sorted_scores = scores_np[sorted_indices]

        if isinstance(masks, torch.Tensor):
            sorted_masks = [masks[i] for i in sorted_indices]
        else:
            sorted_masks = [masks[i] for i in sorted_indices]

        return sorted_masks, sorted_scores, sorted_indices.tolist()
    else:
        if isinstance(masks, torch.Tensor):
            sorted_masks = [masks[i] for i in range(len(masks))]
        else:
            sorted_masks = [masks[i] for i in range(len(masks))]
        return sorted_masks, None, list(range(len(masks)))


def remove_top_connected_components(mask, top_rows=50):
    """
    删除所有与顶部 top_rows 行连通的掩码分量（洪水填充）
    使用连通分量分析识别并删除顶部区域
    """
    if mask is None or not np.any(mask):
        return mask

    # 挤压成2维
    if isinstance(mask, np.ndarray):
        mask_2d = np.squeeze(mask).copy()
        if mask_2d.ndim != 2:
            print(f"    警告：掩码维度为 {mask_2d.ndim}，无法处理顶部删除，返回原掩码")
            return mask
    else:
        return mask

    h, w = mask_2d.shape
    top_rows = min(top_rows, h)

    # 对整个掩码进行连通分量标记（8连通）
    mask_uint8 = (mask_2d > 0).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(mask_uint8, connectivity=8)

    # 收集顶部区域内的标签（排除背景0）
    top_region = labels[:top_rows, :]
    top_labels = set(top_region.flatten()) - {0}

    if not top_labels:
        return mask

    print(f"    顶部 {top_rows} 行内发现 {len(top_labels)} 个连通分量，将删除这些分量")

    # 创建新掩码，只保留不在 top_labels 中的标签
    new_mask = np.zeros_like(mask_2d, dtype=mask_2d.dtype)
    for label in range(1, num_labels):
        if label not in top_labels:
            new_mask[labels == label] = 1

    # 恢复原始形状
    if isinstance(mask, np.ndarray) and mask.ndim > 2:
        new_mask = new_mask.reshape(mask.shape)
    return new_mask


def get_connected_components(mask, kernel_size=3, opening_iterations=2):
    """
    获取掩码的所有连通分量，通过形态学开运算断开细小的连接

    开运算 = 先腐蚀后膨胀，作用：
    1. 断开通过细丝连接的多个区域
    2. 去除孤立的噪点
    3. 平滑区域边界

    Args:
        mask: 二值掩码
        kernel_size: 形态学核大小（奇数，默认3），越大断开连接的能力越强
        opening_iterations: 开运算迭代次数（默认2），越多分离越彻底

    Returns:
        [(component_mask, bottom_y), ...] 按底部y坐标从大到小排序
    """
    if mask is None or not np.any(mask):
        return []

    # 挤压成2维
    mask_2d = np.squeeze(mask).copy()
    if mask_2d.ndim != 2:
        return []

    # 转换为uint8格式
    mask_uint8 = (mask_2d > 0).astype(np.uint8)

    # 形态学开运算：先腐蚀后膨胀，断开细小的连接
    # 执行2次开运算，更彻底地断开连接
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask_opened = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel, iterations=opening_iterations)

    # 统计断开前后的连通分量数量变化
    original_labels, _ = cv2.connectedComponents(mask_uint8, connectivity=8)
    opened_labels, _ = cv2.connectedComponents(mask_opened, connectivity=8)
    if opened_labels > original_labels:
        print(
            f"      形态学开运算({opening_iterations}次)将 {original_labels - 1} 个连通分量分离为 {opened_labels - 1} 个")

    # 对处理后的掩码进行连通分量标记
    num_labels, labels = cv2.connectedComponents(mask_opened, connectivity=8)

    components = []
    for label in range(1, num_labels):
        component = (labels == label).astype(mask_2d.dtype)
        # 计算该分量的底部y坐标
        y_coords = np.where(component)[0]
        if len(y_coords) > 0:
            bottom_y = int(np.max(y_coords))
            components.append((component, bottom_y))

    # 按底部y坐标从大到小排序（从底部到顶部）
    components.sort(key=lambda x: x[1], reverse=True)
    return components


def select_masks_by_bottom_accumulation(mask, threshold_percent=60, kernel_size=3, opening_iterations=2):
    """
    从底部开始累加连通分量，直到累加面积占比达到阈值

    流程：
    1. 先通过开运算分离通过细丝连接的独立区域（执行2次开运算）
    2. 从最底部（y值最大）的连通分量开始
    3. 逐步向上累加分量
    4. 直到累加面积占总面积的比例达到阈值（默认60%）

    Args:
        mask: 已经删除顶部连通分量后的掩码
        threshold_percent: 面积占比阈值（默认60%）
        kernel_size: 形态学核大小（默认3）
        opening_iterations: 开运算迭代次数（默认2）

    Returns:
        selected_mask: 选中的掩码组合
        selected_components: 选中的分量列表
        total_area_ratio: 累加面积占总面积的比例
    """
    if mask is None or not np.any(mask):
        return None, [], 0

    # 获取所有连通分量，应用形态学开运算断开细丝连接
    components = get_connected_components(mask, kernel_size=kernel_size, opening_iterations=opening_iterations)

    if not components:
        return None, [], 0

    # 计算总像素面积（所有分量之和）
    total_pixels = 0
    for comp, _ in components:
        total_pixels += np.sum(comp)

    if total_pixels == 0:
        return None, [], 0

    # 计算阈值像素数
    threshold_area = int(total_pixels * threshold_percent / 100)
    print(f"    面积阈值: {threshold_percent}% = {threshold_area}/{total_pixels} 像素")

    # 从底部开始累加
    accumulated_mask = np.zeros_like(components[0][0], dtype=bool)
    accumulated_pixels = 0
    selected_components = []

    for comp, bottom_y in components:
        comp_pixels = np.sum(comp)
        accumulated_mask = accumulated_mask | comp
        accumulated_pixels += comp_pixels
        selected_components.append((comp, bottom_y))

        percentage = (accumulated_pixels / total_pixels) * 100
        print(
            f"      累加分量 (底部y={bottom_y}, 面积={comp_pixels}): 累加面积={accumulated_pixels}/{total_pixels} ({percentage:.1f}%)")

        if accumulated_pixels >= threshold_area:
            print(f"      累加面积达到阈值 {threshold_percent}%，停止累加")
            break

    return accumulated_mask, selected_components, accumulated_pixels / total_pixels


# 加载模型
model = build_sam3_image_model(checkpoint_path=r"E:\VS code\SAM3\sam3.pt")

# 设置路径
input_folder = r"E:\VS code\data\water_video_ROI\X"#"E:\VS code\SAM3\ROI_input"
output_folder = r"E:\VS code\data\water_video_ROI\X\result"#"E:\VS code\SAM3\initial_picture_output"
prompt = "black water"
gt_file = r"E:\VS code\SAM3\initial_high.xlsx"
mask_output_folder = r"C:\Users\ShenYuLong\Desktop\water_video_ROI\last_frame_mask"#"E:\VS code\SAM3\initial_mask_output"

save_masks = input("是否要将分割完成的掩码保存到指定文件夹？(y/n): ").lower().strip() in ['y', 'yes']
clear_output_folder(output_folder)
Path(output_folder).mkdir(parents=True, exist_ok=True)

if save_masks:
    Path(mask_output_folder).mkdir(parents=True, exist_ok=True)
    print(f"将保存分割掩码到: {mask_output_folder}")

print("\n加载真实水位线数据...")
gt_dicts = load_ground_truth_waterline(gt_file)

# 获取所有图片
image_files = set()
for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']:
    image_files.update(Path(input_folder).glob(ext))
for ext in ['*.JPG', '*.JPEG', '*.PNG', '*.BMP', '*.GIF']:
    image_files.update(Path(input_folder).glob(ext))

image_files = sorted(list(image_files))
failed = []
success_results = []

print(f"\n找到 {len(image_files)} 张图片")

# 逐张处理
for img_path in image_files:
    try:
        print(f"\n处理: {img_path.name}")

        image = Image.open(str(img_path)).convert('RGB')
        image_array = np.array(image)
        img_width, img_height = image.size
        print(f"  图像尺寸: {img_width}x{img_height}")



        inference_state = None
        used_threshold = None
        best_confidence = None
        found_valid_mask = False

        thresholds_to_try = [0.5, 0.4, 0.3, 0.2, 0.1, 0.0]

        for threshold in thresholds_to_try:
            print(f"  尝试阈值: {threshold:.1f}")

            processor = Sam3Processor(model, confidence_threshold=threshold)
            temp_inference_state = processor.set_image(image)
            temp_inference_state = processor.set_text_prompt(state=temp_inference_state, prompt=prompt)

            if 'masks' not in temp_inference_state or temp_inference_state['masks'] is None:
                print(f"    无掩码")
                continue

            masks = temp_inference_state['masks']
            scores = temp_inference_state.get('scores', [])

            num_masks = len(masks)
            print(f"    找到 {num_masks} 个掩码")

            if num_masks == 0:
                continue

            sorted_masks, sorted_scores, sorted_indices = get_all_masks_sorted_by_confidence(masks, scores)

            for i, mask in enumerate(sorted_masks):
                mask_idx = sorted_indices[i]
                score = sorted_scores[i] if sorted_scores is not None else None

                # 转换为numpy数组并挤压成2维
                if isinstance(mask, torch.Tensor):
                    mask_np = mask.cpu().numpy() > 0.5
                else:
                    mask_np = mask > 0.5
                mask_np = np.squeeze(mask_np)

                if mask_np.ndim != 2:
                    print(f"    掩码[{mask_idx}]: 维度异常 {mask_np.ndim}，跳过")
                    continue

                # 步骤1: 删除所有与顶部50行连通的掩码分量
                mask_after_top_removal = remove_top_connected_components(mask_np, top_rows=50)

                if not np.any(mask_after_top_removal):
                    print(f"    掩码[{mask_idx}]: 删除顶部连通分量后为空，舍弃")
                    continue

                # 步骤2: 检查删除顶部后的掩码底部距离是否 ≤ 8 像素
                bottom_y = get_mask_bottom_y(mask_after_top_removal)
                if bottom_y == -1:
                    print(f"    掩码[{mask_idx}]: 删除顶部后无法计算底部位置，舍弃")
                    continue

                distance_to_bottom = img_height - 1 - bottom_y
                score_display = f"{score:.4f}" if score is not None else "N/A"
                print(
                    f"    掩码[{mask_idx}]: 置信度={score_display}, 删除顶部后底部y={bottom_y}, 到底部距离={distance_to_bottom}")

                # 条件: 底部距离 ≤ 8 像素
                if distance_to_bottom > 8:
                    print(f"    掩码[{mask_idx}]: 删除顶部后不满足底部距离要求，舍弃")
                    continue

                # 步骤3: 从底部开始累加连通分量，直到面积占比达到60%
                selected_mask, selected_components, area_ratio = select_masks_by_bottom_accumulation(
                    mask_after_top_removal,
                    threshold_percent=60,  # 60%阈值
                    kernel_size=3,
                    opening_iterations=2
                )

                if selected_mask is None or not np.any(selected_mask):
                    print(f"    掩码[{mask_idx}]: 无法通过底部累加选择有效掩码，舍弃")
                    continue

                print(
                    f"    掩码[{mask_idx}]: 选中{len(selected_components)}个分量, 累加面积占比={area_ratio * 100:.1f}%")

                # 将选中的掩码转换回原始类型
                if isinstance(mask, torch.Tensor):
                    processed_mask = torch.from_numpy(selected_mask.astype(np.float32)).to(mask.device)
                else:
                    processed_mask = selected_mask.astype(np.float32)

                # 构建 inference_state
                if isinstance(masks, torch.Tensor):
                    selected_masks = processed_mask.unsqueeze(0)
                else:
                    selected_masks = np.expand_dims(processed_mask, axis=0)

                if score is not None:
                    selected_scores = torch.tensor([score], dtype=torch.float32)
                else:
                    selected_scores = torch.tensor([], dtype=torch.float32)

                inference_state = {
                    'masks': selected_masks,
                    'scores': selected_scores
                }

                if 'boxes' in temp_inference_state and temp_inference_state['boxes'] is not None:
                    inference_state['boxes'] = temp_inference_state['boxes'][mask_idx:mask_idx + 1]

                used_threshold = threshold
                best_confidence = score
                found_valid_mask = True
                print(f"    掩码[{mask_idx}] 满足所有条件，使用此掩码")
                break

            if found_valid_mask:
                break

        if not found_valid_mask:
            print(f"  未找到满足条件的掩码: {img_path.name}")
            failed.append(img_path.name)
            if save_masks:
                empty_mask_path = Path(mask_output_folder) / f"{img_path.stem}_mask_empty.png"
                empty_mask = np.zeros((img_height, img_width), dtype=np.uint8)
                cv2.imwrite(str(empty_mask_path), empty_mask)
                print(f"  已保存空掩码: {empty_mask_path.name}")
            continue

        if save_masks:
            create_and_save_masks(inference_state['masks'], (img_height, img_width), mask_output_folder, img_path.stem)

        plt.clf()
        plt.close('all')
        plot_results(image, inference_state)

        waterline_y, mask_top_y = calculate_waterline_by_percentage(
            inference_state['masks'],
            img_height,
            threshold_percent=60,  # 60%阈值
            min_continuous_rows=3
        )

        if waterline_y is not None:
            ax = plt.gca()
            ax.axhline(y=waterline_y, color='blue', linewidth=2, linestyle='--')
            for text in ax.texts:
                text.remove()
            ax.set_title('')
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis('off')

            print(f"  水位线位置: {waterline_y}")
            success_results.append((img_path.stem, waterline_y, mask_top_y, best_confidence, used_threshold))

            gt_value = find_ground_truth(img_path.stem, gt_dicts)
            if gt_value is not None:
                diff = waterline_y - gt_value
                print(f"  真实水位线: {gt_value}, 差值: {diff}")
            else:
                print(f"  未找到真实水位线")
        else:
            print("  无法计算水位线")
            failed.append(img_path.name)
            ax = plt.gca()
            for text in ax.texts:
                text.remove()
            ax.set_title('')
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis('off')

        save_path = Path(output_folder) / f"{img_path.stem}_result.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.clf()
        plt.close('all')
        print(f"  已保存包含水位线的结果: {save_path.name}")

    except Exception as e:
        print(f"  错误: {img_path.name} - {e}")
        failed.append(img_path.name)
        import traceback

        traceback.print_exc()
    finally:
        plt.close('all')

save_results_to_excel_with_comparison(success_results, failed, output_folder, gt_dicts)

print("\n" + "=" * 50)
print("处理完成！")
print(f"总图片数: {len(image_files)}")
print(f"成功找到水位线: {len(success_results)} 张")
print(f"未找到水位线: {len(failed)} 张")
if save_masks:
    print(f"原始图片和分割掩码保存到: {mask_output_folder}")
    print(f"  - 原始图片: [文件名]_original.[原扩展名]")
    print(f"  - 单个掩码: [文件名]_mask_XXX.png")
    print(f"  - 合并掩码: [文件名]_mask_combined.png")
    print(f"  - 空掩码: [文件名]_mask_empty.png")
print(f"包含水位线的结果保存到: {output_folder}")
print("=" * 50)