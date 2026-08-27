import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import sobel
import os
import glob
import pandas as pd
import traceback
from pathlib import Path
from PIL import Image
import warnings
from collections import deque

warnings.filterwarnings('ignore')

# ============ 调试保存函数（✅ 已关闭） ============
def save_debug_image(folder, base_name, tag, img):
    return


# ============ 修正的洪泛算法 ============
def binary_flood_fill(image_with_red_points, base_name):
    result = image_with_red_points.copy()
    height, width = result.shape[:2]

    all_red_points = []
    for y in range(height):
        for x in range(width):
            pixel = result[y, x]
            if pixel[0] == 0 and pixel[1] == 0 and pixel[2] == 255:
                all_red_points.append((x, y))

    if not all_red_points:
        print("   没有找到红色点")
        return result, np.zeros((height, width), dtype=np.uint8)

    print(f"   找到 {len(all_red_points)} 个红色点")

    center_x, center_y = width // 2, height // 2
    distances = [
        (np.sqrt((x - center_x)**2 + (y - center_y)**2), (x, y))
        for (x, y) in all_red_points
    ]
    distances.sort(key=lambda x: x[0])
    start_point = distances[0][1]
    start_x, start_y = start_point

    print(f"   起始点: {start_point}")

    white_count = black_count = 0
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            nx, ny = start_x + dx, start_y + dy
            if 0 <= nx < width and 0 <= ny < height:
                pixel = result[ny, nx]
                if not (pixel[0] == 0 and pixel[1] == 0 and pixel[2] == 255):
                    if pixel[0] == 255 and pixel[1] == 255 and pixel[2] == 255:
                        white_count += 1
                    elif pixel[0] == 0 and pixel[1] == 0 and pixel[2] == 0:
                        black_count += 1

    start_color = 'white' if white_count > black_count else 'black'
    print(f"   ✅ 起始点判定结果：{start_color.upper()} 区域")

    other_color = [255, 255, 255] if start_color == 'white' else [0, 0, 0]
    for (x, y) in all_red_points:
        if (x, y) != start_point:
            result[y, x] = other_color

    if start_color == 'white':
        return flood_white_strategy(result, start_point, all_red_points, base_name)
    else:
        return flood_black_strategy(result, start_point, all_red_points, base_name)


# ============ 白色策略 ============
def flood_white_strategy(image, start_point, all_red_points, base_name):
    result = image.copy()
    height, width = result.shape[:2]
    start_x, start_y = start_point

    visited = np.zeros((height, width), dtype=bool)
    flood_mask = np.zeros((height, width), dtype=bool)
    directions = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

    queue = deque([(start_x, start_y)])
    visited[start_y, start_x] = True
    flood_mask[start_y, start_x] = True

    while queue:
        x, y = queue.popleft()
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if nx < 0 or nx >= width or ny < 0 or ny >= height:
                continue
            if visited[ny, nx]:
                continue
            pixel = result[ny, nx]
            if pixel[0] == 255 and pixel[1] == 255 and pixel[2] == 255:
                visited[ny, nx] = True
                flood_mask[ny, nx] = True
                queue.append((nx, ny))

    filled_mask = fill_holes_simple(flood_mask)
    return image.copy(), filled_mask.astype(np.uint8) * 255


# ============ 黑色策略 ============
def flood_black_strategy(image, start_point, all_red_points, base_name):
    result = image.copy()
    height, width = result.shape[:2]
    start_x, start_y = start_point

    visited = np.zeros((height, width), dtype=bool)
    flood_mask = np.zeros((height, width), dtype=bool)
    directions = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

    queue = deque([(start_x, start_y)])
    visited[start_y, start_x] = True
    flood_mask[start_y, start_x] = True

    black_pixels = 0
    black_regions = []

    EDGE_DIST = 2

    while queue:
        x, y = queue.popleft()
        black_pixels += 1
        black_regions.append((x, y))

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if nx < 0 or nx >= width or ny < 0 or ny >= height:
                continue
            if (
                nx <= EDGE_DIST or
                nx >= width - 1 - EDGE_DIST or
                ny <= EDGE_DIST or
                ny >= height - 1 - EDGE_DIST
            ):
                continue
            if visited[ny, nx]:
                continue
            pixel = result[ny, nx]
            if pixel[0] == 0 and pixel[1] == 0 and pixel[2] == 0:
                visited[ny, nx] = True
                flood_mask[ny, nx] = True
                queue.append((nx, ny))

    white_border = []
    if black_pixels > 0:
        for (bx, by) in black_regions:
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = bx + dx, by + dy
                    if 0 <= nx < width and 0 <= ny < height and not flood_mask[ny, nx]:
                        pixel = result[ny, nx]
                        if pixel[0] == 255 and pixel[1] == 255 and pixel[2] == 255:
                            white_border.append((nx, ny))

        white_queue = deque(white_border)
        for (wx, wy) in white_border:
            visited[wy, wx] = True
            flood_mask[wy, wx] = True

        while white_queue:
            x, y = white_queue.popleft()
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if nx < 0 or nx >= width or ny < 0 or ny >= height:
                    continue
                if visited[ny, nx]:
                    continue
                pixel = result[ny, nx]
                if pixel[0] == 255 and pixel[1] == 255 and pixel[2] == 255:
                    visited[ny, nx] = True
                    flood_mask[ny, nx] = True
                    white_queue.append((nx, ny))

    filled_mask = fill_holes_simple(flood_mask)
    return image.copy(), filled_mask.astype(np.uint8) * 255


# ============ 空洞修补 ============
def fill_holes_simple(mask):
    if np.sum(mask) == 0:
        return mask
    mask_uint8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mask_uint8, contours, -1, 255, -1)
    return mask_uint8.astype(bool)


# =========================
# ✅ 新增函数 1：查找所有蓝色区域的外接矩形
# =========================
def find_all_blue_component_boxes(mask, min_area=20):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        cx = x + w // 2
        cy = y + h // 2
        boxes.append((x, y, w, h, cx, cy))
    return boxes


# =========================
# ✅ 新增函数 2：画框并记录
# =========================
def draw_and_record_blue_boxes(image, blue_mask, box_list, round_id):
    boxes = find_all_blue_component_boxes(blue_mask)
    for x, y, w, h, cx, cy in boxes:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        box_list.append({
            "round": round_id,
            "component_id": len(box_list) + 1,
            "x": x,
            "y": y,
            "width": w,
            "height": h,
            "center_x": cx,
            "center_y": cy
        })


# ============ 截取水体区域 ============
def crop_water_region_by_excel(img_path, excel_path, img_num):
    try:
        df = pd.read_excel(excel_path)
        seq_col = '图片序号名'
        mask_top_col = '掩码最高位置'

        match = df[df[seq_col] == img_num]
        if len(match) == 0:
            print(f"   ❌ Excel 中找不到图片序号: {img_num}")
            return None, None, False

        mask_top_raw = match.iloc[0][mask_top_col]
        if pd.isna(mask_top_raw):
            print(f"   ❌ 掩码最高位置为空: {img_num}")
            return None, None, False

        mask_top = int(float(mask_top_raw))

        img = Image.open(img_path)
        img_array = np.array(img)

        if len(img_array.shape) == 3 and img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
        elif len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        elif len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)

        height = img_array.shape[0]
        mask_top = max(0, min(mask_top, height - 1))

        cropped_region = img_array[mask_top:, :]
        return cropped_region, mask_top, True

    except Exception as e:
        print(f"   ❌ 裁剪水体区域失败: {e}")
        return None, None, False


# ============ TLBP 相关 ============
def orientation_adaptive_tlbp_binary(image, P=8, R=1, use_local_orientation=True,
                                     fixed_angle=None, orientation_threshold=None,
                                     tlbp_threshold=5, binary_threshold=0):
    h, w = image.shape
    tlbp_binary = np.zeros((h, w), dtype=np.uint8)

    std_points = []
    for i in range(P):
        theta = 2 * np.pi * i / P
        std_points.append(np.array([R * np.cos(theta), R * np.sin(theta)]))

    if use_local_orientation:
        orientations = compute_gradient_orientation(image)
        orientations = orientations % 180
    else:
        orientations = np.full((h, w), fixed_angle % 180)

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            center = image[i, j]
            if orientation_threshold is not None and use_local_orientation:
                angle = orientations[i, j]
                if angle > 90: angle = 180 - angle
                points = rotate_points(std_points, angle) if angle < orientation_threshold else std_points
            elif use_local_orientation:
                points = rotate_points(std_points, orientations[i, j])
            else:
                points = std_points

            active_count = 0
            for dx, dy in points:
                xi, yi = j + dx, i + dy
                if 0 <= xi < w and 0 <= yi < h:
                    val = bilinear_interp(image, xi, yi)
                else:
                    val = center
                if abs(int(val) - int(center)) > tlbp_threshold:
                    active_count += 1

            if active_count > binary_threshold:
                tlbp_binary[i, j] = 255

    return tlbp_binary


def image_entropy_gray(image):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
    prob = hist / image.size
    prob = prob[prob > 0]
    return -np.sum(prob * np.log2(prob))


def compute_gradient_orientation(image):
    gx = sobel(image, axis=1)
    gy = sobel(image, axis=0)
    return np.rad2deg(np.arctan2(gy, gx))


def rotate_points(points, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array([[c, -s], [s, c]])
    return [R @ p for p in points]


def bilinear_interp(image, x, y):
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    x1, y1 = x0 + 1, y0 + 1
    if x1 >= image.shape[1] or y1 >= image.shape[0]:
        return image[y0, x0]
    wa = (x1 - x) * (y1 - y)
    wb = (x - x0) * (y1 - y)
    wc = (x1 - x) * (y - y0)
    wd = (x - x0) * (y - y0)
    return wa * image[y0, x0] + wb * image[y0, x1] + wc * image[y1, x0] + wd * image[y1, x1]


# ============ 保存函数 ============
def safe_imwrite(save_path, image, verbose=False):
    try:
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        if image.dtype != np.uint8:
            if np.issubdtype(image.dtype, np.floating):
                image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
            image = image.astype(np.uint8)

        cv2.imwrite(save_path, image)
        return True
    except Exception:
        return False


# ============ 单张图片处理 ============
# ============ 单张图片处理 ============
# ============ 单张图片处理 ============
def process_single_image_binary_flood(excel_path, input_folder, output_folder, img_num,
                                      tlbp_threshold=4, min_brightness=200):
    try:
        # ===============================
        # ✅ 四位数图片名（0081.png）
        # ===============================
        filename = f"{img_num:04d}.png"
        img_path = os.path.join(input_folder, filename)

        if not os.path.exists(img_path):
            print(f"   ❌ 图片不存在: {img_path}")
            return False, 0, 0, 0

        # ===============================
        # ✅ 关键修复：防止 UnboundLocalError
        # ===============================
        red_points = []
        tlbp_binary = None

        cropped_img, water_level, success = crop_water_region_by_excel(
            img_path, excel_path, img_num
        )
        if not success or cropped_img is None:
            print(f"   ❌ 裁剪失败: {img_num}")
            return False, 0, 0, 0

        cropped_gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)

        green_mask = np.zeros_like(cropped_gray, dtype=np.uint8)
        h, w = green_mask.shape
        border = 2
        green_mask[:border, :] = 255
        green_mask[-border:, :] = 255
        green_mask[:, :border] = 255
        green_mask[:, -border:] = 255

        iterations = 0
        all_boxes = []
        round_id = 0

        while iterations < 10:
            iterations += 1
            round_id += 1

            modified_brightness = cropped_gray.copy()
            modified_brightness[green_mask == 255] = 0

            _, max_val, _, _ = cv2.minMaxLoc(modified_brightness)
            if max_val < min_brightness:
                break

            red_points = np.where(modified_brightness == max_val)
            red_points = list(zip(red_points[1], red_points[0]))
            if not red_points:
                break

            tlbp_binary = orientation_adaptive_tlbp_binary(
                cv2.medianBlur(cropped_gray, 5),
                P=8, R=1, use_local_orientation=True,
                orientation_threshold=30,
                tlbp_threshold=tlbp_threshold,
                binary_threshold=0
            )

            kernel = np.ones((5, 5), np.uint8)
            tlbp_binary = cv2.morphologyEx(tlbp_binary, cv2.MORPH_CLOSE, kernel, iterations=2)

            tlbp_with_seeds = cv2.cvtColor(tlbp_binary, cv2.COLOR_GRAY2BGR)
            tlbp_with_seeds[green_mask == 255] = [0, 0, 0]
            for x, y in red_points:
                tlbp_with_seeds[y, x] = [0, 0, 255]

            _, blue_mask = binary_flood_fill(
                tlbp_with_seeds, f"{img_num}_round{iterations}"
            )

            blue_mask_binary = (blue_mask > 0).astype(np.uint8)
            blue_pixels = np.sum(blue_mask_binary)
            remaining_pixels = np.sum(green_mask == 0)

            if remaining_pixels > 0 and blue_pixels / remaining_pixels > 0.6:
                break

            draw_and_record_blue_boxes(
                cropped_img,
                blue_mask,
                all_boxes,
                round_id
            )

            green_mask[blue_mask_binary == 1] = 255

        green_mask[:border, :] = 0
        green_mask[-border:, :] = 0
        green_mask[:, :border] = 0
        green_mask[:, -border:] = 0

        green_mask_path = os.path.join(output_folder, f"{img_num}_green_mask.png")
        cv2.imwrite(green_mask_path, green_mask)

        boxed_mask_path = os.path.join(
            output_folder, f"{img_num}_green_mask_boxes.png"
        )
        cv2.imwrite(boxed_mask_path, cropped_img)

        if all_boxes:
            box_df = pd.DataFrame(all_boxes)
            box_csv_path = os.path.join(
                output_folder, f"{img_num}_blue_boxes.csv"
            )
            box_df.to_csv(box_csv_path, index=False, encoding="utf-8-sig")
            print(f"   ✅ 已生成 {len(all_boxes)} 个框")

        # ================= 无绿色框图片可视化 =================
        try:
            has_green_box = len(all_boxes) > 0

            if not has_green_box:
                vis_dir = r"E:\VS code\SAM3\image_without_solar_flow.xlsx"
                os.makedirs(vis_dir, exist_ok=True)

                vis_save_path = os.path.join(vis_dir, f"{img_num}_flood_overlay.png")

                clean_img = cropped_img.copy()

                visualize_flood_result_clean(
                    clean_img,
                    green_mask,
                    vis_save_path
                )

        except Exception as e:
            print(f"   ⚠️ 可视化失败: {e}")

        # ===============================
        # ✅ 安全 return（不会再崩）
        # ===============================
        entropy = image_entropy_gray(tlbp_binary) if tlbp_binary is not None else 0.0
        return True, len(red_points), entropy, iterations

    except Exception as e:
        print(f"   ❌ 处理异常: {e}")
        traceback.print_exc()
        return False, 0, 0, 0
# ============ 批量处理 ============
def batch_process_binary_flood(excel_path, input_folder, output_folder,
                               tlbp_threshold=4, min_brightness=200):
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(excel_path)
    image_numbers = df['图片序号名'].dropna().astype(int).tolist()

    print(f"✅ 共发现 {len(image_numbers)} 张图片，开始批量处理...\n")

    stats = {'total': len(image_numbers), 'processed': 0, 'with_seeds': 0, 'without_seeds': 0, 'failed': 0}
    excel_data = []

    for idx, img_num in enumerate(image_numbers, 1):
        print(f"\n[{idx}/{len(image_numbers)}] 处理图片: {img_num}")
        has_seeds, seed_count, entropy_tlbp, flood_iterations = process_single_image_binary_flood(
            excel_path, input_folder, output_folder, img_num,
            tlbp_threshold, min_brightness
        )

        stats['processed'] += 1

        if has_seeds:
            status = "成功(有种子的TLBP图)"
            stats['with_seeds'] += 1
        else:
            status = "成功(无种子的TLBP图)"
            stats['without_seeds'] += 1

        excel_data.append({
            '图片序号': img_num,
            '处理状态': status,
            '是否有种子': '是' if has_seeds else '否',
            '种子点数量': seed_count,
            'TLBP熵值': round(entropy_tlbp, 6) if entropy_tlbp > 0 else 0,
            '洪泛迭代次数': flood_iterations
        })

        print(f"   ✅ 状态: {status}")

    pd.DataFrame(excel_data).to_excel(
        os.path.join(output_folder, 'binary_flood_results.xlsx'),
        index=False, engine='openpyxl'
    )

    print("\n==================== 处理完成 ====================")
    print(f"总计: {stats['total']}")
    print(f"成功: {stats['processed']}")
    print(f"有种子: {stats['with_seeds']}")
    print(f"无种子: {stats['without_seeds']}")
    print("=================================================\n")


# ============ 新增：蓝色洪泛结果可视化 ============
def visualize_flood_result_on_image(
    original_img, flood_mask, save_path
):
    if original_img is None or flood_mask is None:
        return

    blue_mask = np.zeros_like(original_img, dtype=np.uint8)
    blue_mask[flood_mask > 0] = (255, 0, 0)

    overlay = cv2.addWeighted(original_img, 0.7, blue_mask, 0.3, 0)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, overlay)


def visualize_flood_result_clean(
    original_img, flood_mask, save_path
):
    if original_img is None or flood_mask is None:
        return

    blue_mask = np.zeros_like(original_img, dtype=np.uint8)
    blue_mask[flood_mask > 0] = (255, 0, 0)

    overlay = cv2.addWeighted(original_img, 0.7, blue_mask, 0.3, 0)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, overlay)


# ============ 主程序 ============
if __name__ == "__main__":
    EXCEL_PATH = r"E:\VS code\SAM3\initial_high.xlsx"
    INPUT_FOLDER = r"E:\VS code\SAM3\ROI_input"
    OUTPUT_FOLDER = r"E:\VS code\SAM3\output-notice"

    if not os.path.exists(EXCEL_PATH) or not os.path.exists(INPUT_FOLDER):
        print("❌ Excel 或输入文件夹不存在")
        exit(1)

    Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)
    batch_process_binary_flood(EXCEL_PATH, INPUT_FOLDER, OUTPUT_FOLDER, 4, 200)