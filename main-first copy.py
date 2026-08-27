import sys
import os
import cv2
import torch
import shutil
import pandas as pd
import re
from datetime import datetime
from pathlib import Path
from PIL import Image
import numpy as np

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_video_predictor
sys.path.append(r"D:\sam3model")



#################################### 核心掩码计算与形态学函数 ####################################

def get_mask_bottom_y(mask):
    if mask is None: return -1
    mask_np = mask.cpu().float().numpy() if isinstance(mask, torch.Tensor) else mask
    mask_np = mask_np > 0.5 if mask_np.dtype in [np.float32, np.float64] else mask_np
    mask_np = np.squeeze(mask_np)
    if mask_np.ndim != 2: return -1
    y_coords = np.where(mask_np)[0]
    return int(np.max(y_coords)) if len(y_coords) > 0 else -1


def get_all_masks_sorted_by_confidence(masks, scores):
    if masks is None or len(masks) == 0: return [], [], []
    if scores is not None and len(scores) > 0:
        scores_np = scores.cpu().float().numpy() if isinstance(scores, torch.Tensor) else np.array(scores)
        sorted_indices = np.argsort(scores_np)[::-1]
        sorted_scores = scores_np[sorted_indices]
        sorted_masks = [masks[i] for i in sorted_indices]
        return sorted_masks, sorted_scores, sorted_indices.tolist()
    else:
        sorted_masks = [masks[i] for i in range(len(masks))]
        return sorted_masks, None, list(range(len(masks)))


def remove_top_connected_components(mask, top_rows=50):
    if mask is None or not np.any(mask): return mask
    mask_2d = np.squeeze(mask).copy()
    if mask_2d.ndim != 2: return mask
    h, w = mask_2d.shape
    top_rows = min(top_rows, h)
    mask_uint8 = (mask_2d > 0).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(mask_uint8, connectivity=8)
    top_labels = set(labels[:top_rows, :].flatten()) - {0}
    if not top_labels: return mask
    new_mask = np.zeros_like(mask_2d)
    for label in range(1, num_labels):
        if label not in top_labels: new_mask[labels == label] = 1
    return new_mask.reshape(mask.shape) if isinstance(mask, np.ndarray) and mask.ndim > 2 else new_mask


def get_connected_components(mask, kernel_size=3, opening_iterations=2):
    if mask is None or not np.any(mask): return []
    mask_2d = np.squeeze(mask).copy()
    if mask_2d.ndim != 2: return []
    mask_uint8 = (mask_2d > 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask_opened = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel, iterations=opening_iterations)
    num_labels, labels = cv2.connectedComponents(mask_opened, connectivity=8)
    components = []
    for label in range(1, num_labels):
        component = (labels == label).astype(mask_2d.dtype)
        y_coords = np.where(component)[0]
        if len(y_coords) > 0: components.append((component, int(np.max(y_coords))))
    components.sort(key=lambda x: x[1], reverse=True)
    return components


def select_masks_by_bottom_accumulation(mask, threshold_percent=60, kernel_size=3, opening_iterations=2):
    if mask is None or not np.any(mask): return None, [], 0
    components = get_connected_components(mask, kernel_size=kernel_size, opening_iterations=opening_iterations)
    if not components: return None, [], 0
    total_pixels = sum(np.sum(comp) for comp, _ in components)
    if total_pixels == 0: return None, [], 0
    threshold_area = int(total_pixels * threshold_percent / 100)
    accumulated_mask = np.zeros_like(components[0][0], dtype=bool)
    accumulated_pixels = 0
    selected_components = []
    for comp, bottom_y in components:
        comp_pixels = np.sum(comp)
        accumulated_mask = accumulated_mask | comp
        accumulated_pixels += comp_pixels
        selected_components.append((comp, bottom_y))
        if accumulated_pixels >= threshold_area: break
    return accumulated_mask, selected_components, accumulated_pixels / total_pixels


def calculate_waterline_by_percentage(masks, image_height, threshold_percent=60, min_continuous_rows=3):
    if masks is None: return None, None
    if isinstance(masks, torch.Tensor): masks = masks.cpu().numpy()
    if masks.dtype in [np.float32, np.float64]: masks = masks > 0.5
    combined_mask = np.squeeze(masks)
    if combined_mask.ndim != 2: return None, None
    if not np.any(combined_mask): return None, None
    h, w = combined_mask.shape
    threshold_count = int(w * threshold_percent / 100)
    found_y = None
    continuous_count = 0
    for y in range(h):
        row_mask_count = np.sum(combined_mask[y, :])
        if row_mask_count >= threshold_count:
            continuous_count += 1
            if continuous_count >= min_continuous_rows:
                found_y = y - continuous_count + 1
                break
        else:
            continuous_count = 0
    if found_y is None:
        for y in range(h):
            if np.sum(combined_mask[y, :]) >= threshold_count:
                found_y = y
                break
        if found_y is None:
            y_coords = np.where(combined_mask)[0]
            if len(y_coords) > 0: found_y = np.max(y_coords)
    all_y = np.where(combined_mask)[0]
    mask_top_y = int(np.min(all_y)) if len(all_y) > 0 else 0
    return (int(found_y) if found_y is not None else None), mask_top_y


def extract_2d_mask(mask_tensor):
    mask_np = mask_tensor.cpu().float().numpy() if torch.is_tensor(mask_tensor) else np.array(mask_tensor)
    mask_np = np.squeeze(mask_np)
    if mask_np.ndim == 3: mask_np = mask_np[0]
    return mask_np > 0.5


def keep_largest_connected_component(mask_2d):
    if mask_2d is None or not np.any(mask_2d): return mask_2d
    mask_uint8 = (mask_2d > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
    if num_labels <= 1: return mask_2d
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = np.argmax(areas) + 1
    return labels == largest_label


def calculate_overlap_ratio(mask_test, mask_ref):
    if mask_ref is None: return 0.0
    intersection = np.logical_and(mask_test, mask_ref).sum()
    area_test = mask_test.sum()
    return intersection / area_test if area_test > 0 else 0.0


def apply_morphology(mask_2d):
    mask_uint8 = (mask_2d.astype(np.uint8) * 255)
    kernel_open = np.ones((5, 5), np.uint8)
    opened = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel_open)
    kernel_close = np.ones((11, 11), np.uint8)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close)
    return closed > 0


def get_water_mask_pipeline(pil_image, model, device, prompt="black water", is_cropped=False):
    img_width, img_height = pil_image.size
    thresholds_to_try = [0.5, 0.4, 0.3, 0.2, 0.1, 0.0]

    for threshold in thresholds_to_try:
        try:
            processor = Sam3Processor(model, device=device, confidence_threshold=threshold)
        except:
            processor = Sam3Processor(model, confidence_threshold=threshold)
        state = processor.set_image(pil_image)
        state = processor.set_text_prompt(state=state, prompt=prompt)
        if 'masks' not in state or state['masks'] is None: continue
        masks = state['masks']
        scores = state.get('scores', [])
        if len(masks) == 0: continue
        sorted_masks, sorted_scores, sorted_indices = get_all_masks_sorted_by_confidence(masks, scores)

        for i, mask in enumerate(sorted_masks):
            score = sorted_scores[i] if sorted_scores is not None else None
            mask_np = mask.cpu().float().numpy() > 0.5 if isinstance(mask, torch.Tensor) else mask > 0.5
            mask_np = np.squeeze(mask_np)
            if mask_np.ndim != 2: continue

            if not is_cropped:
                mask_after_top_removal = remove_top_connected_components(mask_np, top_rows=50)
            else:
                mask_after_top_removal = mask_np.copy()

            if not np.any(mask_after_top_removal): continue
            bottom_y = get_mask_bottom_y(mask_after_top_removal)
            if bottom_y == -1: continue
            if (img_height - 1 - bottom_y) > 8: continue

            selected_mask, selected_components, area_ratio = select_masks_by_bottom_accumulation(
                mask_after_top_removal, threshold_percent=60, kernel_size=3, opening_iterations=2
            )
            if selected_mask is None or not np.any(selected_mask): continue
            return selected_mask, score, threshold
    return None, None, None


#################################### 高阶核心图片处理流程 ####################################

def process_single_image(pil_image, model, prompt):
    device = next(model.parameters()).device if hasattr(model, 'parameters') else "cuda"

    try:
        pil_image = pil_image.convert("RGB")
        image_np = np.array(pil_image)
        h_orig, w_orig = image_np.shape[:2]
    except Exception as e:
        print(f"  ❌ 无法转图片: {e}")
        return False, -1

    ref_water_mask, score1, thresh1 = get_water_mask_pipeline(pil_image, model, device, prompt=prompt, is_cropped=False)
    ref_waterline_y, ref_mask_top_y = -1, -1

    if ref_water_mask is not None:
        cleaned_ref_mask = apply_morphology(ref_water_mask)
        calc_y, top_y = calculate_waterline_by_percentage(cleaned_ref_mask, h_orig)
        if calc_y is not None:
            ref_waterline_y, ref_mask_top_y = calc_y, top_y
            print(f"  [原水体] 置信度:{score1:.3f}, Y={ref_waterline_y}")

    bank_masks, bank_scores = [], []
    for bank_thresh in [0.4, 0.2, 0.0]:
        try:
            processor_bank = Sam3Processor(model, device=device, confidence_threshold=bank_thresh)
        except:
            processor_bank = Sam3Processor(model, confidence_threshold=bank_thresh)

        state_bank = processor_bank.set_image(pil_image)
        processor_bank.set_text_prompt("bank", state_bank)
        if state_bank.get("masks") is not None and len(state_bank.get("masks")) > 0:
            bank_masks, bank_scores = state_bank.get("masks"), state_bank.get("scores")
            break

    best_bank_mask, best_bank_score = None, -1.0
    if len(bank_masks) > 0:
        sorted_bank_masks, sorted_bank_scores, _ = get_all_masks_sorted_by_confidence(bank_masks, bank_scores)
        top_thresh = max(50, int(h_orig * 0.1))

        for i, mask in enumerate(sorted_bank_masks):
            m_2d = extract_2d_mask(mask)
            m_2d = keep_largest_connected_component(m_2d)
            s_val = float(sorted_bank_scores[i]) if sorted_bank_scores is not None else 0.0

            y_coords, _ = np.where(m_2d)
            if len(y_coords) == 0:
                continue
            min_y = np.min(y_coords)
            overlap = calculate_overlap_ratio(m_2d, ref_water_mask)

            if min_y <= top_thresh and overlap <= 0:
                if s_val > best_bank_score:
                    best_bank_score, best_bank_mask = s_val, m_2d

    crop_y = 0
    if best_bank_mask is not None:
        lowest_y = np.max(np.where(best_bank_mask)[0])
        crop_y = max(0, lowest_y - 10)

    cropped_img_np = image_np[crop_y:, :, :]
    cropped_pil = Image.fromarray(cropped_img_np)
    best_crop_water_mask, score2, thresh2 = get_water_mask_pipeline(cropped_pil, model, device, prompt=prompt,
                                                                    is_cropped=True)

    final_full_mask = np.zeros((h_orig, w_orig), dtype=bool)
    new_waterline_y, new_mask_top_y = -1, -1

    if best_crop_water_mask is not None:
        cleaned_crop_mask = apply_morphology(best_crop_water_mask)
        final_full_mask[crop_y:, :] = cleaned_crop_mask
        calc_new_y, top_y = calculate_waterline_by_percentage(final_full_mask, h_orig)
        if calc_new_y is not None:
            new_waterline_y, new_mask_top_y = calc_new_y, top_y
            print(f"  [裁剪水体] 置信度:{score2:.3f}, Y={new_waterline_y}")

    final_y = -1
    final_mask = None
    final_score = -1.0

    if ref_waterline_y != -1 and new_waterline_y != -1:
        if ref_waterline_y < new_waterline_y:
            print(f"  [决策] 原水体(Y={ref_waterline_y}) 更靠上，弃用裁剪。")
            final_y = ref_waterline_y
            final_mask = cleaned_ref_mask
            final_score = score1
        else:
            print(f"  [决策] 裁剪后(Y={new_waterline_y}) 更靠上，采用裁剪。")
            final_y = new_waterline_y
            final_mask = final_full_mask
            final_score = score2
    elif new_waterline_y != -1:
        final_y = new_waterline_y
        final_mask = final_full_mask
        final_score = score2
    elif ref_waterline_y != -1:
        final_y = ref_waterline_y
        final_mask = cleaned_ref_mask
        final_score = score1
    else:
        print("  ❌ 提取失败，未找到合法水体")
        return False, -1, None, -1.0

    if final_mask is None:
        print("  ❌ 提取失败，未找到合法mask")
        return False, -1, None, -1.0

    return True, final_y, final_mask, final_score 


def save_frames_as_mp4(frames, output_path, fps=30):
    if not frames: return False
    h, w = frames[0].shape[:2]
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for frame in frames: out.write(frame)
    out.release()
    return True


def overlay_mask(frame, outputs, alpha=0.5):
    if not outputs: return frame
    overlay = frame.copy()
    h, w = frame.shape[:2]
    masks = outputs.get('out_binary_masks', [])
    for mask in masks:
        if len(mask.shape) == 3: mask = mask.squeeze()
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
            mask = (mask > 0.5).astype(np.uint8)
        mask_bool = mask > 0.5
        for c in range(3):
            overlay[:, :, c][mask_bool] = (alpha * 255 * (c == 0) + (1 - alpha) * frame[:, :, c][mask_bool]).astype(
                np.uint8)
    return overlay


def save_per_target_overlays(frame, frame_idx, outputs, video_name, save_dir, alpha=0.5):
    """将单帧中识别到的每个目标分别保存为掩码叠加原帧的图片，命名格式：video_name_f000_obj0.png"""
    if not outputs:
        return
    os.makedirs(save_dir, exist_ok=True)
    h, w = frame.shape[:2]
    masks = outputs.get('out_binary_masks', [])
    out_obj_ids = outputs.get('out_obj_ids', [])
    if len(masks) == 0:
        print(f"    帧 {frame_idx}: 无目标，未保存图片")
        return

    for obj_id, mask in zip(out_obj_ids, masks):
        overlay = frame.copy()
        if len(mask.shape) == 3:
            mask = mask.squeeze()
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
            mask = (mask > 0.5).astype(np.uint8)
        mask_bool = mask > 0.5
        for c in range(3):
            overlay[:, :, c][mask_bool] = (alpha * 255 * (c == 0) + (1 - alpha) * frame[:, :, c][mask_bool]).astype(
                np.uint8)
        filename = f"{video_name}_f{frame_idx:03d}_obj{int(obj_id)}.png"
        save_path = os.path.join(save_dir, filename)
        cv2.imwrite(save_path, overlay)
        print(f"    已保存目标图片: {save_path}")


def propagate_in_video(predictor, session_id):
    outputs_per_frame = {}
    for response in predictor.handle_stream_request(request=dict(type="propagate_in_video", session_id=session_id)):
        outputs_per_frame[response["frame_index"]] = response["outputs"]
    return outputs_per_frame


def propagate_in_video_tracker_only(predictor, session_id):
    outputs_per_frame = {}
    for response in predictor.handle_stream_request(request=dict(type="propagate_in_video_tracker_only", session_id=session_id, propagation_direction="both")):
        outputs_per_frame[response["frame_index"]] = response["outputs"]
    return outputs_per_frame



def get_box_prompts_from_frames(frames, image_model):
    """针对首帧、时序中间帧、尾帧分别调用单帧水位线识别，返回各帧像素Y坐标、框提示、mask及score"""
    frame_indices = sorted({0, len(frames) // 2, len(frames) - 1})
    waterline_ys = []
    box_prompts = []
    masks = []
    bboxes = []
    scores = []
    for frame_idx in frame_indices:
        frame = frames[frame_idx]

        # 将帧 numpy 数组转为 PIL 图像，直接用于单帧水位线识别 (无需临时文件)
        pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        success, final_y, final_mask, final_score = process_single_image(
            pil_image=pil_frame,
            model=image_model,
            prompt="black water"
        )

        if not success:
            raise ValueError(f"帧 {frame_idx} 水位线自动识别失败，无法生成Prompt")

        waterline_y = final_y
        h = frame.shape[0]
        y_top = waterline_y
        y_bottom = h
        y_norm = (y_top + y_bottom) / 2.0 / h
        h_norm = (y_bottom - y_top) / h
        frame_box = [[0.5, y_norm, 1.0, h_norm]]

        # 从 mask 计算 bbox [x1, y1, x2, y2]
        ys, xs = np.where(final_mask)
        if len(ys) > 0:
            bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        else:
            bbox = [0, waterline_y, frame.shape[1] - 1, h - 1]

        waterline_ys.append(waterline_y)
        box_prompts.append(frame_box)
        masks.append(final_mask)
        bboxes.append(bbox)
        scores.append(final_score)
        print(f"  - 帧 {frame_idx} 水位线 Y 像素坐标: {waterline_y}, 归一化框: {frame_box}, bbox: {bbox}, score: {final_score:.3f}")

    return frame_indices, waterline_ys, box_prompts, masks, bboxes, scores


def batch_process_videos_auto(input_dir, output_dir, checkpoint_path):
    """视频批量自动化分割流水线 (首/中/尾帧 -> 图像识别 -> 释放 -> 视频传播)"""
    print("\n" + "=" * 60)
    print("🚀 启动视频自动化分割流水线 (以图引频模式)")
    print("=" * 60)

    video_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.mp4')]
    if not video_files:
        print("❌ 未在输入目录找到 .mp4 文件！")
        return

    for idx, v_name in enumerate(video_files, 1):
        print(f"\n▶️ 正在处理视频 ({idx}/{len(video_files)}): {v_name}")
        v_path = os.path.join(input_dir, v_name)
        out_path = os.path.join(output_dir, v_name)

        # 1. 读取视频并抽取所有帧
        cap = cv2.VideoCapture(v_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret: break
            frames.append(frame)
        cap.release()

        if not frames:
            print(f"  ❌ 视频 {v_name} 无有效帧，跳过。")
            continue

        # 计算首帧 / 中间帧 / 尾帧
        last_idx = len(frames) - 1
        mid_idx = len(frames) // 2
        frame_indices = sorted({0, mid_idx, last_idx})
        print(f"  - 视频共 {len(frames)} 帧，提取首帧({frame_indices[0]})、中间帧({mid_idx})、尾帧({last_idx})进行图像分析...")

        image_model = None
        video_predictor = None
        session_id = None
        try:
            # 2. 初始化图像模型，对首/中/尾三帧分别做单帧水位线识别 (直接使用内存帧，无临时文件)
            print("  正在加载 SAM3 Image 模型进行单帧识别...")
            image_model = build_sam3_image_model(checkpoint_path=checkpoint_path)
            frame_indices, waterline_ys, box_prompts, masks, bboxes, single_frame_scores = get_box_prompts_from_frames(
                frames, image_model
            )
            print(f"  - 已获取各帧水位线 Y 坐标: {waterline_ys}")

            # 3. 释放 image_model 以释放内存，再初始化 video_predictor
            print("  正在释放 SAM3 Image 模型以释放内存...")
            del image_model
            image_model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            print("  正在加载 SAM3 Video 预测模型...")
            video_predictor = build_sam3_video_predictor(checkpoint_path)
            print("  SAM3 Video 模型加载完毕。")

            # 4. 启动视频会话
            session_id = video_predictor.handle_request(
                dict(type="start_session", resource_path=v_path)
            )["session_id"]

            outputs_per_frame = {}
            desktop_dir = os.path.join(os.path.expanduser("~"), "Desktop")
            tracker_only_failed = False

            try:
                # 4.1 尝试用单帧分割的 mask 直接初始化 tracker masklet（替代 detector）
                seeds = [
                    {
                        "frame_idx": fi,
                        "mask": mask,
                        "bbox": bbox,
                        "score": float(score),
                    }
                    for fi, mask, bbox, score in zip(frame_indices, masks, bboxes, single_frame_scores)
                ]
                video_predictor.handle_request(dict(
                    type="init_video_from_single_frame_seeds",
                    session_id=session_id,
                    seeds=seeds,
                ))
                print(f"  - 已用单帧分割 mask 初始化 {len(seeds)} 个关键帧的 tracker masklet")

                # 4.2 关键帧分割结果与保存（直接用单帧分割的 mask）
                print("  - 关键帧分割结果（来自单帧分割）：")
                for fi, mask, bbox in zip(frame_indices, masks, bboxes):
                    num_targets = 1 if mask.any() else 0
                    print(f"    帧 {fi}: bbox {bbox}, 识别到目标数: {num_targets}")
                    if num_targets > 0:
                        seed_outputs = {
                            "out_obj_ids": np.array([0], dtype=np.int64),
                            "out_binary_masks": mask.astype(np.uint8)[None, ...],
                        }
                        save_per_target_overlays(frames[fi], fi, seed_outputs, v_name, desktop_dir)

                # 4.3 tracker-only 传播（无 detector）
                print("  - 开始 tracker-only 时序传播...")
                outputs_per_frame = propagate_in_video_tracker_only(video_predictor, session_id)

            except Exception as e:
                tracker_only_failed = True
                print(f"  ⚠️ tracker-only 流失败，回退到原 detector 流: {e}")

            if tracker_only_failed:
                # 5. 回退：原 detector 流（一次性注入框提示）
                prompts = [
                    {
                        "frame_idx": fi,
                        "text_str": "water",
                        "boxes_xywh": box,
                        "box_labels": [1],
                    }
                    for fi, box in zip(frame_indices, box_prompts)
                ]
                add_prompts_response = video_predictor.handle_request(dict(
                    type="add_prompts", session_id=session_id, prompts=prompts
                ))
                print(f"  - 已一次性注入 {len(prompts)} 帧提示: water + {box_prompts}")

                # 5.1 关键帧分割结果与保存
                outputs_per_frame = add_prompts_response["outputs_per_frame"]
                print("  - 关键帧分割结果：")
                for fi, box in zip(frame_indices, box_prompts):
                    outputs = outputs_per_frame.get(fi, {})
                    out_obj_ids = outputs.get("out_obj_ids", [])
                    num_targets = len(out_obj_ids)
                    print(f"    帧 {fi}: 提示框 {box}，识别到目标数: {num_targets}")
                    save_per_target_overlays(frames[fi], fi, outputs, v_name, desktop_dir)

                # 5.2 原 detector 时序传播
                print("  - 开始原 detector 时序传播...")
                outputs_per_frame = propagate_in_video(video_predictor, session_id)

            # 6. 渲染与导出
            result_frames = []
            fallback_output = outputs_per_frame.get(0, None)
            for i, frame in enumerate(frames):
                out = outputs_per_frame.get(i, fallback_output)
                result_frames.append(overlay_mask(frame, out))

            save_frames_as_mp4(result_frames, out_path, fps)
            print(f"  ✅ 视频导出成功: {out_path}")

        except Exception as e:
            print(f"  ❌ 视频处理失败: {e}")

        finally:
            # 7. 无论如何都关闭会话、释放两个模型并回收显存，再进行下一个视频
            if session_id is not None and video_predictor is not None:
                try:
                    video_predictor.handle_request(dict(type="close_session", session_id=session_id))
                except Exception:
                    pass
            print("  正在关闭并释放 SAM3 Video 模型...")
            del video_predictor
            print("  正在释放 SAM3 Image 模型...")
            del image_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

    print("\n🎉 所有视频处理完毕！")


def main():
    # 1. 初始化路径配置
    paths =  {
        "checkpoint_path": r"E:\VS code\SAM3\sam3.pt",
        "input_video_folder": r"E:\VS code\data\water_video_ROI\video_segment_with_problem",
        "output_video_folder": r"E:\VS code\data\water_video_ROI\video_segment_with_problem\test"
    }
    checkpoint_path = paths["checkpoint_path"]

    # ==========================================
    # 视频全自动分割流水线 (先图片识别后释放，再视频传播)
    # ==========================================

    Path(paths["output_video_folder"]).mkdir(parents=True, exist_ok=True)
    batch_process_videos_auto(
        input_dir=paths["input_video_folder"],
        output_dir=paths["output_video_folder"],
        checkpoint_path=checkpoint_path
    )
    return  # 视频处理结束后，直接退出程序


if __name__ == "__main__":
    main()