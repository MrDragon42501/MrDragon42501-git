"""
SAM3 视频分割 - 批量版（首尾双帧提示 + Excel读取框坐标）
功能：
1. 批量处理指定文件夹内所有 .mp4 文件
2. BOX_PROMPTS1 从第一个 Excel 表格读取：按文件名匹配，第3列为上Y、第4列为下Y
3. BOX_PROMPTS2 从第二个 Excel 表格读取：规则同上
4. 输出视频保存到指定文件夹，文件名与源文件一致
修复点：
- 修复numpy数组if probs布尔判断报错
- 读取视频原生帧率，输出视频速度和原视频一致
- 增加文件存在性校验，提前提示路径错误
- 优化路径获取，兼容脚本任意运行目录
- 完善空掩码、空帧鲁棒逻辑
"""
import os
import sys
# 自动获取脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
# 模型根目录路径
sys.path.append(r"D:\sam3model")
import torch
import numpy as np
import cv2
import pandas as pd
from sam3.model_builder import build_sam3_video_predictor


def propagate_in_video(predictor, session_id):
    """传播分割结果到所有帧"""
    outputs_per_frame = {}
    frame_count = 0
    for response in predictor.handle_stream_request(
        request=dict(
            type="propagate_in_video",
            session_id=session_id,
        )
    ):
        frame_idx = response["frame_index"]
        outputs = response["outputs"]
        outputs_per_frame[frame_idx] = outputs

        if frame_count < 5 or frame_count % 50 == 0:
            print(f"    [propagate] 帧{frame_idx}: keys={outputs.keys() if outputs else 'None'}")
            if outputs:
                print(f"        obj_ids={list(outputs.get('out_obj_ids', []))}")
                print(f"        masks_len={len(outputs.get('out_binary_masks', []))}")

        frame_count += 1

    print(f"    [propagate] 总计返回 {frame_count} 帧")
    return outputs_per_frame


def read_video_frames(video_path):
    """读取视频所有帧，同时返回原生真实帧率和尺寸"""
    video_frames = []
    cap = cv2.VideoCapture(video_path)
    real_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        video_frames.append(frame)
    cap.release()
    return video_frames, real_fps, frame_height, frame_width


def overlay_mask(frame, outputs, alpha=0.5):
    """在帧上叠加掩码"""
    if outputs is None:
        return frame

    overlay = frame.copy()
    height, width = frame.shape[:2]

    masks = outputs.get('out_binary_masks', [])
    for mask in masks:
        if len(mask.shape) == 3:
            mask = mask.squeeze()

        if mask.shape[:2] != (height, width):
            mask_resized = cv2.resize(mask.astype(np.float32),
                                      (width, height),
                                      interpolation=cv2.INTER_LINEAR)
            mask = (mask_resized > 0.5).astype(np.uint8)

        mask_bool = mask > 0.5
        if mask_bool.sum() == 0:
            continue

        # 蓝色 (BGR)
        for c in range(3):
            overlay[:, :, c][mask_bool] = (
                alpha * 255 * (c == 0) +
                (1 - alpha) * frame[:, :, c][mask_bool]
            ).astype(np.uint8)

    return overlay


def save_frames_as_mp4(frames, output_path, fps=30):
    """保存视频"""
    if not frames:
        print("错误：没有帧可保存")
        return False

    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not video_writer.isOpened():
        print(f"错误：无法创建视频文件，检查输出路径权限")
        return False

    for i, frame in enumerate(frames):
        video_writer.write(frame)
        if i % 20 == 0:
            print(f"  写入帧 {i+1}/{len(frames)}")

    video_writer.release()
    print(f"\n视频已保存: {output_path}")
    return True


def load_box_from_excel(excel_path, filename, frame_height):
    """
    从Excel表格中按文件名查找框坐标，转为归一化 [x, y, w, h] 格式
    Excel列：第1列=文件名，第3列=上端面Y，第4列=下端面Y（1-based）
    归一化框：水平方向全覆盖 (x=0.5, w=1.0)，垂直方向由上下Y坐标确定
    """
    df = pd.read_excel(excel_path, header=None)
    name_col = df.iloc[:, 0].astype(str).str.strip()
    base_name = os.path.splitext(filename)[0]  # 去掉扩展名

    # 尝试匹配：带扩展名 / 不带扩展名
    match_mask = (name_col == filename) | (name_col == base_name)
    matched = df[match_mask]

    if matched.empty:
        raise ValueError(f"Excel中未找到文件名匹配的行: {filename}")

    row = matched.iloc[0]
    y_top = float(row.iloc[2])      # 第3列（索引2）：上端面 Y
    y_bottom = float(row.iloc[3])   # 第4列（索引3）：下端面 Y

    # 转为归一化坐标 [x_center, y_center, width, height]
    x_norm = 0.5
    w_norm = 1.0
    y_norm = (y_top + y_bottom) / 2.0 / frame_height
    h_norm = (y_bottom - y_top) / frame_height

    return [[x_norm, y_norm, w_norm, h_norm]]


def process_single_video(predictor, video_path, output_path, excel1_path, excel2_path, prompt_text, box_labels):
    """处理单个视频"""
    video_name = os.path.basename(video_path)
    print(f"\n{'=' * 60}")
    print(f"处理视频: {video_name}")
    print(f"{'=' * 60}")

    # 1. 读取视频 + 获取原生帧率和尺寸
    print("\n[1/5] 读取视频...")
    video_frames, FPS, frame_height, frame_width = read_video_frames(video_path)
    total_frames = len(video_frames)
    last_frame_idx = total_frames - 1
    print(f"    共 {total_frames} 帧")
    print(f"    帧尺寸: {frame_width} x {frame_height}")
    print(f"    视频原生帧率: {FPS}")

    # 2. 从Excel读取框坐标（归一化）
    print("\n[2/5] 从Excel读取框坐标...")
    box_prompts1 = load_box_from_excel(excel1_path, video_name, frame_height)
    box_prompts2 = load_box_from_excel(excel2_path, video_name, frame_height)
    print(f"    首帧框 (Excel1): {box_prompts1}")
    print(f"    末帧框 (Excel2): {box_prompts2}")

    # 3. 启动会话
    print("\n[3/5] 启动会话...")
    response = predictor.handle_request(dict(
        type="start_session",
        resource_path=video_path,
    ))
    session_id = response["session_id"]
    print(f"    会话ID: {session_id}")

    # 4. 添加提示（第一帧 + 最后一帧 均添加 文本+框）
    print(f"\n[4/5] 添加提示...")
    print(f"    >>> 第 0 帧（首帧）: 文本='{prompt_text}' + 框={box_prompts1}")

    # 第一帧添加提示
    response = predictor.handle_request(dict(
        type="add_prompt",
        session_id=session_id,
        frame_index=0,
        text=prompt_text,
        bounding_boxes=box_prompts1,
        bounding_box_labels=box_labels,
    ))

    out = response["outputs"]
    masks = out.get('out_binary_masks', [])
    obj_ids = list(out.get('out_obj_ids', []))
    probs = out.get('out_probs', [])

    print(f"    首帧掩码数量: {len(masks)}")
    print(f"    首帧对象ID: {obj_ids}")
    if len(probs) > 0:
        print(f"    首帧置信度: {[f'{p:.3f}' for p in probs]}")

    # 最后一帧添加提示
    print(f"\n    >>> 第 {last_frame_idx} 帧（末帧）: 文本='{prompt_text}' + 框={box_prompts2}")
    response_last = predictor.handle_request(dict(
        type="add_prompt",
        session_id=session_id,
        frame_index=last_frame_idx,
        text=prompt_text,
        bounding_boxes=box_prompts2,
        bounding_box_labels=box_labels,
    ))

    out_last = response_last["outputs"]
    masks_last = out_last.get('out_binary_masks', [])
    obj_ids_last = list(out_last.get('out_obj_ids', []))
    probs_last = out_last.get('out_probs', [])

    print(f"    末帧掩码数量: {len(masks_last)}")
    print(f"    末帧对象ID: {obj_ids_last}")
    if len(probs_last) > 0:
        print(f"    末帧置信度: {[f'{p:.3f}' for p in probs_last]}")

    # 5. 传播 + 生成视频
    print("\n[5/5] 传播分割结果并生成视频...")
    outputs_per_frame = propagate_in_video(predictor, session_id)
    print(f"    传播完成: {len(outputs_per_frame)} 帧")

    # 生成结果帧
    result_frames = []
    first_frame_masks = masks
    first_frame_output = {
        'out_binary_masks': first_frame_masks,
        'out_obj_ids': obj_ids,
        'out_probs': probs if len(probs) > 0 else []
    }

    for frame_idx in range(len(video_frames)):
        if frame_idx in outputs_per_frame:
            frame_outputs = outputs_per_frame[frame_idx]
            frame_masks = frame_outputs.get('out_binary_masks', []) if frame_outputs else []
        else:
            frame_masks = []

        if len(frame_masks) == 0:
            frame_masks = first_frame_masks
            frame_outputs = first_frame_output

        if len(frame_masks) > 0:
            frame_with_mask = overlay_mask(video_frames[frame_idx], frame_outputs)
        else:
            frame_with_mask = video_frames[frame_idx]

        result_frames.append(frame_with_mask)

    # 保存视频
    save_frames_as_mp4(result_frames, output_path, FPS)

    # 关闭会话释放资源
    predictor.handle_request(dict(type="close_session", session_id=session_id))

    print(f"\n视频 {video_name} 处理完成!")


if __name__ == "__main__":
    # ========== 配置区（按需修改） ==========
    # 模型权重绝对路径
    MODEL_PATH = r"E:\VS code\SAM3\sam3.pt"
    # 输入视频文件夹（批量处理该目录下所有 .mp4）
    INPUT_DIR = r"E:\VS code\data\water_video_ROI\video_segment_with_problem"
    # 输出视频文件夹（结果保存在此，文件名与源文件相同）
    OUTPUT_DIR = r"E:\VS code\data\water_video_ROI\video_output"
    # 第一个 Excel 表格路径（BOX_PROMPTS1 / 首帧框）
    EXCEL1_PATH = r"E:\VS code\data\water_video_ROI\first_high.xlsx"
    # 第二个 Excel 表格路径（BOX_PROMPTS2 / 末帧框）
    EXCEL2_PATH = r"E:\VS code\data\water_video_ROI\last_high.xlsx"
    # 文本提示
    PROMPT_TEXT = "water"
    BOX_LABELS = [1]  # 1表示正样本（要分割的对象）
    # =======================================

    # 前置校验
    if not os.path.isfile(MODEL_PATH):
        print(f"【致命错误】模型文件不存在：{MODEL_PATH}")
        sys.exit(1)
    if not os.path.isdir(INPUT_DIR):
        print(f"【致命错误】输入文件夹不存在：{INPUT_DIR}")
        sys.exit(1)
    if not os.path.isfile(EXCEL1_PATH):
        print(f"【致命错误】Excel1 不存在：{EXCEL1_PATH}")
        sys.exit(1)
    if not os.path.isfile(EXCEL2_PATH):
        print(f"【致命错误】Excel2 不存在：{EXCEL2_PATH}")
        sys.exit(1)

    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 收集所有 mp4 文件
    video_files = sorted([
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith('.mp4')
    ])
    if not video_files:
        print(f"【错误】输入文件夹中没有找到 .mp4 文件：{INPUT_DIR}")
        sys.exit(1)

    print("=" * 60)
    print("SAM3 视频分割 - 批量版（Excel读取框坐标 + 首尾双帧提示）")
    print("=" * 60)
    print(f"\n输入目录: {INPUT_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"Excel1 (首帧框): {EXCEL1_PATH}")
    print(f"Excel2 (末帧框): {EXCEL2_PATH}")
    print(f"文本提示: '{PROMPT_TEXT}'")
    print(f"待处理视频数: {len(video_files)}")
    for vf in video_files:
        print(f"  - {vf}")

    # 加载模型（只加载一次）
    print("\n加载模型...")
    predictor = build_sam3_video_predictor(MODEL_PATH)
    print("模型加载完成\n")

    # 批量处理
    success_count = 0
    fail_count = 0
    fail_list = []

    for idx, video_name in enumerate(video_files, 1):
        print(f"\n【进度 {idx}/{len(video_files)}】", end="")
        video_path = os.path.join(INPUT_DIR, video_name)
        output_path = os.path.join(OUTPUT_DIR, video_name)

        try:
            process_single_video(
                predictor=predictor,
                video_path=video_path,
                output_path=output_path,
                excel1_path=EXCEL1_PATH,
                excel2_path=EXCEL2_PATH,
                prompt_text=PROMPT_TEXT,
                box_labels=BOX_LABELS,
            )
            success_count += 1
        except Exception as e:
            fail_count += 1
            fail_list.append((video_name, str(e)))
            print(f"\n【错误】视频 {video_name} 处理失败: {e}")
            continue

    # 汇总
    print("\n" + "=" * 60)
    print("全部处理完成!")
    print(f"成功: {success_count} 个")
    print(f"失败: {fail_count} 个")
    if fail_list:
        print("\n失败列表:")
        for name, err in fail_list:
            print(f"  - {name}: {err}")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 60)
