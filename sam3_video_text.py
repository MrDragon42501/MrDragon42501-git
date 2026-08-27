"""
SAM3 视频分割 - 快速版
跳过matplotlib可视化，直接输出结果视频
"""

import os
import sys

sys.path.append(r"D:\sam3model")

# 先导入 PIL 避免 DLL 问题
from PIL import Image

import numpy as np
import cv2

from sam3.model_builder import build_sam3_video_predictor


def propagate_in_video(predictor, session_id):
    """传播分割结果到所有帧"""
    outputs_per_frame = {}
    for response in predictor.handle_stream_request(
            request=dict(
                type="propagate_in_video",
                session_id=session_id,
            )
    ):
        outputs_per_frame[response["frame_index"]] = response["outputs"]
    return outputs_per_frame


def read_video_frame(video_path):
    """读取视频所有帧"""
    video_frames = []
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        video_frames.append(frame)
    cap.release()
    return video_frames


def overlay_mask(frame, masks):
    """在帧上叠加掩码（红色标注）"""
    if masks is None or (isinstance(masks, (list, np.ndarray)) and len(masks) == 0):
        return frame

    # 合并所有掩码
    combined_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=bool)
    for mask in masks:
        m = mask.squeeze() if len(mask.shape) > 2 else mask
        combined_mask = np.logical_or(combined_mask, m)

    # 创建叠加效果
    overlay = frame.copy()
    overlay[combined_mask] = np.clip(
        (frame[combined_mask].astype(float) * 0.6 +
         np.array([0, 0, 255]) * 0.4).astype(np.uint8),
        0, 255
    )
    return overlay


def save_frames_as_mp4(frames, output_path, fps=30):
    """将帧列表保存为 MP4 视频"""
    if not frames:
        print("错误：没有帧可保存")
        return False

    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not video_writer.isOpened():
        print(f"错误：无法创建视频文件 {output_path}")
        return False

    for i, frame in enumerate(frames):
        video_writer.write(frame)
        if i % 20 == 0:
            print(f"  写入帧 {i + 1}/{len(frames)}")

    video_writer.release()
    print(f"\n视频已保存: {output_path}")
    return True


if __name__ == "__main__":
    # ========== 配置 ==========
    MODEL_PATH = "sam3.pt"
    VIDEO_PATH = r"C:\Users\71758\Desktop\大创资料\a6b88ddf15aa3bc6593b428611841a80.mp4"
    OUTPUT_VIDEO = r"C:\Users\71758\Desktop\大创资料\sam3_output_fast.mp4"
    PROMPT_TEXT = "water"
    PROMPT_FRAME = 0
    FPS = 30
    # ========================

    print("=" * 50)
    print("SAM3 视频分割")
    print("=" * 50)

    # 1. 加载模型
    print("\n[1/5] 加载模型...")
    predictor = build_sam3_video_predictor(MODEL_PATH)
    print("    完成")

    # 2. 读取视频
    print("\n[2/5] 读取视频...")
    video_frames = read_video_frame(VIDEO_PATH)
    print(f"    共 {len(video_frames)} 帧")

    # 3. 启动会话
    print("\n[3/5] 启动会话...")
    response = predictor.handle_request(dict(
        type="start_session",
        resource_path=VIDEO_PATH,
    ))
    session_id = response["session_id"]
    print(f"    会话ID: {session_id}")

    # 4. 添加提示
    print(f"\n[4/5] 添加提示: '{PROMPT_TEXT}'")
    response = predictor.handle_request(dict(
        type="add_prompt",
        session_id=session_id,
        frame_index=PROMPT_FRAME,
        text=PROMPT_TEXT,
    ))
    out = response["outputs"]
    print(f"    提示帧掩码数量: {len(out.get('out_binary_masks', []))}")

    # 5. 传播
    print("\n[5/5] 传播分割结果...")
    outputs_per_frame = propagate_in_video(predictor, session_id)
    print(f"    传播完成: {len(outputs_per_frame)} 帧")

    # 6. 生成可视化结果
    print("\n[6/6] 生成结果视频...")
    result_frames = []
    for frame_idx in range(len(video_frames)):
        if frame_idx in outputs_per_frame:
            masks = outputs_per_frame[frame_idx].get("out_binary_masks", [])
            if len(masks) > 0:
                frame_with_mask = overlay_mask(video_frames[frame_idx], masks)
            else:
                frame_with_mask = video_frames[frame_idx]
        else:
            frame_with_mask = video_frames[frame_idx]
        result_frames.append(frame_with_mask)

    # 保存视频
    save_frames_as_mp4(result_frames, OUTPUT_VIDEO, FPS)

    # 关闭会话
    predictor.handle_request(dict(type="close_session", session_id=session_id))

    print("\n" + "=" * 50)
    print("完成!")
    print(f"结果: {OUTPUT_VIDEO}")
    print("=" * 50)