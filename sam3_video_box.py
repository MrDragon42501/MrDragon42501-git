"""
SAM3 视频分割 - 文本提示 + 框提示组合版
修复点：
1. 修复numpy数组if probs布尔判断报错
2. 读取视频原生帧率，输出视频速度和原视频一致
3. 增加文件存在性校验，提前提示路径错误
4. 优化路径获取，兼容脚本任意运行目录
5. 完善空掩码、空帧鲁棒逻辑
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
    """读取视频所有帧，同时返回原生真实帧率"""
    video_frames = []
    cap = cv2.VideoCapture(video_path)
    real_fps = cap.get(cv2.CAP_PROP_FPS)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        video_frames.append(frame)
    cap.release()
    return video_frames, real_fps


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


if __name__ == "__main__":
    # ========== 配置区（按需修改） ==========
    # 模型权重绝对路径，根据你实际存放位置修改
    MODEL_PATH = r"E:\VS code\SAM3\sam3.pt"
    VIDEO_PATH = r"C:\Users\ShenYuLong\Desktop\water_video_ROI\0005.mp4"
    OUTPUT_VIDEO = r"C:\Users\ShenYuLong\Desktop\water_video_ROI\0005(1).mp4"
    # 文本提示
    PROMPT_TEXT = "water"
    # 框提示 - 归一化坐标 [x, y, w, h]
    BOX_PROMPTS = [[0.5, 0.685, 1.0, 0.63]]
    BOX_LABELS = [1]  # 1表示正样本（要分割的对象）
    PROMPT_FRAME = 0
    # =======================================

    # 前置文件校验，提前报错
    if not os.path.isfile(MODEL_PATH):
        print(f"【致命错误】模型文件不存在：{MODEL_PATH}")
        sys.exit(1)
    if not os.path.isfile(VIDEO_PATH):
        print(f"【致命错误】输入视频不存在：{VIDEO_PATH}")
        sys.exit(1)

    print("=" * 50)
    print("SAM3 视频分割 - 文本提示 + 框提示组合版")
    print("=" * 50)
    print(f"\n文本提示: '{PROMPT_TEXT}'")
    print(f"框提示: {BOX_PROMPTS}")

    # 1. 加载模型
    print("\n[1/6] 加载模型...")
    predictor = build_sam3_video_predictor(MODEL_PATH)
    print("    完成")

    # 2. 读取视频 + 获取原生帧率
    print("\n[2/6] 读取视频...")
    video_frames, FPS = read_video_frames(VIDEO_PATH)
    print(f"    共 {len(video_frames)} 帧")
    print(f"    帧形状: {video_frames[0].shape}")
    print(f"    视频原生帧率: {FPS}")

    # 3. 启动会话
    print("\n[3/6] 启动会话...")
    response = predictor.handle_request(dict(
        type="start_session",
        resource_path=VIDEO_PATH,
    ))
    session_id = response["session_id"]
    print(f"    会话ID: {session_id}")

    # 4. 添加提示（文本 + 框）
    print(f"\n[4/6] 添加提示...")
    print(f"    文本: '{PROMPT_TEXT}'")
    print(f"    框: {BOX_PROMPTS}")

    response = predictor.handle_request(dict(
        type="add_prompt",
        session_id=session_id,
        frame_index=PROMPT_FRAME,
        text=PROMPT_TEXT,
        bounding_boxes=BOX_PROMPTS,
        bounding_box_labels=BOX_LABELS,
    ))

    out = response["outputs"]
    masks = out.get('out_binary_masks', [])
    obj_ids = list(out.get('out_obj_ids', []))
    probs = out.get('out_probs', [])

    print(f"    掩码数量: {len(masks)}")
    print(f"    对象ID: {obj_ids}")
    # 修复：用长度判断替代直接判断数组，消除ValueError
    if len(probs) > 0:
        print(f"    置信度: {[f'{p:.3f}' for p in probs]}")

    # 5. 传播
    print("\n[5/6] 传播分割结果...")
    outputs_per_frame = propagate_in_video(predictor, session_id)
    print(f"    传播完成: {len(outputs_per_frame)} 帧")

    # 检查传播结果
    print("\n    传播结果检查:")
    for idx in [0, 50, 100, 150]:
        if idx in outputs_per_frame:
            sample = outputs_per_frame[idx]
            count = len(sample.get('out_binary_masks', []))
            ids = list(sample.get('out_obj_ids', []))
            print(f"    第{idx}帧: 掩码{count}个, ID{ids}")

    # 6. 生成视频
    print("\n[6/6] 生成结果视频...")

    result_frames = []

    # 使用第一帧结果作为静态掩码兜底
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

        # 如果传播结果为空，使用第一帧的静态掩码兜底
        if len(frame_masks) == 0:
            frame_masks = first_frame_masks
            frame_outputs = first_frame_output

        if len(frame_masks) > 0:
            frame_with_mask = overlay_mask(video_frames[frame_idx], frame_outputs)
        else:
            frame_with_mask = video_frames[frame_idx]

        result_frames.append(frame_with_mask)

        if frame_idx == 0:
            cv2.imwrite(r"C:\Users\ShenYuLong\Desktop\water-flow(1).png", frame_with_mask)
            print(f"    第一帧已保存")

    # 保存视频（使用原视频真实帧率，不再写死30）
    save_frames_as_mp4(result_frames, OUTPUT_VIDEO, FPS)

    # 关闭会话释放资源
    predictor.handle_request(dict(type="close_session", session_id=session_id))

    print("\n" + "=" * 50)
    print("全部执行完成!")
    print(f"结果视频路径: {OUTPUT_VIDEO}")
    print("=" * 50)