import os
import cv2

def cut_last_3_seconds_opencv(mp4_path):
    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        print(f"[失败] 无法打开：{os.path.basename(mp4_path)}")
        return False

    # 获取视频基本信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 时长不足3秒则跳过
    if total_frames / fps <= 3:
        print(f"[跳过] 时长不足3秒：{os.path.basename(mp4_path)}")
        cap.release()
        return False

    # 计算最后3秒对应的起始帧
    start_frame = int(total_frames - fps * 3)
    if start_frame < 0:
        start_frame = 0

    temp_path = mp4_path + ".tmp.mp4"
    # MP4 编码格式，Windows 通用
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        print(f"[失败] 无法创建输出文件：{os.path.basename(mp4_path)}")
        cap.release()
        return False

    try:
        # 跳转到起始帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)

        writer.release()
        cap.release()

        # 覆盖原文件
        os.replace(temp_path, mp4_path)
        print(f"[成功] 保留最后3秒：{os.path.basename(mp4_path)}")
        return True

    except Exception as e:
        print(f"[失败] {os.path.basename(mp4_path)}：{str(e)}")
        cap.release()
        writer.release()
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return False

def process_folder(folder):
    if not os.path.isdir(folder):
        print("文件夹不存在")
        return

    files = [f for f in os.listdir(folder) if f.lower().endswith(".mp4")]
    print(f"共找到 {len(files)} 个 MP4，开始处理...\n")

    for f in files:
        full_path = os.path.join(folder, f)
        cut_last_3_seconds_opencv(full_path)

    print("\n全部处理完成！")

if __name__ == "__main__":
    # 你的目标文件夹
    TARGET_FOLDER = r"E:\VS code\data\water_video_ROI\video_segment_with_problem"
    process_folder(TARGET_FOLDER)