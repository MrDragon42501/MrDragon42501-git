# -*- coding: utf-8 -*-
"""
相机内参标定程序（棋盘格）
==================================================
功能：
  1. 从指定文件夹读取若干张带棋盘格的照片（支持中文路径）；
  2. 自动检测棋盘格角点，使用 OpenCV 标定相机内参；
  3. 将标定结果按指定格式输出到终端；
  4. 将结果保存为 txt 文件到照片所在文件夹下。

用法：
  python 相机内参标定_棋盘格.py
  然后按提示输入照片文件夹路径（也可直接作为命令行参数传入）。

依赖：
  pip install opencv-python numpy
"""

import os
import sys
import glob
from datetime import datetime

import numpy as np
import cv2

# ======================== 配置区（请按实际修改） ========================
# 棋盘格内角点（交叉点）数目 = (每行黑白交叉点数, 每列黑白交叉点数)
# 即 (棋盘格横向格子数 - 1, 纵向格子数 - 1)。例如 10x7 格棋盘 => (9, 6)
CHESSBOARD_INNER_CORNERS = (11, 8)

# 棋盘格单格边长（毫米），用于建立真实世界坐标
SQUARE_SIZE_MM = 15.0

# 像元物理尺寸（m/px）。注意：该值无法由标定得出，需查相机传感器规格表
# 手动填写。若传感器为 1/2.7 英寸 800 万像素等，常见值约为 1.4e-6。
PIXEL_SIZE_M = 1.4e-6

# 最少成功检测张数（建议 >= 10，越多越稳）
MIN_SUCCESS_IMAGES = 12

# 支持的图片扩展名
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
# ========================================================================


def imread_unicode(path: str):
    """读取图片，支持中文路径（cv2.imread 不支持中文）。"""
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path: str, img):
    """保存图片，支持中文路径。"""
    ext = os.path.splitext(path)[1]
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(path)
    return ok


def fmt_sci(x: float) -> str:
    """把 1.4e-06 格式化为 1.4e-6。"""
    mant, exp = f"{x:.1e}".split("e")
    return f"{mant}e{int(exp)}"


def collect_image_paths(folder: str) -> list:
    paths = []
    for ext in IMAGE_EXTS:
        paths.extend(glob.glob(os.path.join(folder, "*" + ext)))
    # 排除上次运行保存的角点标注图，避免被重复检测
    paths = [p for p in paths if not os.path.basename(p).startswith("角点检测_")]
    return sorted(paths)


def build_result_block(fx, fy, cx, cy, k1, k2, p1, p2) -> str:
    """按指定格式生成标定结果文本块。"""
    return "\n".join([
        f"    fx={fx:.6f},      # 焦距 x（像素）",
        f"    fy={fy:.6f},      # 焦距 y（像素）",
        f"    cx={cx:.6f},      # 主点 x",
        f"    cy={cy:.6f},      # 主点 y",
        f"    k1={k1:.6f},        # 径向畸变",
        f"    k2={k2:.6f},",
        f"    p1={p1:.6f},        # 切向畸变",
        f"    p2={p2:.6f},",
        f"    pixel_size_s={fmt_sci(PIXEL_SIZE_M)}  # 像元物理尺寸（m/px）",
    ])


def main():
    # -------- 1. 确定照片文件夹路径 --------
    folder = r"C:\Users\ShenYuLong\Desktop\photo_test" #照片文件夹路径
    if len(sys.argv) > 1:
        folder = sys.argv[1].strip().strip('"')
    if not folder:
        folder = input("请输入存放棋盘格照片的文件夹路径：").strip().strip('"')
    if not os.path.isdir(folder):
        print(f"[错误] 文件夹不存在：{folder}")
        return

    # -------- 2. 收集图片 --------
    image_paths = collect_image_paths(folder)
    if not image_paths:
        print("[错误] 文件夹中没有找到任何图片。")
        return
    print(f"[信息] 共找到 {len(image_paths)} 张图片。")

    cols, rows = CHESSBOARD_INNER_CORNERS
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # 棋盘格角点的真实世界坐标（单位：米），z 恒为 0
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_MM / 1000.0

    find_flags = (cv2.CALIB_CB_ADAPTIVE_THRESH
                  + cv2.CALIB_CB_NORMALIZE_IMAGE
                  + getattr(cv2, "CALIB_CB_FILTER_QUASI", 0))

    obj_points = []   # 世界坐标
    img_points = []   # 图像坐标
    used_paths = []   # 成功检测的图片
    failed_paths = []
    image_size = None

    # -------- 3. 逐张检测角点 --------
    for p in image_paths:
        img = imread_unicode(p)
        if img is None:
            print(f"  [跳过] 无法读取：{os.path.basename(p)}")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = gray.shape[::-1]

        found, corners = cv2.findChessboardCorners(gray, (cols, rows), None, find_flags)
        if not found:
            failed_paths.append(p)
            print(f"  [失败] 未检测到 {cols}x{rows} 内角点：{os.path.basename(p)}")
            continue

        corners_sub = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        obj_points.append(objp)
        img_points.append(corners_sub)
        used_paths.append(p)
        print(f"  [成功] {os.path.basename(p)}  角点数={len(corners_sub)}")

        # 保存带角点标注的成功检测图（同一文件夹下）
        disp = cv2.drawChessboardCorners(img.copy(), (cols, rows), corners_sub, found)
        save_name = "角点检测_" + os.path.basename(p)
        save_path = os.path.join(folder, save_name)
        if imwrite_unicode(save_path, disp):
            print(f"    [已保存] {save_name}")

        # 可选：可视化检测结果（无 GUI 环境自动跳过）
        try:
            cv2.imshow("chessboard", cv2.resize(disp, (960, 540)))
            cv2.waitKey(50)
        except cv2.error:
            pass

    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass

    # -------- 4. 标定 --------
    print(f"\n[信息] 成功检测 {len(used_paths)} 张，失败 {len(failed_paths)} 张。")
    if len(used_paths) < 3:
        print("[错误] 成功检测的图片不足 3 张，无法标定。")
        print("       请检查 CHESSBOARD_INNER_CORNERS 是否与棋盘格一致，")
        print("       或换用更清晰、光照更均匀的照片。")
        return
    if len(used_paths) < MIN_SUCCESS_IMAGES:
        print(f"[警告] 成功检测数({len(used_paths)})少于建议值 {MIN_SUCCESS_IMAGES}，"
              "标定结果可能不稳定，建议增加照片。")

    rms, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, image_size, None, None)

    fx, fy = mtx[0, 0], mtx[1, 1]
    cx, cy = mtx[0, 2], mtx[1, 2]
    if len(dist) > 0 and len(dist[0]) >= 4:
        k1, k2, p1, p2 = dist[0][:4]
    else:
        k1 = k2 = p1 = p2 = 0.0

    # -------- 5. 输出结果 --------
    block = build_result_block(fx, fy, cx, cy, k1, k2, p1, p2)

    print("\n============================================================")
    print("  相机内参标定结果")
    print("============================================================")
    print(f"  成功张数       : {len(used_paths)}")
    print(f"  图像尺寸       : {image_size[0]} x {image_size[1]}")
    print(f"  重投影误差 RMS : {rms:.4f} px   (越小越好，通常 < 0.5)")
    print("------------------------------------------------------------")
    print(block)
    print("============================================================\n")

    # -------- 6. 保存为 txt（照片所在文件夹） --------
    txt_path = os.path.join(folder, "相机内参标定结果.txt")
    header = (
        "相机内参标定结果\n"
        "========================================\n"
        f"标定时间        : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"成功检测张数    : {len(used_paths)}\n"
        f"图像尺寸        : {image_size[0]} x {image_size[1]}\n"
        f"重投影误差 RMS  : {rms:.4f} px\n"
        f"棋盘格内角点数  : {cols} x {rows}\n"
        f"方格边长        : {SQUARE_SIZE_MM:.1f} mm\n"
        "可直接复制以下内容到 CameraParams：\n"
        "----------------------------------------\n"
    )
    with open(txt_path, "w", encoding="utf-8-sig") as f:
        f.write(header)
        f.write(block)
        f.write("\n")
    print(f"[完成] 结果已保存到：{txt_path}")


if __name__ == "__main__":
    main()
