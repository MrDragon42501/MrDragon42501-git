import os
import cv2
import math
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Optional


# =========================================================
# 1. 参数类
# =========================================================
@dataclass
class CameraParams:
    """
    相机参数
    fx, fy: 焦距（px）
    k1, k2, p1, p2: 畸变系数
    pixel_size_s: 像素物理尺寸（m/px）
    """
    fx: float
    fy: float
    cx: float
    cy: float
    k1: float
    k2: float
    p1: float
    p2: float
    pixel_size_s: float


@dataclass
class SiteParams:
    """
    测站几何参数
    Yc: 摄像机光心对应断面起点距
    Zc: 摄像机光心高程
    """
    Yc: float
    Zc: float


@dataclass
class GaugeParams:
    """
    虚拟水尺参数
    ruler_x_center: 虚拟水尺绘制中心横坐标（在整幅旋转图坐标系下）
    ruler_half_width: 虚拟水尺半宽
    gauge_z_min: 虚拟水尺最小高程（自动生成）
    gauge_z_max: 虚拟水尺最大高程（自动生成）
    gauge_step: 虚拟水尺细刻度间隔
    """
    ruler_x_center: int
    ruler_half_width: int
    gauge_z_min: float
    gauge_z_max: float
    gauge_step: float


# =========================================================
# 2. 中文路径读写
# =========================================================
def load_image_chinese(path: str, flag=cv2.IMREAD_COLOR) -> np.ndarray:
    """
    支持中文路径读取图片
    """
    try:
        img_np = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(img_np, flag)
    except Exception as e:
        raise FileNotFoundError(f"读取图片失败: {path} | 错误: {str(e)}")

    if img is None:
        raise FileNotFoundError(f"无法解析图片: {path}")
    return img


def save_image_chinese(path: str, img: np.ndarray) -> None:
    """
    支持中文路径保存图片
    """
    ext = os.path.splitext(path)[1]
    if ext == "":
        ext = ".jpg"
        path = path + ext

    success, img_encode = cv2.imencode(ext, img)
    if not success:
        raise ValueError(f"图片编码失败: {path}")

    img_encode.tofile(path)


# =========================================================
# 3. 断面地形数据处理
# =========================================================
def load_section_csv(section_csv: str) -> pd.DataFrame:
    """
    读取断面地形数据
    第一列：起点距Y
    第二列：高程Z
    """
    df = pd.read_csv(section_csv, header=None)
    if df.shape[1] < 2:
        raise ValueError("断面地形CSV至少需要两列：第一列起点距，第二列高程")
    df = df.iloc[:, :2].copy()
    df.columns = ["Y", "Z"]
    df = df.sort_values("Y").reset_index(drop=True)
    return df


def extract_monotonic_bank_suffix(df: pd.DataFrame, eps: float = 0.01) -> pd.DataFrame:
    """
    提取对岸单调有效观测段
    从末端向前扫描，保留高程单调递增后缀
    """
    Z = df["Z"].values
    start_idx = len(df) - 2
    for i in range(len(df) - 2, -1, -1):
        if Z[i + 1] - Z[i] >= eps:
            start_idx = i
        else:
            break

    out = df.iloc[start_idx:].copy().reset_index(drop=True)
    if len(out) < 2:
        raise ValueError("提取的有效观测段点数不足")
    return out


def interpolate_Y_from_Z(valid_df: pd.DataFrame, Z_query: float) -> Optional[float]:
    """
    在单调段中根据高程Z反求起点距Y
    """
    Y = valid_df["Y"].values
    Z = valid_df["Z"].values

    if Z_query < Z.min() or Z_query > Z.max():
        return None

    idx = np.searchsorted(Z, Z_query, side="right") - 1
    idx = max(0, min(idx, len(Z) - 2))

    Z0, Z1 = Z[idx], Z[idx + 1]
    Y0, Y1 = Y[idx], Y[idx + 1]

    if abs(Z1 - Z0) < 1e-12:
        return float(Y0)

    return float(Y0 + (Y1 - Y0) * (Z_query - Z0) / (Z1 - Z0))


def build_dense_section_by_Z(valid_df: pd.DataFrame, z_min: float, z_max: float, z_step: float) -> pd.DataFrame:
    """
    按高程等间隔生成稠密地形点
    这里的高程序列不是从 z_min 开始，而是从 0.00m 的整刻度体系对齐
    例如 step=0.01 时，高程序列为：
        ..., 0.00, 0.01, 0.02, ...
    然后截取 [z_min, z_max] 范围内的点
    """
    z_valid_min = float(valid_df["Z"].min())
    z_valid_max = float(valid_df["Z"].max())

    print(f"[DEBUG] 请求生成稠密点高程范围: {z_min:.6f} ~ {z_max:.6f}")
    print(f"[DEBUG] 有效断面高程范围: {z_valid_min:.6f} ~ {z_valid_max:.6f}")

    if z_max < z_valid_min or z_min > z_valid_max:
        raise ValueError(
            f"虚拟水尺高程范围 [{z_min:.6f}, {z_max:.6f}] 与有效断面高程范围 "
            f"[{z_valid_min:.6f}, {z_valid_max:.6f}] 不重合，无法生成稠密断面点"
        )

    z_values = build_aligned_z_list(z_min, z_max, z_step, start_base=0.0)

    rows = []
    for z in z_values:
        y = interpolate_Y_from_Z(valid_df, z)
        if y is not None:
            rows.append([y, z])

    out = pd.DataFrame(rows, columns=["Y", "Z"])
    if len(out) == 0:
        raise ValueError("无法生成有效稠密断面点，请检查高程范围设置是否正确")
    return out


def build_aligned_z_list(z_min: float, z_max: float, step: float, start_base: float = 0.0):
    """
    在 [z_min, z_max] 内，生成从 start_base 对齐的刻度序列
    例如：
        start_base=0.0, step=0.05
        -> ..., 0.00, 0.05, 0.10, ...
    """
    if step <= 0:
        raise ValueError("step 必须 > 0")

    k0 = math.ceil((z_min - start_base) / step)
    first = start_base + k0 * step

    z_list = []
    z = first
    while z <= z_max + 1e-12:
        z_list.append(round(z, 6))
        z += step
    return z_list


# =========================================================
# 4. 图像畸变校正
# =========================================================
def undistort_image(img: np.ndarray, cam: CameraParams) -> Tuple[np.ndarray, np.ndarray]:
    """
    图像畸变校正
    返回：
        undistorted: 畸变校正图
        intrinsics: 内参矩阵
    """
    intrinsics = np.eye(3, dtype=np.float32)
    intrinsics[0, 0] = cam.fx
    intrinsics[1, 1] = cam.fy
    intrinsics[0, 2] = cam.cx
    intrinsics[1, 2] = cam.cy

    distortion_coeff = np.array(
        [cam.k1, cam.k2, cam.p1, cam.p2, 0],
        dtype=np.float32
    )

    undistorted = cv2.undistort(img, intrinsics, distortion_coeff)

    # 与原逻辑保持一致：统一转灰度后再回BGR
    undistorted = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
    undistorted = cv2.cvtColor(undistorted, cv2.COLOR_GRAY2BGR)

    return undistorted, intrinsics


# =========================================================
# 5. 横滚角与水平校正
# =========================================================
def compute_roll_angle(pt0: Tuple[float, float], pt1: Tuple[float, float]) -> float:
    """
    根据当前水面边缘两点计算横滚角
    φ = - arctan((v1-v0)/(u1-u0))
    返回弧度
    """
    u0, v0 = pt0
    u1, v1 = pt1
    if abs(u1 - u0) < 1e-12:
        raise ValueError("两点横坐标过近，无法计算横滚角")
    phi = -math.atan((v1 - v0) / (u1 - u0))
    return phi


def rotate_image_keep_all(img: np.ndarray, angle_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    图像旋转并扩展画布，避免裁剪
    返回：
        rotated_img
        M: 仿射矩阵
    """
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)

    cosv = abs(M[0, 0])
    sinv = abs(M[0, 1])

    new_w = int(h * sinv + w * cosv)
    new_h = int(h * cosv + w * sinv)

    M[0, 2] += new_w / 2.0 - cx
    M[1, 2] += new_h / 2.0 - cy

    rotated = cv2.warpAffine(img, M, (new_w, new_h))
    return rotated, M


def transform_point(M: np.ndarray, pt: Tuple[float, float]) -> Tuple[float, float]:
    """
    用仿射矩阵变换点
    """
    u, v = pt
    uv1 = np.array([u, v, 1.0], dtype=np.float64)
    out = M @ uv1
    return float(out[0]), float(out[1])


# =========================================================
# 6. 俯仰角计算
# =========================================================
def compute_pitch_angle(
    v_line_star: float,
    reference_water_level: float,
    valid_df: pd.DataFrame,
    site: SiteParams,
    pixel_size_s: float,
    image_height_hp: int,
    focal_length_f: float
) -> float:
    """
    根据当前参考水位计算俯仰角
    """
    Ys_ref = interpolate_Y_from_Z(valid_df, reference_water_level)
    if Ys_ref is None:
        raise ValueError("参考水位超出有效观测段范围")

    beta = math.atan((site.Zc - reference_water_level) / (Ys_ref - site.Yc))
    alpha = math.atan(pixel_size_s * (v_line_star - image_height_hp / 2.0) / focal_length_f)
    omega = beta - alpha
    return omega


# =========================================================
# 7. 虚拟水尺与查找表
# =========================================================
def project_level_to_vstar(
    Z_sj: float,
    Y_sj: float,
    site: SiteParams,
    omega: float,
    pixel_size_s: float,
    image_height_hp: int,
    focal_length_f: float
) -> float:
    """
    物理高程 -> 水平校正图像像素纵坐标
    """
    geom_angle = math.atan((site.Zc - Z_sj) / (Y_sj - site.Yc))
    v_star = image_height_hp / 2.0 + (focal_length_f / pixel_size_s) * math.tan(geom_angle - omega)
    return float(v_star)


def build_lookup_table_from_original_section(
    valid_df: pd.DataFrame,
    site: SiteParams,
    omega: float,
    pixel_size_s: float,
    image_height_hp: int,
    focal_length_f: float
) -> pd.DataFrame:
    """
    用原始有效断面采样点生成查找表：Z, Y, v_star
    为了便于按 Z 插值，这里按 Z 排序
    """
    rows = []
    for _, row in valid_df.iterrows():
        Y_sj = float(row["Y"])
        Z_sj = float(row["Z"])
        v_star = project_level_to_vstar(
            Z_sj=Z_sj,
            Y_sj=Y_sj,
            site=site,
            omega=omega,
            pixel_size_s=pixel_size_s,
            image_height_hp=image_height_hp,
            focal_length_f=focal_length_f
        )
        rows.append([Z_sj, Y_sj, v_star])

    lookup_df = pd.DataFrame(rows, columns=["Z", "Y", "v_star"])
    lookup_df = lookup_df.sort_values("Z").reset_index(drop=True)
    return lookup_df

def draw_virtual_gauge_on_full_image_in_roi(
    img_rot: np.ndarray,
    lookup_df: pd.DataFrame,
    roi: Tuple[int, int, int, int],
    z_min: float,
    z_max: float,
    step_main: float = 0.05,
    color=(0, 0, 255),
    text_color=(0, 255, 0),
    axis_offset_right: int = 3,
    tick_len_main: int = 12,
    tick_thickness: int = 2,
    draw_roi_box: bool = True,
    draw_axis: bool = True
) -> np.ndarray:
    """
    在校正后的整幅图上，仅在 ROI 区域内绘制虚拟水尺（右侧短刻度样式）
    - 水尺主轴放在 ROI 右侧
    - 刻度只向左画，不再横穿整个 ROI
    - 标注放在刻度左侧
    - 每隔 step_main 一个主刻度
    """
    out = img_rot.copy()
    h_img, w_img = out.shape[:2]

    x, y, w, h = roi

    # ROI 右侧主轴位置，留少量边距
    x_axis = min(x + w - 1, x + w - axis_offset_right)
    x_axis = max(x, x_axis)

    # 刻度长度不能超过 ROI 宽度
    tick_len = max(4, min(tick_len_main, max(1, x_axis - x)))

    z_arr = lookup_df["Z"].values.astype(np.float64)
    vstar_arr = lookup_df["v_star"].values.astype(np.float64)

    # 右侧竖直主轴
    if draw_axis:
        cv2.line(out, (x_axis, y), (x_axis, y + h - 1), color, 1)

    z_list = build_aligned_z_list(z_min, z_max, step_main, start_base=0.0)

    for z in z_list:
        v = np.interp(z, z_arr, vstar_arr)
        v = int(round(v))

        if y <= v < y + h and 0 <= v < h_img:
            # 只向左画短刻度
            x0 = max(x, x_axis - tick_len)
            x1 = x_axis
            cv2.line(out, (x0, v), (x1, v), color, tick_thickness)

            # 标注放刻度左侧
            text_x = max(x, x0 - 42)
            text_y = min(max(v + 5, 15), h_img - 5)
            cv2.putText(
                out, f"{z:.2f}",
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,text_color, 1, cv2.LINE_AA
            )

    if draw_roi_box:
        cv2.rectangle(out, (x, y), (x + w - 1, y + h - 1), (255, 255, 0), 2)

    return out


def draw_virtual_gauge_on_roi_ruler_style(
    roi_img: np.ndarray,
    lookup_df: pd.DataFrame,
    roi_offset_y: int,
    z_min: float,
    z_max: float,
    ruler_x_center: Optional[int] = None,
    ruler_half_width: int = 12,
    short_step: float = 0.01,
    mid_step: float = 0.05,
    long_step: float = 0.10,
    color=(0, 0, 255),
    text_color=(0, 255, 0),
    axis_offset_right: int = 3,
    short_ratio: float = 0.20,
    mid_ratio: float = 0.40,
    long_ratio: float = 0.60,
    min_short_len: int = 4,
    min_mid_len: int = 7,
    min_long_len: int = 10
) -> np.ndarray:
    """
    在 ROI 图像中绘制靠右直尺样式虚拟水尺：
      - 0.01m: 短刻度
      - 0.05m: 中刻度
      - 0.10m: 长刻度 + 标注
    样式：
      - 主轴在右侧
      - 刻度只向左画
      - 标注在刻度左边
    """
    out = roi_img.copy()
    h, w = out.shape[:2]

    # 若未指定位置，则默认贴右侧
    if ruler_x_center is None:
        x_center = w - axis_offset_right
    else:
        # 即使外部传了值，也限制在图像内
        x_center = int(np.clip(ruler_x_center, 0, w - 1))

    # 为了稳定成“靠右水尺”，再限制不要离右边太远
    x_center = min(x_center, w - axis_offset_right)
    x_center = max(0, x_center)

    # 可用于向左画刻度的最大长度
    max_len = min(ruler_half_width, x_center)
    max_len = max(8, max_len)

    # 这里就是你要控制“刻度短一点”的关键参数
    short_len = max(min_short_len, int(max_len * short_ratio))
    mid_len   = max(min_mid_len,   int(max_len * mid_ratio))
    long_len  = max(min_long_len,  int(max_len * long_ratio))

    z_arr = lookup_df["Z"].values.astype(np.float64)
    vstar_arr = lookup_df["v_star"].values.astype(np.float64)

    # 右侧主轴线
    cv2.line(out, (x_center, 0), (x_center, h - 1), color, 1)

    z_list = build_aligned_z_list(z_min, z_max, short_step, start_base=0.0)

    for z in z_list:
        v_global = np.interp(z, z_arr, vstar_arr)
        v_local = int(round(v_global - roi_offset_y))

        if not (0 <= v_local < h):
            continue

        is_long = abs((z / long_step) - round(z / long_step)) < 1e-6
        is_mid = abs((z / mid_step) - round(z / mid_step)) < 1e-6

        if is_long:
            tick_len = long_len
            thickness = 2
        elif is_mid:
            tick_len = mid_len
            thickness = 2
        else:
            tick_len = short_len
            thickness = 1

        # 只向左画刻度
        x0 = max(0, x_center - tick_len)
        x1 = x_center
        cv2.line(out, (x0, v_local), (x1, v_local), color, thickness)

        # 长刻度标注放左侧
        if is_long:
            text_x = max(2, x0 - 35)
            text_y = min(max(v_local + 5, 15), h - 5)
            cv2.putText(
                out, f"{z:.2f}",
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA
            )

    return out


# =========================================================
# 8. 鼠标交互：选两点计算横滚角
# =========================================================
clicked_points = []


def mouse_click_points(event, x, y, flags, param):
    global clicked_points
    img_show = param["img_show"]

    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))
        cv2.circle(img_show, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(
            img_show,
            f"({x}, {y})",
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )
        if len(clicked_points) >= 2:
            cv2.line(img_show, clicked_points[-2], clicked_points[-1], (255, 0, 0), 2)
        cv2.imshow(param["win_name"], img_show)


def select_two_points(img: np.ndarray, win_name: str = "Click 2 points on waterline") -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    鼠标点击两点，按ESC结束
    """
    global clicked_points
    clicked_points = []

    img_show = img.copy()
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win_name, mouse_click_points, {"img_show": img_show, "win_name": win_name})

    print("请在图像上点击两点作为当前水面参考线，按 ESC 结束。")

    while True:
        cv2.imshow(win_name, img_show)
        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            break

    cv2.destroyWindow(win_name)

    if len(clicked_points) < 2:
        raise ValueError("未成功选取两点")
    return clicked_points[0], clicked_points[1]


def input_two_points_from_console(img: np.ndarray) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    手动输入两点坐标（基于 undistorted 图像坐标系）
    """
    h, w = img.shape[:2]
    print(f"请输入两点坐标，当前图像尺寸为 W={w}, H={h}")

    x1 = float(input("点1 x1 = ").strip())
    y1 = float(input("点1 y1 = ").strip())
    x2 = float(input("点2 x2 = ").strip())
    y2 = float(input("点2 y2 = ").strip())

    if not (0 <= x1 < w and 0 <= y1 < h):
        raise ValueError(f"点1坐标超出范围: ({x1}, {y1})")
    if not (0 <= x2 < w and 0 <= y2 < h):
        raise ValueError(f"点2坐标超出范围: ({x2}, {y2})")
    if abs(x2 - x1) < 1e-12:
        raise ValueError("两点横坐标过近或相同，无法计算横滚角")

    return (x1, y1), (x2, y2)


# =========================================================
# 9. ROI 交互与输入
# =========================================================
def fit_image_to_screen(img: np.ndarray, max_width: int = 1400, max_height: int = 900):
    """
    将图像按比例缩放到较适合屏幕显示的大小
    返回:
        img_show: 缩放后图像
        scale: 原图 -> 显示图 的缩放比例
    """
    h, w = img.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    new_w = int(w * scale)
    new_h = int(h * scale)

    if scale < 1.0:
        img_show = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        img_show = img.copy()

    return img_show, scale


def select_roi_interactive_adaptive(
    img: np.ndarray,
    win_name: str = "Select ROI",
    max_width: int = 1400,
    max_height: int = 900
) -> Tuple[int, int, int, int]:
    """
    自适应窗口大小的 ROI 框选
    返回原图坐标系下的 x, y, w, h
    """
    print("请拖框选择 ROI，按 Enter/Space 确认，按 c 重选。")

    img_show, scale = fit_image_to_screen(img, max_width=max_width, max_height=max_height)

    roi_show = cv2.selectROI(win_name, img_show, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(win_name)

    xs, ys, ws, hs = roi_show
    if ws <= 0 or hs <= 0:
        raise ValueError("未成功选取ROI")

    x = int(round(xs / scale))
    y = int(round(ys / scale))
    w = int(round(ws / scale))
    h = int(round(hs / scale))

    H, W = img.shape[:2]
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))

    return x, y, w, h


def input_roi_from_console(img: np.ndarray) -> Tuple[int, int, int, int]:
    """
    手动输入 ROI 坐标
    """
    H, W = img.shape[:2]
    print(f"请输入 ROI 坐标，图像尺寸为 W={W}, H={H}")
    x = int(input("x = ").strip())
    y = int(input("y = ").strip())
    w = int(input("w = ").strip())
    h = int(input("h = ").strip())

    if w <= 0 or h <= 0:
        raise ValueError("ROI 的 w 和 h 必须 > 0")
    if x < 0 or y < 0 or x + w > W or y + h > H:
        raise ValueError("输入的 ROI 超出图像范围")

    return x, y, w, h


def get_roi(
    img: np.ndarray,
    mode: str = "interactive"
) -> Tuple[int, int, int, int]:
    """
    ROI 获取统一入口
    mode:
        "interactive" -> 交互选择
        "input"       -> 手动输入
    """
    mode = mode.strip().lower()
    if mode == "interactive":
        return select_roi_interactive_adaptive(img)
    elif mode == "input":
        return input_roi_from_console(img)
    else:
        raise ValueError("mode 必须是 'interactive' 或 'input'")


# =========================================================
# 10. 主流程
# =========================================================
def main():
    # -----------------------------------------------------
    # A. 输入路径
    # -----------------------------------------------------
    image_path = r"C:\Users\ShenYuLong\Desktop\frame_000_optimized.jpg"
    section_csv = r"C:\Users\ShenYuLong\Desktop\aaa.csv"
    output_dir = r"C:\Users\ShenYuLong\Desktop\frame_000_optimized(1)"
    os.makedirs(output_dir, exist_ok=True)

    # -----------------------------------------------------
    # B. 相机参数
    # -----------------------------------------------------
    cam = CameraParams(
        fx=285.203792,
        fy=285.213505,
        cx=164.696065,
        cy=116.753570,
        k1=0,
        k2=0,
        p1=0,
        p2=0,
        pixel_size_s=1.4e-6
    )

    # -----------------------------------------------------
    # C. 测站几何参数
    # -----------------------------------------------------
    site = SiteParams(
        Yc=0,
        Zc=4.8
    )

    # -----------------------------------------------------
    # D. 当前参考水位（已知）
    # -----------------------------------------------------
    reference_water_level = 4
    # -------------------------------

    # -----------------------------------------------------
    # E. 虚拟水尺参数
    # gauge_z_min / gauge_z_max 后面自动赋值
    # -----------------------------------------------------
    gauge_params = GaugeParams(
        ruler_x_center=850,
        ruler_half_width=25,
        gauge_z_min=0.0,
        gauge_z_max=0.0,
        gauge_step=2
    )

    # -----------------------------------------------------
    # E1. ROI 选择模式
    # "interactive" = 交互框选
    # "input"       = 手动输入坐标
    # -----------------------------------------------------
    roi_mode = "interactive"

    # -----------------------------------------------------
    # E2. 横滚角获取模式
    # "click_points" -> 鼠标点两点
    # "input_points" -> 手动输入两点完整坐标
    # -----------------------------------------------------
    roll_mode = "click_points"

    # -----------------------------------------------------
    # F. 断面有效段提取参数
    # -----------------------------------------------------
    monotonic_eps = 0.01

    # -----------------------------------------------------
    # G. 读取图像与断面地形数据
    # -----------------------------------------------------
    img = load_image_chinese(image_path, cv2.IMREAD_COLOR)
    section_df = load_section_csv(section_csv)
    valid_df = extract_monotonic_bank_suffix(section_df, eps=monotonic_eps)

    print("原始断面点数:", len(section_df))
    print("有效断面点数:", len(valid_df))
    print("有效断面高程范围: ", valid_df["Z"].min(), "~", valid_df["Z"].max())
    print(valid_df.head())
    print(valid_df.tail())

    gauge_params.gauge_z_min = float(valid_df["Z"].min())
    gauge_params.gauge_z_max = float(valid_df["Z"].max())

    print(f"自动设置虚拟水尺高程范围: {gauge_params.gauge_z_min} ~ {gauge_params.gauge_z_max}")

    valid_df.to_csv(
        os.path.join(output_dir, "valid_section_original_points.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    # -----------------------------------------------------
    # H. 畸变校正
    # -----------------------------------------------------
    undistorted, intrinsics = undistort_image(img, cam)
    save_image_chinese(os.path.join(output_dir, "01_undistorted.jpg"), undistorted)

    # 等效焦距（m）
    f = cam.pixel_size_s * (cam.fx + cam.fy) / 2.0

    # -----------------------------------------------------
    # I. 获取两点 -> 计算横滚角 -> 水平校正
    # -----------------------------------------------------
    if roll_mode == "click_points":
        pt0, pt1 = select_two_points(undistorted, "Click 2 points on waterline")
    elif roll_mode == "input_points":
        pt0, pt1 = input_two_points_from_console(undistorted)
    else:
        raise ValueError("roll_mode 必须是 'click_points' 或 'input_points'")

    phi = compute_roll_angle(pt0, pt1)
    phi_deg = math.degrees(phi)

    rotated, M = rotate_image_keep_all(undistorted, -phi_deg)

    p0_star = transform_point(M, pt0)
    p1_star = transform_point(M, pt1)
    v_line_star = (p0_star[1] + p1_star[1]) / 2.0

    rotated_vis = rotated.copy()
    cv2.circle(rotated_vis, (int(round(p0_star[0])), int(round(p0_star[1]))), 5, (0, 0, 255), -1)
    cv2.circle(rotated_vis, (int(round(p1_star[0])), int(round(p1_star[1]))), 5, (0, 255, 0), -1)
    cv2.line(
        rotated_vis,
        (int(round(p0_star[0])), int(round(p0_star[1]))),
        (int(round(p1_star[0])), int(round(p1_star[1]))),
        (255, 0, 0), 2
    )

    v_line_int = int(round(v_line_star))
    if 0 <= v_line_int < rotated_vis.shape[0]:
        cv2.line(
            rotated_vis,
            (0, v_line_int),
            (rotated_vis.shape[1] - 1, v_line_int),
            (0, 255, 255),
            2
        )
        cv2.putText(
            rotated_vis,
            f"v_line_star={v_line_star:.2f}",
            (10, max(25, v_line_int - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )


    save_image_chinese(os.path.join(output_dir, "02_rotated_horizontal.jpg"), rotated_vis)
    # 给 03 单独准备一个“无 v_line_star 文字”的底图
    rotated_vis_no_text = rotated.copy()
    cv2.circle(rotated_vis_no_text, (int(round(p0_star[0])), int(round(p0_star[1]))), 5, (0, 0, 255), -1)
    cv2.circle(rotated_vis_no_text, (int(round(p1_star[0])), int(round(p1_star[1]))), 5, (0, 255, 0), -1)
    cv2.line(
        rotated_vis_no_text,
        (int(round(p0_star[0])), int(round(p0_star[1]))),
        (int(round(p1_star[0])), int(round(p1_star[1]))),
        (255, 0, 0), 2
    )

    if 0 <= v_line_int < rotated_vis_no_text.shape[0]:
        cv2.line(
            rotated_vis_no_text,
            (0, v_line_int),
            (rotated_vis_no_text.shape[1] - 1, v_line_int),
            (0, 255, 255),
            2
        )


    # -----------------------------------------------------
    # J. 根据参考水位计算俯仰角
    # -----------------------------------------------------
    Hp = rotated.shape[0]
    omega = compute_pitch_angle(
        v_line_star=v_line_star,
        reference_water_level=reference_water_level,
        valid_df=valid_df,
        site=site,
        pixel_size_s=cam.pixel_size_s,
        image_height_hp=Hp,
        focal_length_f=f
    )
    omega_deg = math.degrees(omega)

    # -----------------------------------------------------
    # K. 生成稠密点（保存用，可选）
    # -----------------------------------------------------
    dense_section_df = build_dense_section_by_Z(
        valid_df=valid_df,
        z_min=gauge_params.gauge_z_min,
        z_max=gauge_params.gauge_z_max,
        z_step=gauge_params.gauge_step
    )

    dense_gauge_rows = []
    for _, row in dense_section_df.iterrows():
        Y_sj = float(row["Y"])
        Z_sj = float(row["Z"])
        v_star = project_level_to_vstar(
            Z_sj=Z_sj,
            Y_sj=Y_sj,
            site=site,
            omega=omega,
            pixel_size_s=cam.pixel_size_s,
            image_height_hp=Hp,
            focal_length_f=f
        )
        dense_gauge_rows.append([v_star, Y_sj, Z_sj])

    dense_gauge_df = pd.DataFrame(dense_gauge_rows, columns=["v_star", "Y", "Z"])
    dense_gauge_df = dense_gauge_df.sort_values("v_star").reset_index(drop=True)
    dense_gauge_df.to_csv(
        os.path.join(output_dir, "dense_virtual_gauge_points.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    # -----------------------------------------------------
    # L. 查找表：用原始有效断面采样点
    # -----------------------------------------------------
    lookup_df = build_lookup_table_from_original_section(
        valid_df=valid_df,
        site=site,
        omega=omega,
        pixel_size_s=cam.pixel_size_s,
        image_height_hp=Hp,
        focal_length_f=f
    )

    lookup_csv = os.path.join(output_dir, "lookup_table.csv")
    lookup_df.to_csv(lookup_csv, index=False, encoding="utf-8-sig")

    # -----------------------------------------------------
    # M. 选 ROI
    # -----------------------------------------------------
    roi = get_roi(rotated, mode=roi_mode)
    x, y, w, h = roi

    delta_v_star = y

    # -----------------------------------------------------
    # N. 在原图中，仅在 ROI 区域绘制虚拟水尺（每 0.05m）
    # -----------------------------------------------------
    gauge_img_full = draw_virtual_gauge_on_full_image_in_roi(
        img_rot=rotated_vis_no_text,
        lookup_df=lookup_df,
        roi=roi,
        z_min=gauge_params.gauge_z_min,
        z_max=gauge_params.gauge_z_max,
        step_main=0.5
    )
    save_image_chinese(os.path.join(output_dir, "03_virtual_gauge_full_in_roi.jpg"), gauge_img_full)

    # -----------------------------------------------------
    # O. 裁出 ROI 原图
    # -----------------------------------------------------
    roi_img_raw = rotated[y:y+h, x:x+w].copy()
    save_image_chinese(os.path.join(output_dir, "04_roi_raw.jpg"), roi_img_raw)

    roi_img_from_full = gauge_img_full[y:y+h, x:x+w].copy()
    save_image_chinese(os.path.join(output_dir, "05_roi_from_full.jpg"), roi_img_from_full)

    # -----------------------------------------------------
    # P. 在 ROI 内绘制直尺样式虚拟水尺
    # -----------------------------------------------------
    # 直接贴右边，更接近你参考脚本的画法
    roi_ruler_x_center = w - 3

    roi_gauge_img = draw_virtual_gauge_on_roi_ruler_style(
        roi_img=roi_img_raw,
        lookup_df=lookup_df,
        roi_offset_y=y,
        z_min=gauge_params.gauge_z_min,
        z_max=gauge_params.gauge_z_max,
        ruler_x_center=roi_ruler_x_center,
        ruler_half_width=12,  # 这里越小，整体刻度越短
        short_step=0.01,
        mid_step=0.05,
        long_step=0.1,
        axis_offset_right=3,

        # 这几个参数就是控制“短一点”的核心
        short_ratio=0.20,
        mid_ratio=0.40,
        long_ratio=0.60,

        min_short_len=4,
        min_mid_len=7,
        min_long_len=10
    )

    save_image_chinese(os.path.join(output_dir, "06_roi_virtual_ruler.jpg"), roi_gauge_img)

    # -----------------------------------------------------
    # Q. 保存关键参数
    # -----------------------------------------------------
    summary_items = [
        ["phi_rad", phi],
        ["phi_deg", phi_deg],
        ["v_line_star", v_line_star],
        ["omega_rad", omega],
        ["omega_deg", omega_deg],
        ["f_meter", f],
        ["image_height_rotated", Hp],
        ["roi_mode", roi_mode],
        ["roll_mode", roll_mode],
        ["pt0_x", pt0[0]],
        ["pt0_y", pt0[1]],
        ["pt1_x", pt1[0]],
        ["pt1_y", pt1[1]],
        ["roi_x", x],
        ["roi_y", y],
        ["roi_w", w],
        ["roi_h", h],
        ["delta_v_star", delta_v_star],
        ["gauge_z_min", gauge_params.gauge_z_min],
        ["gauge_z_max", gauge_params.gauge_z_max],
        ["gauge_step", gauge_params.gauge_step],
        ["ruler_x_center_global", gauge_params.ruler_x_center],
        ["ruler_half_width", gauge_params.ruler_half_width],
        ["ruler_x_center_roi", roi_ruler_x_center],

        ["fx", cam.fx],
        ["fy", cam.fy],
        ["cx", cam.cx],
        ["cy", cam.cy],
        ["s", cam.pixel_size_s],
        ["pixel_size_s", cam.pixel_size_s],
        ["k1", cam.k1],
        ["k2", cam.k2],
        ["p1", cam.p1],
        ["p2", cam.p2],

        ["intrinsics_00_fx", float(intrinsics[0, 0])],
        ["intrinsics_11_fy", float(intrinsics[1, 1])],
        ["intrinsics_02_cx", float(intrinsics[0, 2])],
        ["intrinsics_12_cy", float(intrinsics[1, 2])],

        ["Yc", site.Yc],
        ["Zc", site.Zc],
        ["reference_water_level", reference_water_level],
        ["monotonic_eps", monotonic_eps],
    ]
    summary = pd.DataFrame(summary_items, columns=["name", "value"])
    summary.to_csv(
        os.path.join(output_dir, "summary_params.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    print("处理完成。输出目录：", output_dir)
    print(f"roll_mode = {roll_mode}")
    print(f"点1 = ({pt0[0]:.3f}, {pt0[1]:.3f})")
    print(f"点2 = ({pt1[0]:.3f}, {pt1[1]:.3f})")
    print(f"横滚角 φ = {phi_deg:.6f} deg")
    print(f"俯仰角 ω = {omega_deg:.6f} deg")
    print(f"ROI mode = {roi_mode}")
    print(f"ROI = (x={x}, y={y}, w={w}, h={h})")
    print(f"delta_v_star = {delta_v_star}")
    print(f"自动生成的虚拟水尺高程范围 = {gauge_params.gauge_z_min} ~ {gauge_params.gauge_z_max}")
    print("原图虚拟水尺：每 0.05m 一个刻度，仅绘制在 ROI 区域内")
    print("ROI 虚拟水尺：0.01m 短刻度，0.05m 中刻度，0.10m 长刻度")
    print(f"原始有效断面点已保存：{os.path.join(output_dir, 'valid_section_original_points.csv')}")
    print(f"稠密虚拟水尺点已保存：{os.path.join(output_dir, 'dense_virtual_gauge_points.csv')}")
    print(f"查找表已保存：{lookup_csv}")


if __name__ == "__main__":
    main()
