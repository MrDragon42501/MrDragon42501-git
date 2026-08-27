import os
import sys
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sam3.model.sam3_image_processor import Sam3Processor
from sam3 import build_sam3_image_model

# ================= 路径配置 =================
IMG_DIR = r"C:\Users\a1595\Desktop\123"
EXCEL_PATH = r"C:\Users\a1595\Desktop\掩码高度.xlsx"
BOX_DIR = r"C:\Users\a1595\Desktop\12345"
OUT_DIR = r"C:\Users\a1595\Desktop\1234567"          # 融合掩码输出目录
WATER_MASK_DIR = r"C:\Users\a1595\Desktop\first"    # 水掩码目录
GLARE_MASK_DIR = r"C:\Users\a1595\Desktop\yaobantu" # 耀斑掩码输出目录

BPE_PATH = r"D:\sam3model\assets\bpe_simple_vocab_16e6.txt.gz"
CHECKPOINT_PATH = r"D:\sam3model\sam3.pt"

CONFIDENCE_THRESHOLD = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(GLARE_MASK_DIR, exist_ok=True)   # 创建耀斑掩码输出文件夹

# ================= 初始化模型 =================
print("正在初始化 SAM3 模型...")
if not os.path.exists(BPE_PATH):
    raise FileNotFoundError(f"BPE 文件不存在: {BPE_PATH}")
if not os.path.exists(CHECKPOINT_PATH):
    raise FileNotFoundError(f"权重文件不存在: {CHECKPOINT_PATH}")

model = build_sam3_image_model(bpe_path=BPE_PATH)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
if isinstance(checkpoint, dict):
    if any(k.startswith('detector.') for k in checkpoint.keys()):
        new_ckpt = {}
        for k, v in checkpoint.items():
            new_k = k.replace('detector.', '', 1) if k.startswith('detector.') else k
            new_ckpt[new_k] = v
        model.load_state_dict(new_ckpt, strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
else:
    model.load_state_dict(checkpoint, strict=False)

model.eval()
model.to(DEVICE)
processor = Sam3Processor(model, device=DEVICE)
print("SAM3 模型初始化完成\n")

# ================= 读取 Excel =================
if not os.path.exists(EXCEL_PATH):
    raise FileNotFoundError(f"Excel 文件不存在: {EXCEL_PATH}")

df_excel = pd.read_excel(EXCEL_PATH)
df_excel["图片序号名"] = df_excel["图片序号名"].astype(str).str.strip()
df_excel = df_excel.set_index("图片序号名")

# ================= 辅助函数 =================
def load_boxes_from_csv(base_name, y_offset):
    csv_path = os.path.join(BOX_DIR, f"{base_name}_blue_boxes.csv")
    if not os.path.exists(csv_path):
        print(f"   ⚠️ 框文件不存在: {csv_path}，跳过框提示")
        return []
    df_boxes = pd.read_csv(csv_path)
    boxes = []
    for _, row in df_boxes.iterrows():
        x = int(row.iloc[2])
        y = int(row.iloc[3]) + y_offset
        w = int(row.iloc[4])
        h = int(row.iloc[5])
        boxes.append((x, y, w, h))
    return boxes

def normalize_box(x, y, w, h, img_w, img_h):
    center_x = (x + w / 2.0) / img_w
    center_y = (y + h / 2.0) / img_h
    norm_w = w / img_w
    norm_h = h / img_h
    return [center_x, center_y, norm_w, norm_h]

def load_water_mask(base_name, target_shape):
    """加载水掩码，返回二值掩码 (H, W) 的 bool 数组"""
    mask_path = os.path.join(WATER_MASK_DIR, f"{base_name}_mask_combined.png")
    if not os.path.exists(mask_path):
        print(f"   ❌ 水掩码不存在: {mask_path}")
        return None
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"   ❌ 无法读取水掩码: {mask_path}")
        return None
    if mask.shape != target_shape:
        mask = cv2.resize(mask, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
    return (mask > 127)

def merge_masks(water_mask, glare_mask):
    """合并水掩码和耀斑掩码：水区域或耀斑区域"""
    return np.logical_or(water_mask, glare_mask)

def save_binary_mask(mask_bool, save_path):
    """将 bool 掩码保存为 0/255 PNG"""
    mask_uint8 = (mask_bool * 255).astype(np.uint8)
    cv2.imwrite(save_path, mask_uint8)
    print(f"   💾 已保存: {save_path}")

# ================= 获取所有图片文件 =================
image_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
image_files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(image_extensions)]
image_files.sort()

if not image_files:
    print("输入文件夹中没有图片，退出。")
    sys.exit(0)

print(f"找到 {len(image_files)} 张图片，开始处理...\n")
error_log = []

for idx, fname in enumerate(image_files, 1):
    print(f"[{idx}/{len(image_files)}] 处理图片: {fname}")
    base_name = os.path.splitext(fname)[0]

    # 1. 加载图片
    img_path = os.path.join(IMG_DIR, fname)
    try:
        pil_image = Image.open(img_path).convert("RGB")
        image_np = np.array(pil_image)
        h_img, w_img = image_np.shape[:2]
    except Exception as e:
        error_log.append(f"{fname}: 加载失败 - {e}")
        continue

    # 2. 加载水掩码
    water_mask = load_water_mask(base_name, (h_img, w_img))
    if water_mask is None:
        error_log.append(f"{fname}: 水掩码缺失，跳过")
        continue

    # 3. 获取偏移量
    if base_name not in df_excel.index:
        y_offset = 0
    else:
        y_offset = int(df_excel.loc[base_name, "掩码最高位置"])

    # 4. 读取耀斑标注框
    boxes_abs = load_boxes_from_csv(base_name, y_offset)
    print(f"   找到 {len(boxes_abs)} 个框")

    # 5. 设置图像
    try:
        state = processor.set_image(pil_image)
        if state is None:
            raise RuntimeError("set_image 返回 None")
    except Exception as e:
        error_log.append(f"{fname}: set_image 失败 - {e}")
        continue

    # 6. 重置提示
    processor.reset_all_prompts(state)

    # 7. 添加文本提示
    try:
        state = processor.set_text_prompt("glint", state)
    except Exception as e:
        error_log.append(f"{fname}: 文本提示失败 - {e}")
        continue

    # 8. 添加框提示
    for (x, y, w, h) in boxes_abs:
        box_norm = normalize_box(x, y, w, h, w_img, h_img)
        try:
            state = processor.add_geometric_prompt(box_norm, True, state)
        except Exception as e:
            print(f"   添加框失败: {e}")

    # 9. 设置置信度阈值
    try:
        state = processor.set_confidence_threshold(CONFIDENCE_THRESHOLD, state)
    except Exception as e:
        print(f"   设置阈值失败: {e}")

    # 10. 获取分割结果（耀斑掩码）
    masks = state.get("masks", [])
    boxes_pred = state.get("boxes", [])
    scores = state.get("scores", [])

    # 判断是否有有效掩码
    has_masks = False
    if masks is not None:
        if isinstance(masks, torch.Tensor):
            has_masks = masks.numel() > 0
        elif isinstance(masks, (list, np.ndarray)):
            has_masks = len(masks) > 0

    # 11. 构建耀斑掩码（合并所有分割结果）
    if has_masks:
        if isinstance(masks, torch.Tensor):
            if masks.dim() == 4 and masks.shape[1] == 1:
                masks_np = masks[:, 0, :, :].cpu().numpy()  # [N, H, W]
            else:
                masks_np = masks.cpu().numpy()
        else:
            masks_np = np.array(masks)

        if masks_np.ndim == 3:
            glare_mask_bool = np.any(masks_np > 0.5, axis=0)
        else:
            glare_mask_bool = (masks_np > 0.5)
        print(f"   ✅ 获得 {masks_np.shape[0] if masks_np.ndim==3 else 1} 个耀斑掩码，合并后前景像素: {np.sum(glare_mask_bool)}")
    else:
        print("   ⚠️ 未获得耀斑分割结果，使用空掩码")
        glare_mask_bool = np.zeros((h_img, w_img), dtype=bool)

    # 12. 保存耀斑掩码（单独输出到 yaobantu 文件夹）
    glare_save_path = os.path.join(GLARE_MASK_DIR, f"{base_name}_glare_mask.png")
    save_binary_mask(glare_mask_bool, glare_save_path)

    # 13. 融合掩码：水掩码 ∪ 耀斑掩码
    fused_mask = merge_masks(water_mask, glare_mask_bool)
    print(f"   融合后水区域像素: {np.sum(fused_mask)}")

    # 14. 保存融合掩码（到 OUT_DIR）
    fused_save_path = os.path.join(OUT_DIR, f"{base_name}_mask_fused.png")
    save_binary_mask(fused_mask, fused_save_path)

print("\n🎉 所有图片处理完成！")
if error_log:
    print("\n以下图片处理失败：")
    for err in error_log:
        print(f"  {err}")