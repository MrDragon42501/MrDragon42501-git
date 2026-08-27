import os
import sys
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
import matplotlib.pyplot as plt
from sam3.model.sam3_image_processor import Sam3Processor
from sam3 import build_sam3_image_model

# ================= 路径配置 =================
IMG_DIR = r"E:\VS code\SAM3\ROI_input"
EXCEL_PATH = r"E:\VS code\SAM3\initial_high.xlsx"
BOX_DIR = r"E:\VS code\SAM3\output-notice"
OUT_DIR = r"E:\VS code\SAM3\output-light"

BPE_PATH = r"E:\VS code\SAM3\assets\bpe_simple_vocab_16e6.txt.gz"
CHECKPOINT_PATH = r"E:\VS code\SAM3\sam3.pt"

CONFIDENCE_THRESHOLD = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUT_DIR, exist_ok=True)

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

    # 2. 获取偏移量
    if base_name not in df_excel.index:
        y_offset = 0
    else:
        y_offset = int(df_excel.loc[base_name, "掩码最高位置"])

    # 3. 读取框
    boxes_abs = load_boxes_from_csv(base_name, y_offset)
    print(f"   找到 {len(boxes_abs)} 个框")

    # 4. 设置图像
    try:
        state = processor.set_image(pil_image)
        if state is None:
            raise RuntimeError("set_image 返回 None")
    except Exception as e:
        error_log.append(f"{fname}: set_image 失败 - {e}")
        continue

    # 5. 重置提示（注意：不接收返回值）
    processor.reset_all_prompts(state)

    # 6. 添加文本提示
    try:
        state = processor.set_text_prompt("glint", state)
    except Exception as e:
        error_log.append(f"{fname}: 文本提示失败 - {e}")
        continue

    # 7. 添加所有框提示
    for (x, y, w, h) in boxes_abs:
        box_norm = normalize_box(x, y, w, h, w_img, h_img)
        try:
            state = processor.add_geometric_prompt(box_norm, True, state)
        except Exception as e:
            print(f"   添加框失败: {e}")

    # 8. 设置置信度阈值
    try:
        state = processor.set_confidence_threshold(CONFIDENCE_THRESHOLD, state)
    except Exception as e:
        print(f"   设置阈值失败: {e}")

    # 9. 获取分割结果
    masks = state.get("masks", [])
    boxes_pred = state.get("boxes", [])
    scores = state.get("scores", [])

    # 判断是否有有效掩码（处理可能是张量的情况）
    has_masks = False
    if masks is not None:
        if isinstance(masks, torch.Tensor):
            has_masks = masks.numel() > 0
        elif isinstance(masks, (list, np.ndarray)):
            has_masks = len(masks) > 0

    out_img = image_np.copy()
    if has_masks:
        # 统一处理为列表形式以便迭代
        if isinstance(masks, torch.Tensor):
            # 假设 masks 形状为 [N, 1, H, W] 或 [N, H, W]
            masks_list = [masks[i] for i in range(masks.shape[0])]
            boxes_list = [boxes_pred[i] for i in range(boxes_pred.shape[0])] if isinstance(boxes_pred, torch.Tensor) else boxes_pred
            scores_list = [scores[i] for i in range(scores.shape[0])] if isinstance(scores, torch.Tensor) else scores
        else:
            masks_list = masks
            boxes_list = boxes_pred
            scores_list = scores

        print(f"   ✅ 获得 {len(masks_list)} 个分割对象")
        for i, (mask, box, score) in enumerate(zip(masks_list, boxes_list, scores_list)):
            # 掩码处理
            if torch.is_tensor(mask):
                mask_np = mask[0].cpu().numpy() if mask.dim() == 4 else mask.cpu().numpy()
            else:
                mask_np = mask[0] if len(mask.shape) == 4 else mask
            # 确保 mask_np 是 2D
            if mask_np.ndim == 3:
                mask_np = mask_np.squeeze(0)
            color = plt.cm.tab10(i % 10)[:3]  # RGB
            # 叠加半透明掩码
            for c in range(3):
                out_img[:, :, c] = np.where(mask_np > 0.5,
                                            out_img[:, :, c] * 0.5 + color[c] * 255 * 0.5,
                                            out_img[:, :, c])
            # 框处理
            if torch.is_tensor(box):
                box = box.cpu().numpy()
            x0, y0, x1, y1 = box.astype(int)
            cv2.rectangle(out_img, (x0, y0), (x1, y1), (int(color[0]*255), int(color[1]*255), int(color[2]*255)), 2)
            cv2.putText(out_img, f"{score:.2f}", (x0, y0-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    else:
        print("   ⚠️ 未获得分割结果")

    # 10. 保存结果
    out_path = os.path.join(OUT_DIR, fname)
    out_img_bgr = cv2.cvtColor(out_img.astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_path, out_img_bgr)
    print(f"   💾 已保存: {out_path}")

print("\n🎉 所有图片处理完成！")
if error_log:
    print("\n以下图片处理失败：")
    for err in error_log:
        print(f"  {err}")