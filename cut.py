import os
import pandas as pd
from PIL import Image
from typing import Dict, Tuple
import re

# ===================== 【用户需修改的配置区域】开始 =====================
# 图片所在的文件夹路径（请替换为你的实际路径）
IMAGE_FOLDER_PATH = r"E:\VS code\data\nomral - X"
# 您上传的Excel文件路径（已适配，无需修改列名）
EXCEL_FILE_PATH = r"E:\VS code\SAM3\initial_high.xlsx"
# 支持的图片格式（可根据需要添加/删除）
SUPPORTED_IMAGE_FORMATS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")
# ===================== 【用户需修改的配置区域】结束 =====================

def read_image_coords_from_excel(excel_path: str) -> Dict[int, int]:
    """
    从您的Excel文件中读取图片序号与对应Y坐标的映射关系
    :param excel_path: Excel文件路径
    :return: 字典，key为图片序号（整数），value为Y坐标整数
    """
    try:
        # 读取Excel文件的Sheet1
        df = pd.read_excel(excel_path, sheet_name="Sheet1")
        # 过滤掉图片序号或Y坐标为空的行
        df = df.dropna(subset=["图片序号名", "测量水位线"])
        # 构建映射字典
        coord_map = {}
        for _, row in df.iterrows():
            try:
                image_serial = int(row["图片序号名"])
                y_coord = int(row["测量水位线"])
                coord_map[image_serial] = y_coord
            except (ValueError, TypeError):
                print(f"⚠️  跳过无效数据：图片序号【{row['图片序号名']}】/坐标【{row['测量水位线']}】不是有效整数")
        print(f"✅ 成功从Excel读取到 {len(coord_map)} 条有效坐标记录")
        return coord_map
    except FileNotFoundError:
        print(f"❌ 错误：Excel文件未找到，请检查路径【{excel_path}】")
        exit(1)
    except Exception as e:
        print(f"❌ 读取Excel文件失败：{str(e)}")
        exit(1)

def extract_serial_from_filename(filename: str) -> int | None:
    """
    从图片文件名中提取数字序号，匹配Excel中的图片序号名
    :param filename: 图片文件名（如785.jpg、IMG_786.png）
    :return: 提取到的整数序号，提取失败返回None
    """
    # 提取文件名中的所有数字
    serial_match = re.search(r'(\d+)', filename)
    if serial_match:
        try:
            return int(serial_match.group(1))
        except ValueError:
            return None
    return None

def get_image_size(image_path: str) -> Tuple[int, int]:
    """
    获取图片的宽度和高度
    :param image_path: 图片文件路径
    :return: 元组 (宽度, 高度)
    """
    try:
        with Image.open(image_path) as img:
            return img.size
    except Exception as e:
        print(f"❌ 无法读取图片【{image_path}】：{str(e)}")
        return 0, 0

def cut_image(image_path: str, y_coord: int) -> bool:
    """
    按Y坐标剪切图片，保留Y坐标以下的部分，覆盖原图
    :param image_path: 图片文件路径
    :param y_coord: 剪切起始Y坐标
    :return: 剪切成功返回True，失败返回False
    """
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            # 验证Y坐标有效性
            if y_coord < 0:
                print(f"⚠️  跳过【{os.path.basename(image_path)}】：Y坐标不能为负数")
                return False
            if y_coord >= height:
                print(f"⚠️  跳过【{os.path.basename(image_path)}】：Y坐标({y_coord})大于等于图片高度({height})")
                return False
            
            # 剪切图片：左、上、右、下 → 保留Y坐标到图片底部的区域
            cropped_img = img.crop((0, y_coord, width, height))
            # 覆盖保存原图
            cropped_img.save(image_path)
            return True
    except Exception as e:
        print(f"❌ 剪切图片【{image_path}】失败：{str(e)}")
        return False

def main():
    # 1. 读取Excel中的坐标映射
    coord_map = read_image_coords_from_excel(EXCEL_FILE_PATH)
    
    # 2. 检查图片文件夹是否存在
    if not os.path.isdir(IMAGE_FOLDER_PATH):
        print(f"❌ 错误：图片文件夹未找到，请检查路径【{IMAGE_FOLDER_PATH}】")
        exit(1)
    
    # 3. 遍历文件夹中的所有图片，批量处理
    success_count = 0
    skip_count = 0
    fail_count = 0
    
    for filename in os.listdir(IMAGE_FOLDER_PATH):
        # 过滤非支持的图片格式
        if not filename.lower().endswith(SUPPORTED_IMAGE_FORMATS):
            continue
        
        image_path = os.path.join(IMAGE_FOLDER_PATH, filename)
        # 跳过文件夹（只处理文件）
        if not os.path.isfile(image_path):
            continue
        
        # 提取图片序号，匹配Excel
        image_serial = extract_serial_from_filename(filename)
        if image_serial is None:
            print(f"⚠️  跳过【{filename}】：无法从文件名中提取有效数字序号")
            skip_count += 1
            continue
        
        # 获取对应Y坐标
        y_coord = coord_map.get(image_serial)
        if y_coord is None:
            print(f"⚠️  跳过【{filename}】：Excel中未找到序号【{image_serial}】对应的坐标")
            skip_count += 1
            continue
        
        # 执行剪切
        if cut_image(image_path, y_coord):
            print(f"✅ 成功处理【{filename}】，序号：{image_serial}，Y坐标：{y_coord}")
            success_count += 1
        else:
            fail_count += 1
    
    # 4. 输出处理结果统计
    print("\n" + "="*50)
    print("📊 批量处理完成，结果统计：")
    print(f"✅ 成功处理：{success_count} 张")
    print(f"⚠️  跳过处理：{skip_count} 张")
    print(f"❌ 处理失败：{fail_count} 张")
    print("="*50)

if __name__ == "__main__":
    main()