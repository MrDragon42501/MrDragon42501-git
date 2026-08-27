import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
import os
from pathlib import Path

def douglas_peucker(points, epsilon):
    """
    Douglas-Peucker算法：简化折线点集
    参数:
        points: 原始点集，格式为[(x1,y1), (x2,y2), ...]
        epsilon: 容差（距离阈值），值越大简化程度越高
    返回:
        简化后的点集
    """
    if len(points) < 2:
        return points.copy()
    
    def perpendicular_distance(point, line_start, line_end):
        """计算点到线段的垂直距离"""
        if line_start == line_end:
            return sqrt((point[0]-line_start[0])**2 + (point[1]-line_start[1])**2)
        
        numerator = abs(
            (line_end[0] - line_start[0]) * (line_start[1] - point[1]) -
            (line_start[0] - point[0]) * (line_end[1] - line_start[1])
        )
        denominator = sqrt((line_end[0] - line_start[0])**2 + (line_end[1] - line_start[1])**2)
        return numerator / denominator if denominator != 0 else 0
    
    max_dist = 0
    max_index = 0
    start_point = points[0]
    end_point = points[-1]
    
    for i in range(1, len(points)-1):
        dist = perpendicular_distance(points[i], start_point, end_point)
        if dist > max_dist:
            max_dist = dist
            max_index = i
    
    result = []
    if max_dist > epsilon:
        left_points = douglas_peucker(points[:max_index+1], epsilon)
        right_points = douglas_peucker(points[max_index:], epsilon)
        result = left_points[:-1] + right_points
    else:
        result = [start_point, end_point]
    
    return result

def process_single_excel(excel_file_path, output_img_path, x_col, y_col, epsilon=5.0):
    """
    处理单个Excel文件，生成对比图并保存
    参数:
        excel_file_path: 单个Excel文件的完整路径
        output_img_path: 输出图片的完整路径
        x_col: 横坐标列名
        y_col: 纵坐标列名
        epsilon: Douglas-Peucker算法的容差
    """
    try:
        # 读取Excel数据
        df = pd.read_excel(excel_file_path)
        
        # 检查列是否存在
        if x_col not in df.columns or y_col not in df.columns:
            raise ValueError(f"未找到列：{x_col} 或 {y_col}")
        
        # 数据清洗：去空值、转数值类型
        df = df[[x_col, y_col]].dropna()
        df[x_col] = pd.to_numeric(df[x_col], errors='coerce')
        df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
        df = df.dropna()
        
        if len(df) < 2:
            raise ValueError("有效数据点不足2个")
        
        # 提取原始点集
        original_points = list(zip(df[x_col].values, df[y_col].values))
        x_original = [p[0] for p in original_points]
        y_original = [p[1] for p in original_points]
        
        # 简化折线
        simplified_points = douglas_peucker(original_points, epsilon)
        x_simplified = [p[0] for p in simplified_points]
        y_simplified = [p[1] for p in simplified_points]
        
        # 绘图
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文
        plt.rcParams['axes.unicode_minus'] = False    # 支持负号
        
        fig, ax = plt.subplots(figsize=(10, 6))
        # 原始折线（蓝色）
        ax.plot(x_original, y_original, 'b-', marker='o', markersize=4, label='原始折线', alpha=0.7)
        # 简化折线（红色粗线）
        ax.plot(x_simplified, y_simplified, 'r-', marker='s', markersize=6, label='简化折线', linewidth=2, alpha=0.8)
        
        # 图表样式设置
        ax.set_title(f'{os.path.basename(excel_file_path)} - 折线简化对比', fontsize=14)
        ax.set_xlabel(x_col, fontsize=12)
        ax.set_ylabel(y_col, fontsize=12)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # 保存图片（高分辨率）
        plt.tight_layout()
        plt.savefig(output_img_path, dpi=150, bbox_inches='tight')
        plt.close()  # 关闭画布，释放内存
        
        # 输出日志
        print(f"✅ 处理完成：{os.path.basename(excel_file_path)}")
        print(f"   原始点：{len(original_points)} | 简化点：{len(simplified_points)} | 保存路径：{output_img_path}")
        
    except Exception as e:
        print(f"❌ 处理失败：{os.path.basename(excel_file_path)} - {str(e)}")

def batch_process_excels(input_folder, output_folder, x_col, y_col, epsilon=5.0):
    """
    批量处理文件夹下的所有Excel文件
    参数:
        input_folder: 输入Excel文件夹路径
        output_folder: 输出图片文件夹路径
        x_col: 横坐标列名
        y_col: 纵坐标列名
        epsilon: 简化容差
    """
    # 检查输入文件夹是否存在
    if not os.path.exists(input_folder):
        print(f"错误：输入文件夹不存在 - {input_folder}")
        return
    
    # 创建输出文件夹（若不存在）
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # 遍历输入文件夹中的Excel文件
    excel_extensions = ['.xlsx', '.xls']
    excel_files = [f for f in os.listdir(input_folder) if os.path.splitext(f)[1].lower() in excel_extensions]
    
    if not excel_files:
        print(f"警告：输入文件夹 {input_folder} 中未找到Excel文件")
        return
    
    # 批量处理每个Excel文件
    print(f"开始处理，共找到 {len(excel_files)} 个Excel文件...")
    for excel_file in excel_files:
        # 构建完整路径
        excel_file_path = os.path.join(input_folder, excel_file)
        # 生成输出图片路径（替换后缀为.png，保持文件名一致）
        img_filename = os.path.splitext(excel_file)[0] + '.png'
        output_img_path = os.path.join(output_folder, img_filename)
        
        # 处理单个文件
        process_single_excel(excel_file_path, output_img_path, x_col, y_col, epsilon)
    
    print("\n📊 批量处理完成！所有图片已保存至：", output_folder)

# 示例调用
if __name__ == "__main__":
    # 请根据实际情况修改以下参数
    INPUT_FOLDER = r"E:\VS code\SAM3\filter_output"    # 输入Excel文件夹路径
    OUTPUT_FOLDER = r"E:\VS code\SAM3\change_points_seeker_output"  # 输出图片文件夹路径
    X_COLUMN = "时间(Y坐标)"               # Excel中横坐标列名
    Y_COLUMN = "滤波后信号强度(灰度值)"               # Excel中纵坐标列名
    EPSILON = 15.0                     # 简化容差（值越大简化越明显）
    
    # 执行批量处理
    batch_process_excels(INPUT_FOLDER, OUTPUT_FOLDER, X_COLUMN, Y_COLUMN, EPSILON)