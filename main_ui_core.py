import numpy as np
import torch
from PIL import Image
from sam3.model.sam3_image_processor import Sam3Processor  # 导入SAM3的图像处理器
from sam3 import build_sam3_image_model  # 导入构建SAM3模型的函数
import os


class SAM3Segmentation:
    def __init__(self):
        # 初始化SAM3模型
        self.init_sam3_model()  # 调用初始化SAM3模型的函数

        # 初始化状态
        self.state = None  # 保存模型状态和中间结果
        self.current_image = None  # 保存上传的图片
        self.current_image_array = None  # 图片数组形式

        # 框选模式，初始化为正向框选
        self.box_mode = "positive"  # 设置框选模式为正向框选

    def init_sam3_model(self):
        """初始化SAM3模型"""
        try:
            print("正在初始化SAM3模型...")  # 输出初始化信息

            # 1. BPE文件路径 - 修改为你的路径
            bpe_path = r"E:\VS code\SAM3\assets\bpe_simple_vocab_16e6.txt.gz"  # 设置BPE文件路径
            # 2. 权重文件路径
            checkpoint_path = r"E:\VS code\SAM3\sam3.pt"  # 设置权重文件路径

            # 检查BPE文件是否存在
            if not os.path.exists(bpe_path):
                raise FileNotFoundError(f"BPE文件不存在: {bpe_path}")  # 如果BPE文件不存在，抛出异常

            # 检查权重文件是否存在
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"权重文件不存在: {checkpoint_path}")  # 如果权重文件不存在，抛出异常
            
            # 设置设备（优先GPU）
            DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

            # 3. 构建模型并加载权重
            print("构建模型中...")  # 输出构建模型的信息
            model = build_sam3_image_model(
            checkpoint_path=str(checkpoint_path),
            bpe_path=str(bpe_path),
            device=DEVICE,
            enable_inst_interactivity=True  #启用交互式分割
            )  # 使用BPE文件路径构建SAM3模型
            print("基础模型构建完成")  # 输出模型构建成功的信息

            print("加载模型权重...")  # 输出加载权重的信息
            # 加载权重
            checkpoint = torch.load(checkpoint_path, map_location="cpu")  # 加载权重文件
            model.load_state_dict(checkpoint, strict=False)  # 将权重加载到模型中，严格模式为False

            # 设置为评估模式
            model.eval()  # 设置模型为评估模式

            # 4. 创建图像处理器
            self.processor = Sam3Processor(model, device=DEVICE)  # 使用构建的模型初始化图像处理器
            self.model = model
            print("SAM3模型初始化成功!")  # 输出模型初始化成功的信息

        except Exception as e:
            print(f"初始化失败: {e}")  # 如果发生异常，输出错误信息
            raise  # 抛出异常，终止程序

    def set_image(self, image):
        """设置当前图片并进行处理"""
        self.current_image = image  # 保存上传的图片
        self.current_image_array = np.array(image)  # 将图片转换为数组

        # 使用处理器将图像传递给SAM3模型进行处理
        self.state = self.processor.set_image(image)  # 将图像传递给SAM3模型，获取处理结果

    def on_text_prompt(self, prompt):
        """根据文本提示进行分割"""
        if not self.state:
            raise ValueError("请先加载图片")  # 如果没有加载图片，抛出异常

        if not prompt:
            raise ValueError("请输入有效的提示")  # 如果文本提示为空，抛出异常

        print(f"使用提示分割: {prompt}...")  # 输出正在使用的文本提示
        self.state = self.processor.set_text_prompt(prompt, self.state)  # 使用文本提示更新状态

    def on_box_mode_change(self, mode):# 选择正负框
        """框选模式变化"""
        if mode not in ["positive", "negative"]:
            raise ValueError("无效的框选模式")  # 如果框选模式无效，抛出异常
        self.box_mode = mode  # 更新框选模式

    def on_confidence_change(self, value):
        """处理置信度变化"""
        confidence = value / 100.0  # 将置信度值转换为0到1之间的小数
        print(f"当前置信度阈值: {confidence:.2f}")  # 输出当前置信度阈值

        if self.state:
            self.state = self.processor.set_confidence_threshold(confidence, self.state)  # 更新状态中的置信度阈值

    def on_box_fenge(self):
        """处理框选分割"""
        if self.state is None or "prompted_boxes" not in self.state:
            return  # 如果没有加载图片或者没有框选区域，直接返回

        prompted_boxes = self.state["prompted_boxes"]  # 获取当前所有框选区域
        self.processor.reset_all_prompts(self.state)  # 重置所有框选提示

        for box_info in prompted_boxes:
            box = box_info.get("box")  # 获取框选区域
            label = box_info.get("label")  # 获取框选区域的标签（正向或负向）
            self.state = self.processor.add_geometric_prompt(box, label, self.state)  # 将框选提示添加到模型中

        print("框选分割完成")  # 输出框选分割完成的信息

    def update_display(self):
        """更新显示（这里只是示范，不做UI部分显示）"""
        if self.current_image_array is None:
            return  # 如果没有图片，直接返回

        print(f"图片尺寸: {self.current_image_array.shape}")  # 输出图片的尺寸

        if self.state is not None and "masks" in self.state:
            masks = self.state.get("masks", [])  # 获取分割结果的掩码
            boxes = self.state.get("boxes", [])  # 获取分割结果的边界框
            scores = self.state.get("scores", [])  # 获取分割结果的置信度

            if len(masks) > 0:
                print(f"找到 {len(masks)} 个对象")  # 输出找到的分割结果数量
                for mask, box, score in zip(masks, boxes, scores):
                    print(f"对象位置: {box}, 置信度: {score:.2f}")  # 输出每个分割结果的边界框和置信度
            else:
                print("未找到高于置信度阈值的对象")  # 如果没有找到对象，输出提示信息
        else:
            print("没有获取到掩码数据")  # 如果没有分割结果，输出提示信息

    def set_box(self, x_min, y_min, x_max, y_max, label="positive"):
        """手动设置框选区域"""
        if self.state is None:
            raise ValueError("请先加载图片")  # 如果没有加载图片，抛出异常

        if "prompted_boxes" not in self.state:
            self.state["prompted_boxes"] = []  # 如果没有框选区域，初始化为空列表

        # 计算相对坐标
        img_h = self.state["original_height"]  # 获取图片的高度
        img_w = self.state["original_width"]  # 获取图片的宽度

        center_x = (x_min + x_max) / 2.0 / img_w  # 计算框选区域中心点的相对横坐标
        center_y = (y_min + y_max) / 2.0 / img_h  # 计算框选区域中心点的相对纵坐标
        width = (x_max - x_min) / img_w  # 计算框选区域的相对宽度
        height = (y_max - y_min) / img_h  # 计算框选区域的相对高度

        box = [center_x, center_y, width, height]  # 将框选区域保存为归一化的四元组
        label = True if label == "positive" else False  # 判断框选模式是正向框选还是负向框选

        self.state["prompted_boxes"].append({
            "box": box,  # 添加框选区域到状态中
            "label": label  # 添加框选标签到状态中
        })

        print(f"框选区域添加成功，框信息：{box}")  # 输出框选区域信息

    def on_clear_prompts(self):
            """清除所有提示"""
            if self.current_image is not None:
                self.set_loading(True, "清除提示并重置...")
                # self.state = self.processor.reset_all_prompts(self.state)
                if self.state is not None and "prompted_boxes" in self.state:
                    del self.state["prompted_boxes"]
                if self.state is not None and "masks" in self.state:
                    del self.state["masks"]
                self.text_input.clear()
                self.set_loading(False)
                self.status_label.setText("✅ 已清除所有提示")
                self.update_display()


# 示例运行
def main():
    # 创建SAM3分割对象
    sam3_seg = SAM3Segmentation()

    # 模拟上传图片
    image = Image.open(r"E:\VS code\SAM3\picture_input\80.jpg")  # 假设有一张图片
    sam3_seg.set_image(image)

    # 选择框选模式
    sam3_seg.on_box_mode_change("positive")

    # 设置框选区域
    sam3_seg.set_box(50, 50, 150, 150)

    # 设置置信度
    sam3_seg.on_confidence_change(80)

    # 执行框选分割
    sam3_seg.on_box_fenge()

    # 更新显示（此处为模拟输出，不涉及UI）
    sam3_seg.update_display()


if __name__ == "__main__":
    main()
