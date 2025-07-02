import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import openvino as ov
from openvino.tools import mo
import torch


def preprocess_and_display(img_path, img_size=640):
    """
    图像预处理和显示流程（不包含模型推理）

    参数:
        img_path: 图像文件路径
        img_size: 目标尺寸（默认640）
    """
    # 1. 检查并读取图像
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"文件不存在: {img_path}")

    orig_img = cv2.imread(img_path)  # 原始BGR图像
    if orig_img is None:
        raise ValueError(f"无法读取图像，请检查格式: {img_path}")

    # 转换颜色空间为RGB
    img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

    # 2. 记录原始尺寸
    h0, w0 = img.shape[:2]  # 原始高度和宽度
    print(f"原始尺寸: {w0}x{h0}")

    # 3. 图像缩放（保持长宽比）
    r = img_size / max(h0, w0)  # 计算缩放比例
    if r != 1:  # 需要缩放
        interp = cv2.INTER_LINEAR if r > 1 else cv2.INTER_AREA  # 上采样用线性，下采样用区域
        resized_img = cv2.resize(img, (math.ceil(w0 * r), math.ceil(h0 * r)), interpolation=interp)
    else:
        resized_img = img.copy()

    # 4. 计算并添加填充
    h1, w1 = resized_img.shape[:2]  # 缩放后的尺寸
    dw = (img_size - w1) / 2  # 宽度填充量的一半
    dh = (img_size - h1) / 2  # 高度填充量的一半

    # 确保填充量为整数
    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))

    color = (114, 114, 114)  # 填充颜色（灰色）
    padded_img = cv2.copyMakeBorder(
        resized_img, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=color
    )

    # 5. 显示处理过程
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(orig_img[:, :, ::-1])  # 显示原始图像（BGR转RGB）
    plt.title(f"原始图像: {w0}x{h0}")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(resized_img)
    plt.title(f"缩放后: {w1}x{h1}")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(padded_img)
    plt.title(f"填充后: {padded_img.shape[1]}x{padded_img.shape[0]}")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # 6. 转换为模型输入格式（不执行推理）
    # HWC to CHW (Height, Width, Channel -> Channel, Height, Width)
    input_tensor = padded_img.transpose((2, 0, 1))
    # 转换为连续内存
    input_tensor = np.ascontiguousarray(input_tensor)
    # 添加batch维度并归一化
    input_tensor = np.expand_dims(input_tensor, axis=0).astype(np.float32) / 255.0

    print(f"预处理后的张量形状: {input_tensor.shape} (B,C,H,W)")
    print(f"像素值范围: {input_tensor.min():.4f} - {input_tensor.max():.4f}")

    return input_tensor, orig_img, resized_img, padded_img


# 使用示例
if __name__ == "__main__":
    # 替换为你的实际图像路径
    image_path = "bus.jpg"  # 确保文件存在

    try:
        # 运行预处理
        input_tensor, orig_img, resized_img, padded_img = preprocess_and_display(
            image_path,
            img_size=640
        )

        # 可以在这里添加其他处理代码
        # 例如保存预处理后的图像
        #cv2.imwrite("resized.jpg", cv2.cvtColor(resized_img, cv2.COLOR_RGB2BGR))
        #cv2.imwrite("padded.jpg", cv2.cvtColor(padded_img, cv2.COLOR_RGB2BGR))

    except Exception as e:
        print(f"发生错误: {str(e)}")

    # 1. 加载 PyTorch 模型
    ir_filename = torch.load("yolov8n.xml")  # 替换为你的模型路径


    def run_inference(input_tensor, device="GPU"):

        # 4. 加载 OpenVINO 模型
        core = ov.Core()
        # 读取模型
        model = core.read_model(model=ir_filename, weights=ir_filename.replace(".xml", ".bin"))

        compiled_model = core.compile_model(model, device)

        # 5. 推理
        preds = compiled_model(input_tensor)[0]
        print(preds.shape)  # (1, 25200, 85)
        return preds  # shape: (1, 25200, 85)


