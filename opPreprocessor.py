import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import openvino as ov
from openvino.tools import mo
import torch


class DetectionResultProcessor:
    """用于处理模型推理结果的类"""

    def __init__(self, conf_thresh=0.5):
        self.conf_thresh = conf_thresh

    def process(self, preds: np.ndarray) -> list:
        """
        处理预测结果
        :param preds: 模型输出的预测张量 (1, 25200, 85)
        :return: 过滤后的检测框列表 [x1, y1, x2, y2, conf, cls]
        """
        if preds.shape[-1] != 85:
            raise ValueError("预测结果维度应为(1,25200,85)")

        detections = []
        for det in preds[0]:  # 遍历25200个预测
            conf = det[4]  # 置信度
            if conf > self.conf_thresh:
                # 提取xywh+conf+class (根据YOLO输出格式调整)
                detections.append(det[:6])
        return detections


def preprocess_and_display(img_path, img_size=640, display=True):
    """
    完整的图像预处理流程（包含可选的可视化）

    参数:
        img_path: 图像路径
        img_size: 目标尺寸
        display: 是否显示处理过程

    返回:
        input_tensor: 预处理后的张量 (1,3,640,640)
        orig_img: 原始图像 (用于后处理可视化)
    """
    # 1. 读取图像
    orig_img = cv2.imread(img_path)
    if orig_img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    # 2. 颜色转换 BGR→RGB
    img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

    # 3. 缩放（保持长宽比）
    h, w = img_rgb.shape[:2]
    r = min(img_size / h, img_size / w)
    new_h, new_w = int(h * r), int(w * r)
    resized_img = cv2.resize(img_rgb, (new_w, new_h),
                             interpolation=cv2.INTER_LINEAR if r > 1 else cv2.INTER_AREA)

    # 4. 填充到正方形
    top = bottom = (img_size - new_h) // 2
    left = right = (img_size - new_w) // 2
    padded_img = cv2.copyMakeBorder(
        resized_img, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )

    # 5. 可选：显示处理过程
    if display:
        plt.figure(figsize=(12, 4))
        plt.subplot(131), plt.imshow(orig_img[..., ::-1]), plt.title(f"原始 {w}x{h}")
        plt.subplot(132), plt.imshow(resized_img), plt.title(f"缩放 {new_w}x{new_h}")
        plt.subplot(133), plt.imshow(padded_img), plt.title(f"填充 {img_size}x{img_size}")
        plt.show()

    # 6. 转换为模型输入格式
    input_tensor = padded_img.transpose(2, 0, 1)  # HWC→CHW
    input_tensor = np.expand_dims(input_tensor, 0).astype(np.float32) / 255.0  # 添加batch维度+归一化

    return input_tensor, orig_img


def main():
    # 1. 预处理
    input_tensor, orig_img, _, _ = preprocess_and_display("bus.jpg", img_size=640)

    # 2. 加载并转换模型
    if not os.path.exists("yolov8n.xml"):
        pt_model = torch.load("yolov8n.pt", map_location='cpu')
        pt_model.eval()
        ov_model = mo.convert_model(
            pt_model,
            input_shape=[1, 3, 640, 640],
            example_input=torch.randn(1, 3, 640, 640)
        )
        ov.serialize(ov_model, "yolov8n.xml", "yolov8n.bin")

    # 3. 推理
    core = ov.Core()
    compiled_model = core.compile_model("yolov8n.xml", "AUTO")  # 自动选择设备
    preds = compiled_model(input_tensor)[0]

    # 4. 将preds传递给处理类
    processor = DetectionResultProcessor(conf_thresh=0.25)
    filtered_dets = processor.process(preds)

    print(f"检测到 {len(filtered_dets)} 个目标:")
    for det in filtered_dets[:3]:  # 打印前3个检测结果
        print(f"- 位置: {det[:4]}, 置信度: {det[4]:.2f}, 类别: {det[5]}")


if __name__ == "__main__":
    main()