import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import openvino as ov
#from openvino.tools import mo
import torch
import opPostprocessor
import postProcessor

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

    return input_tensor, orig_img, resized_img, padded_img, (dw, dh, r, w0, h0)


def run_inference(input_tensor, xml_path, device="CPU"):
    """
    使用 OpenVINO IR 模型进行推理

    参数:
        input_tensor: 预处理后的图像 (B,C,H,W)，float32
        xml_path: OpenVINO .xml 模型路径
        device: 推理设备，"CPU" 或 "GPU"
    返回:
        推理结果：numpy 数组
    """
    core = ov.Core()
    model = core.read_model(model=xml_path, weights=xml_path.replace(".xml", ".bin"))
    compiled_model = core.compile_model(model, device)

    # 输入节点名
    input_layer = compiled_model.input(0)
    result = compiled_model([input_tensor])[compiled_model.output(0)]

    print(f" 推理完成，输出 shape: {result.shape}")
    return result


def ini ():
    image_path = "OIP.jpg"
    xml_path = "openvino_models/yolov8n.xml"

    try:
        # 图像预处理
        input_tensor, orig_img, resized_img, padded_img, (dw, dh, r, w0, h0) = preprocess_and_display(
            image_path,
            img_size=640
        )
        # 模型推理
        preds = run_inference(input_tensor, xml_path=xml_path, device="CPU")
        # preds 形状：(1, 84, 8400)
        print("模型输出前几行（前6个框，每个框前6维）：")
        print(preds[0][:6, :6])  # [x, y, w, h, obj, cls0]
        print("模型输出范围:", preds.min(), preds.max())




        # 后处理
        results =postProcessor.post_process(preds, dw, dh, r, w0, h0)
        print("最终检测结果:")
        print(results)


        """
        postprocessor =opPostprocessor.YOLOv8PostProcessor(conf_thres=0.01, iou_thres=0.45)
  
        #results =postProcessor.post_process(preds, dw, dh, r, w0, h0)

        results = postprocessor.post_process(preds, dw, dh, r, w0, h0)

        # 可视化结果
        orig_img=cv2.imread(image_path)
        vis_image = postprocessor.visualize(orig_img.copy(), results)

        # 显示或保存结果
        vis_image.show()
        # vis_image.save("result.jpg")

        # 打印检测结果
        print(f"检测到 {len(results)} 个物体:")
        for i, det in enumerate(results):
            x, y, w, h, conf, cls_id = det
            cls_name = postprocessor.coco_names[int(cls_id) - 1]
            print(f"{i + 1}: {cls_name} ({conf:.2f}) 位置: [{x:.1f}, {y:.1f}, {w:.1f}, {h:.1f}]")
        print("最终检测结果:")
        print(results)
"""

    except Exception as e:
        print(f"发生错误: {str(e)}")



if __name__ == "__main__":
    ini()