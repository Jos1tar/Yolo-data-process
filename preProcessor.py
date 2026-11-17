import cv2
import numpy as np
import matplotlib.pyplot as plt
import onnxruntime as rt
import os
from functools import lru_cache
import openvino as ov

try:  # torch 只在 xyxy2xywh 的类型判断里用到，可选
    import torch
except ImportError:  # pragma: no cover
    torch = None

import opPostprocessor


def preprocess_and_display(img_path, img_size=640, show=False):
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
        interp = cv2.INTER_LINEAR  # 与 ultralytics 一致；原来下采样用 INTER_AREA，conf 会差约 0.01
        resized_img = cv2.resize(img, (min(img_size, round(w0 * r)), min(img_size, round(h0 * r))), interpolation=interp)
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

    # 5. 显示处理过程（服务端调用时 show=False，否则 plt.show() 会阻塞请求）
    if show:
        _show_steps(orig_img, resized_img, padded_img, w0, h0, w1, h1)

    # 6. 转换为模型输入格式（不执行推理）
    input_tensor = padded_img.transpose((2, 0, 1))
    input_tensor = np.ascontiguousarray(input_tensor)
    input_tensor = np.expand_dims(input_tensor, axis=0).astype(np.float32) / 255.0
    return input_tensor, orig_img, resized_img, padded_img, (dw, dh, r, w0, h0)


def _show_steps(orig_img, resized_img, padded_img, w0, h0, w1, h1):
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


def run_inference(input_tensor, xml_path, device="CPU"):
    """
       使用 OpenVINO IR 模型进行推理（适配 YOLOv8 输出）

       参数:
           input_tensor: 预处理后的图像 (B,C,H,W)，float32
           xml_path: OpenVINO .xml 模型路径
           device: 推理设备，"CPU" 或 "GPU"
       返回:
           preds: numpy 数组，形状为 (1, 84, 8400)
       """
    core = ov.Core()
    model = core.read_model(model=xml_path, weights=xml_path.replace(".xml", ".bin"))
    compiled_model = core.compile_model(model, device)


    output_layer = compiled_model.output(0)
    result = compiled_model([input_tensor])[output_layer]  # shape: (1, 84, 8400)
    preds = np.array(result)

    print(f"推理完成，输出 shape: {preds.shape}")
    return preds


# （左上角坐标，右下角坐标）转 （检测框中心坐标，检测框宽高）
def xyxy2xywh(x):
    """Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right."""
    y = x.clone() if torch is not None and isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y

# （检测框中心坐标，检测框宽高）转（左上角坐标，右下角坐标）
def xywh2xyxy(x):
    """Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right."""
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


def std_output(pred):
    """
    将（1，84，8400）处理成（8400， 85）  85= box:4  conf:1 cls:80
    """
    pred = np.squeeze(pred)
    pred = np.transpose(pred, (1, 0))
    pred_class = pred[..., 4:]
    pred_conf = np.max(pred_class, axis=-1)
    pred = np.insert(pred, 4, pred_conf, axis=-1)
    return pred



@lru_cache(maxsize=4)
def _get_session(model_path):
    return rt.InferenceSession(model_path, providers=["CPUExecutionProvider"])


def ini(image_path: str = "runs/true.jpg", model_path: str = "yolov8n.onnx"):
    try:
        input_tensor, orig_img, resized_img, padded_img, meta = preprocess_and_display(image_path, img_size=640)

        # 载入 ONNX 模型并进行推理
        sess = _get_session(model_path)  # 原来每个请求都重新创建 InferenceSession
        input_name = sess.get_inputs()[0].name
        pred = sess.run(None, {input_name: input_tensor})[0]

        # 后处理（原代码调用了不存在的 cod_trf 且 post_process 参数签名不匹配）
        postprocessor = opPostprocessor.YOLOv8PostProcessor(conf_thres=0.5, iou_thres=0.4)
        mapped, drawn = postprocessor.post_pipeline(pred, [meta], [orig_img], [padded_img])
        result = mapped[0]

        out_dir = "runs/detect/myPredict"
        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, os.path.basename(image_path))
        cv2.imwrite(save_path, drawn[0])
        print(f"结果保存至 {save_path}")
        return {"status": "success", "save_path": save_path, "results": result}

    except Exception as e:
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    ini()