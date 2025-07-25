from typing import List

import cv2
import numpy as np
import matplotlib.pyplot as plt
import onnxruntime as rt
import os
import math
import openvino as ov
#from openvino.tools import mo
import torch
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

import opPostprocessor


def preprocess_batch(img_paths: List[str], img_size=640, visualize=False):
    """
    多张图像的预处理和（可选）显示流程（不包含模型推理）

    参数:
        img_paths: 图像文件路径列表
        img_size: 输入图像尺寸（正方形）
        visualize: 是否显示每张图的预处理过程

    返回:
        batch_tensor: 预处理后的batch张量 (B, C, H, W)
        orig_imgs: 原始图像列表
        resized_imgs: 缩放图像列表
        padded_imgs: 填充图像列表
        meta_list: 每张图的(dw, dh, r, w0, h0)元信息列表
    """
    input_tensors = []
    orig_imgs = []
    resized_imgs = []
    padded_imgs = []
    meta_list = []

    for idx, img_path in enumerate(img_paths):
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"文件不存在: {img_path}")

        orig_img = cv2.imread(img_path)
        if orig_img is None:
            raise ValueError(f"无法读取图像，请检查格式: {img_path}")
        img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        h0, w0 = img.shape[:2]

        r = img_size / max(h0, w0)
        if r != 1:
            interp = cv2.INTER_LINEAR if r > 1 else cv2.INTER_AREA
            resized_img = cv2.resize(img, (math.ceil(w0 * r), math.ceil(h0 * r)), interpolation=interp)
        else:
            resized_img = img.copy()

        h1, w1 = resized_img.shape[:2]
        dw = (img_size - w1) / 2
        dh = (img_size - h1) / 2
        top = int(round(dh - 0.1))
        bottom = int(round(dh + 0.1))
        left = int(round(dw - 0.1))
        right = int(round(dw + 0.1))

        color = (114, 114, 114)
        padded_img = cv2.copyMakeBorder(
            resized_img, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=color
        )

        # 转换为模型输入格式
        input_tensor = padded_img.transpose((2, 0, 1))  # HWC -> CHW
        input_tensor = np.ascontiguousarray(input_tensor)
        input_tensor = np.expand_dims(input_tensor, axis=0).astype(np.float32) / 255.0

        input_tensors.append(input_tensor)
        orig_imgs.append(orig_img)
        resized_imgs.append(resized_img)
        padded_imgs.append(padded_img)
        meta_list.append((dw, dh, r, w0, h0))

        # 可视化
        if visualize:
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            plt.imshow(orig_img[:, :, ::-1])
            plt.title(f"[{idx}] original {w0}x{h0}")
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.imshow(resized_img)
            plt.title(f"[{idx}] scaled {w1}x{h1}")
            plt.axis('off')

            plt.subplot(1, 3, 3)
            plt.imshow(padded_img)
            plt.title(f"[{idx}] padded {padded_img.shape[1]}x{padded_img.shape[0]}")
            plt.axis('off')

            plt.tight_layout()
            plt.show()

    # 合并为batch张量
    batch_tensor = np.vstack(input_tensors)
    print(f"批量张量形状: {batch_tensor.shape} (B,C,H,W)")

    return batch_tensor, orig_imgs, resized_imgs, padded_imgs, meta_list


def run_trt_inference(input_tensor, engine_path="yolov8n.engine"):
    """
    使用 TensorRT engine 文件进行推理
    输入：
        input_tensor: numpy 数组 (1, 3, 640, 640)，float32
    输出：
        pred: numpy 数组，shape=(1, 84, 8400)
    """
    assert input_tensor.dtype == np.float32, "输入必须是 float32"
    assert input_tensor.shape[0] == 1, "仅支持 batch_size=1"

    TRT_LOGGER = trt.Logger()
    with open(engine_path, "rb") as f:
        engine = trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    # ✅ 设置动态 shape（YOLOv8 是动态模型）
    input_name = engine.get_binding_name(0)  # "images"
    context.set_binding_shape(0, input_tensor.shape)  # 如 (1, 3, 640, 640)

    # 获取输入输出索引
    input_idx = engine.get_binding_index(input_name)
    output_idx = engine.get_binding_index("output0")  # 通常为 "output0"

    # 分配 GPU 显存
    input_tensor = np.ascontiguousarray(input_tensor)  # 保证内存连续
    input_nbytes = input_tensor.nbytes

    output_shape = context.get_binding_shape(output_idx)  # eg: (1, 84, 8400)
    output_nbytes = np.prod(output_shape) * np.dtype(np.float32).itemsize

    d_input = cuda.mem_alloc(input_nbytes)
    d_output = cuda.mem_alloc(output_nbytes)

    # 拷贝输入到 GPU
    cuda.memcpy_htod(d_input, input_tensor)

    # 推理
    bindings = [None] * engine.num_bindings
    bindings[input_idx] = int(d_input)
    bindings[output_idx] = int(d_output)

    context.execute_v2(bindings)

    # 拷贝输出回 CPU
    output = np.empty(output_shape, dtype=np.float32)
    cuda.memcpy_dtoh(output, d_output)

    print(f"✅ 推理完成，输出 shape: {output.shape}")
    return output

# （左上角坐标，右下角坐标）转 （检测框中心坐标，检测框宽高）
def xyxy2xywh(x):
    """Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right."""
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
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



def ini(image_path: str = "runs/true.jpg"):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图像：{image_path}")

    try:
        post_processor = opPostprocessor.YOLOv8PostProcessor(conf_thres=0.4, iou_thres=0.6)
        image_paths = ["runs/bus.jpg", "runs/OIP.jpg", "runs/true.jpg"]
        #, "runs/OIP.jpg", "runs/true.jpg"
        batch_tensor, orig_imgs, resized_imgs, padded_imgs, meta_list = preprocess_batch(image_paths, img_size=640,visualize=True)
        """
        # 载入 ONNX 模型并进行推理
        sess = rt.InferenceSession(
            'yolov8n.onnx',
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']  # CUDA for GPU
        )
        input_name = sess.get_inputs()[0].name
        label_name = sess.get_outputs()[0].name
        pred = sess.run([label_name], {input_name: input_tensor})[0]
        """
        # 使用 TensorRT 推理 ONNX 模型
        sess = rt.InferenceSession("yolov8n.onnx", providers=["TensorrtExecutionProvider", "CUDAExecutionProvider"])
        print("实际使用的 provider:", sess.get_providers())
        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name
        # 假设 input_tensor 是 shape=(1,3,640,640)，float32
        pred = sess.run([output_name], {input_name: batch_tensor})[0]
        print("推理完成，输出 shape:", pred.shape)


        mapped_results,drawn_imgs = post_processor.post_pipeline(pred, meta_list,orig_imgs, padded_imgs)

        # 保存结果
        os.makedirs("runs/detect/myPredict", exist_ok=True)
        for i, img in enumerate(drawn_imgs):
            save_path = f"runs/detect/myPredict/{os.path.basename(image_paths[i])}"
            cv2.imwrite(save_path, img)
            print(f"[{i}] 结果保存至 {save_path}")

    except Exception as e:
        return {"status": "error", "message": str(e)}




if __name__ == "__main__":
    ini()

