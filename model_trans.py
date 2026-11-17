"""PyTorch -> ONNX -> OpenVINO IR 转换脚本（实习期原型）。

修正记录：
- 原函数名叫 convert_yolov8_to_onnx 却加载 yolov5n.pt，之后又去转换 yolov5nu.onnx；改为参数化。
- 原 OpenVINO 转换写入了 ImageNet mean/std 和 reverse_input_channels，
  但 preProcessor 已经做了 BGR->RGB 和 /255，YOLO 也不使用 ImageNet 归一化，
  等于重复/错误归一化，输出会整体偏移。现在 IR 不嵌入任何预处理。
- openvino.tools.mo 已弃用，改用 openvino.convert_model / save_model。
"""
import os

import openvino as ov


def convert_pt_to_onnx(pt_path: str = "yolov8n.pt", imgsz: int = 640) -> str:
    from ultralytics import YOLO  # 仅导出阶段需要

    model = YOLO(pt_path)
    onnx_path = model.export(format="onnx", dynamic=True, simplify=True, opset=12, imgsz=(imgsz, imgsz))
    print(f"ONNX 转换完成: {onnx_path}")
    return str(onnx_path)


def convert_onnx_to_xml(onnx_path: str, output_dir: str = "./openvino_models", fp16: bool = True):
    os.makedirs(output_dir, exist_ok=True)
    model_name = os.path.splitext(os.path.basename(onnx_path))[0]
    xml_path = os.path.join(output_dir, f"{model_name}.xml")

    print(f"开始转换 {onnx_path} 到 OpenVINO IR 格式...")
    try:
        ov_model = ov.convert_model(onnx_path)
        ov_model.reshape([1, 3, 640, 640])  # 固定输入尺寸
        ov.save_model(ov_model, xml_path, compress_to_fp16=fp16)
        print(f"转换成功！\nXML: {xml_path}\nBIN: {xml_path.replace('.xml', '.bin')}")
        return xml_path
    except Exception as e:
        print(f"转换失败: {e}")
        return None


if __name__ == "__main__":
    onnx_path = convert_pt_to_onnx("yolov8n.pt")
    convert_onnx_to_xml(onnx_path)
