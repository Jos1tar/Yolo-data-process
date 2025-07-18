from ultralytics import YOLO
from openvino.tools import mo
from openvino.runtime import serialize
import os
import torch


def convert_yolov8_to_onnx():
    # 1. 加载预训练模型
    model = YOLO('yolov5n.pt')  # 替换为你的.pt文件路径

    # 2. 导出为ONNX格式
    model.export(
        format='onnx',  # 输出格式
        dynamic=True,  # 支持动态batch
        simplify=True,  # 简化模型
        opset=12,  # ONNX算子集版本
        imgsz=(640, 640)  # 输入尺寸
    )
    print("ONNX转换完成！")




def convert_onnx_to_xml(onnx_path: str, output_dir: str = "./openvino_models"):
    """
    将ONNX模型转换为OpenVINO的XML/BIN格式
    Args:
        onnx_path: 输入的ONNX模型路径
        output_dir: 输出目录（默认当前目录下的openvino_models文件夹）
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 模型名称（不带后缀）
    model_name = os.path.splitext(os.path.basename(onnx_path))[0]

    # 设置输出路径
    xml_path = os.path.join(output_dir, f"{model_name}.xml")

    print(f"开始转换 {onnx_path} 到 OpenVINO IR格式...")

    try:
        # 使用OpenVINO模型优化器转换ONNX
        ov_model = mo.convert_model(
            input_model=onnx_path,
            input_shape=[1, 3, 640, 640],  # YOLOv8的默认输入尺寸
            compress_to_fp16=True,  # 压缩为FP16格式
            reverse_input_channels=True,  # BGR->RGB（如果模型需要RGB输入）
            mean_values=[123.675, 116.28, 103.53],  # ImageNet均值
            scale_values=[58.395, 57.12, 57.375]  # ImageNet标准差a
        )

        # 序列化为XML和BIN文件
        serialize(ov_model, xml_path)

        print(f"转换成功！模型已保存为：\nXML: {xml_path}\nBIN: {xml_path.replace('.xml', '.bin')}")
        return xml_path

    except Exception as e:
        print(f"转换失败: {str(e)}")
        return None


if __name__ == "__main__":
    convert_yolov8_to_onnx()

    onnx_path = "yolov5nu.onnx"
    # 转换为OpenVINO格式
    convert_onnx_to_xml(onnx_path)