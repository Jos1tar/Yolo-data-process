import torch
print(torch.__version__)  # 查看 PyTorch 版本（需 ≥1.8）
print(torch.cuda.is_available())  # 检查 CUDA 是否可用

from ultralytics import YOLO
model = YOLO("yolov8n.pt")  # 自动下载预训练模型
  # 测试图片
model("bus.jpg",show=True,save=True)