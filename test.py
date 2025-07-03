from ultralytics import YOLO

a1= YOLO('yolov8n.pt')

a1("OIP.jpg", show =True, save = True)