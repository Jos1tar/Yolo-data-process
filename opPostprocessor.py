import cv2
import numpy as np
import torch
import torchvision

from PIL import Image, ImageDraw, ImageFont


class YOLOv8PostProcessor:
    def __init__(self, conf_thres=0.25, iou_thres=0.45):
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        self.coco_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
            'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
            'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
            'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
            'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
            'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
            'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]

        # 图像尺寸（模型训练尺寸）
        self.model_size = 640

    def xywh2xyxy(self, x):
        y = np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2  # x1
        y[..., 1] = x[..., 1] - x[..., 3] / 2  # y1
        y[..., 2] = x[..., 0] + x[..., 2] / 2  # x2
        y[..., 3] = x[..., 1] + x[..., 3] / 2  # y2
        return y

    def post_process(self, preds, dw, dh, r, w0, h0):
        """完全重构的后处理函数"""
        # 转换形状 (B, C, N) -> (B, N, C)
        if isinstance(preds, np.ndarray):
            preds = np.transpose(preds, (0, 2, 1))
        elif isinstance(preds, torch.Tensor):
            preds = preds.permute(0, 2, 1).cpu().numpy()

        batch_size = preds.shape[0]
        output = [np.zeros((0, 6))] * batch_size

        for b in range(batch_size):
            pred = preds[b]  # [num_preds, 84]

            # 关键修复：模型输出的是相对坐标，需要转换为绝对坐标
            # 分离边界框、物体置信度和类别分数
            bbox = pred[:, :4]  # [x, y, w, h] 相对坐标
            objectness = pred[:, 4]  # 物体置信度
            class_scores = pred[:, 5:]  # 类别分数

            # 应用sigmoid激活函数
            objectness = 1 / (1 + np.exp(-objectness))
            class_scores = 1 / (1 + np.exp(-class_scores))

            # 计算每个框的最大类别分数
            class_ids = np.argmax(class_scores, axis=1)
            class_max_scores = class_scores[np.arange(len(class_scores)), class_ids]

            # 综合置信度 = 物体置信度 * 最大类别分数
            conf_scores = objectness * class_max_scores

            # 置信度筛选
            conf_mask = conf_scores > self.conf_thres
            bbox = bbox[conf_mask]
            conf_scores = conf_scores[conf_mask]
            class_ids = class_ids[conf_mask]

            if bbox.shape[0] == 0:
                continue

            # 关键修复：将相对坐标转换为绝对坐标（在640x640空间）
            # 转换为绝对坐标（相对于640x640图像）
            bbox_abs = np.copy(bbox)
            bbox_abs[:, 0] = bbox[:, 0] * self.model_size  # x center
            bbox_abs[:, 1] = bbox[:, 1] * self.model_size  # y center
            bbox_abs[:, 2] = bbox[:, 2] * self.model_size  # width
            bbox_abs[:, 3] = bbox[:, 3] * self.model_size  # height

            # 转换坐标格式 (xywh -> xyxy)
            boxes = self.xywh2xyxy(bbox_abs)

            # 构建检测结果 [x1, y1, x2, y2, conf, class_id]
            detections = np.column_stack([boxes, conf_scores, class_ids.astype(float)])

            # 应用NMS
            if detections.shape[0] > 0:
                boxes_tensor = torch.from_numpy(detections[:, :4])
                scores_tensor = torch.from_numpy(detections[:, 4])
                keep_indices = torchvision.ops.nms(boxes_tensor, scores_tensor, self.iou_thres)
                detections = detections[keep_indices.numpy()]

                # 坐标反变换到原始图像空间
                detections[:, 0] = (detections[:, 0] - dw) / r  # x1
                detections[:, 1] = (detections[:, 1] - dh) / r  # y1
                detections[:, 2] = (detections[:, 2] - dw) / r  # x2
                detections[:, 3] = (detections[:, 3] - dh) / r  # y2

                # 裁剪到图像边界
                detections[:, [0, 2]] = detections[:, [0, 2]].clip(0, w0)
                detections[:, [1, 3]] = detections[:, [1, 3]].clip(0, h0)

                # 过滤无效框
                valid_mask = (detections[:, 2] > detections[:, 0]) & (detections[:, 3] > detections[:, 1])
                detections = detections[valid_mask]

                output[b] = detections

        if batch_size == 1:
            output = output[0]
            return output if output.shape[0] > 0 else np.zeros((0, 6))

        return output

    def visualize(self, image, detections):
        """可视化检测结果"""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype("arial.ttf", 15)
        except:
            font = ImageFont.load_default()

        for det in detections:
            x1, y1, x2, y2, conf, cls_id = det
            cls_id = int(cls_id)

            # 跳过无效框
            if x2 <= x1 or y2 <= y1 or conf < 0.01:
                continue

            # 绘制边界框
            color = (255, 0, 0)  # 红色
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

            # 显示标签
            if 0 <= cls_id < len(self.coco_names):
                label = f"{self.coco_names[cls_id]}: {conf:.2f}"
            else:
                label = f"unknown({cls_id}): {conf:.2f}"

            # 获取文本尺寸
            if hasattr(draw, 'textbbox'):
                bbox = draw.textbbox((x1, y1), label, font=font)
                text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
            else:
                text_width, text_height = font.getsize(label)

            # 确保标签在图像范围内
            text_y = max(0, y1 - text_height - 5)

            # 绘制标签背景
            draw.rectangle([x1, text_y, x1 + text_width, text_y + text_height], fill=color)
            draw.text((x1, text_y), label, fill="white", font=font)

        return image