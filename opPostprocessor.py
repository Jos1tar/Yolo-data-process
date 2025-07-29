import cv2
import numpy as np
import torch
import torchvision

from PIL import Image, ImageDraw, ImageFont


class YOLOv8PostProcessor:
    def __init__(self, conf_thres=0.25, iou_thres=0.45):
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        # 简化的COCO类别列表（保持原样）
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
        self.coco_dict = {idx: name for idx, name in enumerate(self.coco_names)}
        self.model_size = 640

    def xywh2xyxy(self, x, y, w, h):
        """将中心点坐标格式转换为左上右下角点格式
        Args:
            x: 中心点x坐标
            y: 中心点y坐标
            w: 宽度
            h: 高度
        Returns:
            [x1, y1, x2, y2]: 左上右下角点坐标
        """
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2
        return np.array([x1, y1, x2, y2])  # 返回numpy数组便于后续计算

    def get_iou(self, box1, box2):
        """计算两个框的IoU
        Args:
            box1: [x1, y1, x2, y2] 格式的框
            box2: [x1, y1, x2, y2] 格式的框
        Returns:
            iou: 交并比
        """
        # 计算交集区域的坐标
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        # 计算交集面积
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)

        # 计算两个框的面积
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        # 计算IoU
        union_area = box1_area + box2_area - inter_area
        iou = inter_area / (union_area + 1e-6)  # 添加epsilon避免除0

        return iou

    def non_max_suppression(self, boxes, scores):
        """简化的非极大值抑制实现"""
        if len(boxes) == 0:
            return []

        # 按置信度排序
        indices = np.argsort(scores)[::-1]
        boxes = boxes[indices]
        scores = scores[indices]

        keep = []
        while boxes.shape[0] > 0:
            # 保留当前最高置信度的框
            keep.append(indices[0])

            if boxes.shape[0] == 1:
                break

            # 计算当前框与其余框的IoU
            ious = np.array([self.get_iou(boxes[0], boxes[i]) for i in range(1, boxes.shape[0])])

            # 找到IoU小于阈值的框
            idx = np.where(ious < self.iou_thres)[0] + 1

            # 保留这些框
            boxes = boxes[idx]
            indices = indices[idx]

        return keep

    def post_process(self, preds, meta_list):
        preds = np.transpose(preds, (0, 2, 1))  # (B, N, C)
        results_all = []

        for b in range(preds.shape[0]):
            pred = preds[b]  # shape (8400, 84)
            # 前面不变：转置后得到 pred (8400,84)
            boxes = pred[:, :4]
            class_confs = pred[:, 4:]  # 80 维 class-aware scores

            # 直接取类别和置信度
            cls_ids = np.argmax(class_confs, axis=1)
            confs = np.max(class_confs, axis=1)

            # 置信度过滤
            keep_mask = confs > self.conf_thres
            boxes = boxes[keep_mask]
            confs = confs[keep_mask]
            cls_ids = cls_ids[keep_mask]

            # 将每个 box 从 xywh 转为 xyxy
            boxes_xyxy = np.array([
                self.xywh2xyxy(x, y, w, h) for x, y, w, h in boxes
            ])


            # 后面逐类 NMS 不变…

            if boxes.shape[0] == 0:
                results_all.append(([], *meta_list[b]))
                continue


            # 2. 拼接成 (M,6)：[xyxy, conf, cls_id]
            detected = np.concatenate([
                boxes,
                confs[:, None],
                cls_ids[:, None]
            ], axis=1)   # shape = (M,6)

            output_box = []
            for cls_id in np.unique(detected[:, 5].astype(int)):
                cls_mask = detected[:, 5] == cls_id
                cls_boxes_xywh = detected[cls_mask, :4]

                # 新增：转换为xyxy格式
                cls_boxes_xyxy = []
                for box in cls_boxes_xywh:
                    x, y, w, h = box
                    cls_boxes_xyxy.append(self.xywh2xyxy(x, y, w, h))
                cls_boxes_xyxy = np.array(cls_boxes_xyxy)

                cls_scores = detected[cls_mask, 4]

                # 传入正确的xyxy格式
                keep_idxs = self.non_max_suppression(cls_boxes_xyxy, cls_scores)
                kept = detected[cls_mask][keep_idxs]
                output_box.extend(kept)

            # 4. 打印并收集结果
            print(f"[图像 {b}] 最终保留框数量: {len(output_box)}")
            for i, box in enumerate(output_box):
                x, y, w, h, conf, cid = box
                print(f"  {i+1}: 类别={self.coco_dict[int(cid)]}({int(cid)}), "
                      f"置信度={conf:.4f}, 框=[{x:.1f},{y:.1f},{w:.1f},{h:.1f}]")

            results_all.append((output_box, *meta_list[b]))

        return results_all

    def cod_trf_batch(self, results_all, orig_imgs, padded_imgs):
        """
        将每张图片的检测框从 letterbox 映射回原图坐标
        :param results_all: [(output_box, dw, dh, r, w0, h0), ...]
        :param orig_imgs: 原图列表
        :param padded_imgs: 经过 letterbox 的图像列表
        :return: List[np.array] -> 每张图的转换后框 (x1,y1,x2,y2,conf,cls)
        """
        transformed_results = []

        for i, (result, dw, dh, r, w0, h0) in enumerate(results_all):
            if not result:
                transformed_results.append([])
                continue

            res = []
            h_pad, w_pad = padded_imgs[i].shape[:2]

            scale = min(w_pad / w0, h_pad / h0)
            new_w = int(w0 * scale)
            new_h = int(h0 * scale)
            pad_x = (w_pad - new_w) / 2
            pad_y = (h_pad - new_h) / 2

            for box in result:
                x, y, w, h, conf, cls_id = box

                # 反 letterbox
                x_orig = (x - pad_x) / scale
                y_orig = (y - pad_y) / scale
                w_orig = w / scale
                h_orig = h / scale

                x1, y1, x2, y2 = self.xywh2xyxy(x_orig, y_orig, w_orig, h_orig)

                # clip 到原图尺寸
                img_h, img_w = orig_imgs[i].shape[:2]
                x1 = max(0, min(img_w - 1, x1))
                y1 = max(0, min(img_h - 1, y1))
                x2 = max(0, min(img_w - 1, x2))
                y2 = max(0, min(img_h - 1, y2))

                res.append([x1, y1, x2, y2, conf, cls_id])

            transformed_results.append(np.array(res))

        return transformed_results

    def draw(self, res, image, cls):
        """
        将预测框绘制在 image 上
        Args:
            res: 预测框数据 (x1, y1, x2, y2, conf, class_id)
            image: 原图
            cls: 类别列表（索引->名称映射，例如 coco.names）
        Returns:
            绘制后的图像
        """
        for r in res:
            x1, y1, x2, y2, conf, class_id = r
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            class_id = int(class_id)

            # 绘制边框
            color = (0, 255, 0)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            # 准备标签文本
            class_name = cls[class_id] if class_id < len(cls) else f"id_{class_id}"
            text = f"{class_name}: {conf:.2f}"

            # 文字大小和位置
            font_scale = max(min((y2 - y1) / 300, 1.0), 0.4)
            font = cv2.FONT_HERSHEY_SIMPLEX
            thickness = 1
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            text_origin = (x1, y1 - 10 if y1 - 10 > 10 else y1 + 10)

            # 背景填充
            cv2.rectangle(image,
                          (text_origin[0], text_origin[1] - text_size[1] - 4),
                          (text_origin[0] + text_size[0] + 4, text_origin[1] + 4),
                          color, -1)
            # 添加文字
            cv2.putText(image, text, (text_origin[0] + 2, text_origin[1]),
                        font, font_scale, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

        return image

    def post_pipeline(self, preds, meta_list, orig_imgs, padded_imgs):
        """
        完整的后处理管道
        Args:
            preds: 模型预测结果，shape=(B,84,8400)
            meta_list: 元数据列表，包含 (dw, dh, r, w0, h0)
            orig_imgs: 原始图片列表
            padded_imgs: letterbox 后的图片列表
        Returns:
            mapped_results: List[np.array]，每张图的框 (x1,y1,x2,y2,conf,cls)
            drawn_imgs: List[np.array]，在原图上画完框的图
        """
        # 1. NMS、filter 得到每张图的原始框（格式：[[x,y,w,h,conf,cls],…]）
        results_all = self.post_process(preds, meta_list)

        # 2. 把框映射回原图
        mapped_results = self.cod_trf_batch(results_all, orig_imgs, padded_imgs)


        # 3. 打印每张图的检测框
        for idx, boxes in enumerate(mapped_results):
            print(f"[Image {idx}] 检测到 {len(boxes)} 个框：")
            for b in boxes:
                x1, y1, x2, y2, conf, cls_id = b
                print(f"    cls={self.coco_dict[int(cls_id)]}({int(cls_id)}), "
                      f"conf={conf:.3f}, box=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})")

        # 4. 在原图上画框
        drawn_imgs = []
        for img, boxes in zip(orig_imgs, mapped_results):
            drawn = self.draw(boxes, img.copy(), self.coco_names)
            drawn_imgs.append(drawn)

        return mapped_results, drawn_imgs
