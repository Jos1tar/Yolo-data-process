import numpy as np
import torch
import torchvision


def post_process(preds, dw, dh, r, w0, h0):
    """
    适配YOLOv8模型输出的后处理

    参数:
    preds: 模型输出，可以是 torch.Tensor 或 numpy.ndarray
           形状应为 (batch, channels, num_preds) 如 (1, 84, 8400)
    dw, dh: 宽高padding
    r: 缩放比例
    w0, h0: 原始图像宽高

    返回:
    output: [n, 6] (xywh, conf, class)
    """

    conf_thres = 0.001
    iou_thres = 0.65
    max_nms = 30000
    max_det = 300

    # COCO类别映射表
    coco80_to_coco91 = [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
        22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
        43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
        62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
        85, 86, 87, 88, 89, 90
    ]

    def xywh2xyxy(x):
        y = np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2
        y[..., 1] = x[..., 1] - x[..., 3] / 2
        y[..., 2] = x[..., 0] + x[..., 2] / 2
        y[..., 3] = x[..., 1] + x[..., 3] / 2
        return y

    def xyxy2xywh(x):
        y = np.copy(x)
        y[..., 0] = (x[..., 0] + x[..., 2]) / 2
        y[..., 1] = (x[..., 1] + x[..., 3]) / 2
        y[..., 2] = x[..., 2] - x[..., 0]
        y[..., 3] = x[..., 3] - x[..., 1]
        return y

    # 确保输入是numpy数组
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()

    # 转换形状 (B, C, N) -> (B, N, C)
    preds = np.transpose(preds, (0, 2, 1))  # 使用numpy的transpose替代permute

    batch_size = preds.shape[0]
    output = [np.zeros((0, 6))] * batch_size

    for b in range(batch_size):
        pred = preds[b]  # [num_preds, 84]

        # 提取框坐标 (xywh) 和类别概率
        bbox = pred[:, :4]  # [n, 4]
        cls_probs = pred[:, 4:]  # [n, 80]

        # 计算每个预测框的最大类别概率和对应的类别索引
        max_cls_prob = np.max(cls_probs, axis=1)  # [n]
        max_cls_idx = np.argmax(cls_probs, axis=1)  # [n]

        # 置信度筛选
        conf_mask = max_cls_prob > conf_thres
        bbox = bbox[conf_mask]
        max_cls_prob = max_cls_prob[conf_mask]
        max_cls_idx = max_cls_idx[conf_mask]

        if bbox.shape[0] == 0:
            continue

        # 转换框坐标 xywh -> xyxy
        boxes = xywh2xyxy(bbox)

        # 构建检测结果 [x1, y1, x2, y2, conf, class]
        detections = np.column_stack([
            boxes,
            max_cls_prob,
            max_cls_idx.astype(float)
        ])

        # 按置信度降序排序
        sorted_indices = np.argsort(detections[:, 4])[::-1]
        detections = detections[sorted_indices[:max_nms]]

        # NMS 处理
        boxes_tensor = torch.from_numpy(detections[:, :4])
        scores_tensor = torch.from_numpy(detections[:, 4])
        keep_indices = torchvision.ops.nms(boxes_tensor, scores_tensor, iou_thres)
        keep_indices = keep_indices[:max_det].numpy()

        output[b] = detections[keep_indices]

    # 只处理 batch=1 的情况方便演示
    if batch_size == 1:
        output = output[0]

        if output.shape[0] == 0:
            return np.zeros((0, 6))

        # 坐标反变换（去掉padding，恢复缩放）
        output[:, [0, 2]] -= dw
        output[:, [1, 3]] -= dh
        output[:, :4] /= r

        # 裁剪边界
        output[:, [0, 2]] = output[:, [0, 2]].clip(0, w0)
        output[:, [1, 3]] = output[:, [1, 3]].clip(0, h0)

        # 类别映射
        class_ids = output[:, 5].astype(int)
        output[:, 5] = np.array(coco80_to_coco91)[class_ids]

        # 转回 xywh 左上角坐标格式
        boxes = xyxy2xywh(output[:, :4])
        boxes[:, :2] -= boxes[:, 2:] / 2
        output[:, :4] = boxes

    return output