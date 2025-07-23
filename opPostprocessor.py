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
        self.coco_dict = {idx: name for idx, name in enumerate(self.coco_names)}


        # 图像尺寸（模型训练尺寸）
        self.model_size = 640

    def xywh2xyxy(self,*box):
        """
        将xywh转换为左上角点和左下角点
        Args:
            box:
        Returns: x1y1x2y2
        """
        ret = [box[0] - box[2] // 2, box[1] - box[3] // 2, \
               box[0] + box[2] // 2, box[1] + box[3] // 2]
        return ret

    def get_inter(self,box1, box2):
        """
        计算相交部分面积
        Args:
            box1: 第一个框
            box2: 第二个
        Returns: 相交部分的面积
        """
        x1, y1, x2, y2 = self.xywh2xyxy(*box1)
        x3, y3, x4, y4 = self.xywh2xyxy(*box2)
        # 验证是否存在交集
        if x1 >= x4 or x2 <= x3:
            return 0
        if y1 >= y4 or y2 <= y3:
            return 0
        # 将x1,x2,x3,x4排序，因为已经验证了两个框相交，所以x3-x2就是交集的宽
        x_list = sorted([x1, x2, x3, x4])
        x_inter = x_list[2] - x_list[1]
        # 将y1,y2,y3,y4排序，因为已经验证了两个框相交，所以y3-y2就是交集的宽
        y_list = sorted([y1, y2, y3, y4])
        y_inter = y_list[2] - y_list[1]
        # 计算交集的面积
        inter = x_inter * y_inter
        return inter

    def get_iou(self,box1, box2):
        """
        计算交并比： (A n B)/(A + B - A n B)
        Args:
            box1: 第一个框
            box2: 第二个框
        Returns:  # 返回交并比的值
        """
        box1_area = box1[2] * box1[3]  # 计算第一个框的面积
        box2_area = box2[2] * box2[3]  # 计算第二个框的面积
        inter_area = self.get_inter(box1, box2)
        union = box1_area + box2_area - inter_area  # (A n B)/(A + B - A n B)
        iou = inter_area / union
        return iou


    def post_process(self, preds, dw, dh, r, w0, h0):
        """完全重构的后处理函数"""
        # 转换形状 (B, C, N) -> (B, N, C)
        if isinstance(preds, np.ndarray):
            preds = np.transpose(preds, (0, 2, 1))
        elif isinstance(preds, torch.Tensor):
            preds = preds.permute(0, 2, 1).cpu().numpy()

        batch_size = preds.shape[0]


        for b in range(batch_size):#其实b只有1，但是当有多batch可以适用
            pred = preds[b] # [num_preds, 84]
          #无激活函数
            pred_class = pred[..., 4:]#后面80个，因为是yolo8没有单独置信度
            pred_conf = np.max(pred_class, axis=-1)#此处是后面84个的最大值
            pred = np.insert(pred, 4, pred_conf, axis=-1)      #此处已经是（8400，85）
            """
            # sigmoid 仅应用到 [4:]（即 80 个类别）
            pred[:, 4:] = 1 / (1 + np.exp(-pred[:, 4:]))
            pred_class = pred[..., 4:]

            # 最大类别概率作为置信度
            pred_conf = np.max(pred[:, 4:], axis=-1)

            # 插入置信度列
            pred = np.insert(pred, 4, pred_conf, axis=-1)  # shape: (8400, 85)

            #bbox = pred[:, :4]  # [x, y, w, h] 相对坐标  
            """


            box = pred[pred[..., 4] > self.conf_thres]  # 置信度筛选

            cls_conf = box[..., 5:]#后面80个置信度筛选后的
            cls = []
            for i in range(len(cls_conf)):
                cls.append(int(np.argmax(cls_conf[i])))#cls根据置信度最大值记录每个框的类别Id

            total_cls = list(set(cls))  # 记录图像内共出现几种物体，
            output_box = []
            # 每个预测类别分开考虑
            for i in range(len(total_cls)):
                clss = total_cls[i]
                cls_box = []
                temp = box[:, :6]#temp 截取了前6列，格式为 [x, y, w, h, conf, class]
                for j in range(len(cls)):#对于置信度筛选后的每个框
                    # 记录[x,y,w,h,conf(最大类别概率),class]值
                    if cls[j] == clss:
                        temp[j][5] = clss
                        cls_box.append(temp[j][:6])
                #  cls_box 里面是[x,y,w,h,conf(最大类别概率),class]，只包含当前class
                cls_box = np.array(cls_box)
                sort_cls_box = sorted(cls_box, key=lambda x: -x[4])  # 将cls_box按置信度从大到小排序

                # 得到置信度最大的预测框
                max_conf_box = sort_cls_box[0]
                output_box.append(max_conf_box)
                sort_cls_box = np.delete(sort_cls_box, 0, 0)
                # 对除max_conf_box外其他的框进行非极大值抑制
                while len(sort_cls_box) > 0:
                    # 得到当前最大的框
                    max_conf_box = output_box[-1]
                    del_index = []
                    for j in range(len(sort_cls_box)):
                        current_box = sort_cls_box[j]
                        iou = self.get_iou(max_conf_box, current_box)
                        if iou > self.iou_thres:
                            # 筛选出与当前最大框Iou大于阈值的框的索引
                            del_index.append(j)
                    # 删除这些索引
                    sort_cls_box = np.delete(sort_cls_box, del_index, 0)
                    if len(sort_cls_box) > 0:
                        output_box.append(sort_cls_box[0])
                        sort_cls_box = np.delete(sort_cls_box, 0, 0)

            print("最终保留的检测框 output_box:")
            for i, box in enumerate(output_box):
                x, y, w, h, conf, cls_id = box
                print(
                    f"{i + 1}: 类别ID={int(cls_id)}, 置信度={conf:.4f}, 框=[x={x:.1f}, y={y:.1f}, w={w:.1f}, h={h:.1f}]")
            return output_box

    def xywh2xyxy(self,*box):
        """
        将xywh转换为左上角点和左下角点
        Args:
            box:
        Returns: x1y1x2y2
        """
        ret = [box[0] - box[2] // 2, box[1] - box[3] // 2, \
               box[0] + box[2] // 2, box[1] + box[3] // 2]
        return ret

    def cod_trf(self,result, pre, after):
        """
        因为预测框是在经过letterbox后的图像上做预测所以需要将预测框的坐标映射回原图像上
        Args:
            result:  [x,y,w,h,conf(最大类别概率),class]
            pre:    原尺寸图像
            after:  经过letterbox处理后的图像
        Returns: 坐标变换后的结果,
        """
        res = np.array(result)
        x, y, w, h, conf, cls = res.transpose((1, 0))
        x1, y1, x2, y2 = self.xywh2xyxy(x, y, w, h)  # 左上角点和右下角的点
        h_pre, w_pre, _ = pre.shape
        h_after, w_after, _ = after.shape
        scale = max(w_pre / w_after, h_pre / h_after)  # 缩放比例
        h_pre, w_pre = h_pre / scale, w_pre / scale  # 计算原图在等比例缩放后的尺寸
        x_move, y_move = abs(w_pre - w_after) // 2, abs(h_pre - h_after) // 2  # 计算平移的量
        ret_x1, ret_x2 = (x1 - x_move) * scale, (x2 - x_move) * scale
        ret_y1, ret_y2 = (y1 - y_move) * scale, (y2 - y_move) * scale
        ret = np.array([ret_x1, ret_y1, ret_x2, ret_y2, conf, cls]).transpose((1, 0))
        return ret

    def draw(self,res, image, cls):
        """
        将预测框绘制在image上
        Args:
            res: 预测框数据
            image: 原图
            cls: 类别列表，类似["apple", "banana", "people"]  根据coco映射
        Returns:
        """
        for r in res:
            # 画框
            image = cv2.rectangle(image, (int(r[0]), int(r[1])), (int(r[2]), int(r[3])), (255, 0, 0), 1)
            # 表明类别
            text = "{}:{}".format(cls[int(r[5])], round(float(r[4]), 2))
            h, w = int(r[3]) - int(r[1]), int(r[2]) - int(r[0])  # 计算预测框的长宽
            font_size = min(h / 640, w / 640) * 3  # 计算字体大小（随框大小调整）
            image = cv2.putText(image, text, (max(10, int(r[0])), max(20, int(r[1]))), cv2.FONT_HERSHEY_COMPLEX,
                                max(font_size, 0.3), (0, 0, 255), 1)  # max()为了确保字体不过界
        # 移除cv2.imshow和cv2.waitKey，不再弹窗展示
        return image

