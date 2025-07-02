import numpy as np
import torch
import openvino as ov
import torchvision

import dsPreProcessor



preds = dsPreProcessor.run_inference()

bs = preds.shape[0]      # batch size
nc = preds.shape[2] - 5  # number of classes
conf_thres = 0.001       # 置信度筛选阈值
max_wh = 7680            # 最大允许的检测框高宽
iou_thres = 0.65         # NMS交并比
max_nms = 30000          # 最多允许30000个框进入NMS
max_det=300              # 保留NMS后的前300个框
output = [np.zeros((0, 6))] * bs# 用来装NMS后的前300个框


# （左上角坐标，右下角坐标）转 （检测框中心坐标，检测框宽高）
def xyxy2xywh(x):
    #Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right.
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



xc = preds[..., 4] > conf_thres  # 用来筛选出置信度 > 0.001的检测框
print('xc.shape', xc.shape)
print('xc', xc)

# 这个for循环是为了遍历每个batch，此时演示的batch=1，所以只会循环一次，循环内x.shape=(25200,85)
for xi, x in enumerate(preds):
    # xi = 当前遍历的batch_index， x = 筛选置信度大于0.001的检测框
    x = x[xc[xi]]
    # 所有coco_class_score* 乘 confidence
    x[:, 5:] *= x[:, 4:5]
    # 检测框坐标转换，（检测框中心坐标，检测框宽高）转（左上角坐标，右下角坐标）
    box = xywh2xyxy(x[:, :4])
    # mask暂时没什么用
    mask = x[:, 85:]
    # i = 满足要求的检测框index， j = 满足要求的coco类别index
    i, j = np.where(x[:, 5:] > conf_thres)
    # 拼接矩阵，最后x矩阵一共6列, x = (0:4检测框坐标, 4coco_class_score, 5coco_class, mask是空不会占一列)
    x = np.concatenate((box[i], x[i, 5 + j, None], j[:, None].astype(float), mask[i]), axis=1)
    print('⭐x.shape', x.shape)
    # x矩阵根据第五列进行降序排序
    sorted_indices = np.argsort(x[:, 4])[::-1]
    # 保留x排序后的前30000行
    x = x[sorted_indices[:max_nms]]
    # 🤓c的作用意义不明
    c = x[:, 5:6] * max_wh  # classes
    # boxes (offset by class), scores
    boxes, scores = x[:, :4] + c, x[:, 4]
    boxes_tensor = torch.from_numpy(boxes)
    scores_tensor = torch.from_numpy(scores)
    # NMS
    i = torchvision.ops.nms(boxes_tensor, scores_tensor, iou_thres)
    # 保留前300个框
    i = i[:max_det]  # limit detections
    output[xi] = x[i]
    output = np.array(output)
    print('⭐output.shape', output.shape)
    print('⭐output👇\n', output)

    # 因为本例子中，batch=1, 所以output的第一个维度可以不要
    print('压缩维度前的output.shape', output.shape)
    output = np.squeeze(output, axis=0)
    print('压缩维度后的output.shape', output.shape)

    # 坐标映射👇
    # dw 和 dh 就是预处理时候填充的宽度和高度
    output[..., [0, 2]] -= dw  # x padding
    output[..., [1, 3]] -= dh  # y padding
    # 这里的r，就是预处理时候的r，图像缩放比例
    output[..., :4] /= r
    # 防止检测框坐标超出图像的范围
    output[..., [0, 2]] = output[..., [0, 2]].clip(0, w0)  # x1, x2
    output[..., [1, 3]] = output[..., [1, 3]].clip(0, h0)  # y1, y2


    # 类别映射👇
    # 映射列表
    def coco80_to_coco91_class():
        """
        Converts COCO 80-class index to COCO 91-class index used in the paper.

        Reference: https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
        """
        # a = np.loadtxt('data/coco.names', dtype='str', delimiter='\n')
        # b = np.loadtxt('data/coco_paper.names', dtype='str', delimiter='\n')
        # x1 = [list(a[i] == b).index(True) + 1 for i in range(80)]  # darknet to coco
        # x2 = [list(b[i] == a).index(True) if any(b[i] == a) else None for i in range(91)]  # coco to darknet
        return [
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            27,
            28,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            40,
            41,
            42,
            43,
            44,
            46,
            47,
            48,
            49,
            50,
            51,
            52,
            53,
            54,
            55,
            56,
            57,
            58,
            59,
            60,
            61,
            62,
            63,
            64,
            65,
            67,
            70,
            72,
            73,
            74,
            75,
            76,
            77,
            78,
            79,
            80,
            81,
            82,
            84,
            85,
            86,
            87,
            88,
            89,
            90,
        ]


    # 创建一个表
    class_map = np.array(coco80_to_coco91_class())
    # class_index留着后面画框用
    class_index = output[:, 5]
    class_index = class_index.astype(int)
    # 映射
    output[:, 5] = class_map[output[:, 5].astype(int)]
    # 最后把检测框坐标（左上角，右下角）转（左上角，宽高）
    box = xyxy2xywh(output[:, :4])  # xywh
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    output[:, :4] = box

    # 🏁🏁🏁至此，output的内容和yolov5的val.py脚本的后处理结果一致🏁🏁🏁
    # 可以在CLI用val.py运行一次：val.py --weights yolov5n_openvino_model --batch-size 1 --data coco.yaml --img 640 --conf 0.001 --iou 0.65
    # 在yolov5/runs/val/exp/yolov5n_openvino_model_predictions.json中查找"image_id": 285可以对比output结果
    print(output)

