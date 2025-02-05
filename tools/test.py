import math
import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import directed_hausdorff

# dice score
def dice_score(data1, data2): 
    smooth = 1e-5
    dice = (2.0 * (data1 * data2).sum() + smooth) / (data1.sum() + data2.sum() + smooth)
    return dice

# distance
def euclidean_distance(p1, p2):
    z1, y1, x1 = p1
    z2, y2, x2 = p2
    return math.sqrt((z2 - z1) ** 2 + (y2 - y1) ** 2 + (x2 - x1) ** 2)

def calculate(pred, true):
    y_pred = pred.flatten()
    y_true = true.flatten()
    cm = confusion_matrix(y_pred, y_true)
    TP = cm[1, 1]
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    specificity = TN / (TN + FP)
    recall = TP / (TP + FN)
    iou = TP / (FP + TP + FN)
    dice = 2*TP / (FP + 2*TP + FN)
    
    return accuracy, precision, specificity, recall, iou, dice

def hausdorff_distance(pred, true):
    points1 = np.array(np.where(pred == 1)).T
    points2 = np.array(np.where(true == 1)).T   
    h1 = directed_hausdorff(points1, points2)[0]
    h2 = directed_hausdorff(points2, points1)[0]
    hd = max(h1, h2)

    dist_seg_label = [min(np.linalg.norm(s - l) for l in points2) for s in points1]
    dist_label_seg = [min(np.linalg.norm(l - s) for s in points1) for l in points2]
    all_dists = sorted(dist_seg_label + dist_label_seg)
    idx_95 = int(0.95 * len(all_dists))
    hd_95 = all_dists[idx_95]

    return hd, hd_95


def bb_iou_3d(min1, max1, min2, max2):
    """
    计算两个三维矩形框的交并比（IoU）
    :param min1: 第一个矩形框的最小点 [z, y, x]
    :param max1: 第一个矩形框的最大点 [z, y, x]
    :param min2: 第二个矩形框的最小点 [z, y, x]
    :param max2: 第二个矩形框的最大点 [z, y, x]
    :return: IoU 值
    """

    # 计算交集的最小点和最大点
    inter_min = [max(min1[0], min2[0]), max(min1[1], min2[1]), max(min1[2], min2[2])]
    inter_max = [min(max1[0], max2[0]), min(max1[1], max2[1]), min(max1[2], max2[2])]

    # 计算交集的尺寸
    inter_dim = [max(0, inter_max[0] - inter_min[0]),
                 max(0, inter_max[1] - inter_min[1]),
                 max(0, inter_max[2] - inter_min[2])]

    # 计算交集体积
    inter_volume = inter_dim[0] * inter_dim[1] * inter_dim[2]
    # 计算第一个矩形框的体积
    volume1 = (max1[0] - min1[0]) * (max1[1] - min1[1]) * (max1[2] - min1[2])
    # 计算第二个矩形框的体积
    volume2 = (max2[0] - min2[0]) * (max2[1] - min2[1]) * (max2[2] - min2[2])
    # 计算并集体积
    union_volume = volume1 + volume2 - inter_volume
    # 计算 IoU
    iou = inter_volume / union_volume if union_volume != 0 else 0

    return iou
