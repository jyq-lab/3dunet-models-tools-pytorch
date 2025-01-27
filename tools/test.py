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