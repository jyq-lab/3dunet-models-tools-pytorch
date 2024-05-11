import torch
import numpy


# accessing dice with '.'
class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def calculate_confusion_matrix(y_pred, y_true):

    if isinstance(y_true, numpy.ndarray) and isinstance(y_pred, numpy.ndarray):
        y_pred, y_true = y_pred.astype(bool), y_true.astype(bool)

        TP = (y_true * y_pred).sum()
        FP = ((1 - y_true) * y_pred).sum()
        FN = (y_true * (1 - y_pred)).sum()
    elif isinstance(y_true, torch.Tensor) and isinstance(y_pred, torch.Tensor):
        y_pred, y_true = y_pred.bool(), y_true.bool()

        TP = (y_true * y_pred).sum()
        FP = ( ~y_true * y_pred ).sum()
        FN = ( y_true * ~y_pred ).sum()
    else:
        print(" type error !")
    
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0.0                                            # precision
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0.0                                               # recall
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0.0 # F1
    return precision, recall, f1_score


# dice score
def dice_score(data1, data2): 
    smooth = 1e-5
    dice = (2.0 * (data1 * data2).sum() + smooth) / (data1.sum() + data2.sum() + smooth)
    return dice
