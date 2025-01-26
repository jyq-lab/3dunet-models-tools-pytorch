import math
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