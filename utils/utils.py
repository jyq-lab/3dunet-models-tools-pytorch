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


def compute_bounding_box(mask):
  """Computes the bounding box of the masks.

  This function generalizes to arbitrary number of dimensions great or equal
  to 1.

  Args:
    mask: The 2D or 3D numpy mask, where '0' means background and non-zero means
      foreground.

  Returns:
    A tuple:
     - The coordinates of the first point of the bounding box (smallest on all
       axes), or `None` if the mask contains only zeros.
     - The coordinates of the second point of the bounding box (greatest on all
       axes), or `None` if the mask contains only zeros.
  """
  num_dims = len(mask.shape)
  bbox_min = np.zeros(num_dims, np.int64)
  bbox_max = np.zeros(num_dims, np.int64)

  # max projection to the x0-axis
  proj_0 = np.amax(mask, axis=tuple(range(num_dims))[1:])
  idx_nonzero_0 = np.nonzero(proj_0)[0]
  if len(idx_nonzero_0) == 0:  # pylint: disable=g-explicit-length-test
    return None, None

  bbox_min[0] = np.min(idx_nonzero_0)
  bbox_max[0] = np.max(idx_nonzero_0)

  # max projection to the i-th-axis for i in {1, ..., num_dims - 1}
  for axis in range(1, num_dims):
    max_over_axes = list(range(num_dims))  # Python 3 compatible
    max_over_axes.pop(axis)  # Remove the i-th dimension from the max
    max_over_axes = tuple(max_over_axes)  # numpy expects a tuple of ints
    proj = np.amax(mask, axis=max_over_axes)
    idx_nonzero = np.nonzero(proj)[0]
    bbox_min[axis] = np.min(idx_nonzero)
    bbox_max[axis] = np.max(idx_nonzero)

  return bbox_min, bbox_max
