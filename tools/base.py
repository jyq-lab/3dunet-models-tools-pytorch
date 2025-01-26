import torch
import numpy

# accessing dict with '.'
class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def neg(idx):
    return None if idx == 0 else -idx



