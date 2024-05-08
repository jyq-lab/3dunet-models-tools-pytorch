import numpy as np
import torch
from typing import Union, Tuple, List
def compute_steps_for_sliding_window(image_size: Tuple[int, ...], tile_size: Tuple[int, ...], tile_step_size: float) -> \
        List[List[int]]:
    """from nnU-Net: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/inference/sliding_window_prediction.py"""
    
    assert [i >= j for i, j in zip(image_size, tile_size)], "image size must be as large or larger than patch_size"
    assert 0 < tile_step_size <= 1, 'step_size must be larger than 0 and smaller or equal to 1'

    # our step width is patch_size*step_size at most, but can be narrower. For example if we have image size of
    # 110, patch size of 64 and step_size of 0.5, then we want to make 3 steps starting at coordinate 0, 23, 46
    target_step_sizes_in_voxels = [i * tile_step_size for i in tile_size]
    
    num_steps = [int(np.ceil((i - k) / j)) + 1 for i, j, k in zip(image_size, target_step_sizes_in_voxels, tile_size)]
    
    steps = []
    for dim in range(len(tile_size)):
        # the highest step value for this dimension is
        max_step_value = image_size[dim] - tile_size[dim]
        if num_steps[dim] > 1:
            actual_step_size = max_step_value / (num_steps[dim] - 1)
        else:
            actual_step_size = 99999999999  # does not matter because there is only one step at 0
            
        steps_here = [int(np.round(actual_step_size * i)) for i in range(num_steps[dim])]

        steps.append(steps_here)

    return steps


class Split_Comb():
    '''overlap split and vote comb'''
    def __init__(self, image_size, patch_size, step_size=1):
        self.image_size = image_size
        self.patch_size  = patch_size
        self.dim = len(self.image_size)
        
        # The starting coordinates of the patch in each dimension. from nnU-Net
        self.steps = compute_steps_for_sliding_window(image_size, patch_size, step_size)
        
    def split(self, image):
        assert image.shape == self.image_size, "input image shape is different from image_size"
        
        slicers, origin_points = [], []
        # make slicers origin_points
        for sx in self.steps[0]:
            for sy in self.steps[1]:
                if self.dim == 3:
                    for sz in self.steps[2]:
                        origin_points.append([sx, sy, sz])
                elif self.dim == 2:
                    origin_points.append([sx, sy])
                    
        # make slicers
        for _origin in origin_points:
            if self.dim == 3:
                slicers.append(
                    image[
                        _origin[0]:_origin[0]+self.patch_size[0],
                        _origin[1]:_origin[1]+self.patch_size[1],
                        _origin[2]:_origin[2]+self.patch_size[2]])
            elif self.dim == 2:
                slicers.append(
                    image[
                        _origin[0]:_origin[0]+self.patch_size[0],
                        _origin[1]:_origin[1]+self.patch_size[1]])
                
        self._origin_points = origin_points
        return slicers
    
    def combine(self, pre_split):
        
        if isinstance(pre_split[0], np.ndarray):
            output = np.zeros(self.image_size)
            occur = np.zeros(self.image_size)
        elif isinstance(pre_split[0], torch.Tensor):
            output = torch.zeros(self.image_size, device=pre_split[0].device)
            occur = torch.zeros(self.image_size, device=pre_split[0].device)
            
        for data, _origin in zip(pre_split, self._origin_points):
            if self.dim == 3:
                output[
                    _origin[0]:_origin[0]+self.patch_size[0],
                    _origin[1]:_origin[1]+self.patch_size[1],
                    _origin[2]:_origin[2]+self.patch_size[2],] += data
                occur[
                    _origin[0]: _origin[0]+self.patch_size[0], 
                    _origin[1]: _origin[1]+self.patch_size[1],
                    _origin[2]: _origin[2]+self.patch_size[2],] += 1
            elif self.dim == 2:
                output[
                    _origin[0]:_origin[0]+self.patch_size[0],
                    _origin[1]:_origin[1]+self.patch_size[1],] += data
                occur[
                    _origin[0]: _origin[0]+self.patch_size[0], 
                    _origin[1]: _origin[1]+self.patch_size[1],] += 1
                
        return output/occur