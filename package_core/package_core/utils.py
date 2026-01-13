from __future__ import division
import shutil
import numpy as np
import torch

from path import Path
import datetime
from collections import OrderedDict
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

def high_res_colormap(low_res_cmap, resolution=1000, max_value=1):
    # Construct the list colormap, with interpolated values for higer resolution
    # For a linear segmented colormap, you can just specify the number of point in
    # cm.get_cmap(name, lutsize) with the parameter lutsize
    x = np.linspace(0, 1, low_res_cmap.N)
    low_res = low_res_cmap(x)
    new_x = np.linspace(0, max_value, resolution)
    high_res = np.stack([np.interp(new_x, x, low_res[:, i])
                         for i in range(low_res.shape[1])], axis=1)
    return ListedColormap(high_res)


def opencv_rainbow(resolution=1000):
    # Construct the opencv equivalent of Rainbow
    opencv_rainbow_data = (
        (0.000, (1.00, 0.00, 0.00)),
        (0.400, (1.00, 1.00, 0.00)),
        (0.600, (0.00, 1.00, 0.00)),
        (0.800, (0.00, 0.00, 1.00)),
        (1.000, (0.60, 0.00, 1.00))
    )

    return LinearSegmentedColormap.from_list('opencv_rainbow', opencv_rainbow_data, resolution)


COLORMAPS = {'rainbow': opencv_rainbow(),
             'magma': high_res_colormap(cm.get_cmap('magma')),
             'bone': cm.get_cmap('bone', 10000)}


def tensor2array(tensor, max_value=None, colormap='rainbow'):
    tensor = tensor.detach().cpu()
    if max_value is None:
        max_value = tensor.max().item()
    if tensor.ndimension() == 2 or tensor.size(0) == 1:
        norm_array = tensor.squeeze().numpy()/max_value
        array = COLORMAPS[colormap](norm_array).astype(np.float32)
        array = array.transpose(2, 0, 1)

    elif tensor.ndimension() == 3:
        assert(tensor.size(0) == 3)
        array = 0.5 + tensor.numpy()*0.5
    return array

def generate_2D_mesh(H, W):
    # Re-implmentation of meshzoo.rectangle to remove dependency
    # Create vertices indices
    idx = np.arange(H * W).reshape(H, W)
    
    # Get indices for quads (top-left corners)
    # We want indices that correspond to the top-left of every quad
    # i.e., all rows except last, all cols except last
    v0 = idx[:-1, :-1].reshape(-1)
    v1 = idx[:-1, 1:].reshape(-1)
    v2 = idx[1:, :-1].reshape(-1)
    v3 = idx[1:, 1:].reshape(-1)
    
    # Create two triangles for each quad: (v0, v1, v2) and (v1, v3, v2)
    faces_a = np.stack([v0, v1, v2], axis=1)
    faces_b = np.stack([v1, v3, v2], axis=1)
    
    # Interleave them to match a zigzag/grid pattern if needed, or just concat
    faces = np.vstack([faces_a, faces_b])

    x = torch.arange(0, W, 1).float().cuda() 
    y = torch.arange(0, H, 1).float().cuda()

    xx = x.repeat(H, 1)
    yy = y.view(H, 1).repeat(1, W)
    
    grid = torch.stack([xx, yy], dim=0) 
        
    return grid, faces

