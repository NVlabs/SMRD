# MIT License
"""
MIT License

Copyright (c) 2021 The University of Texas Computational Sensing and Imaging Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

"""
------------------------------------------------------------------------------
Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

This work is made available under the Nvidia Source Code License.
To view a copy of this license, visit
https://github.com/batuozt/SMRD/blob/master/LICENSE.md

Written by Batu Ozturkler
------------------------------------------------------------------------------
"""

import torch
import torch.fft as torch_fft
import numpy as np
import sigpy as sp
from torch import nn
from torch.nn import functional as F
from PIL import Image
import os
from torchvision import transforms
import shutil
from collections import OrderedDict
import math
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import torchvision
from datetime import datetime
import glob
import torch.distributed as dist
import h5py
import functools
import logging
import warnings
import pickle
import re
import copy
from torch.optim import Adam
from typing import Optional
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def ssim(
    gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None
) -> np.ndarray:
    """Compute Structural Similarity Index Metric (SSIM)"""
    if not gt.ndim == 3:
        raise ValueError("Unexpected number of dimensions in ground truth.")
    if not gt.ndim == pred.ndim:
        raise ValueError("Ground truth dimensions does not match pred.")

    maxval = gt.max() if maxval is None else maxval

    ssim = np.array([0])
    for slice_num in range(gt.shape[0]):
        ssim = ssim + structural_similarity(
            gt[slice_num], pred[slice_num], data_range=maxval
        )

    return ssim / gt.shape[0]

def psnr(
    gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None
) -> np.ndarray:
    """Compute Peak Signal to Noise Ratio metric (PSNR)"""
    if maxval is None:
        maxval = gt.max()
    return peak_signal_noise_ratio(gt, pred, data_range=maxval)

def scale(img):
    img = img.detach().cpu().numpy()
    magnitude_vals = np.abs(img).reshape(-1)
    if img.shape[0] == 320:
        k = int(round(0.015 * torch.from_numpy(magnitude_vals).numel()))
    else:
        k = int(round(0.02 * torch.from_numpy(magnitude_vals).numel()))
    scale = torch.min(torch.topk(torch.from_numpy(magnitude_vals), k).values)
    img = torch.clip(img / scale, 0, 1)
    return img

def normalize(gen_img, estimated_mvue):
    '''
        Estimate mvue from coils and normalize with 99% percentile.
    '''
    scaling = torch.quantile(estimated_mvue.abs(), 0.99)
    return gen_img * scaling

def unnormalize(gen_img, estimated_mvue):
    '''
        Estimate mvue from coils and normalize with 99% percentile.
    '''
    scaling = torch.quantile(estimated_mvue.abs(), 0.99)
    return gen_img / scaling

# Multicoil forward operator for MRI
class MulticoilForwardMRI(nn.Module):
    def __init__(self, orientation):
        self.orientation = orientation
        super(MulticoilForwardMRI, self).__init__()
        return

    # Centered, orthogonal ifft in torch >= 1.7
    def _ifft(self, x):
        x = torch_fft.ifftshift(x, dim=(-2, -1))
        x = torch_fft.ifft2(x, dim=(-2, -1), norm='ortho')
        x = torch_fft.fftshift(x, dim=(-2, -1))
        return x

    # Centered, orthogonal fft in torch >= 1.7
    def _fft(self, x):
        x = torch_fft.fftshift(x, dim=(-2, -1))
        x = torch_fft.fft2(x, dim=(-2, -1), norm='ortho')
        x = torch_fft.ifftshift(x, dim=(-2, -1))
        return x


    '''
    Inputs:
     - image = [B, H, W] torch.complex64/128    in image domain
     - maps  = [B, C, H, W] torch.complex64/128 in image domain
     - mask  = [B, W] torch.complex64/128 w/    binary values
    Outputs:
     - ksp_coils = [B, C, H, W] torch.complex64/128 in kspace domain
    '''
    def forward(self, image, maps, mask):
        # Broadcast pointwise multiply
        coils = image[:, None] * maps

        # Convert to k-space data
        ksp_coils = self._fft(coils)

        if self.orientation == 'vertical':
            # Mask k-space phase encode lines
            ksp_coils = ksp_coils * mask[:, None, None, :]
        elif self.orientation == 'horizontal':
            # Mask k-space frequency encode lines
            ksp_coils = ksp_coils * mask[:, None, :, None]
        else:
            if len(mask.shape) == 3:
                ksp_coils = ksp_coils * mask[:, None, :, :]
            else:
                raise NotImplementedError('mask orientation not supported')


        # Return downsampled k-space
        return ksp_coils

def get_mvue(kspace, s_maps):
    ''' Get mvue estimate from coil measurements '''
    return np.sum(sp.ifft(kspace, axes=(-1, -2)) * np.conj(s_maps), axis=1) / np.sqrt(np.sum(np.square(np.abs(s_maps)), axis=1))
    
def get_all_files(folder, pattern='*'):
    files = [x for x in glob.iglob(os.path.join(folder, pattern))]
    return sorted(files)

# Source: https://stackoverflow.com/questions/3229419/how-to-pretty-print-nested-dictionaries
def pretty(d, indent=0):
    ''' Print dictionary '''
    for key, value in d.items():
      print('\t' * indent + str(key))
      if isinstance(value, dict):
         pretty(value, indent+1)
      else:
         print('\t' * (indent+1) + str(value))

def save_images(samples, loc, normalize=False):
    torchvision.utils.save_image(
        samples,
        loc,
        nrow=int(samples.shape[0] ** 0.5),
        normalize=normalize,
        scale_each=True)

def load_dict(model, ckpt, device='cuda'):
    state_dict = torch.load(ckpt, map_location=device)
    try:
        model.load_state_dict(state_dict)
    except:
        print('Loading model failed... Trying to remove the module from the keys...')
        new_state_dict = OrderedDict()
        for key, value in state_dict.items():
            new_state_dict[key[len('module.'):]] = value
        model.load_state_dict(new_state_dict)
    return model

def mp_setup(rank, world_size, port=12345):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def mp_cleanup():
    dist.destroy_process_group()


def update_pbar_desc(pbar, metrics, labels):
    pbar_string = ''
    for metric, label in zip(metrics, labels):
        pbar_string += f'{label}: {metric:.7f}; '
    pbar.set_description(pbar_string)


class MpLogger:
    def __init__(self, logger, rank):
        self.logger = logger
        self.rank = rank

    def info(self, message):
        if self.rank == 0:
            self.logger.info(message)

