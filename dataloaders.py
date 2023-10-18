#!/usr/bin/env python
#

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

from scipy.ndimage.interpolation import rotate
from torch.utils.data import Dataset
import os
import numpy as np
import torch
import h5py
import sigpy as sp
from xml.etree import ElementTree as ET
from utils import *

from monai.data.fft_utils import ifftn_centered, fftn_centered
from monai.transforms.utils import resize_center
from monai.apps.reconstruction.complex_utils import complex_conj
from monai.apps.reconstruction.transforms.array import RandomKspaceMask, EquispacedKspaceMask
from monai.apps.reconstruction.fastmri_reader import FastMRIReader

def _ifft_centered_single_last_axis(x:torch.Tensor, axis_dim:int=-1, ):
    '''
    ifftn_centered for single last axis, 
    axis_dim=-1 then last dim is the axis to ifft, 
    axis_dim=-2 then second last dim is the axis to ifft

    Args:
    x: input complex tensor
    axis_dim: axis to ifft [-1]
    '''

    assert axis_dim in [-1, -2], 'axis_dim must be -1 or -2'

    ifft_inp = x
    if axis_dim == -2:
        ifft_inp = torch.transpose(ifft_inp, -1, -2)
    ifft_inp = torch.view_as_real(ifft_inp)
    ifft = ifftn_centered(ifft_inp, spatial_dims=1, is_complex=True)
    ifft = torch.view_as_complex(ifft)
    if axis_dim == -2:
        ifft = torch.transpose(ifft, -1, -2)
    return ifft

def _fft_centered_single_last_axis(x:torch.Tensor, axis_dim:int=-1, ):
    '''
    fftn_centered for single last axis, 
    axis_dim=-1 then last dim is the axis to ifft, 
    axis_dim=-2 then second last dim is the axis to ifft

    Args:
    x: input complex tensor
    axis_dim: axis to ifft [-1]
    '''

    assert axis_dim in [-1, -2], 'axis_dim must be -1 or -2'

    fft_inp = x
    if axis_dim == -2:
        fft_inp = torch.transpose(fft_inp, -1, -2)
    fft_inp = torch.view_as_real(fft_inp)
    fft = fftn_centered(fft_inp, spatial_dims=1, is_complex=True)
    fft = torch.view_as_complex(fft)
    if axis_dim == -2:
        fft = torch.transpose(fft, -1, -2)
    return fft

def get_mvue_monai(
        kspace: torch.Tensor,
        s_maps: torch.Tensor,
        ):
    ''' Get mvue estimate from coil measurements '''
    ifft_kspace = torch.view_as_complex(ifftn_centered(torch.view_as_real(kspace), spatial_dims=2, is_complex=True))
    s_maps_conj = torch.view_as_complex(complex_conj(torch.view_as_real(s_maps)))
    mvue = torch.sum(ifft_kspace * s_maps_conj, dim=1) / torch.sqrt(torch.sum(torch.square(torch.abs(s_maps)), dim=1))
    return mvue

class MVU_Estimator_Brain(Dataset):
    def __init__(self, file_list, maps_dir, input_dir,
                 project_dir='./',
                 R=1,
                 image_size=(384,384),
                 acs_size=26,
                  pattern='random',
                 orientation='vertical'):
        # Attributes
        self.project_dir = project_dir
        self.file_list    = file_list
        self.maps_dir     = maps_dir
        self.input_dir      = input_dir
        self.image_size = image_size
        self.R            = R
        self.pattern      = pattern
        self.orientation  = orientation

        # Access meta-data of each scan to get number of slices
        self.num_slices = np.zeros((len(self.file_list,)), dtype=int)
        for idx, file in enumerate(self.file_list):
            input_file = os.path.join(self.input_dir, os.path.basename(file))
            with h5py.File(os.path.join(self.project_dir, input_file), 'r') as data:
                self.num_slices[idx] = int(np.array(data['kspace']).shape[0])
        # Create cumulative index for mapping
        self.slice_mapper = np.cumsum(self.num_slices) - 1 # Counts from '0'

    def __len__(self):
        return int(np.sum(self.num_slices)) # Total number of slices from all scans

    # Phase encode random mask generator
    def _get_mask(self, acs_lines=30, total_lines=384, R=1, pattern='random'):
        # Overall sampling budget
        num_sampled_lines = np.floor(total_lines / R)
        # Get locations of ACS lines
        # !!! Assumes k-space is even sized and centered, true for fastMRI
        center_line_idx = np.arange((total_lines - acs_lines) // 2,
                             (total_lines + acs_lines) // 2)
        
        # Find remaining candidates
        outer_line_idx = np.setdiff1d(np.arange(total_lines), center_line_idx)
        if pattern == 'random':
            # Sample remaining lines from outside the ACS at random
            random_line_idx = np.random.choice(outer_line_idx,
                       size=int(num_sampled_lines - acs_lines), replace=False)
        elif pattern == 'equispaced':
            # Sample equispaced lines
            # !!! Only supports integer for now
            random_line_idx = outer_line_idx[::int(R)]
        else:
            raise NotImplementedError('Mask pattern not implemented')

        # Create a mask and place ones at the right locations
        mask = np.zeros((total_lines))
        mask[center_line_idx] = 1.
        mask[random_line_idx] = 1.

        return mask

    # Cropping utility - works with numpy / tensors
    def _crop(self, x, wout, hout):
        w, h = x.shape[-2:]
        x1 = int(np.ceil((w - wout) / 2.))
        y1 = int(np.ceil((h - hout) / 2.))

        return x[..., x1:x1+wout, y1:y1+hout]

    def __getitem__(self, idx):
        # Convert to numerical
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get scan and slice index
        # First scan for which index is in the valid cumulative range
        scan_idx = int(np.where((self.slice_mapper - idx) >= 0)[0][0])
        # Offset from cumulative range
        slice_idx = int(idx) if scan_idx == 0 else \
            int(idx - self.slice_mapper[scan_idx] + self.num_slices[scan_idx] - 1)
        #slice_idx = 0
        # Load maps for specific scan and slice
        maps_file = os.path.join(self.maps_dir,
                                 os.path.basename(self.file_list[scan_idx]))
        with h5py.File(os.path.join(self.project_dir, maps_file), 'r') as data:
            # Get maps
            maps = np.asarray(data['s_maps'][slice_idx])

        # Load raw data for specific scan and slice
        raw_file = os.path.join(self.input_dir,
                                os.path.basename(self.file_list[scan_idx]))
        with h5py.File(os.path.join(self.project_dir, raw_file), 'r') as data:
            # Get maps
            gt_ksp = np.asarray(data['kspace'][slice_idx])
        # Crop extra lines and reduce FoV in phase-encode

        gt_ksp = sp.resize(gt_ksp, ( gt_ksp.shape[0], gt_ksp.shape[1], self.image_size[1]))

        # Reduce FoV by half in the readout direction
        gt_ksp = sp.ifft(gt_ksp, axes=(-2,))
        gt_ksp = sp.resize(gt_ksp, (gt_ksp.shape[0], self.image_size[0], gt_ksp.shape[2]))
        gt_ksp = sp.fft(gt_ksp, axes=(-2,)) # Back to k-space


        # Crop extra lines and reduce FoV in phase-encode
        maps = sp.fft(maps, axes=(-2, -1)) # These are now maps in k-space
        maps = sp.resize(maps, (maps.shape[0], maps.shape[1], self.image_size[1]))

        # Reduce FoV by half in the readout direction
        maps = sp.ifft(maps, axes=(-2,))
        maps = sp.resize(maps, (maps.shape[0], self.image_size[0], maps.shape[2]))
        maps = sp.fft(maps, axes=(-2,)) # Back to k-space
        maps = sp.ifft(maps, axes=(-2, -1)) # Finally convert back to image domain
        # find mvue image
        mvue = get_mvue(gt_ksp.reshape((1,) + gt_ksp.shape), maps.reshape((1,) + maps.shape))

        # !!! Removed ACS-based scaling if handled on the outside
        scale_factor = 1.

        # Scale data
        mvue   = mvue   / scale_factor
        gt_ksp = gt_ksp / scale_factor

        # Compute ACS size based on R factor and sample size
        total_lines = gt_ksp.shape[-1]
        if 1 < self.R <= 6:
            # Keep 8% of center samples
            acs_lines = np.floor(0.08 * total_lines).astype(int)
        else:
            # Keep 4% of center samples
            acs_lines = np.floor(0.04 * total_lines).astype(int)

        # Get a mask
        mask = self._get_mask(acs_lines, total_lines,
                              self.R, self.pattern)
        # Mask k-space
        if self.orientation == 'vertical':
            gt_ksp *= mask[None, None, :]
        elif self.orientation == 'horizontal':
            gt_ksp *= mask[None, :, None]
        else:
            raise NotImplementedError

        ## name for mvue file
        mvue_file = os.path.join(self.input_dir,
                                 os.path.basename(self.file_list[scan_idx]))
        # Output
        sample = {
                  'mvue': mvue,
                  'maps': maps,
                  'ground_truth': gt_ksp,
                  'mask': mask,
                  'scale_factor': scale_factor,
                  # Just for feedback
                  'scan_idx': scan_idx,
                  'slice_idx': slice_idx,
                  'mvue_file': mvue_file}
        return sample

class MVU_Estimator_Brain_monai(Dataset):
    def __init__(self, file_list, maps_dir, input_dir,
                 project_dir='./',
                 R=1,
                 image_size=(384,384),
                 acs_size=26,
                  pattern='random',
                 orientation='vertical'):
        # Attributes
        self.project_dir = project_dir
        self.file_list    = file_list
        self.maps_dir     = maps_dir
        self.input_dir      = input_dir
        self.image_size = image_size
        self.R            = R
        self.pattern      = pattern
        self.orientation  = orientation #only vertical orientation supported by this loader right now.
        self.reader = FastMRIReader()

        # Access meta-data of each scan to get number of slices
        self.num_slices = np.zeros((len(self.file_list,)), dtype=int)
        for idx, file in enumerate(self.file_list):
            input_file = os.path.join(self.input_dir, os.path.basename(file))
            monai_data = self.reader.read(os.path.join(self.project_dir, input_file))
            self.num_slices[idx] = int(np.array(monai_data['kspace']).shape[0])
        # Create cumulative index for mapping
        self.slice_mapper = np.cumsum(self.num_slices) - 1 # Counts from '0'

    def __len__(self):
        return int(np.sum(self.num_slices)) # Total number of slices from all scans

    def __getitem__(self, idx):
        # Convert to numerical
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get scan and slice index
        # First scan for which index is in the valid cumulative range
        scan_idx = int(np.where((self.slice_mapper - idx) >= 0)[0][0])
        # Offset from cumulative range
        slice_idx = int(idx) if scan_idx == 0 else \
            int(idx - self.slice_mapper[scan_idx] + self.num_slices[scan_idx] - 1)

        # Load maps for specific scan and slice
        maps_file = os.path.join(
                 self.maps_dir,
                 os.path.basename(self.file_list[scan_idx]),
                 )
        inputs_file = os.path.join(
                 self.input_dir,
                 os.path.basename(self.file_list[scan_idx]),
                 )
        
        monai_data_smap = self.reader.read(os.path.join(self.project_dir, maps_file))
        monai_data = self.reader.read(os.path.join(self.project_dir, inputs_file))

        maps = np.asarray(monai_data_smap['s_maps'][slice_idx])
        gt_ksp = np.asarray(monai_data['kspace'][slice_idx])
        gt_ksp = resize_center(gt_ksp, *( gt_ksp.shape[0], gt_ksp.shape[1], self.image_size[1]), inplace=True)
        # Reduce FoV by half in the readout direction
        gt_ksp = _ifft_centered_single_last_axis(torch.from_numpy(gt_ksp), axis_dim=-2).numpy()
        gt_ksp = resize_center( gt_ksp, *( gt_ksp.shape[0], self.image_size[0], gt_ksp.shape[2]), inplace=True)
        gt_ksp = _fft_centered_single_last_axis(torch.from_numpy(gt_ksp), axis_dim=-2).numpy()

        maps = torch.view_as_complex(fftn_centered(torch.view_as_real(torch.from_numpy(maps)), spatial_dims=2, is_complex=True)).numpy()
        maps = resize_center(maps, *( maps.shape[0], maps.shape[1], self.image_size[1]), inplace=True)
        
        maps = _ifft_centered_single_last_axis(torch.from_numpy(maps), axis_dim=-2).numpy()
        maps = resize_center(maps, *( maps.shape[0], self.image_size[0], maps.shape[2]), inplace=True)
        maps = _fft_centered_single_last_axis(torch.from_numpy(maps), axis_dim=-2).numpy()
        maps = torch.view_as_complex(ifftn_centered(torch.view_as_real(torch.from_numpy(maps)), spatial_dims=2, is_complex=True)).numpy()

        mvue = get_mvue_monai(
                torch.from_numpy(gt_ksp.reshape((1,) + gt_ksp.shape)),
                torch.from_numpy(maps.reshape((1,) + maps.shape)), 
                ).numpy()
        
        # !!! Removed ACS-based scaling if handled on the outside
        scale_factor = 1.

        # Scale data
        mvue   = mvue   / scale_factor
        gt_ksp = gt_ksp / scale_factor

        # Compute ACS size based on R factor and sample size
        total_lines = gt_ksp.shape[-1]
        if 1 < self.R <= 6:
            # Keep 8% of center samples
            center_fractions = 0.08
            acs_lines = np.floor(center_fractions * total_lines).astype(int)
        else:
            # Keep 4% of center samples
            center_fractions = 0.04
            acs_lines = np.floor(center_fractions * total_lines).astype(int)
        adjusted_R = self.R * (total_lines) / ((self.R-1) * acs_lines + total_lines)
            
        gt_ksp_torch = torch.from_numpy(gt_ksp)
        gt_ksp_torch = torch.view_as_real(gt_ksp_torch)

        if self.pattern == 'random':
            RandomKspace = RandomKspaceMask(center_fractions=[center_fractions],accelerations=[adjusted_R],spatial_dims=2,is_complex=True)
            masked_monai = RandomKspace(gt_ksp_torch)
            mask_monai = RandomKspace.mask
        if self.pattern == 'equispaced':
            EquispacedKspace = EquispacedKspaceMask(center_fractions=[center_fractions],accelerations=[adjusted_R],spatial_dims=2,is_complex=True)
            masked_monai = EquispacedKspace(gt_ksp_torch)
            mask_monai = EquispacedKspace.mask
        masked_gt = torch.view_as_complex(masked_monai[0])
        mask_monai=np.squeeze(np.asarray(mask_monai))

        mvue_file = os.path.join(self.input_dir,
                                 os.path.basename(self.file_list[scan_idx]))
        # Output
        sample = {
                  'mvue': mvue,
                  'maps': maps,
                  'ground_truth': masked_gt,
                  'mask': mask_monai,
                  'scale_factor': scale_factor,
                  'scan_idx': scan_idx,
                  'slice_idx': slice_idx,
                  'mvue_file': mvue_file}
        return sample
    
class MVU_Estimator_Stanford_Knees(Dataset):
    def __init__(self, file_list, maps_dir, input_dir,
                 project_dir='./',
                 R=1,
                 image_size=(320,320),
                 acs_size=26,
                 pattern='random',
                 orientation='vertical'):
        # Attributes
        self.project_dir = project_dir
        self.acs_size     = acs_size
        self.maps_dir     = maps_dir
        self.input_dir      = input_dir
        self.R = R
        self.image_size = image_size
        self.pattern      = pattern
        self.orientation  = orientation
        self.file_list = sorted(file_list)
        if len(self.file_list) == 0:
            raise IOError('No image files found in the specified path')

    def num_slices(self):
        num_slices = np.zeros((len(self.file_list,)), dtype=int)
        for idx, file in enumerate(self.file_list):
            with h5py.File(os.path.join(self.project_dir, file), 'r') as data:
                num_slices[idx] = np.array(data['kspace']).shape[0]
        return num_slices

    @property
    def slice_mapper(self):
        return np.cumsum(self.num_slices) - 1 # Counts from '0'

    def __len__(self):
        return int(np.sum(self.num_slices)) # Total number of slices from all scans

    def __getitem__(self, idx):
        # Convert to numerical
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get scan and slice index
        # First scan for which index is in the valid cumulative range
        scan_idx = int(np.where((self.slice_mapper - idx) >= 0)[0][0])
        # Offset from cumulative range
        slice_idx = int(idx) if scan_idx == 0 else \
            int(idx - self.slice_mapper[scan_idx] + self.num_slices[scan_idx] - 1)
        
        # Load specific slice from specific scan
        with h5py.File(os.path.join(self.project_dir, self.file_list[scan_idx]), 'r') as data:
            # Get maps, kspace, masks
            gt_ksp = np.asarray(data['kspace'])[slice_idx]
            gt_ksp = (gt_ksp-gt_ksp.mean())/gt_ksp.std()
            maps = np.asarray(data['maps'])[slice_idx]
            maps = maps.squeeze(-1).transpose(2,0,1)
            gt_ksp = gt_ksp.transpose(2,0,1)
            #maps = np.asarray(data['s_maps'])[slice_idx]
            #mask = np.asarray(data['masks'])[slice_idx]
        seed = 1111
        nky = gt_ksp.shape[1]    
        nkz = gt_ksp.shape[2]  
        acceleration = self.R
        self.calib_size = (20,20)
        
        mask = poisson(
            (nky, nkz),
            acceleration,
            calib=self.calib_size,
            dtype=np.float32,
            seed=seed,
        )
        
        # find mvue image
        mvue = get_mvue(gt_ksp.reshape((1,) + gt_ksp.shape), maps.reshape((1,) + maps.shape))

        # # Load MVUE slice from specific scan
        mvue_file = os.path.join(self.input_dir,
                                 os.path.basename(self.file_list[scan_idx]))

        # !!! Removed ACS-based scaling if handled on the outside
        scale_factor = 1.

        # Scale data
        mvue   = mvue / scale_factor
        gt_ksp = gt_ksp / scale_factor

        # apply mask
        gt_ksp *= mask[None, :, :]

        # Output
        sample = {
                  'mvue': mvue,
                  'maps': maps,
                  'ground_truth': gt_ksp,
                  'mask': mask,
                  'scale_factor': scale_factor,
                  # Just for feedback
                  'scan_idx': scan_idx,
                  'slice_idx': slice_idx,
                  'mvue_file': mvue_file}
        return sample

def poisson(
    img_shape,
    accel,
    K=30,
    calib=(0, 0),
    dtype=np.complex,
    crop_corner=True,
    return_density=False,
    seed=0,
):
    """Generate Poisson-disc sampling pattern

    Args:
        img_shape (tuple of ints): length-2 image shape.
        accel (float): Target acceleration factor. Greater than 1.
        K (float): maximum number of samples to reject.
        calib (tuple of ints): length-2 calibration shape.
        dtype (Dtype): data type.
        crop_corner (bool): Toggle whether to crop sampling corners.
        return_density (bool): Toggle whether to return sampling density.
        seed (int): Random seed.

    Returns:
        array: Poisson-disc sampling mask.

    References:
        Bridson, Robert. "Fast Poisson disk sampling in arbitrary dimensions."
        SIGGRAPH sketches. 2007.

    """
    y, x = np.mgrid[: img_shape[-2], : img_shape[-1]]
    x = np.maximum(abs(x - img_shape[-1] / 2) - calib[-1] / 2, 0)
    x /= x.max()
    y = np.maximum(abs(y - img_shape[-2] / 2) - calib[-2] / 2, 0)
    y /= y.max()
    r = np.sqrt(x ** 2 + y ** 2)

    slope_max = 40
    slope_min = 0
    if seed is not None:
        rand_state = np.random.get_state()
    else:
        seed = -1  # numba does not play nicely with None types
    while slope_min < slope_max:
        slope = (slope_max + slope_min) / 2.0
        R = 1.0 + r * slope
        mask = _poisson(img_shape[-1], img_shape[-2], K, R, calib, seed)
        if crop_corner:
            mask *= r < 1

        est_accel = img_shape[-1] * img_shape[-2] / np.sum(mask[:])

        if abs(est_accel - accel) < 0.1:
            break
        if est_accel < accel:
            slope_min = slope
        else:
            slope_max = slope

    if seed is not None and seed > 0:
        np.random.set_state(rand_state)
    mask = mask.reshape(img_shape).astype(dtype)
    if return_density:
        return mask, r
    else:
        return mask

#@nb.jit(nopython=True, cache=True)  # pragma: no cover
def _poisson(nx, ny, K, R, calib, seed=None):

    mask = np.zeros((ny, nx))
    f = ny / nx

    if seed is not None and seed > 0:
        np.random.seed(int(seed))

    pxs = np.empty(nx * ny, np.int32)
    pys = np.empty(nx * ny, np.int32)
    pxs[0] = np.random.randint(0, nx)
    pys[0] = np.random.randint(0, ny)
    m = 1
    while m > 0:

        i = np.random.randint(0, m)
        px = pxs[i]
        py = pys[i]
        rad = R[py, px]

        # Attempt to generate point
        done = False
        k = 0
        while not done and k < K:

            # Generate point randomly from R and 2R
            rd = rad * (np.random.random() * 3 + 1) ** 0.5
            t = 2 * np.pi * np.random.random()
            qx = px + rd * np.cos(t)
            qy = py + rd * f * np.sin(t)

            # Reject if outside grid or close to other points
            if qx >= 0 and qx < nx and qy >= 0 and qy < ny:

                startx = max(int(qx - rad), 0)
                endx = min(int(qx + rad + 1), nx)
                starty = max(int(qy - rad * f), 0)
                endy = min(int(qy + rad * f + 1), ny)

                done = True
                for x in range(startx, endx):
                    for y in range(starty, endy):
                        if mask[y, x] == 1 and (
                            ((qx - x) / R[y, x]) ** 2 + ((qy - y) / (R[y, x] * f)) ** 2 < 1
                        ):
                            done = False
                            break

                    if not done:
                        break

            k += 1

        # Add point if done else remove active
        if done:
            pxs[m] = qx
            pys[m] = qy
            mask[int(qy), int(qx)] = 1
            m += 1
        else:
            pxs[i] = pxs[m - 1]
            pys[i] = pys[m - 1]
            m -= 1

    # Add calibration region
    mask[
        int(ny / 2 - calib[-2] / 2) : int(ny / 2 + calib[-2] / 2),
        int(nx / 2 - calib[-1] / 2) : int(nx / 2 + calib[-1] / 2),
    ] = 1

    return mask
