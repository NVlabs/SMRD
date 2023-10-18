"""
------------------------------------------------------------------------------
Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

This work is made available under the Nvidia Source Code License.
To view a copy of this license, visit
https://github.com/batuozt/SMRD/blob/master/LICENSE.md

Written by Batu Ozturkler
------------------------------------------------------------------------------
"""

import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
from torch import nn
import hydra
import os
import logging
import random
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from dataloaders import MVU_Estimator_Brain, MVU_Estimator_Brain_monai, MVU_Estimator_Stanford_Knees
import multiprocessing
from torch.utils.data.distributed import DistributedSampler
from utils import *
import csv
import scipy.stats as stats

from ncsnv2.models import get_sigmas
from ncsnv2.models.ema import EMAHelper
from ncsnv2.models.ncsnv2 import NCSNv2Deepest
import argparse

class Diffusion:
    def __init__(self, beta_schedule="linear", beta_start=1e-4, beta_end=2e-2, num_diffusion_timesteps=1000, given_betas=None):
        if given_betas is None:
            if beta_schedule == "quad":
                betas = (
                    np.linspace(
                        beta_start**0.5,
                        beta_end**0.5,
                        num_diffusion_timesteps,
                        dtype=np.float64,
                    )
                    ** 2
                )
            elif beta_schedule == "linear":
                betas = np.linspace(
                    beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
                )
            elif beta_schedule == "const":
                betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
            elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
                betas = 1.0 / np.linspace(
                    num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
                )
            else:
                raise NotImplementedError(beta_schedule)
            assert betas.shape == (num_diffusion_timesteps,)
            betas = torch.from_numpy(betas)
        else:
            betas = given_betas
        self.betas = torch.cat([torch.zeros(1).to(betas.device), betas], dim=0).cuda().float()
        self.alphas = (1 - self.betas).cumprod(dim=0).cuda().float()
        self.num_diffusion_timesteps = num_diffusion_timesteps
    
    def alpha(self, t):
        return self.alphas.index_select(0, t+1)

class REDdiff_MRIOptimizer(torch.nn.Module):
    def __init__(self, config, logger, project_dir='./'):
        super().__init__()

        self.config = config

        self.REDdiff_MRI_config = self._dict2namespace(self.config['langevin_config'])
        self.device = config['device']
        self.REDdiff_MRI_config.device = config['device']

        self.project_dir = project_dir
        self.score = NCSNv2Deepest(self.REDdiff_MRI_config).to(self.device)
        self.sigmas_torch = get_sigmas(self.REDdiff_MRI_config)

        self.sigmas = self.sigmas_torch.cpu().numpy()

        states = torch.load(os.path.join(project_dir, config['gen_ckpt']))#, map_location=self.device)

        self.score = torch.nn.DataParallel(self.score)

        self.score.load_state_dict(states[0], strict=True)
        if self.REDdiff_MRI_config.model.ema:
            ema_helper = EMAHelper(mu=self.REDdiff_MRI_config.model.ema_rate)
            ema_helper.register(self.score)
            ema_helper.load_state_dict(states[-1])
            ema_helper.ema(self.score)
        del states

        self.index = 0
        self.logger = logger

    def _dict2namespace(self,REDdiff_MRI_config):
        namespace = argparse.Namespace()
        for key, value in REDdiff_MRI_config.items():
            if isinstance(value, dict):
                new_value = dict2namespace(value)
            else:
                new_value = value
            setattr(namespace, key, new_value)
        return namespace

    def _initialize(self):
        self.gen_outs = []

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

    def _sample(self, y):
        ref, mvue, maps, batch_mri_mask = y
        estimated_mvue = torch.tensor(
            get_mvue(ref.cpu().numpy(),
            maps.cpu().numpy()), device=ref.device)
        self.logger.info(f"Running {self.REDdiff_MRI_config.model.num_classes} steps of REDdiff_MRI.")

        pbar = tqdm(range(self.REDdiff_MRI_config.model.num_classes), disable=(self.config['device'] != 0))
        pbar_labels = ['class', 'step_size', 'error', 'mean', 'max']
        step_lr = self.REDdiff_MRI_config.sampling.step_lr
        forward_operator = lambda x: MulticoilForwardMRI(self.config['orientation'])(torch.complex(x[:, 0], x[:, 1]), maps, batch_mri_mask)
        inverse_operator = lambda x: torch.view_as_real(torch.sum(self._ifft(x) * torch.conj(maps), axis=1) ).permute(0,3,1,2)
        
        samples = torch.rand(y[0].shape[0], self.REDdiff_MRI_config.data.channels,
                                 self.config['image_size'][0],
                                 self.config['image_size'][1], device=self.device)
        
        diffusion = Diffusion(**self.config.diffusion)
            
        n = samples.size(0)
        zf = unnormalize(inverse_operator(ref), estimated_mvue)
        dtype = torch.FloatTensor
        mu = torch.autograd.Variable(zf, requires_grad=True)   #, device=device).type(dtype)
        optimizer = torch.optim.Adam([mu], lr=self.config.lr, betas=(0.9, 0.99), weight_decay=0.0)   
            
        for c in pbar:
            if c <= self.config['start_iter']:
                continue
            sigma = self.sigmas[c]
            labels = torch.ones(samples.shape[0], device=samples.device) * c
            labels = labels.long()
            step_size = step_lr * (sigma / self.sigmas[-1]) ** 2

            n_steps_each=1
            for s in range(n_steps_each):
                with torch.enable_grad():
                    samples = samples.to('cuda')
                    noise = torch.randn_like(samples) * np.sqrt(step_size * 2)
                    
                    estimated_mvue = estimated_mvue.clone().to('cuda')

                    torch.autograd.set_detect_anomaly(True)
                    
                    n = samples.size(0) #batch size i guess
                                        
                    mu = mu.to(self.device)
                    mu = mu.requires_grad_(True)
                    ti = 2311-c
                    
                    n = mu.size(0)
                    t = torch.ones(n).to(self.device).long() * ti
                    alpha_t = diffusion.alpha(t).view(-1, 1, 1, 1)
                    sigma_t =  (1 - alpha_t).sqrt() 
                    
                    noise_x0 = torch.randn_like(mu)
                    noise_xt = torch.randn_like(mu)

                    x0_pred = mu 
                    x_t = alpha_t.sqrt() * x0_pred + sigma_t * noise_xt
                    
                    x_t = x_t.to(self.device)

                    x_t = x_t.type(torch.FloatTensor)
                    with torch.no_grad():
                        p_grad = self.score(x_t, labels)
                    p_grad = p_grad.requires_grad_(True)
                    et = -sigma_t * p_grad

                    meas = forward_operator(normalize(mu, estimated_mvue)) #H x hat t, ref = y
                    
                    ref_chans=torch.view_as_real(ref)
                    meas_chans=torch.view_as_real(meas)
                    meas_chans = meas_chans.requires_grad_(True)
                    e_obs = ref_chans - meas_chans

                    scale_loss = 1/((ref_chans**2).mean()/2)
                    loss_obs = scale_loss*(e_obs**2).mean()/2
                    
                    noise_xt = noise_xt.to(self.device)
                    x_t = x_t.to(self.device)
                    
                    loss_noise = torch.mul((et - noise_xt).detach(), x0_pred).mean()
                    
                    snr_inv = (1-alpha_t[0]).sqrt()/alpha_t[0].sqrt()
                    
                    grad_term_weight = self.config.grad_term_weight
                    w_t = grad_term_weight*snr_inv
                    v_t = 1.0

                    loss = w_t*loss_noise + v_t*loss_obs

                    optimizer.zero_grad()  #initialize
                    loss.backward()
                    optimizer.step()
                    samples = mu

                    # compute metrics
                    metrics = [c, step_size, (meas-ref).norm()/len(meas), (p_grad).abs().mean(), (p_grad).abs().max()]
                    update_pbar_desc(pbar, metrics, pbar_labels)

                    #RED-Diff
                    
                    if np.isnan((meas - ref).norm().cpu().detach().numpy()):
                        return normalize(samples, estimated_mvue)
                if self.config['save_images']:
                    if (c+1) % self.config['save_iter'] ==0 :
                        estimated_mvue = estimated_mvue.cpu()
                        img_gen = normalize(samples, estimated_mvue)
                        to_display = torch.view_as_complex(img_gen.permute(0, 2, 3, 1).reshape(-1, self.config['image_size'][0], self.config['image_size'][1], 2).contiguous()).abs()
                        if self.config['anatomy'] == 'brain':
                            # flip vertically
                            to_display = to_display.flip(-2)
                        elif self.config['anatomy'] == 'stanford_knees':
                            # do nothing
                            pass
                        else:
                            pass
                        for i, exp_name in enumerate(self.config['exp_names']):
                            if self.config['repeat'] == 1:
                                file_name = f'{exp_name}_R={self.config["R"]}_{c}.jpg'
                                save_images(to_display[i:i+1], file_name, normalize=True)
                            else:
                                for j in range(self.config['repeat']):
                                    file_name = f'{exp_name}_R={self.config["R"]}_sample={j}_{c}.jpg'
                                    save_images(to_display[j:j+1], file_name, normalize=True)

                # if c>=0:
                #     break
        return normalize(samples, estimated_mvue)

    def sample(self, y):
        self._initialize()
        mvue = self._sample(y)
        outputs = []
        for i in range(y[0].shape[0]):
            outputs_ = {
                'mvue': mvue[i:i+1],
            }
            outputs.append(outputs_)
        return outputs

def mp_run(rank, config, project_dir, working_dir, files):
    if config['multiprocessing']:
        mp_setup(rank, config['world_size'])
    logger = multiprocessing.log_to_stderr()
    logger.setLevel(logging.INFO)
    logger = MpLogger(logger, rank)

    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    logger.info(f'Logging to {working_dir}')

    project_name = config['anatomy']
    pretty(config)

    config['device'] = rank
    # load appropriate dataloader
    if config['anatomy'] == 'stanford_knees':
        dataset = MVU_Estimator_Stanford_Knees(files,
                            input_dir=config['input_dir'],
                            maps_dir=config['maps_dir'],
                            project_dir=project_dir,
                            image_size = config['image_size'],
                            R=config['R'],
                            pattern=config['pattern'],
                            orientation=config['orientation'])
    elif config['anatomy'] == 'brain':
        dataset = MVU_Estimator_Brain_monai(files,
                                input_dir=config['input_dir'],
                                maps_dir=config['maps_dir'],
                                project_dir=project_dir,
                                image_size = config['image_size'],
                                R=config['R'],
                                pattern=config['pattern'],
                                orientation=config['orientation'])
    else:
        raise NotImplementedError('anatomy not implemented, please write dataloader to process kspace appropriately')

    sampler = DistributedSampler(dataset, rank=rank, shuffle=True) if config['multiprocessing'] else None
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])

    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=config['batch_size'],
                                         sampler=sampler,
                                         #shuffle=True if sampler is None else False)
                                         shuffle=False)

    REDdiff_MRI_optimizer = REDdiff_MRIOptimizer(config, logger, project_dir)
    if config['multiprocessing']:
        REDdiff_MRI_optimizer = DDP(REDdiff_MRI_optimizer, device_ids=[rank]).module
    REDdiff_MRI_optimizer.to(rank)

    for index, sample in enumerate(tqdm(loader)):
        '''
                    ref: one complex image per coil
                    mvue: one complex image reconstructed using the coil images and the sensitivity maps
                    maps: sensitivity maps for each one of the coils
                    mask: binary valued kspace mask
        '''
        
        ref, mvue, maps, mask = sample['ground_truth'], sample['mvue'], sample['maps'], sample['mask']

        ref = ref.to(rank).type(torch.complex128)
        mask = mask.to(rank)
        noise_std = config['noise_std']

        if config['batch_size'] == 1:
            noise = mask[None, :, :] * torch.view_as_complex(torch.randn(ref.shape+(2,)).to(rank)) * noise_std * torch.abs(ref).max()
        elif config['batch_size'] > 1:
            noise = mask[:,None, None,:] * torch.view_as_complex(torch.randn(ref.shape+(2,)).to(rank)) * noise_std * torch.abs(ref).max()

        ref = ref + noise.to(rank)
        mvue = mvue.to(rank)
        maps = maps.to(rank)
        
        estimated_mvue = torch.tensor(
            get_mvue(ref.cpu().numpy(),
            maps.cpu().numpy()), device=ref.device)

        exp_names = []
        for batch_idx in range(config['batch_size']):

            exp_name = 'noise_' + str(config['noise_std']) + '_lamdainit_' + str(config['lambda_init']) + '_lamdaend_' + str(config['lambda_end']) + '_' + str(config['lambda_func']) + '_' + sample['mvue_file'][batch_idx].split('/')[-1] + '|REDdiff_MRI|' + f'slide_idx_{sample["slice_idx"][batch_idx].item()}'
            exp_names.append(exp_name)
            print(exp_name)
            if config['save_images']:
                file_name = f'{exp_name}_R={config["R"]}_estimated_mvue.jpg'
                save_images(estimated_mvue[batch_idx:batch_idx+1].abs().flip(-2), file_name, normalize=True)

                file_name = f'{exp_name}_input.jpg'
                save_images(mvue[batch_idx:batch_idx+1].abs().flip(-2), file_name, normalize=True)

        REDdiff_MRI_optimizer.config['exp_names'] = exp_names
        REDdiff_MRI_optimizer.slice_id = index
        if config['repeat'] > 1:
            repeat = config['repeat']
            ref, mvue, maps, mask, estimated_mvue = ref.repeat(repeat,1,1,1), mvue.repeat(repeat,1,1,1), maps.repeat(repeat,1,1,1), mask.repeat(repeat,1), estimated_mvue.repeat(repeat,1,1,1)
        outputs = REDdiff_MRI_optimizer.sample((ref, mvue, maps, mask))
        outputs[0] = outputs[0]['mvue'].permute(0,2,3,1)
        outputs[0] = torch.view_as_complex(outputs[0])
        norm_output = torch.abs(outputs[0]).detach().cpu().numpy()
        gt = torch.abs(sample['mvue']).squeeze(1).cpu().numpy()
        PSNR = psnr(norm_output,gt)
        SSIM = ssim(norm_output,gt)[0]
        img = scale(torch.from_numpy(norm_output))
        file_name = f'{exp_name}_final_recon.jpg'
        batch_idx = 0
        save_images(img[batch_idx:batch_idx+1].abs().flip(-2), file_name, normalize=True)
        
        logger.info(f"PSNR is {PSNR}")
        for i, exp_name in enumerate(exp_names):
            if config['repeat'] == 1:
                torch.save(outputs[i], f'{exp_name}_R={config["R"]}_outputs.pt')
            else:
                for j in range(config['repeat']):
                    torch.save(outputs[j], f'{exp_name}_R={config["R"]}_sample={j}_outputs.pt')
        result_file_path = 'results_'+config['exp_name']+'_main_reddiff.csv'
        header = ['Lamda','Slice id','Noise Level','PSNR','SSIM']
        data = [[config['lambda_end'],index,config['noise_std'],PSNR,SSIM],]
        with open(result_file_path, 'a', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            # write multiple rows
            writer.writerows(data)
            f.close()

        # todo: delete after testing
        if index >= 0:
            break

    if config['multiprocessing']:
        mp_cleanup()

@hydra.main(config_path='configs')
def main(config):
    """ setup """
    
    working_dir = os.getcwd()
    project_dir = hydra.utils.get_original_cwd()

    folder_path = os.path.join(project_dir, config['input_dir'])
    files = get_all_files(folder_path, pattern='*.h5')

    if not config['multiprocessing']:
        mp_run(0, config, project_dir, working_dir, files)
    else:
        mp.spawn(mp_run,
                args=(config, project_dir, working_dir, files),
                nprocs=config['world_size'],
                join=True)

if __name__ == '__main__':
    main()
