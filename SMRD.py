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
from cg import ConjugateGradient
import csv
import pylab 
import scipy.stats as stats

from ncsnv2.models import get_sigmas
from ncsnv2.models.ema import EMAHelper
from ncsnv2.models.ncsnv2 import NCSNv2Deepest
import argparse

from matplotlib.pyplot import *

class SMRDOptimizer(torch.nn.Module):
    def __init__(self, config, logger, project_dir='./'):
        super().__init__()

        self.config = config
        
        #Load configurations used in the CSGM-Langevin paper for the score function
        self.SMRD_config = self._dict2namespace(self.config['langevin_config'])
        self.device = config['device']
        self.SMRD_config.device = config['device']

        self.project_dir = project_dir
        self.score = NCSNv2Deepest(self.SMRD_config).to(self.device)
        self.sigmas_torch = get_sigmas(self.SMRD_config)

        self.sigmas = self.sigmas_torch.cpu().numpy()

        states = torch.load(os.path.join(project_dir, config['gen_ckpt']))

        self.score = torch.nn.DataParallel(self.score)

        self.score.load_state_dict(states[0], strict=True)
        if self.SMRD_config.model.ema:
            ema_helper = EMAHelper(mu=self.SMRD_config.model.ema_rate)
            ema_helper.register(self.score)
            ema_helper.load_state_dict(states[-1])
            ema_helper.ema(self.score)
        del states

        self.index = 0
        self.logger = logger

    def _dict2namespace(self,SMRD_config):
        namespace = argparse.Namespace()
        for key, value in SMRD_config.items():
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
        self.logger.info(f"Running {self.SMRD_config.model.num_classes} steps of SMRD.")

        pbar = tqdm(range(self.SMRD_config.model.num_classes), disable=(self.config['device'] != 0))
        pbar_labels = ['class', 'step_size', 'error', 'mean', 'max']
        step_lr = self.SMRD_config.sampling.step_lr
        forward_operator = lambda x: MulticoilForwardMRI(self.config['orientation'])(torch.complex(x[:, 0], x[:, 1]), maps, batch_mri_mask)
        inverse_operator = lambda x: torch.view_as_real(torch.sum(self._ifft(x) * torch.conj(maps), axis=1) ).permute(0,3,1,2)
        
        samples = torch.rand(y[0].shape[0], self.SMRD_config.data.channels,
                                 self.config['image_size'][0],
                                 self.config['image_size'][1], device=self.device)
        
        n_steps_each = 3
        window_size = self.config['window_size'] * n_steps_each 
        gt_losses = []
        lambdas = []
        SURES = []
        
        lamda_init = self.config.lambda_init
        lamda_end = self.config.lambda_end
        if self.config.lambda_func == 'cnst':
            lamdas = lamda_init * torch.ones(len(self.sigmas), device=samples.device)
        elif self.config.lambda_func == 'linear':
            lamdas = torch.linspace(lamda_init, lamda_end, len(self.sigmas))
        elif self.config.lambda_func == 'learnable':
            with torch.enable_grad():
                init = torch.tensor(lamda_init, dtype=torch.float32)
                lamda = torch.nn.Parameter(init, requires_grad=True)
            lambda_lr = self.config['lambda_lr']
            optimizer = torch.optim.Adam([lamda], lr=lambda_lr)            
        
        with torch.no_grad():
            
            for c in pbar:
                if c <= self.config['start_iter']:
                    continue
                if c <= 1800:
                    n_steps_each = 3
                else:
                    n_steps_each = self.SMRD_config.sampling.n_steps_each
                sigma = self.sigmas[c]
                labels = torch.ones(samples.shape[0], device=samples.device) * c
                labels = labels.long()
                step_size = step_lr * (sigma / self.sigmas[-1]) ** 2
                
                for s in range(n_steps_each):
                    with torch.enable_grad():
                        if self.config.lambda_func == 'learnable':
                            optimizer.zero_grad()
                        samples = samples.to('cuda')
                        if self.config.lambda_func == 'learnable':
                            samples = samples.requires_grad_(True)
                        noise = torch.randn_like(samples) * np.sqrt(step_size * 2)
                        # get score from model
                        with torch.no_grad():
                            p_grad = self.score(samples, labels)
                        estimated_mvue = estimated_mvue.clone().to('cuda')

                        if self.config.lambda_func == 'learnable':
                            samples = samples.requires_grad_(True)
                        
                        torch.autograd.set_detect_anomaly(True)
                        if self.config.lambda_func == 'learnable':
                            pass
                        else:
                            lamda = lamdas[c]
                            
                        if lamda.detach().cpu().numpy() > 0:
                            lamda_applied = lamda.clone()
                        else:
                            #If learning results in a negative lamda, apply initial lamda
                            lamda_applied = torch.tensor(lamda_init, dtype=torch.float32)
                        
                        model_normal = lambda m: torch.view_as_complex((unnormalize(inverse_operator(forward_operator(normalize(torch.view_as_real(m).permute(0,-1,1,2),estimated_mvue))),estimated_mvue) + lamda_applied.clone() * torch.view_as_real(m).permute(0,-1,1,2)).permute(0,2,3,1))
                        cg_solve = ConjugateGradient(model_normal, self.config['num_cg_iter'])
                        n = samples.size(0)
                        meas = forward_operator(samples) #H x hat t, ref = y
                                                
                        zf = inverse_operator(ref)
                        zf = unnormalize(zf, estimated_mvue)
                        
                        samples_input = samples
                        
                        samples = samples.to(self.device)
                        
                        ### REVERSE DIFFUSION
                        samples = samples + step_size * (p_grad) + noise 
                        
                        cg_in = torch.view_as_complex((zf + lamda_applied.clone() * samples).permute(0,2,3,1))

                        samples = cg_solve(torch.view_as_complex(zf.permute(0,2,3,1)),cg_in)

                        samples = torch.view_as_real(samples).permute(0,-1,1,2).type(torch.FloatTensor)
                        
                        if self.config.lambda_func == 'learnable':
                            samples = samples.requires_grad_(True)
                        samples = samples.to(self.device)

                        # compute metrics
                        metrics = [c, step_size, (meas-ref).norm(), (p_grad).abs().mean(), (p_grad).abs().max()]
                        update_pbar_desc(pbar, metrics, pbar_labels)
                        
                        # create perturbed input for monte-carlo SURE
                        perturb_noise = torch.randn_like(samples)
                        eps = torch.abs(zf.max())/1000
                        samples_perturbed = samples_input + eps * perturb_noise
                        with torch.no_grad():
                            perturbed_p_grad = self.score(samples_perturbed, labels)
                            
                        ### START PERTURBED PATH
                        samples_perturbed = samples_perturbed + step_size * (perturbed_p_grad) + noise
                        
                        cg_in = torch.view_as_complex((zf + lamda_applied.clone() * samples_perturbed).permute(0,2,3,1))

                        samples_perturbed = cg_solve(torch.view_as_complex(zf.permute(0,2,3,1)),cg_in)

                        samples_perturbed = torch.view_as_real(samples_perturbed).permute(0,-1,1,2).type(torch.FloatTensor)
                        if self.config.lambda_func == 'learnable':
                            samples_perturbed = samples_perturbed.requires_grad_(True)
                        samples_perturbed = samples_perturbed.to(self.device)
                        ### END PERTURBED PATH
                        
                        if self.config.lambda_func == 'learnable':
                            samples = samples.requires_grad_(True)
                        samples = samples.to(self.device)
                        
                        samples_loss = torch.view_as_complex(samples.permute(0,2,3,1))
                        
                        samples_loss = samples_loss.to(self.device)
                        l2_loss_fn = nn.MSELoss(reduction='mean')
                        l2_loss_fn_gt = nn.MSELoss(reduction='mean')
                        
                        if self.config.lambda_func == 'learnable':
                            zf = zf.requires_grad_(True)
                        zf = zf.type(torch.float32)
                        zf_loss = torch.view_as_complex(zf.permute(0,2,3,1))
                        l2_loss = l2_loss_fn(torch.abs(zf_loss-samples_loss),torch.zeros_like(torch.abs(samples_loss)))
                        
                        gt_l2_loss = l2_loss_fn_gt(torch.abs(mvue.squeeze(1)-samples_loss),torch.zeros_like(torch.abs(samples_loss)))
                        
                        div_term = (samples_perturbed - samples)
                        divergence = torch.sum(1/eps * torch.matmul(perturb_noise.permute(0,1,3,2),div_term))
                        
                        SURE = l2_loss * divergence/(torch.numel(samples))
                        
                        SURES.append(SURE.detach().cpu().numpy())
                        gt_losses.append(gt_l2_loss.detach().cpu().numpy())
                        lambdas.append(lamda.clone().detach().cpu().numpy())

                        init_lambda_update = self.config['init_lambda_update']
                        last_lambda_update = self.config['last_lambda_update']
                        if c>init_lambda_update:
                            if c<last_lambda_update:
                                if self.config.lambda_func == 'learnable':
                                    loss = SURE
                                    loss.backward(retain_graph=True)
                                    optimizer.step()
                                
                        #if c%5==0:
                        if self.config.lambda_func == 'learnable':
                            samples = samples.detach()
                            samples_perturbed = samples_perturbed.detach()
                            lamda = lamda.detach()
                            zf = zf.detach()
                            loss = loss.detach()
                    
                    if self.config.early_stop == 'stop':
                        # EARLY STOPPING USING SURE LOSS
                        # check the self-validation loss for early stopping
                        if len(SURES) > 3 * window_size:
                            if c > 3*window_size:
                                if np.mean(SURES[-window_size:]) > np.mean(SURES[-2*window_size:-window_size]): 
                                    print('\nAutomatic early stopping activated.')
                                    print(c)
                                    np.save('SURES_slice_'+str(self.slice_id)+str(c),np.asarray(SURES))
                                    np.save('gt_losses_slice_'+str(self.slice_id)+str(c),np.asarray(gt_losses))
                                    np.save('lambdas_slice_'+str(self.slice_id)+str(c),np.asarray(lambdas))
                                    outputs = normalize(samples, estimated_mvue)
                                    outputs = outputs.permute(0,2,3,1)
                                    outputs = outputs.contiguous()
                                    outputs = torch.view_as_complex(outputs)
                                    norm_output = torch.abs(outputs).cpu().numpy()
                                    gt = torch.abs(mvue).squeeze(1).cpu().numpy()
                                    PSNR = psnr(norm_output,gt)
                                    SSIM = ssim(norm_output,gt)[0]

                                    img_gen = normalize(samples, estimated_mvue)
                                    to_display = torch.view_as_complex(img_gen.permute(0, 2, 3, 1).reshape(-1, self.config['image_size'][0], self.config['image_size'][1], 2).contiguous()).abs()
                                    if self.config['anatomy'] == 'brain':
                                        # flip vertically
                                        to_display = to_display.flip(-2)
                                    elif self.config['anatomy'] == 'knees':
                                        # flip vertically and horizontally
                                        to_display = to_display.flip(-2)
                                        #to_display = to_display.flip(-1)
                                    elif self.config['anatomy'] == 'stanford_knees':
                                        # do nothing
                                        pass
                                    elif self.config['anatomy'] == 'abdomen':
                                        # flip horizontally
                                        to_display = to_display.flip(-1)
                                    else:
                                        pass
                                    to_display = scale(to_display)
                                    file_name = f'_sure_ES_output_recon_slice'+str(self.slice_id)+'_R='+str(self.config["R"])+'_'+str(c)+'.jpg'
                                    save_images(to_display[0:1], file_name, normalize=True)
                                    return normalize(samples, estimated_mvue)
                    else:
                        pass

                    if np.isnan((meas - ref).norm().cpu().numpy()):
                        return normalize(samples, estimated_mvue)
                if self.config['save_images']:
                    if (c+1) % self.config['save_iter'] ==0 :
                        estimated_mvue = estimated_mvue.cpu()
                        img_gen = normalize(samples, estimated_mvue)
                        
                        outputs = normalize(samples, estimated_mvue)
                        outputs = outputs.permute(0,2,3,1)
                        outputs = outputs.contiguous()
                        outputs = torch.view_as_complex(outputs)
                        norm_output = torch.abs(outputs).cpu().numpy()
                        gt = torch.abs(mvue).squeeze(1).cpu().numpy()
                        PSNR = psnr(norm_output,gt)
                        SSIM = ssim(norm_output,gt)[0]
                        result_file_path = 'results'+self.config['exp_name']+'_iterations.csv'
                        try:
                            with open(result_file_path, mode='a') as csv_file:
                                reader = csv.reader(csv_file)
                                items = []
                                for item in reader:
                                    if item[0] == 'Iter':
                                        is_header = True
                                    else:
                                        is_header = False
                                    break
                        except:
                            is_header = False
                        header = ['Iter','Noise Level','PSNR','SSIM']
                        data = [[c,self.config['noise_std'],PSNR,SSIM],]
                        with open(result_file_path, 'a', encoding='UTF8', newline='') as f:
                            writer = csv.writer(f)
                            # write the header
                            if not is_header:
                                writer.writerow(header)
                            # write multiple rows
                            writer.writerows(data)
                            f.close()
                        
                        to_display = torch.view_as_complex(img_gen.permute(0, 2, 3, 1).reshape(-1, self.config['image_size'][0], self.config['image_size'][1], 2).contiguous()).abs()
                        if self.config['anatomy'] == 'brain':
                            # flip vertically
                            to_display = to_display.flip(-2)
                        elif self.config['anatomy'] == 'stanford_knees':
                            # do nothing
                            pass
                        else:
                            pass
                        to_display = scale(to_display)
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
        samples = samples.detach()
        np.save('SURES_slice_'+str(self.slice_id),np.asarray(SURES))
        np.save('gt_losses_slice_'+str(self.slice_id),np.asarray(gt_losses))
        np.save('lambdas_slice_'+str(self.slice_id),np.asarray(lambdas))
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
        '''
        dataset = MVU_Estimator_Brain(files,
                                input_dir=config['input_dir'],
                                maps_dir=config['maps_dir'],
                                project_dir=project_dir,
                                image_size = config['image_size'],
                                R=config['R'],
                                pattern=config['pattern'],
                                orientation=config['orientation'])
        
        '''
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

    SMRD_optimizer = SMRDOptimizer(config, logger, project_dir)
    if config['multiprocessing']:
        SMRD_optimizer = DDP(SMRD_optimizer, device_ids=[rank]).module
    SMRD_optimizer.to(rank)

    for index, sample in enumerate(tqdm(loader)):
        '''
                    ref: one complex image per coil
                    mvue: one complex image reconstructed using the coil images and the sensitivity maps
                    maps: sensitivity maps for each one of the coils
                    mask: binary valued kspace mask
        '''
        
        ref, mvue, maps, mask = sample['ground_truth'], sample['mvue'], sample['maps'], sample['mask']

        # move everything to cuda
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

            exp_name = 'noise_' + str(config['noise_std']) + '_lamdainit_' + str(config['lambda_init']) + '_lamdaend_' + str(config['lambda_end']) + '_' + str(config['lambda_func']) + '_' + sample['mvue_file'][batch_idx].split('/')[-1] + '|SMRD|' + f'slide_idx_{sample["slice_idx"][batch_idx].item()}'
            exp_names.append(exp_name)
            print(exp_name)
            if config['save_images']:
                file_name = f'{exp_name}_R={config["R"]}_estimated_mvue.jpg'
                estimated_mvuevis = torch.abs(estimated_mvue)
                estimated_mvuevis = scale(estimated_mvuevis)
                save_images(estimated_mvuevis[batch_idx:batch_idx+1].abs().flip(-2), file_name, normalize=True)
                mvuevis = torch.abs(mvue)
                mvuevis = scale(mvuevis)
                file_name = f'{exp_name}_input.jpg'
                save_images(mvuevis[batch_idx:batch_idx+1].abs().flip(-2), file_name, normalize=True)

        SMRD_optimizer.config['exp_names'] = exp_names
        SMRD_optimizer.slice_id = index
        if config['repeat'] > 1:
            repeat = config['repeat']
            ref, mvue, maps, mask, estimated_mvue = ref.repeat(repeat,1,1,1), mvue.repeat(repeat,1,1,1), maps.repeat(repeat,1,1,1), mask.repeat(repeat,1), estimated_mvue.repeat(repeat,1,1,1)
        outputs = SMRD_optimizer.sample((ref, mvue, maps, mask))
        outputs[0] = outputs[0]['mvue'].permute(0,2,3,1)
        outputs[0] = torch.view_as_complex(outputs[0])
        norm_output = torch.abs(outputs[0]).cpu().numpy()
        gt = torch.abs(sample['mvue']).squeeze(1).cpu().numpy()
        PSNR = psnr(norm_output,gt)
        SSIM = ssim(norm_output,gt)[0]
        logger.info(f"PSNR is {PSNR}")
        img = scale(torch.from_numpy(norm_output))
        file_name = f'{exp_name}_final_recon.jpg'
        batch_idx = 0
        save_images(img[batch_idx:batch_idx+1].abs().flip(-2), file_name, normalize=True)
        for i, exp_name in enumerate(exp_names):
            if config['repeat'] == 1:
                torch.save(outputs[i], f'{exp_name}_R={config["R"]}_outputs.pt')
            else:
                for j in range(config['repeat']):
                    torch.save(outputs[j], f'{exp_name}_R={config["R"]}_sample={j}_outputs.pt')
        result_file_path = 'results'+config['exp_name']+'final_recon.csv'
        header = ['Slice id','Noise Level','PSNR','SSIM']
        data = [[index,config['noise_std'],PSNR,SSIM],]
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
