import pandas as pd
import numpy as np
import torch.nn as nn
import torch

from params import data_pars as dps
from utils import trim

sqrt2pi = torch.sqrt(torch.tensor(torch.pi * 2))
EPS = 1e-10
sigma_growth = np.sqrt(np.linspace(10., 1., 24))

def normal_loglikelihood(x, mu, sigma):
    sig = torch.tensor(sigma).to(x.device)
    return torch.tensor(-0.5) * torch.square((x - mu)/(sig+EPS)) - torch.log(sig*sqrt2pi + EPS)

def loglik_of_station_pixels(pred, station_dict, station_num_obs, sigma=0.01, penalise_zeros=False):
    station_loglik = torch.zeros((), dtype=torch.float32).to(pred.device)
    n_elements = 0
    for b in range(pred.shape[0]):        
        yx = station_dict['coords_yx'][b]
        for i in range(yx.shape[0]):
            mask = station_dict['var_present'][b][i,:] # variable mask
            vals_true = station_dict['values'][b][i,:][mask]
            vals_pred = pred[b,:,yx[i,0],yx[i,1]][mask]
            #s_mult = sigma_growth[station_num_obs[b][i]-1]
            #station_loglik += normal_loglikelihood(vals_pred, vals_true, sigma*s_mult).sum()
            station_loglik += normal_loglikelihood(vals_pred, vals_true, sigma).sum()
            
            if penalise_zeros:
                zmask = vals_true==0
                ovzmask = vals_pred[zmask] > 0
                station_loglik += torch.tensor(-250) * vals_pred[zmask][ovzmask].sum() # -500
            
            n_elements += mask.sum()
    if n_elements==0:
        return station_loglik
    else:
        return station_loglik / n_elements.to(torch.float32)

def physical_constraints(pred, constraints, sea_mask=None, sigma=0.1):
    if sea_mask is None:
        sea_mask = torch.ones(constraints[:,0:1,:,:].shape, device=pred.device)
    sea_mask = sea_mask.squeeze(1).to(torch.bool)
    constraint_loglik = torch.zeros((), dtype=torch.float32).to(pred.device)
    for i in range(constraints.shape[1]):
        mask = torch.isfinite(constraints[:,i,:,:][sea_mask])
        loglik_var = normal_loglikelihood(pred[:,i,:,:][sea_mask][mask],
                                         constraints[:,i,:,:][sea_mask][mask],
                                         sigma)
        constraint_loglik += loglik_var.sum() / loglik_var.shape[0]
        
    return constraint_loglik

def preserve_coarse_grid_average(pred, coarse_inputs, sea_mask=None, sigma=0.1):
    scale = int(pred.shape[-1] / coarse_inputs.shape[-1])
    if sea_mask is None:
        sea_mask = torch.ones(coarse_inputs[:,0:1,:,:].shape, device=pred.device).to(torch.bool)
    sea_mask = sea_mask.expand(sea_mask.shape[0], pred.shape[1],
                               sea_mask.shape[2], sea_mask.shape[3])
    coarse_pred = nn.functional.avg_pool2d(pred, scale, stride=scale)
    return normal_loglikelihood(coarse_pred[sea_mask],
                                coarse_inputs[sea_mask], sigma).mean()

def enforce_local_continuity(pred, sigma=0.01):
    EX_2 = torch.square(nn.functional.avg_pool2d(pred, 4, stride=1))
    E_X2 = nn.functional.avg_pool2d(torch.square(pred), 4, stride=1)
    local_std = torch.sqrt(E_X2 - EX_2)
    return normal_loglikelihood(local_std, 0, sigma).mean()

def make_loss_func(train_pars, use_unseen_sites=True, penalise_zeros=False):
    
    def combined_loglik(pred, batch, null_stations=False):                            
        if null_stations:
            context_station_pixel_L = torch.tensor(0.).to(pred.device)
            target_station_pixel_L = torch.tensor(0.).to(pred.device)
        else:
            # likelihood at seen met sites
            context_station_pixel_L = loglik_of_station_pixels(
                pred,
                batch.context_station_dict,
                batch.context_num_obs,
                sigma=train_pars.sigma_context_stations,
                penalise_zeros=penalise_zeros
            )
            if use_unseen_sites:
                # likelihood at unseen met sites
                target_station_pixel_L = loglik_of_station_pixels(
                    pred,
                    batch.target_station_dict,
                    batch.target_num_obs,
                    sigma=train_pars.sigma_target_stations,
                    penalise_zeros=penalise_zeros
                )
            else:
                target_station_pixel_L = torch.zeros((), dtype=torch.float32).to(pred.device)
        
        # sea mask for not counting sea pixels
        sea_mask = trim(batch.fine_inputs, dps.scale)[:,0:1,:,:] # landfrac channel
        phys_constraints_L = physical_constraints(
            trim(pred, dps.scale),
            trim(batch.constraint_targets, dps.scale),
            sea_mask=sea_mask,
            sigma=train_pars.sigma_constraints
        )
        
        # coarsen sea mask and threshold        
        sea_mask = nn.functional.avg_pool2d(sea_mask, dps.scale, stride=dps.scale)
        sea_mask = sea_mask > 0.5
        grid_avg_L = preserve_coarse_grid_average(
            trim(pred, dps.scale),
            trim(batch.coarse_inputs, 1),
            sea_mask=sea_mask,
            sigma=train_pars.sigma_gridavg
        )

        # # local continuity: do we need this?
        # loc_cont_L = enforce_local_continuity(
            # trim(pred, dps.scale),
            # sigma=pars.sigma_localcont
        # )
        # return (context_station_pixel_L, target_station_pixel_L,
                # phys_constraints_L, grid_avg_L, loc_cont_L)
        
        return {'context_station_pixel_L':context_station_pixel_L,
                'target_station_pixel_L':target_station_pixel_L,
                'phys_constraints_L':phys_constraints_L,
                'grid_avg_L':grid_avg_L}

    return combined_loglik
