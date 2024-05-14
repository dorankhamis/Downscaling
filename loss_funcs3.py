import pandas as pd
import numpy as np
import torch.nn as nn
import torch

from params2 import data_pars as dps
from params2 import normalisation as nm
from utils2 import trim

sqrt2pi = torch.sqrt(torch.tensor(torch.pi * 2))
EPS = 1e-10

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
            station_loglik += normal_loglikelihood(vals_pred, vals_true, sigma).sum()
            
            if penalise_zeros:
                zmask = vals_true==0
                ovzmask = vals_pred[zmask] > 0
                station_loglik += torch.tensor(-50) * vals_pred[zmask][ovzmask].sum()
                
                zmask = vals_true>0
                ovzmask = vals_pred[zmask] == 0
                station_loglik += torch.tensor(-50) * vals_true[zmask][ovzmask].sum()
            
            n_elements += mask.sum()
    if n_elements==0:
        return station_loglik # zeros
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

def preserve_coarse_grid_average(pred, coarse_inputs, sea_mask=None,
                                 sigma=0.1, divisor_override=None):
    scale = int(pred.shape[-1] / coarse_inputs.shape[-1])
    if sea_mask is None:
        sea_mask = torch.ones(coarse_inputs[:,0:1,:,:].shape, device=pred.device).to(torch.bool)
    sea_mask = sea_mask.expand(sea_mask.shape[0], pred.shape[1],
                               sea_mask.shape[2], sea_mask.shape[3])
    coarse_pred = nn.functional.avg_pool2d(pred, scale, stride=scale,
                                           divisor_override=divisor_override)
    return normal_loglikelihood(coarse_pred[sea_mask],
                                coarse_inputs[sea_mask], sigma).mean()

def enforce_local_continuity(pred, sigma=0.01):
    ## unused
    EX_2 = torch.square(nn.functional.avg_pool2d(pred, 4, stride=1))
    E_X2 = nn.functional.avg_pool2d(torch.square(pred), 4, stride=1)
    local_std = torch.sqrt(E_X2 - EX_2)
    return normal_loglikelihood(local_std, 0, sigma).mean()

def make_loss_func(train_pars, use_unseen_sites=True, penalise_zeros=False, var=None):
    if type(var)==str: var = [var]
    
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
                sigma=train_pars.sigma_context_stations[var[0]],
                penalise_zeros=penalise_zeros
            )        
            
            if use_unseen_sites:
                # likelihood at unseen met sites
                target_station_pixel_L = loglik_of_station_pixels(
                    pred,
                    batch.target_station_dict,
                    batch.target_num_obs,
                    sigma=train_pars.sigma_target_stations[var[0]],
                    penalise_zeros=penalise_zeros
                )
            else:
                target_station_pixel_L = torch.zeros((), dtype=torch.float32).to(pred.device)
        
        # sea mask for not counting sea pixels
        #sea_mask = trim(batch.fine_inputs, dps.scale)[:,0:1,:,:] # landfrac channel
        
        # ireland and shetland mask instead, to leave a bit of sea around
        # the land for better averaging?
        lat_up = 55.3
        lat_down = 51.0
        lon_right = -5.4
        lon_left = -8.3
        lat_shet = 59.5
        ireland_shetland_mask = ~torch.bitwise_or(
            torch.bitwise_and(
                torch.bitwise_and(batch.fine_inputs[:,-2:-1,:,:]<(lat_up/nm.lat_norm),
                                  batch.fine_inputs[:,-2:-1,:,:]>(lat_down/nm.lat_norm)),
                torch.bitwise_and(batch.fine_inputs[:,-1:,:,:]<(lon_right/nm.lon_norm),
                                  batch.fine_inputs[:,-1:,:,:]>(lon_left/nm.lon_norm))
            ), batch.fine_inputs[:,-2:-1,:,:]>(lat_shet/nm.lat_norm)
        )
        sea_mask = trim(ireland_shetland_mask, dps.scale)
        
        phys_constraints_L = physical_constraints(
            trim(pred, dps.scale),
            trim(batch.constraint_targets, dps.scale),
            sea_mask=sea_mask,
            sigma=train_pars.sigma_constraints[var[0]]
        )
        
        # coarsen sea mask and threshold        
        sea_mask_coarse = nn.functional.avg_pool2d(sea_mask.to(torch.float32), dps.scale, stride=dps.scale)
        sea_mask_coarse = sea_mask_coarse > 0.5
        if var=='PRECIP':
            divisor_override = 1. # to preserve grid sum rather than average
            coarse_mult = dps.scale # multiply up coarse value by resolution
        else:
            divisor_override = None
            coarse_mult = 1.
            
        grid_avg_L = preserve_coarse_grid_average(
            trim(pred, dps.scale),
            coarse_mult * trim(batch.coarse_inputs, 1),
            sea_mask=sea_mask_coarse,
            sigma=train_pars.sigma_gridavg[var[0]],
            divisor_override=divisor_override
        )
        
        if var[0]=='WS':
            # unnormalise first and then renormalise as different norms betwee, UX,VY and WS 
            ws_pred = torch.sqrt(torch.square(pred[:,0:1,:,:] * nm.ws_sd) + torch.square(pred[:,1:2,:,:] * nm.ws_sd))
            ws_pred = (ws_pred - nm.ws_mu) / nm.ws_sd
                    
            ws_constraint = torch.sqrt(
                torch.square(batch.constraint_targets[:,0:1,:,:] * nm.ws_sd) + 
                torch.square(batch.constraint_targets[:,1:2,:,:] * nm.ws_sd)
            )
            ws_constraint = (ws_constraint - nm.ws_mu) / nm.ws_sd
            
            ws_coarse = torch.sqrt(
                torch.square(batch.coarse_inputs[:,0:1,:,:] * nm.ws_sd) + 
                torch.square(batch.coarse_inputs[:,1:2,:,:] * nm.ws_sd)
            )
            ws_coarse = (ws_coarse - nm.ws_mu) / nm.ws_sd
            
            if null_stations:
                context_station_pixel_L_ws = torch.tensor(0.).to(pred.device)
                target_station_pixel_L_ws = torch.tensor(0.).to(pred.device)
            else:
                ws_context_station_dict = {}
                ws_context_station_dict['coords_yx'] = [torch.clone(t) for t in batch.context_station_dict['coords_yx']]
                ws_context_station_dict['var_present'] = [torch.clone(t) for t in batch.context_station_dict['var_present']]
                ws_context_station_dict['values'] = []
                for b in range(len(batch.context_station_dict['values'])):
                    context_ws = torch.sqrt(
                        torch.square(batch.context_station_dict['values'][b][:,0]) + 
                        torch.square(batch.context_station_dict['values'][b][:,1])
                    )
                    ws_context_station_dict['values'].append(context_ws[...,None])
                    var_pred = (batch.context_station_dict['var_present'][b].sum(dim=1, keepdims=True) // 2)
                    ws_context_station_dict['var_present'][b] = var_pred.to(torch.bool)
                
                # likelihood at seen met sites
                context_station_pixel_L_ws = loglik_of_station_pixels(
                    ws_pred,
                    ws_context_station_dict,
                    batch.context_num_obs,
                    sigma=train_pars.sigma_context_stations[var[0]]*2,
                    penalise_zeros=penalise_zeros
                )        
                
                if use_unseen_sites:
                    ws_target_station_dict = {}
                    ws_target_station_dict['coords_yx'] = [torch.clone(t) for t in batch.target_station_dict['coords_yx']]
                    ws_target_station_dict['var_present'] = [torch.clone(t) for t in batch.target_station_dict['var_present']]
                    ws_target_station_dict['values'] = []
                    for b in range(len(batch.target_station_dict['values'])):
                        target_ws = torch.sqrt(
                            torch.square(batch.target_station_dict['values'][b][:,0]) + 
                            torch.square(batch.target_station_dict['values'][b][:,1])
                        )
                        ws_target_station_dict['values'].append(target_ws[...,None])
                        var_pred = (batch.target_station_dict['var_present'][b].sum(dim=1, keepdims=True) // 2)
                        ws_target_station_dict['var_present'][b] = var_pred.to(torch.bool)
                        
                    # likelihood at unseen met sites
                    target_station_pixel_L_ws = loglik_of_station_pixels(
                        ws_pred,
                        ws_target_station_dict,
                        batch.target_num_obs,
                        sigma=train_pars.sigma_target_stations[var[0]]*2,
                        penalise_zeros=penalise_zeros
                    )
                else:
                    target_station_pixel_L_ws = torch.zeros((), dtype=torch.float32).to(pred.device)
            
            phys_constraints_L_ws = physical_constraints(
                trim(ws_pred, dps.scale),
                trim(ws_constraint, dps.scale),
                sea_mask=sea_mask,
                sigma=train_pars.sigma_constraints[var[0]]
            )
            
            grid_avg_L_ws = preserve_coarse_grid_average(
                trim(ws_pred, dps.scale),
                trim(ws_coarse, 1),
                sea_mask=sea_mask_coarse,
                sigma=train_pars.sigma_gridavg[var[0]],
                divisor_override=divisor_override
            )
            return {'context_station_pixel_L':context_station_pixel_L + context_station_pixel_L_ws,
                    'target_station_pixel_L':target_station_pixel_L + target_station_pixel_L_ws,
                    'phys_constraints_L':phys_constraints_L + phys_constraints_L_ws,
                    'grid_avg_L':grid_avg_L + grid_avg_L_ws}
                           
        return {'context_station_pixel_L':context_station_pixel_L,
                'target_station_pixel_L':target_station_pixel_L,
                'phys_constraints_L':phys_constraints_L,
                'grid_avg_L':grid_avg_L}

    return combined_loglik
