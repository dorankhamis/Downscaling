import pandas as pd
import numpy as np
import torch.nn as nn
import torch

from params import data_pars as dps
from utils import trim

sqrt2pi = torch.sqrt(torch.tensor(torch.pi * 2))
EPS = 1e-10

def normal_loglikelihood(x, mu, sigma):
    sig = torch.tensor(sigma).to(x.device)
    return torch.tensor(-0.5) * torch.square((x - mu)/(sig+EPS)) - torch.log(sig*sqrt2pi + EPS)

def loglik_of_station_pixels(pred, station_dict, sigma=0.01):
    station_loglik = torch.zeros((), dtype=torch.float32).to(pred.device)
    n_elements = 0
    for b in range(pred.shape[0]):
        yx = station_dict['coords_yx'][b]
        for i in range(yx.shape[0]):
            mask = station_dict['var_present'][b][i,:] # variable mask
            vals_true = station_dict['values'][b][i,:][mask]
            vals_pred = pred[b,:,yx[i,0],yx[i,1]][mask]
            station_loglik += normal_loglikelihood(vals_pred, vals_true, sigma).sum()
            n_elements += mask.sum()
    if n_elements==0:
        return station_loglik
    else:
        return station_loglik / n_elements.to(torch.float32)

def physical_constraints(pred, constraints, sea_mask=None, sigma=0.1):
    if sea_mask is None:
        sea_mask = torch.ones(constraints[:,0:1,:,:].shape, device=pred.device)
    sea_mask = sea_mask.squeeze(1).to(torch.bool)
    # constrant variable order:
    # ['TA', 'PA', 'SWIN', 'LWIN', 'RH']
    
    # TA
    mask = torch.isfinite(constraints[:,0,:,:][sea_mask])
    loglik_TA = normal_loglikelihood(pred[:,0,:,:][sea_mask][mask],
                                     constraints[:,0,:,:][sea_mask][mask],
                                     sigma)
    
    # PA
    mask = torch.isfinite(constraints[:,1,:,:][sea_mask])
    loglik_PA = normal_loglikelihood(pred[:,1,:,:][sea_mask][mask],
                                     constraints[:,1,:,:][sea_mask][mask],
                                     sigma)

    # SWIN
    mask = torch.isfinite(constraints[:,2,:,:][sea_mask])
    loglik_SWIN = normal_loglikelihood(pred[:,2,:,:][sea_mask][mask],
                                       constraints[:,2,:,:][sea_mask][mask],
                                       sigma)
                                      
    # LWIN
    mask = torch.isfinite(constraints[:,3,:,:][sea_mask])
    loglik_LWIN = normal_loglikelihood(pred[:,3,:,:][sea_mask][mask],
                                       constraints[:,3,:,:][sea_mask][mask],
                                       sigma)
    
    # currently we don't have a WS constraint

    # RH
    mask = torch.isfinite(constraints[:,-1,:,:][sea_mask])
    #nelements_RH = mask.sum()
    loglik_RH = normal_loglikelihood(pred[:,-1,:,:][sea_mask][mask],
                                     constraints[:,-1,:,:][sea_mask][mask],
                                     sigma)
    # possible error thrown here if masks have no True elements
    # e.g. all sea tile
    return (loglik_TA.sum() / loglik_TA.shape[0] + 
            loglik_PA.sum() / loglik_PA.shape[0] + 
            loglik_SWIN.sum() / loglik_SWIN.shape[0] + 
            loglik_LWIN.sum() / loglik_LWIN.shape[0] + 
            loglik_RH.sum() / loglik_RH.shape[0])

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

def make_loss_func(train_pars):
    
    def combined_loglik(pred, batch, null_stations=False):                            
        if null_stations:
            context_station_pixel_L = torch.tensor(0.).to(pred.device)
            target_station_pixel_L = torch.tensor(0.).to(pred.device)
        else:
            # likelihood at seen met sites
            context_station_pixel_L = loglik_of_station_pixels(
                pred,
                batch.context_station_dict,
                sigma=train_pars.sigma_context_stations
            )
        
            # likelihood at unseen met sites
            target_station_pixel_L = loglik_of_station_pixels(
                pred,
                batch.target_station_dict,
                sigma=train_pars.sigma_target_stations
            )
        
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

# def enforce_local_continuity(pred, coarse_inputs, sigma=0.1):
    # # enforce local continuity  
    # EX_2 = tf.square(self.av_pool_local(pred))
    # E_X2 = self.av_pool_local(tf.square(pred))
    # local_std = tf.sqrt(E_X2 - EX_2)
    # local_cont_loss = tf.reduce_mean(local_std)
    # return local_cont_loss


# def crps(y_true, y_pred):
    # # added by me, doran
    # # not sure of the dimensions of y_true and y_pred?
    # # https://github.com/palikar/edward/blob/master/edward/criticisms/evaluate.py
    # # and based on 
    # # https://github.com/properscoring/properscoring/blob/master/properscoring/_crps.py       
    
    # score = torch.mean(torch.abs(y_pred - y_true), dim=0) # length: nvars
    
    # # should cross over ensemble rather than variable
    # diff = torch.unsqueeze(y_pred, 0) - torch.unsqueeze(y_pred, 1) # ens x ens # nvars
    # score = torch.add(score, torch.multiply(torch.tensor(-0.5, dtype=diff.dtype),
                                            # torch.mean(torch.abs(diff), dim=(0, 1))))
    # # using reduce_sum rather than reduce_mean as we divide by the total
    # # number of elements (time points and variables present at each pixel)
    # # in the ensemble station loss function
    # return torch.sum(score, dim=0)

# def station_pixel_ensemble_crps(preds, station_dict):
    # station_loss = torch.zeros((), dtype=torch.float32) # needs gradient??
    # n_elements = 0
    # for b in range(preds.shape[1]):
        # yx = station_dict['coords_yx'][b]
        # for i in range(yx.shape[0]):
            # mask = station_dict['var_present'][b][i,:] # variable mask
            # vals_true = station_dict['values'][b][i,:][mask]
            # vals_ens_pred = torch.transpose(
                # torch.transpose(preds[:, b, :, yx[i,0], yx[i,1]], 0, 1)[mask],
            # 0, 1)
            # station_loss += crps(vals_true, vals_ens_pred)
            # n_elements += torch.sum(mask)
    # if n_elements==0:
        # return station_loss
    # else:
        # return station_loss / n_elements.to(torch.float32)
