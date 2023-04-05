import pandas as pd
import numpy as np
import torch.nn as nn
import torch
import pkbar
import shutil
from pathlib import Path

from setupdata import Batch
from utils import save_checkpoint, prepare_run

sqrt2pi = torch.sqrt(torch.tensor(torch.pi * 2))

def normal_loglikelihood(x, mu, sigma):
    sig = torch.tensor(sigma).to(x.device)
    return torch.tensor(-0.5) * torch.square((x - mu)/sig) - torch.log(sig*sqrt2pi)

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

def physical_constraints(pred, constraints, sigma=0.1):
    mask = torch.isfinite(constraints[:,0,:,:])
    nelements_TA = mask.sum()
    loglik_TA = normal_loglikelihood(pred[:,0,:,:][mask], constraints[:,0,:,:][mask], sigma) # TA
    
    mask = torch.isfinite(constraints[:,1,:,:])
    nelements_PA = mask.sum()
    loglik_PA = normal_loglikelihood(pred[:,1,:,:][mask], constraints[:,1,:,:][mask], sigma) # PA
    
    mask = torch.isfinite(constraints[:,2,:,:])
    nelements_RH = mask.sum()
    loglik_RH = normal_loglikelihood(pred[:,-1,:,:][mask], constraints[:,2,:,:][mask], sigma) # RH
    return (loglik_TA.sum() / nelements_TA + 
            loglik_PA.sum() / nelements_PA + 
            loglik_RH.sum() / nelements_RH)

def preserve_coarse_grid_average(pred, coarse_inputs, sigma=0.1):
    scale = int(pred.shape[-1] / coarse_inputs.shape[-1])
    return normal_loglikelihood(nn.functional.avg_pool2d(pred, scale, stride=scale),
                                coarse_inputs, sigma).mean()
    
def make_loss_func(sigma_stations=0.05, sigma_constraints=0.1, sigma_gridavg=0.1):
    
    def loglik_step(pred, coarse_inputs, station_dict, constraints):        
        station_pixel_L = loglik_of_station_pixels(pred, station_dict, sigma=sigma_stations)
        phys_constraints_L = physical_constraints(pred, constraints, sigma=sigma_constraints)
        grid_avg_L = preserve_coarse_grid_average(pred, coarse_inputs, sigma=sigma_gridavg)
        return station_pixel_L, phys_constraints_L, grid_avg_L
        
    return loglik_step

def make_train_step(model, optimizer, loglik_step):
    def train_step(batch, kl_weight):
        model.train()
                
        # run data through model
        vae_normal_dists, KL = model.encode(batch.coarse_inputs, batch.fine_inputs, calc_kl=True)
        pred = model.decode(vae_normal_dists, batch.fine_inputs)
        
        # calculate log likelihood
        station_pixel_L, phys_constraints_L, grid_avg_L = loglik_step(
            pred,
            batch.coarse_inputs,
            batch.station_dict,
            batch.constraint_targets
        )
        loglik = station_pixel_L + phys_constraints_L + grid_avg_L
        
        # calculate ELBO
        ELBO = loglik - KL*kl_weight
        mean_neg_ELBO = -ELBO / float(len(batch.station_dict['values']))
        
        # propagate derivatives
        mean_neg_ELBO.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        return {'mean_neg_ELBO':mean_neg_ELBO.item(),
                'ELBO':ELBO.item(),
                'loglikelihood':loglik.item(),
                'KL':KL.item(),
                'station_pixel_L':station_pixel_L.item(),
                'phys_constraints_L':phys_constraints_L.item(),
                'grid_avg_L':grid_avg_L.item()}
    return train_step

def make_val_step(model, loglik_step):
    def val_step(batch, kl_weight):
        model.eval()        
        with torch.no_grad(): 
            # run data through model
            vae_normal_dists, KL = model.encode(batch.coarse_inputs, batch.fine_inputs, calc_kl=True)
            pred = model.decode(vae_normal_dists, batch.fine_inputs)
            
            # calculate log likelihood
            station_pixel_L, phys_constraints_L, grid_avg_L = loglik_step(
                pred,
                batch.coarse_inputs,
                batch.station_dict,
                batch.constraint_targets
            )
            loglik = station_pixel_L + phys_constraints_L + grid_avg_L
            
            # calculate ELBO
            ELBO = loglik - KL*kl_weight
            mean_neg_ELBO = -ELBO / float(len(batch.station_dict['values']))
            
        return {'mean_neg_ELBO':mean_neg_ELBO.item(),
                'ELBO':ELBO.item(),
                'loglikelihood':loglik.item(),
                'KL':KL.item(),
                'station_pixel_L':station_pixel_L.item(),
                'phys_constraints_L':phys_constraints_L.item(),
                'grid_avg_L':grid_avg_L.item()}
    return val_step

def create_running_loss_dict():
    return {'mean_neg_ELBO':[],
            'loglikelihood':[],
            'KL':[],
            'station_pixel_L':[],
            'phys_constraints_L':[],
            'grid_avg_L':[]}

def update_checkpoint(epoch, model, optimizer, best_loss, losses, val_losses):
    return {'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_loss': best_loss,
            'torch_random_state': torch.random.get_rng_state(),
            'numpy_random_state': np.random.get_state(),
            'losses': losses,
            'val_losses': val_losses}

def fit(model, optimizer, loglikelihood, datgen, 
        train_len, val_len, batch_size, KLW_scheduler, outdir='/logs/', 
        LR_scheduler=None, max_epochs=25, warmup_epochs=2, 
        checkpoint=None, device=None):
    ## setup
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    is_best = False
    train_step = make_train_step(model, optimizer, loglikelihood)
    val_step = make_val_step(model, loglikelihood)
    losses, val_losses, curr_epoch, best_loss = prepare_run(checkpoint, warmup_epochs)    
    #metrics_file = outdir + 'metrics.csv' # append running_ls to this?
    
    ## train
    for epoch in range(curr_epoch, max_epochs):
        model.train()
        kbar = pkbar.Kbar(target=train_len, epoch=epoch, num_epochs=max_epochs,
                          width=15, always_stateful=False)
        kl_weight = KLW_scheduler[epoch + warmup_epochs] # as we start from -warmup_epochs        
        running_ls = create_running_loss_dict()        
        for bidx in range(train_len):
            batch = datgen.get_batch(batch_size=batch_size,
                                     batch_type='train',
                                     load_binary_batch=True)
            loss_dict = train_step(Batch(batch, device=device), kl_weight)
            for loss_key in running_ls:
                running_ls[loss_key].append(loss_dict[loss_key])            
            print_values = [(key, running_ls[key][-1]) for key in running_ls]
            kbar.update(bidx, values=print_values)
        losses.append(np.mean(running_ls['mean_neg_ELBO'])) # append epoch average loss
        
        # save running_ls to metrics log file?
        
        if (not LR_scheduler is None) and (epoch>2):
            LR_scheduler.step()
        
        with torch.no_grad():
            model.eval()
            running_ls = create_running_loss_dict()
            for bidx in range(val_len):
                batch = datgen.get_batch(batch_size=batch_size,
                                         batch_type='val',
                                         load_binary_batch=True)
                loss_dict = val_step(Batch(batch, device=device), kl_weight)
                for loss_key in running_ls:
                    running_ls[loss_key].append(loss_dict[loss_key])
            val_losses.append(np.mean(running_ls['mean_neg_ELBO']))
            kbar.add(1, values=[("val_mean_neg_ELBO", val_losses[-1])])
                        
            is_best = bool(val_losses[-1] < best_loss)
            best_loss = min(val_losses[-1], best_loss)
            checkpoint = update_checkpoint(epoch, model, optimizer,
                                           best_loss, losses, val_losses)
            save_checkpoint(checkpoint, is_best, outdir)
        print("Done epoch %d" % epoch)
    return model, losses, val_losses


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
