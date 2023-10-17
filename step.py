import pandas as pd
import numpy as np
import torch.nn as nn
import torch
import pkbar

from setupdata import Batch, make_mask_list
from utils import save_checkpoint, prepare_run
from params import data_pars as dps

def model_step(model, batch, loglikelihood, kl_weight=1, null_stations=False):
    # run data through model
    vae_normal_dists, KL = model.encode(batch.coarse_inputs, batch.fine_inputs,
                                        calc_kl=True, mask=None)
    pred = model.decode(vae_normal_dists, batch.fine_inputs, batch.context_data,
                        batch.context_locs, batch.grid_locs, masks=batch.masks,
                        context_mask=batch.context_mask)
    
    # calculate log likelihood, trimming edge pixels at coarse scale
    loss_dict = loglikelihood(pred, batch, null_stations=null_stations)        
    loglik = sum(loss_dict.values())
    
    # calculate ELBO
    ELBO = loglik - KL*kl_weight
    neg_ELBO = -ELBO
    
    # add vars to loss dict and return
    loss_dict['neg_ELBO'] = -ELBO
    loss_dict['loglikelihood'] = loglik
    loss_dict['KL'] = KL
    return loss_dict

def make_train_step(model, optimizer, loglikelihood):
    def train_step(batch, kl_weight, null_stations=False):
        model.train()
                
        # take step           
        losses = model_step(model, batch, loglikelihood,
                            kl_weight=kl_weight,
                            null_stations=null_stations)
        
        # propagate derivatives        
        losses['neg_ELBO'].backward()
        optimizer.step()
        optimizer.zero_grad()
        
        return {k:losses[k].item() for k in losses}
                
    return train_step

def make_val_step(model, loglikelihood):
    def val_step(batch, kl_weight, null_stations=False):
        model.eval()
        
        with torch.no_grad(): 
            # take step
            losses = model_step(model, batch, loglikelihood,
                                kl_weight=kl_weight,
                                null_stations=null_stations)
        
        return {k:losses[k].item() for k in losses}
    
    return val_step

def update_checkpoint(epoch, model, optimizer, best_loss, losses, val_losses):
    return {'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_loss': best_loss,
            'torch_random_state': torch.random.get_rng_state(),
            'numpy_random_state': np.random.get_state(),
            'losses': losses,
            'val_losses': val_losses}

def update_running_loss(running_ls, loss_dict):
    if running_ls is None:
        running_ls = {k:[loss_dict[k]] for k in loss_dict}
    else:
        for loss_key in running_ls:
            running_ls[loss_key].append(loss_dict[loss_key])
    return running_ls

def nullify_station_data(batch):
    batch.context_data.fill_(0)
    batch.context_mask.fill_(0)
    batch.context_locs.fill_(0)
    batch.context_mask[:,:,0] = 1
    batch.context_locs[:,:,0] = -1
    #batch.context_data[:,::2,0] = 1
    return batch

def fit(model, optimizer, loglikelihood, datgen, train_pars,
        KLW_scheduler, LR_scheduler=None, null_batch_prob=0.2,
        outdir='/logs/', checkpoint=None, device=None):
    ## setup
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    is_best = False
    train_step = make_train_step(model, optimizer, loglikelihood)
    val_step = make_val_step(model, loglikelihood)
    losses, val_losses, curr_epoch, best_loss = prepare_run(checkpoint)    
    
    # pre-make masks
    masks = make_mask_list(datgen.dim_l, datgen.dim_l, datgen.scale)
    
    ## train
    for epoch in range(curr_epoch, train_pars.max_epochs):
        model.train()
        kbar = pkbar.Kbar(target=train_pars.train_len, epoch=epoch,
                          num_epochs=train_pars.max_epochs,
                          width=15, always_stateful=False)
        kl_weight = KLW_scheduler[epoch]    
        running_ls = None
        for bidx in range(train_pars.train_len):
            null_stations = False
            batch = datgen.get_batch(batch_size=train_pars.batch_size,
                                     batch_type='train',
                                     load_binary_batch=True)
            batch = Batch(batch, masks, datgen.X1, device=device)
            
            if np.random.uniform() < null_batch_prob:
                batch = nullify_station_data(batch)
                null_stations = True
                
            loss_dict = train_step(batch, kl_weight, null_stations=null_stations)
            running_ls = update_running_loss(running_ls, loss_dict)
            print_values = [(key, running_ls[key][-1]) for key in running_ls]
            kbar.update(bidx, values=print_values)
        losses.append(np.mean(running_ls['neg_ELBO'])) # append epoch average loss
        
        if (not LR_scheduler is None) and (epoch>2):
            LR_scheduler.step()
        
        # validation
        with torch.no_grad():
            model.eval()
            running_ls = None
            for bidx in range(train_pars.val_len):
                null_stations = False
                batch = datgen.get_batch(batch_size=train_pars.batch_size,
                                         batch_type='val',
                                         load_binary_batch=True)
                batch = Batch(batch, masks, datgen.X1, device=device)
                
                if np.random.uniform() < null_batch_prob:
                    batch = nullify_station_data(batch)
                    null_stations = True
                    
                loss_dict = val_step(batch, kl_weight, null_stations=null_stations)
                running_ls = update_running_loss(running_ls, loss_dict)
            val_losses.append(np.mean(running_ls['neg_ELBO']))
            kbar.add(1, values=[("val_neg_ELBO", val_losses[-1])])
                        
            is_best = bool(val_losses[-1] < best_loss)
            best_loss = min(val_losses[-1], best_loss)
            checkpoint = update_checkpoint(epoch, model, optimizer,
                                           best_loss, losses, val_losses)
            save_checkpoint(checkpoint, is_best, outdir)
        print("Done epoch %d" % (epoch+1))
    return model, losses, val_losses
