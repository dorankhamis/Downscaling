import pandas as pd
import numpy as np
import torch.nn as nn
import torch
import pkbar

from setupdata2 import Batch
from utils import save_checkpoint, prepare_run, make_mask_list_from_filters
from params import data_pars as dps, train_pars, model_pars as mps

def model_step(model, batch, loglikelihood, null_stations=False):
    # run data through model
    pred = model(batch.coarse_inputs, batch.fine_inputs,
                  batch.context_data, batch.context_locs)
            
    # calculate log likelihood, trimming edge pixels at coarse scale
    loss_dict = loglikelihood(pred, batch, null_stations=null_stations)        
    loglik = sum(loss_dict.values())
    
    # add vars to loss dict and return
    loss_dict['nll'] = -loglik
    loss_dict['loglikelihood'] = loglik    
    return loss_dict

def make_train_step(model, optimizer, loglikelihood):
    def train_step(batch, null_stations=False):        
        model.train()
        
        # take step        
        losses = model_step(model, batch, loglikelihood,                            
                            null_stations=null_stations)
        
        # propagate derivatives        
        losses['nll'].backward()
        optimizer.step()
        optimizer.zero_grad()
        
        return {k:losses[k].item() for k in losses}
                
    return train_step

def make_val_step(model, loglikelihood):
    def val_step(batch, null_stations=False):
        model.eval()
             
        with torch.no_grad():
            # take step
            losses = model_step(model, batch, loglikelihood,                                
                                null_stations=null_stations)
        
        return {k:losses[k].item() for k in losses}
    
    return val_step

def update_checkpoint(epoch, model, optimizer, best_loss, losses, val_losses):
    return {'epoch': epoch,
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
    
def create_null_batch(batch, constraints=True):
    batch2 = Batch(0, constraints=constraints)
    batch2.copy(batch)
    
    zero_locs = [torch.zeros((1, batch.context_locs[0].shape[1], 0)) for b in range(batch.coarse_inputs.shape[0])]
    zero_data = [torch.zeros((1, batch.context_data[0].shape[1], 0)) for b in range(batch.coarse_inputs.shape[0])]
    
    batch2.context_locs = zero_locs
    batch2.context_data = zero_data    
    return batch2

def fit(model, optimizer, loglikelihood, var, datgen, 
        batch_prob_hourly, LR_scheduler=None, null_batch_prob=0.5,
        outdir='/logs/', checkpoint=None, device=None):
    ## setup
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    is_best = False
    train_step = make_train_step(model, optimizer, loglikelihood)
    val_step = make_val_step(model, loglikelihood)
    losses, val_losses, curr_epoch, best_loss = prepare_run(checkpoint)    
    
    ## train
    for epoch in range(curr_epoch, train_pars.max_epochs):
        model.train()
        kbar = pkbar.Kbar(target=train_pars.train_len, epoch=epoch,
                          num_epochs=train_pars.max_epochs,
                          width=15, always_stateful=False)
        p_hourly = batch_prob_hourly[epoch]    
        running_ls = None
        for bidx in range(1, train_pars.train_len+1):
            null_stations = False
            context_frac = None
            if np.random.uniform() < null_batch_prob:                
                null_stations = True
                context_frac = 0
                                
            batch = datgen.get_chess_batch(
                var,
                batch_size=train_pars.batch_size,
                batch_type='train',
                load_binary_batch=True,
                context_frac=context_frac,
                p_hourly=p_hourly
            )
            batch = Batch(batch, var_list=var, device=device)
                        
            loss_dict = train_step(batch, null_stations=null_stations)
            running_ls = update_running_loss(running_ls, loss_dict)
            
            print_values = [(key, running_ls[key][-1]) for key in running_ls]
            kbar.update(bidx, values=print_values)
        losses.append(np.mean(running_ls['nll'])) # append epoch average loss
        
        if (not LR_scheduler is None) and (epoch>2):
            LR_scheduler.step()
        
        # validation
        with torch.no_grad():
            model.eval()
            running_ls = None
            for bidx in range(1, train_pars.val_len+1):
                null_stations = False
                context_frac = 0.75 # fix constant for more stable validation?
                if np.random.uniform() < null_batch_prob:                
                    null_stations = True
                    context_frac = 0
                #print('here1')
                batch = datgen.get_chess_batch(
                    var,
                    batch_size=train_pars.batch_size,
                    batch_type='val',
                    load_binary_batch=True,
                    context_frac=context_frac,
                    p_hourly=p_hourly
                )
                #print('here2')
                batch = Batch(batch, var_list=var, device=device)
                #print('here3')
                loss_dict = val_step(batch, null_stations=null_stations)
                #print('here4')
                running_ls = update_running_loss(running_ls, loss_dict)
            val_losses.append(np.mean(running_ls['nll']))
            kbar.add(1, values=[("val_nll", val_losses[-1])])
            
            is_best = bool(val_losses[-1] < best_loss)
            if epoch > (train_pars.warmup_epochs + train_pars.increase_epochs):
                best_loss = min(val_losses[-1], best_loss)
            else:
                best_loss = np.inf
            checkpoint = update_checkpoint(epoch, model, optimizer,
                                           best_loss, losses, val_losses)
            save_checkpoint(checkpoint, is_best, outdir)
        print("Done epoch %d" % (epoch+1))
    return model, losses, val_losses
