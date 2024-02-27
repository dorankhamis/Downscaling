import pandas as pd
import numpy as np
import torch.nn as nn
import torch
import pkbar
import datetime

from setupdata3 import Batch
from utils import save_checkpoint, prepare_run, create_attention_masks
from params import data_pars as dps, train_pars, model_pars as mps

batch_quantiles_path = '/gws/nopw/j04/ceh_generic/netzero/downscaling/training_data/var_quantile_samples/'

def model_step(model, batch, loglikelihood, var=None, use_masks=False):
    if use_masks:
        # prepare attention masks
        masks = create_attention_masks(model, batch, var)
    
    # run data through model
    if use_masks:
        pred = model(
            batch.coarse_inputs,
            batch.fine_inputs,
            batch.context_data,
            batch.context_locs,
            context_masks=masks['context_masks'],
            context_soft_masks=masks['context_soft_masks'],
            pixel_passers=masks['pixel_passers']
        )
    else:
        pred = model(
            batch.coarse_inputs,
            batch.fine_inputs,
            batch.context_data,
            batch.context_locs
        )        
    
    # run data through model with no context sites
    batch2 = create_null_batch(batch)
    pred2 = model(batch2.coarse_inputs, batch2.fine_inputs,
                  batch2.context_data, batch2.context_locs)
    
    clip_output = False
    if type(var)==list:
        if ('SWIN' in var) or ('PRECIP' in var):
            clip_output = True
    elif type(var)==str:
        if (var=='SWIN') or (var=='PRECIP'):
            clip_output = True
    if clip_output:
        pred = nn.functional.relu(pred) # cannot be negative
        pred2 = nn.functional.relu(pred2) # cannot be negative
            
    # calculate log likelihood, trimming edge pixels at coarse scale
    loss_dict = loglikelihood(pred, batch, null_stations=False)
    loss_dict_null = loglikelihood(pred2, batch, null_stations=True)
    loglik = 0.9 * sum(loss_dict.values()) + 0.1 * sum(loss_dict_null.values())
    
    # add vars to loss dict and return
    loss_dict['nll'] = -loglik
    loss_dict['loglikelihood'] = loglik
    for k in loss_dict_null:
        if not 'station' in k:
            loss_dict[k+'_nc'] = loss_dict_null[k]
    return loss_dict

def make_train_step(model, optimizer, loglikelihood):
    def train_step(batch, var=None):
        model.train()
        
        # take step        
        losses = model_step(model, batch, loglikelihood, var=var)
        
        # propagate derivatives        
        losses['nll'].backward()
        optimizer.step()
        optimizer.zero_grad()
        
        return {k:losses[k].item() for k in losses}
                
    return train_step

def make_val_step(model, loglikelihood):
    def val_step(batch, var=None):
        model.eval()
             
        with torch.no_grad():
            # take step
            losses = model_step(model, batch, loglikelihood, var=var)
        
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
    zero_c_numobs = [None for b in range(len(batch.context_num_obs))]
    zero_t_numobs = [None for b in range(len(batch.target_num_obs))]
    zero_c_station_dict = {k:None for k in batch.context_station_dict}
    zero_t_station_dict = {k:None for k in batch.target_station_dict}
    zero_raw_station_df = [{'context':None, 'target':None} for i in range(len(batch.raw_station_dict))]
    
    batch2.context_locs = zero_locs
    batch2.context_data = zero_data
    batch2.context_num_obs = zero_c_numobs
    batch2.target_num_obs = zero_t_numobs
    batch2.context_station_dict = zero_c_station_dict
    batch2.target_station_dict = zero_t_station_dict
    batch2.raw_station_dict = zero_raw_station_df    
    return batch2

def read_qbin(var, nbin, batch_type='train'):
    return pd.read_pickle(batch_quantiles_path + f'/{var}_bin_{nbin}_{batch_type}.pkl',
                          compression={'method': 'gzip', 'compresslevel': 5})

def quantile_bin_choice(bin_dict, qkeys, q_ind, nqbins, num_q_samples, batch_type='train'):
    # choose a quantile bin to extract samples from
    vv, nb = bin_dict[qkeys[q_ind % nqbins]]
    bin_samples = read_qbin(vv, nb, batch_type=batch_type)
    maxii = bin_samples.shape[0]
    ii = np.random.randint(low=0, high=maxii)
    this_sample = bin_samples.iloc[ii:(ii+1)]
    other_samples = bin_samples[
        bin_samples['DATE_TIME'].dt.date == this_sample.DATE_TIME.dt.date.values[0]
    ]
    n_samp = max(1, min(num_q_samples, other_samples.shape[0]))
    other_samples = other_samples.sample(n_samp)
    date = this_sample.DATE_TIME.dt.date.values[0]
    parent_pixel_ids = list(other_samples.parent_pixel_id)
    times = list(other_samples.DATE_TIME.dt.hour)
    if (q_ind % nqbins)==(nqbins-1): np.random.shuffle(qkeys)
    q_ind += 1
    return date, times, parent_pixel_ids, qkeys, q_ind

def fit(model, optimizer, loglikelihood, var, datgen, 
        batch_prob_hourly, LR_scheduler=None,
        outdir='/logs/', checkpoint=None, device=None):
    ## setup
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    if type(var)==str: var = [var]
    is_best = False
    train_step = make_train_step(model, optimizer, loglikelihood)
    val_step = make_val_step(model, loglikelihood)
    losses, val_losses, curr_epoch, best_loss = prepare_run(checkpoint)    
    
    bin_dict = {}
    nqbins = 10
    for v in var:
        for b in range(nqbins):
            bin_dict[f'{v}_{b}'] = (v, b)
    qkeys = list(bin_dict.keys())
    zero_timedelta = datetime.timedelta(days=0)
    num_q_samples = int(np.ceil(0.25*train_pars.batch_size))
           
    ## train
    for epoch in range(curr_epoch, train_pars.max_epochs):
        model.train()
        kbar = pkbar.Kbar(target=train_pars.train_len, epoch=epoch,
                          num_epochs=train_pars.max_epochs,
                          width=15, always_stateful=False)
        p_hourly = 1 #batch_prob_hourly[epoch]
        running_ls = None
        q_ind = 0
        np.random.shuffle(qkeys)
        for bidx in range(1, train_pars.train_len+1):
            context_frac = None # sample randomly each time
 
            # choose a quantile bin to extract samples from
            date, times, parent_pixel_ids, qkeys, q_ind = quantile_bin_choice(
                bin_dict, qkeys, q_ind, nqbins, num_q_samples, batch_type='train'
            )
                
            batch = datgen.get_batch(
                var,
                batch_size=train_pars.batch_size,
                batch_type='train',
                context_frac=context_frac,
                p_hourly=p_hourly,
                date=date,
                parent_pixel_ids=parent_pixel_ids,
                times=times
            )
            batch = Batch(batch, var_list=var, device=device)
                        
            loss_dict = train_step(batch, var=var)
            running_ls = update_running_loss(running_ls, loss_dict)
            
            print_values = [(key, running_ls[key][-1]) for key in running_ls]
            kbar.update(bidx, values=print_values)
        losses.append(np.mean(running_ls['nll'])) # append epoch average loss
        
        if (not LR_scheduler is None) and (epoch>2):
            LR_scheduler.step()
        
        # validation
        with torch.no_grad():
            kbarv = pkbar.Kbar(target=train_pars.val_len, epoch=epoch,
                               num_epochs=train_pars.max_epochs,
                               width=15, always_stateful=False)
            model.eval()
            running_ls = None
            q_ind = 0
            np.random.shuffle(qkeys)
            for bidx in range(1, train_pars.val_len+1):
                context_frac = 0.75 # fix constant for more stable validation?

                # choose a quantile bin to extract samples from
                date, times, parent_pixel_ids, qkeys, q_ind = quantile_bin_choice(
                    bin_dict, qkeys, q_ind, nqbins, num_q_samples, batch_type='val'
                )

                                
                batch = datgen.get_batch(
                    var,
                    batch_size=train_pars.batch_size,
                    batch_type='val',
                    context_frac=context_frac,
                    p_hourly=p_hourly
                )                
                batch = Batch(batch, var_list=var, device=device)
                
                loss_dict = val_step(batch, var=var)
                running_ls = update_running_loss(running_ls, loss_dict)
            
                print_values = [(key, running_ls[key][-1]) for key in running_ls]
                kbarv.update(bidx, values=print_values)
                
            val_losses.append(np.mean(running_ls['nll']))
            kbar.add(1, values=[("val_nll", val_losses[-1])])
            
            is_best = bool(val_losses[-1] < best_loss)
            best_loss = min(val_losses[-1], best_loss)
            checkpoint = update_checkpoint(epoch, model, optimizer,
                                           best_loss, losses, val_losses)
            save_checkpoint(checkpoint, is_best, outdir)
        print("Done epoch %d" % (epoch+1))
    return model, losses, val_losses

