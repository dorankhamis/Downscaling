import numpy as np
import pandas as pd
import torch
import argparse
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from pathlib import Path

from setupdata3 import data_generator, Batch
from utils import frange_cycle_linear, setup_checkpoint
from model2 import SimpleDownscaler
from params import data_pars, model_pars, train_pars
from loss_funcs2 import make_loss_func
from step3 import fit

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__=="__main__":
    var_names = ['TA', 'PA', 'SWIN', 'LWIN', 'WS', 'RH', 'PRECIP']
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("var_num", help = "variable to use")
        args = parser.parse_args()
        var = var_names[int(args.var_num)]
    except: # not running as batch script
        var = 'TA'
    
    print('\nDevice: %s \n' % device)
    
    # file paths
    log_dir = '/home/users/doran/projects/downscaling/logs/'
    model_name = f'dwnsamp_{var}'
    model_outdir = f'{log_dir}/{model_name}/'
    Path(model_outdir).mkdir(parents=True, exist_ok=True)

    # training flags
    load_prev_chkpnt = True
    specify_chkpnt = f'{model_name}/checkpoint.pth' # if None, load best, otherwise "modelname/checkpoint.pth"
    reset_chkpnt = False

    ## create data generator
    datgen = data_generator()
    if var=='PRECIP': datgen.load_EA_rain_gauge_data()

    ## dummy batch for model param fetching
    batch = datgen.get_batch(var,
                             batch_size=train_pars.batch_size,
                             batch_type='train',
                             p_hourly=1)
                                  
    ## create model
    model = SimpleDownscaler(
        input_channels=batch['coarse_inputs'].shape[1],
        hires_fields=batch['fine_inputs'].shape[1],
        output_channels=batch['coarse_inputs'].shape[1],
        context_channels=batch['station_data'][0].shape[1],
        filters=model_pars.filters,
        dropout_rate=model_pars.dropout_rate,
        scale=data_pars.scale,
        scale_factor=model_pars.scale_factor,
        attn_heads=model_pars.attn_heads,
        ds_cross_attn=model_pars.ds_cross_attn,
        pe=model_pars.pe
    )

    model.to(device)
    del(batch)

    ## create optimizer, schedulers and loss function
    optimizer = Adam(model.parameters(), lr=train_pars.lr)
    LR_scheduler = ExponentialLR(optimizer, gamma=train_pars.gamma)
    
    if False:
        ## schedule for increasing prob of hourly samples in a batch
        batch_prob_hourly = np.zeros(train_pars.warmup_epochs, np.float32)
        batch_prob_hourly = np.hstack([
            batch_prob_hourly,
            np.linspace(0, train_pars.p_hourly_max, train_pars.increase_epochs)
        ])
        batch_prob_hourly = np.hstack([
            batch_prob_hourly,
            np.ones(train_pars.cooldown_epochs, np.float32)*train_pars.p_hourly_max
        ])
    else:
        # or all hourly samples!
        batch_prob_hourly = np.ones((train_pars.warmup_epochs + 
                                     train_pars.increase_epochs + 
                                     train_pars.cooldown_epochs), dtype=np.float32)
    
    penalise_zeros = False
    #if var=='SWIN':
    #    penalise_zeros = True
        
    loglikelihood = make_loss_func(train_pars,
                                   use_unseen_sites=train_pars.use_unseen_sites,
                                   penalise_zeros=penalise_zeros)

    ## load checkpoint
    model, optimizer, checkpoint = setup_checkpoint(
        model, optimizer, device, load_prev_chkpnt,
        model_outdir, log_dir,
        specify_chkpnt=specify_chkpnt,
        reset_chkpnt=reset_chkpnt
    )

    ## train
    model, losses, val_losses = fit(model,
                                    optimizer,
                                    loglikelihood,
                                    var,
                                    datgen,                                
                                    batch_prob_hourly,
                                    LR_scheduler=LR_scheduler,
                                    outdir=model_outdir,
                                    checkpoint=checkpoint,
                                    device=device)
                                    

