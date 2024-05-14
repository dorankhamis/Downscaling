import numpy as np
import pandas as pd
import torch
import argparse
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from pathlib import Path

from setupdata4 import data_generator, Batch
from utils2 import frange_cycle_linear, setup_checkpoint
from model3 import ConvDownscaler, Resolver
from params2 import data_pars, model_pars, train_pars
from loss_funcs3 import make_loss_func
from step4 import fit

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__=="__main__":
    var_names = ['TA', 'PA', 'SWIN', 'LWIN', 'WS', 'RH', 'PRECIP']
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("var_num", help = "variable to use")
        args = parser.parse_args()
        var = var_names[int(args.var_num)]
    except: # not running as batch script
        var = 'PRECIP'
    
    print('\nDevice: %s \n' % device)
    use_resolver_vars = ['SWIN']
    
    # file paths
    log_dir = '/home/users/doran/projects/downscaling/logs/'
    if var in use_resolver_vars:
        model_name = f'resolver_{var}'
    else:
        model_name = f'mergedwnsamp_{var}' # constraint/obs merge model
    model_outdir = f'{log_dir}/{model_name}/'
    Path(model_outdir).mkdir(parents=True, exist_ok=True)

    # training flags
    load_prev_chkpnt = True
    specify_chkpnt = f'{model_name}/checkpoint.pth' # if None, load best, otherwise "modelname/checkpoint.pth"
    reset_chkpnt = False

    ## create data generator
    dg = data_generator()
    if var=='PRECIP': dg.load_EA_rain_gauge_data()

    ## create model
    final_relu = True if var in ['SWIN', 'PRECIP'] else False
    if var in use_resolver_vars:
        model = Resolver(            
            hires_fields=len(model_pars.fine_variable_order[var]),
            output_channels=model_pars.output_channels[var],        
            filters=model_pars.resolver_filters,
            dropout_rate=0.02,
            final_relu=final_relu
        )
    else:
        model = ConvDownscaler(
            input_channels=model_pars.in_channels[var],
            hires_fields=len(model_pars.fine_variable_order[var]),
            output_channels=model_pars.output_channels[var],        
            filters=model_pars.filters,
            dropout_rate=model_pars.dropout_rate,
            scale=data_pars.scale,
            scale_factor=model_pars.scale_factor,
            final_relu=final_relu
        )

    model.to(device)   

    ## create optimizer, schedulers and loss function
    optimizer = Adam(model.parameters(), lr=train_pars.lr)
    LR_scheduler = ExponentialLR(optimizer, gamma=train_pars.gamma)
    
    batch_prob_hourly = np.ones((train_pars.warmup_epochs + 
                                 train_pars.increase_epochs + 
                                 train_pars.cooldown_epochs), dtype=np.float32)

    if False: #var=='PRECIP':# or var=='SWIN':
        penalise_zeros = True
    else:
        penalise_zeros = False
    loglikelihood = make_loss_func(train_pars,
                                   use_unseen_sites=train_pars.use_unseen_sites,
                                   penalise_zeros=penalise_zeros,
                                   var=var)

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
                                    dg,                                
                                    batch_prob_hourly,
                                    LR_scheduler=LR_scheduler,
                                    outdir=model_outdir,
                                    checkpoint=checkpoint,
                                    device=device)
                                    

