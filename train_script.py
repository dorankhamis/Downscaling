import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from pathlib import Path

from setupdata import data_generator
from step import fit, make_loss_func
from utils import frange_cycle_linear, setup_checkpoint
from torch_model import MetVAE
from params import data_pars, model_pars, train_pars

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# file paths
log_dir = './logs/'
model_name = 'dwnsamp'
model_outdir = f'{log_dir}/{model_name}/'
Path(model_outdir).mkdir(parents=True, exist_ok=True)

# training flags
load_prev_chkpnt = True
specify_chkpnt = f'{model_name}/checkpoint.pth' # if None, load best, otherwise "modelname/checkpoint.pth"
reset_chkpnt = False

## create data generator
datgen = data_generator(data_pars.train_years, data_pars.val_years, data_pars.heldout_years,
                        dim_l=data_pars.dim_l, res=data_pars.res, scale=data_pars.scale)
                              
## create model
model = MetVAE(input_channels=model_pars.input_channels,
               hires_fields=model_pars.hires_fields,
               output_channels=model_pars.output_channels,
               latent_variables=model_pars.latent_variables,
               filters=model_pars.filters,
               dropout_rate=model_pars.dropout_rate,
               scale=data_pars.scale)
model.to(device)

## create optimizer, schedulers and loss function
optimizer = Adam(model.parameters(), lr=train_pars.lr)
LR_scheduler = ExponentialLR(optimizer, gamma=train_pars.gamma)
KLW_scheduler = frange_cycle_linear(train_pars.max_epochs, start=0.0, stop=1.0, n_cycle=3, ratio=0.6)
KLW_scheduler = np.hstack([np.zeros(train_pars.warmup_epochs), KLW_scheduler])
#KLW_scheduler *= kl_weight_max
loglikelihood = make_loss_func(sigma_stations=train_pars.sigma_stations,
                               sigma_constraints=train_pars.sigma_constraints,
                               sigma_gridavg=train_pars.sigma_gridavg)

## load checkpoint
model, optimizer, checkpoint = setup_checkpoint(model, optimizer, device, load_prev_chkpnt,
                                                model_outdir, log_dir,
                                                specify_chkpnt=specify_chkpnt,
                                                reset_chkpnt=reset_chkpnt)

# train
model, losses, val_losses = fit(model, optimizer, loglikelihood, datgen, 
                                train_pars.train_len,
                                train_pars.val_len,
                                train_pars.batch_size,
                                KLW_scheduler,
                                outdir=model_outdir,
                                LR_scheduler=LR_scheduler,
                                max_epochs=train_pars.max_epochs,
                                warmup_epochs=train_pars.warmup_epochs,
                                checkpoint=checkpoint,
                                device=device)
