import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from pathlib import Path

from setupdata import data_generator
from utils import frange_cycle_linear, setup_checkpoint
from torch_model import MetVAE
from params import data_pars, model_pars, train_pars
from loss_funcs import make_loss_func
from step import fit

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
Ideas:
   - cross-attention between coarse ERA5 and coarsened fine inputs
    rather than concatenation?
   - use definition of density/value channel from the conditional
    neural processes paper, where a convolution is applied to
    [Mc, Yc], the pixel mask of context points Mc and the context
    values Yc = Mc . I, element wise product with the full image I.
    We don't have a full image so Yc is just the station values
    inserted into a zero image. The convolution takes the place of
    the kernel in the 1D form. h = conv([Mc, Yc]) where the first
    channel of h is the density channel. Then we pass h through a
    CNN, h_cnn = CNN(h).
   - Then we join this to the refined ERA5 data, separately to the other
    fine scale data? (Which NEVER gets joined and only has an effect 
    through cross attention?)   
'''

# file paths
log_dir = './logs/'
model_name = 'dwnsamp6'
model_outdir = f'{log_dir}/{model_name}/'
Path(model_outdir).mkdir(parents=True, exist_ok=True)

# training flags
load_prev_chkpnt = True
specify_chkpnt = f'{model_name}/checkpoint.pth' # if None, load best, otherwise "modelname/checkpoint.pth"
reset_chkpnt = False
flat_kl = False

## create data generator
datgen = data_generator() # creation of this gives log value warning
                              
## create model
model = MetVAE(input_channels=model_pars.input_channels,
               hires_fields=model_pars.hires_fields,
               output_channels=model_pars.output_channels,
               latent_variables=model_pars.latent_variables,
               filters=model_pars.filters,
               dropout_rate=model_pars.dropout_rate,
               scale=data_pars.scale,
               attn_heads=model_pars.attn_heads,
               d_cross_attn=model_pars.d_cross_attn,
               context_channels=model_pars.context_channels)
model.to(device)

## create optimizer, schedulers and loss function
optimizer = Adam(model.parameters(), lr=train_pars.lr)
LR_scheduler = ExponentialLR(optimizer, gamma=train_pars.gamma)
if not flat_kl:
    KLW_scheduler = frange_cycle_linear(train_pars.cycling_epochs, start=0.0, stop=1.0, n_cycle=4, ratio=0.6)
    KLW_scheduler = np.hstack([np.zeros(train_pars.warmup_epochs), KLW_scheduler])
    KLW_scheduler = np.hstack([KLW_scheduler, np.ones(train_pars.cooldown_epochs)])
else:
    KLW_scheduler = np.ones(train_pars.max_epochs)

#KLW_scheduler *= kl_weight_max
loglikelihood = make_loss_func(train_pars)

## load checkpoint
model, optimizer, checkpoint = setup_checkpoint(model, optimizer, device, load_prev_chkpnt,
                                                model_outdir, log_dir,
                                                specify_chkpnt=specify_chkpnt,
                                                reset_chkpnt=reset_chkpnt)

## train
model, losses, val_losses = fit(model, optimizer, loglikelihood, datgen,
                                train_pars, KLW_scheduler,
                                LR_scheduler=LR_scheduler,
                                null_batch_prob=0.25,
                                outdir=model_outdir,
                                checkpoint=checkpoint,
                                device=device)

