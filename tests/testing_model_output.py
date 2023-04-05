import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from pathlib import Path

from setupdata import data_generator, Batch
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
loglikelihood = make_loss_func(sigma_stations=train_pars.sigma_stations,
                               sigma_constraints=train_pars.sigma_constraints,
                               sigma_gridavg=train_pars.sigma_gridavg)

## load checkpoint
model, _, _ = setup_checkpoint(model, None, device, load_prev_chkpnt,
                                model_outdir, log_dir,
                                specify_chkpnt=specify_chkpnt,
                                reset_chkpnt=reset_chkpnt)
                   
model.eval()
batch = datgen.get_batch(batch_size=train_pars.batch_size,
                         batch_type='train',
                         load_binary_batch=True)
station_targets = batch[2]
batch = Batch(batch)

vae_normal_dists = model.encode(batch.coarse_inputs, batch.fine_inputs, calc_kl=False)
pred = model.decode(vae_normal_dists, batch.fine_inputs)
pred = pred.detach().cpu()
grid_av_pred = torch.nn.functional.avg_pool2d(pred, data_pars.scale, stride=data_pars.scale)

fine_vars = ['landfrac', 'elev', 'stdev', 'slope', 'aspect',
             'topi', 'stdtopi', 'fdepth', 'rainfall',
             'year_sin', 'year_cos', 'hour_sin', 'hour_cos']
met_vars = ['TA', 'PA', 'SWIN', 'LWIN', 'WS', 'RH']

n_fine = batch.fine_inputs.shape[1]
n_vars = pred.shape[1]
nplots = n_fine + 3 * n_vars
ncols = int(np.ceil(np.sqrt(nplots)))
nrows = int(np.ceil(nplots / ncols))

fig, axs = plt.subplots(nrows, ncols)
b = 0
for i in range(n_fine):
    axs[i//ncols, i%ncols].imshow(batch.fine_inputs.numpy()[b,i,:,:])
    axs[i//ncols, i%ncols].title.set_text(fine_vars[i])
cx = i + 1
for j in range(n_vars):    
    axs[(cx+j)//ncols, (cx+j)%ncols].imshow(pred.numpy()[b,j,:,:])
    axs[(cx+j)//ncols, (cx+j)%ncols].title.set_text(met_vars[j])
cx += j + 1
for j in range(n_vars):
    axs[(cx+j)//ncols, (cx+j)%ncols].imshow(batch.coarse_inputs.numpy()[b,j,:,:])
    axs[(cx+j)//ncols, (cx+j)%ncols].title.set_text('ERA5 ' + met_vars[j])
cx += j + 1
for j in range(n_vars):    
    axs[(cx+j)//ncols, (cx+j)%ncols].imshow(grid_av_pred.numpy()[b,j,:,:])
    axs[(cx+j)//ncols, (cx+j)%ncols].title.set_text('Grid mean ' + met_vars[j])
for i in range(nrows):
    for j in range(ncols):
        axs[i,j].get_xaxis().set_visible(False)
        axs[i,j].get_yaxis().set_visible(False)
plt.show()

## look at station values
for b in range(preds.shape[1]):
    yx = batch.station_dict['coords_yx'][b].numpy()
    for i in range(yx.shape[0]):        
        site_pred = pd.DataFrame(pred.numpy()[b:(b+1),:, yx[i,0], yx[i,1]])
        site_pred.columns = met_vars
        site_obs = pd.DataFrame(batch.station_dict['values'][b].numpy()[i:(i+1),:])
        site_obs.columns = met_vars
        site_obs = site_obs.assign(SITE_ID = station_targets[b].index[i], dat_type='site_obs')
        site_pred = site_pred.assign(SITE_ID = station_targets[b].index[i], dat_type='model_pred')
        print(site_obs)
        compare = pd.concat([site_obs.melt(id_vars=['SITE_ID','dat_type']),
                             site_pred.melt(id_vars=['SITE_ID','dat_type'])], axis=0)
        fig, ax = plt.subplots()
        sns.pointplot(x='variable', y='value', hue='dat_type',
                      join=False, data=compare, ax=ax).legend()
        plt.title(station_targets[b].index[i])
        plt.show()

