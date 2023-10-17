import numpy as np
import pandas as pd
import xarray as xr
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from pathlib import Path
from adjustText import adjust_text
from matplotlib.colors import Normalize

from setupdata import data_generator, Batch, make_mask_list
from utils import (frange_cycle_linear, setup_checkpoint, zeropad_strint,
                   trim, pooling, pool_4D_arr)
from params import data_pars, model_pars, train_pars
from loss_funcs import make_loss_func
from torch_model import MetVAE
from params import data_pars, model_pars, train_pars
from params import normalisation as nm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# file paths
log_dir = './logs/'
model_name = 'dwnsamp6'
model_outdir = f'{log_dir}/{model_name}/'
Path(model_outdir).mkdir(parents=True, exist_ok=True)

# training flags
load_prev_chkpnt = True
#specify_chkpnt = f'{model_name}/checkpoint.pth' # if None, load best, otherwise "modelname/checkpoint.pth"
specify_chkpnt = None
reset_chkpnt = False

## create data generator
datgen = data_generator()

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
loglikelihood = make_loss_func(train_pars)

## load checkpoint
model, opt, chk = setup_checkpoint(model, None, device, load_prev_chkpnt,
                                model_outdir, log_dir,
                                specify_chkpnt=specify_chkpnt,
                                reset_chkpnt=reset_chkpnt)
model.eval()

## testing on one batch of data
###############################
masks = make_mask_list(datgen.dim_l, datgen.dim_l, datgen.scale)

b = datgen.get_batch(batch_size=train_pars.batch_size,
                         batch_type='train',
                         load_binary_batch=True)
station_targets = b[2]
sample_metadata = b[7]
batch = Batch(b, masks, datgen.X1, device=device)

sea_mask = batch.fine_inputs[:,0:1,:,:].clone().cpu().numpy() # landfrac channel
sea_mask[sea_mask==0] = np.nan
# sea_mask = trim(batch.fine_inputs, data_pars.scale)[:,0:1,:,:] # landfrac channel

met_vars = datgen.coarse_variable_order
fine_vars = datgen.fine_variable_order
constraint_vars = datgen.constraint_variable_order

with torch.no_grad():
    vae_normal_dists = model.encode(batch.coarse_inputs, batch.fine_inputs,
                                    calc_kl=False, mask=None)
    pred = model.decode(vae_normal_dists, batch.fine_inputs, 
                        batch.context_data, batch.context_locs,
                        batch.grid_locs, masks=batch.masks,
                        context_mask=batch.context_mask)
    #grid_av_pred = torch.nn.functional.avg_pool2d(pred, data_pars.scale, stride=data_pars.scale)
    #grid_av_pred = grid_av_pred.cpu().numpy()
    pred = pred.cpu().numpy()


def to_np(arr):
    if type(arr)==type(torch.tensor(0)):
        return arr.cpu().numpy() 
    else:
        return arr


def plot_var_tiles(batch, pred, met_vars, constraint_vars,
                   b=0, v='TA', sea_mask=None, trim_edges=False, norm=None):
    if sea_mask is None:
        sea_mask = np.ones(pred.shape)
        sea_mask = sea_mask[:,0:1,:,:]
    
    nplots = 4
    ncols = 2
    nrows = 2    
    fig, axs = plt.subplots(nrows, ncols)
    
    idx_m = met_vars.index(v)
    try:
        idx_c = constraint_vars.index(v)
    except:
        idx_c = -99
    
    if not norm is None:
        cmap = cm.get_cmap('viridis')
        norm = Normalize(np.min(pred[b,idx_m,:,:]), np.max(pred[b,idx_m,:,:]))
        im = cm.ScalarMappable(norm=norm)

    # take grid average -- do this in a more intelligent way to ignore sea pixels?
    #grid_av_pred = torch.nn.functional.avg_pool2d(pred, data_pars.scale, stride=data_pars.scale)
    grid_av_pred = pool_4D_arr(pred, (data_pars.scale, data_pars.scale),
                               train_pars.batch_size, method='mean')        

    if trim_edges:
        toplot = trim(pred*sea_mask[b,:,:,:], data_pars.scale)[b,idx_m,:,:]
    else:
        toplot = (pred*sea_mask[b,:,:,:])[b,idx_m,:,:]
    axs[0,0].imshow(to_np(toplot)[::-1,:], cmap=cmap, norm=norm)
    axs[0,0].title.set_text(met_vars[idx_m])

    if idx_c != -99:
        if trim_edges:
            toplot = trim(batch.constraint_targets*sea_mask[b,:,:,:], data_pars.scale)[b,idx_c,:,:]
        else:
            toplot = (batch.constraint_targets*sea_mask[b,:,:,:])[b,idx_c,:,:]
        axs[0,1].imshow(to_np(toplot)[::-1,:], cmap=cmap, norm=norm)
        axs[0,1].title.set_text('Constraint')
    
        
    if trim_edges:
        toplot = trim(batch.coarse_inputs, 1)[b,idx_m,:,:]
    else:
        toplot = batch.coarse_inputs[b,idx_m,:,:]    
    axs[1,0].imshow(to_np(toplot)[::-1,:], cmap=cmap, norm=norm)
    axs[1,0].title.set_text('ERA5 input')
    
    
    if trim_edges:
        toplot = trim(grid_av_pred, 1)[b,idx_m,:,:]
    else:
        toplot = grid_av_pred[b,idx_m,:,:]    
    axs[1,1].imshow(to_np(toplot)[::-1,:], cmap=cmap, norm=norm)
    axs[1,1].title.set_text('Grid mean')
    
    for i in range(nrows):
        for j in range(ncols):
            axs[i,j].get_xaxis().set_visible(False)
            axs[i,j].get_yaxis().set_visible(False)
    
    if not norm is None:
        fig.colorbar(im, ax=axs.ravel().tolist())
    plt.show()

def plot_batch_tiles(batch, pred, fine_vars, met_vars,
                     b=0, sea_mask=None, trim_edges=False, norm=None):
    if sea_mask is None:
        sea_mask = np.ones(pred.shape)
        sea_mask = sea_mask[:,0:1,:,:]
    
    n_fine = batch.fine_inputs.shape[1]
    n_vars = pred.shape[1]
    nplots = n_fine + 3 * n_vars
    ncols = int(np.ceil(np.sqrt(nplots)))
    nrows = int(np.ceil(nplots / ncols))
    
    fig, axs = plt.subplots(nrows, ncols)
    cmap = cm.get_cmap('viridis')
    if not norm is None:
        norm = Normalize(-2.5, 2.5)
        im = cm.ScalarMappable(norm=norm)

    # take grid average -- do this in a more intelligent way to ignore sea pixels?
    #grid_av_pred = torch.nn.functional.avg_pool2d(pred, data_pars.scale, stride=data_pars.scale)
    grid_av_pred = pool_4D_arr(pred, (data_pars.scale, data_pars.scale),
                               train_pars.batch_size, method='mean')
    for i in range(n_fine):
        if trim_edges:
            toplot = trim(batch.fine_inputs, data_pars.scale)[b,i,:,:]
        else:
            toplot = batch.fine_inputs[b,i,:,:]
        #if type(toplot)==type(dummy_tensor): toplot.cpu().numpy()        
        axs[i//ncols, i%ncols].imshow(to_np(toplot)[::-1,:], cmap=cmap, norm=norm)
        axs[i//ncols, i%ncols].title.set_text(fine_vars[i])
    cx = i + 1
    for j in range(n_vars):
        if trim_edges:
            toplot = trim(pred*sea_mask[b,:,:,:], data_pars.scale)[b,j,:,:]
        else:
            toplot = (pred*sea_mask[b,:,:,:])[b,j,:,:]
        #if type(toplot)==type(dummy_tensor): toplot.cpu().numpy()        
        axs[(cx+j)//ncols, (cx+j)%ncols].imshow(to_np(toplot)[::-1,:], cmap=cmap, norm=norm)
        axs[(cx+j)//ncols, (cx+j)%ncols].title.set_text(met_vars[j])
    cx += j + 1
    for j in range(n_vars):
        if trim_edges:
            toplot = trim(batch.coarse_inputs, 1)[b,j,:,:]
        else:
            toplot = batch.coarse_inputs[b,j,:,:]
        #if type(toplot)==type(dummy_tensor): toplot.cpu().numpy()   
        axs[(cx+j)//ncols, (cx+j)%ncols].imshow(to_np(toplot)[::-1,:], cmap=cmap, norm=norm)
        axs[(cx+j)//ncols, (cx+j)%ncols].title.set_text('ERA5 ' + met_vars[j])
    cx += j + 1
    for j in range(n_vars):
        if trim_edges:
            toplot = trim(grid_av_pred, 1)[b,j,:,:]
        else:
            toplot = grid_av_pred[b,j,:,:]
        #if type(toplot)==type(dummy_tensor): toplot.cpu().numpy()  
        axs[(cx+j)//ncols, (cx+j)%ncols].imshow(to_np(toplot)[::-1,:], cmap=cmap, norm=norm)
        axs[(cx+j)//ncols, (cx+j)%ncols].title.set_text('Grid mean ' + met_vars[j])
    for i in range(nrows):
        for j in range(ncols):
            axs[i,j].get_xaxis().set_visible(False)
            axs[i,j].get_yaxis().set_visible(False)
    if not norm is None:
        fig.colorbar(im, ax=axs.ravel().tolist())
    plt.show()

def plot_preds_vs_site_data(preds, b, batch, station_targets,
                            site_type='context', model_names=None):
    if not type(preds)==list:
        preds = [preds]
    if model_names is None:
        model_names = [f'model_pred_{kk}' for kk in range(len(preds))]
    if site_type=='context':
        yx = to_np(batch.context_station_dict['coords_yx'][b])
    elif site_type=='target':
        yx = to_np(batch.target_station_dict['coords_yx'][b])
    nplots = yx.shape[0]
    ncols = int(np.ceil(np.sqrt(nplots)))
    nrows = int(np.ceil(nplots / ncols))
    fig, axs = plt.subplots(nrows, ncols)
    ii = 0
    jj = 0
    for i in range(yx.shape[0]):        
        site_preds = [pd.DataFrame(to_np(p)[b:(b+1),:, yx[i,0], yx[i,1]]) for p in preds]
        for sp in site_preds: sp.columns = met_vars
        if site_type=='context':
            site_obs = pd.DataFrame(to_np(batch.context_station_dict['values'][b])[i:(i+1),:])
        elif site_type=='target':
            site_obs = pd.DataFrame(to_np(batch.target_station_dict['values'][b])[i:(i+1),:])
        site_obs.columns = met_vars
        site_obs = site_obs.assign(SITE_ID = station_targets[b][site_type].index[i],
                                   dat_type='site_obs')
        site_preds = [sp.assign(SITE_ID = station_targets[b][site_type].index[i],
                                dat_type = model_names[kk]) for kk, sp in enumerate(site_preds)]
        print(site_obs)
        site_preds = [sp.melt(id_vars=['SITE_ID','dat_type']) for sp in site_preds]
        compare = pd.concat([site_obs.melt(id_vars=['SITE_ID','dat_type'])] + site_preds, axis=0)
        ii = i // ncols
        jj = i % ncols
        if i==0:
            sns.pointplot(x='variable', y='value', hue='dat_type',
                          join=False, data=compare, ax=axs[ii,jj],
                          dodge=True, scale=0.5).legend()
        else:
            sns.pointplot(x='variable', y='value', hue='dat_type',
                          join=False, data=compare, ax=axs[ii,jj],
                          dodge=True, scale=0.5)
        axs[ii,jj].set_title(station_targets[b][site_type].index[i])
        axs[ii,jj].set_xlabel('')
    # sort out legend
    handles, labels = axs[0,0].get_legend_handles_labels()
    [[c.get_legend().remove() for c in r if (not c.get_legend() is None)] for r in axs]
    fig.legend(handles, labels, loc='upper right')
    plt.show()

def extract_site_predictions(preds, station_targets, model_names=None):
    if not type(preds)==list:
        preds = [preds]
    if model_names is None:
        model_names = [f'model_pred_{kk}' for kk in len(preds)]
    df_out = pd.DataFrame()
    for b in range(preds[0].shape[0]):
        for site_type in ['context', 'target']:
            if site_type=='context':
                yx = to_np(batch.context_station_dict['coords_yx'][b])
            elif site_type=='target':
                yx = to_np(batch.target_station_dict['coords_yx'][b])        
            for i in range(yx.shape[0]):        
                site_preds = [pd.DataFrame(to_np(p)[b:(b+1),:, yx[i,0], yx[i,1]]) for p in preds]
                for sp in site_preds: sp.columns = met_vars
                if site_type=='context':
                    site_obs = pd.DataFrame(to_np(batch.context_station_dict['values'][b])[i:(i+1),:])
                elif site_type=='target':
                    site_obs = pd.DataFrame(to_np(batch.target_station_dict['values'][b])[i:(i+1),:])
                site_obs.columns = met_vars
                site_obs = site_obs.assign(SITE_ID = station_targets[b][site_type].index[i],
                                           dat_type='site_obs')
                site_preds = [sp.assign(SITE_ID = station_targets[b][site_type].index[i],
                                        dat_type = model_names[kk]) for kk, sp in enumerate(site_preds)]
                site_preds = [sp.melt(id_vars=['SITE_ID','dat_type']) for sp in site_preds]
                compare = pd.concat([site_obs.melt(id_vars=['SITE_ID','dat_type'])] + site_preds, axis=0)
                df_out = pd.concat([df_out, compare.assign(site_type = site_type)], axis=0)
    return df_out

def plot_station_locations(station_targets, fine_inputs, b):
    dim_h = data_pars.dim_l * data_pars.scale
    station_targets[b]['context']['adj_y'] = dim_h - station_targets[b]['context'].sub_y
    station_targets[b]['target']['adj_y'] = dim_h - station_targets[b]['target'].sub_y

    fig, ax = plt.subplots()
    ax.imshow(to_np(fine_inputs)[b, 0, ::-1, :], alpha=0.6, cmap='Greys')
    ax.scatter(station_targets[b]['context'].sub_x, station_targets[b]['context'].adj_y,
               s=18, c='#1f77b4', marker='s')
    texts = [plt.text(station_targets[b]['context'].sub_x.iloc[i],
                      station_targets[b]['context'].adj_y.iloc[i],
                      station_targets[b]['context'].index[i], fontsize=9) 
                        for i in range(station_targets[b]['context'].shape[0])]

    ax.scatter(station_targets[b]['target'].sub_x, station_targets[b]['target'].adj_y,
               s=18, c='#17becf', marker='o')
    texts += [plt.text(station_targets[b]['target'].sub_x.iloc[i],
                       station_targets[b]['target'].adj_y.iloc[i],
                       station_targets[b]['target'].index[i], fontsize=9) 
                            for i in range(station_targets[b]['target'].shape[0])]
    adjust_text(texts, arrowprops=dict(arrowstyle="-", color='k', lw=0.5))
    plt.axis('off')
    plt.show()

''' remove sites that are in Ireland! we don't have the 
ancilliary data there! '''

# look at tiles and station values
b = 1
plot_batch_tiles(batch, pred, fine_vars, met_vars, b=b,
                 sea_mask=sea_mask, trim_edges=True)

v = 'LWIN'
plot_var_tiles(batch, pred, met_vars, constraint_vars,
               b=b, v=v, sea_mask=sea_mask, trim_edges=True, norm=True)

plot_station_locations(station_targets, batch.fine_inputs, b)
plot_preds_vs_site_data(pred, b, batch, station_targets, site_type='context')
plot_preds_vs_site_data(pred, b, batch, station_targets, site_type='target')    

# run again but zero all the context inputs to see what happens without them
# i.e. add a single unmasked null tag to the context set
from step import nullify_station_data
batch2 = Batch(0,0,0)
batch2.copy(batch)
batch2 = nullify_station_data(batch2)
with torch.no_grad():
    vae_normal_dists2 = model.encode(batch2.coarse_inputs, batch2.fine_inputs,
                                    calc_kl=False, mask=None)
    pred2 = model.decode(vae_normal_dists2, batch2.fine_inputs, 
                        batch2.context_data, batch2.context_locs,
                        batch2.grid_locs, masks=batch2.masks,
                        context_mask=batch2.context_mask)
    pred2 = pred2.cpu().numpy()

# look at station values
b = 1
plot_preds_vs_site_data([pred, pred2], b, batch, station_targets, site_type='context')
plot_preds_vs_site_data([pred, pred2], b, batch, station_targets, site_type='target')

## should we be loading up the CHESS met data as comparison here?
# also test against UKV?
    
def load_process_chess(sample_metadata, datgen, b):
    ## load
    chessmet_dir = '/gws/nopw/j04/hydro_jules/data/uk/driving_data/chess/chess-met/daily/'
    chess_var_names = ['dtr', 'huss', 'precip', 'psurf', 
                       'rlds', 'rsds', 'sfcWind', 'tas']
                       
    datgen.var_name_map = datgen.var_name_map.assign(
    chess = np.array(['tas', 'psurf', 'rsds', 
                      'rlds', 'sfcWind', 'huss'])
    )    
    station_loc_ref = datgen.site_metadata[['SITE_ID', 'chess_y', 'chess_x']]
    
    year = sample_metadata[b]['timestamp'].year
    month = sample_metadata[b]['timestamp'].month
    day = sample_metadata[b]['timestamp'].day
    chess_dat = xr.open_mfdataset(chessmet_dir + \
        f'/chess-met_*_gb_1km_daily_{year}{zeropad_strint(month)}*.nc')

    ## rescale
    chess_dat.psurf.values = chess_dat.psurf.values / 100. # Pa to hPa
    chess_dat.huss.values = 100*(0.263 * chess_dat.psurf.values * chess_dat.huss.values * 
        np.exp((-17.67 * (chess_dat.tas.values - 273.15)) / (chess_dat.tas.values - 29.65))) # sphum to RH
    chess_dat.tas.values = chess_dat.tas.values - 273.15 # K to C

    ## normalise
    # incoming radiation                        
    chess_dat.rlds.values = (chess_dat.rlds.values - nm.lwin_mu) / nm.lwin_sd
    chess_dat.rsds.values = np.log(1. + chess_dat.rsds.values)
    chess_dat.rsds.values = (chess_dat.rsds.values - nm.logswin_mu) / nm.logswin_sd
    # air pressure
    chess_dat.psurf.values = (chess_dat.psurf.values - nm.p_mu) / nm.p_sd
    # relative humidity            
    chess_dat.huss.values = (chess_dat.huss.values - nm.rh_mu) / nm.rh_sd
    # temperature
    chess_dat.tas.values = (chess_dat.tas.values - nm.temp_mu) / nm.temp_sd
    # wind speed            
    chess_dat.sfcWind.values = (chess_dat.sfcWind.values - nm.ws_mu) / nm.ws_sd

    ## create "chess-pred"
    pred3 = np.zeros(pred.shape)
    for bb in range(len(sample_metadata)):    
        subdat = chess_dat.isel(y=sample_metadata[bb]['y_inds'],
                                x=sample_metadata[bb]['x_inds'])
        for j, cv in enumerate(datgen.coarse_variable_order):
            var = datgen.var_name_map[datgen.var_name_map['fine']==cv]['chess'].values[0]
            pred3[bb,j,:,:] = subdat[var].values[day-1,:,:]
    return pred3

b = 1
pred3 = load_process_chess(sample_metadata, datgen, b)
plot_preds_vs_site_data([pred, pred2, pred3], b, batch, station_targets, site_type='context')
plot_preds_vs_site_data([pred, pred2, pred3], b, batch, station_targets, site_type='target')

# site scatters
site_preds = extract_site_predictions([pred, pred2, pred3], station_targets,
                                      model_names=['model', 'model_no_context', 'chess'])
site_preds = site_preds.pivot_table(columns='dat_type', values='value',
                                    index=['SITE_ID', 'variable', 'site_type'])
sites = list(set([site_preds.index[i][0] for i in range(site_preds.shape[0])]))

nplots = len(datgen.coarse_variable_order)
ncols = int(np.ceil(np.sqrt(nplots)))
nrows = int(np.ceil(nplots / ncols))
fig, axs = plt.subplots(nrows, ncols)
for i,var in enumerate(datgen.coarse_variable_order):
    ii = i // ncols
    jj = i % ncols
    thisdat = site_preds.loc[(sites, var, 'context')]
    thisdat = thisdat.melt(id_vars='site_obs')
    sns.scatterplot(x='site_obs', y='value', hue='dat_type',
                    data=thisdat, ax=axs[ii,jj]).legend()
    axs[ii,jj].set_title(var)
    xx = np.mean(axs[ii,jj].get_xlim())
    axs[ii,jj].axline((xx,xx), slope=1, linestyle='--', color='k')
handles, labels = axs[0,0].get_legend_handles_labels()
[[c.get_legend().remove() for c in r if (not c.get_legend() is None)] for r in axs]
fig.legend(handles, labels, loc='upper right')    
plt.show()

import sklearn.metrics as metrics

def efficiencies(y_pred, y_true):
    alpha = np.std(y_pred) / np.std(y_true)    
    beta_nse = (np.mean(y_pred) - np.mean(y_true)) / np.std(y_true)
    beta_kge = np.mean(y_pred) / np.mean(y_true)
    rho = np.corrcoef(y_pred, y_true)[1,0]
    NSE = -beta_nse*beta_nse - alpha*alpha + 2*alpha*rho # Nash-Sutcliffe
    KGE = 1 - np.sqrt((beta_kge-1)**2 + (alpha-1)**2 + (rho-1)**2) # Kling-Gupta
    KGE_mod = 1 - np.sqrt(beta_nse**2 + (alpha-1)**2 + (rho-1)**2)
    LME = 1 - np.sqrt((beta_kge-1)**2 + (rho*alpha - 1)**2) # Liu-Mean
    LME_mod = 1 - np.sqrt(beta_nse**2 + (rho*alpha - 1)**2)
    return {'NSE':NSE, 'KGE':KGE, 'KGE_mod':KGE_mod, 'LME':LME, 'LME_mod':LME_mod}

def calc_metrics(df, var1, var2):
    effs = efficiencies(df[[var1, var2]].dropna()[var2],
                        df[[var1, var2]].dropna()[var1])
    effs['r2'] = metrics.r2_score(df[[var1, var2]].dropna()[var1],
                                  df[[var1, var2]].dropna()[var2])
    effs['mae'] = metrics.mean_absolute_error(df[[var1, var2]].dropna()[var1],
                                              df[[var1, var2]].dropna()[var2])
    effs['medae'] = metrics.median_absolute_error(df[[var1, var2]].dropna()[var1],
                                                  df[[var1, var2]].dropna()[var2])
    return effs


calc_metrics(site_preds, 'site_obs', 'chess')
calc_metrics(site_preds, 'site_obs', 'model')
calc_metrics(site_preds, 'site_obs', 'model_no_context')

fig,ax = plt.subplots(2,2)
b=1
v=3
ax[0,0].imshow(pred[b,v,:,:])
ax[0,1].imshow(pred2[b,v,:,:])
ax[1,0].imshow(pred3[b,v,:,:])
ax[1,1].imshow(batch.constraint_targets[b,v,:,:].cpu().numpy())
plt.show()

chess_av_pool = pool_4D_arr(pred3, (data_pars.scale, data_pars.scale),
                            train_pars.batch_size, method='mean')
fig,ax = plt.subplots(1,3)
b=2
v=2
ax[0].imshow(batch.coarse_inputs[b,v,:,:].cpu().numpy())
ax[1].imshow(chess_av_pool[b,v,:,:])
ax[2].imshow(pred3[b,v,:,:])
plt.show()

## do we want to plot a scatter of the grid averages (non-sea mask) pixels
## against the raw ERA5? and the fine pixels against the constraints?
grid_av_pred = pool_4D_arr(pred*sea_mask, (data_pars.scale, data_pars.scale),
                           train_pars.batch_size, method='mean')
grid_av_pred2 = pool_4D_arr(pred2*sea_mask, (data_pars.scale, data_pars.scale),
                           train_pars.batch_size, method='mean')
grid_av_pred3 = pool_4D_arr(pred3, (data_pars.scale, data_pars.scale),
                           train_pars.batch_size, method='mean')
coarse_df = pd.DataFrame()
for bb in range(grid_av_pred.shape[0]):    
    for i, var in enumerate(datgen.coarse_variable_order):
        this_var = pd.DataFrame()
        this_var['model'] = grid_av_pred[bb,i,:,:].flatten()
        this_var['model_no_context'] = grid_av_pred2[bb,i,:,:].flatten()
        this_var['chess'] = grid_av_pred3[bb,i,:,:].flatten()
        this_var['era5'] = batch.coarse_inputs[bb,i,:,:].flatten()
        this_var['variable'] = var
        this_var['batch'] = bb
        coarse_df = pd.concat([coarse_df, this_var], axis=0)

nplots = len(datgen.coarse_variable_order)
ncols = int(np.ceil(np.sqrt(nplots)))
nrows = int(np.ceil(nplots / ncols))
fig, axs = plt.subplots(nrows, ncols)
for i, var in enumerate(datgen.coarse_variable_order):
    ii = i // ncols
    jj = i % ncols
    thisdat = coarse_df[coarse_df['variable']==var].drop('variable', axis=1)
    thisdat = thisdat.melt(id_vars=['era5', 'batch'])
    sns.scatterplot(x='era5', y='value', hue='variable',
                    data=thisdat, ax=axs[ii,jj]).legend()
    axs[ii,jj].set_title(var)
    xx = np.mean(axs[ii,jj].get_xlim())
    axs[ii,jj].axline((xx,xx), slope=1, linestyle='--', color='k')
handles, labels = axs[0,0].get_legend_handles_labels()
[[c.get_legend().remove() for c in r if (not c.get_legend() is None)] for r in axs]
fig.legend(handles, labels, loc='upper right')    
plt.show()

calc_metrics(coarse_df, 'era5', 'chess')
calc_metrics(coarse_df, 'era5', 'model')
calc_metrics(coarse_df, 'era5', 'model_no_context')







### testing on a whole UK map batch
###############################
# create a data generator with a larger tile size
# tilesize = 16
# tile_overlap = 4
# discard_overlap = tile_overlap//2
# datgen = data_generator()

# get tile(s) of whole UK
# coarse_inputs, fine_inputs, ixs, iys = datgen.batch_all_space(
    # batch_type='train', load_binary_batch=True, min_overlap=4)

coarse_inputs, fine_inputs = datgen.get_all_space(batch_type='train',
                                                  load_binary_batch=True)
masks = make_mask_list(coarse_inputs.shape[-2], coarse_inputs.shape[-1],
                       datgen.scale, batch_size=coarse_inputs.shape[0])

max_batch_size = 4
ii = 0
preds = []
while ii<coarse_inputs.shape[0]:
    iinext = min(ii+max_batch_size, coarse_inputs.shape[0])
    with torch.no_grad():
        vae_normal_dists = model.encode(coarse_inputs[ii:iinext,...],
                                        fine_inputs[ii:iinext,...],
                                        calc_kl=False, mask=None)
        pred = model.decode(vae_normal_dists, fine_inputs[ii:iinext,...], masks=masks)
        pred = pred.cpu()
    preds.append(pred.cpu())
    del(pred)
    ii += max_batch_size
preds = torch.cat(preds, dim=0)
grid_av_preds = torch.nn.functional.avg_pool2d(preds, data_pars.scale, stride=data_pars.scale)
# could also mask out the ocean by multiplying ~landfrac by NaNs and then
# doing nan-aware pooling?

plot_batch_tiles(batch, pred, fine_vars, met_vars, b=b)

fine_vars = ['landfrac', 'elev', 'stdev', 'slope', 'aspect', 'rainfall',
             'year_sin', 'year_cos', 'hour_sin', 'hour_cos', 'lat', 'lon']
pre_valdense_len = len(fine_vars)
met_vars = ['TA', 'PA', 'SWIN', 'LWIN', 'WS', 'RH']
val_dense_chans = [m + s for m in met_vars for s in ['_value', '_density']]
fine_vars += val_dense_chans


n_fine = fine_inputs.shape[1]
n_vars = preds.shape[1]
nplots = n_fine + 3 * n_vars
ncols = int(np.ceil(np.sqrt(nplots)))
nrows = int(np.ceil(nplots / ncols))

b = 0
fig, axs = plt.subplots(nrows, ncols)
for i in range(n_fine):
    axs[i//ncols, i%ncols].imshow(fine_inputs.numpy()[b,i,::-1,:])
    axs[i//ncols, i%ncols].title.set_text(fine_vars[i])
cx = i + 1
for j in range(n_vars):    
    axs[(cx+j)//ncols, (cx+j)%ncols].imshow(preds.numpy()[b,j,::-1,:])
    axs[(cx+j)//ncols, (cx+j)%ncols].title.set_text(met_vars[j])
cx += j + 1
for j in range(n_vars):
    axs[(cx+j)//ncols, (cx+j)%ncols].imshow(coarse_inputs.numpy()[b,j,::-1,:])
    axs[(cx+j)//ncols, (cx+j)%ncols].title.set_text('ERA5 ' + met_vars[j])
cx += j + 1
for j in range(n_vars):    
    axs[(cx+j)//ncols, (cx+j)%ncols].imshow(grid_av_preds.numpy()[b,j,::-1,:])
    axs[(cx+j)//ncols, (cx+j)%ncols].title.set_text('Grid mean ' + met_vars[j])
for i in range(nrows):
    for j in range(ncols):
        axs[i,j].get_xaxis().set_visible(False)
        axs[i,j].get_yaxis().set_visible(False)
plt.show()


land_sea_mask_nans = fine_inputs.numpy()[b,0,::-1,:]
land_sea_mask_nans[land_sea_mask_nans==0] = np.nan

b = 0
fig, axs = plt.subplots(nrows, ncols)
for i in range(n_fine):
    axs[i//ncols, i%ncols].imshow(fine_inputs.numpy()[b,i,::-1,:])
    axs[i//ncols, i%ncols].title.set_text(fine_vars[i])
cx = i + 1
for j in range(n_vars):    
    axs[(cx+j)//ncols, (cx+j)%ncols].imshow(preds.numpy()[b,j,::-1,:] * land_sea_mask_nans)
    axs[(cx+j)//ncols, (cx+j)%ncols].title.set_text(met_vars[j])
cx += j + 1
for j in range(n_vars):
    axs[(cx+j)//ncols, (cx+j)%ncols].imshow(coarse_inputs.numpy()[b,j,::-1,:])
    axs[(cx+j)//ncols, (cx+j)%ncols].title.set_text('ERA5 ' + met_vars[j])
cx += j + 1
for j in range(n_vars):    
    axs[(cx+j)//ncols, (cx+j)%ncols].imshow(grid_av_preds.numpy()[b,j,::-1,:])
    axs[(cx+j)//ncols, (cx+j)%ncols].title.set_text('Grid mean ' + met_vars[j])
for i in range(nrows):
    for j in range(ncols):
        axs[i,j].get_xaxis().set_visible(False)
        axs[i,j].get_yaxis().set_visible(False)
# fig.subplots_adjust(right=0.8)
# cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
# fig.colorbar(ax, cax=cbar_ax)
plt.show()


v = 0
fig, ax = plt.subplots(1,3)
ax[0].imshow(coarse_inputs.numpy()[b,v,::-1,:])
ax[1].imshow(preds.numpy()[b,v,::-1,:] * fine_inputs.numpy()[b,0,::-1,:])
ax[2].imshow(grid_av_preds.numpy()[b,v,::-1,:])
ax[0].set_title(met_vars[v]+' ERA5 25km')
ax[1].set_title(met_vars[v]+' downscaled 1km')
ax[2].set_title(met_vars[v]+' grid average 1km->25km')
plt.show()


from mpl_toolkits.axes_grid1 import ImageGrid
v = 1
# Set up figure and image grid
fig = plt.figure(figsize=(9.75, 3))
grid = ImageGrid(fig, 111,          # as in plt.subplot(111)
                 nrows_ncols=(1,3),
                 axes_pad=0.15,
                 share_all=True,
                 cbar_location="right",
                 cbar_mode="single",
                 cbar_size="7%",
                 cbar_pad=0.15,
                 )
# Add data to image grid
for ax in grid:
    im = ax.imshow(np.random.random((10,10)), vmin=0, vmax=1)
im1 = grid.axes_all[0].imshow(coarse_inputs.numpy()[b,v,::-1,:])
im2 = grid.axes_all[1].imshow(preds.numpy()[b,v,::-1,:] * fine_inputs.numpy()[b,0,::-1,:])
im3 = grid.axes_all[2].imshow(grid_av_preds.numpy()[b,v,::-1,:])
# Colorbar
ax.cax.colorbar(im3)
ax.cax.toggle_label(True)
#plt.tight_layout()    # Works, but may still require rect paramater to keep colorbar labels visible
plt.show()


# first try to stack up the grid_av_preds as these are less cumbersome
def stack_map(data, iys, ixs, tile_size, ol=1):
    ystacks = []
    for k in range(len(ixs)):
        for j in range(len(iys)):
            kk = (k*len(iys))+j
            if j==0:                
                stacked_grid = data[kk,...][:,:(tile_size-ol),:]
            elif j<(len(iys)-1):                
                stacked_grid = torch.cat([stacked_grid,
                                          data[kk,...][:,ol:(tile_size-ol),:]], dim=1)
            else:
                pix_overlap = iys[j-1] + tile_size - iys[j] - ol
                stacked_grid = torch.cat([stacked_grid,
                                          data[kk,...][:,pix_overlap:,:]], dim=1)
        ystacks.append(torch.clone(stacked_grid))
        
    for j in range(len(ixs)):
        if j==0:
            full_map = ystacks[j][:,:,:(tile_size-ol)]
        elif j<(len(ixs)-1):
            full_map = torch.cat([full_map, ystacks[j][:,:,ol:(tile_size-ol)]], dim=2)
        else:
            pix_overlap = ixs[j-1] + tile_size - ixs[j] - ol
            full_map = torch.cat([full_map,
                                  ystacks[j][:,:,pix_overlap:]], dim=2)
    return full_map.numpy()[:,::-1,:]
    
ol = 1
full_map_grid_av_pred = stack_map(grid_av_preds, iys, ixs, datgen.dim_l, ol=ol)
full_map_preds = stack_map(preds,
                           [v*data_pars.scale for v in iys],
                           [v*data_pars.scale for v in ixs],
                           datgen.dim_h, ol=ol*data_pars.scale)
full_static_ins = stack_map(fine_inputs,
                            [v*data_pars.scale for v in iys],
                            [v*data_pars.scale for v in ixs],
                            datgen.dim_h, ol=ol*data_pars.scale)
full_coarse_ins = stack_map(coarse_inputs, iys, ixs, datgen.dim_l, ol=ol)

v = 0
fig, ax = plt.subplots(1,3)
ax[0].imshow(full_coarse_ins[v,:,:])
ax[1].imshow(full_map_preds[v,:,:]*full_static_ins[0])
ax[2].imshow(full_map_grid_av_pred[v,:,:])
ax[0].set_title(met_vars[v]+' ERA5 25km')
ax[1].set_title(met_vars[v]+' downscaled 1km')
ax[2].set_title(met_vars[v]+' grid average 1km->25km')
plt.show()

fig, ax = plt.subplots(3, len(met_vars))
for v in range(len(met_vars)):
    ax[0,v].imshow(full_coarse_ins[v,:,:])
    ax[1,v].imshow(full_map_preds[v,:,:])#*full_static_ins[0])
    ax[2,v].imshow(full_map_grid_av_pred[v,:,:])
    ax[0,v].set_title(met_vars[v]+' ERA5 25km')
    ax[1,v].set_title(met_vars[v]+' downscaled 1km')
    ax[2,v].set_title(met_vars[v]+' grid average 1km->25km')
plt.show()

'''
- test running larger tile sizes that dim_l = 4 to combat the tile 
    edge effect X
- test a batching of overlapping tiles where we only take the results
    from the inner pixels X
- can we train with larger tiles on Orchid compared to Lotus_gpu?
- look into how we treat the sea. Is the uniform noise ruining results?
- To combat tiling/edge effect:
    4x4 and 8x8 batches centred on same point/time, then central 2x2 
    subwindow result should be the same! Extra loss term?
- Constrain self attention to local patches using mask (attempt in layers.py)
    and possibly only at the intermediate spatial layer, not the central
    layer (which would share information across all pixels)
'''

