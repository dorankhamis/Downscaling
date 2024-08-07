import numpy as np
import pandas as pd
import xarray as xr
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from torch.optim import Adam
from pathlib import Path
from sklearn.metrics import r2_score

from setupdata3 import (data_generator, Batch, create_chess_pred,
                        load_process_chess, interp_to_grid, reflect_pad_nans)
from model2 import SimpleDownscaler
from params import data_pars, model_pars, train_pars
from loss_funcs2 import make_loss_func
from step3 import create_null_batch
from params import normalisation as nm
from utils import *
from plotting import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## create data generator
datgen = data_generator()

# file paths
#var_names = ['TA', 'PA', 'SWIN', 'LWIN', 'WS', 'RH', 'PRECIP']
var = 'SWIN'
log_dir = './logs/'
model_name = f'dwnsamp_{var}'
model_outdir = f'{log_dir}/{model_name}/'

# training flags
load_prev_chkpnt = True
# specify_chkpnt: if None, load best, otherwise "modelname/checkpoint.pth"
specify_chkpnt = f'{model_name}/checkpoint.pth'
reset_chkpnt = False

## create model              
model = SimpleDownscaler(
    input_channels=model_pars.in_channels[var],
    hires_fields=model_pars.hires_fields[var],
    output_channels=model_pars.output_channels[var],
    context_channels=model_pars.context_channels[var],
    filters=model_pars.filters,
    dropout_rate=model_pars.dropout_rate,
    scale=data_pars.scale,
    scale_factor=model_pars.scale_factor,
    attn_heads=model_pars.attn_heads,
    ds_cross_attn=model_pars.ds_cross_attn,
    pe=model_pars.pe
 )

model.to(device)

## create loss function
loglikelihood = make_loss_func(train_pars)

## load checkpoint
model, opt, chk = setup_checkpoint(model, None, device,
                                   load_prev_chkpnt,
                                   model_outdir, log_dir,
                                   specify_chkpnt=specify_chkpnt,
                                   reset_chkpnt=reset_chkpnt)
plt.plot(chk['losses'])
plt.plot(chk['val_losses'])
plt.show()

model.eval()

####################
var_names = ['TA', 'PA', 'SWIN', 'LWIN', 'WS', 'RH', 'PRECIP']
for var in var_names:
    ## dummy batch for model param fetching
    batch = datgen.get_batch(var, batch_size=1,
                             batch_type='train')
    print(var)
    print(f'input_channels = {batch["coarse_inputs"].shape[1]}')
    print(f'hires_fields = {batch["fine_inputs"].shape[1]}')
    print(f'output_channels = {batch["coarse_inputs"].shape[1]}')
    print(f'context_channels = {batch["station_data"][0].shape[1]}')
    print(f'coarse variable order = {batch["batch_metadata"][0]["coarse_var_order"]}')
    print(f'fine variable order = {batch["batch_metadata"][0]["fine_var_order"]}')
    print(f'context variable order = {batch["batch_metadata"][0]["context_var_order"]}')
    

############################
var_names = ['TA', 'PA', 'SWIN', 'LWIN', 'WS', 'RH', 'PRECIP']
for var in var_names:
    print(var)
    model_name = f'dwnsamp_{var}'
    model_outdir = f'{log_dir}/{model_name}/'
    specify_chkpnt = f'{model_name}/checkpoint.pth'
                                  
    ## create model              
    model = SimpleDownscaler(
        input_channels=model_pars.in_channels[var],
        hires_fields=model_pars.hires_fields[var],
        output_channels=model_pars.output_channels[var],
        context_channels=model_pars.context_channels[var],
        filters=model_pars.filters,
        dropout_rate=model_pars.dropout_rate,
        scale=data_pars.scale,
        scale_factor=model_pars.scale_factor,
        attn_heads=model_pars.attn_heads,
        ds_cross_attn=model_pars.ds_cross_attn,
        pe=model_pars.pe
     )

    ## load checkpoint
    model, opt, chk = setup_checkpoint(model, None, device, load_prev_chkpnt,
                                       model_outdir, log_dir,
                                       specify_chkpnt=specify_chkpnt,
                                       reset_chkpnt=reset_chkpnt)
    plt.plot(chk['losses'])
    plt.plot(chk['val_losses'])
    plt.title(var)
    plt.show()


###################################
## UK plots for all vars
var_names = ['TA', 'PA', 'SWIN', 'LWIN', 'WS', 'RH', 'PRECIP']
log_dir = './logs/'
load_prev_chkpnt = True
reset_chkpnt = False
PLOT = True
output_dir = '/home/users/doran/projects/downscaling/output/test_plots/'
Path(output_dir).mkdir(parents=True, exist_ok=True)
for var in var_names:
    model_name = f'dwnsamp_{var}'
    model_outdir = f'{log_dir}/{model_name}/'
    Path(model_outdir).mkdir(parents=True, exist_ok=True)
    
    if var=='PRECIP': datgen.load_EA_rain_gauge_data()

    #specify_chkpnt = None # f'{model_name}/checkpoint.pth' 
    specify_chkpnt = f'{model_name}/checkpoint.pth'

    ## dummy batch for model param fetching
    batch = datgen.get_batch(var, batch_size=train_pars.batch_size,
                             batch_type='train')
                                  
    ## create model              
    model = SimpleDownscaler(
        input_channels=model_pars.in_channels[var],
        hires_fields=model_pars.hires_fields[var],
        output_channels=model_pars.output_channels[var],
        context_channels=model_pars.context_channels[var],
        filters=model_pars.filters,
        dropout_rate=model_pars.dropout_rate,
        scale=data_pars.scale,
        scale_factor=model_pars.scale_factor,
        attn_heads=model_pars.attn_heads,
        ds_cross_attn=model_pars.ds_cross_attn,
        pe=model_pars.pe
     )

    del(batch)
    model.to(device)

    ## load checkpoint
    model, opt, chk = setup_checkpoint(model, None, device, load_prev_chkpnt,
                                       model_outdir, log_dir,
                                       specify_chkpnt=specify_chkpnt,
                                       reset_chkpnt=reset_chkpnt)
    model.eval()

    # get tile(s) of whole UK
    date_string = "20140809" #"20140101"
    it = 8 #10
    tile = False
    context_frac = 0.7
    constraints = True
    batch = datgen.get_all_space(var, batch_type='train',
                                 context_frac=context_frac,
                                 date_string=date_string, it=it,
                                 timestep='hourly',
                                 tile=tile,
                                 return_constraints=constraints)
    ixs = batch['ixs']
    iys = batch['iys']

    batch = Batch(batch, var_list=var, device=device, constraints=constraints)
    
    masks = create_attention_masks(model, batch, var,
                                   dist_lim = 30,
                                   poly_exp = None,
                                   diminish_model = None,
                                   dist_pixpass = 100,
                                   pass_exp = 6)
    
    if False:
        just_dist_masks = create_attention_masks(model, batch, "TA",
                                       dist_lim = None,
                                       dist_lim_far = None,
                                       attn_eps = None,
                                       poly_exp = None,
                                       diminish_model = None,
                                       dist_pixpass = None,
                                       pass_exp = None)
                                       
    # visualise_softmask_spatially(masks, level=0, b=0, site=0, exp=True)
    # visualise_pixelpasser(masks, level=0, b=0)

    station_targets = batch.raw_station_dict
    sample_metadata = batch.batch_metadata
    met_vars = datgen.coarse_variable_order
    fine_vars = datgen.fine_variable_order
    
    with torch.no_grad():
        pred = model(batch.coarse_inputs, batch.fine_inputs,
                     batch.context_data, batch.context_locs,
                     context_masks=masks['context_masks'],
                     context_soft_masks=masks['context_soft_masks'],
                     pixel_passers=masks['pixel_passers'])
        pred = pred.cpu().numpy()
    if (var=='SWIN') or (var=='PRECIP'): pred = pred.clip(min=0)
    
    if False:
        with torch.no_grad():
            pred_jd = model(batch.coarse_inputs, batch.fine_inputs,
                         batch.context_data, batch.context_locs,
                         context_masks=just_dist_masks['context_masks'],
                         context_soft_masks=just_dist_masks['context_soft_masks'],
                         pixel_passers=just_dist_masks['pixel_passers'])
            pred_jd = pred_jd.cpu().numpy()
        if (var=='SWIN') or (var=='PRECIP'): pred_jd = pred_jd.clip(min=0)
        era5 = batch.coarse_inputs.cpu().numpy()
        era5_unnorm = unnormalise_img(era5, var)
        pred_unnorm = unnormalise_img(pred, var)
        pred_jd_unnorm = unnormalise_img(pred_jd, var)

        cmap = cm.get_cmap('plasma')
        norm = Normalize(np.min(pred_unnorm), np.max(pred_unnorm))
        im = cm.ScalarMappable(cmap=cmap, norm=norm)        
        fig, ax = plt.subplots(1,3)
        ax[0].imshow(era5_unnorm[0,0,::-1,:], cmap=cmap, norm=norm)
        ax[2].imshow(pred_unnorm[0,0,::-1,:], cmap=cmap, norm=norm)
        ax[1].imshow(pred_jd_unnorm[0,0,::-1,:], cmap=cmap, norm=norm)
        ax[0].set_title(f'ERA5 {var}')
        ax[2].set_title(f'Downscaled {var}, intelligent masking')
        ax[1].set_title(f'Downscaled {var}, distance masking')
        fig.colorbar(im, ax=ax.ravel().tolist())
        plt.show()
        
    
    #np.save(output_dir + f'coarse_inputs_{var}_{date_string}_{it}.npy', batch.coarse_inputs.cpu().numpy())
    #np.save(output_dir + f'prediction_{var}_{date_string}_{it}.npy', pred)


    if PLOT:
        era5 = batch.coarse_inputs.cpu().numpy()
        if var=='WS':
            #era5_ws_unnorm = unnormalise_img(datgen.parent_pixels['hourly'].isel(time=it).ws.values, 'WS')
            era5_unnorm = unnormalise_img(era5, 'UX')
            pred_unnorm = unnormalise_img(pred, 'UX')
        else:
            era5_unnorm = unnormalise_img(era5, var)
            pred_unnorm = unnormalise_img(pred, var)

        cmap = cm.get_cmap('plasma')
        norm = Normalize(np.min(pred_unnorm), np.max(pred_unnorm))
        im = cm.ScalarMappable(cmap=cmap, norm=norm)
        if var=='WS':
            fig, ax = plt.subplots(2,2)
            ax[0,0].imshow(era5_unnorm[0,0,::-1,:], cmap=cmap, norm=norm)
            ax[0,1].imshow(pred_unnorm[0,0,::-1,:], cmap=cmap, norm=norm)
            ax[1,0].imshow(era5_unnorm[0,1,::-1,:], cmap=cmap, norm=norm)
            ax[1,1].imshow(pred_unnorm[0,1,::-1,:], cmap=cmap, norm=norm)
            ax[0,0].set_title(f'ERA5 UX')
            ax[0,1].set_title(f'Downscaled UX')
            ax[1,0].set_title(f'ERA5 VY')
            ax[1,1].set_title(f'Downscaled VY')
        else:
            fig, ax = plt.subplots(1,2)
            ax[0].imshow(era5_unnorm[0,0,::-1,:], cmap=cmap, norm=norm)
            ax[1].imshow(pred_unnorm[0,0,::-1,:], cmap=cmap, norm=norm)
            ax[0].set_title(f'ERA5 {var}')
            ax[1].set_title(f'Downscaled {var}')
        fig.colorbar(im, ax=ax.ravel().tolist())
        plt.show()
        
        if constraints:
            constraint_unnorm = unnormalise_img(batch.constraint_targets.detach().numpy(), var)

            fig, ax = plt.subplots(1,3)
            cmap = cm.get_cmap('plasma')
            norm = Normalize(np.min(pred_unnorm), np.max(pred_unnorm))
            im = cm.ScalarMappable(cmap=cmap, norm=norm)
            ax[0].imshow(era5_unnorm[0,0,::-1,:], cmap=cmap, norm=norm)
            ax[1].imshow(pred_unnorm[0,0,::-1,:], cmap=cmap, norm=norm)
            ax[2].imshow(constraint_unnorm[0,0,::-1,:], cmap=cmap, norm=norm)
            ax[0].set_title(f'ERA5 {var}')
            ax[1].set_title(f'Downscaled {var}')
            ax[2].set_title(f'Hi-res constraint {var}')
            fig.colorbar(im, ax=ax.ravel().tolist())
            plt.show()
        
        plot_batch_tiles(batch, pred, fine_vars, met_vars, b=0,
                         sea_mask=None, trim_edges=False, constraints=constraints)
        '''
        cloud cover attention might be causing unnatural peaks in SWIN
        '''

    # run again but zero all the context inputs to see what happens without them
    batch2 = create_null_batch(batch)
    with torch.no_grad():
        pred2 = model(batch2.coarse_inputs, batch2.fine_inputs,
                      batch2.context_data, batch2.context_locs)
        pred2 = pred2.cpu().numpy()
    if (var=='SWIN') or (var=='PRECIP'): pred2 = pred2.clip(min=0)

    if PLOT:
        era5 = batch.coarse_inputs.cpu().numpy()
        if var=='WS':
            #era5_ws_unnorm = unnormalise_img(datgen.parent_pixels['hourly'].isel(time=it).ws.values, 'WS')
            era5_unnorm = unnormalise_img(era5, 'UX')
            pred_unnorm = unnormalise_img(pred, 'UX')
            pred2_unnorm = unnormalise_img(pred2, 'UX')
        else:
            era5_unnorm = unnormalise_img(era5, var)
            pred_unnorm = unnormalise_img(pred, var)
            pred2_unnorm = unnormalise_img(pred2, var)

        cmap = cm.get_cmap('plasma')
        norm = Normalize(np.min(pred_unnorm), np.max(pred_unnorm))
        im = cm.ScalarMappable(cmap=cmap, norm=norm)
        if var=='WS':
            fig, ax = plt.subplots(2,3)
            ax[0,0].imshow(era5_unnorm[0,0,::-1,:], cmap=cmap, norm=norm)
            ax[0,1].imshow(pred2_unnorm[0,0,::-1,:], cmap=cmap, norm=norm)
            ax[0,2].imshow(pred_unnorm[0,0,::-1,:], cmap=cmap, norm=norm)
            ax[1,0].imshow(era5_unnorm[0,1,::-1,:], cmap=cmap, norm=norm)
            ax[1,1].imshow(pred2_unnorm[0,1,::-1,:], cmap=cmap, norm=norm)
            ax[1,2].imshow(pred_unnorm[0,1,::-1,:], cmap=cmap, norm=norm)
            ax[0,0].set_title(f'ERA5 UX')
            ax[0,1].set_title(f'Downscaled UX nc')
            ax[0,2].set_title(f'Downscaled UX')
            ax[1,0].set_title(f'ERA5 VY')
            ax[1,1].set_title(f'Downscaled VY nc')
            ax[1,2].set_title(f'Downscaled VY')
        else:
            fig, ax = plt.subplots(1,3)        
            ax[0].imshow(era5_unnorm[0,0,::-1,:], cmap=cmap, norm=norm)
            ax[1].imshow(pred2_unnorm[0,0,::-1,:], cmap=cmap, norm=norm)
            ax[2].imshow(pred_unnorm[0,0,::-1,:], cmap=cmap, norm=norm)
            ax[0].set_title(f'ERA5 {var}')
            ax[1].set_title(f'Downscaled_nc {var}')
            ax[2].set_title(f'Downscaled {var}')
        fig.colorbar(im, ax=ax.ravel().tolist())
        plt.show()
        
        b = 0
        plot_batch_tiles(batch, pred, fine_vars, met_vars, b=b,
                         sea_mask=None, trim_edges=False, constraints=False)
        plot_batch_tiles(batch, pred2, fine_vars, met_vars, b=b,
                         sea_mask=None, trim_edges=False, constraints=False)

    ## look at station values
    if PLOT:
        b = 0          
        site_res = plot_context_and_target_preds(
            [pred, pred2], b,
            batch, station_targets, var, 
            model_names=['this_model', 'this_model_nc']
        )
        
        print(calc_metrics(site_res['target']
            .pivot(columns='variable', index='SITE_ID', values='value'),
             var, 'this_model_nc')
        )
        print(calc_metrics(site_res['target']
            .pivot(columns='variable', index='SITE_ID', values='value'),
             var, 'this_model')
        )
        
        print(calc_metrics(site_res['context']
            .pivot(columns='variable', index='SITE_ID', values='value'),
            var, 'this_model_nc')
        )
        print(calc_metrics(site_res['context']
            .pivot(columns='variable', index='SITE_ID', values='value'),
            var, 'this_model')
        )

        plot_station_locations(batch.raw_station_dict, batch.fine_inputs, b,
                               plot_target=True, labels=False)





# diminish_mask = distances > model_pars.dist_lim
# softmask = np.ones(distances.shape, dtype=np.float32)
# softmask2 = np.ones(distances.shape, dtype=np.float32)
# #if diminish_model=="gaussian":
# attn_sigmasq = -(model_pars.dist_lim_far - model_pars.dist_lim)**2 / np.log(model_pars.attn_eps)
# softmask[diminish_mask] = np.exp(- (distances[diminish_mask] - model_pars.dist_lim)**2 / attn_sigmasq)
# #elif diminish_model=="polynomial":
# softmask2[diminish_mask] = 1. / (distances[diminish_mask] / model_pars.dist_lim)**6#model_pars.poly_exp
# plt.scatter(softmask[0,1,:,:].flatten(), softmask2[0,1,:,:].flatten())
# plt.show()




#################
## visualising intelligent masking
if type(var)==str: var = [var]    
dist_lim = model_pars.dist_lim[var[0]]
dist_lim_far = model_pars.dist_lim_far
attn_eps=model_pars.attn_eps
poly_exp = model_pars.poly_exp[var[0]]
diminish_model = model_pars.diminish_model
dist_pixpass = model_pars.dist_pixpass[var[0]]
pass_exp = model_pars.pass_exp[var[0]]

attn_masks = {
    'context_soft_masks':[None, None, None, None],
    'pixel_passers':     [None, None, None, None],
    'context_masks':     [None, None, None, None]
}
b = 0
site_meta = (pd.concat([batch.raw_station_dict[b]['context'],
                        batch.raw_station_dict[b]['target']], axis=0)
    .reset_index()
)
distances, softmask, scale_factors, site_yx, cntxt_stats = prepare_attn(
    model,
    batch,
    site_meta,
    None,
    context_sites=None,
    b=b,
    dist_lim=dist_lim,
    dist_lim_far=dist_lim_far,
    attn_eps=attn_eps,
    poly_exp=poly_exp,
    diminish_model=diminish_model
)

cntxt_stats=cntxt_stats
fine_inputs=batch.fine_inputs
fine_variable_order=batch.batch_metadata[b]['fine_var_order']

if type(var)==list: var=var[0]
# context_soft_masks = [softmask]
raw_context_masks = [torch.clone(softmask)]
for i, sf in enumerate(scale_factors):
    raw_context_masks.append(nn.functional.interpolate(
        raw_context_masks[i],
        scale_factor=sf,
        mode='bilinear')
    )
    # context_soft_masks.append(nn.functional.interpolate(
        # context_soft_masks[i],
        # scale_factor=sf,
        # mode='bilinear')
    # )

# apply shade / cloud affect on high-resolution soft masking
if var=='SWIN' or var=='LWIN' or var=='PRECIP':
    cloud_ind = fine_variable_order.index('cloud_cover')
    if var=='SWIN':
        shade_ind = fine_variable_order.index('shade_map')
        illum_ind = fine_variable_order.index('illumination_map')

    for i in range(cntxt_stats.shape[0]):
        if var=='SWIN':
            solar_dist = (
                (fine_inputs[:,cloud_ind,:,:] - cntxt_stats.iloc[i].cloud_cover).square() + 
                (fine_inputs[:,shade_ind,:,:] - cntxt_stats.iloc[i].shade_map).square() + 
                (fine_inputs[:,illum_ind,:,:] - cntxt_stats.iloc[i].illumination_map).square()
            ).sqrt()
        else:
            solar_dist = (fine_inputs[:,cloud_ind,:,:] - cntxt_stats.iloc[i].cloud_cover).abs()
        
        context_soft_masks[-1][:,i,:,:] = context_soft_masks[-1][:,i,:,:] - 5.*solar_dist
        
        # interpolate for lower res masks
        solar_dists_sf = []
        for j in range(2, len(scale_factors)+2):
            context_soft_masks[-j][:,i:(i+1),:,:] = (context_soft_masks[-j][:,i:(i+1),:,:] -
                5 * nn.functional.interpolate(
                    solar_dist[None,...],
                    size=context_soft_masks[-j][0,i,:,:].shape,
                    mode='bilinear'
                )
            )
        
    ## visualise
    fig, ax = plt.subplots(1,2, sharex=True, sharey=True)
    j=3
    i=44
    ax[0].imshow(context_soft_masks[j].detach().numpy()[0,i,::-1,:], cmap='plasma')
    ax[1].imshow(raw_context_masks[j].detach().numpy()[0,i,::-1,:], cmap='plasma')
    plt.show()
    
    fig, ax = plt.subplots(1,2, sharex=True, sharey=True)
    j=3
    i=45
    ax[0].imshow(np.exp(raw_context_masks[j].detach().numpy()[0,i,::-1,:]), cmap='plasma')
    ax[1].imshow(np.exp(context_soft_masks[j].detach().numpy()[0,i,::-1,:]), cmap='plasma')    
    plt.show()




################


## loading and visualising
var = 'RH'
era5 = np.load(f'./output/test_plots/coarse_inputs_{var}_20170101_10.npy')
pred = np.load(f'./output/test_plots/prediction_{var}_20170101_10.npy')

era5 = unnormalise_img(era5, var)
pred = unnormalise_img(pred, var)

#if var=='SWIN' or var=='LWIN' or var=='WS':
#   pred = pred.clip(min=0)

fig, ax = plt.subplots(1,2)
cmap = cm.get_cmap('plasma')
norm = Normalize(np.min(pred), np.max(pred))
im = cm.ScalarMappable(cmap=cmap, norm=norm)
ax[0].imshow(era5[0,0,::-1,:], cmap=cmap, norm=norm)
ax[1].imshow(pred[0,0,::-1,:], cmap=cmap, norm=norm)
ax[0].set_title(f'ERA5 {var}')
ax[1].set_title(f'Downscaled {var}')
fig.colorbar(im, ax=ax.ravel().tolist())
plt.show()

# compare against simply interpolated ERA5 at site pixels?
pred_era5 = interp_to_grid(datgen.parent_pixels['hourly'][datgen.var_name_map.loc[var].coarse][it,:,:], datgen.fine_grid, coords=['lat', 'lon'])
pred_era5 = reflect_pad_nans(pred_era5)
pred_era5 = pred_era5.values[None, None, ...]

plot_context_and_target_preds(
    [pred, pred2, pred_era5], b,
    batch, station_targets, var, 
    model_names=['this_model', 'this_model_nc', 'era5_interp']
)

## site scatters (entire batch)
site_preds = extract_site_predictions([pred, pred2, pred_era5], station_targets, batch, [var],
                                      model_names=['this_model', 'this_model_nc', 'era5_interp'])
site_preds = site_preds.pivot_table(columns='dat_type', values='value',
                                    index=['SITE_ID', 'variable', 'site_type'])
sites = list(set([site_preds.index[i][0] for i in range(site_preds.shape[0])]))

mm = 'NSE'
print(calc_metrics(site_preds.loc[:,:,'context'], 'site_obs', 'this_model_nc')[mm])
print(calc_metrics(site_preds.loc[:,:,'context'], 'site_obs', 'this_model')[mm])
print(calc_metrics(site_preds.loc[:,:,'context'], 'site_obs', 'era5_interp')[mm])

print(calc_metrics(site_preds.loc[:,:,'target'], 'site_obs', 'this_model_nc')[mm])
print(calc_metrics(site_preds.loc[:,:,'target'], 'site_obs', 'this_model')[mm])
print(calc_metrics(site_preds.loc[:,:,'target'], 'site_obs', 'era5_interp')[mm])

fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10,5))
for i, point_type in enumerate(['context', 'target']):
    thisdat = site_preds.loc[(sites, var, point_type)]
    thisdat = thisdat.melt(id_vars='site_obs')    
    sns.scatterplot(x='site_obs', y='value', hue='dat_type',
                    data=thisdat, ax=axs[i], s=15).legend()
    axs[i].set_title(point_type)
    xx = np.mean(axs[i].get_xlim())
    axs[i].axline((xx,xx), slope=1, linestyle='--', color='k')
plt.show()





## testing on one batch of data
###############################
p_hourly = 1
context_frac = 0.7
batch = datgen.get_batch(var, batch_size=train_pars.batch_size,
                       batch_type='train',
                       context_frac=context_frac,
                       p_hourly=p_hourly)
                           
batch = Batch(batch, var_list=var, device=device)
station_targets = batch.raw_station_dict
sample_metadata = batch.batch_metadata
met_vars = datgen.coarse_variable_order
fine_vars = datgen.fine_variable_order

sea_mask = batch.fine_inputs[:,0:1,:,:].clone().cpu().numpy() # landfrac channel
sea_mask[sea_mask==0] = np.nan

mask_sites = True
if mask_sites:
    # create the flat YX grid for attention
    raw_H = batch.fine_inputs.shape[-2]
    raw_W = batch.fine_inputs.shape[-1]
    masks = []
    
    for b in range(len(batch.raw_station_dict)):
        # build context masks   
        site_yx = off_grid_site_lat_lon(batch.raw_station_dict[b]['context'].LATITUDE,
                                        batch.raw_station_dict[b]['context'].LONGITUDE,
                                        batch.fine_inputs[b,-2,:,:].cpu().numpy() * nm.lat_norm,
                                        batch.fine_inputs[b,-1,:,:].cpu().numpy() * nm.lon_norm,
                                        datgen.X1, raw_H, raw_W)

        masks_b = build_context_masks(
            site_yx, datgen.X1,
            batch.coarse_inputs.shape[-2:],
            batch.fine_inputs.shape[-2:],
            model, device, 
            dist_lim=25,
            dist_lim_far=100, # dist limits in 1km pixels
            attn_eps=1e-8,
            binary_masks=False,
            soft_masks=True,
            pixel_pass_mask=False
        )
        masks.append(masks_b)

if mask_sites:
    ii = 0
    pred = []
    max_batch_size = 1
    while ii<batch.coarse_inputs.shape[0]:
        iinext = min(ii+max_batch_size, batch.coarse_inputs.shape[0])
        with torch.no_grad():
            out = model(batch.coarse_inputs[ii:iinext,...],
                        batch.fine_inputs[ii:iinext,...],
                        batch.context_data[ii:iinext],
                        batch.context_locs[ii:iinext],
                        context_soft_masks=masks[ii]['context_soft_masks'])
                        #context_masks=batch_elem_masks)        
        pred.append(out.cpu())
        del(out)    
        ii += max_batch_size
    pred = torch.cat(pred, dim=0).numpy()
else:
    with torch.no_grad():    
        pred = model(batch.coarse_inputs, batch.fine_inputs,
                      batch.context_data, batch.context_locs)
        pred = pred.cpu().numpy()

# run again but zero all the context inputs to see what happens without them
batch2 = create_null_batch(batch)
with torch.no_grad():
    pred2 = model(batch2.coarse_inputs, batch2.fine_inputs,
                  batch2.context_data, batch2.context_locs)
    pred2 = pred2.cpu().numpy()

# also create a chess batch for comparison
#pred3 = create_chess_pred(sample_metadata, datgen, var, pred)

## look at tiles and station values
b = 0
plot_batch_tiles(batch, pred, fine_vars, met_vars, b=b,
                 sea_mask=sea_mask, trim_edges=True)
plot_context_and_target_preds(
    [pred, pred2], b,
    batch, station_targets, var, 
    model_names=['this_model', 'this_model_nc']
)

plot_station_locations(station_targets, batch.fine_inputs, b)

plot_context_and_target_preds(
    [pred, pred2, pred3], b,
    batch, station_targets, var, 
    model_names=['this_model', 'this_model_no_point_obs', 'CHESS']
)

## grid scatters
bsize = batch.coarse_inputs.shape[0]
fig, ax = plt.subplots(bsize,3, sharex=True, sharey=True, figsize=(14,3.5*bsize))
for b in range(bsize):
    ax[b,0].scatter(to_np(trim(pred*sea_mask[b,:,:,:], data_pars.scale)[b,0,:,:]).flatten(),
                  to_np(trim(pred2*sea_mask[b,:,:,:], data_pars.scale)[b,0,:,:]).flatten(),
                  alpha=1, s=1.5)
    ax[b,1].scatter(to_np(trim(pred*sea_mask[b,:,:,:], data_pars.scale)[b,0,:,:]).flatten(),
                  to_np(trim(pred3*sea_mask[b,:,:,:], data_pars.scale)[b,0,:,:]).flatten(),
                  alpha=1, s=1.5)
    ax[b,2].scatter(to_np(trim(pred2*sea_mask[b,:,:,:], data_pars.scale)[b,0,:,:]).flatten(),
                  to_np(trim(pred3*sea_mask[b,:,:,:], data_pars.scale)[b,0,:,:]).flatten(),
                  alpha=1, s=1.5)
    if b==0:
        ax[b,0].set_title('model vs model_no_point_obs')
        ax[b,1].set_title('model vs chess')
        ax[b,2].set_title('model_no_point_obs vs chess')
for aaa in ax:
    for aa in aaa:
        xx = np.mean(aa.get_xlim())
        aa.axline((xx,xx), slope=1, linestyle='--', color='k')
plt.show()

## site scatters (entire batch)
site_preds = extract_site_predictions([pred, pred2, pred3], station_targets, batch, [var],
                                      model_names=['model', 'model_no_point_obs', 'chess'])
site_preds = site_preds.pivot_table(columns='dat_type', values='value',
                                    index=['SITE_ID', 'variable', 'site_type'])
sites = list(set([site_preds.index[i][0] for i in range(site_preds.shape[0])]))

fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10,5))
for i, point_type in enumerate(['context', 'target']):
    thisdat = site_preds.loc[(sites, var, point_type)]
    thisdat = thisdat.melt(id_vars='site_obs')    
    sns.scatterplot(x='site_obs', y='value', hue='dat_type',
                    data=thisdat, ax=axs[i], s=15).legend()
    axs[i].set_title(point_type)
    xx = np.mean(axs[i].get_xlim())
    axs[i].axline((xx,xx), slope=1, linestyle='--', color='k')
plt.show()

calc_metrics(site_preds, 'site_obs', 'chess')
calc_metrics(site_preds, 'site_obs', 'model')
calc_metrics(site_preds, 'site_obs', 'model_no_point_obs')

if False:
    ## other plots
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

if False:
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

if False:
    # plot all sites
    datgen.site_metadata['adj_y'] = datgen.fine_grid.landfrac.shape[-2] - datgen.site_metadata['chess_y']
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(datgen.fine_grid.landfrac.values[::-1, :], alpha=0.5, cmap='Greys')
    ax[0].scatter(datgen.site_metadata[datgen.site_metadata['SITE_NAME'].isna()].chess_x,
                  datgen.site_metadata[datgen.site_metadata['SITE_NAME'].isna()].adj_y,
                  s=7, c='#1f77b4', marker='o')
    ax[1].imshow(datgen.fine_grid.landfrac.values[::-1, :], alpha=0.5, cmap='Greys')
    ax[1].scatter(datgen.site_metadata[~datgen.site_metadata['SITE_NAME'].isna()].chess_x,
                  datgen.site_metadata[~datgen.site_metadata['SITE_NAME'].isna()].adj_y,
                  s=7, c='#550080', marker='s')
    ax[0].axis('off')
    ax[1].axis('off')
    plt.show()


################################
if False:
    ## testing era5 swin versus solar elevation
    date_string = "20170415"
    era5_fldr = '/gws/nopw/j04/hydro_jules/data/uk/driving_data/era5/28km_grid/'
    era5_vars = ['msdwswrf']
    era5_filelist = [f'{era5_fldr}/{v}/era5_{date_string}_{v}.nc' for v in era5_vars]
    era5_dat = xr.open_mfdataset(era5_filelist)

    nz_train_path = '/gws/nopw/j04/ceh_generic/netzero/downscaling/training_data/'
    doy = pd.to_datetime(date_string, format='%Y%m%d', utc=True).day_of_year - 1 # zero indexed
    shading_array = np.load(
        nz_train_path + f'/terrain_shading/shading_mask_day_{doy}_merged_to_7999.npy')

    yy = 24; xx = 9
    sol_elev = []
    for hh in range(24):
        sp = SolarPosition(pd.to_datetime(date_string + f' {zeropad_strint(hh)}:00:00', format='%Y%m%d %H:%M:%S', utc=True), timezone=0) # utc
        zea = sp.calc_solar_angles_return(dg.coarse_grid.lat.values[yy, xx],
                                          dg.coarse_grid.lon.values[yy, xx])
        sol_elev.append(zea.solar_elevation)
    sol_elev = np.array(sol_elev)

    shade_map = (dg.fine_grid.landfrac.copy()).astype(np.float32)
    shade_map.values[:,:] = 1
    shade_vals = []
    for hh in range(24):
        shade_map.values[dg.fine_grid.landfrac.values==1] = -(shading_array[hh,:]-1)
        shade_map_c = shade_map.interp_like(dg.coarse_grid)
        shade_vals.append(shade_map_c.values[yy,xx])

    fig, ax = plt.subplots()
    ax.plot(era5_dat.msdwswrf.values[:,yy,xx] / np.max(era5_dat.msdwswrf.values[:,yy,xx]), '.')
    ax.plot(sol_elev / np.max(sol_elev))
    ax.plot(shade_vals)
    plt.show()



### testing on a whole UK map batch
###############################
# get tile(s) of whole UK
date_string = "20170101"
it = 5
tile = False
context_frac = 0.7
batch = datgen.get_all_space(var, batch_type='train',
                             context_frac=context_frac,
                             date_string=date_string, it=it,
                             timestep='hourly',
                             tile=tile)
ixs = batch['ixs']
iys = batch['iys']

batch = Batch(batch, var_list=var, device=device, constraints=False)
station_targets = batch.raw_station_dict
sample_metadata = batch.batch_metadata
met_vars = datgen.coarse_variable_order
fine_vars = datgen.fine_variable_order

sea_mask = batch.fine_inputs[:,0:1,:,:].clone().cpu().numpy() # landfrac channel
sea_mask[sea_mask==0] = np.nan

'''
should mask site locations that are "far" from grid points.
Too many context locs is dragging results down to the average.

Perhaps we can route individual pixels through the "no context points"
branch of the model by judging distance from closest point
or clever attention masking, then checking if whole row is False and
skipping pixel?
'''
   
# create the flat YX grid for attention
end_sizes = data_pars.scale * (np.array(datgen.fine_grid.landfrac.shape)//data_pars.scale)
raw_H = end_sizes[0] # == fine_inputs.shape[-2]
raw_W = end_sizes[1] # == fine_inputs.shape[-1]
X1 = np.where(np.ones((raw_H, raw_W)))
X1 = np.hstack([X1[0][...,np.newaxis],
                X1[1][...,np.newaxis]])

# build context masks for all possible context sites
cntxt_stats = datgen.site_metadata.set_index('SITE_ID').loc[batch.raw_station_dict[0]['context'].index]
cntxt_stats = cntxt_stats.assign(s_idx = np.arange(cntxt_stats.shape[0]))

site_yx = off_grid_site_lat_lon(cntxt_stats.LATITUDE,
                                cntxt_stats.LONGITUDE,
                                datgen.fine_grid.lat.values[:raw_H, :raw_W],
                                datgen.fine_grid.lon.values[:raw_H, :raw_W],
                                X1, raw_H, raw_W)


# context_masks = build_context_masks(site_yx, X1,
                                    # batch.coarse_inputs.shape[-2:],
                                    # batch.fine_inputs.shape[-2:],
                                    # model, device, dist_lim=dist_lim)
                                    
## creating "soft masks" for context sites
masks = build_context_masks(
    site_yx, X1,
    batch.coarse_inputs.shape[-2:],
    batch.fine_inputs.shape[-2:],
    model, device, 
    dist_lim=100, dist_lim_far=400,
    attn_eps=1e-8, binary_masks=False,
    soft_masks=True, pixel_pass_mask=False
)

with torch.no_grad():
    pred = model(batch.coarse_inputs, batch.fine_inputs,
                 batch.context_data, batch.context_locs,
                 context_soft_masks=masks['context_soft_masks'])
                 #pixel_passer=pixel_passer)
                 #context_masks=context_masks,
                 #context_soft_masks=context_soft_masks)
    pred = pred.cpu().numpy()

b = 0
plot_batch_tiles(batch, pred, fine_vars, met_vars, b=b,
                 sea_mask=None, trim_edges=False, constraints=False)

plot_station_locations(station_targets, batch.fine_inputs, b, labels=False)

plot_context_and_target_preds(
    [pred], b,
    batch, station_targets, var, 
    model_names=['this_model']
)

# compare against simply interpolated ERA5 at site pixels?
pred_era5 = datgen.parent_pixels['hourly'][datgen.var_name_map.loc[var].coarse][it,:,:].interp_like(datgen.fine_grid)
pred_era5 = pred_era5.values[None, None, ...]
plot_context_and_target_preds(
    [pred, pred_era5], b,
    batch, station_targets, var, 
    model_names=['this_model', 'era5_interp']
)


# run again but zero all the context inputs to see what happens without them
batch2 = create_null_batch(batch, constraints=False)
if tile:
    max_batch_size = 6
    ii = 0
    pred2 = []
    while ii<batch2.coarse_inputs.shape[0]:
        iinext = min(ii+max_batch_size, batch2.coarse_inputs.shape[0])
        with torch.no_grad():
            out = model(batch2.coarse_inputs[ii:iinext,...],
                        batch2.fine_inputs[ii:iinext,...],
                        batch2.context_data[ii:iinext],
                        batch2.context_locs[ii:iinext])        
        pred2.append(out.cpu())
        del(out)
        ii += max_batch_size
    pred2 = torch.cat(pred2, dim=0).numpy()
else:
    with torch.no_grad():
        pred2 = model(batch2.coarse_inputs, batch2.fine_inputs,
                      batch2.context_data, batch2.context_locs)
        pred2 = pred2.cpu().numpy()

plot_context_and_target_preds(
    [pred, pred2], b,
    batch, station_targets, var, 
    model_names=['this_model', 'this_model_nc']
)


## site scatters (entire batch)
site_preds = extract_site_predictions([pred, pred2], station_targets, batch, [var],
                                      model_names=['model', 'model_no_point_obs'])
site_preds = site_preds.pivot_table(columns='dat_type', values='value',
                                    index=['SITE_ID', 'variable', 'site_type'])
sites = list(set([site_preds.index[i][0] for i in range(site_preds.shape[0])]))

fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10,5))
for i, point_type in enumerate(['context', 'target']):
    thisdat = site_preds.loc[(sites, var, point_type)]
    thisdat = thisdat.melt(id_vars='site_obs')    
    sns.scatterplot(x='site_obs', y='value', hue='dat_type',
                    data=thisdat, ax=axs[i], s=15).legend()
    axs[i].set_title(point_type)
    xx = np.mean(axs[i].get_xlim())
    axs[i].axline((xx,xx), slope=1, linestyle='--', color='k')
plt.show()

print(calc_metrics(site_preds, 'site_obs', 'model'))
print(calc_metrics(site_preds, 'site_obs', 'model_no_point_obs'))

## grid scatter
fig, ax = plt.subplots(1, figsize=(7, 7))
ax.scatter(to_np(trim(pred*sea_mask[b,:,:,:], data_pars.scale)[b,0,:,:]).flatten(),
              to_np(trim(pred2*sea_mask[b,:,:,:], data_pars.scale)[b,0,:,:]).flatten(),
              alpha=1, s=1.5)    
ax.set_title('model vs model_no_point_obs')
xx = np.mean(ax.get_xlim())
ax.axline((xx,xx), slope=1, linestyle='--', color='k')
plt.show()





''' do whole day and take average -> compare with chess? '''
date_string = "20170523"
it = list(np.arange(24))
tile = False
min_overlap = 3
batch = datgen.get_all_space(var, batch_type='train',
                             context_frac=context_frac,                             
                             date_string=date_string, it=it,
                             timestep='hourly',
                             tile=tile,
                             min_overlap=min_overlap)
ixs = batch['ixs']
iys = batch['iys']

batch = Batch(batch, var_list=var, device=device, constraints=False)
station_targets = batch.raw_station_dict
sample_metadata = batch.batch_metadata
met_vars = datgen.coarse_variable_order
fine_vars = datgen.fine_variable_order

sea_mask = batch.fine_inputs[0,0:1,:,:].clone().cpu().numpy() # landfrac channel
sea_mask[sea_mask==0] = np.nan

max_batch_size = 1
ii = 0
pred = []
while ii<batch.coarse_inputs.shape[0]:
    iinext = min(ii+max_batch_size, batch.coarse_inputs.shape[0])
    with torch.no_grad():
        out = model(batch.coarse_inputs[ii:iinext,...],
                    batch.fine_inputs[ii:iinext,...],
                    batch.context_data[ii:iinext],
                    batch.context_locs[ii:iinext])        
    pred.append(out.cpu())
    del(out)
    ii += max_batch_size
pred = torch.cat(pred, dim=0).numpy()
pred_dayav = pred.mean(axis=0)

# run again but zero all the context inputs to see what happens without them
batch2 = create_null_batch(batch, constraints=False)
ii = 0
pred2 = []
while ii<batch2.coarse_inputs.shape[0]:
    iinext = min(ii+max_batch_size, batch2.coarse_inputs.shape[0])
    with torch.no_grad():
        out = model(batch2.coarse_inputs[ii:iinext,...],
                    batch2.fine_inputs[ii:iinext,...],
                    batch2.context_data[ii:iinext],
                    batch2.context_locs[ii:iinext])        
    pred2.append(out.cpu())
    del(out)
    ii += max_batch_size
pred2 = torch.cat(pred2, dim=0).numpy()
pred2_dayav = pred2.mean(axis=0)

# # also create a chess batch for comparison
# chess_var = datgen.var_name_map.loc[var].chess
# pred3 = load_process_chess(datgen.td.year, datgen.td.month, datgen.td.day,
                           # var=chess_var, normalise=True)
# pred3 = pred3[chess_var].values[:(-(len(pred3.y) - pred.shape[-2])),
                                # :(-(len(pred3.x) - pred.shape[-1]))]
# fig, ax = plt.subplots(1,2)
# ax[0].imshow((pred_dayav * sea_mask)[0,::-1,:])
# ax[1].imshow(pred3[::-1,:])
# plt.show()

daily_data = datgen.site_metadata[
    ['SITE_ID', 'EASTING', 'NORTHING', 'LATITUDE', 'LONGITUDE',
     'ALTITUDE', 'chess_y', 'chess_x', 'parent_pixel_id']]
day_av = []
for sid in daily_data.SITE_ID:
    try:
        if datgen.site_points_present[sid].loc[datgen.td, var]==24:
            day_av.append(datgen.daily_site_data[sid].loc[datgen.td, var])
        else:
            day_av.append(np.nan)
    except:
        day_av.append(np.nan)
daily_data[var] = day_av

site_preds_d = pred_dayav[0,daily_data.chess_y.values, daily_data.chess_x.values]
site_preds2_d = pred2_dayav[0,daily_data.chess_y.values, daily_data.chess_x.values]
site_preds_chess = pred3[daily_data.chess_y.values, daily_data.chess_x.values]
site_obs = daily_data[var]

a = pd.DataFrame({'obs':site_obs, 'preds':site_preds_chess}).dropna()
metrics.r2_score(a.obs, a.preds)

a = pd.DataFrame({'obs':site_obs, 'preds':site_preds_d}).dropna()
metrics.r2_score(a.obs, a.preds)

a = pd.DataFrame({'obs':site_obs, 'preds':site_preds2_d}).dropna()
metrics.r2_score(a.obs, a.preds)

#################################
## plots for presentation
station_targets[b]['context']['adj_y'] = fine_inputs.shape[-2] - station_targets[b]['context'].sub_y
station_targets[b]['target']['adj_y'] = fine_inputs.shape[-2] - station_targets[b]['target'].sub_y

fig, ax = plt.subplots()
ax.imshow(to_np(fine_inputs)[b, 0, ::-1, :], alpha=0.6, cmap='Greys')
ax.scatter(station_targets[b]['context'].sub_x, station_targets[b]['context'].adj_y,
           s=18, c='#1f77b4', marker='s')
ax.scatter(station_targets[b]['target'].sub_x, station_targets[b]['target'].adj_y,
           s=18, c='#17becf', marker='o')
if labels:
    texts = [plt.text(station_targets[b]['context'].sub_x.iloc[i],
                      station_targets[b]['context'].adj_y.iloc[i],
                      station_targets[b]['context'].index[i], fontsize=9) 
                        for i in range(station_targets[b]['context'].shape[0])]
    texts += [plt.text(station_targets[b]['target'].sub_x.iloc[i],
                       station_targets[b]['target'].adj_y.iloc[i],
                       station_targets[b]['target'].index[i], fontsize=9) 
                            for i in range(station_targets[b]['target'].shape[0])]
    adjust_text(texts, arrowprops=dict(arrowstyle="-", color='k', lw=0.5))
plt.axis('off')
plt.show()




# fig, ax = plt.subplots(3, 4, sharex=True, sharey=True)
# ax[0,0].imshow(pred[0,0,::-1,:])
# ax[0,1].imshow(batch.fine_inputs[0,0,:,:].cpu().numpy()[::-1,:])
# ax[0,2].imshow(batch.fine_inputs[0,1,:,:].cpu().numpy()[::-1,:])
# ax[0,3].imshow(batch.fine_inputs[0,2,:,:].cpu().numpy()[::-1,:])
# ax[1,0].imshow(batch.fine_inputs[0,3,:,:].cpu().numpy()[::-1,:])
# ax[1,1].imshow(batch.fine_inputs[0,4,:,:].cpu().numpy()[::-1,:])
# ax[1,2].imshow(batch.fine_inputs[0,5,:,:].cpu().numpy()[::-1,:])
# ax[1,3].imshow(batch.fine_inputs[0,6,:,:].cpu().numpy()[::-1,:])
# ax[2,0].imshow(batch.fine_inputs[0,7,:,:].cpu().numpy()[::-1,:])
# ax[2,1].imshow(batch.fine_inputs[0,8,:,:].cpu().numpy()[::-1,:])
# plt.show()

########################
## testing timing of batch generation

# def get_batch(self, var, batch_size=1, batch_type='train',
          # context_frac=None, p_hourly=1,
          # date=None, parent_pixel_ids=[], times=[]):
''' 
when using date, parent_pixel_ids, times to select batches,
date must be a datetime.date object
parent_pixel_ids must be list of integers
times must be a list of integers in {0,23}
'''
import time
var = 'SWIN'
batch_size=1
batch_type='train'    
context_frac=None
p_hourly=1
date=None
parent_pixel_ids=[]
times=[]

ta0 = time.time()
if not date is None: date_string = f'{date.year}{zeropad_strint(date.month)}{zeropad_strint(date.day)}'
if parent_pixel_ids is None: parent_pixel_ids = []
if times is None: times = []
if type(parent_pixel_ids)==int: parent_pixel_ids = [parent_pixel_ids]
if type(times)==int: times = [times]        
if type(var)==str: var = [var]
ta1 = time.time()
ta = ta1-ta0
''' quick '''

tb0 = time.time()
datgen.parent_pixels = {}
if date is None:
    datgen.parent_pixels['hourly'] = datgen.read_parent_pixel_day(batch_type=batch_type)
else:                
    datgen.parent_pixels['hourly'] = datgen.read_parent_pixel_day(batch_type=batch_type,
                                                              date_string=date_string)
    datgen.td = pd.to_datetime(datgen.parent_pixels['hourly'].time[0].values, utc=True)
tb1 = time.time()
tb = tb1-tb0
''' quick '''

tc0 = time.time()
## work out the timesteps of the batch        
batch_timesteps = np.random.choice(['daily', 'hourly'], batch_size,
                                   p=[1-p_hourly, p_hourly])
if ('daily' in batch_timesteps) and ('hourly' in batch_timesteps):
    batch_tsteps = 'mix'
elif batch_timesteps[0]=='daily':
    batch_tsteps = 'daily'
else:
    batch_tsteps = 'hourly'

if (batch_tsteps!='hourly'):
    ## load chess data (don't normalise in case using in constraint preparation)
    datgen.chess_var = list(datgen.var_name_map.loc[var].chess)
    datgen.chess_dat = load_process_chess(datgen.td.year, datgen.td.month, datgen.td.day,
                                        datgen.chess_var, normalise=False)
tc1 = time.time()
tc = tc1-tc0
''' quick '''

td0 = time.time()
## process and normalise ERA5 parent_pixels
datgen.prepare_era5_pixels(var, batch_tsteps=batch_tsteps)
td1 = time.time()
td = td1-td0
''' slowest '''

#prepare_era5_pixels:
constraints=True
if batch_tsteps=='daily':
    ## average over the day
    datgen.parent_pixels['daily'] = datgen.parent_pixels['hourly'].mean('time')
    datgen.parent_pixels['hourly'] = None
    tstep = 'daily'
elif batch_tsteps=='hourly' or batch_tsteps=='mix':
    tstep = 'hourly'
    
## convert wind vectors and dewpoint temp to wind speed and relative humidity        
datgen.parent_pixels[tstep]['rh'] = relhum_from_dewpoint(
    datgen.parent_pixels[tstep]['t2m'] - 273.15,
    datgen.parent_pixels[tstep]['d2m'] - 273.15
)
datgen.parent_pixels[tstep]['ws'] = (np.sqrt(
    np.square(datgen.parent_pixels[tstep]['v10']) +
    np.square(datgen.parent_pixels[tstep]['u10']))
)        

td0a = time.time()
if batch_tsteps!='daily':
    if constraints:
        # this requires particular units - do before changing them!
        datgen.prepare_constraints(var)
td1a = time.time()
tda = td1a-td0a
''' slowest! '''

## load other large grids that we only wan to load once per batch
td0b = time.time()
if 'SWIN' in var:
    # terrain shading used in hourly AND daily
    doy = datgen.td.day_of_year - 1 # zero indexed
    # datgen.shading_array = np.load(
        # nz_train_path + f'/terrain_shading/shading_mask_day_{doy}_merged_to_7999.npy')
    # datgen.sea_shading_array = np.load(
        # nz_train_path + f'/terrain_shading/sea_shading_mask_day_{doy}_merged_to_7999.npy')
    datgen.shading_ds = xr.open_dataset(nz_train_path + f'./terrain_shading/shading_mask_{doy}.nc')
    datgen.shading_ds = datgen.shading_ds.shading
        
if ('SWIN' in var) or ('LWIN' in var) or ('PRECIP' in var) :
    # cloud cover
    tcc = xr.open_dataset(era5_fldr + f'/tcc/era5_{datgen.td.year}{zeropad_strint(datgen.td.month)}{zeropad_strint(datgen.td.day)}_tcc.nc')
    datgen.tcc_1km = interp_to_grid(tcc.tcc, datgen.fine_grid, coords=['lat', 'lon'])
    datgen.tcc_1km = reflect_pad_nans(datgen.tcc_1km.copy())
td1b = time.time()
tdb = td1b-td0b

td0c = time.time()
if ('PRECIP' in var):
    datgen.gear = xr.open_dataset(precip_fldr + f'/{datgen.td.year}/CEH-GEAR-1hr-v2_{datgen.td.year}{zeropad_strint(datgen.td.month)}.nc')
    # label centre of pixel
    datgen.gear['x'] = datgen.gear.x + 500
    datgen.gear['y'] = datgen.gear.y + 500
    # subset to domain
    datgen.gear = datgen.gear.sel(
        x = np.intersect1d(datgen.gear.x, datgen.fine_grid.x),
        y = np.intersect1d(datgen.gear.y, datgen.fine_grid.y)
    )
    # subset to day
    day_inds = np.where(pd.to_datetime(datgen.gear.time.values, utc=True).day == datgen.td.day)[0]
    datgen.gear = datgen.gear.isel(time = day_inds)
td1c = time.time()
tdc = td1c-td0c


td0d = time.time()
## change units
datgen.parent_pixels[tstep]['t2m'].values -= 273.15 # K -> Celsius 
datgen.parent_pixels[tstep]['d2m'].values -= 273.15 # K -> Celsius
datgen.parent_pixels[tstep]['sp'].values /= 100.    # Pa -> hPa
datgen.parent_pixels[tstep]['mtpr'].values *= 3600. # kg m2 s-1 -> mm in hour

#datgen.parent_pixels[tstep] = datgen.parent_pixels[tstep].drop(['u10', 'v10', 'd2m'])
datgen.parent_pixels[tstep] = datgen.parent_pixels[tstep].drop(['d2m'])

## normalise ERA5        
datgen.parent_pixels[tstep]['t2m'].values = (datgen.parent_pixels[tstep]['t2m'].values - nm.temp_mu) / nm.temp_sd        
datgen.parent_pixels[tstep]['sp'].values = (datgen.parent_pixels[tstep]['sp'].values - nm.p_mu) / nm.p_sd        
datgen.parent_pixels[tstep]['msdwlwrf'].values = (datgen.parent_pixels[tstep]['msdwlwrf'].values - nm.lwin_mu) / nm.lwin_sd        
datgen.parent_pixels[tstep]['msdwswrf'].values = datgen.parent_pixels[tstep]['msdwswrf'].values / nm.swin_norm
datgen.parent_pixels[tstep]['ws'].values = (datgen.parent_pixels[tstep]['ws'].values - nm.ws_mu) / nm.ws_sd
datgen.parent_pixels[tstep]['u10'].values = (datgen.parent_pixels[tstep]['u10'].values) / nm.ws_sd
datgen.parent_pixels[tstep]['v10'].values = (datgen.parent_pixels[tstep]['v10'].values) / nm.ws_sd
datgen.parent_pixels[tstep]['rh'].values = (datgen.parent_pixels[tstep]['rh'].values - nm.rh_mu) / nm.rh_sd
datgen.parent_pixels[tstep]['mtpr'].values = datgen.parent_pixels[tstep]['mtpr'].values / nm.precip_norm

datgen.parent_pixels[tstep]['msdwswrf'] = datgen.parent_pixels[tstep]['msdwswrf'].clip(min=0)
datgen.parent_pixels[tstep]['mtpr'] = datgen.parent_pixels[tstep]['mtpr'].clip(min=0)
td1d = time.time()
tdd = td1d-td0d

td0e = time.time()
## drop variables we don't need
datgen.era5_var = list(datgen.var_name_map.loc[var].coarse)
to_drop = list(datgen.parent_pixels[tstep].keys())
for v in datgen.era5_var: to_drop.remove(v)
#to_drop.remove('pixel_id') # not in era5 data anymore, moved to datgen.coarse_grid
if 'RH' in var:
    for exvar in datgen.rh_extra_met_vars:
        cvar = datgen.var_name_map.loc[exvar].coarse
        if cvar in to_drop:
            to_drop.remove(cvar)
if 'LWIN' in var:
    for exvar in datgen.lwin_extra_met_vars:
        cvar = datgen.var_name_map.loc[exvar].coarse
        if cvar in to_drop:
            to_drop.remove(cvar)
if 'SWIN' in var:
    for exvar in datgen.swin_extra_met_vars:
        cvar = datgen.var_name_map.loc[exvar].coarse
        if cvar in to_drop:
            to_drop.remove(cvar)
if 'PRECIP' in var:
    for exvar in datgen.precip_extra_met_vars:
        cvar = datgen.var_name_map.loc[exvar].coarse
        if cvar in to_drop:
            to_drop.remove(cvar)
if 'WS' in var:
    to_drop.remove('u10')
    to_drop.remove('v10')
    
datgen.parent_pixels[tstep] = datgen.parent_pixels[tstep].drop(to_drop)
td1e = time.time()
tde = td1e-td0e

td0f = time.time()
# load data
datgen.parent_pixels[tstep].load()
td1f = time.time()
tdf = td1f-td0f

if batch_tsteps=='mix':
    # do time-averaging for daily AFTER processing if mixed batch            
    datgen.parent_pixels['daily'] = datgen.parent_pixels['hourly'].mean('time')   


te0 = time.time()
## get additional fine scale met vars for input
datgen.prepare_fine_met_inputs(var, batch_tsteps)
te1 = time.time()
te = te1-te0
''' slow '''

if batch_tsteps!='hourly': # cover daily and mix
    # pool to scale and fill in the parent pixels
    for v in var:
        cv = datgen.var_name_map.loc[v].chess
        ev = datgen.var_name_map.loc[v].coarse
        chess_pooled = pooling(datgen.chess_dat[cv].values, (datgen.scale, datgen.scale), method='mean')
        chess_mask = ~np.isnan(chess_pooled)
        datgen.parent_pixels['daily'][ev].values[chess_mask] = chess_pooled[chess_mask]   
    
    # solar values for daily timestep
    datgen.calculate_day_av_solar_vars(var)

coarse_inputs = []
fine_inputs = []
station_targets = []
station_data = []
station_num_obs = []
constraint_targets = []
context_locations = []
batch_metadata = []
for b in range(batch_size):
    # try to select the parent pixel id and hour
    if b <= (len(parent_pixel_ids)-1):
        ppid = parent_pixel_ids[b]
    else:
        ppid = None
    if b <= (len(times)-1): # assuming entire batch is hourly samples
        _it = times[b]
    else:
        _it = None
    
    # generate batch from parent pixels
    # sample = datgen.get_sample(var,
                             # batch_type=batch_type,
                             # context_frac=context_frac,
                             # timestep=batch_timesteps[b],
                             # parent_pixel_id=ppid,
                             # it=_it)
       # def get_sample(self, var, batch_type='train',
                   # context_frac=None, 
                   # timestep='hourly',
                   # sample_xyt=True,
                   # ix=None, iy=None, it=None,
                   # return_constraints=True,
                   # SID=None, parent_pixel_id=None):

    batch_type=batch_type
    context_frac=context_frac
    timestep=batch_timesteps[b]
    parent_pixel_id=ppid
    it=_it
    sample_xyt=True
    ix=None
    iy=None
    it=None
    return_constraints=True
    SID=None
    parent_pixel_id=None
    
    tf0 = time.time()
    if sample_xyt:
        ## sample a dim_l x dim_l tile
        ix, iy, it = datgen.sample_xyt(batch_type=batch_type,
                                     timestep=timestep,
                                     SID=SID,
                                     parent_pixel_id=parent_pixel_id,
                                     it=it)
    tf1 = time.time()
    tf = tf1-tf0
    ''' quick '''
    
    if it is None: it = 0 # hack for daily
    
    tg0 = time.time()
    ## load input data
    (coarse_input, fine_input, subdat,
        x_inds, y_inds, lat_grid, lon_grid, timestamp) = datgen.get_input_data(
        var, ix, iy, it, timestep=timestep
    )
    tg1 = time.time()
    tg = tg1-tg0
    ''' slow '''
    
    th0 = time.time()
    ## get station targets within the dim_l x dim_l tile        
    (station_targets, station_npts, 
        station_data, context_locations) = datgen.get_station_targets(
        subdat, x_inds, y_inds, timestamp, var,
        np.stack([lat_grid, lon_grid], axis=0),
        fine_input,
        batch_type=batch_type,
        context_frac=context_frac,            
        timestep=timestep            
    )
    th1 = time.time()
    th = th1-th0
    ''' quick '''
    
    ti0 = time.time()
    constraints = []
    if return_constraints:
        if timestep=='daily':
            ## grab subgrid chess constraints                
            for v in list(datgen.var_name_map.loc[var].chess):
                constraints.append(torch.from_numpy(
                    datgen.chess_dat.isel(x=x_inds, y=y_inds)[v].values)
                    .to(torch.float32)
                )
        else:
            ## use physically reasoned constraints
            constraints = datgen.get_constraints(x_inds, y_inds, it, var)               
    else:
        constraints.append(torch.zeros((0,0)).to(torch.float32))
    constraints = torch.stack(constraints, dim=-1)
    ti1 = time.time()
    ti = ti1-ti0
    ''' quick '''
    
    ## capture sample description
    sample_meta = {
        'timestamp':datgen.td,
        'x_inds':x_inds,
        'y_inds':y_inds,
        't_ind':it,
        'coarse_xl':ix,
        'coarse_yt':iy,
        'timestep':timestep,
        'fine_var_order':datgen.fine_variable_order,
        'coarse_var_order':datgen.coarse_variable_order,
        'context_var_order':datgen.context_variable_order
    }

   
'''
main culprits to pre-load:
    self.prepare_constraints(var)
    self.prepare_fine_met_inputs(var, batch_tsteps)

'''
   
   

##################

           
'''
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
'''

if False:
    # this doesn't work with variable overlap!
    def stack_map(data, iys, ixs, tile_size, ol=1):
        ystacks = []
        for k in range(len(ixs)):
            for j in range(len(iys)):
                kk = (k*len(iys))+j
                if j==0:                
                    stacked_grid = data[kk,...][:,:(tile_size-ol),:]
                elif j<(len(iys)-1):                
                    # stacked_grid = torch.cat([stacked_grid,
                                              # data[kk,...][:,ol:(tile_size-ol),:]],
                                              # dim=1)
                    stacked_grid = np.concatenate([stacked_grid,
                                                  data[kk,...][:,ol:(tile_size-ol),:]],
                                                  axis=1)
                else:
                    pix_overlap = iys[j-1] + tile_size - iys[j] - ol
                    # stacked_grid = torch.cat([stacked_grid,
                                              # data[kk,...][:,pix_overlap:,:]],
                                              # dim=1)
                    stacked_grid = np.concatenate([stacked_grid,
                                                  data[kk,...][:,pix_overlap:,:]],
                                                  axis=1)
            #ystacks.append(torch.clone(stacked_grid))
            ystacks.append(stacked_grid.copy())
            
        for j in range(len(ixs)):
            if j==0:
                full_map = ystacks[j][:,:,:(tile_size-ol)]
            elif j<(len(ixs)-1):
                #full_map = torch.cat([full_map, ystacks[j][:,:,ol:(tile_size-ol)]], dim=2)
                full_map = np.concatenate([full_map, ystacks[j][:,:,ol:(tile_size-ol)]], axis=2)
            else:
                pix_overlap = ixs[j-1] + tile_size - ixs[j] - ol
                #full_map = torch.cat([full_map, ystacks[j][:,:,pix_overlap:]], dim=2)
                full_map = np.concatenate([full_map, ystacks[j][:,:,pix_overlap:]], axis=2)
        #return full_map.numpy()[:,::-1,:]
        return full_map[:,::-1,:]
        
    ol = 3
    #full_map_grid_av_pred = stack_map(grid_av_preds, iys, ixs, datgen.dim_l, ol=ol)
    full_map_pred = stack_map(pred,
                              [v*data_pars.scale for v in iys],
                              [v*data_pars.scale for v in ixs],
                              datgen.dim_h, ol=ol*data_pars.scale)
    full_static_ins = stack_map(batch.fine_inputs.cpu().numpy(),
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

