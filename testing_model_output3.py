import numpy as np
import pandas as pd
import xarray as xr
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from torch.optim import Adam
from pathlib import Path
from sklearn.metrics import r2_score

from setupdata4 import (data_generator, Batch, create_chess_pred,
                        load_process_chess, interp_to_grid, reflect_pad_nans)
from model3 import ConvDownscaler, Resolver
from params2 import data_pars, model_pars, train_pars
from loss_funcs3 import make_loss_func
from step4 import create_null_batch
from params2 import normalisation as nm
from utils2 import *
from plotting2 import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## create data generator
datgen = data_generator()

# file paths
#var_names = ['TA', 'PA', 'SWIN', 'LWIN', 'WS', 'RH', 'PRECIP']
var = 'SWIN'
use_resolver_vars = ['SWIN']

log_dir = './logs/'
load_prev_chkpnt = True
reset_chkpnt = False

if var in use_resolver_vars:
    model_name = f'resolver_{var}'
else:
    model_name = f'mergedwnsamp_{var}' # constraint/obs merge model
model_outdir = f'{log_dir}/{model_name}/'

# specify_chkpnt: if None, load best, otherwise "modelname/checkpoint.pth"
specify_chkpnt = f'{model_name}/checkpoint.pth'


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


###################################
## UK plots for all vars
var_names = ['TA', 'PA', 'SWIN', 'LWIN', 'WS', 'RH', 'PRECIP']
use_resolver_vars = ['SWIN']
log_dir = './logs/'
load_prev_chkpnt = True
reset_chkpnt = False
PLOT = True
output_dir = '/home/users/doran/projects/downscaling/output/test_plots/'
Path(output_dir).mkdir(parents=True, exist_ok=True)
for var in var_names:
    print(var)
    if var in use_resolver_vars:
        model_name = f'resolver_{var}'
    else:
        model_name = f'mergedwnsamp_{var}' # constraint/obs merge model
        #model_name = f'convdwnsamp_{var}'
    model_outdir = f'{log_dir}/{model_name}/'
    Path(model_outdir).mkdir(parents=True, exist_ok=True)
    
    if var=='PRECIP': datgen.load_EA_rain_gauge_data()

    #specify_chkpnt = None # f'{model_name}/checkpoint.pth' 
    specify_chkpnt = f'{model_name}/checkpoint.pth'
                                  
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

    ## load checkpoint
    model, opt, chk = setup_checkpoint(model, None, device, load_prev_chkpnt,
                                       model_outdir, log_dir,
                                       specify_chkpnt=specify_chkpnt,
                                       reset_chkpnt=reset_chkpnt)
    
    plt.plot(chk['losses'])
    plt.plot(chk['val_losses'])
    plt.show()
    model.eval()

    # get tile(s) of whole UK
    date_string = "20180101" # "20140809"
    it = 10 #8
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

    station_targets = batch.raw_station_dict
    sample_metadata = batch.batch_metadata
    met_vars = datgen.coarse_variable_order
    fine_vars = datgen.fine_variable_order
    if var=='WS':
        constraint_inds = [
            datgen.fine_variable_order.index("constraint_ux_raw"),
            datgen.fine_variable_order.index("constraint_vy_raw")
        ]
    else:
        constraint_inds = [datgen.fine_variable_order.index("constraint_raw")]
    
    with torch.no_grad():
        pred = model(batch.coarse_inputs, batch.fine_inputs, constraint_inds=constraint_inds)
        pred = pred.cpu().numpy()    
    
    np.save(output_dir + f'coarse_inputs_{var}_{date_string}_{it}.npy', batch.coarse_inputs.cpu().numpy())
    np.save(output_dir + f'prediction_{var}_{date_string}_{it}.npy', pred)

    if PLOT:
        era5 = batch.coarse_inputs.cpu().numpy()
        if var=='WS':            
            era5_unnorm = unnormalise_img(era5, 'UX')
            pred_unnorm = unnormalise_img(pred, 'UX')
        else:
            era5_unnorm = unnormalise_img(era5, var)
            pred_unnorm = unnormalise_img(pred, var)

        cmap = cm.get_cmap('plasma')
        norm = Normalize(min(np.min(pred_unnorm), np.min(era5_unnorm)),
                         max(np.max(pred_unnorm), np.max(era5_unnorm)))
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
        
        plot_batch_tiles(batch, pred, fine_vars, met_vars, b=0,
                         sea_mask=None, trim_edges=False,
                         constraints=constraints)


    ## look at station values
    if PLOT:
        b = 0
        if constraints:
            pred2 = batch.constraint_targets
        else:
            pred2 = pred
        pred3 = nn.functional.interpolate(batch.coarse_inputs, scale_factor=data_pars.scale, mode='bilinear')
        site_res = plot_context_and_target_preds(
            [pred, pred2, pred3], b,
            batch, station_targets, var, 
            model_names=['this_model', 'constraint', 'era5_interp']
        )
    
        def print_metrics(site_res, this_var, model_name):
            print(this_var)
            print(f'{model_name}')
            print("target:")            
            print(calc_metrics(site_res['target']
                .pivot(columns='variable', index='SITE_ID', values='value'),
                 this_var, model_name)
            )
            print("....")
            print("context:")            
            print(calc_metrics(site_res['context']
                .pivot(columns='variable', index='SITE_ID', values='value'),
                 this_var, model_name)
            )
            print("########")
        

        if var=="WS":
            both_components = site_res.copy()
            
            for this_var in ['UX', 'VY']:
                site_res = both_components[this_var].copy()
                for model_name in ['this_model', 'constraint', 'era5_interp']:
                    print_metrics(site_res, this_var, model_name)
        else:            
            for model_name in ['this_model', 'constraint', 'era5_interp']:
                print_metrics(site_res, var, model_name)


        plot_station_locations(batch.raw_station_dict, batch.fine_inputs, b,
                               plot_target=True, labels=False)



