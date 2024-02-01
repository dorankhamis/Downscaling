import numpy as np
import pandas as pd
import xarray as xr
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import argparse
import torch.nn as nn

from torch.optim import Adam
from pathlib import Path
from sklearn.neighbors import NearestNeighbors

from setupdata3 import (data_generator, Batch, create_chess_pred,
                        load_process_chess, interp_to_grid, reflect_pad_nans)
from model2 import SimpleDownscaler
from params import data_pars, model_pars, train_pars
from loss_funcs2 import make_loss_func
from step3 import create_null_batch
from params import normalisation as nm
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

projdir = '/home/users/doran/projects/downscaling/'
var_names = ['TA', 'PA', 'SWIN', 'LWIN', 'WS', 'RH', 'PRECIP']

if __name__=="__main__":            
    try:
        parser = argparse.ArgumentParser()        
        parser.add_argument("year", help = "year to run")
        parser.add_argument("day", help = "day to run") # zero indexed
        args = parser.parse_args()
        year = int(args.year)  
        month = int(args.month)        
        day = int(args.day)
    except: # not running as batch script        
        year = 2015
        month = 5
        day = 14
    
    datgen = data_generator()
    
    ## output directory, shared by variables and hourly/daily data
    outdir = projdir + f'/output/'
    Path(outdir).mkdir(parents=True, exist_ok=True)
    
    log_dir = projdir + '/logs/'
    load_prev_chkpnt = True
    
    # set params and structures
    it = list(np.arange(24))
    tile = False
    p_hourly = 1
    max_batch_size = 1
    context_frac = 1
    date_string = f'{year}{zeropad_strint(month)}{zeropad_strint(day)}'    
        
    ## run through all vars
    result = {}
    for var in var_names:        
        model_name = f'dwnsamp_{var}'
        model_outdir = f'{log_dir}/{model_name}/'        
        specify_chkpnt = f'{model_name}/checkpoint.pth' # latest       
        
        if var=='PRECIP':
            datgen.load_EA_rain_gauge_data()
        
        ## dummy batch for model param fetching
        batch = datgen.get_batch(
            var,
            batch_size=1,
            batch_type='train',                                 
            p_hourly=1
        )
                                      
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

        del(batch)
        model.to(device)        

        ## load checkpoint
        optimizer = None
        model, opt, chk = setup_checkpoint(
            model, optimizer, device, load_prev_chkpnt,
            model_outdir, log_dir,
            specify_chkpnt=specify_chkpnt,
            reset_chkpnt=reset_chkpnt
        )
        
        model.eval()

        hourly_results = pd.DataFrame()    
         
        batch = datgen.get_all_space(var,
                                     batch_type='train',
                                     load_binary_batch=False,
                                     context_frac=context_frac,
                                     date_string=date_string,
                                     it=it,
                                     timestep='hourly',
                                     tile=tile)            
        
        print(var)
        print(date_string)
        
        batch = Batch(batch, var_list=var, device=device, constraints=False)        
        ii = 0
        pred = []
        while ii<batch.coarse_inputs.shape[0]:
            iinext = min(ii+max_batch_size, batch.coarse_inputs.shape[0])
            
            masks = create_attention_masks(model, batch, var)
            
            with torch.no_grad():
                out = model(batch.coarse_inputs[ii:iinext,...],
                            batch.fine_inputs[ii:iinext,...],
                            batch.context_data[ii:iinext],
                            batch.context_locs[ii:iinext],
                            context_masks=masks['context_masks'],
                            context_soft_masks=masks['context_soft_masks'],
                            pixel_passer=masks['pixel_passers'])            

            pred.append(out.cpu())
            del(out)            
            ii += max_batch_size
        pred = torch.cat(pred, dim=0).numpy()
        
        # un normalise the output back to COSMOS units
        if var=='WS':
            pred = unnormalise_img(pred.copy(), 'UX')
            result['UX'] = pred[:,0:1,:,:].copy()
            result['VY'] = pred[:,1:,:,:].copy()
        else:
            pred = unnormalise_img(pred.copy(), var)
            result[var] = pred.copy()
        del(pred)
        
    # current units are:
    
    # create specific humidity (rh -> d2m; then d2m, p -> q)
    
    # transform to Jules units
    
    # output as single netcdf with ERA5 variable names and chess grid
    # e.g.
    chess_latlon = chess_latlon.assign_coords(
        coords={'time':np.array(sub_time.index, dtype=np.datetime64)}
    )
    chess_latlon['theta'] = (['time','y','x'], sm_template,
        {'notes':'soil moisture, as volumetric water content',
         'units':'fractional VWC'
        }
    )
    chess_latlon['tsoil'] = (['time','y','x'], lst_template,
        {'notes':'topsoil temperature',
         'units':'degrees Celsius'
        }
    )
    if OUTPUT_PRECIP:
        chess_latlon['precip'] = (['time','y','x'], precip_template,
            {'notes':'total precipitation', 'units':'mm'}
        ) 
    chess_latlon = chess_latlon.drop(['landfrac', 'data_present', 'lat', 'lon'])


