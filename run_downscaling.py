import numpy as np
import pandas as pd
import xarray as xr
import torch
import datetime
import argparse

from pathlib import Path

from setupdata3 import data_generator, Batch
from model2 import SimpleDownscaler
from params import data_pars, model_pars, train_pars
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
        year = 2018
        month = 5
        day = 14
    
    datgen = data_generator()
    
    ## output directory, shared by variables and hourly/daily data
    outdir = projdir + f'/output/'
    Path(outdir).mkdir(parents=True, exist_ok=True)
    
    log_dir = projdir + '/logs/'
    load_prev_chkpnt = True
    reset_chkpnt = False
    
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
                                     batch_type='run',                                     
                                     context_frac=context_frac,
                                     date_string=date_string,
                                     it=it,
                                     timestep='hourly',
                                     tile=tile)            
        
        print(var)
        print(date_string)
        
        batch = Batch(batch, var_list=var, device=device, constraints=False)
        masks = create_attention_masks(model, batch, var)    
        ii = 0
        pred = []
        while ii<batch.coarse_inputs.shape[0]:
            iinext = min(ii+max_batch_size, batch.coarse_inputs.shape[0])
            
            context_masks = [[masks['context_masks'][r][ii]] for r in range(len(masks['context_masks']))]
            context_soft_masks = [[masks['context_soft_masks'][r][ii]] for r in range(len(masks['context_soft_masks']))]
            pixel_passer = [[masks['pixel_passers'][r][ii]] for r in range(len(masks['pixel_passers']))]
            
            with torch.no_grad():
                out = model(batch.coarse_inputs[ii:iinext,...],
                            batch.fine_inputs[ii:iinext,...],
                            batch.context_data[ii:iinext],
                            batch.context_locs[ii:iinext],
                            context_masks=context_masks,
                            context_soft_masks=context_soft_masks,
                            pixel_passers=pixel_passer)            

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
        
    '''
    current units are:
        pressure: hPa
        temperature: degC
        relative humidity: %
        radiation: W m-2
        precip: mm hour-1
        wind components: m s-1
    '''
    
    # create specific humidity (rh -> d2m; then d2m, p -> q)
    d2m = (17.625 * result['TA'])/(243.04 + result['TA']) + np.log(result['RH']/100.)
    e = 6.112 * np.exp((17.67*d2m)/(d2m + 243.5))
    q = (0.622 * e)/(result['PA'] - (0.378 * e))
    result['q'] = q
    
    # transform to Jules units
    result['TA'] += 273.15 # Celsius -> Kelvin 
    result['PA'] *= 100.   # hPa -> Pa
    result['PRECIP'] /= 3600. # mm in hour -> kg m2 s-1
    
    units = {
        'TA': 'K',
        'PA': 'Pa',
        'SWIN': 'W m-2',
        'LWIN': 'W m-2',
        'q': 'kg kg-1',
        'UX': 'm s-1',
        'VY': 'm s-1',
        'PRECIP': 'kg m-2 s-1'            
    }
    var_long_names = {
        'TA': 'Air temperature',
        'PA': 'Air pressure',
        'SWIN': 'Downwelling shortwave radiation flux',
        'LWIN': 'Downwelling longwave radiation flux',
        'q': 'Specific humidity',
        'UX': 'X component of wind velocity (east positive, west negative)',
        'VY': 'Y component of wind velocity (north positive, south negative)',
        'PRECIP': 'Precipitation rate'
    }
    
    # output as single netcdf with ERA5 variable names and chess grid
    # e.g.
    out_nc = datgen.fine_grid.assign_coords(
        coords={'time':np.array(pd.date_range(date_string, freq='H', periods=24), dtype=np.datetime64)}
    )
    out_nc = out_nc.where(out_nc.landfrac>0) # nan on ocean
    
    for var in var_names:
        if var=='WS':
            # UX and VY as u10 and v10
            for this_var in ['UX', 'VY']:
                era5_name = datgen.var_name_map.loc[this_var].coarse
                out_nc[era5_name] = (['time','y','x'],
                    result[this_var][:,0,:,:] * out_nc.landfrac.values[None,...],
                    {'description':var_long_names[this_var],
                     'units':units[this_var]
                    }
                )
        elif var=='RH':
            # q
            this_var = 'q'
            out_nc[era5_name] = (['time','y','x'],
                result[this_var][:,0,:,:] * out_nc.landfrac.values[None,...],
                {'description':var_long_names[this_var],
                 'units':units[this_var]
                }
            )
        else:
            era5_name = datgen.var_name_map.loc[var].coarse
            out_nc[era5_name] = (['time','y','x'],
                result[var][:,0,:,:] * out_nc.landfrac.values[None,...],
                {'description':var_long_names[var],
                 'units':units[var]
                }
            )

    out_nc = out_nc.drop(['landfrac', 'era5_nbr', 'lat', 'lon', 'transverse_mercator'])

    chess_grid = xr.load_dataset('original smaller chess grid')
    # trim out_nc to size of original chess grid 
    
    # outout to /gws/nopw/j04/hydro_jules/data/uk/driving_data/era5/1km_grid/era5_1km_{date}.nc
    
    # also need to automatically run the regridding to 28km bng  
    
    # and get a copy of the working model / supporting files onto hydro_jules / netzero GWS
    # along with a virtual environment with torch etc installed for sourcing from anyone
    

