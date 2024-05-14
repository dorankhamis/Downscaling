import numpy as np
import pandas as pd
import xarray as xr
import torch
import datetime
import argparse
import sys, os

from pathlib import Path

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'downscaling'))

from downscaling.setupdata import data_generator, Batch
from downscaling.model import ConvDownscaler, Resolver
from downscaling.params import data_pars, model_pars, train_pars
from downscaling.params import normalisation as nm
from downscaling.params import folders as fld
from downscaling.utils import *
from regrid_to_28km_bng import regrid_all_vars

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

prj_fldr = fld.prj_fldr
outdir = fld.outdir
Path(outdir).mkdir(parents=True, exist_ok=True)
var_names = ['TA', 'PA', 'SWIN', 'LWIN', 'WS', 'RH', 'PRECIP']

if __name__=="__main__":            
    try:
        parser = argparse.ArgumentParser()        
        parser.add_argument("year", help = "year to run")
        parser.add_argument("month", help = "month to run")
        parser.add_argument("day", help = "day to run")
        args = parser.parse_args()
        year = int(args.year)
        month = int(args.month)
        day = int(args.day)
    except: # not running as batch script        
        year = 2018
        month = 5
        day = 14
    
    date_string = f'{year}{zeropad_strint(month)}{zeropad_strint(day)}'
    print(date_string)

    ## first need to automatically run the regridding to 28km bng  
    ##################
    regrid_all_vars(date_string)

    ## then run the downscaling on the outputs
    ##################
    datgen = data_generator()
       
    # set params and structures
    log_dir = prj_fldr + '/logs/'
    load_prev_chkpnt = True
    reset_chkpnt = False
    it = list(np.arange(24))
    tile = False
    p_hourly = 1
    max_batch_size = 1
    context_frac = 1
    constraints = True
    use_resolver_vars = ['SWIN']
        
    ### FOR TESTING TRIM VARS AND/OR TIMES
    #it = [0]
    #var_names = ['TA']
        
    ## run through all vars
    result = {}
    #raw_era5 = {}
    for var in var_names:
        print(var)
        if var in use_resolver_vars:
            model_name = f'resolver_{var}'
        else:
            model_name = f'mergedwnsamp_{var}' # constraint/obs merge model        
        model_outdir = f'{log_dir}/{model_name}/'        
        specify_chkpnt = f'{model_name}/checkpoint.pth' # latest
        
        ## create model
        final_relu = True if var in ['SWIN', 'PRECIP'] else False
        if var in use_resolver_vars:
            model = Resolver(            
                hires_fields=len(model_pars.fine_variable_order[var]),
                output_channels=model_pars.output_channels[var],        
                filters=model_pars.resolver_filters,
                dropout_rate=model_pars.dropout_rate,
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
        optimizer = None
        model, opt, chk = setup_checkpoint(
            model, optimizer, device, load_prev_chkpnt,
            model_outdir, log_dir,
            specify_chkpnt=specify_chkpnt,
            reset_chkpnt=reset_chkpnt
        )        
        model.eval()
         
        batch = datgen.get_all_space(var,
                                     batch_type='run',                                     
                                     context_frac=context_frac,
                                     date_string=date_string,
                                     it=it,
                                     timestep='hourly',
                                     tile=tile,
                                     return_constraints=constraints)
        
        batch = Batch(batch, var_list=var, device=device, constraints=constraints)
        if var=='WS':
            constraint_inds = [
                datgen.fine_variable_order.index("constraint_ux_raw"),
                datgen.fine_variable_order.index("constraint_vy_raw")
            ]
        else:
            constraint_inds = datgen.fine_variable_order.index("constraint_raw")
        
        ii = 0
        pred = []
        while ii<batch.coarse_inputs.shape[0]:
            iinext = min(ii+max_batch_size, batch.coarse_inputs.shape[0])            
            with torch.no_grad():
                out = model(batch.coarse_inputs[ii:iinext,...],
                            batch.fine_inputs[ii:iinext,...],
                            constraint_inds=constraint_inds)

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
  
  
    if False:
        # visualise
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import seaborn as sns        
        from matplotlib.colors import Normalize
        
        fig, ax = plt.subplots(4,6, sharex=True, sharey=True)
        norm = Normalize(np.min(result[var]), np.max(result[var]))
        im = cm.ScalarMappable(norm=norm, cmap='plasma')
        for i in range(24):
            ax[i//6, i%6].imshow(result[var][i,0,::-1,:], cmap='plasma', norm=norm)
        fig.colorbar(im, ax=ax.ravel().tolist())
        plt.show()
        
           
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
    d2m = dewpoint_from_relhum(result['TA'], result['RH']) # for temperatures in degrees C, relative humidity in %, returns d2m in degC    
    e = vapour_pressure_from_dewpoint(d2m + 273.15) # dewpoint in K, returns e in hPa
    q = specifichum_from_vapour_pressure(e, result['PA']) # e, PA in same hPA units, returns q in kg/kg
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
        'PRECIP': 'kg m-2 s-1',
        'RH': '%'
    }
    var_long_names = {
        'TA': 'Air temperature',
        'PA': 'Air pressure',
        'SWIN': 'Downwelling shortwave radiation flux',
        'LWIN': 'Downwelling longwave radiation flux',
        'q': 'Specific humidity',
        'UX': 'X component of wind velocity (east positive, west negative)',
        'VY': 'Y component of wind velocity (north positive, south negative)',
        'PRECIP': 'Precipitation rate',
        'RH': 'Relative humidity'
    }
    
    # output as single netcdf with ERA5 variable names and chess grid
    # e.g.
    out_nc = datgen.fine_grid.assign_coords(
        coords={'time':np.array(pd.date_range(date_string, freq='H', periods=len(it)), dtype=np.datetime64)}
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
            out_nc[this_var] = (['time','y','x'],
                result[this_var][:,0,:,:] * out_nc.landfrac.values[None,...],
                {'description':var_long_names[this_var],
                 'units':units[this_var]
                }
            )
            # rh            
            out_nc['rh'] = (['time','y','x'],
                result['RH'][:,0,:,:] * out_nc.landfrac.values[None,...],
                {'description':var_long_names['RH'],
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
    
    # add attributes
    now = datetime.datetime.today()
    thedate = f'{now.year}/{zeropad_strint(now.month)}/{zeropad_strint(now.day)}'
    thetime = f'{zeropad_strint(now.hour)}:{zeropad_strint(now.minute)}:{zeropad_strint(now.second)}'
    out_nc.attrs= {
        'title': 'Downscaled meteorology',
        'description': 'ERA5 meteorology downscaled to 1km British National Grid from 0.25 degrees.',
        'institution': 'UKCEH Wallingford',
        'history': f'Created {thedate} {thetime}',
        'date_created': thedate,
        'creator_name': 'Doran Khamis, Matthew Wiggins, Amulya Chevuturi, Emma Robinson, Richard Smith, Matt Fry',
        'creator_email': 'dorkha@ceh.ac.uk',
        'publisher_name': 'UK Centre for Ecology and Hydrology',
        'publisher_url': 'http://www.ceh.ac.uk',
        'publisher_email': 'enquiries@ceh.ac.uk',
        'source': 'UKSCAPE',
        'licence': 'Licensing conditions apply (datalicensing@ceh.ac.uk)'
    }
                
    # trim out_nc to size of original chess grid 
    chess_grid = xr.open_dataset(fld.target_grid_path)
    out_nc = out_nc.isel(x = np.where(out_nc.x == chess_grid.x)[0], y = np.where(out_nc.y == chess_grid.y)[0])

    # output to /gws/nopw/j04/hydro_jules/data/uk/driving_data/era5/1km_grid/era5_1km_{date}.nc
    # add encoding to compress output?
    out_nc.to_netcdf(f'{fld.outdir}/era5_1km_{date_string}.nc')

