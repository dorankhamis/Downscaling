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
from plotting import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## create data generator
datgen = data_generator()

projdir = '/home/users/doran/projects/downscaling/'

if __name__=="__main__":
    var_names = ['TA', 'PA', 'SWIN', 'LWIN', 'WS', 'RH', 'PRECIP']
    
    try:
        parser = argparse.ArgumentParser()        
        parser.add_argument("year", help = "year to run")
        parser.add_argument("day", help = "day to run") # zero indexed
        args = parser.parse_args()
        year = int(args.year)        
        day = int(args.day)
    except: # not running as batch script        
        year = 2015
        day = 0
    
    ## output directory, shared by variables and hourly/daily data
    outdir = projdir + f'/output/uk_tests/{year}/'
    Path(outdir).mkdir(parents=True, exist_ok=True)
    
    datgen.load_EA_rain_gauge_data()
    
    ## hold out some training sites to check against
    ## then set context frac as 1 so all training sites are seen as context
    context_sites = pd.DataFrame({'SITE_ID':datgen.train_sites}).sample(frac=0.6, random_state=42)
    removed_sites = np.setdiff1d(datgen.train_sites, context_sites.SITE_ID.values)
    datgen.train_sites = list(np.setdiff1d(datgen.train_sites, removed_sites))
    datgen.heldout_sites += list(removed_sites)

    # save an index column for ordered recall later (s_idx)
    context_sites = (context_sites.reset_index(drop=True)
        .assign(s_idx = np.arange(context_sites.shape[0]))
        .set_index('SITE_ID')
    )    
        
    # run this day for all vars, if we haven't jumped to the next year
    curr_date = pd.to_datetime(f'{year}0101', format='%Y%m%d', utc=True)
    curr_date = curr_date + datetime.timedelta(days=day)
    thisday = curr_date.day
    thismonth = curr_date.month
    if curr_date.year == year: # we are still within the same year    
        
        ## build hourly data
        tstamps = [curr_date + datetime.timedelta(hours=dt) for dt in range(24)]    
        hourly_data = pd.DataFrame()
        sites = list(datgen.site_data.keys())
        for sid in sites:            
            thisdf = datgen.site_data[sid].reindex(index=tstamps, columns=var_names+['UX', 'VY'], fill_value=np.nan)
            thisdf.dropna(how='all')            
            hourly_data = pd.concat([hourly_data, thisdf.assign(SITE_ID = sid)], axis=0)
        
        ## run through all vars
        for var in var_names:
            log_dir = projdir + '/logs/'
            model_name = f'dwnsamp_{var}'
            model_outdir = f'{log_dir}/{model_name}/'
            Path(model_outdir).mkdir(parents=True, exist_ok=True)
           
            # training flags
            load_prev_chkpnt = True
            # specify_chkpnt: if None, load best, otherwise "modelname/checkpoint.pth"
            #specify_chkpnt = None                          # best
            specify_chkpnt = f'{model_name}/checkpoint.pth' # latest
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

            ## create optimizer, schedulers and loss function
            loglikelihood = make_loss_func(train_pars)
            optimizer = None

            ## load checkpoint
            model, opt, chk = setup_checkpoint(
                model, optimizer, device, load_prev_chkpnt,
                model_outdir, log_dir,
                specify_chkpnt=specify_chkpnt,
                reset_chkpnt=reset_chkpnt
            )
            
            model.eval()

            it = list(np.arange(24))
            tstamps = [curr_date + datetime.timedelta(hours=int(dt)) for dt in it]
            tile = False
            p_hourly = 1
            max_batch_size = 1
            context_frac = 1
 
            daily_results = pd.DataFrame()
            hourly_results = pd.DataFrame()
            if var=='TA': dtr_daily_results = pd.DataFrame()
            
            site_chess_yx = datgen.site_metadata[['SITE_ID', 'chess_y', 'chess_x']]    
            date_string = f'{year}{zeropad_strint(thismonth)}{zeropad_strint(thisday)}'
            
            batch = datgen.get_all_space(var,
                                         batch_type='run',                                         
                                         context_frac=context_frac,
                                         date_string=date_string,
                                         it=it,
                                         timestep='hourly',
                                         tile=tile)            
            
            print(var)
            print(date_string)
            print(datgen.td)
            
            batch = Batch(batch, var_list=var, device=device, constraints=False)
            batch2 = create_null_batch(batch, constraints=False)
            masks = create_attention_masks(model, batch, var)
            
            ii = 0
            pred = []
            pred2 = []
            pred_era5 = []
            while ii<batch.coarse_inputs.shape[0]:
                iinext = min(ii+max_batch_size, batch.coarse_inputs.shape[0])
    
                # extract this timepoint from the masks lists
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
                    out2 = model(batch2.coarse_inputs[ii:iinext,...],
                                 batch2.fine_inputs[ii:iinext,...],
                                 batch2.context_data[ii:iinext],
                                 batch2.context_locs[ii:iinext])                

                pred.append(out.cpu())
                pred2.append(out2.cpu())
                del(out)
                del(out2)
                
                # create era5 interped pred
                if var=='WS':
                    era5_interp = interp_to_grid(datgen.parent_pixels['hourly'][
                        datgen.var_name_map.loc[var].coarse][ii,:,:],
                        datgen.fine_grid, coords=['lat', 'lon'])
                    era5_interp = reflect_pad_nans(era5_interp).values[None, None, ...]
                else:
                    era5_interp_u10 = interp_to_grid(
                        datgen.parent_pixels['hourly']['u10'][ii,:,:],
                        datgen.fine_grid, coords=['lat', 'lon'])
                    era5_interp_v10 = interp_to_grid(
                        datgen.parent_pixels['hourly']['v10'][ii,:,:],
                        datgen.fine_grid, coords=['lat', 'lon'])                        
                    era5_interp_u10 = reflect_pad_nans(era5_interp_u10)
                    era5_interp_v10 = reflect_pad_nans(era5_interp_v10)
                    era5_interp = np.stack([
                        era5_interp_u10.values[None, ...],
                        era5_interp_v10.values[None, ...]], axis=1)
                pred_era5.append(era5_interp)
                
                ii += max_batch_size
                
            pred = torch.cat(pred, dim=0).numpy()
            pred2 = torch.cat(pred2, dim=0).numpy()
            pred_era5 = np.concatenate(pred_era5, axis=0)
            if var=='WS':
                # add wind speed as third element                
                ws_pred = np.sqrt(np.square(pred[:,0:1,:,:] * nm.ws_sd) + np.square(pred[:,1:2,:,:] * nm.ws_sd))
                ws_pred = (ws_pred - nm.ws_mu) / nm.ws_sd
                pred = np.concatenate([pred, ws_pred], axis=1)
                
                ws_pred = np.sqrt(np.square(pred2[:,0:1,:,:] * nm.ws_sd) + np.square(pred2[:,1:2,:,:] * nm.ws_sd))
                ws_pred = (ws_pred - nm.ws_mu) / nm.ws_sd
                pred2 = np.concatenate([pred2, ws_pred], axis=1)
                
                ws_pred = np.sqrt(np.square(pred_era5[:,0:1,:,:] * nm.ws_sd) + np.square(pred_era5[:,1:2,:,:] * nm.ws_sd))
                ws_pred = (ws_pred - nm.ws_mu) / nm.ws_sd
                pred_era5 = np.concatenate([pred_era5, ws_pred], axis=1)
            
            pred_dayav = pred.mean(axis=0)            
            pred2_dayav = pred2.mean(axis=0)            
            pred_era5_dayav = pred_era5.mean(axis=0)
            del(batch2)
            
            ''' also output the day and night averages for each var
            and day, night length in hours? '''
                                   
            # also create a chess batch for comparison
            chess_var = datgen.var_name_map.loc[var].chess
            pred3 = load_process_chess(datgen.td.year, datgen.td.month, datgen.td.day,
                                       chess_var, normalise=True)            
            pred3 = pred3[chess_var].values
            
            # pad to new size                
            pred3 = np.hstack([pred3, np.ones((pred3.shape[0], pred.shape[-1] - pred3.shape[-1]))*np.nan])
            pred3 = np.vstack([pred3, np.ones((pred.shape[-2] - pred3.shape[0], pred3.shape[-1]))*np.nan])

            if var=='TA':
                # also load chess daily temperature range
                pred_cdtr = load_process_chess(datgen.td.year, datgen.td.month, datgen.td.day,
                                               'dtr', normalise=True)
                
                pred_cdtr = pred_cdtr['dtr'].values
                # pad to new size                
                pred_cdtr = np.hstack([pred_cdtr, np.ones((pred_cdtr.shape[0], pred.shape[-1] - pred_cdtr.shape[-1]))*np.nan])
                pred_cdtr = np.vstack([pred_cdtr, np.ones((pred.shape[-2] - pred_cdtr.shape[0], pred_cdtr.shape[-1]))*np.nan])
                
                # and calculate predicted temperature range
                # (though this is 12-12 rather than CHESS's 9-9)
                pred_dtr = pred.max(axis=0, keepdims=True) - pred.min(axis=0, keepdims=True)
                pred_dtrnc = pred2.max(axis=0, keepdims=True) - pred2.min(axis=0, keepdims=True)
                pred_dtr_era5 = pred_era5.max(axis=0, keepdims=True) - pred_era5.min(axis=0, keepdims=True)
                    

            ########
            ## from here we only retain pixels containing met sites
            
            # retrieve day-averaged site data
            
            if False:
                datgen.daily_site_data = {}
                for SID in list(datgen.site_data.keys()):
                    datgen.daily_site_data[SID] = datgen.site_data[SID].resample('1D').mean()
            
            daily_data = datgen.site_metadata[
                ['SITE_ID', 'LATITUDE', 'LONGITUDE', 'chess_y', 'chess_x', 'parent_pixel_id']
            ]            
            if var=='WS':
                sel_vars = ['UX', 'VY', 'WS']
            else:
                sel_vars = [var]
            day_av = np.zeros((len(daily_data.SITE_ID), len(sel_vars)))
            for i, sid in enumerate(daily_data.SITE_ID):
                try:
                    if datgen.site_points_present[sid].loc[datgen.td, var]==24:
                        day_av[i,:] = datgen.daily_site_data[sid].loc[datgen.td, sel_vars].values
                    else:
                        day_av[i,:] = np.array([np.nan]*len(sel_vars))
                except:
                    day_av[i,:] = np.array([np.nan]*len(sel_vars))
            daily_data[sel_vars] = day_av

            # extract site values from grids
            site_preds_d = pred_dayav[:, site_chess_yx.chess_y.values, site_chess_yx.chess_x.values]
            site_preds2_d = pred2_dayav[:, site_chess_yx.chess_y.values, site_chess_yx.chess_x.values]
            site_preds_era5_d = pred_era5_dayav[:, site_chess_yx.chess_y.values, site_chess_yx.chess_x.values]
            site_preds_chess = pred3[site_chess_yx.chess_y.values, site_chess_yx.chess_x.values]
            site_obs = daily_data[sel_vars]
            
            daily_data[[s+'_pred_model' for s in sel_vars]] = site_preds_d.T
            daily_data[[s+'_pred_model_nc' for s in sel_vars]] = site_preds2_d.T
            daily_data[[s+'_pred_era5_interp' for s in sel_vars]] = site_preds_era5_d.T
            daily_data[[var + '_pred_chess']] = site_preds_chess[:,None]
            daily_data = daily_data.assign(DATE_TIME = curr_date)
            
            daily_results = pd.concat([daily_results, daily_data], axis=0)
            daily_results = daily_results.reset_index().drop('index', axis=1)
           
            # save daily data                
            dd_out = outdir + f'/{var}_{year}{zeropad_strint(thismonth)}{zeropad_strint(thisday)}_daily.csv'
            print(f'Saving daily results to {dd_out}')
            daily_results.to_csv(dd_out, index=False)

            if var=='TA':
                ''' add daily temperature range retrieval from sites here'''
                tmax = hourly_data[['SITE_ID', 'TA']].groupby('SITE_ID').agg(np.nanmax)
                tmin = hourly_data[['SITE_ID', 'TA']].groupby('SITE_ID').agg(np.nanmin)
                site_trange = tmax - tmin
                site_trange = site_trange.rename({'TA':'DTR'}, axis=1).reset_index()

                filt_site_chess_yx = site_chess_yx[site_chess_yx['SITE_ID'].isin(site_trange.SITE_ID)]
                # extract site values from grids
                site_preds_d = pred_dtr[0, 0, filt_site_chess_yx.chess_y.values, filt_site_chess_yx.chess_x.values]
                site_preds2_d = pred_dtrnc[0, 0, filt_site_chess_yx.chess_y.values, filt_site_chess_yx.chess_x.values]
                site_preds_era5_d = pred_dtr_era5[0, 0, filt_site_chess_yx.chess_y.values, filt_site_chess_yx.chess_x.values]
                site_preds_chess = pred_cdtr[filt_site_chess_yx.chess_y.values, filt_site_chess_yx.chess_x.values]
                
                site_trange['pred_model'] = site_preds_d
                site_trange['pred_model_nc'] = site_preds2_d
                site_trange['pred_era5_interp'] = site_preds_era5_d
                site_trange['pred_chess'] = site_preds_chess
                site_trange = site_trange.assign(DATE_TIME = curr_date)
                
                dtr_daily_results = pd.concat([dtr_daily_results, site_trange], axis=0)
                dtr_daily_results = dtr_daily_results.reset_index().drop('index', axis=1)
                
                dtr_daily_results.to_csv(outdir +f'/DTR_{year}{zeropad_strint(thismonth)}{zeropad_strint(thisday)}_daily.csv', index=False)
            
            ## get hourly results
            site_preds_h_arr = pred[:, :, site_chess_yx.chess_y.values, site_chess_yx.chess_x.values]
            site_preds2_h_arr = pred2[:, :, site_chess_yx.chess_y.values, site_chess_yx.chess_x.values]
            site_preds_era5_h_arr = pred_era5[:, :, site_chess_yx.chess_y.values, site_chess_yx.chess_x.values]

            del(pred)
            del(pred2)
            del(pred_era5)
            
            for i, v in enumerate(sel_vars):
                sph = pd.DataFrame(site_preds_h_arr[:,i,:])
                sph.columns = daily_data.SITE_ID.values
                sph = sph.assign(DATE_TIME = tstamps)            
                sph = sph.melt(id_vars='DATE_TIME',
                               var_name='SITE_ID',
                               value_name=v+'_pred_model')
                
                sp2h = pd.DataFrame(site_preds2_h_arr[:,i,:])
                sp2h.columns = daily_data.SITE_ID.values
                sp2h = sp2h.assign(DATE_TIME = tstamps)
                sp2h = sp2h.melt(id_vars='DATE_TIME',
                                 var_name='SITE_ID',
                                 value_name=v+'_pred_model_nc')
                
                spe5h = pd.DataFrame(site_preds_era5_h_arr[:,i,:])
                spe5h.columns = daily_data.SITE_ID.values
                spe5h = spe5h.assign(DATE_TIME = tstamps)
                spe5h = spe5h.melt(id_vars='DATE_TIME',
                                   var_name='SITE_ID',
                                   value_name=v+'_pred_era5_interp')
                
                if i==0:
                    site_preds_h = sph.copy()
                    site_preds2_h = sp2h.copy()
                    site_preds_era5_h = spe5h.copy()
                else:
                    site_preds_h = pd.merge(site_preds_h, sph, on=['DATE_TIME', 'SITE_ID'], how='left')
                    site_preds2_h = pd.merge(site_preds2_h, sp2h, on=['DATE_TIME', 'SITE_ID'], how='left')
                    site_preds_era5_h = pd.merge(site_preds_era5_h, spe5h, on=['DATE_TIME', 'SITE_ID'], how='left')
                
            hourly_res = (site_preds_h
                .merge(site_preds2_h, on=['DATE_TIME', 'SITE_ID'], how='left')
                .merge(site_preds_era5_h, on=['DATE_TIME', 'SITE_ID'], how='left')
            )
            hourly_res['DATE_TIME'] = pd.to_datetime(hourly_res.DATE_TIME, utc=True)
            hourly_res = hourly_res.merge(hourly_data.reset_index()[['DATE_TIME', 'SITE_ID'] + sel_vars],
                                          on=['DATE_TIME', 'SITE_ID'], how='left')
            del(site_preds_h)
            del(site_preds2_h)
            del(site_preds_era5_h)
                        
            hourly_results = pd.concat([hourly_results, hourly_res], axis=0)
            hourly_results = hourly_results.reset_index(drop=True)
                
            # save hourly data
            hh_out = outdir + f'/{var}_{year}{zeropad_strint(thismonth)}{zeropad_strint(thisday)}_hourly.csv'
            print(f'Saving hourly results to {hh_out}')
            hourly_results.to_csv(hh_out, index=False)            

