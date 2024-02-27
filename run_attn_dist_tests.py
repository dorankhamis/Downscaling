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

from setupdata3 import data_generator, Batch, create_chess_pred, load_process_chess
from model2 import SimpleDownscaler
from params import data_pars, model_pars, train_pars
from loss_funcs2 import make_loss_func
from step3 import create_null_batch
from params import normalisation as nm
from utils import *
from plotting import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calc_site_metrics_for_batch(pred, batch, var, model_label = 'pred_model'):
    if type(var)==str: var = [var]
    site_preds = pd.DataFrame()
    for ii in range(batch.coarse_inputs.shape[0]):
        # this_context = (batch.raw_station_dict[ii]['context']
            # .assign(pred = 
                # pred[ii, 0, 
                    # batch.raw_station_dict[ii]['context'].sub_y.values,
                    # batch.raw_station_dict[ii]['context'].sub_x.values
                # ],
                # dat_type = 'context')
            # .reset_index() # move SITE_ID from index
            # .rename({'pred':model_label}, axis=1)[['SITE_ID', var, model_label, 'dat_type']]
        # )
        # this_target = (batch.raw_station_dict[ii]['target']
            # .assign(pred = 
                # pred[ii, 0, 
                    # batch.raw_station_dict[ii]['target'].sub_y.values,
                    # batch.raw_station_dict[ii]['target'].sub_x.values
                # ],
                # dat_type = 'target')
            # .rename({'pred':model_label}, axis=1)[['SITE_ID', var, model_label, 'dat_type']]
        # )
        
        this_context = batch.raw_station_dict[ii]['context'].copy()
        this_context[[s+f'_{model_label}' for s in var]] = pred[
            ii,
            :, 
            batch.raw_station_dict[ii]['context'].sub_y.values,
            batch.raw_station_dict[ii]['context'].sub_x.values
        ].T
        this_context = (this_context.assign(dat_type = 'context')
            .reset_index() # move SITE_ID from index
        )
        
        this_target = batch.raw_station_dict[ii]['target'].copy()
        this_target[[s+f'_{model_label}' for s in var]] = pred[
            ii,
            :, 
            batch.raw_station_dict[ii]['target'].sub_y.values,
            batch.raw_station_dict[ii]['target'].sub_x.values
        ].T
        this_context = (this_context.assign(dat_type = 'target')
            .reset_index() # move SITE_ID from index
        )
                            
        site_preds = pd.concat([site_preds, this_context, this_target], axis=0)
                    
    ''' calc metric function needs altering to allow multiple vars for WS preds '''
    metrics_context = calc_metrics(site_preds[site_preds['dat_type']=='context'], var, model_label)
    metrics_target = calc_metrics(site_preds[site_preds['dat_type']=='target'], var, model_label)
    metrics_allsites = calc_metrics(site_preds, var, model_label)
    
    metrics_context = pd.DataFrame(metrics_context, index=['context'])[['NSE', 'KGE', 'mae']]
    metrics_target = pd.DataFrame(metrics_target, index=['target'])[['NSE', 'KGE', 'mae']]
    metrics_allsites = pd.DataFrame(metrics_allsites, index=['allsites'])[['NSE', 'KGE', 'mae']]
    this_out = pd.concat([metrics_context, metrics_target, metrics_allsites], axis=0).reset_index().rename({'index':'dat_type'}, axis=1)
    return this_out


## create data generator
datgen = data_generator()

projdir = '/home/users/doran/projects/downscaling/'

if __name__=="__main__":
    var_names = ['TA', 'PA', 'SWIN', 'LWIN', 'WS', 'RH']
    
    try:
        parser = argparse.ArgumentParser()        
        parser.add_argument("year", help = "year to run")
        parser.add_argument("day", help = "day to run") # zero indexed
        parser.add_argument("varnum", help = "var num to run") # [0,5]
        args = parser.parse_args()
        year = int(args.year)        
        day = int(args.day)
        var = var_names[int(args.varnum)]
    except: # not running as batch script        
        year = 2017
        day = 0
        var = 'TA'
    
    if var=='PRECIP': datgen.load_EA_rain_gauge_data()
    
    ## output directory, shared by variables and hourly/daily data
    outdir = projdir + f'/output/attn_dist_tests/{year}/'
    Path(outdir).mkdir(parents=True, exist_ok=True)    
    
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
        
        ## build hourly data for heldout sites
        it = list(np.arange(24))
        tstamps = [curr_date + datetime.timedelta(hours=int(dt)) for dt in it]
        hourly_data = pd.DataFrame()
        if var=='WS': sel_vars = ['UX', 'VY', 'WS']
        else: sel_vars = [var]
        for sid in datgen.heldout_sites:
            thisdf = datgen.site_data[sid].reindex(index=tstamps, columns=sel_vars, fill_value=np.nan)
            thisdf.dropna(how='all')            
            hourly_data = pd.concat([hourly_data, thisdf.assign(SITE_ID = sid)], axis=0)
        hourly_data = hourly_data.dropna()
        
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
        
        # load the 24 hour batches
        tile = False
        p_hourly = 1
        max_batch_size = 1
        context_frac = 1        
        date_string = f'{year}{zeropad_strint(thismonth)}{zeropad_strint(thisday)}'        
        batch = datgen.get_all_space(var,
                                     batch_type='run',
                                     context_frac=context_frac,
                                     date_string=date_string,
                                     it=it,
                                     timestep='hourly',
                                     tile=tile)
        batch = Batch(batch, var_list=var, device=device, constraints=False)
        
        print(var)
        print(date_string)
        print(datgen.td)
        
        # add target data to batch.raw_station_dict
        for ii, dt in enumerate(tstamps):
            batch.raw_station_dict[ii]['target'] = (hourly_data.loc[dt]
                .merge(datgen.site_metadata[['SITE_ID', 'chess_y', 'chess_x',
                                             'LONGITUDE', 'LATITUDE']],
                    on = 'SITE_ID', how='left')
                .rename({'chess_y':'sub_y', 'chess_x':'sub_x'}, axis=1)
            )
        
        ## loop over attention distance params
        '''
        Sample dist_lim in [1, 250]
        and    dist_pixpass in [1, 250]
        and    poly_exp in [1, 10]
        in large steps to begin with
        '''
        results = pd.DataFrame()
        
        param_sweep_out = outdir + f'/{var}_{year}{zeropad_strint(thismonth)}{zeropad_strint(thisday)}_paramsweep.csv'
        #param_sweep_out = outdir + f'/{var}_{year}{zeropad_strint(thismonth)}{zeropad_strint(thisday)}_zoomin_paramsweep.csv'
        
        for dist_lim in [2, 10, 50, 100, 200]:
            for poly_exp in [2, 4, 6, 8]:           
                for dist_pixpass in [75, 100, 125]:

                    masks = create_attention_masks(model, batch, var,
                                                   dist_lim = dist_lim,
                                                   dist_lim_far = None,
                                                   attn_eps = None,
                                                   poly_exp = poly_exp,
                                                   diminish_model = None,
                                                   dist_pixpass = dist_pixpass,
                                                   pass_exp = None)
                                    
                    ii = 0
                    pred = []
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
                        pred.append(out.cpu())            
                        del(out)            
                        ii += max_batch_size
                    pred = torch.cat(pred, dim=0).numpy()
                    
                    if var=='WS':
                        # add wind speed as third element                
                        ws_pred = np.sqrt(np.square(pred[:,0:1,:,:] * nm.ws_sd) + np.square(pred[:,1:2,:,:] * nm.ws_sd))
                        ws_pred = (ws_pred - nm.ws_mu) / nm.ws_sd
                        pred = np.concatenate([pred, ws_pred], axis=1)

                    ## grab site predictions and metrics
                    this_out = calc_site_metrics_for_batch(pred, batch, var, model_label = 'pred_model')
                    this_out = this_out.assign(
                        dist_lim = dist_lim,
                        dist_pixpass = dist_pixpass,
                        model_type = 'pred_model'
                    )
                    
                    # store metrics for parameter set
                    results = pd.concat([results, this_out], axis=0)
                    
                    print(f'Saving intermediate results to {param_sweep_out}')
                    results.to_csv(param_sweep_out, index=False)
            
        # then calculate metrics for no-context model run
        batch2 = create_null_batch(batch, constraints=False)
        ii = 0
        pred2 = []            
        while ii<batch2.coarse_inputs.shape[0]:
            iinext = min(ii+max_batch_size, batch2.coarse_inputs.shape[0])

            with torch.no_grad():
                out2 = model(batch2.coarse_inputs[ii:iinext,...],
                             batch2.fine_inputs[ii:iinext,...],
                             batch2.context_data[ii:iinext],
                             batch2.context_locs[ii:iinext])
            pred2.append(out2.cpu())            
            del(out2)
            ii += max_batch_size        
        pred2 = torch.cat(pred2, dim=0).numpy()        

        ## grab site predictions and metrics
        this_out_nc = calc_site_metrics_for_batch(pred2, batch2, model_label = 'pred_model_nc')
        this_out_nc = this_out_nc.assign(
            dist_lim = np.nan,
            dist_pixpass = np.nan,
            model_type = 'pred_model_nc'
        )
        del(batch2)
        
        # store metrics
        results = pd.concat([results, this_out_nc], axis=0)
            
        # save data
        print(f'Saving full results to {param_sweep_out}')
        results.to_csv(param_sweep_out, index=False)

