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
from model2 import MetVAE, SimpleDownscaler
from params import data_pars, model_pars, train_pars
from loss_funcs2 import make_loss_func
from step3 import create_null_batch
from params import normalisation as nm
from utils import *
from plotting import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calc_site_metrics_for_batch(pred, batch, model_label = 'pred_model'):
    site_preds = pd.DataFrame()
    for ii in range(batch.coarse_inputs.shape[0]):
        this_context = (batch.raw_station_dict[ii]['context']
            .assign(pred = 
                pred[ii, 0, 
                    batch.raw_station_dict[ii]['context'].sub_y.values,
                    batch.raw_station_dict[ii]['context'].sub_x.values
                ],
                dat_type = 'context')
            .reset_index() # move SITE_ID from index
            .rename({'pred':model_label}, axis=1)[['SITE_ID', var, model_label, 'dat_type']]
        )
        
        this_target = (batch.raw_station_dict[ii]['target']
            .assign(pred = 
                pred[ii, 0, 
                    batch.raw_station_dict[ii]['target'].sub_y.values,
                    batch.raw_station_dict[ii]['target'].sub_x.values
                ],
                dat_type = 'target')
            .rename({'pred':model_label}, axis=1)[['SITE_ID', var, model_label, 'dat_type']]
        )
                            
        site_preds = pd.concat([site_preds, this_context, this_target], axis=0)
                    
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
        for sid in datgen.heldout_sites:
            thisdf = datgen.site_data[sid].reindex(index=tstamps, columns=[var], fill_value=np.nan)
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

        ## dummy batch for model param fetching
        batch = datgen.get_batch(var, batch_size=train_pars.batch_size,
                                 batch_type='train',
                                 load_binary_batch=False,
                                 p_hourly=1)
                                      
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
                                     batch_type='train',
                                     load_binary_batch=False,
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
        Sample dist_lim in [1, 1000], and dist_lim_far == dist_lim + 50
        and    dist_pixpass in [1, 1000]
        in large steps to begin with
        '''        
        attn_eps = 1e-6 # 1e-6
        poly_exp = 6.
        diminish_model = "polynomial" # ["gaussian", "polynomial"]
        pass_exp = 1. # 1.        
        soft_masks = True
        pixel_pass_masks = True
        binary_masks = False
        results = pd.DataFrame()
        
        #param_sweep_out = outdir + f'/{var}_{year}{zeropad_strint(thismonth)}{zeropad_strint(thisday)}_paramsweep.csv'
        #param_sweep_out = outdir + f'/{var}_{year}{zeropad_strint(thismonth)}{zeropad_strint(thisday)}_zoomin_paramsweep.csv'
        param_sweep_out = outdir + f'/{var}_{year}{zeropad_strint(thismonth)}{zeropad_strint(thisday)}_{diminish_model}_paramsweep.csv'
        
        #for dist_lim in [1, 10, 50, 100, 200, 400, 800]:
        #for dist_lim in list(range(10,210,10)):
        for dist_lim in list(range(20,220,20)):
            
            dist_lim_far = dist_lim + 50
            
            for dist_pixpass in [1, 10, 25, 50, 75, 100]:
            #for dist_pixpass in [100]:
                    
                ii = 0
                pred = []
                while ii<batch.coarse_inputs.shape[0]:
                    iinext = min(ii+max_batch_size, batch.coarse_inputs.shape[0])

                    distances, softmask, scale_factors, site_yx = prepare_attn(
                        model,
                        batch,
                        datgen.site_metadata,
                        datgen.fine_grid,
                        context_sites=None,
                        b=ii,
                        dist_lim=dist_lim,
                        dist_lim_far=dist_lim_far,
                        attn_eps=attn_eps,
                        poly_exp=poly_exp,
                        diminish_model=diminish_model
                    )
                    
                    masks = {
                        'context_soft_masks':[None,None,None,None],
                        'pixel_passers':     [None,None,None,None],
                        'context_masks':     [None,None,None,None]
                    }
                    if soft_masks:
                        masks['context_soft_masks'] = build_soft_masks(softmask, scale_factors, device)
                    if pixel_pass_masks:        
                        masks['pixel_passers'] = build_pixel_passers(distances, scale_factors, dist_pixpass, pass_exp, device)
                        # reshaping is done in the model...
                    if binary_masks:
                        masks['context_masks'] = build_binary_masks(distances, scale_factors, dist_lim_far, device)
                    
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

                ## grab site predictions and metrics
                this_out = calc_site_metrics_for_batch(pred, batch, model_label = 'pred_model')
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

