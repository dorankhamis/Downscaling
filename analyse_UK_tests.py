## Loading UK test runs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn import metrics

from setupdata3 import data_generator
from params import normalisation as nm
from utils import *
from plotting import *

datgen = data_generator(load_site_data=False)
    
# hold out some training sites to check against (the same ones each time)
context_sites = pd.DataFrame({'sites':datgen.train_sites}).sample(frac=0.6, random_state=42)
removed_sites = np.setdiff1d(datgen.train_sites, context_sites.sites.values)
datgen.train_sites = list(np.setdiff1d(datgen.train_sites, removed_sites))
datgen.heldout_sites += list(removed_sites)

def calc_var_timestep_metrics(dat, sites, var, timestep):
    out_df = pd.DataFrame()
    for sid in sites:            
        subdat = dat[dat['SITE_ID']==sid][[var, 'pred_model', 'pred_model_nc']]
        try:
            mets = calc_metrics(subdat, var, 'pred_model')
            mets_nc = calc_metrics(subdat, var, 'pred_model_nc')
            
            met_df = (pd.DataFrame(mets, index=[0])
                .assign(SITE_ID = sid,
                        timestep = timestep,
                        model = 'full',
                        var = var)
            )            
            met_nc_df = (pd.DataFrame(mets_nc, index=[0])
                .assign(SITE_ID = sid,
                        timestep = timestep,
                        model = 'nc',
                        var = var)
            )
            out_df = pd.concat([out_df, met_df, met_nc_df], axis=0)
            if timestep=='daily':
                subdat = dat[dat['SITE_ID']==sid][[var, 'pred_chess']]
                mets_chess = calc_metrics(subdat, var, 'pred_chess')
                met_chess_df = (pd.DataFrame(mets_chess, index=[0])
                    .assign(SITE_ID = sid,
                            timestep = timestep,
                            model = 'chess',
                            var = var)
                )
                out_df = pd.concat([out_df, met_chess_df], axis=0)            
        except:
            continue
    return out_df

years = [2015]#, 2017, 2018, 2019]
basedir = '/home/users/doran/projects/downscaling/'
pltdir = basedir + '/output/uk_tests/plots/'
Path(pltdir).mkdir(parents=True, exist_ok=True)
metrics_all = pd.DataFrame()
metrics_cntx = pd.DataFrame()
metrics_targ = pd.DataFrame()
for var in ['TA', 'PA', 'SWIN', 'LWIN', 'WS', 'RH']:#, 'DTR']: # adding daily temp range
    for timestep in ['daily', 'hourly']:
        ## load data
        dat = pd.DataFrame()
        for year in years:
            for m in range(1,13):
                for d in range(1,32):
                    try:
                        dat = pd.concat([
                            dat,
                            pd.read_csv(basedir + f'/output/uk_tests/{year}/{var}_{year}{zeropad_strint(m)}{zeropad_strint(d)}_{timestep}.csv')
                        ], axis=0)
                    except:
                        continue
        dat['DATE_TIME'] = pd.to_datetime(dat.DATE_TIME)
        
        modelnames = ['pred_model', 'pred_model_nc']
        if timestep=='daily':
            modelnames += ['pred_chess']
        
        ## un-normalise results back to original units
        dat = unnormalise(dat, var, modelnames)        
        
        # site lists
        sites = list(dat.SITE_ID.unique())
        tr_sites = list(np.intersect1d(datgen.train_sites, sites))
        ho_sites = list(np.intersect1d(datgen.heldout_sites, sites))
        
        if False:
            ## timeseries plots
            for SID in ['SHEEP', 'BUNNY', 'LULLN']:            
                fig, ax = plt.subplots(figsize=(18,7))            
                (dat[dat['SITE_ID']==SID].set_index('DATE_TIME')
                    .plot(y=[var]+modelnames, ax=ax))
                ax.set_title(f'{SID}, {var} {year} {timestep}')
                plt.show()
                plt.savefig(pltdir + f'/{var}_{timestep}_{SID}_timeseries.png', bbox_inches='tight')
                #plt.show()
                plt.close()

            ## scatter plots               
            model_compare_site_density_scatter(dat, var, modelnames, year, timestep,
                                               outname='allsites', sites=sites,
                                               title_text='All sites', pltdir=pltdir)
                                               
            model_compare_site_density_scatter(dat, var, modelnames, year, timestep,
                                               outname='contextsites', sites=tr_sites,
                                               title_text='Context sites', pltdir=pltdir)
                                               
            model_compare_site_density_scatter(dat, var, modelnames, year, timestep,
                                               outname='targetsites', sites=ho_sites,
                                               title_text='Target sites', pltdir=pltdir)

        ## calculate metrics
        thisdf = calc_var_timestep_metrics(dat, sites, var, timestep)
        metrics_all = pd.concat([metrics_all, thisdf], axis=0)
        
        thisdf = calc_var_timestep_metrics(dat, tr_sites, var, timestep)
        metrics_cntx = pd.concat([metrics_cntx, thisdf], axis=0)
        
        thisdf = calc_var_timestep_metrics(dat, ho_sites, var, timestep)
        metrics_targ = pd.concat([metrics_targ, thisdf], axis=0)

metrics_all.reset_index(drop=True).to_csv(basedir+f'/output/uk_tests/metrics_{year}_allsites.csv', index=False)
metrics_cntx.reset_index(drop=True).to_csv(basedir+f'/output/uk_tests/metrics_{year}_contextsites.csv', index=False)
metrics_targ.reset_index(drop=True).to_csv(basedir+f'/output/uk_tests/metrics_{year}_targetsites.csv', index=False)

for var in ['TA', 'PA', 'SWIN', 'LWIN', 'WS', 'RH']:
    for metric_to_plot in ['NSE', 'KGE']:        
        fig, ax = plt.subplots(figsize=(5,5))        
        sns.boxplot(x='model', y=metric_to_plot, hue='model',
                    width=0.9, dodge=False, ax=ax,
                    data=metrics_all[(metrics_all['timestep']=='daily') &
                                     (metrics_all['var']==var)])
        ax.set_title(f'All sites, {var} {year} daily, metric: {metric_to_plot}')
        plt.ylim(0,1)
        plt.savefig(pltdir + f'/{var}_{year}_daily_{metric_to_plot}_boxplot_allsites.png', bbox_inches='tight')        
        plt.close()
        #plt.show()

        fig, ax = plt.subplots(figsize=(5,5))
        g = sns.FacetGrid(data = metrics_all[(metrics_all['timestep']=='hourly') &
                                     (metrics_all['var']==var)],
                         col='model', hue='model')
        g.map(sns.histplot, metric_to_plot)
        plt.show()
        sns.histplot(x=metric_to_plot, hue='model', bins=25, ax=ax,
                    # data=metrics_all[(metrics_all['timestep']=='hourly') &
                                     # (metrics_all['var']==var)])
        sns.boxplot(x='model', y=metric_to_plot, hue='model',
                    width=0.9, dodge=False, ax=ax,
                    data=metrics_all[(metrics_all['timestep']=='hourly') &
                                     (metrics_all['var']==var)])
        plt.ylim(0,1)
        ax.set_title(f'All sites, {var} {year} hourly, metric: {metric_to_plot}')        
        plt.savefig(pltdir + f'/{var}_{year}_hourly_{metric_to_plot}_hist_allsites.png', bbox_inches='tight')
        plt.close()
        #plt.show()

# sns.violinplot(x='model', y='NSE', hue='model', width=0.9, dodge=False, gridsize=1000,
            # data=metrics_all[(metrics_all['timestep']=='daily') & 
                             # (metrics_all['var']=='WS')])
                    
