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

extract_counts = False

if extract_counts:
    datgen = data_generator(load_site_data=True)
else:
    datgen = data_generator(load_site_data=False)
    
# hold out some training sites to check against (the same ones each time)
context_sites = pd.DataFrame({'sites':datgen.train_sites}).sample(frac=0.6, random_state=42)
removed_sites = np.setdiff1d(datgen.train_sites, context_sites.sites.values)
datgen.train_sites = list(np.setdiff1d(datgen.train_sites, removed_sites))
datgen.heldout_sites += list(removed_sites)

if False:
    # save train / heldout sites
    pd.to_pickle({
        'context_sites':datgen.train_sites,
        'target_sites':datgen.heldout_sites
    }, "./output/site_splits.pkl")

def calc_var_timestep_metrics(dat, sites, var, model_names):
    out_df = pd.DataFrame()
    for sid in sites:            
        subdat = dat[dat['SITE_ID']==sid][[var] + model_names]
        try:
            for mod in model_names:
                mets = calc_metrics(subdat, var, mod)
            
                met_df = (pd.DataFrame(mets, index=[0])
                    .assign(SITE_ID = sid,
                            timestep = timestep,
                            model = mod.replace('pred_',''),
                            var = var)
                )
                out_df = pd.concat([out_df, met_df], axis=0)
        except:
            continue
    return out_df

years = [2015, 2016, 2017, 2018, 2019]
basedir = '/home/users/doran/projects/downscaling/'
pltdir = basedir + '/output/uk_tests/plots/'
Path(pltdir).mkdir(parents=True, exist_ok=True)
metrics_all = pd.DataFrame()
metrics_cntx = pd.DataFrame()
metrics_targ = pd.DataFrame()
for var in ['TA', 'PA', 'SWIN', 'LWIN', 'WS', 'RH', 'DTR']: # adding daily temp range
    #for timestep in ['daily', 'hourly']:
    for timestep in ['daily']:
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
        
        if len(years)>1: year = '' # fix year label
        
        if extract_counts and timestep == "daily":
            if var != 'DTR':
                obs = pd.DataFrame()
                obs_counts = pd.DataFrame()
                for si in dat.SITE_ID.unique():
                    obs = pd.concat([
                        obs,
                        datgen.site_data[si][[var]].resample("D").mean().reset_index().assign(SITE_ID = si)],
                        axis=0
                    )
                    obs_counts = pd.concat([
                        obs_counts,
                        (datgen.site_data[si][[var]].dropna()
                        .resample("D")
                        .agg("count")
                        .rename({var:'obs_count'}, axis=1)
                        .reset_index()
                        .assign(SITE_ID = si))],
                        axis=0
                    )
                obs_counts.to_csv(f'./output/uk_tests/obs_counts_hoursinday_{var}.csv', index=False)
            else:
                obs = pd.DataFrame()
                obs_counts = pd.DataFrame()
                for si in dat.SITE_ID.unique():
                    maxs = datgen.site_data[si][['TA']].resample("D").max()
                    mins = datgen.site_data[si][['TA']].resample("D").min()
                    obs = pd.concat([
                        obs,
                        (maxs-mins).reset_index().assign(SITE_ID = si)],
                        axis=0
                    )
                    obs_counts = pd.concat([
                        obs_counts,
                        (datgen.site_data[si][['TA']].dropna()
                        .resample("D")
                        .agg("count")
                        .rename({'TA':'obs_count'}, axis=1)
                        .reset_index()
                        .assign(SITE_ID = si))],
                        axis=0
                    )
                obs = obs.rename({'TA':'DTR'}, axis=1)
                # don't need to save obs_counts for DTR as it is the same as TA
                
            # if False:
                # #checks
                # si = 'BUNNY'
                # dat[dat['SITE_ID']==si][['SITE_ID', 'DATE_TIME', var]].dropna()
                # (dat[dat['SITE_ID']==si].drop(var, axis=1)
                    # .merge(obs, on=['SITE_ID', 'DATE_TIME'], how='left')
                    # [['SITE_ID', 'DATE_TIME', var]].dropna()
                # )
                # obs[obs['SITE_ID']==si].dropna()
                # obs[(obs['SITE_ID']==si) & (obs['DATE_TIME'] < pd.to_datetime("2015-12-31", utc=True))].dropna()
            
            # join re-extracted obs and obs counts
            dat = (dat.drop(var, axis=1)
                .merge(obs, on=['SITE_ID', 'DATE_TIME'], how='left')
                .merge(obs_counts, on=['SITE_ID', 'DATE_TIME'], how='left')
            )

        modelnames = ['pred_model', 'pred_model_nc', 'pred_era5_interp']
        if timestep=='daily':
            modelnames += ['pred_chess']
        
        ## un-normalise results back to original units
        dat = unnormalise(dat, var, modelnames)
        
        # cutoff for missing hours in a day
        if timestep == "daily":          
            dat = dat[dat['obs_count']>20] 
        
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
                plt.savefig(pltdir + f'/{var}_{timestep}_{SID}_timeseries.png', bbox_inches='tight')
                #plt.show()
                plt.close()

        ## scatter plots               
        model_compare_site_density_scatter(dat, var, modelnames, year, timestep,
                                           outname='allsites', sites=sites,
                                           title_text='All sites', pltdir=pltdir)# save=True/False
                                           
        model_compare_site_density_scatter(dat, var, modelnames, year, timestep,
                                           outname='contextsites', sites=tr_sites,
                                           title_text='Context sites', pltdir=pltdir)
                                           
        model_compare_site_density_scatter(dat, var, modelnames, year, timestep,
                                           outname='targetsites', sites=ho_sites,
                                           title_text='Target sites', pltdir=pltdir)

        ## calculate metrics
        thisdf = calc_var_timestep_metrics(dat, sites, var, modelnames)
        metrics_all = pd.concat([metrics_all, thisdf], axis=0)
        
        thisdf = calc_var_timestep_metrics(dat, tr_sites, var, modelnames)
        metrics_cntx = pd.concat([metrics_cntx, thisdf], axis=0)
        
        thisdf = calc_var_timestep_metrics(dat, ho_sites, var, modelnames)
        metrics_targ = pd.concat([metrics_targ, thisdf], axis=0)

metrics_all.reset_index(drop=True).to_csv(basedir+f'/output/uk_tests/metrics_{year}_allsites.csv', index=False)
metrics_cntx.reset_index(drop=True).to_csv(basedir+f'/output/uk_tests/metrics_{year}_contextsites.csv', index=False)
metrics_targ.reset_index(drop=True).to_csv(basedir+f'/output/uk_tests/metrics_{year}_targetsites.csv', index=False)


dat['elev_bins'] = pd.cut(dat['ALTITUDE'], 50)

def qf(qval):
    def quant(x):
        if len(x)==0:
            return np.nan
        return np.nanquantile(x,qval)
    return quant

bins_dat = dat[['elev_bins','pred_model']].groupby('elev_bins').agg(('mean',qf(0.25), qf(0.75))).reset_index()
mean_elev_bins = dat[['elev_bins','ALTITUDE']].groupby('elev_bins').agg('mean').reset_index()
bins_dat.columns = ['elev_bins', 'WS_mean', 'WS_q25', 'WS_q75']
bins_dat['lerr'] = bins_dat['WS_mean'] - bins_dat['WS_q25']
bins_dat['uerr'] = bins_dat['WS_q75'] - bins_dat['WS_mean']
bins_dat = bins_dat.merge(mean_elev_bins, on='elev_bins', how='left')

bins_dat2 = dat[['elev_bins','WS']].groupby('elev_bins').agg((np.nanmean,qf(0.25), qf(0.75))).reset_index()
mean_elev_bins2 = dat[['elev_bins','ALTITUDE']].groupby('elev_bins').agg('mean').reset_index()
bins_dat2.columns = ['elev_bins', 'WS_mean', 'WS_q25', 'WS_q75']
bins_dat2['lerr'] = bins_dat2['WS_mean'] - bins_dat2['WS_q25']
bins_dat2['uerr'] = bins_dat2['WS_q75'] - bins_dat2['WS_mean']
bins_dat2 = bins_dat2.merge(mean_elev_bins2, on='elev_bins', how='left')

fig, ax = plt.subplots(1,2,sharey=True)
ax[0].errorbar(bins_dat['ALTITUDE'], bins_dat['WS_mean'],
            yerr=bins_dat[['lerr', 'uerr']].values.T, fmt='none')
ax[0].plot(bins_dat['ALTITUDE'], bins_dat['WS_mean'], 'o')
ax[0].set_xlabel('Elevation')
ax[0].set_ylabel('Wind speed')
ax[1].errorbar(bins_dat2['ALTITUDE'], bins_dat2['WS_mean'],
            yerr=bins_dat2[['lerr', 'uerr']].values.T, fmt='none')
ax[1].plot(bins_dat2['ALTITUDE'], bins_dat2['WS_mean'], 'o')
ax[1].set_xlabel('Elevation')
plt.show()



for var in ['TA', 'PA', 'SWIN', 'LWIN', 'WS', 'RH', 'DTR']:
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
                    
