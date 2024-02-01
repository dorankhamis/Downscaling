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

def qf(q):
    def quant(x):
        return np.nanquantile(x, q)
    return quant


years = [2016]#, 2017, 2018, 2019]
basedir = '/home/users/doran/projects/downscaling/'
pltdir = basedir + '/output/attn_dist_tests/plots/'
Path(pltdir).mkdir(parents=True, exist_ok=True)
#runid = 'zoomin_paramsweep' # 'paramsweep'
allvars = pd.DataFrame()
for var in ['TA', 'PA', 'SWIN', 'LWIN', 'WS', 'RH']:    
    ## load data
    dat = pd.DataFrame()
    for rd in ['polynomial_paramsweep']:#['zoomin_paramsweep', 'paramsweep']:
        for year in years:
            for m in range(1,13):
                for d in range(1,32):
                    try:
                        thisdat = pd.read_csv(basedir + f'/output/attn_dist_tests/{year}/{var}_{year}{zeropad_strint(m)}{zeropad_strint(d)}_{rd}.csv')
                        thisdat = thisdat.assign(DATE_TIME = pd.to_datetime(f'{year}{zeropad_strint(m)}{zeropad_strint(d)}'))
                        dat = pd.concat([dat, thisdat], axis=0)
                    except:
                        continue    
    sumdat = (dat[(dat['model_type']=='pred_model')]
        .groupby(['dat_type', 'dist_pixpass', 'dist_lim'])
        .agg(NSE_q25=('NSE', qf(0.25)), NSE_q50=('NSE', qf(0.5)), NSE_q75=('NSE', qf(0.75)),
             KGE_q25=('KGE', qf(0.25)), KGE_q50=('KGE', qf(0.5)), KGE_q75=('KGE', qf(0.75)),
             mae_q25=('mae', qf(0.25)), mae_q50=('mae', qf(0.5)), mae_q75=('mae', qf(0.75)))
        .reset_index()
    )
        
    #sumdat = sumdat[sumdat['dist_pixpass']==100]
    #sumdat['pixpass'] = sumdat.dist_pixpass.astype('category')

    sumdat_nc = (dat[(dat['model_type']=='pred_model_nc')]
        .groupby(['dat_type'])
        .agg(NSE_q25=('NSE', qf(0.25)), NSE_q50=('NSE', qf(0.5)), NSE_q75=('NSE', qf(0.75)),
             KGE_q25=('KGE', qf(0.25)), KGE_q50=('KGE', qf(0.5)), KGE_q75=('KGE', qf(0.75)),
             mae_q25=('mae', qf(0.25)), mae_q50=('mae', qf(0.5)), mae_q75=('mae', qf(0.75)))
        .reset_index()
    )
    
    ## category line plots
    #errmet = 'NSE_q50'
    for errmet in ['NSE_q50', 'KGE_q50', 'mae_q50']:
        fig, ax = plt.subplots(2,2, sharex=True, sharey=True, figsize=(10,10))
        sns.lineplot(x='dist_lim', y=errmet,
                     data=sumdat[(sumdat['dat_type']=='context')],
                     hue='dist_pixpass', ax=ax[0,0])
        ax[0,0].axhline(y=sumdat_nc.loc[sumdat_nc['dat_type']=='context', errmet].values[0],
                      linestyle='--', color='k', linewidth=2)
        ax[0,0].set_title(f'{errmet}, Context sites, {var}')
        
        sns.lineplot(x='dist_lim', y=errmet,
                     data=sumdat[(sumdat['dat_type']=='target')],
                     hue='dist_pixpass', ax=ax[0,1])
        ax[0,1].axhline(y=sumdat_nc.loc[sumdat_nc['dat_type']=='target', errmet].values[0],
                      linestyle='--', color='k', linewidth=2)
        ax[0,1].set_title(f'{errmet}, Target sites, {var}')
        
        sns.lineplot(x='dist_pixpass', y=errmet,
                     data=sumdat[(sumdat['dat_type']=='context')],
                     hue='dist_lim', ax=ax[1,0])
        ax[1,0].axhline(y=sumdat_nc.loc[sumdat_nc['dat_type']=='context', errmet].values[0],
                      linestyle='--', color='k', linewidth=2)
        ax[1,0].set_title(f'{errmet}, Context sites, {var}')
        
        sns.lineplot(x='dist_pixpass', y=errmet,
                     data=sumdat[(sumdat['dat_type']=='target')],
                     hue='dist_lim', ax=ax[1,1])
        ax[1,1].axhline(y=sumdat_nc.loc[sumdat_nc['dat_type']=='target', errmet].values[0],
                      linestyle='--', color='k', linewidth=2)
        ax[1,1].set_title(f'{errmet}, Target sites, {var}')
        
        #plt.show()

        plt.savefig(pltdir + f'/{var}_{errmet}_{rd}.png', bbox_inches='tight')
        plt.show()
        plt.close()
    
    allvars = pd.concat([allvars, sumdat.assign(var = var)], axis=0)

for errmet in ['NSE_q50', 'KGE_q50', 'mae_q50']:
    nse_res = allvars.set_index(['dat_type', 'var', 'dist_lim', 'dist_pixpass'])[[errmet]]
    nse_res = nse_res.loc['target'].reset_index()    
    for var in ['TA', 'PA', 'SWIN', 'LWIN', 'WS', 'RH']:
        nse_res.loc[nse_res['var']==var, errmet] = nse_res.loc[nse_res['var']==var, errmet] / nse_res.loc[nse_res['var']==var, errmet].max()

    fig, ax = plt.subplots(1, 2, figsize=(12,5))
    sns.lineplot(x='dist_lim', y=errmet, hue='var',
                 data=nse_res, ax=ax[0])
    sns.lineplot(x='dist_pixpass', y=errmet, hue='var',
                 data=nse_res, ax=ax[1])
    plt.show()

    if errmet == 'mae_q50':
        (nse_res.groupby('var')
            .apply(lambda x: x['dist_lim'][x[errmet].idxmin()])
            .to_csv(f'./output/attn_dist_tests/var_dist_lims_{rd}.csv', index=False)
        )
        (nse_res.groupby('var')
            .apply(lambda x: x['dist_pixpass'][x[errmet].idxmin()])
            .to_csv(f'./output/attn_dist_tests/var_dist_pixpass_{rd}.csv', index=False)
        )
    else:
        (nse_res.groupby('var')
            .apply(lambda x: x['dist_lim'][x[errmet].idxmax()])
            .to_csv(f'./output/attn_dist_tests/var_dist_lims_{rd}.csv', index=False)
        )
        (nse_res.groupby('var')
            .apply(lambda x: x['dist_pixpass'][x[errmet].idxmax()])
            .to_csv(f'./output/attn_dist_tests/var_dist_pixpass_{rd}.csv', index=False)
        )

# or take average:
print(nse_res.groupby('var').apply(lambda x: x['dist_lim'][x['val'].idxmax()]).mean())

if False:
    ## errbars
    fig, ax = plt.subplots()
    for pixpass in sumdat['dist_pixpass'].unique():
        pldat = sumdat[(sumdat['dat_type']=='context') & (sumdat['dist_pixpass']==pixpass)]
        pldat[('NSE', 'q25')] = pldat[('NSE', 'q50')] - pldat[('NSE', 'q25')]
        pldat[('NSE', 'q75')] = pldat[('NSE', 'q75')] - pldat[('NSE', 'q50')]
        
        markers, caps, bars = ax.errorbar(pldat['dist_lim'], pldat[('NSE', 'q50')],
                                          yerr=pldat[[('NSE', 'q25'), ('NSE', 'q75')]].T,
                                          fmt='o', ms=2, capsize=4, capthick=1)
    plt.draw()
    plt.show()


    ## heatmap
    arr = sumdat.loc[sumdat['dat_type']=='context',[('dist_lim',''), ('dist_pixpass', ''), ('NSE','q50')]]
