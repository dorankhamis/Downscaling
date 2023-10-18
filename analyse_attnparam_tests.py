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

def q025(x): return np.quantile(x, 0.025)
def q25(x): return np.quantile(x, 0.25)
def q50(x): return np.quantile(x, 0.5)
def q75(x): return np.quantile(x, 0.75)
def q975(x): return np.quantile(x, 0.975)

years = [2015]#, 2017, 2018, 2019]
basedir = '/home/users/doran/projects/downscaling/'
pltdir = basedir + '/output/attn_dist_tests/plots/'
Path(pltdir).mkdir(parents=True, exist_ok=True)
#runid = 'zoomin_paramsweep' # 'paramsweep'
allvars = pd.DataFrame()
for var in ['TA', 'PA', 'SWIN', 'LWIN', 'WS', 'RH']:    
    ## load data
    dat = pd.DataFrame()
    for rd in ['zoomin_paramsweep', 'paramsweep']:
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
        .agg({'NSE':[q25, q50, q75], 'KGE':[q25, q50, q75], 'mae':[q25, q50, q75]})
        .reset_index()
    )
    sumdat = sumdat[sumdat['dist_pixpass']==100]
    sumdat['pixpass'] = sumdat.dist_pixpass.astype('category')
    
    allvars = pd.concat([allvars, sumdat.assign(var = var)], axis=0)
    
    sumdat_nc = (dat[(dat['model_type']=='pred_model_nc')]
        .groupby(['dat_type'])
        .agg({'NSE':[q25, q50, q75], 'KGE':[q25, q50, q75], 'mae':[q25, q50, q75]})
        .reset_index()
    )
    
    ## category line plots
    fig, ax = plt.subplots(1,2, sharex=True, sharey=True, figsize=(10,5))
    sns.lineplot(x='dist_lim', y=('NSE', 'q50'),
                 data=sumdat[(sumdat['dat_type']=='context')],
                 hue='pixpass', ax=ax[0])
    ax[0].axhline(y=sumdat_nc.loc[sumdat_nc['dat_type']=='context', ('NSE','q50')].values[0],
                  linestyle='--', color='k', linewidth=2)
    ax[0].set_title(f'NSE, Context sites, {var}')
    
    sns.lineplot(x='dist_lim', y=('NSE', 'q50'),
                 data=sumdat[(sumdat['dat_type']=='target')],
                 hue='pixpass', ax=ax[1])
    ax[1].axhline(y=sumdat_nc.loc[sumdat_nc['dat_type']=='target', ('NSE','q50')].values[0],
                  linestyle='--', color='k', linewidth=2)
    ax[1].set_title(f'NSE, Target sites, {var}')

    #plt.savefig(pltdir + f'/{var}_NSE_zoomedout_paramsweep.png', bbox_inches='tight')
    plt.savefig(pltdir + f'/{var}_NSE_zoomedin_paramsweep.png', bbox_inches='tight')
    plt.show()
    plt.close()
    
    
    fig, ax = plt.subplots(1,2, sharex=True, sharey=True, figsize=(10,5))
    sns.lineplot(x='dist_lim', y=('mae', 'q50'),
                 data=sumdat[(sumdat['dat_type']=='context')],
                 hue='pixpass', ax=ax[0])
    ax[0].axhline(y=sumdat_nc.loc[sumdat_nc['dat_type']=='context', ('mae','q50')].values[0],
                  linestyle='--', color='k', linewidth=2)
    ax[0].set_title(f'MAE, Context sites, {var}')
    
    sns.lineplot(x='dist_lim', y=('mae', 'q50'),
                 data=sumdat[(sumdat['dat_type']=='target')],
                 hue='pixpass', ax=ax[1])
    ax[1].axhline(y=sumdat_nc.loc[sumdat_nc['dat_type']=='target', ('mae','q50')].values[0],
                  linestyle='--', color='k', linewidth=2)
    ax[1].set_title(f'MAE, Target sites, {var}')

    #plt.savefig(pltdir + f'/{var}_MAE_zoomedout_paramsweep.png', bbox_inches='tight')
    plt.savefig(pltdir + f'/{var}_MAE_zoomedin_paramsweep.png', bbox_inches='tight')
    #plt.show()
    plt.close()

nse_res = allvars.set_index(['dat_type', 'var', 'dist_lim'])[[('NSE', 'q50')]]
nse_res = nse_res.loc['target'].reset_index()
nse_res.columns = ['var', 'dist_lim', 'val']
for var in ['TA', 'PA', 'SWIN', 'LWIN', 'WS', 'RH']:
    nse_res.loc[nse_res['var']==var, 'val'] = nse_res.loc[nse_res['var']==var, 'val'] / nse_res.loc[nse_res['var']==var, 'val'].max()

fig, ax = plt.subplots(1, figsize=(7,5))
sns.lineplot(x='dist_lim', y='val', hue='var',
             data=nse_res, ax=ax)
plt.show()

(nse_res.groupby('var')
    .apply(lambda x: x['dist_lim'][x['val'].idxmax()])
    .to_csv('./output/attn_dist_tests/var_dist_lims.csv', index=False)
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
