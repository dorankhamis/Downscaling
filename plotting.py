import numpy as np
import pandas as pd
import xarray as xr
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from adjustText import adjust_text
from matplotlib.colors import Normalize

from utils import *
from params import *

def to_np(arr):
    if type(arr)==type(torch.tensor(0)):
        return arr.cpu().numpy() 
    else:
        return arr

def density_scatter(x, y, ax=None, fig=None, sort=True, bins=20, 
                    cmap_name='hot', **kwargs):
    from matplotlib import colormaps    
    from scipy.interpolate import interpn
    
    if ax is None :
        fig , ax = plt.subplots()
    data , x_e, y_e = np.histogram2d(x, y, bins=bins, density=True)
    z = interpn((0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1])), data,
                np.vstack([x,y]).T, method = "splinef2d", bounds_error = False)

    # To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0

    # Sort the points by density, so that the densest points are plotted last
    if sort :
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]
    ax.scatter(x, y, c=z,  cmap=colormaps[cmap_name], **kwargs)
    return ax

def model_site_density_scatter(dat, var, modelname, year, timestep,
                               outname='', sites=None,
                               title_text='All sites', save=True):
    pltdat = (dat[['SITE_ID', var, modelname]].dropna()
        .reset_index(drop=True)
        .set_index('SITE_ID')
    )
    if not sites is None:
        pltdat = pltdat.loc[np.intersect1d(sites, pltdat.index.unique())]
    fig, ax = plt.subplots(1, figsize=(7,6))    
    ax = density_scatter(x=pltdat[var], y=pltdat[modelname],
                         fig=fig, ax=ax, s=3.5, bins=30)
    xx = np.mean(ax.get_xlim())
    ax.axline((xx,xx), slope=1, linestyle='--', color='k')
    ax.set_title(f'{title_text}, {var} {year} {timestep}, model: {modelname}')
    if save:
        plt.savefig(pltdir + f'/{var}_{year}_{timestep}_scatter_{outname}.png', bbox_inches='tight')            
        plt.close()
    else:
        return fig, ax
        
def model_compare_site_density_scatter(dat, var, modelnames, year, timestep,
                                       outname='allsites', sites=None,
                                       title_text='All sites', save=True,
                                       pltdir='./'):
    if type(modelnames)==str:
        modelnames = [modelnames]
    if not type(modelnames)==list:
        modelnames = list(modelnames)
    
    pltdat = (dat[['SITE_ID', var] + modelnames].dropna()
        .reset_index(drop=True)
        .set_index('SITE_ID')
    )
    if not sites is None:
        pltdat = pltdat.loc[np.intersect1d(sites, pltdat.index.unique())]
        
    xx_lab = pltdat[var].values.flatten().min()
    yy_lab = pltdat[modelnames].values.flatten().max()
    
    if len(modelnames)<4:
        fig, ax = plt.subplots(1, len(modelnames), figsize=(6*len(modelnames),5),
                               sharex=True, sharey=True)
        for i in range(len(modelnames)):        
            ax[i] = density_scatter(x=pltdat[var], y=pltdat[modelnames[i]],
                                    fig=fig, ax=ax[i], s=3., bins=30)
            xx = np.mean(ax[i].get_xlim())
            ax[i].axline((xx,xx), slope=1, linestyle='--', color='k')
            ax[i].set_title(f'{title_text}, {var} {year} {timestep}, model: {modelnames[i]}')
            # add metrics
            mets = calc_metrics(pltdat, var, modelnames[i])
            newlab = f'NSE = {np.around(mets["NSE"], 3)}  KGE = {np.around(mets["KGE"], 3)}  MAE = {np.around(mets["mae"], 3)}'
            ax[i].text(xx_lab, yy_lab, newlab, fontsize = 10)
            ax[i].set_xlabel('Observed')
        ax[0].set_ylabel('Modelled')
    if len(modelnames)>=4:
        ncols = int(np.ceil(np.sqrt(len(modelnames))))
        nrows = int(np.ceil(len(modelnames) / ncols))
        fig, ax = plt.subplots(nrows, ncols, figsize=(6*ncols-1, 5*nrows-1),
                               sharex=True, sharey=True)
        for i in range(len(modelnames)):        
            ax[i//ncols, i%ncols] = density_scatter(x=pltdat[var], y=pltdat[modelnames[i]],
                                    fig=fig, ax=ax[i//ncols, i%ncols], s=3., bins=30)
            xx = np.mean(ax[i//ncols, i%ncols].get_xlim())
            ax[i//ncols, i%ncols].axline((xx,xx), slope=1, linestyle='--', color='k')
            ax[i//ncols, i%ncols].set_title(f'{title_text}, {var} {year} {timestep}, model: {modelnames[i]}')
            # add metrics
            mets = calc_metrics(pltdat, var, modelnames[i])
            newlab = f'NSE = {np.around(mets["NSE"], 3)}  KGE = {np.around(mets["KGE"], 3)}  MAE = {np.around(mets["mae"], 3)}'
            ax[i//ncols, i%ncols].text(xx_lab, yy_lab, newlab, fontsize = 10)            
        for j in range(nrows): ax[j,0].set_ylabel('Modelled')
        for j in range(ncols): ax[-1,j].set_xlabel('Observed')
        
    if save:
        plt.savefig(pltdir + f'/{var}_{timestep}_scatter_{outname}.png', bbox_inches='tight')            
        plt.close()
    else:
        return fig, ax

def plot_var_tiles(batch, pred, met_vars, b=0, v='TA', sea_mask=None,
                   trim_edges=False, norm=None):
    if sea_mask is None:
        sea_mask = np.ones(pred.shape)
        sea_mask = sea_mask[:,0:1,:,:]
    
    nplots = 4
    ncols = 2
    nrows = 2    
    fig, axs = plt.subplots(nrows, ncols)
    
    idx_m = met_vars.index(v)
    idx_c = idx_m    
    
    if not norm is None:
        cmap = cm.get_cmap('viridis')
        norm = Normalize(np.min(pred[b,idx_m,:,:]), np.max(pred[b,idx_m,:,:]))
        im = cm.ScalarMappable(norm=norm)

    # take grid average -- do this in a more intelligent way to ignore sea pixels?
    #grid_av_pred = torch.nn.functional.avg_pool2d(pred, data_pars.scale, stride=data_pars.scale)
    grid_av_pred = pool_4D_arr(pred, (data_pars.scale, data_pars.scale),
                               train_pars.batch_size, method='mean')        

    if trim_edges:
        toplot = trim(pred*sea_mask[b,:,:,:], data_pars.scale)[b,idx_m,:,:]
    else:
        toplot = (pred*sea_mask[b,:,:,:])[b,idx_m,:,:]
    axs[0,0].imshow(to_np(toplot)[::-1,:], cmap=cmap, norm=norm)
    axs[0,0].title.set_text(met_vars[idx_m])

    if idx_c != -99:
        if trim_edges:
            toplot = trim(batch.constraint_targets*sea_mask[b,:,:,:], data_pars.scale)[b,idx_c,:,:]
        else:
            toplot = (batch.constraint_targets*sea_mask[b,:,:,:])[b,idx_c,:,:]
        axs[0,1].imshow(to_np(toplot)[::-1,:], cmap=cmap, norm=norm)
        axs[0,1].title.set_text('Constraint')
    
        
    if trim_edges:
        toplot = trim(batch.coarse_inputs, 1)[b,idx_m,:,:]
    else:
        toplot = batch.coarse_inputs[b,idx_m,:,:]    
    axs[1,0].imshow(to_np(toplot)[::-1,:], cmap=cmap, norm=norm)
    axs[1,0].title.set_text('ERA5 input')
    
    
    if trim_edges:
        toplot = trim(grid_av_pred, 1)[b,idx_m,:,:]
    else:
        toplot = grid_av_pred[b,idx_m,:,:]    
    axs[1,1].imshow(to_np(toplot)[::-1,:], cmap=cmap, norm=norm)
    axs[1,1].title.set_text('Grid mean')
    
    for i in range(nrows):
        for j in range(ncols):
            axs[i,j].get_xaxis().set_visible(False)
            axs[i,j].get_yaxis().set_visible(False)
    
    if not norm is None:
        fig.colorbar(im, ax=axs.ravel().tolist())
    plt.show()

def plot_batch_tiles(batch, pred, fine_vars, met_vars,
                     b=0, sea_mask=None, trim_edges=False,
                     norm=None, constraints=True):
    if sea_mask is None:
        sea_mask = np.ones(pred.shape)
        sea_mask = sea_mask[:,0:1,:,:]
    
    n_fine = batch.fine_inputs.shape[1]
    n_vars = pred.shape[1]
    nplots = n_fine + 3 * n_vars
    if constraints:
        nplots += 2 * n_vars
    
    ncols = int(np.ceil(np.sqrt(nplots)))
    nrows = int(np.ceil(nplots / ncols))
    
    fig, axs = plt.subplots(nrows, ncols)
    cmap = cm.get_cmap('viridis')
    if not norm is None:
        norm = Normalize(-2.5, 2.5)
        im = cm.ScalarMappable(norm=norm)

    # take grid average -- do this in a more intelligent way to ignore sea pixels?
    #grid_av_pred = torch.nn.functional.avg_pool2d(pred, data_pars.scale, stride=data_pars.scale)
    grid_av_pred = pool_4D_arr(pred, (data_pars.scale, data_pars.scale),
                               pred.shape[0], method='mean')
    if constraints:
        constraint_av_pred = pool_4D_arr(batch.constraint_targets, (data_pars.scale, data_pars.scale),
                                         batch.constraint_targets.shape[0], method='mean')
    for i in range(n_fine):
        if trim_edges:
            toplot = trim(batch.fine_inputs, data_pars.scale)[b,i,:,:]
        else:
            toplot = batch.fine_inputs[b,i,:,:]
        #if type(toplot)==type(dummy_tensor): toplot.cpu().numpy()        
        axs[i//ncols, i%ncols].imshow(to_np(toplot)[::-1,:], cmap=cmap, norm=norm)
        axs[i//ncols, i%ncols].title.set_text(fine_vars[i])
    cx = i + 1
    for j in range(n_vars):
        if trim_edges:
            toplot = trim(pred*sea_mask[b,:,:,:], data_pars.scale)[b,j,:,:]
        else:
            toplot = (pred*sea_mask[b,:,:,:])[b,j,:,:]
        axs[(cx+j)//ncols, (cx+j)%ncols].imshow(to_np(toplot)[::-1,:], cmap=cmap, norm=norm)
        axs[(cx+j)//ncols, (cx+j)%ncols].title.set_text(met_vars[j])
    cx += j + 1
    for j in range(n_vars):
        if trim_edges:
            toplot = trim(batch.coarse_inputs, 1)[b,j,:,:]
        else:
            toplot = batch.coarse_inputs[b,j,:,:]
        axs[(cx+j)//ncols, (cx+j)%ncols].imshow(to_np(toplot)[::-1,:], cmap=cmap, norm=norm)
        axs[(cx+j)//ncols, (cx+j)%ncols].title.set_text('ERA5 ' + met_vars[j])
    cx += j + 1
    for j in range(n_vars):
        if trim_edges:
            toplot = trim(grid_av_pred, 1)[b,j,:,:]
        else:
            toplot = grid_av_pred[b,j,:,:]        
        axs[(cx+j)//ncols, (cx+j)%ncols].imshow(to_np(toplot)[::-1,:], cmap=cmap, norm=norm)
        axs[(cx+j)//ncols, (cx+j)%ncols].title.set_text('Grid mean ' + met_vars[j])
    cx += j + 1
    if constraints:
        for j in range(n_vars):
            if trim_edges:
                toplot = trim(batch.constraint_targets, data_pars.scale)[b,j,:,:]
            else:
                toplot = batch.constraint_targets[b,j,:,:]        
            axs[(cx+j)//ncols, (cx+j)%ncols].imshow(to_np(toplot)[::-1,:], cmap=cmap, norm=norm)
            axs[(cx+j)//ncols, (cx+j)%ncols].title.set_text('Hi-res constraint ' + met_vars[j])
        cx += j + 1
        for j in range(n_vars):
            if trim_edges:
                toplot = trim(constraint_av_pred, 1)[b,j,:,:]
            else:
                toplot = constraint_av_pred[b,j,:,:]        
            axs[(cx+j)//ncols, (cx+j)%ncols].imshow(to_np(toplot)[::-1,:], cmap=cmap, norm=norm)
            axs[(cx+j)//ncols, (cx+j)%ncols].title.set_text('Grid mean of constraint ' + met_vars[j])
    for i in range(nrows):
        for j in range(ncols):
            axs[i,j].get_xaxis().set_visible(False)
            axs[i,j].get_yaxis().set_visible(False)
    if not norm is None:
        fig.colorbar(im, ax=axs.ravel().tolist())
    plt.show()

def plot_singlevar_preds_vs_site_data(preds, b, batch, station_targets, var,
                                      site_type='context', model_names=None):
    if not type(preds)==list:
        preds = [preds]
    if model_names is None:
        model_names = [f'model_pred_{kk}' for kk in range(len(preds))]    
    yinds = station_targets[b][site_type].sub_y.values[None,...]
    xinds = station_targets[b][site_type].sub_x.values[None,...]   
    #site_preds = pred[b, 0, yinds, xinds]
    site_preds = [pd.DataFrame(to_np(p)[b, 0, yinds, xinds]).melt() for p in preds]
    for kk, sp in enumerate(site_preds):
        sp.index = station_targets[b][site_type].index
        sp.columns = ['id', model_names[kk]]    
    joined = station_targets[b][site_type]
    for sp in site_preds:
        joined = joined.merge(sp, on='SITE_ID', how='left')    
    joined = joined[[var] + model_names].reset_index().melt(id_vars='SITE_ID')
    #sns.scatterplot(x='SITE_ID', y='value', hue='variable', data=joined).legend()
    
    fig, ax = plt.subplots()
    sns.pointplot(x='SITE_ID', y='value', hue='variable', data=joined,
                    dodge=0.1, join=False, ax=ax).legend()
    plt.xticks(rotation = 90)
    for points in ax.collections: 
        size = points.get_sizes().item()
        new_sizes = [size * 0.33 for name in ax.get_yticklabels()]
        points.set_sizes(new_sizes)
    plt.show()

def plot_context_and_target_preds(preds, b, batch, station_targets, var,
                                  model_names=None):
    if not type(preds)==list:
        preds = [preds]
    if model_names is None:
        model_names = [f'model_pred_{kk}' for kk in range(len(preds))]
    fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    output = {}
    for axi, site_type in enumerate(['context', 'target']):
        yinds = station_targets[b][site_type].sub_y.values[None,...]
        xinds = station_targets[b][site_type].sub_x.values[None,...]
        site_preds = [pd.DataFrame(to_np(p)[b, 0, yinds, xinds]).melt() for p in preds]
        for kk, sp in enumerate(site_preds):
            sp.index = station_targets[b][site_type].index
            sp.columns = ['id', model_names[kk]]    
        joined = station_targets[b][site_type]
        for sp in site_preds:
            joined = joined.merge(sp, on='SITE_ID', how='left')    
        joined = joined[[var] + model_names].reset_index().melt(id_vars='SITE_ID')
        output[site_type] = joined
        if joined.shape[0]>0:
            sns.pointplot(x='SITE_ID', y='value', hue='variable', data=joined,
                            dodge=0.1, join=False, ax=ax[axi]).legend()        
        ax[axi].set_xticklabels(ax[axi].get_xticklabels(), rotation=45,
                                ha='right', rotation_mode='anchor')
        ax[axi].set_title(site_type)
        for points in ax[axi].collections: 
            size = points.get_sizes().item()
            new_sizes = [size * 0.33 for name in ax[axi].get_yticklabels()]
            points.set_sizes(new_sizes)
    plt.show()
    return output

def plot_preds_vs_site_data(preds, b, batch, station_targets,
                            site_type='context', model_names=None):
    if not type(preds)==list:
        preds = [preds]
    if model_names is None:
        model_names = [f'model_pred_{kk}' for kk in range(len(preds))]
    if site_type=='context':
        yx = to_np(batch.context_station_dict['coords_yx'][b])
    elif site_type=='target':
        yx = to_np(batch.target_station_dict['coords_yx'][b])
    nplots = yx.shape[0]
    ncols = int(np.ceil(np.sqrt(nplots)))
    nrows = int(np.ceil(nplots / ncols))
    fig, axs = plt.subplots(nrows, ncols)
    ii = 0
    jj = 0
    for i in range(yx.shape[0]):        
        site_preds = [pd.DataFrame(to_np(p)[b:(b+1),:, yx[i,0], yx[i,1]]) for p in preds]
        for sp in site_preds: sp.columns = met_vars
        if site_type=='context':
            site_obs = pd.DataFrame(to_np(batch.context_station_dict['values'][b])[i:(i+1),:])
        elif site_type=='target':
            site_obs = pd.DataFrame(to_np(batch.target_station_dict['values'][b])[i:(i+1),:])
        site_obs.columns = met_vars
        site_obs = site_obs.assign(SITE_ID = station_targets[b][site_type].index[i],
                                   dat_type='site_obs')
        site_preds = [sp.assign(SITE_ID = station_targets[b][site_type].index[i],
                                dat_type = model_names[kk]) for kk, sp in enumerate(site_preds)]
        print(site_obs)
        site_preds = [sp.melt(id_vars=['SITE_ID','dat_type']) for sp in site_preds]
        compare = pd.concat([site_obs.melt(id_vars=['SITE_ID','dat_type'])] + site_preds, axis=0)
        ii = i // ncols
        jj = i % ncols
        if i==0:
            sns.pointplot(x='variable', y='value', hue='dat_type',
                          join=False, data=compare, ax=axs[ii,jj],
                          dodge=True, scale=0.5).legend()
        else:
            sns.pointplot(x='variable', y='value', hue='dat_type',
                          join=False, data=compare, ax=axs[ii,jj],
                          dodge=True, scale=0.5)
        axs[ii,jj].set_title(station_targets[b][site_type].index[i])
        axs[ii,jj].set_xlabel('')
    # sort out legend
    handles, labels = axs[0,0].get_legend_handles_labels()
    [[c.get_legend().remove() for c in r if (not c.get_legend() is None)] for r in axs]
    fig.legend(handles, labels, loc='upper right')
    plt.show()

def extract_site_predictions(preds, station_targets, batch, met_vars, model_names=None):
    if not type(preds)==list:
        preds = [preds]
    if model_names is None:
        model_names = [f'model_pred_{kk}' for kk in len(preds)]
    df_out = pd.DataFrame()
    for b in range(preds[0].shape[0]):
        for site_type in ['context', 'target']:
            if site_type=='context':
                yx = to_np(batch.context_station_dict['coords_yx'][b])
            elif site_type=='target':
                yx = to_np(batch.target_station_dict['coords_yx'][b])        
            for i in range(yx.shape[0]):        
                site_preds = [pd.DataFrame(to_np(p)[b:(b+1),:, yx[i,0], yx[i,1]]) for p in preds]
                for sp in site_preds: sp.columns = met_vars
                if site_type=='context':
                    site_obs = pd.DataFrame(to_np(batch.context_station_dict['values'][b])[i:(i+1),:])
                elif site_type=='target':
                    site_obs = pd.DataFrame(to_np(batch.target_station_dict['values'][b])[i:(i+1),:])
                site_obs.columns = met_vars
                site_obs = site_obs.assign(SITE_ID = station_targets[b][site_type].index[i],
                                           dat_type='site_obs')
                site_preds = [sp.assign(SITE_ID = station_targets[b][site_type].index[i],
                                        dat_type = model_names[kk]) for kk, sp in enumerate(site_preds)]
                site_preds = [sp.melt(id_vars=['SITE_ID','dat_type']) for sp in site_preds]
                compare = pd.concat([site_obs.melt(id_vars=['SITE_ID','dat_type'])] + site_preds, axis=0)
                df_out = pd.concat([df_out, compare.assign(site_type = site_type)], axis=0)
    return df_out

def plot_station_locations(station_targets, fine_inputs, b, labels=True,
                           plot_context=True, plot_target=True):
    # generalise this for any tile size!
    #dim_h = data_pars.dim_l * data_pars.scale
    #station_targets[b]['context']['adj_y'] = dim_h - station_targets[b]['context'].sub_y
    #station_targets[b]['target']['adj_y'] = dim_h - station_targets[b]['target'].sub_y
    station_targets[b]['context']['adj_y'] = fine_inputs.shape[-2] - station_targets[b]['context'].sub_y
    station_targets[b]['target']['adj_y'] = fine_inputs.shape[-2] - station_targets[b]['target'].sub_y

    fig, ax = plt.subplots()
    ax.imshow(to_np(fine_inputs)[b, 0, ::-1, :], alpha=0.6, cmap='Greys')
    if plot_context:
        ax.scatter(station_targets[b]['context'].sub_x, station_targets[b]['context'].adj_y,
                   s=18, c='#1f77b4', marker='s')
    if plot_target:
        ax.scatter(station_targets[b]['target'].sub_x, station_targets[b]['target'].adj_y,
                   s=18, c='#17becf', marker='o')
    if labels:
        texts = []
        if plot_context:
            texts += [plt.text(station_targets[b]['context'].sub_x.iloc[i],
                               station_targets[b]['context'].adj_y.iloc[i],
                               station_targets[b]['context'].index[i], fontsize=9) 
                                    for i in range(station_targets[b]['context'].shape[0])]
        if plot_target:
            texts += [plt.text(station_targets[b]['target'].sub_x.iloc[i],
                               station_targets[b]['target'].adj_y.iloc[i],
                               station_targets[b]['target'].index[i], fontsize=9) 
                                    for i in range(station_targets[b]['target'].shape[0])]
        adjust_text(texts, arrowprops=dict(arrowstyle="-", color='k', lw=0.5))
    plt.axis('off')
    plt.show()
