import pandas as pd
import numpy as np
import torch
import shutil
import warnings
import datetime

from pathlib import Path
from sklearn.neighbors import NearestNeighbors
import sklearn.metrics as metrics
import torch.nn as nn

from downscaling.params import normalisation as nm
from downscaling.params import data_pars, model_pars

EPS = 1e-30

def trim(tensor, scale):
    return tensor[:,:,scale:-scale,scale:-scale]

def unnormalise(dat, var, names):
    if type(names)==str:
        names = [names]
    if not type(names)==list:
        names = list(names)
    if var=='TA':
        dat[[var]+names] = dat[[var]+names] * nm.temp_sd + nm.temp_mu
    elif var=='PA':
        dat[[var]+names] = dat[[var]+names] * nm.p_sd + nm.p_mu
    if var=='SWIN':
        dat[[var]+names] = dat[[var]+names] * nm.swin_norm
    if var=='LWIN':
        dat[[var]+names] = dat[[var]+names] * nm.lwin_sd + nm.lwin_mu
    if var=='WS':
        dat[[var]+names] = dat[[var]+names] * nm.ws_sd + nm.ws_mu
        dat[['UX']+names] = dat[['UX']+names] * nm.ws_sd
        dat[['VY']+names] = dat[['VY']+names] * nm.ws_sd
    if var=='RH':
        dat[[var]+names] = dat[[var]+names] * nm.rh_sd + nm.rh_mu
    if var=='DTR': # daily temp range
        dat[[var]+names] = dat[[var]+names] * nm.temp_sd
    return dat
    
def unnormalise_img(dat, var):
    if var=='TA':
        dat = dat * nm.temp_sd + nm.temp_mu
    elif var=='PA':
        dat = dat * nm.p_sd + nm.p_mu
    if var=='SWIN':
        dat = dat * nm.swin_norm
    if var=='LWIN':
        dat = dat * nm.lwin_sd + nm.lwin_mu
    if var=='WS':
        dat = dat * nm.ws_sd + nm.ws_mu
    if var=='UX' or var=='VY':
        dat = dat * nm.ws_sd
    if var=='RH':
        dat = dat * nm.rh_sd + nm.rh_mu
    if var=='DTR': # daily temp range
        dat = dat * nm.temp_sd
    return dat

def decide_scale_factor(fsize, csize, scale):
    sf_1 = (fsize[-1]/scale) / csize[-1]
    sf_2 = (fsize[-2]/scale) / csize[-2]
    
    a11 = (sf_1 * csize[-1]) // 1
    a12 = (sf_1 * csize[-2]) // 1
    
    a21 = (sf_2 * csize[-1]) // 1
    a22 = (sf_2 * csize[-2]) // 1
    
    if sf_2 == sf_1:
        return sf_1 # take either
    elif sf_2 > sf_1:
        if (a11 == a21) and (a12 == a22):
            return sf_1 # take either
        elif (a11 < a21) and (a12 == a22):
            return sf_1 # unwanted change, take the smaller
        elif (a11 == a21) and (a12 < a22):
            return sf_2 # desired change, take the larger
        else:
            w = 0.5            
            wup = 1
            wdown = 0
            while True:
                sf = (w*sf_1 + (1-w)*sf_2)
                aw1 = (sf * csize[-1]) // 1
                aw2 = (sf * csize[-2]) // 1
                if aw1 == a11 and aw2 == a22:
                    return sf
                elif aw1 == a11 and aw2 < a22:
                    wdown = w
                    w = 0.5*(wup - w)
                elif aw1 > a11 and aw2 == a22:
                    wup = w
                    w = 0.5*(w - wdown)
    elif sf_1 > sf_2:
        if (a11 == a21) and (a12 == a22):
            return sf_1
        elif (a11 == a21) and (a12 > a22):
            return sf_2 # unwanted change, take the smaller
        elif (a11 > a21) and (a12 == a22):
            return sf_1 # desired change, take the larger
        else:
            w = 0.5
            wup = 1
            wdown = 0
            while True:
                sf = (w*sf_1 + (1-w)*sf_2)
                aw1 = (sf * csize[-1]) // 1
                aw2 = (sf * csize[-2]) // 1
                if aw1 == a11 and aw2 == a22:
                    return sf
                elif aw1 < a11 and aw2 == a22:
                    wdown = w
                    w = 0.5*(wup - w)
                elif aw1 == a11 and aw2 > a22:
                    wup = w
                    w = 0.5*(w - wdown)

def find_num_pools(scale, max_num=5, factor=3, threshold=1.5):
    for i in range(max_num):
        val = scale/(factor**i)
        if val<=threshold: break
    return max(0, i-1)

def off_grid_site_lat_lon(raw_site_lats, raw_site_lons,
                          raw_grid_lats, raw_grid_lons,
                          grid_yx_flat, raw_H, raw_W):
    # get off-grid lat lon of sites
    lls = pd.concat([pd.Series(raw_site_lats) / nm.lat_norm, pd.Series(raw_site_lons) / nm.lon_norm], axis=1).values
    latlon_grid = np.stack([raw_grid_lats / nm.lat_norm, raw_grid_lons / nm.lon_norm], axis=0)
    latlon_grid = np.reshape(latlon_grid, (2, raw_H*raw_W)).T #self.dim_h*self.dim_h
    neigh = NearestNeighbors(n_neighbors=4)
    neigh.fit(latlon_grid)
    dists, inds = neigh.kneighbors(X=lls)    
    weights = 1 - (dists / dists.sum(axis=1, keepdims=True))
    site_yx = np.stack([(grid_yx_flat[inds][s] * weights[s][...,None]).sum(axis=0) / weights[s].sum() for s in range(weights.shape[0])])
    site_yx = site_yx.astype(np.float32)
    return site_yx

def mask_to_softmask(distances, dist_lim, dist_lim_far=None, attn_eps=None,
                     poly_exp=None, diminish_model="polynomial"):    
    diminish_mask = distances > dist_lim
    softmask = np.ones(distances.shape, dtype=np.float32)
    if diminish_model=="gaussian":
        attn_sigmasq = -(dist_lim_far - dist_lim)**2 / np.log(attn_eps)
        softmask[diminish_mask] = np.exp(- (distances[diminish_mask] - dist_lim)**2 / attn_sigmasq)
    elif diminish_model=="polynomial":
        softmask[diminish_mask] = 1. / (distances[diminish_mask] / dist_lim)**poly_exp
    # but we want to add to scores in exponentiated space (before the softmax),
    # so take log, meaning the softmask when distance=0 is 0 (log(1)==0) 
    # and it is negative for finite separation distances.
    return np.log(softmask + EPS)

def build_soft_masks(softmask, scale_factors, device,
                     var=None, cntxt_stats=None,
                     fine_inputs=None, fine_variable_order=None):
    if type(var)==list: var=var[0]
    context_soft_masks = [torch.clone(softmask)]
    for i, sf in enumerate(scale_factors):
        context_soft_masks.append(nn.functional.interpolate(
            context_soft_masks[i],
            scale_factor=sf,
            mode='bilinear')
        )
    
    # apply shade / cloud affect on high-resolution soft masking
    if var=='SWIN' or var=='LWIN' or var=='PRECIP':
        cloud_ind = fine_variable_order.index('cloud_cover')
        if var=='SWIN':
            shade_ind = fine_variable_order.index('shade_map')
            illum_ind = fine_variable_order.index('illumination_map')
    
        for i in range(cntxt_stats.shape[0]):
            if var=='SWIN':
                solar_dist = (
                    (fine_inputs[:,cloud_ind,:,:] - cntxt_stats.iloc[i].cloud_cover).square() + 
                    (fine_inputs[:,shade_ind,:,:] - cntxt_stats.iloc[i].shade_map).square() + 
                    (fine_inputs[:,illum_ind,:,:] - cntxt_stats.iloc[i].illumination_map).square()
                ).sqrt()
            else:
                solar_dist = (fine_inputs[:,cloud_ind,:,:] - cntxt_stats.iloc[i].cloud_cover).abs()
            
            context_soft_masks[-1][:,i,:,:] = context_soft_masks[-1][:,i,:,:] - 5.*solar_dist
            
            # interpolate for lower res masks
            solar_dists_sf = []
            for j in range(2, len(scale_factors)+2):
                context_soft_masks[-j][:,i:(i+1),:,:] = (context_soft_masks[-j][:,i:(i+1),:,:] -
                    5 * nn.functional.interpolate(
                        solar_dist[None,...],
                        size=context_soft_masks[-j][0,i,:,:].shape,
                        mode='bilinear'
                    )
                )

    context_soft_masks = [torch.transpose(
            torch.reshape(mm, (mm.shape[0], mm.shape[1], mm.shape[2]*mm.shape[3])), -2, -1
        )[None,...].to(device) for mm in context_soft_masks
    ]
    return context_soft_masks

def build_pixel_passers(distances, scale_factors, dist_lim_pixpass, pass_exp, device):
    min_dists = distances.min(dim=1)[0]
    min_dists[min_dists < dist_lim_pixpass] = 1.
    min_dists[min_dists >= dist_lim_pixpass] = 1. / (min_dists[min_dists >= dist_lim_pixpass] / dist_lim_pixpass)**pass_exp
    pixel_passers = [min_dists.unsqueeze(0)]
    for i, sf in enumerate(scale_factors):
        pixel_passers.append(nn.functional.interpolate(
            pixel_passers[i],
            scale_factor=sf,
            mode='bilinear')
        )
    return pixel_passers

def build_binary_masks(distances, scale_factors, dist_lim_far, device):
    hardmask = distances < dist_lim_far
    context_binary_masks = [hardmask.to(torch.float32)]
    for i, sf in enumerate(scale_factors):            
        context_binary_masks.append(nn.functional.interpolate(
            context_binary_masks[i],
            scale_factor=sf,
            mode='bilinear')
        )
    # re-bool
    context_binary_masks = [m > 0.5 for m in context_binary_masks]
    context_binary_masks = [
        torch.transpose(
            torch.reshape(mm, (mm.shape[0], mm.shape[1], mm.shape[2]*mm.shape[3])), -2, -1
        )[None,...].to(device) for mm in context_binary_masks
    ]
    return context_binary_masks

def prepare_attn(model, batch, site_meta, fine_grid,
                 context_sites=None, b=0,
                 dist_lim=80, dist_lim_far=130,
                 attn_eps=1e-6, poly_exp=4,
                 diminish_model="polynomial"):
    # create the flat YX grid for attention
    raw_H = batch.fine_inputs.shape[-2]
    raw_W = batch.fine_inputs.shape[-1]
    X1 = np.where(np.ones((raw_H, raw_W)))
    X1 = np.hstack([X1[0][...,np.newaxis],
                    X1[1][...,np.newaxis]])

    # build context masks for all possible context sites    
    if context_sites is None:
        cntxt_stats = (site_meta
            .set_index('SITE_ID')
            .loc[batch.raw_station_dict[b]['context'].index]
        )
    else:
        # use all sites
        cntxt_stats = (site_meta
            .set_index('SITE_ID')
            .loc[context_sites]
        )
    cntxt_stats = cntxt_stats.assign(s_idx = np.arange(cntxt_stats.shape[0]))
    
    # check if we have sites
    if cntxt_stats.shape[0]==0:
        return None, None, None, None, None
    
    if fine_grid is None:
        fine_lats = batch.fine_inputs[b,-2,:,:].cpu().numpy() * nm.lat_norm
        fine_lons = batch.fine_inputs[b,-1,:,:].cpu().numpy() * nm.lon_norm
    else:
        fine_lats = fine_grid.lat.isel(x=batch.batch_metadata[b]['x_inds'],
                                       y=batch.batch_metadata[b]['y_inds']).values,
        fine_lons = fine_grid.lon.isel(x=batch.batch_metadata[b]['x_inds'],
                                       y=batch.batch_metadata[b]['y_inds']).values,
    
    site_yx = off_grid_site_lat_lon(
        cntxt_stats.LATITUDE,
        cntxt_stats.LONGITUDE,
        fine_lats,
        fine_lons,
        X1, raw_H, raw_W
    )

    raw_H_c = batch.coarse_inputs.shape[-2]
    raw_W_c = batch.coarse_inputs.shape[-1]
    X2 = np.where(np.ones((raw_H_c, raw_W_c)))
    X2 = np.hstack([X2[0][...,np.newaxis],
                    X2[1][...,np.newaxis]])
    grid_yx_flat = (X2 * data_pars.scale) + data_pars.scale/2 # centre yx km of 28km pixels    

    # do knn at most coarse res to save memory
    # therefore grid_yx_flat should be at raw ERA5 (y,x)
    nbrs = NearestNeighbors(n_neighbors = site_yx.shape[0], algorithm='ball_tree').fit(site_yx)
    distances, indices = nbrs.kneighbors(grid_yx_flat)
    distances = np.take_along_axis(distances, np.argsort(indices, axis=1), axis=1)
    
    distances = np.reshape(distances.T, (site_yx.shape[0], raw_H_c, raw_W_c))
    softmask = mask_to_softmask(
        distances,
        dist_lim=dist_lim,
        dist_lim_far=dist_lim_far,
        attn_eps=attn_eps,
        poly_exp=poly_exp,
        diminish_model=diminish_model
    )
    
    distances = torch.from_numpy(distances)[None,...].to(torch.float32).to(batch.coarse_inputs.device)
    softmask = torch.from_numpy(softmask)[None,...].to(torch.float32).to(batch.coarse_inputs.device)
    
    scale_factors = [3 for i in range(model.nups)] + [raw_H / (3**model.nups * raw_H_c)]
    return distances, softmask, scale_factors, site_yx, cntxt_stats
    
def subset_context_masks(batch, context_sites, context_masks, b=0):
    bsites = np.array(batch.raw_station_dict[b]['context'].index)
    #s_idxs = cntxt_stats.assign(s_idx = np.arange(cntxt_stats.shape[0])).loc[bsites].s_idx.values
    s_idxs = context_sites.loc[bsites].s_idx.values # s_idx already assigned
    batch_masks = [m[:,:,:,s_idxs] for m in context_masks]
    return batch_masks

def create_attention_masks(model, batch, var,
                           dist_lim = None,
                           dist_lim_far = None,
                           attn_eps = None,
                           poly_exp = None,
                           diminish_model = None,
                           dist_pixpass = None,
                           pass_exp = None):
    '''
    Output format is:
        outer list of length n_resolutions
        inner lists of length batch_size
        containing tensors of pixels x pixels or pixels x context sites
    '''
    if type(var)==str: var = [var]    
    if dist_lim is None: 
        dist_lim = model_pars.dist_lim[var[0]]
    if dist_lim_far is None:
        dist_lim_far = model_pars.dist_lim_far
    if attn_eps is None:
        attn_eps=model_pars.attn_eps
    if poly_exp is None:
        poly_exp = model_pars.poly_exp[var[0]]
    if diminish_model is None:
        diminish_model = model_pars.diminish_model
    if dist_pixpass is None:
        dist_pixpass = model_pars.dist_pixpass[var[0]]
    if pass_exp is None:
        pass_exp = model_pars.pass_exp[var[0]]
    
    attn_masks = {
        'context_soft_masks':[None, None, None, None],
        'pixel_passers':     [None, None, None, None],
        'context_masks':     [None, None, None, None]
    }
    for b in range(len(batch.batch_metadata)):
        site_meta = (pd.concat([batch.raw_station_dict[b]['context'],
                                batch.raw_station_dict[b]['target']], axis=0)
            .reset_index()
        )
        distances, softmask, scale_factors, site_yx, cntxt_stats = prepare_attn(
            model,
            batch,
            site_meta,
            None,
            context_sites=None,
            b=b,
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
        
        # if no context stations just return
        if not (distances is None):
            if model_pars.soft_masks:
                # do we want to allow attention head-dependent 
                # scaling of the soft masks as per ALiBi
                # https://arxiv.org/pdf/2108.12409.pdf
                masks['context_soft_masks'] = build_soft_masks(
                    softmask,
                    scale_factors,
                    batch.coarse_inputs.device,
                    var=var,
                    cntxt_stats=cntxt_stats,
                    fine_inputs=batch.fine_inputs,
                    fine_variable_order=batch.batch_metadata[b]['fine_var_order']
                )
            if model_pars.pixel_pass_masks:        
                masks['pixel_passers'] = build_pixel_passers(
                    distances,
                    scale_factors,
                    dist_pixpass,
                    pass_exp,
                    batch.coarse_inputs.device
                )
                # reshaping is done in the model...
            if model_pars.binary_masks:
                masks['context_masks'] = build_binary_masks(
                    distances,
                    scale_factors,
                    dist_lim_far,
                    batch.coarse_inputs.device
                )            

        for mtype in masks.keys():
            for ll in range(len(masks[mtype])):
                if b==0:
                    if masks[mtype][ll] is None:
                        attn_masks[mtype][ll] = [None]
                    else:
                        attn_masks[mtype][ll] = [masks[mtype][ll][0]]
                else:
                    if masks[mtype][ll] is None:
                        attn_masks[mtype][ll].append(None)
                    else:
                        attn_masks[mtype][ll].append(masks[mtype][ll][0])
    return attn_masks

def pooling(mat, ksize, method='max', pad=False):
    # from https://stackoverflow.com/questions/42463172/how-to-perform-max-mean-pooling-on-a-2d-array-using-numpy
    '''Non-overlapping pooling on 2D or 3D data.

    <mat>: ndarray, input array to pool.
    <ksize>: tuple of 2, kernel size in (ky, kx).
    <method>: str, 'max for max-pooling, 
                   'mean' for mean-pooling.
    <pad>: bool, pad <mat> or not. If no pad, output has size
           n//f, n being <mat> size, f being kernel size.
           if pad, output has size ceil(n/f).

    Return <result>: pooled matrix.
    '''

    m, n = mat.shape[:2]
    ky, kx = ksize

    _ceil = lambda x,y: int(np.ceil(x/float(y)))

    if pad:
        ny = _ceil(m,ky)
        nx = _ceil(n,kx)
        size = (ny*ky, nx*kx) + mat.shape[2:]
        mat_pad = np.full(size,np.nan)
        mat_pad[:m,:n,...] = mat
    else:
        ny = m//ky
        nx = n//kx
        mat_pad = mat[:ny*ky, :nx*kx, ...]

    new_shape = (ny,ky,nx,kx) + mat.shape[2:]
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if method=='max':
            result = np.nanmax(mat_pad.reshape(new_shape),axis=(1,3))
        else:
            result = np.nanmean(mat_pad.reshape(new_shape),axis=(1,3))

    return result


def pool_4D_arr(arr, kernel, batch_size, method='mean', pad=False):    
    return np.stack(
        [pooling(arr[b,:,:,:].T, kernel, method=method, pad=pad).T for b in range(batch_size)]
    )

def asStride(arr, sub_shape, stride):
    '''Get a strided sub-matrices view of an ndarray.
    See also skimage.util.shape.view_as_windows()
    '''
    s0,s1 = arr.strides[:2]
    m1,n1 = arr.shape[:2]
    m2,n2 = sub_shape
    view_shape = (1+(m1-m2)//stride[0], 1+(n1-n2)//stride[1], m2, n2) + arr.shape[2:]
    strides = (stride[0]*s0, stride[1]*s1, s0, s1) + arr.strides[2:]
    subs = np.lib.stride_tricks.as_strided(arr, view_shape, strides=strides)
    return subs

def poolingOverlap(mat,ksize,stride=None,method='max',pad=False):
    '''Overlapping pooling on 2D or 3D data.

    <mat>: ndarray, input array to pool.
    <ksize>: tuple of 2, kernel size in (ky, kx).
    <stride>: tuple of 2 or None, stride of pooling window.
              If None, same as <ksize> (non-overlapping pooling).
    <method>: str, 'max for max-pooling,
                   'mean' for mean-pooling.
    <pad>: bool, pad <mat> or not. If no pad, output has size
           (n-f)//s+1, n being <mat> size, f being kernel size, s stride.
           if pad, output has size ceil(n/s).

    Return <result>: pooled matrix.
    '''

    m, n = mat.shape[:2]
    ky,kx = ksize
    if stride is None:
        stride = (ky,kx)
    sy,sx = stride

    _ceil = lambda x,y: int(numpy.ceil(x/float(y)))

    if pad:
        ny = _ceil(m,sy)
        nx = _ceil(n,sx)
        size = ((ny-1)*sy+ky, (nx-1)*sx+kx) + mat.shape[2:]
        mat_pad = np.full(size, numpy.nan)
        mat_pad[:m,:n,...] = mat
    else:
        mat_pad = mat[:(m-ky)//sy*sy+ky, :(n-kx)//sx*sx+kx, ...]

    view = asStride(mat_pad, ksize, stride)

    if method=='max':
        result = np.nanmax(view,axis=(2,3))
    else:
        result = np.nanmean(view,axis=(2,3))

    return result

def zeropad_strint(integer, num=1):
    if num==1:
        if integer<10:
            return '0' + str(integer)
        else:
            return str(integer)
    elif num==2:
        if integer<10:
            return '00' + str(integer)
        elif integer<100:
            return '0' + str(integer)
        else:
            return str(integer)

def find_chess_tile(lat, lon, latlon_ref):
    # assumes equal length lat/lon vectors in latlon_ref
    dist_diff = np.sqrt(np.square(latlon_ref.lat.values - lat) +
                        np.square(latlon_ref.lon.values - lon))
    chesstile_yx = np.where(dist_diff == np.min(dist_diff))
    return chesstile_yx

def frange_cycle_linear(n_iter, start=0.0, stop=1.0, n_cycle=4, ratio=0.5):
    # varying KL weight in cycles
    L = np.ones(n_iter) * stop
    period = n_iter/n_cycle
    step = (stop-start)/(period*ratio) # linear schedule

    for c in range(n_cycle):
        v , i = start , 0
        while v <= stop and (int(i+c*period) < n_iter):
            L[int(i+c*period)] = v
            v += step
            i += 1
    return L

def prepare_run(checkpoint):
    if checkpoint is None:
        curr_epoch = 0
        best_loss = np.inf
        losses = []
        val_losses = []
    else:
        curr_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['best_loss']
        torch.random.set_rng_state(checkpoint['torch_random_state'])
        np.random.set_state(checkpoint['numpy_random_state'])
        try:
            losses = checkpoint['losses']
            val_losses = checkpoint['val_losses']
        except:
            losses = []
            val_losses = []
    return losses, val_losses, curr_epoch, best_loss

def save_checkpoint(state, is_best, outdir):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    f_path = outdir + '/checkpoint.pth'
    torch.save(state, f_path)
    if is_best:
        print ("=> Saving a new best")
        best_fpath = outdir + '/best_model.pth'
        shutil.copyfile(f_path, best_fpath)
    else:
        print ("=> Validation loss did not improve")

def load_checkpoint(checkpoint_fpath, model, optimizer, device):
    if device.type=='cpu':
        checkpoint = torch.load(checkpoint_fpath, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint.pop('state_dict'))
    if not optimizer is None:
        optimizer.load_state_dict(checkpoint.pop('optimizer'))  
    return model, optimizer, checkpoint

def setup_checkpoint(model, optimizer, device, load_prev_chkpnt,
                     model_outdir, log_dir, specify_chkpnt=None,
                     reset_chkpnt=False):
    if load_prev_chkpnt:
        if specify_chkpnt is None:
            loadmodel = model_outdir + 'best_model.pth'
            print('Loading best checkpoint...')
        else:
            ## to load different weights to begin        
            # specify_chkpnt of form "modelname/checkpoint.pth" or 
            # "OTHERMODEL/best_model.pth" or "OTHERMODEL/checkpoint.pth"
            loadmodel = f'{log_dir}/{specify_chkpnt}'
            print(f'Loading {log_dir}/{specify_chkpnt}...')
        try:
            model, optimizer, checkpoint = load_checkpoint(loadmodel, model, optimizer, device)
            print('Loaded checkpoint successfully')
            print(f'Best loss: {checkpoint["best_loss"]}')
            print(f'current epoch: {checkpoint["epoch"]}')
        except:
            print('Failed loading checkpoint')
            checkpoint = None
    else: 
      checkpoint = None
      loadmodel = None

    if reset_chkpnt is True:        
        checkpoint = None # adding this to reset best loss and loss trajectory
    
    return model, optimizer, checkpoint

def rand_cmap(nlabels, ctype='bright', first_color_black=True, last_color_black=False):
    from matplotlib.colors import LinearSegmentedColormap
    import colorsys    
    if ctype not in ('bright', 'soft'):
        print ('Please choose "bright" or "soft" for type')
        return
        
    # Generate color map for bright colors, based on hsv
    if ctype == 'bright':
        randHSVcolors = [(np.random.uniform(low=0.0, high=1),
                          np.random.uniform(low=0.2, high=1),
                          np.random.uniform(low=0.9, high=1)) for i in range(nlabels)]
        # Convert HSV list to RGB
        randRGBcolors = []
        for HSVcolor in randHSVcolors:
            randRGBcolors.append(colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2]))
            
    # Generate soft pastel colors, by limiting the RGB spectrum
    if ctype == 'soft':
        low = 0.6
        high = 0.95
        randRGBcolors = [(np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high)) for i in range(nlabels)]
    if first_color_black:
        randRGBcolors[0] = [0, 0, 0]
    if last_color_black:
        randRGBcolors[-1] = [0, 0, 0]
    random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)
    return random_colormap

def generate_random_date(year_range, utc=True):
    start_date = pd.to_datetime(f'{np.min(year_range)}0101', utc=utc)
    end_date = pd.to_datetime(f'{np.max(year_range)+1}0101', utc=utc)
    days_between_dates = (end_date - start_date).days        
    random_number_of_days = np.random.randint(0, days_between_dates)
    random_date = start_date + datetime.timedelta(days=random_number_of_days)
    return random_date

def calculate_1km_pixels_in_25km_cells():
    raw_era5_fldr = hj_base + '/data/uk/driving_data/era5/hourly_single_levels/'
    dat = xr.open_dataset(raw_era5_fldr + '/t2m/era5_20111014_t2m.nc') # any random era5 file
    chess_grid = xr.open_dataset(home_data_dir + '/chess/chess_lat_lon.nc')

    ### if using raw lat/lon ERA5 data
    ## cut down large era5 spatial range to the chess grid extent
    latmin = float(chess_grid.lat.min())
    latmax = float(chess_grid.lat.max())
    lonmin = float(chess_grid.lon.min())
    lonmax = float(chess_grid.lon.max())
    dat = dat.loc[dict(longitude = dat.longitude[(dat.longitude < (lonmax)) & (dat.longitude > (lonmin))],
                       latitude = dat.latitude[(dat.latitude < (latmax)) & (dat.latitude > (latmin))])]
    ## reproject onto BNG
    wgs84_epsg = 4326
    bng_epsg = 27700
    dat = dat.rio.write_crs(rasterio.crs.CRS.from_epsg(wgs84_epsg))
    dat = dat.rio.reproject(f"EPSG:{bng_epsg}") # too slow!
    
    # generate a chess grid at the same 25km resolution as the raw ERA5    
    ynew = np.array(range(dp.res//2, dp.res * (int(chess_grid.y.max()) // dp.res) + dp.res//2, dp.res))
    xnew = np.array(range(dp.res//2, dp.res * (int(chess_grid.x.max()) // dp.res) + dp.res//2, dp.res))
        
    # pad right/bottom to not lose non-full-tile data
    # ynew = np.array(range(dp.res//2, 
                          # (dp.res * 
                            # (int(chess_grid.y.max() +
                                # (dp.res - chess_grid.y.max() % dp.res)) // 
                            # dp.res) + 
                            # dp.res//2),
                          # dp.res))
    # xnew = np.array(range(dp.res//2,                                          
                          # (dp.res * 
                            # (int(chess_grid.x.max() + 
                                # (dp.res - chess_grid.x.max() % dp.res)) // 
                            # dp.res) + 
                            # dp.res//2),
                          # dp.res))
    
    dat = dat.interp(y=ynew, x=xnew)

    # given a 25km pixel from the era5 lat lon grid, which 1km chess pixels lie inside?
    # does this have a unique solution or is there judgement calls about edge pixels?
    # the netCDF version of ERA5 is on a regular lat/lon grid and should be thought of as 
    # point values at (lat, lon) (or as centroids of tiles, though the interpolation from GRIB to netCDF
    # does not preserve area so this is slightly inaccurate).  

    ## on re-gridded era5 we can do this via (h*(y//h) + h//2, w*(x//w) + w//2)
    # to calculate the centroid of the parent coarse pixel
    exs, eys = np.meshgrid(dat.x.values, dat.y.values)
    era_xy = pd.merge(
        pd.DataFrame(exs)
            .assign(i=range(exs.shape[0]))
            .melt(id_vars=['i'], var_name='j', value_name='x'),
        pd.DataFrame(eys)
            .assign(i=range(eys.shape[0]))
            .melt(id_vars=['i'], var_name='j', value_name='y'),
        how='left', on=['i', 'j'])
    era_xy = era_xy.assign(pixel_id = range(era_xy.shape[0]))
    pixel_ids = np.array(era_xy[['y','x','pixel_id']]
        .pivot(index='y',columns='x',values='pixel_id'), dtype=np.int32)
    dat['pixel_id'] = (['y', 'x'],  pixel_ids)

    cxs, cys = np.meshgrid(dp.res * (chess_grid.x.values // dp.res) + dp.res//2,
                           dp.res * (chess_grid.y.values // dp.res) + dp.res//2)
    chess_xy = pd.merge(
        pd.DataFrame(cxs)
            .assign(i=range(cxs.shape[0]))
            .melt(id_vars=['i'], var_name='j', value_name='x'),
        pd.DataFrame(cys)
            .assign(i=range(cys.shape[0]))
            .melt(id_vars=['i'], var_name='j', value_name='y'),
        how='left', on=['i', 'j'])
    chess_xy = chess_xy.merge(era_xy[['x','y','pixel_id']], on=['x','y'], how='left')
    nbr_array = np.array(chess_xy[['i','j','pixel_id']]
        .pivot(index='i',columns='j',values='pixel_id'))
    chess_grid['era5_nbr'] = (['y', 'x'], nbr_array)
    
    ## output dat_bng_chess['pixel_id'] with (y,x) coords
    ## and chess_grid['era5_nbr'] with (y,x) coords    
    chess_grid.to_netcdf('/home/users/doran/data_dump/chess/chess_1km_25km_parent_pixel_ids.nc')
    dat.drop(['t2m']).to_netcdf('/home/users/doran/data_dump/chess/chess_25km_pixel_ids.nc')
  
    ## then, knowing which 25km cell each 1km pixel relates to, we can enforce
    ## approximate conservation of cell averages. Also, the above only
    ## needs to be done once and then the arrays re-used.

def context_target_split(sites, context_frac=0.5, random_state=22):
    df = pd.DataFrame({'SITE_ID':sites})
    context = (df.sample(int(context_frac * df.shape[0]),
                             random_state=random_state).SITE_ID)
    targets = np.setdiff1d(sites, context)
    return list(context), list(targets)

def tensorise_station_targets(station_targets, var_list=None, max_coord=99, device=None):
    # converts list of dataframes to [xy, value, mask] tensor list
    outputs = {'coords_yx':[], 'values':[], 'var_present':[]}
    
    for b in range(len(station_targets)):
        outputs['coords_yx'].append(
            torch.from_numpy(station_targets[b][['sub_y', 'sub_x']].to_numpy()).to(torch.long) #(torch.float32)
        )
        if var_list is None:
            outputs['values'].append(
                torch.from_numpy(station_targets[b].iloc[:,2:].to_numpy()).to(torch.float32)
            )
            
            outputs['var_present'].append(
                torch.from_numpy(~station_targets[b].iloc[:,2:].isna().to_numpy()).to(torch.bool)
            )
        else:
            if type(var_list)==str: var_list = [var_list]
            outputs['values'].append(
                torch.from_numpy(station_targets[b].loc[:,var_list].to_numpy()).to(torch.float32)
            )
            
            outputs['var_present'].append(
                torch.from_numpy(~station_targets[b].loc[:,var_list].isna().to_numpy()).to(torch.bool)
            )            
    if not device is None:
        outputs = {k:[tnsr.to(device) for tnsr in outputs[k]] for k in outputs}
    return outputs

# def denormalise(output, normalisations):
    # # eg normalisations from datagen.normalisations
    # rad_norm = normalisations['rad_norm']
    # temp_mu = normalisations['temp_mu']
    # temp_sd = normalisations['temp_sd']
    # p_mu = normalisations['p_mu']
    # p_sd = normalisations['p_sd']
    # rh_norm = normalisations['rh_norm']
    # ws_mu = normalisations['ws_mu']
    # ws_sd = normalisations['ws_sd']
    
    # # denormalise model output
    # output_numpy = output.numpy()
    # output_numpy[..., 0] = output_numpy[..., 0] * temp_sd + temp_mu
    # output_numpy[..., 1] = output_numpy[..., 1] * p_sd + p_mu
    # output_numpy[..., 2] = output_numpy[..., 2] * rad_norm
    # output_numpy[..., 3] = output_numpy[..., 3] * rad_norm
    # output_numpy[..., 4] = output_numpy[..., 4] * ws_sd + ws_mu
    # output_numpy[..., 5] = output_numpy[..., 5] * rh_norm

    # # denormalise coarse and fine inputs

def efficiencies(y_pred, y_true):
    alpha = np.std(y_pred) / np.std(y_true)    
    beta_nse = (np.mean(y_pred) - np.mean(y_true)) / np.std(y_true)
    beta_kge = np.mean(y_pred) / np.mean(y_true)
    rho = np.corrcoef(y_pred, y_true)[1,0]
    NSE = -beta_nse*beta_nse - alpha*alpha + 2*alpha*rho # Nash-Sutcliffe
    KGE = 1 - np.sqrt((beta_kge-1)**2 + (alpha-1)**2 + (rho-1)**2) # Kling-Gupta
    KGE_mod = 1 - np.sqrt(beta_nse**2 + (alpha-1)**2 + (rho-1)**2)
    LME = 1 - np.sqrt((beta_kge-1)**2 + (rho*alpha - 1)**2) # Liu-Mean
    LME_mod = 1 - np.sqrt(beta_nse**2 + (rho*alpha - 1)**2)
    return {'NSE':NSE, 'KGE':KGE, 'KGE_mod':KGE_mod, 'LME':LME, 'LME_mod':LME_mod}

def calc_metrics(df, var1, var2): # ytrue, ypred
    effs = efficiencies(df[[var1, var2]].dropna()[var2],
                        df[[var1, var2]].dropna()[var1]) # ypred, ytrue
    effs['r2'] = metrics.r2_score(df[[var1, var2]].dropna()[var1],
                                  df[[var1, var2]].dropna()[var2]) # ytrue, ypred
    effs['mae'] = metrics.mean_absolute_error(df[[var1, var2]].dropna()[var1],
                                              df[[var1, var2]].dropna()[var2])
    effs['medae'] = metrics.median_absolute_error(df[[var1, var2]].dropna()[var1],
                                                  df[[var1, var2]].dropna()[var2])
    return effs
    
def pad_contexts(arr, maxsize):
    pad_size = maxsize - arr.shape[1]
    pad_with = np.zeros((arr.shape[0], pad_size))
    return np.hstack([arr, pad_with])

def define_tile_start_inds(full_length, tile_size, min_overlap):
    central_tiles = int(np.ceil((full_length - 2*(tile_size-min_overlap)) / (tile_size-min_overlap)))    
    tile_starts = [0]
    for i in range(central_tiles):
        tile_starts.append(tile_starts[i] + tile_size - min_overlap)
    if tile_starts[-1] < full_length - tile_size:
        tile_starts.append(full_length - tile_size)
    if tile_starts[-1] > full_length - tile_size:
        tile_starts[-1] = full_length - tile_size
    return tile_starts    

def create_filter(filt_size, filt_vals=None):
    gfilt = np.zeros((filt_size, filt_size))
    if filt_vals is None:
        filt_vals = np.linspace(1, 0, filt_size//2 + 2)
    gfilt[filt_size//2, filt_size//2] += filt_vals[0]
    for i in range(1, filt_size//2 + 1):        
        gfilt[filt_size//2 - i, (filt_size//2 - i):(filt_size//2 + i)]  += filt_vals[i]
        gfilt[(filt_size//2 - (i - 1)):(filt_size//2 + i + 1), filt_size//2 - i] += filt_vals[i]
        gfilt[filt_size//2 + i, (filt_size//2 - (i - 1)):(filt_size//2 + i + 1)] += filt_vals[i]
        gfilt[(filt_size//2 - i):(filt_size//2 + i), filt_size//2 + i]  += filt_vals[i]
    return gfilt

def relhum_from_dewpoint(air_temp, dewpoint_temp):
    ''' for temperatures in degrees C '''
    # RH calc from https://www.omnicalculator.com/physics/relative-humidity
    return 100 * (np.exp((17.625 * dewpoint_temp)/(243.04 + dewpoint_temp)) / 
                    np.exp((17.625 * air_temp)/(243.04 + air_temp)))

def make_mask(dim_i, dim_j):
    # this could be rewritten using the flexible-sized filter 
    # to have an argument of how big the unmasked area should be 
    # at each level.
    mask = torch.zeros((dim_i, dim_j, dim_i, dim_j), dtype=torch.bool)
    for i in range(dim_i):
        for j in range(dim_j):            
            mask[i,j,i,j] = 1.
            if i>0:
                mask[i,j,i-1,j] = 1
            if j>0:
                mask[i,j,i,j-1] = 1
            if i>0 and j>0:
                mask[i,j,i-1,j-1] = 1
            if i<(dim_i-1):
                mask[i,j,i+1,j] = 1
            if j<(dim_j-1):
                mask[i,j,i,j+1] = 1
            if i<(dim_i-1) and j<(dim_j-1):
                mask[i,j,i+1,j+1] = 1
            if i>0 and j<(dim_j-1):
                mask[i,j,i-1,j+1] = 1
            if i<(dim_i-1) and j>0:
                mask[i,j,i+1,j-1] = 1
    return mask

def make_mask_with_batch_dim(dim_i, dim_j, batch_size):
    # make mask and add batch dim
    mask = make_mask(dim_i, dim_j)
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)
    return mask

def make_mask_list(coarse_dim_i, coarse_dim_j, scale, scale_factor=3, batch_size=None):
    npools = find_num_pools(scale, factor=scale_factor) # number of Nx upsamples
    sizes = [(coarse_dim_i, coarse_dim_j)]
    for i in range(npools): sizes.append((sizes[i][0]*scale_factor, sizes[i][1]*scale_factor))
    # no mask/attention at hi res size to save mem/compute        
    if batch_size is None:
        return [make_mask(sy,sx) for (sy,sx) in sizes]
    else:
        return [make_mask_with_batch_dim(sy, sx, batch_size) for (sy,sx) in sizes]

def make_context_mask(n_gridpts, n_cntxpts, batch_mask_guide):
    mask = torch.ones((len(batch_mask_guide), n_gridpts, n_cntxpts), dtype=torch.bool)
    for b in range(len(batch_mask_guide)):
        if batch_mask_guide[b]>0:
            for i in range(1, batch_mask_guide[b]+1):
                mask[b, :, -i] = 0
    return mask

def make_context_mask2(n_gridpts, n_cntxpts):
    mask = torch.ones((1, n_gridpts, n_cntxpts), dtype=torch.bool)
    mask[0, :, -1] = 0
    return mask
    
def mask_from_filter(dim_i, dim_j, filter_size, batch_dim=0)  :  
    hfs = filter_size//2
    filt_vals = np.repeat(True, hfs + 2)
    gfilt = create_filter(filter_size, filt_vals=filt_vals)    
    mask = torch.zeros((dim_i, dim_j, dim_i, dim_j), dtype=torch.bool)
    for i in range(dim_i):
        for j in range(dim_j):
            # calculate filter edge locations and account for overhangs
            pt = i, j
            le = max(0, hfs - pt[0])
            re = max(0, hfs - (dim_i - 1 - pt[0]))
            te = max(0, hfs - pt[1])
            be = max(0, hfs - (dim_j - 1 - pt[1]))
            lb = - (hfs - le)
            rb = hfs - re + 1
            tb = - (hfs - te)
            bb = hfs - be + 1
            mask[i, j, (pt[0]+lb):(pt[0]+rb), (pt[1]+tb):(pt[1]+bb)] += \
                    gfilt[(hfs+lb):(hfs+rb), (hfs+tb):(hfs+bb)]
    if batch_dim==0:
        return mask
    else:
        return mask.unsqueeze(0).expand((batch_dim, -1, -1, -1 , -1))

def make_mask_list_from_filters(coarse_dim_i, coarse_dim_j, scale,
                                scale_factor=3, batch_size=None, filter_sizes=None):
    # number of Nx upsamples
    npools = find_num_pools(scale, factor=scale_factor)
    if filter_sizes is None:
        filter_sizes = np.repeat(3, npools)
    sizes = [(coarse_dim_i, coarse_dim_j)]
    for i in range(npools):
        sizes.append((sizes[i][0]*scale_factor, sizes[i][1]*scale_factor))

    # no mask/attention at hi res size to save mem/compute        
    return [mask_from_filter(sizes[i][0],
                             sizes[i][1],
                             filter_sizes[i],
                             batch_dim=batch_size) for i in range(len(sizes))]
