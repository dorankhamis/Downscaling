import pandas as pd
import numpy as np
import torch

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

def prepare_run(checkpoint, warmup_epochs):
    if checkpoint is None:
        curr_epoch = -warmup_epochs
        best_loss = np.inf
        losses = []
        val_losses = []
    else:
        curr_epoch = checkpoint['epoch']
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

    # if loadmodel != model_outdir + 'best_model.pth' and not checkpoint is None:
        # checkpoint['best_loss'] = np.inf
        # checkpoint['epoch'] = -warmup_epochs
    
    return model, optimizer, checkpoint
