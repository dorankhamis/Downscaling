import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

if __name__=="__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("day", help = "day to run, 0-indexed")        
        args = parser.parse_args()
        day = int(args.day)        
    except:
        print('Not running as script')
        day = 0
        
    of_n_chunks = 8000
    outdir = '/gws/nopw/j04/ceh_generic/netzero/downscaling/training_data/terrain_shading/'
    
    successfully_merged = []
    fname = 'sea_shading_mask' # 'shading_mask'
    shading_cur = np.load(outdir + f'/{fname}_day_{day}_chunk0.npy')
    successfully_merged.append(0)
    for ch in range(1, of_n_chunks): # already loaded 0
        try:
            shading_chunk = np.load(outdir + f'/{fname}_day_{day}_chunk{ch}.npy')
            shading_cur = np.hstack([shading_cur, shading_chunk])
            successfully_merged.append(ch)
        except:
            # reached gap in contiguous chunks
            break
    
    # # deal with wrong order of hours (subtracting rather than adding) FIXED IN NEW RUNS
    # shading_cur = shading_cur[1:,:][::-1,:]
    # # add fully shaded zero-hour at correct side of the day!
    # shading_cur = np.vstack([np.ones((1,shading_cur.shape[1]), dtype=np.float32), shading_cur])
    # # roll day back
    # if day==0: true_day = 365
    # else: true_day = day - 1
    true_day = day
    
    # save current merged array
    np.save(outdir+f'{fname}_day_{true_day}_merged_to_{successfully_merged[-1]}.npy', shading_cur)
    
    if False:
        # visualise the shade map!
        import xarray as xr
        grid = xr.open_dataset('/home/users/doran/data_dump/bng_grids/bng_1km_28km_parent_pixel_ids.nc')
        grid = grid.load() # force load data
        
        shademap = grid.landfrac.copy()
        shademap.values[np.where(grid.landfrac.values > 0)] = 1 - shading_cur[9,:]
        shademap.plot();
        plt.show()
          
    # remove files that have been successfully merged... but only after we have got everything! 
    if successfully_merged[-1] == of_n_chunks-1:
        for ch in successfully_merged:
            os.remove(outdir + f'/{fname}_day_{day}_chunk{ch}.npy')
