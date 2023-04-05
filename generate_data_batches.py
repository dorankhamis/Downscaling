import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import glob
#import tensorflow as tf
import argparse

import setupdata
from soil_moisture.utils import zeropad_strint


def main(run_n, batch_type,
         train_years, val_years, heldout_years,
         dim_l, res, scale):
    try:
        yr_ind = run_n // 12
        mn_ind = run_n % 12
        if batch_type=='train':
            year = train_years[yr_ind]
        elif batch_type=='val':
            year = val_years[yr_ind]
        else:
            year = heldout_years[yr_ind]
        month = zeropad_strint(mn_ind+1)
        possible_files = glob.glob(f'{setupdata.era5_fldr}/t2m/era5_{year}{month}*_t2m.nc')
        datestrings = [f.split('era5_')[-1].split('_')[0] for f in possible_files]
    except:
        return 1

    datagen = setupdata.data_generator(        
        train_years, val_years, heldout_years,
        dim_l=dim_l, res=res, scale=scale
    )

    batch_outpath = setupdata.binary_batch_path + f'/{batch_type}_batches/'
    Path(batch_outpath).mkdir(parents=True, exist_ok=True)

    for ds in datestrings:
        if Path(f'{batch_outpath}/era5_bng_25km_pixels_{ds}.pkl').exists():
            continue
        parpix = datagen.read_parent_pixel_day(batch_type=batch_type, date_string=ds)
        pd.to_pickle(parpix, f'{batch_outpath}/era5_bng_25km_pixels_{ds}.pkl')
        

if __name__ == "__main__":      
    parser = argparse.ArgumentParser()
    parser.add_argument("run_n", help = "run number")
    parser.add_argument("batch_type", help = "train, val or test")
    args = parser.parse_args()
    run_n = int(args.run_n)
    batch_type = str(args.batch_type)
    
    dim_l = 4 # number of raw ERA5 pixels in each dim to take
    res = 25000 # or raw ERA5 in metres
    scale = 25 # how much we are downscaling
    train_years = [2015, 2016, 2017, 2018]
    val_years = [2019, 2020]
    heldout_years = [2021]
    
    main(run_n, batch_type,
         train_years, val_years, heldout_years,
         dim_l, res, scale)
        
