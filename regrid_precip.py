import numpy as np
import pandas as pd
import cf
import sys
from pathlib import Path

path = '/gws/nopw/j04/hydro_jules/data/uk/driving_data/era5/'
output_original = path+'/hourly_single_levels/'
output_regrid = path+'/28km_grid/'
name = 'mtpr'
output_regridded = output_regrid+name+'/'
Path(output_regridded).mkdir(parents=True, exist_ok=True)
        
#dt = int(sys.argv[1])
for dt in np.arange(0,366):
    for year in np.arange(2000, 2022):
        dates = [str(s).replace('-', '').replace(' 00:00:00', '') for s in pd.date_range(f'{year}0101', f'{year+1}0101')[:-1]]
        try:
            date = dates[dt]
            print(date)
            yr = date[0:4]
            mon = date[4:6]
            day = date[6:8]
            
            original = cf.read(output_original+name+'/era5_'+date+'_'+name+'.nc')[0]
            print(original)
            base = cf.read('/gws/nopw/j04/hydro_jules/data/uk/soil_moisture_map/ancillaries/bng_grid_28km.nc')
            lon = cf.AuxiliaryCoordinate(source=base.select('longitude')[0])
            lat = cf.AuxiliaryCoordinate(source=base.select('latitude')[0])
            base.select('lwe_thickness_of_precipitation_amount')[0].set_construct(lon, axes=['Y', 'X'])
            base.select('lwe_thickness_of_precipitation_amount')[0].set_construct(lat, axes=['Y', 'X'])
            print("Doing regridding...")
            regridded = original.regrids(base.select('lwe_thickness_of_precipitation_amount')[0], method='linear')
            regridded.set_property('File_conversion', 'Converted from ERA5 lat/lon to 28km eqBNG')
            print("Saving...")
            cf.write(regridded, output_regridded+'/era5_'+date+'_'+name+'.nc', global_attributes='File_conversion')
        except:
            continue # for leap year days, i.e. dt=365
