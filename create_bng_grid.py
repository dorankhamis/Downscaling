import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pathlib

hj_base = '/gws/nopw/j04/hydro_jules/data/uk'
hj_ancil_fldr = hj_base + '/ancillaries/'
hj_sm_ancil = hj_base + '/soil_moisture_map/ancillaries/'
era5_fldr = hj_base + '/driving_data/era5/bn_grid/'
nz_base = '/gws/nopw/j04/ceh_generic/netzero/'
binary_batch_path = nz_base + '/downscaling/training_data/'
precip_fldr = '/badc/ukmo-hadobs/data/insitu/MOHC/HadOBS/HadUK-Grid/v1.1.0.0/1km/rainfall/day/latest/'
home_data_dir = '/home/users/doran/data_dump/'
midas_fldr = home_data_dir + '/MetOffice/midas_data/'
chessmet_dir = hj_base + '/driving_data/chess/chess-met/daily/'
res = 28 # km

# load the raw grids
hadukgrid = xr.load_dataset(precip_fldr + '/rainfall_hadukgrid_uk_1km_day_19550801-19550831.nc')
chessgrid = xr.open_dataset(hj_ancil_fldr + '/chess_lat_lon.nc')

# trim unwanted variables
newgrid = hadukgrid.isel(time=0).drop(["time_bnds", "projection_y_coordinate_bnds", "projection_x_coordinate_bnds", "time"])
newgrid = newgrid.rename({'projection_y_coordinate':'y', 'projection_x_coordinate':'x'})
newgrid.rainfall.values[~np.isnan(newgrid.rainfall.values)] = 1
newgrid.rainfall.values[np.isnan(newgrid.rainfall.values)] = 0
newgrid = newgrid.rename({'rainfall':'landfrac'})
newgrid = newgrid.assign(lat=lambda x: x.latitude, lon=lambda x: x.longitude)
newgrid = newgrid.drop(['longitude', 'latitude'])

# align bottom-left corner with chess grid
y_start = np.where(newgrid.y.values == chessgrid.y[0].values)[0][0]
x_start = np.where(newgrid.x.values == chessgrid.x[0].values)[0][0]
newgrid.landfrac[y_start:, x_start:].plot()

# trim y,x
ybtrim = np.where(newgrid.y.values >= newgrid.y.values[y_start])[0]
xbtrim = np.where(newgrid.x.values >= newgrid.x.values[x_start])[0]
newgrid = newgrid.isel(y=ybtrim, x=xbtrim)

# define new grid
ny_res = int(np.floor(len(ybtrim) / res))
nx_res = int(np.floor(len(xbtrim) / res))
new_n_y = ny_res * 28
new_n_x = nx_res * 28
ynew = np.arange(res/2, res * ny_res + res/2, res) * 1000
xnew = np.arange(res/2, res * nx_res + res/2, res) * 1000

coarsegrid = newgrid.interp(y=ynew, x=xnew[:-1]) # trim a pixel from the right 
newgrid = newgrid.isel(y=np.arange(len(newgrid.y))[:new_n_y], x=np.arange(len(newgrid.x))[:(new_n_x-28)])

newgrid.attrs['title'] = 'A slightly larger version of the CHESS grid based on the MO HadUK grid'
coarsegrid.attrs['title'] = 'A slightly larger version of the CHESS grid coarsened to 28km pixels'
newgrid.attrs['comment'] = ''
coarsegrid.attrs['comment'] = ''

## save
newgrid.to_netcdf(hj_sm_ancil + '/bng_grid_1km.nc')
coarsegrid.to_netcdf(hj_sm_ancil + '/bng_grid_28km.nc')

outdir = '/home/users/doran/data_dump/bng_grids/'
pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)

newgrid.to_netcdf(outdir + '/bng_grid_1km.nc')
coarsegrid.to_netcdf(outdir + '/bng_grid_28km.nc')

#####################################################
import rasterio
## now do 28km elevation / regrid other chess grid ancils
## and parent pixel ids?
# load new grids
finegrid = xr.load_dataset(hj_sm_ancil + '/bng_grid_1km.nc')
coarsegrid = xr.load_dataset(hj_sm_ancil + '/bng_grid_28km.nc')

chess_grid = xr.open_dataset(hj_ancil_fldr + '/chess_lat_lon.nc')
height_grid = xr.open_dataset(hj_ancil_fldr+'/uk_ihdtm_topography+topoindex_1km.nc')
# easting/northing of SW corner of grid box, so redefine x,y
height_grid = height_grid.assign_coords({'x':chess_grid.x.values,
                                         'y':chess_grid.y.values})
height_grid.eastings.values = height_grid.eastings.values + 500
height_grid.northings.values = height_grid.northings.values + 500

# open a test raw era5 dataset
raw_era5_fldr = hj_base + '/driving_data/era5/hourly_single_levels/'
dat = xr.open_dataset(raw_era5_fldr + '/t2m/era5_20150101_t2m.nc') # any random era5 file
dat = dat.isel(time=0).drop('time')

# reproject onto BNG
wgs84_epsg = 4326
bng_epsg = 27700
dat = dat.rio.write_crs(rasterio.crs.CRS.from_epsg(wgs84_epsg))
dat_rp = dat.rio.reproject(f"EPSG:{bng_epsg}") # too slow!

# regrid       
#dat = dat.interp(y=coarsegrid.y.values.astype(np.int32), x=coarsegrid.x.values.astype(np.int32))
dat_rg = dat_rp.interp_like(coarsegrid)

fig, ax = plt.subplots(1,3)
dat.t2m.plot(ax=ax[0])
dat_rp.t2m.plot(ax=ax[1])
dat_rg.t2m.plot(ax=ax[2])
plt.show()

# load amulya test
dat_test = xr.load_dataset('/gws/nopw/j04/hydro_jules/data/uk/driving_data/era5/28km_grid/t2m/era5_20150101_t2m.nc')
dat_test = dat_test.isel(time=0).drop('time')
fig, ax = plt.subplots(1,4)
dat.t2m.plot(ax=ax[0])
dat_rp.t2m.plot(ax=ax[1])
dat_rg.t2m.plot(ax=ax[2])
dat_test.t2m.plot(ax=ax[3])
plt.show()

plt.plot(dat_test.t2m.values.flatten(), dat_rg.t2m.values.flatten())

## on re-gridded era5 we can do this via (h*(y//h) + h//2, w*(x//w) + w//2)
# to calculate the centroid of the parent coarse pixel
finegrid.landfrac.plot()

exs, eys = np.meshgrid(dat_test.x.values, dat_test.y.values)
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
dat_test['pixel_id'] = (['y', 'x'],  pixel_ids)

res = 28000
cxs, cys = np.meshgrid(res * (finegrid.x.values // res) + res//2,
                       res * (finegrid.y.values // res) + res//2)
fine_xy = pd.merge(
    pd.DataFrame(cxs)
        .assign(i=range(cxs.shape[0]))
        .melt(id_vars=['i'], var_name='j', value_name='x'),
    pd.DataFrame(cys)
        .assign(i=range(cys.shape[0]))
        .melt(id_vars=['i'], var_name='j', value_name='y'),
    how='left', on=['i', 'j'])
fine_xy = fine_xy.merge(era_xy[['x','y','pixel_id']], on=['x','y'], how='left')
nbr_array = np.array(fine_xy[['i','j','pixel_id']]
    .pivot(index='i',columns='j',values='pixel_id'))
finegrid['era5_nbr'] = (['y', 'x'], nbr_array)

## interp elevation map to coarse grid
height_grid_f = height_grid.interp_like(finegrid)
height_grid_c = height_grid_f.interp_like(coarsegrid)
dat_test['elev'] = height_grid_c.elev

# and save fine scale version
height_grid_f.drop(['eastings', 'northings']).to_netcdf('/home/users/doran/data_dump/height_map/topography_bng_1km.nc')

## interp landfrac to coarse grid
lf_c = finegrid.interp_like(coarsegrid)
dat_test['landfrac'] = lf_c.landfrac

## output dat_bng_chess['pixel_id'] with (y,x) coords
## and chess_grid['era5_nbr'] with (y,x) coords    
finegrid.to_netcdf('/home/users/doran/data_dump/bng_grids/bng_1km_28km_parent_pixel_ids.nc')
dat_test.drop('t2m').to_netcdf('/home/users/doran/data_dump/bng_grids/bng_28km_pixel_ids.nc')


