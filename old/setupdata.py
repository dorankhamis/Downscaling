import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
import pickle
import glob
import datetime
import torch
import skimage.measure
import warnings
import math

from pathlib import Path
from sklearn.neighbors import NearestNeighbors

from data_classes.met_data import ERA5Data
from data_classes.cosmos_data import CosmosMetaData, CosmosData
from utils import *
from params import normalisation as nm
from params import data_pars as dp
from solar_position import SolarPosition

EPS = 1e-10

#era5_fldr = '/gws/nopw/j04/hydro_jules/data/uk/driving_data/era5/hourly_single_levels/'
# ERA5 data on Jasmin badc:
# /badc/ecmwf-era5/data/oper
# /badc/ecmwf-era5t/data/oper
# /badc/ecmwf-era5/data/invariants

hj_base = '/gws/nopw/j04/hydro_jules/'
hj_ancil_fldr = hj_base + '/data/uk/ancillaries/'
era5_fldr = hj_base + '/data/uk/driving_data/era5/bn_grid/'
nz_base = '/gws/nopw/j04/ceh_generic/netzero/'
binary_batch_path = nz_base + '/downscaling/training_data/'
precip_fldr = '/badc/ukmo-hadobs/data/insitu/MOHC/HadOBS/HadUK-Grid/v1.1.0.0/1km/rainfall/day/latest/'
home_data_dir = '/home/users/doran/data_dump/'
midas_fldr = home_data_dir + '/MetOffice/midas_data/'

def read_one_cosmos_site_met(SID, missing_val=-9999.0):
    data = CosmosData(SID)    
    data.read_subhourly()
    data.preprocess_all(missing_val, 'DATE_TIME')
    return data

def provide_cosmos_met_data(metadata, met_vars, sites=None, missing_val=-9999.0, forcenew=False):
    Path(home_data_dir+'/met_pickles/').mkdir(exist_ok=True, parents=True)
    fname = home_data_dir+'/met_pickles/cosmos_site_met.pkl'
    if forcenew is False and Path(fname).is_file():
        with open(fname, 'rb') as fo:
            metdat = pickle.load(fo)
        met_data = metdat.pop('data')        
    else:        
        met_data = {}
        if sites is None: sites = metadata.site['SITE_ID']
        for SID in sites:            
            data = read_one_cosmos_site_met(SID, missing_val=-9999.0)    
            data = data.subhourly[met_vars]
            # check that this aggregation labelling matches ERA5 hourly meaning!
            met_data[SID] = data.resample('1H', origin='end_day').mean()
        metdat = dict(data=met_data)
        with open(fname, 'wb') as fs:
            pickle.dump(metdat, fs)
    return met_data
    
def provide_midas_met_data(metadata, met_vars, sites=None, forcenew=False):
    Path(home_data_dir+'/met_pickles/').mkdir(exist_ok=True, parents=True)
    fname = home_data_dir+'/met_pickles/midas_site_met.pkl'
    if forcenew is False and Path(fname).is_file():
        with open(fname, 'rb') as fo:
            metdat = pickle.load(fo)
        met_data = metdat.pop('data')
    else:        
        met_data = {}
        if sites is None: sites = metadata['SITE_ID']
        for SID in sites:
            if Path(midas_fldr+SID+'.pkl').exists():
                with open(midas_fldr+SID+'.pkl', 'rb') as fo:
                    data = pickle.load(fo)
                try:
                    met_data[SID] = data[met_vars]
                except:
                    print(f'Met vars missing from {SID}')
        metdat = dict(data=met_data)
        with open(fname, 'wb') as fs:
            pickle.dump(metdat, fs)
    return met_data

def site_splits(use_sites=None, holdoutfrac=0.1, random_state=22):
    metadata = CosmosMetaData()    
    midasfiles = glob.glob(midas_fldr+'/*.pkl')
    midasfiles = [s.split('/')[-1].split('.pkl')[0] for s in midasfiles]
    
    if use_sites is None: 
        use_sites = list(metadata.site.SITE_ID) + midasfiles
    
    # drop sites with missing static data (Northern Ireland, mainly)
    missing_sites = ['FIVET', 'GLENW', 'HILLB']
    use_sites = list(np.setdiff1d(use_sites, missing_sites))

    # create splits
    df = pd.DataFrame({'SITE_ID':use_sites})
    train_sites = (df.sample(int((1-holdoutfrac) * df.shape[0]),
                             random_state=random_state).SITE_ID)
    heldout_sites = np.setdiff1d(use_sites, train_sites)
    return list(train_sites), list(heldout_sites)

def partition_interp_solar_radiation(swin_coarse, fine_grid, sp,
                                     press_coarse, press_fine,
                                     height_grid, shading_array):
    if np.all(swin_coarse.values==0):
        # short cut for night time
        return fine_grid.landfrac * 0
    
    # from Proposal of a regressive model for the hourly diffuse
    # solar radiation under all sky conditions, J.A. Ruiz-Arias, 2010
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ## calculate diffuse fraction
        I0 = 1361.5 # solar constant, W m-2    
        sw_in = swin_coarse.interp_like(fine_grid, method='linear')
        
        mask = sp.solar_zenith_angle <= 90
                
        # the clearness index, ratio of cloud top solar to surface incident solar rad
        kt = fine_grid.landfrac.copy()
        kt.values[:,:] = 0        
        kt.values[mask] = sw_in.values[mask] / \
            (I0 * np.cos(np.deg2rad(sp.solar_zenith_angle.values[mask])))
        
        # diffuse fraction
        k = 0.952 - 1.041 * np.exp(-np.exp(2.300 - 4.702*kt))
        
        ## partitioning into direct and diffuse:
        S_dir = sw_in * (1 - k)
        S_dif = sw_in * k
        
        ## adjustments to SW components
        # cloud cover: already accounted for by ERA5?
        #visible_sky_portion = 1
        #S_dif *= visible_sky_portion
        
        # terrain illumination/shading
        cos_solar_illum = fine_grid.landfrac.copy()
        cos_solar_illum.values[:,:] = 0        
        cos_solar_illum.values[mask] = (
            np.cos(np.deg2rad(sp.solar_zenith_angle.values[mask])) * 
            np.cos(np.deg2rad(height_grid.slope.values[mask])) + 
            np.sin(np.deg2rad(sp.solar_zenith_angle.values[mask])) * 
            np.sin(np.deg2rad(height_grid.slope.values[mask]))# *
            #np.cos(np.deg2rad(sp.solar_azimuth_angle.values[mask] - height_grid.aspect.values[mask]))
        )
        # we replace the azimuthal/aspect contribution with our shade mask:
        shade_map = (fine_grid.landfrac.copy()).astype(np.uint8)
        shade_map.values[:,:] = 1
        shade_map.values[fine_grid.landfrac.values==1] = -(shading_array-1) # invert shade to zero
        
        # broadband attenuation    
        p_interp = press_coarse.interp_like(fine_grid, method='linear')
        broadband_attenuation = fine_grid.landfrac.copy()
        broadband_attenuation.values[:,:] = 1        
        broadband_attenuation.values[mask] = - (
            (np.log(I0 * np.cos(np.deg2rad(sp.solar_zenith_angle.values[mask])) + EPS) - 
                np.log(sw_in.values[mask] + EPS)) / p_interp.values[mask]
        )
        S_dir *= shade_map * cos_solar_illum * np.exp(broadband_attenuation * (press_fine - p_interp))  
        
        # final SW 1km                
        Sw_1km = S_dir + S_dif
        
        return Sw_1km

class Batch:
    def __init__(self, batch, masks, grid_locs, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.coarse_inputs = batch[0].to(device)
            self.fine_inputs = batch[1].to(device)
            #self.station_dict = tensorise_station_targets(batch[2], device=device)
            self.context_station_dict = tensorise_station_targets([s['context'] for s in batch[2]], device=device)
            self.target_station_dict = tensorise_station_targets([s['target'] for s in batch[2]], device=device)
            self.constraint_targets = batch[3].to(device)            
            self.context_data = batch[4].to(device)
            self.context_locs = batch[5].to(device)
            
            n_gridpts = batch[1].shape[-1] * batch[1].shape[-2]
            n_cntxpts = batch[4].shape[-1]
            n_batch = batch[0].shape[0]
            
            self.grid_locs = torch.from_numpy(grid_locs.T).type(torch.float32).to(device)
            self.grid_locs = self.grid_locs.unsqueeze(0).expand(n_batch, -1, -1)
            
            self.context_mask = make_context_mask(n_gridpts, n_cntxpts, batch[6]).to(device)            
            self.masks = [
                (m.unsqueeze(0)
                  .expand(n_batch, -1, -1, -1, -1)
                  .type(torch.bool)
                  .to(device)) for m in masks
            ]
        except:
            self.coarse_inputs = None
            self.fine_inputs = None
            self.context_station_dict = None
            self.target_station_dict = None
            self.constraint_targets = None
            self.masks = None
            self.context_data = None
            self.context_locs = None
            self.context_mask = None
            self.grid_locs = None
    
    def copy(self, other_batch):
        self.coarse_inputs = other_batch.coarse_inputs.clone()
        self.fine_inputs = other_batch.fine_inputs.clone()
        self.context_station_dict = other_batch.context_station_dict.copy()
        self.target_station_dict = other_batch.target_station_dict.copy()
        self.constraint_targets = other_batch.constraint_targets.clone()
        self.masks = [m.clone() for m in other_batch.masks]
        self.context_data = other_batch.context_data
        self.context_locs = other_batch.context_locs
        self.context_mask = other_batch.context_mask
        self.grid_locs = other_batch.grid_locs
        

class data_generator():
    def __init__(self, train_sites=None, heldout_sites=None):
        
        # load 1km chess grid        
        self.chess_grid = xr.open_dataset(hj_ancil_fldr+'/chess_lat_lon.nc')
        self.chess_grid = self.chess_grid.load() # force load data
        self.latmin = float(self.chess_grid.lat.min())
        self.latmax = float(self.chess_grid.lat.max())
        self.lonmin = float(self.chess_grid.lon.min())
        self.lonmax = float(self.chess_grid.lon.max())
        
        self.met_cls = ERA5Data()
        
        self.wgs84_epsg = 4326
        self.bng_epsg = 27700
        self.res = dp.res # resolution of lo-res image in m
        self.scale = dp.scale # downscaling factor
        self.dim_l = dp.dim_l # size of lo-res image in 25km pixels (size of subest of UK to train on each sample)
        self.dim_h = self.dim_l*dp.scale # size of hi-res image in 1km pixels
        
        # create a coarse res chess grid
        ''' should change this to pad out the right/bottom to not shrink resulting hi res grid? 
        (int(dg.chess_grid.y.max() + (dp.res - dg.chess_grid.y.max() % dp.res)) // dp.res)
        (int(dg.chess_grid.x.max() + (dp.res - dg.chess_grid.x.max() % dp.res)) // dp.res)
        '''
        self.y_chess_25k = np.array(range(dp.res//2, 
                                          #dp.res * (int(self.chess_grid.y.max()) // dp.res) + dp.res//2,
                                          (dp.res * 
                                            (int(self.chess_grid.y.max() +
                                                (dp.res - self.chess_grid.y.max() % dp.res)) // 
                                            dp.res) + 
                                            dp.res//2),
                                          dp.res))
        self.x_chess_25k = np.array(range(dp.res//2,
                                          #dp.res * (int(self.chess_grid.x.max()) // dp.res) + dp.res//2,
                                          (dp.res * 
                                            (int(self.chess_grid.x.max() + 
                                                (dp.res - self.chess_grid.x.max() % dp.res)) // 
                                            dp.res) + 
                                            dp.res//2),
                                          dp.res))
        
        # load parent pixel ids on reprojected/regridded ERA5 25km cells
        ''' should be recalculated if extending the coarse chess grid from above!'''
        self.parent_pixels = xr.open_dataset(home_data_dir + '/chess/chess_25km_pixel_ids.nc')
        self.parent_pixels = self.parent_pixels.load()
        
        # load child pixels on 1km chess grid labelled by parent pixel IDs
        ''' likely should be recalculated if extending the coarse chess grid from above!'''
        self.child_parent_map = xr.open_dataset(home_data_dir + '/chess/chess_1km_25km_parent_pixel_ids.nc')
        self.child_parent_map = self.child_parent_map.load()
    
        # define ERA5 (coarse) and COSMOS (fine) variables to load
        var_name_map = dict(
            name   = ['Air_Temp', 'Pressure', 'Short_Wave_Rad_In', 'Long_Wave_Rad_In', 'Wind_Speed', 'Relative_Humidity'],
            coarse = ['t2m'     , 'sp'      , 'msdwswrf'         , 'msdwlwrf'        , 'ws'        , 'rh'],
            fine   = ['TA'      , 'PA'      , 'SWIN'             , 'LWIN'            , 'WS'        , 'RH']
        )        
        self.var_name_map = pd.DataFrame(var_name_map)
        self.var_name_map = self.var_name_map.assign(
            chess = np.array(['tas', 'psurf', 'rsds', 
                              'rlds', 'sfcWind', 'huss'])
            # huss is originally specific humidity but we transform it to RH
        )
        self.var_name_map.index = self.var_name_map.fine

        self.targ_var_depends = {}
        self.targ_var_depends['TA'] = ['landfrac', 'elev', 'stdev']
        self.targ_var_depends['PA'] = ['landfrac', 'elev', 'stdev']
        self.targ_var_depends['SWIN'] = ['landfrac', 'elev', 'stdev', 'slope',
                                         'aspect',  'solar_azimuth', 'solar_altitude']
        self.targ_var_depends['LWIN'] = ['landfrac', 'elev', 'stdev', 'slope',
                                         'aspect',  'solar_azimuth', 'solar_altitude']
        self.targ_var_depends['WS'] = ['landfrac', 'elev', 'stdev', 'slope']
        self.targ_var_depends['RH'] = ['landfrac']        
        
        # load site metadata/locations
        self.site_metadata = CosmosMetaData()
        midas_metadata = pd.read_csv(home_data_dir + '/MetOffice/midas_site_locations.csv')

        # load site data
        ''' we need to remove the Ireland sites from here as we 
        don't currently have the ancillary data for them! '''
        self.site_data = provide_cosmos_met_data(self.site_metadata,
                                                 self.var_name_map['fine'],
                                                 forcenew=False)
        self.site_metadata = self.site_metadata.site

        #midas_vars = ['air_temperature', 'stn_pres', 'wind_speed', 'rltv_hum']
        midas_data = provide_midas_met_data(midas_metadata,
                                            self.var_name_map['fine'],
                                            forcenew=True)        

        # rescale midas air temperature to Kelvin to match COSMOS and add to self.site_dat
        for sid in midas_data.keys():
            midas_data[sid]['TA'] = midas_data[sid]['TA'] + 273.15
            self.site_data[sid] = midas_data[sid]
            self.site_metadata = pd.concat([
                self.site_metadata, 
                midas_metadata[midas_metadata['SITE_ID']==sid]], axis=0)
        self.site_metadata = self.site_metadata.reset_index().drop('index', axis=1)

        # for each site find the 1km chess tile it sits within
        cosmos_chess_y = []
        cosmos_chess_x = []
        coarse_parent_pixel_id = []
        for i in self.site_metadata.index:
            this_dat = self.site_metadata.loc[i]
            ccyx = find_chess_tile(this_dat['LATITUDE'], this_dat['LONGITUDE'],
                                   self.chess_grid)
            cosmos_chess_y.append(ccyx[0][0])
            cosmos_chess_x.append(ccyx[1][0])
            try:
                coarse_parent_pixel_id.append(
                    int(self.child_parent_map.era5_nbr[ccyx[0][0], ccyx[1][0]].values)
                )
            except:
                coarse_parent_pixel_id.append(np.nan)

        # add to site metadata but filter out missing vals    
        self.site_metadata = self.site_metadata.assign(chess_y = cosmos_chess_y,
                                                       chess_x = cosmos_chess_x,
                                                       parent_pixel_id = coarse_parent_pixel_id)
        missing_sites = self.site_metadata[self.site_metadata['parent_pixel_id'].isna()]['SITE_ID']
        self.site_metadata = self.site_metadata[~self.site_metadata['parent_pixel_id'].isna()]
        for sid in missing_sites: 
            self.site_data.pop(sid)
            midas_data[sid] = None
        del(midas_data)

        self.train_years = dp.train_years
        self.val_years = dp.val_years
        self.heldout_years = dp.heldout_years
        if train_sites is None:
            train_sites, heldout_sites = site_splits(use_sites=list(self.site_metadata.SITE_ID),
                                                     holdoutfrac=0.1, random_state=22)        
        self.train_sites = train_sites
        self.heldout_sites = heldout_sites
        
        # load hi res static data        
        self.height_grid = xr.open_dataset(hj_ancil_fldr+'/uk_ihdtm_topography+topoindex_1km.nc')
        self.height_grid = self.height_grid.drop(['topi','stdtopi', 'fdepth','area']).load()        
        
        self.elev_vars = ['elev', 'stdev', 'slope', 'aspect']
        # easting/northing of SW corner of grid box, so redefine x,y
        self.height_grid = self.height_grid.assign_coords({'x':self.chess_grid.x.values,
                                                           'y':self.chess_grid.y.values})
        self.height_grid.eastings.values = self.height_grid.eastings.values + 500
        self.height_grid.northings.values = self.height_grid.northings.values + 500
        # now we are labelling the tile centroid
        
        # treat NaNs in height grid        
        # 1: elev, sea NaNs should be elev==0        
        self.height_grid.elev.values[np.isnan(self.height_grid.elev.values)] = 0
        # 2: stdev, sea NaNs should be stdev==0        
        self.height_grid.stdev.values[np.isnan(self.height_grid.stdev.values)] = 0
        # 3: slope, sea NaNs should be slope==0
        self.height_grid.slope.values[np.isnan(self.height_grid.slope.values)] = 0
        # 4: aspect, sea NaNs are tricky, aspect is "straight up", stick with uniform noise
        asp_mask = np.isnan(self.height_grid.aspect.values)
        self.height_grid.aspect.values[asp_mask] =  np.random.uniform(
            low=0, high=360, size=len(np.where(asp_mask)[0])
        )        
        
        # find 25km elevation data by averaging 1km data
        ys = self.parent_pixels.y.shape[0]
        xs = self.parent_pixels.x.shape[0]        
        self.parent_pixels['elev'] = (['y', 'x'], np.ones((ys, xs), dtype=np.float32)*np.nan)
        source = self.height_grid['elev'].values
        self.parent_pixels['elev'] = (('y', 'x'), 
            skimage.measure.block_reduce(
                source, (self.scale, self.scale), np.nanmean
            )[:ys,:xs] # trim right/bottom edges
        )
        del(source)
               
        # normalise site data
        for SID in list(self.site_data.keys()):
            # incoming radiation                        
            self.site_data[SID].loc[:, 'LWIN'] = (self.site_data[SID].loc[:, 'LWIN'] - nm.lwin_mu) / nm.lwin_sd
            # first log transform SWIN
            self.site_data[SID].loc[:, 'SWIN'] = np.log(1. + self.site_data[SID].loc[:, 'SWIN'])
            # then standardise
            self.site_data[SID].loc[:, 'SWIN'] = (self.site_data[SID].loc[:, 'SWIN'] - nm.logswin_mu) / nm.logswin_sd
            
            # air pressure
            self.site_data[SID].loc[:, ['PA']] = (self.site_data[SID].loc[:, ['PA']] - nm.p_mu) / nm.p_sd
            
            # relative humidity            
            self.site_data[SID].loc[:, ['RH']] = (self.site_data[SID].loc[:, ['RH']] - nm.rh_mu) / nm.rh_sd
            
            # temperature
            #print(self.site_data[SID].loc[:, ['TA']].mean()) check T is in Kelvin?
            self.site_data[SID].loc[:, ['TA']] = (self.site_data[SID].loc[:, ['TA']] - 273.15 - nm.temp_mu) / nm.temp_sd
            
            # wind speed            
            self.site_data[SID].loc[:, ['WS']] = (self.site_data[SID].loc[:, ['WS']] - nm.ws_mu) / nm.ws_sd

        # load list of binary batches
        self.bin_batches = {}
        self.bin_batches['train'] = glob.glob(f'{binary_batch_path}/train_batches/*.pkl')
        self.bin_batches['val'] = glob.glob(f'{binary_batch_path}/val_batches/*.pkl')
        self.bin_batches['test'] = glob.glob(f'{binary_batch_path}/test_batches/*.pkl')
        
        # location array for hi-res grid (for cross-attention with context points)
        X1 = np.where(np.ones((self.dim_h, self.dim_h)))
        self.X1 = np.hstack([X1[0][...,np.newaxis], X1[1][...,np.newaxis]]) / (self.dim_h-1)

    def find_1km_pixel(self, lat, lon):        
        dist_diff = np.sqrt(np.square(self.chess_grid.lat.values - lat) +
                            np.square(self.chess_grid.lon.values - lon))
        chesstile_yx = np.where(dist_diff == np.min(dist_diff))
        return chesstile_yx        

    def read_parent_pixel_day(self, batch_type='train', date_string=None):
        if date_string is None:
            if batch_type=='train':
                self.td = generate_random_date(self.train_years)
            elif batch_type=='val':
                self.td = generate_random_date(self.val_years)
            elif batch_type=='test':
                self.td = generate_random_date(self.heldout_years)
            date_string = f'{self.td.year}{zeropad_strint(self.td.month)}{zeropad_strint(self.td.day)}'
        
        # load ERA5 data for that date and trim lat/lon
        era5_vars = self.var_name_map[(self.var_name_map['coarse']!='ws') &
                                      (self.var_name_map['coarse']!='rh')]['coarse']
        era5_vars = list(era5_vars) + ['u10', 'v10', 'd2m']
                
        era5_filelist = [f'{era5_fldr}/{var}/era5_{date_string}_{var}.nc' for var in era5_vars]
        self.era5_dat = xr.open_mfdataset(era5_filelist)
        
        # because these are already on the BNG at 1km, find are averages of 25x25 squares        
        self.parent_pixels = self.parent_pixels.assign_coords(
            {'time':self.era5_dat.time.values}
        )
        ys = self.parent_pixels.y.shape[0] - 1
        xs = self.parent_pixels.x.shape[0] - 1
        ts = self.parent_pixels.time.shape[0]
        for var in era5_vars:
            self.parent_pixels[var] = (['time', 'y', 'x'],
                                      np.ones((ts, ys, xs), dtype=np.float32)*np.nan)
            source = self.era5_dat[var].values
            self.parent_pixels[var] = (('time', 'y', 'x'), 
                skimage.measure.block_reduce(
                    source, (1, self.scale, self.scale), np.mean
                )[:ts,:ys,:xs] # trim right/bottom edges
            )
        del(source)
        del(self.era5_dat)
        
        ## if loading raw lat/lon projection
        # self.era5_dat = self.era5_dat.loc[dict(
            # longitude = self.era5_dat.longitude[
                # (self.era5_dat.longitude < self.lonmax) & (self.era5_dat.longitude > self.lonmin)],
            # latitude = self.era5_dat.latitude[
                # (self.era5_dat.latitude < self.latmax) & (self.era5_dat.latitude > self.latmin)]
        # )]

        # reproject and regrid onto 25km BNG
        # self.era5_dat = self.era5_dat.rio.write_crs(rasterio.crs.CRS.from_epsg(self.wgs84_epsg))
        # self.era5_dat = self.era5_dat.rio.reproject(f"EPSG:{self.bng_epsg}")
        # self.era5_dat = self.era5_dat.interp(y=self.y_chess_25k, x=self.x_chess_25k)
        
        return self.parent_pixels

    def prepare_constraints(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # ignore warnings thrown by interp_like() which seem to be needless
            
            ### constraints for training help 
            ## Air Temp
            # reduce to sea level with lapse rate of -0.0065 K/m
            lapse_val = -0.0065
            self.parent_pixels['t_sealevel'] = (['time', 'y', 'x'],
                                               self.parent_pixels['t2m'].values)
            self.parent_pixels['t_sealevel'] = self.parent_pixels['t_sealevel'] - self.parent_pixels['elev'] * lapse_val

            # bicubic spline to interpolate from 25km to 1km... cubic doesn't work here!
            t_sealevel_interp = self.parent_pixels['t_sealevel'].interp_like(self.chess_grid, method='linear')
            t_sealevel_interp = t_sealevel_interp.transpose('time', 'y', 'x')

            # adjust to the 1km elevation using same lapse rate
            self.t_1km_elev = t_sealevel_interp + self.height_grid['elev'] * lapse_val
            self.t_1km_elev = self.t_1km_elev.transpose('time', 'y', 'x')
            
            ## Air Pressure:
            # integral of hypsometric equation using the 1km Air Temp?
            T_av = 0.5*(t_sealevel_interp + self.t_1km_elev) # K
            p_1 = 1013 # hPa, standard sea level pressure value
            R = 287 # J/kg·K = kg.m2/s2/kg.K = m2/s2.K  # specific gas constant of dry air
            g = 9.81 # m/s2 
            self.p_1km_elev = p_1 * np.exp(-g * self.height_grid['elev'] / (R * T_av))
            self.p_1km_elev = self.p_1km_elev.transpose('time', 'y', 'x')
            del(T_av)
            del(t_sealevel_interp)
            
            ## Relative Humidity
            # Assumed constant with respect to elevation, so can be simply 
            # interpolated from 25km to 1km using a bicubic spline.
            t2m_C = self.parent_pixels['t2m'] - 273.15 # K -> Celsius 
            d2m_C = self.parent_pixels['d2m'] - 273.15 # K -> Celsius
            rh_25km = relhum_from_dewpoint(t2m_C, d2m_C)            
            self.rh_1km_interp = rh_25km.interp_like(self.chess_grid, method='linear')
            self.rh_1km_interp = self.rh_1km_interp.transpose('time', 'y', 'x')
            
            # ## Dew point temperature
            # # reduce to sea level with lapse rate of -0.002 K/m
            # # from https://www.atmos.illinois.edu/~snodgrss/Airflow_over_mtn.html#:~:text=Dew%20Point%20Lapse%20Rate%20%EF%BF%BD,is%20equal%20to%20the%20MALR.
            # lapse_val = -0.002
            # self.parent_pixels['td_sealevel'] = (['time', 'y', 'x'],
                                               # self.parent_pixels['d2m'].values)
            # self.parent_pixels['td_sealevel'] = self.parent_pixels['td_sealevel'] - self.parent_pixels['elev'] * lapse_val
            # # bicubic spline to interpolate from 25km to 1km... cubic doesn't work here!
            # td_sealevel_interp = self.parent_pixels['td_sealevel'].interp_like(self.chess_grid, method='linear')
            # # adjust to the 1km elevation using same lapse rate
            # self.td_1km_elev = td_sealevel_interp + self.height_grid['elev'] * lapse_val
            
            ## Saturated and Actual vapour pressure
            # using Buck equation
            es_25km = 6.1121 * np.exp(
                (18.678 - (self.parent_pixels['t2m']-273.15)/234.5) *
                ((self.parent_pixels['t2m']-273.15)/(257.14 + (self.parent_pixels['t2m']-273.15)))
            )
            es_1km = 6.1121 * np.exp(
                (18.678 - (self.t_1km_elev-273.15)/234.5) *
                ((self.t_1km_elev-273.15)/(257.14 + (self.t_1km_elev-273.15)))
            )
            # from RH
            ea_25km = (rh_25km/100) * es_25km
            ea_1km = (self.rh_1km_interp/100) * es_1km
            del(rh_25km)
            
            ## Longwave radiation and Emissivity
            # method from Real-time and retrospective forcing in the North American Land
            #   Data Assimilation System (NLDAS) project, Brian A. Cosgrove,
            # which borrows model from
            # Satterlund, Donald R., An improved equation for estimating long‐wave radiation from the atmosphere
            # and I have taken calibrated constants from  M. Li, Y. Jiang, C.F.M. Coimbra
            #   On the determination of atmospheric longwave irradiance under all-sky conditions
            #   Sol. Energy, 144 (2017), pp. 40-48
            emms_25km = 1.02 * (1 - np.exp(-ea_25km*(self.parent_pixels['t2m']/1564.94)))
            emms_25km_interp = emms_25km.interp_like(self.chess_grid, method='linear')  
            emms_1km = 1.02 * (1 - np.exp(-ea_1km*(self.t_1km_elev/1564.94)))
            emms_ratio = emms_1km / emms_25km_interp
            
            Lw_25km_interp = self.parent_pixels['msdwlwrf'].interp_like(self.chess_grid, method='linear')  
            self.Lw_1km = emms_ratio * Lw_25km_interp
            self.Lw_1km = self.Lw_1km.transpose('time', 'y', 'x')
            # there is "supposed" to be a ratio of temperatures between downscaled/nondownscaled
            # of form (T_dwns / T_raw)^4 but this might enforce an elevation dependence 
            # on the longwave radiation which we might not want?
            
            ## Shortwave radiation terrain shading            
            doy = self.td.day_of_year - 1 # zero indexed
            self.shading_array = np.load(binary_batch_path + f'/terrain_shading/shading_mask_day_{doy}.npy')

    def load_precip_grid(self):
        # load the correct hi-res daily [or hourly] precipitation field
        precip_file = glob.glob(precip_fldr + f'/rainfall_hadukgrid_uk_1km_day_{self.td.year}{zeropad_strint(self.td.month)}01*.nc')[0]
        precip_dat = xr.open_dataset(precip_file) # total in mm
        self.precip_dat = precip_dat.loc[dict(
            projection_y_coordinate = precip_dat.projection_y_coordinate[
                (precip_dat.projection_y_coordinate <= float(self.chess_grid.y.max().values)) & 
                (precip_dat.projection_y_coordinate >= float(self.chess_grid.y.min().values))],
            projection_x_coordinate = precip_dat.projection_x_coordinate[
                (precip_dat.projection_x_coordinate <= float(self.chess_grid.x.max().values)) & 
                (precip_dat.projection_x_coordinate >= float(self.chess_grid.x.min().values))]
        )]
        # subset to correct day and take just rainfall data array 
        self.precip_dat = self.precip_dat.isel(time=self.td.day-1).rainfall.load()
        # get the corresponding interpolated mtpr (PRECIP) from ERA5:
        datestring = f'{self.td.year}{zeropad_strint(self.td.month)}{zeropad_strint(self.td.day)}'
        fname = hj_base + f'/data/uk/driving_data/era5/bn_grid/mtpr/era5_{datestring}_mtpr.nc'
        era5_precip = xr.open_dataset(fname)
        era5_precip = era5_precip.mtpr.sum(dim='time')*60*60 # convert from hourly kg/m2/s to daily total mm        
        # infill offshore precip with interpolated ERA5
        indices = np.where(np.isnan(self.precip_dat.values))
        self.precip_dat.values[indices] = era5_precip.values[indices]

    def sample_xyt(self, batch_type='train'):
        # choose a station and find its 1km pixel
        if batch_type=='train' or batch_type=='val':
            SID = np.random.choice(self.train_sites)
        elif batch_type=='test':
            SID = np.random.choice(self.train_sites + self.heldout_sites) 
        targ_site = self.site_metadata[self.site_metadata['SITE_ID']==SID]
        targ_loc = self.site_metadata[self.site_metadata['SITE_ID']==SID][['LATITUDE','LONGITUDE']]
        targ_yx = np.where(self.parent_pixels.pixel_id.values == targ_site['parent_pixel_id'].values)

        # grab a random dim_l x dim_l tile that contains that 1km pixel        
        ix = np.random.randint(max(0, targ_yx[1][0] - self.dim_l + 1),
                               min(self.parent_pixels.x.shape[0] - self.dim_l + 1, targ_yx[1][0] + 1))
        iy = np.random.randint(max(0, targ_yx[0][0] - self.dim_l + 1),
                               min(self.parent_pixels.y.shape[0] - self.dim_l + 1, targ_yx[0][0] + 1))
        it = np.random.randint(0, 24)
        return ix, iy, it

    def get_input_data(self, ix=None, iy=None, it=None, return_intermediates=False):
        if (ix is None) and (iy is None) and not (it is None):
            # for rectangular grid over whole space
            subdat = self.parent_pixels.isel(time=it)            
        else:
            subdat = self.parent_pixels.isel(time=it,
                                             y=range(iy, iy+self.dim_l),
                                             x=range(ix, ix+self.dim_l))
        
        timestamp = pd.to_datetime(self.parent_pixels.time.values[it], utc=True)
        #doy = timestamp.day_of_year
        #year_sin = np.sin(doy / 365. * 2*np.pi - np.pi/2.)
        #year_cos = np.cos(doy / 365. * 2*np.pi - np.pi/2.)
        #hour = timestamp.hour
        #hour_sin = np.sin(hour / 24. * 2*np.pi - 3*np.pi/4.)
        #hour_cos = np.cos(hour / 24. * 2*np.pi - 3*np.pi/4.)
         
        self.sp = SolarPosition(timestamp, timezone=0) # utc
        self.sp.calc_solar_angles(self.chess_grid.lat, self.chess_grid.lon)
        
        # subset hi-res fields to the chosen coarse tile
        x_inds = np.intersect1d(np.where(self.chess_grid.x.values < int(subdat.x.max().values) + self.res//2),
                                np.where(self.chess_grid.x.values > int(subdat.x.min().values) - self.res//2))
        y_inds = np.intersect1d(np.where(self.chess_grid.y.values < int(subdat.y.max().values) + self.res//2),
                                np.where(self.chess_grid.y.values > int(subdat.y.min().values) - self.res//2))
        sub_chess = self.chess_grid.isel(y=y_inds, x=x_inds)
        sub_topog = self.height_grid.isel(y=y_inds, x=x_inds)
        # precip_dat is 9am Day X - 9am Day X+1 and is labelled as 12 noon Day X
        sub_precip = self.precip_dat.isel(projection_y_coordinate=y_inds,
                                          projection_x_coordinate=x_inds)
        sub_sol_azimuth = self.sp.solar_azimuth_angle.isel(y=y_inds, x=x_inds)
        sub_sol_altitude = self.sp.solar_elevation.isel(y=y_inds, x=x_inds)

        # convert wind vectors and dewpoint temp to wind speed and relative humidity        
        subdat['t2m'].values -= 273.15 # K -> Celsius 
        subdat['d2m'].values -= 273.15 # K -> Celsius
        subdat['sp'].values /= 100.    # Pa -> hPa
        subdat['rh'] = relhum_from_dewpoint(subdat['t2m'], subdat['d2m'])
        subdat['ws'] = np.sqrt(np.square(subdat['v10']) + np.square(subdat['u10']))
        subdat = subdat.drop(['u10','v10', 'd2m'])
        
        # normalise lo res met data
        subdat['t2m'].values = (subdat['t2m'].values - nm.temp_mu) / nm.temp_sd
        subdat['sp'].values = (subdat['sp'].values - nm.p_mu) / nm.p_sd
        subdat['msdwlwrf'].values = (subdat['msdwlwrf'].values - nm.lwin_mu) / nm.lwin_sd
        subdat['msdwswrf'].values = np.log(1. + subdat['msdwswrf'].values) # log(1 + swin)
        subdat['msdwswrf'].values = (subdat['msdwswrf'].values - nm.logswin_mu) / nm.logswin_sd
        subdat['ws'].values = (subdat['ws'].values - nm.ws_mu) / nm.ws_sd
        subdat['rh'].values = (subdat['rh'].values - nm.rh_mu) / nm.rh_sd

        # normalise hi res data        
        sub_precip.values = sub_precip.values / nm.precip_norm # Should we log transform this too?
        sub_topog['aspect'].values = sub_topog['aspect'].values / 360. - 0.5 # so goes between -0.5 and 0.5
        for var in self.elev_vars:
            if var=='aspect': continue
            sub_topog[var].values = (sub_topog[var].values - nm.s_means.loc[var]) / nm.s_stds.loc[var]
        sub_sol_azimuth = np.deg2rad(sub_sol_azimuth.values)
        sub_sol_altitude = np.deg2rad(sub_sol_altitude.values)
        lat_grid = sub_chess.lat.values / 90.
        lon_grid = sub_chess.lon.values / 180.

        # create tensors with batch index and channel dim last
        coarse_input = torch.stack([torch.from_numpy(subdat[var].values).to(torch.float32) 
                                        for var in self.var_name_map.coarse], dim=-1)
        # therefore coarse scale variable order (in cosmos parlance) is
        self.coarse_variable_order = list(self.var_name_map['fine'])

        # get static inputs at the high resolution
        fine_input = torch.stack([torch.from_numpy(sub_topog[var].values).to(torch.float32) 
                                    for var in self.elev_vars], dim=-1)
        fine_input = torch.cat([torch.from_numpy(sub_chess['landfrac'].values).to(torch.float32)[...,None],
                                fine_input], dim=-1)

        # and join on precipitation
        fine_input = torch.cat([fine_input, 
            torch.from_numpy(sub_precip.values).to(torch.float32)[...,None]], dim = -1)
        
        # initial fine scale variable order is
        self.fine_variable_order = ['landfrac'] + self.elev_vars + ['rainfall']
        
        # add sun angles and position
        fine_input = torch.cat([fine_input,
                                torch.from_numpy(sub_sol_azimuth).to(torch.float32)[...,None],
                                torch.from_numpy(sub_sol_altitude).to(torch.float32)[...,None],
                                torch.from_numpy(lat_grid).to(torch.float32)[...,None],
                                torch.from_numpy(lon_grid).to(torch.float32)[...,None]
                                ], dim = -1)
                                        
        # add to fine scale variable order        
        self.fine_variable_order += ['solar_azimuth', 'solar_altitude', 'lat', 'lon']

        if return_intermediates:
            return coarse_input, fine_input, subdat, x_inds, y_inds, timestamp
        else:
            return coarse_input, fine_input

    def get_station_targets(self, subdat, x_inds, y_inds, timestamp,
                            batch_type='train', trim_edge_sites=True):
        ## get station targets within the dim_l x dim_l tile, ignoring stations on edge pixels       
        if trim_edge_sites:
            parents = subdat.pixel_id.values[1:-1,1:-1].flatten()
        else:
            parents = subdat.pixel_id.values.flatten()
        if batch_type=='train' or batch_type=='val':
            contained_sites = self.site_metadata[(self.site_metadata['SITE_ID'].isin(self.train_sites)) & 
                                                 (self.site_metadata['parent_pixel_id'].isin(parents))]
        elif batch_type=='test':
            contained_sites = self.site_metadata[(self.site_metadata['SITE_ID'].isin(self.train_sites + self.heldout_sites)) &
                                                 (self.site_metadata['parent_pixel_id'].isin(parents))]
        # then we use all these sites as targets for the loss calculation
        # (if they have data for the particular variable on the particular day)
        
        # find location of contained sites in local subset and pull out data
        contained_sites = contained_sites.set_index('SITE_ID')
        contained_sites['sub_x'] = -1
        contained_sites['sub_y'] = -1
        for var in self.var_name_map.fine:
            if var=='TD': continue
            contained_sites[var] = -1
        for sid in contained_sites.index:            
            this_x = np.where(x_inds == contained_sites.loc[sid,'chess_x'])[0]
            this_y = np.where(y_inds == contained_sites.loc[sid,'chess_y'])[0]
            contained_sites.loc[sid,'sub_x'] = int(this_x)
            contained_sites.loc[sid,'sub_y'] = int(this_y)
            try:
                this_dat = self.site_data[sid].loc[timestamp, :]
            except:
                this_dat = pd.Series(np.nan, self.var_name_map.fine)            
            for var in self.var_name_map.fine:
                if var=='TD': continue                
                contained_sites.loc[sid,var] = this_dat[var]
               
        # define station targets
        station_targets = contained_sites[
            ['sub_x', 'sub_y'] + list(self.var_name_map.fine)
        ]
        
        # trim sites with no data
        keepsites = station_targets[self.var_name_map.fine].dropna(how='all').index
        station_targets = station_targets.loc[keepsites]
        
        # create context/target splits
        context, targets = context_target_split(
            station_targets.index, context_frac=0.5, random_state=np.random.randint(1000))
        
        # create value/density pairs vector for context points (+1 for null tag)
        val_dense_vecs = np.zeros((2*self.var_name_map.shape[0], len(context)+1), dtype=np.float32)        
        for i, sid in enumerate(context):
            vardat = station_targets.loc[sid]
            for j, var in enumerate(self.var_name_map.fine):            
                thisval = vardat[var]
                if not np.isnan(thisval):
                    val_dense_vecs[2*j,i] = thisval
                    val_dense_vecs[2*j+1,i] = 1.
        
        # create location array to append
        X0 = station_targets.loc[context][['sub_y', 'sub_x']]
        X0 = np.array(X0) / (self.dim_h-1)
        X0 = np.vstack([X0, np.array([[-1, -1]])]) # add null tag
                
        if False:
            # if we want to scale the attention by distance:
            nbrs = NearestNeighbors(n_neighbors = X0.shape[0], algorithm='ball_tree').fit(X0)
            distances, indices = nbrs.kneighbors(self.X1)
            # these are ordered by nearest rather than in order of X0, so 
            # reorder by indices vectors
            distances = np.take_along_axis(distances, np.argsort(indices, axis=1), axis=1)
        
        # if False:
            ### old method using value/density grids
            # # create density and value grids for context stations
            # fs = 1
            # hfs = fs//2
            # gfilt = create_filter(fs)        
            # ysize = subdat.y.shape[0]*self.scale
            # xsize = subdat.x.shape[0]*self.scale
            # densities = np.zeros((ysize, xsize, self.var_name_map.shape[0]), dtype=np.float32)
            # values = np.zeros((ysize, xsize, self.var_name_map.shape[0]), dtype=np.float32)
            # #numbers = np.zeros((ysize, xsize, self.var_name_map.shape[0]), dtype=np.float32)
            # for i, var in enumerate(self.var_name_map.fine):            
                # vardat = station_targets.loc[context][['sub_x','sub_y',var]].dropna()
                # for j in range(vardat.shape[0]):
                    # # calculate filter edge locations and account for overhangs
                    # pt = vardat.sub_y.iloc[j], vardat.sub_x.iloc[j]
                    # le = max(0, hfs - pt[0])
                    # re = max(0, hfs - (ysize - 1 - pt[0]))
                    # te = max(0, hfs - pt[1])
                    # be = max(0, hfs - (xsize - 1 - pt[1]))
                    # lb = - (hfs - le)
                    # rb = hfs - re + 1
                    # tb = - (hfs - te)
                    # bb = hfs - be + 1
                    # densities[(pt[0]+lb):(pt[0]+rb), (pt[1]+tb):(pt[1]+bb), i] += \
                            # gfilt[(hfs+lb):(hfs+rb), (hfs+tb):(hfs+bb)]
                    # values[(pt[0]+lb):(pt[0]+rb), (pt[1]+tb):(pt[1]+bb), i] += \
                            # vardat[var].iloc[j] * gfilt[(hfs+lb):(hfs+rb), (hfs+tb):(hfs+bb)]
                    # #numbers[(pt[0]+lb):(pt[0]+rb), (pt[1]+tb):(pt[1]+bb), i] += 1.
                            
            # # then divide by densities to get weighted mean of values
            # mask = densities!=0
            # values[mask] /= densities[mask]
            # #values[mask] /= numbers[mask]
            
            # densities = torch.from_numpy(densities)
            # values = torch.from_numpy(values)
            
            # # return station data and interleaved density/value tensors
            # return ({'context':station_targets.loc[context], 'target':station_targets.loc[targets]},
                    # torch.flatten(torch.stack([values, densities], dim=-1), start_dim=-2, end_dim=-1))

        # return station data, value/density array and context locations
        return ({'context':station_targets.loc[context],
                 'target':station_targets.loc[targets]},
                 val_dense_vecs, X0)

    def get_constraints(self, x_inds, y_inds, it):
        ## Shortwave radiation, time of day dependent           
        self.Sw_1km = partition_interp_solar_radiation(
            self.parent_pixels['msdwswrf'][it,:,:],
            self.chess_grid,
            self.sp,
            self.parent_pixels['sp'][it,:,:],
            self.p_1km_elev[it,:,:],
            self.height_grid,
            self.shading_array[it,:]
        )
                
        # subset
        sub_temp_constraint = self.t_1km_elev.isel(time=it, y=y_inds, x=x_inds)
        sub_pres_constraint = self.p_1km_elev.isel(time=it, y=y_inds, x=x_inds)
        sub_relh_constraint = self.rh_1km_interp.isel(time=it, y=y_inds, x=x_inds)
        sub_lwin_constraint = self.Lw_1km.isel(time=it, y=y_inds, x=x_inds)
        sub_swin_constraint = self.Sw_1km.isel(y=y_inds, x=x_inds)
        
        # normalise
        sub_temp_constraint = (sub_temp_constraint - 273.15 - nm.temp_mu) / nm.temp_sd
        sub_pres_constraint = (sub_pres_constraint - nm.p_mu) / nm.p_sd
        sub_relh_constraint = (sub_relh_constraint - nm.rh_mu) / nm.rh_sd       
        sub_lwin_constraint = (sub_lwin_constraint - nm.lwin_mu) / nm.lwin_sd
        sub_swin_constraint = np.log(1. + sub_swin_constraint) # log(1 + swin)
        sub_swin_constraint = (sub_swin_constraint - nm.logswin_mu) / nm.logswin_sd
        # subdat['ws'] = (subdat['ws'] - nm.ws_mu) / nm.ws_sd
         
        # combine into tensor        
        constraints = torch.stack([
            torch.from_numpy(sub_temp_constraint.values).to(torch.float32),
            torch.from_numpy(sub_pres_constraint.values).to(torch.float32),
            torch.from_numpy(sub_swin_constraint.values).to(torch.float32),
            torch.from_numpy(sub_lwin_constraint.values).to(torch.float32),
            torch.from_numpy(sub_relh_constraint.values).to(torch.float32),
        ], dim=-1) # (Y, X, C)

        # therefore constraint variable order is
        self.constraint_variable_order = ['TA', 'PA', 'SWIN', 'LWIN', 'RH']
        return constraints

    def get_sample(self, batch_type='train'):
        ## sample a dim_l x dim_l tile
        ix, iy, it = self.sample_xyt(batch_type=batch_type)
        
        ## load input data
        coarse_input, fine_input, subdat, x_inds, y_inds, timestamp = self.get_input_data(
            ix, iy, it, return_intermediates=True
        )
        
        ## get station targets within the dim_l x dim_l tile        
        # context_grids
        station_targets, context_valdense_array, context_locations = self.get_station_targets(
            subdat, x_inds, y_inds, 
            timestamp, batch_type=batch_type
        )
        
        ## define constraints
        constraints = self.get_constraints(x_inds, y_inds, it)
        
        ## capture sample description
        sample_meta = {
            'timestamp':timestamp,
            'x_inds':x_inds,
            'y_inds':y_inds,
            'time_ind':it,
            'coarse_xl':ix,
            'coarse_yt':iy
        }
        
        #return coarse_input, fine_input, station_targets, constraints, context_grids
        return (coarse_input, fine_input, station_targets, constraints,
                context_valdense_array, context_locations, sample_meta)

    def get_batch(self, batch_size=1, batch_type='train', load_binary_batch=True):
        if load_binary_batch is False:
            self.parent_pixels = self.read_parent_pixel_day(batch_type=batch_type)            
            self.td = pd.to_datetime(self.parent_pixels.time[0].values)
        else:
            bfn = self.bin_batches[batch_type][np.random.randint(0, len(self.bin_batches[batch_type]))]            
            self.parent_pixels = pd.read_pickle(bfn)
            try:
                self.td = pd.to_datetime(bfn.split('_')[-1].replace('.pkl',''))
            except:
                self.td = pd.to_datetime(self.parent_pixels.time[0].values)        
        
        # calculate constraint grids
        self.prepare_constraints()
                
        # load the correct hi-res daily [or hourly] precipitation field
        self.load_precip_grid()

        coarse_inputs = []
        fine_inputs = []
        station_targets = []
        constraint_targets = []
        context_vd_arrays = []
        context_locations = []
        batch_metadata = []
        for b in range(batch_size):
            # generate batch from parent pixels
            # coarse_input, fine_input, station_target, constraints, context_grids = \
                # self.get_sample(batch_type=batch_type)
            (coarse_input, fine_input, station_target, constraints,
                context_valdense_array, context_locs, sample_meta) = \
                    self.get_sample(batch_type=batch_type)
            
            # append to storage
            coarse_inputs.append(coarse_input)
            #fine_inputs.append(torch.cat([fine_input, context_grids], dim=-1)) # cat on channel dim
            fine_inputs.append(fine_input)
            station_targets.append(station_target)
            constraint_targets.append(constraints)
            context_vd_arrays.append(context_valdense_array)
            context_locations.append(context_locs.T)
            batch_metadata.append(sample_meta)

        # pad context data to max size across batch
        c_sizes = [c.shape[-1] for c in context_vd_arrays]        
        maxsize = max(c_sizes)
        context_padding = [maxsize-cs for cs in c_sizes]
        context_vd_arrays = [torch.from_numpy(pad_contexts(arr, maxsize)).type(torch.float32) 
            for arr in context_vd_arrays]
        context_locations = [torch.from_numpy(pad_contexts(arr, maxsize)).type(torch.float32) 
            for arr in context_locations]
            
        coarse_inputs = torch.stack(coarse_inputs, dim=0)
        fine_inputs = torch.stack(fine_inputs, dim=0)
        constraint_targets = torch.stack(constraint_targets, dim=0)
        context_vd_arrays = torch.stack(context_vd_arrays, dim=0)
        context_locations = torch.stack(context_locations, dim=0)
        
        # return (coarse_inputs.permute(0,3,1,2), fine_inputs.permute(0,3,1,2),
                # station_targets, constraint_targets.permute(0,3,1,2))
        return (coarse_inputs.permute(0,3,1,2),
                fine_inputs.permute(0,3,1,2),
                station_targets,
                constraint_targets.permute(0,3,1,2),
                context_vd_arrays,
                context_locations,
                context_padding,
                batch_metadata)

    def get_chess_batch(self, var, batch_size=1, batch_type='train',
                        load_binary_batch=True):            
        if load_binary_batch is False:
            self.parent_pixels = self.read_parent_pixel_day(batch_type=batch_type)            
            self.td = pd.to_datetime(self.parent_pixels.time[0].values)
        else:
            bfn = self.bin_batches[batch_type][np.random.randint(0, len(self.bin_batches[batch_type]))]            
            self.parent_pixels = pd.read_pickle(bfn)
            try:
                self.td = pd.to_datetime(bfn.split('_')[-1].replace('.pkl',''))
            except:
                self.td = pd.to_datetime(self.parent_pixels.time[0].values)        

        ## average over the day
        self.parent_pixels = self.parent_pixels.mean('time')
        
        ## convert wind vectors and dewpoint temp to wind speed and relative humidity        
        self.parent_pixels['t2m'].values -= 273.15 # K -> Celsius 
        self.parent_pixels['d2m'].values -= 273.15 # K -> Celsius
        self.parent_pixels['sp'].values /= 100.    # Pa -> hPa
        self.parent_pixels['rh'] = relhum_from_dewpoint(self.parent_pixels['t2m'],
            self.parent_pixels['d2m'])
        self.parent_pixels['ws'] = (np.sqrt(np.square(self.parent_pixels['v10']) +
            np.square(self.parent_pixels['u10'])))
        self.parent_pixels = self.parent_pixels.drop(['u10','v10', 'd2m'])
        
        ## drop variables we don't need
        era5_var = self.var_name_map.loc[var].coarse
        to_drop = list(self.parent_pixels.keys())
        to_drop.remove(era5_var)
        to_drop.remove('pixel_id')
        self.parent_pixels = self.parent_pixels.drop(to_drop)        
        
        ## load chess data
        chess_var = self.var_name_map.loc[var].chess
        chess_dat = load_process_chess(self.td.year, self.td.month, self.td.day,
                                       var=chess_var)
                                       
        ## pool to scale and fill in the parent pixels
        chess_pooled = pooling(chess_dat[chess_var].values, (self.scale, self.scale),
                               method='mean')
        chess_mask = ~np.isnan(chess_pooled)
        self.parent_pixels[era5_var].values[chess_mask] = chess_pooled[chess_mask]        

        coarse_inputs = []
        fine_inputs = []
        station_targets = []
        constraint_targets = []
        context_vd_arrays = []
        context_locations = []
        batch_metadata = []
        for b in range(batch_size):
            # generate batch from parent pixels            
            (coarse_input, fine_input, station_target, constraints,
                context_valdense_array, context_locs, sample_meta) = \
                    self.get_sample(batch_type=batch_type)
            
            # append to storage
            coarse_inputs.append(coarse_input)
            #fine_inputs.append(torch.cat([fine_input, context_grids], dim=-1)) # cat on channel dim
            fine_inputs.append(fine_input)
            station_targets.append(station_target)
            constraint_targets.append(constraints)
            context_vd_arrays.append(context_valdense_array)
            context_locations.append(context_locs.T)
            batch_metadata.append(sample_meta)

        # pad context data to max size across batch
        c_sizes = [c.shape[-1] for c in context_vd_arrays]        
        maxsize = max(c_sizes)
        context_padding = [maxsize-cs for cs in c_sizes]
        context_vd_arrays = [torch.from_numpy(pad_contexts(arr, maxsize)).type(torch.float32) 
            for arr in context_vd_arrays]
        context_locations = [torch.from_numpy(pad_contexts(arr, maxsize)).type(torch.float32) 
            for arr in context_locations]
            
        coarse_inputs = torch.stack(coarse_inputs, dim=0)
        fine_inputs = torch.stack(fine_inputs, dim=0)
        constraint_targets = torch.stack(constraint_targets, dim=0)
        context_vd_arrays = torch.stack(context_vd_arrays, dim=0)
        context_locations = torch.stack(context_locations, dim=0)
        
        # return (coarse_inputs.permute(0,3,1,2), fine_inputs.permute(0,3,1,2),
                # station_targets, constraint_targets.permute(0,3,1,2))
        return (coarse_inputs.permute(0,3,1,2),
                fine_inputs.permute(0,3,1,2),
                station_targets,
                constraint_targets.permute(0,3,1,2),
                context_vd_arrays,
                context_locations,
                context_padding,
                batch_metadata)

    def get_big_small_batch(self):
        ''' TODO: define function that gives one target batch and another
            set of inputs that are bigger or smaller to run alongside the 
            normal batch and then use the central pixels as comparison in
            new loss component? '''
        pass

    def batch_all_space(self, batch_type='train', load_binary_batch=True, min_overlap=0):
        if load_binary_batch is False:
            self.parent_pixels = self.read_parent_pixel_day(batch_type=batch_type)
        else:
            bfn = self.bin_batches[batch_type][np.random.randint(0, len(self.bin_batches[batch_type]))]            
            self.parent_pixels = pd.read_pickle(bfn)
        
        # extract timestamp
        self.td = pd.to_datetime(self.parent_pixels.time[0].values)
        
        # calculate constraint grids
        self.prepare_constraints()
                
        # load the correct hi-res daily [or hourly] precipitation field
        self.load_precip_grid()

        # choose a time slice
        it = np.random.randint(0, 24)
        
        # divide space up into dim_l x dim_l tiles
        ixs = define_tile_start_inds(self.parent_pixels.x.shape[0], self.dim_l, min_overlap)
        iys = define_tile_start_inds(self.parent_pixels.y.shape[0], self.dim_l, min_overlap)
        coarse_inputs = []
        fine_inputs = []
        yx_tiles = []
        for ix in ixs:
            for iy in iys:
                coarse_input, fine_input, subdat, x_inds, y_inds, timestamp = self.get_input_data(
                    ix, iy, it, return_intermediates=True
                )
                coarse_inputs.append(coarse_input)
                fine_inputs.append(fine_input)
                
                ## get all station targets / context, x_inds, y_inds = all
                #station_targets, context_grids = self.get_station_targets(subdat, x_inds, y_inds, timestamp, batch_type=batch_type)
                
                yx_tiles.append((iy, ix))

        coarse_inputs = torch.stack(coarse_inputs, dim=0)
        fine_inputs = torch.stack(fine_inputs, dim=0)

        return (coarse_inputs.permute(0,3,1,2),
                fine_inputs.permute(0,3,1,2), ixs, iys)

    def get_all_space(self, batch_type='train', load_binary_batch=True):
        if load_binary_batch is False:
            self.parent_pixels = self.read_parent_pixel_day(batch_type=batch_type)
        else:
            bfn = self.bin_batches[batch_type][np.random.randint(0, len(self.bin_batches[batch_type]))]            
            self.parent_pixels = pd.read_pickle(bfn)
        
        # extract timestamp
        self.td = pd.to_datetime(self.parent_pixels.time[0].values)
        
        # calculate constraint grids
        self.prepare_constraints()
                
        # load the correct hi-res daily [or hourly] precipitation field
        self.load_precip_grid()

        # choose a time slice
        it = np.random.randint(0, 24)
        
        # take one big (rectangular) tile
        ix = None
        iy = None
        coarse_input, fine_input, subdat, x_inds, y_inds, timestamp = self.get_input_data(
            ix, iy, it, return_intermediates=True
        )        
        station_targets, context_grids = self.get_station_targets(
            subdat, x_inds, y_inds, timestamp,
            batch_type=batch_type,
            trim_edge_sites=False
        )
        fine_input = torch.cat([fine_input, context_grids], dim=-1)      
        
        # add batch dim at 0  
        coarse_inputs = torch.stack([coarse_input], dim=0)
        fine_inputs = torch.stack([fine_input], dim=0)

        return (coarse_inputs.permute(0,3,1,2),
                fine_inputs.permute(0,3,1,2))


if __name__=="__main__":
    if False:
        dg = data_generator()
        
        batch_size = 3
        batch_type = 'train'
        load_binary_batch = True
              
        masks = make_mask_list(dg.dim_l, dg.dim_l, dg.scale)
        batch = dg.get_batch(batch_size=batch_size,
                             batch_type=batch_type,
                             load_binary_batch=load_binary_batch)

        b = Batch(batch, masks, dg.X1, device=None)
        gridYX = torch.from_numpy(dg.X1.T).type(torch.float32)
        gridYX = gridYX.unsqueeze(0).expand(batch_size, -1, -1)


        ###########
        a1 = time.time()
        bfn = dg.bin_batches[batch_type][np.random.randint(0, len(dg.bin_batches[batch_type]))]
        dg.parent_pixels = pd.read_pickle(bfn)
        dg.td = pd.to_datetime(dg.parent_pixels.time[0].values)
        b1 = time.time()
        
        a2 = time.time()
        # calculate constraint grids
        dg.prepare_constraints()
        b2 = time.time()
                
        a3 = time.time()
        # load the correct hi-res daily [or hourly] precipitation field
        dg.load_precip_grid()
        b3 = time.time()
       
        # coarse_input, fine_input, station_target, constraints, context_grids = \
                # dg.get_sample(batch_type=batch_type)
        
        a4 = time.time()
        ## examining station pixel locations
        ix, iy, it = dg.sample_xyt(batch_type=batch_type)
        b4 = time.time()
        
        # fig, ax = plt.subplots(1,2)
        # dg.parent_pixels.t2m[it,:,:].plot(ax=ax[0])
        # dg.parent_pixels.t2m[it,iy:(iy+8),ix:(ix+8)].plot(ax=ax[1])
        # plt.show()
        
        a5 = time.time()
        ## load input data
        coarse_input, fine_input, subdat, x_inds, y_inds, timestamp = dg.get_input_data(
            ix, iy, it, return_intermediates=True
        )
        b5 = time.time()

        station_targets, context_valdense_array, context_locations = dg.get_station_targets(
            subdat, x_inds, y_inds, 
            timestamp, batch_type=batch_type
        )
        
        ## define constraints
        constraints = dg.get_constraints(x_inds, y_inds, it)

        # fig, ax = plt.subplots(1,2)
        # dg.chess_grid.landfrac.plot(ax=ax[0])
        # dg.chess_grid.isel(y=y_inds, x=x_inds).landfrac.plot(ax=ax[1])
        # plt.show()

        ## outputting an interim batch:
        # pd.to_pickle(
            # dict(parent_pixels = dg.parent_pixels,
                 # precip_dat = dg.precip_dat,
                 # T_constraint = dg.t_1km_elev,
                 # P_constraint = dg.p_1km_elev,
                 # RH_constraint = dg.rh_1km_interp,
                 # LWIN_constraint = dg.Lw_1km),
            # '/home/users/doran/temp/test_batch.pkl',
            # compression={'method': 'bz2', 'compresslevel': 9}
        # )
        #sub_swin_constraint = self.Sw_1km.isel(y=y_inds, x=x_inds)
        


        #############################
        ## plotting fields
        pltdir = './plots/example_fields/'
        Path(pltdir).mkdir(parents=True, exist_ok=True)
        # coarse inputs
        for i in range(dg.var_name_map.shape[0]):
            fig, ax = plt.subplots()
            ax.imshow(dg.parent_pixels[dg.var_name_map.coarse.iloc[i]].values[9,::-1,:])
            plt.axis('off')
            plt.savefig(pltdir + f'/{dg.var_name_map.fine.iloc[i]}_coarse.png', bbox_inches='tight')
            plt.close()
            
        # hi-res inputs
        for i,n in enumerate(dg.fine_variable_order[:-2]):
            fig, ax = plt.subplots()
            if n=='rainfall':
                ax.imshow(dg.precip_dat.values[::-1,:])
            elif n=='solar_azimuth':
                ax.imshow(dg.sp.solar_azimuth_angle.values[::-1,:])
            elif n=='solar_altitude':
                ax.imshow(dg.sp.solar_elevation.values[::-1,:])
            elif n=='landfrac':
                ax.imshow(dg.chess_grid[n].values[::-1,:])
            else:
                ax.imshow(dg.height_grid[n].values[::-1,:])
            
            plt.axis('off')
            plt.savefig(pltdir + f'/{n}_fine.png', bbox_inches='tight')
            plt.close()
            
        # point observations
        fine_input = fine_input.numpy()
        station_target['context']['adj_y'] = dg.dim_h - station_target['context'].sub_y
        station_target['target']['adj_y'] = dg.dim_h - station_target['target'].sub_y
        
        fig, ax = plt.subplots()
        ax.imshow(fine_input[::-1,:,0], alpha=0.7, cmap='Greys')
        ax.scatter(station_target['context'].sub_x, station_target['context'].adj_y,
                   s=20, c='#1f77b4', marker='s')
        texts = [plt.text(station_target['context'].sub_x.iloc[i],
                          station_target['context'].adj_y.iloc[i],
                          station_target['context'].index[i]) 
                            for i in range(station_target['context'].shape[0])]
        
        ax.scatter(station_target['target'].sub_x, station_target['target'].adj_y,
                   s=20, c='#17becf', marker='^')
        texts = [plt.text(station_target['target'].sub_x.iloc[i],
                          station_target['target'].adj_y.iloc[i],
                          station_target['target'].index[i]) 
                            for i in range(station_target['context'].shape[0])]
        adjust_text(texts, arrowprops=dict(arrowstyle="-", color='k', lw=0.5))
        plt.axis('off')
        plt.savefig(pltdir + f'/context_target_sites.png', bbox_inches='tight')
        plt.close()
        
        # constraints        
        for i,n in enumerate(dg.constraint_variable_order):
            fig, ax = plt.subplots()
            if n=='TA':
                ax.imshow(dg.t_1km_elev.values[9,::-1,:])
            elif n=='PA':
                ax.imshow(dg.p_1km_elev.values[9,::-1,:])
            elif n=='SWIN':
                ax.imshow(dg.Sw_1km.values[::-1,:])
            elif n=='LWIN':
                ax.imshow(dg.Lw_1km.values[9,::-1,:])
            elif n=='RH':
                ax.imshow(dg.rh_1km_interp.values[9,::-1,:])
            plt.axis('off')
            plt.savefig(pltdir + f'/{n}_constraint.png', bbox_inches='tight')
            plt.close()
        
        
        ## looking at KNN of context points with points on grid
        from sklearn.neighbors import NearestNeighbors
        X0 = station_targets[1]['context'][['sub_y', 'sub_x']]
        X0 = np.array(X0)
        X1 = np.where(np.ones((200,200)))
        X1 = np.hstack([X1[0][...,np.newaxis], X1[1][...,np.newaxis]])
        nbrs = NearestNeighbors(n_neighbors=200*200, algorithm='ball_tree').fit(X1)
        distances, indices = nbrs.kneighbors(X0)
        
        a = torch.randn((3, 16, 200, 200)) # embedding dim == 12
        b = torch.randn((3, 16, 12)) # 12 context points, say        
        
        xs = np.linspace(0, 1, 200)
        ys = np.linspace(0, 1, 200)
        xx, yy = np.meshgrid(xs, ys)
        xx = torch.from_numpy(xx).type(torch.float32).unsqueeze(0).unsqueeze(0)
        yy = torch.from_numpy(yy).type(torch.float32).unsqueeze(0).unsqueeze(0)
        xx = xx.expand((a.shape[0], -1, -1, -1))
        yy = yy.expand((a.shape[0], -1, -1, -1))
        
        a1 = torch.cat([a, yy], dim=1)
        a1 = torch.cat([a1, xx], dim=1)
        a1 = torch.reshape(a1, (a1.shape[0], a1.shape[1], a1.shape[2]*a1.shape[3]))
     

        
