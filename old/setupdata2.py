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
from scipy.ndimage import gaussian_filter

from timeout import Timeout
from data_classes.met_data import ERA5Data
from data_classes.cosmos_data import CosmosMetaData, CosmosData
from utils import *
from params import normalisation as nm
from params import data_pars as dp
from solar_position import SolarPosition

EPS = 1e-10

hj_base = '/gws/nopw/j04/hydro_jules/'
hj_ancil_fldr = hj_base + '/data/uk/ancillaries/'
era5_fldr = hj_base + '/data/uk/driving_data/era5/bn_grid/'
nz_base = '/gws/nopw/j04/ceh_generic/netzero/'
binary_batch_path = nz_base + '/downscaling/training_data/'
precip_fldr = '/badc/ukmo-hadobs/data/insitu/MOHC/HadOBS/HadUK-Grid/v1.1.0.0/1km/rainfall/day/latest/'
home_data_dir = '/home/users/doran/data_dump/'
midas_fldr = home_data_dir + '/MetOffice/midas_data/'
chessmet_dir = hj_base + '/data/uk/driving_data/chess/chess-met/daily/'

def load_process_chess(year, month, day, var=None, normalise=True):
    ## load
    if var is None:
        chess_dat = xr.open_mfdataset(chessmet_dir + \
            f'/chess-met_*_gb_1km_daily_{year}{zeropad_strint(month)}*.nc')
        chess_dat = chess_dat.isel(time=day-1)
    elif var=='huss':
        chess_dat = xr.open_mfdataset(
            [glob.glob(chessmet_dir + f'/chess-met_{var}_gb_1km_daily_{year}{zeropad_strint(month)}*.nc')[0],
             glob.glob(chessmet_dir + f'/chess-met_tas_gb_1km_daily_{year}{zeropad_strint(month)}*.nc')[0],
             glob.glob(chessmet_dir + f'/chess-met_psurf_gb_1km_daily_{year}{zeropad_strint(month)}*.nc')[0]
            ]
        )
        chess_dat = chess_dat.isel(time=day-1)
    elif var=='rlds':
        chess_dat = xr.open_mfdataset(
            [glob.glob(chessmet_dir + f'/chess-met_{var}_gb_1km_daily_{year}{zeropad_strint(month)}*.nc')[0],
             glob.glob(chessmet_dir + f'/chess-met_huss_gb_1km_daily_{year}{zeropad_strint(month)}*.nc')[0],
             glob.glob(chessmet_dir + f'/chess-met_psurf_gb_1km_daily_{year}{zeropad_strint(month)}*.nc')[0],
             glob.glob(chessmet_dir + f'/chess-met_tas_gb_1km_daily_{year}{zeropad_strint(month)}*.nc')[0]
            ]
        )
        chess_dat = chess_dat.isel(time=day-1)
    elif var=='rsds':
        chess_dat = xr.open_mfdataset(
            [glob.glob(chessmet_dir + f'/chess-met_{var}_gb_1km_daily_{year}{zeropad_strint(month)}*.nc')[0],
             glob.glob(chessmet_dir + f'/chess-met_psurf_gb_1km_daily_{year}{zeropad_strint(month)}*.nc')[0]
            ]
        )
        chess_dat = chess_dat.isel(time=day-1)
    else:        
        chess_dat = xr.open_dataset(glob.glob(chessmet_dir + \
            f'/chess-met_{var}_gb_1km_daily_{year}{zeropad_strint(month)}*.nc')[0])
        chess_dat = chess_dat.isel(time=day-1)

    ## rescale
    if var=='huss' or var=='rlds' or var is None:
        # specific humidity to RH, requires psurf in Pa and T in K
        chess_dat.huss.values = (0.263 * chess_dat.psurf.values * chess_dat.huss.values * 
            np.exp((-17.67 * (chess_dat.tas.values - 273.16)) /
                    (chess_dat.tas.values - 29.65)))
    if var=='tas' or var=='huss' or var is None:
         # K to C
        chess_dat.tas.values = chess_dat.tas.values - 273.15
    if var=='psurf' or var=='huss' or var=='rsds' or var is None:
        # Pa to hPa
        chess_dat.psurf.values = chess_dat.psurf.values / 100.
    if normalise:
        return normalise_chess_data(chess_dat, var)
    else:
        return chess_dat

def normalise_chess_data(chess_dat, var):
    ## normalise
    if var=='rlds' or var is None:
        # incoming longwave radiation
        chess_dat.rlds.values = (chess_dat.rlds.values - nm.lwin_mu) / nm.lwin_sd
    if var=='rsds' or var is None:
        # incoming shortwave radiation
        #chess_dat.rsds.values = np.log(1. + chess_dat.rsds.values)
        #chess_dat.rsds.values = (chess_dat.rsds.values - nm.logswin_mu) / nm.logswin_sd
        chess_dat.rsds.values = chess_dat.rsds.values / nm.swin_norm
    if var=='psurf' or var=='huss' or var=='rsds' or var is None:
        # air pressure
        chess_dat.psurf.values = (chess_dat.psurf.values - nm.p_mu) / nm.p_sd
    if var=='huss' or var=='rlds' or var is None:
        # relative humidity            
        chess_dat.huss.values = (chess_dat.huss.values - nm.rh_mu) / nm.rh_sd
    if var=='tas' or var=='huss' or var is None:
        # temperature
        chess_dat.tas.values = (chess_dat.tas.values - nm.temp_mu) / nm.temp_sd
    if var=='sfcWind' or var is None:
        # wind speed            
        chess_dat.sfcWind.values = (chess_dat.sfcWind.values - nm.ws_mu) / nm.ws_sd
    return chess_dat

def create_chess_pred(sample_metadata, datgen, var, pred):
    ## load
    chess_dat = load_process_chess(sample_metadata[0]['timestamp'].year,
                                   sample_metadata[0]['timestamp'].month,
                                   sample_metadata[0]['timestamp'].day,
                                   var=datgen.var_name_map.loc[var].chess)    
    ## create "chess-pred"
    pred3 = np.zeros(pred.shape)
    for bb in range(len(sample_metadata)):    
        subdat = chess_dat.isel(y=sample_metadata[bb]['y_inds'],
                                x=sample_metadata[bb]['x_inds'])
        for j, cv in enumerate(datgen.coarse_variable_order):
            vv = datgen.var_name_map[datgen.var_name_map['fine']==cv]['chess'].values[0]
            pred3[bb,j,:,:] = subdat[vv]
    return pred3

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

def load_midas_elevation():
    # load midas site elevation
    midas_meta = pd.read_csv('/badc/ukmo-midas/metadata/SRCE/SRCE.DATA.COMMAS_REMOVED')
    mid_cols = ['SRC_ID', 'SRC_NAME','HIGH_PRCN_LAT',# - Latitude in 0.001 deg
                'HIGH_PRCN_LON',# - Longitude in 0.001 deg
                'LOC_GEOG_AREA_ID', 'REC_ST_IND', 'SRC_BGN_DATE',
                'SRC_TYPE','GRID_REF_TYPE', 'EAST_GRID_REF',
                'NORTH_GRID_REF','HYDR_AREA_ID', 'POST_CODE', 
                'SRC_END_DATE', 'ELEVATION', # - Metres
                'WMO_REGION_CODE', 'PARENT_SRC_ID', 'ZONE_TIME',
                'DRAINAGE_STREAM_ID', 'SRC_UPD_DATE',
                'MTCE_CTRE_CODE', 'PLACE_ID',
                'LAT_WGs84',# - SRC higher precision latitude to 5 dp - please see note below
                'LONG_WGS84',# - SRC higer precision longitude to 5dp - please see note below
                'SRC_GUID', 'SRC_GEOM', 'SRC_LOCATION_TYPE']
    midas_meta.columns = mid_cols[:24]
    return midas_meta[['SRC_ID', 'SRC_NAME', 'ELEVATION']]

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

def load_landcover():
    import rasterio
    fldr = hj_base+'/data/uk/soil_moisture_map/ancillaries/land_cover_map/2015/data/'
    fnam = '/LCM2015_GB_1km_percent_cover_aggregate_class.tif'
    lcm = rasterio.open(fldr + fnam)
    lcm = lcm.read()
    class_names = ['Broadleaf_woodland',
                   'Coniferous_woodland',
                   'Arable',
                   'Improved_grassland',
                   'Semi-natural_grassland',
                   'Mountain_heath_bog',
                   'Saltwater',
                   'Freshwater',
                   'Coastal',
                   'Urban']
    l_wooded = lcm[:2,:,:].sum(axis=0)
    l_open = lcm[2:5,:,:].sum(axis=0) + lcm[6:9,:,:].sum(axis=0)
    l_high = lcm[5,:,:]
    l_urban = lcm[9,:,:]
    if False:
        fig, ax = plt.subplots(2,2)
        ax[0,0].imshow(l_wooded)
        ax[0,1].imshow(l_open)
        ax[1,0].imshow(l_high)
        ax[1,1].imshow(l_urban)
        plt.show()
    # access using lcm[:, lcm.shape[1]-1-chess_y, chess_x] # as indexes y-inverted compared to chess
    return np.stack([l_wooded, l_open, l_high, l_urban], axis=0).astype(np.float32)/100.

def scale_by_coarse_data(fine_data, coarse_data, fine_grid, scale):
    # renorm to match the input coarse data        
    c_coarse = coarse_data.copy()    
    c_coarse.values = pooling(fine_data.values.T, (scale, scale), method='mean').T
    grid_ratio = coarse_data / c_coarse
    grid_ratio_interp = grid_ratio.interp_like(fine_grid, method='linear')
    return fine_data * grid_ratio_interp

def partition_interp_solar_radiation(swin_coarse, fine_grid, sp,
                                     press_coarse, press_fine,
                                     height_grid, shading_array, scale):
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
            np.sin(np.deg2rad(height_grid.slope.values[mask])) *
            np.cos(np.deg2rad(sp.solar_azimuth_angle.values[mask] - height_grid.aspect.values[mask]))
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
        
        # renorm to match the input coarse SWIN        
        # to counteract errors in partitioning
        Sw_1km = scale_by_coarse_data(Sw_1km.copy(), swin_coarse, fine_grid, scale)        
        return Sw_1km
    

class Batch:
    def __init__(self, batch, masks=None, constraints=True,
                 var_list=None, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.raw_station_dict = batch['station_targets']
            self.batch_metadata = batch['batch_metadata']
            
            self.coarse_inputs = batch['coarse_inputs'].to(device)
            self.fine_inputs = batch['fine_inputs'].to(device)
            self.context_station_dict = tensorise_station_targets(
                [s['context'] for s in batch['station_targets']],
                device=device,
                var_list=var_list
            )
            self.target_station_dict = tensorise_station_targets(
                [s['target'] for s in batch['station_targets']],
                device=device,
                var_list=var_list
            )
            if constraints:
                self.constraint_targets = batch['constraints'].to(device)
            else:
                self.constraint_targets = None
            
            #self.context_data = batch['station_data'].to(device)
            #self.context_locs = batch['context_locations'].to(device)
            self.context_data = [a.to(device) for a in batch['station_data']]
            self.context_locs = [a.to(device) for a in batch['context_locations']]
            
            # number of hourly observations aggregated to get daily avg
            self.context_num_obs = [s['context'].values.astype(np.int32).flatten()
                for s in batch['station_num_obs']]
            self.target_num_obs = [s['target'].values.astype(np.int32).flatten()
                for s in batch['station_num_obs']]
            
            n_gridpts = batch['fine_inputs'].shape[-1] * batch['fine_inputs'].shape[-2]
            #n_cntxpts = batch['context_locations'].shape[-1]
            n_batch = batch['coarse_inputs'].shape[0]
                       
            # self.context_mask = make_context_mask(n_gridpts, n_cntxpts,
                                                  # batch['context_padding']).to(device)
            #self.context_mask = [make_context_mask2(n_gridpts, batch['context_locations'][b].shape[-1]).to(device) for b in range(n_batch)]
            #self.masks = [m.type(torch.bool).to(device) for m in masks]        
            
        except:
            print('Creating empty batch')
            self.raw_station_dict = None
            self.batch_metadata = None
            self.coarse_inputs = None
            self.fine_inputs = None
            self.context_station_dict = None
            self.target_station_dict = None
            self.constraint_targets = None
            #self.masks = None
            self.context_data = None
            self.context_locs = None
            #self.context_mask = None            
            self.context_num_obs = None
            self.target_num_obs = None
    
    def copy(self, other_batch):
        self.raw_station_dict = other_batch.raw_station_dict.copy()
        self.batch_metadata = other_batch.batch_metadata.copy()
        self.coarse_inputs = other_batch.coarse_inputs.clone()
        self.fine_inputs = other_batch.fine_inputs.clone()
        self.context_station_dict = other_batch.context_station_dict.copy()
        self.target_station_dict = other_batch.target_station_dict.copy()
        try:
            self.constraint_targets = other_batch.constraint_targets.clone()
        except:
            self.constraint_targets = None
        #self.masks = [m.clone() for m in other_batch.masks]
        self.context_data = [m.clone() for m in other_batch.context_data]
        self.context_locs = [m.clone() for m in other_batch.context_locs]
        #self.context_mask = [m.clone() for m in other_batch.context_mask]        
        self.context_num_obs = [m.copy() for m in other_batch.context_num_obs]
        self.target_num_obs = [m.copy() for m in other_batch.target_num_obs]
        

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
        (int(self.chess_grid.y.max() + (dp.res - self.chess_grid.y.max() % dp.res)) // dp.res)
        (int(self.chess_grid.x.max() + (dp.res - self.chess_grid.x.max() % dp.res)) // dp.res)
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
        parent_pixels = xr.open_dataset(home_data_dir + '/chess/chess_25km_pixel_ids.nc')
        parent_pixels = parent_pixels.load()
        
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
        self.targ_var_depends['TA'] = ['landfrac', 'elev']
        self.targ_var_depends['PA'] = ['landfrac', 'elev']
        self.targ_var_depends['SWIN'] = ['landfrac', 'elev',
                                         'illumination_map',                                         
                                         'solar_altitude',
                                         'sunlight_hours', 'PA']
        self.targ_var_depends['LWIN'] = ['landfrac', 'elev',
                                         'sunlight_hours',
                                         'land_cover', 'RH']
        self.targ_var_depends['WS'] = ['landfrac', 'elev', 'stdev',
                                       'slope', 'land_cover']
        self.targ_var_depends['RH'] = ['landfrac', 'TA', 'PA']
        
        # load site metadata/locations
        self.site_metadata = CosmosMetaData()
        midas_metadata = pd.read_csv(home_data_dir + '/MetOffice/midas_site_locations.csv')
        midas_site_ids = midas_metadata.SITE_ID.str.split('_', expand=True)
        midas_site_ids.columns = ['SRC_ID', 'SRC_NAME']
        midas_site_ids['SRC_ID'] = midas_site_ids['SRC_ID'].astype(int)
        midas_elevs = load_midas_elevation()
        midas_elevs.columns  = ['SRC_ID', 'SRC_NAME', 'ALTITUDE']
        midas_metadata = (pd.concat([midas_metadata, midas_site_ids], axis=1)
            .merge(midas_elevs.drop('SRC_NAME', axis=1), on='SRC_ID'))        

        # load site data
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

        ''' we need to remove the Ireland sites from here as we 
        don't have the ancillary data for them currently '''
        lat_up = 55.3
        lat_down = 53.0
        lon_right = -5.4
        lon_left = -8.3        
        ireland_sites = self.site_metadata[
            ((self.site_metadata['LATITUDE']>lat_down) & 
             (self.site_metadata['LATITUDE']<lat_up) &
             (self.site_metadata['LONGITUDE']>lon_left) &
             (self.site_metadata['LONGITUDE']<lon_right))
        ]
        for sid in ireland_sites.SITE_ID:
            self.site_data[sid] = None
            self.site_data.pop(sid)
        self.site_metadata = self.site_metadata[~self.site_metadata['SITE_ID'].isin(ireland_sites['SITE_ID'])]

        self.site_metadata['ALTITUDE'] = self.site_metadata.ALTITUDE.astype(np.float32)

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
        ys = parent_pixels.y.shape[0]
        xs = parent_pixels.x.shape[0]        
        parent_pixels['elev'] = (['y', 'x'], np.ones((ys, xs), dtype=np.float32)*np.nan)
        source = self.height_grid['elev'].values
        parent_pixels['elev'] = (('y', 'x'), 
            skimage.measure.block_reduce(
                source, (self.scale, self.scale), np.nanmean
            )[:ys,:xs] # trim right/bottom edges
        )
        del(source)
        
        # load land cover for wind model
        self.land_cover = load_landcover()
        self.lcm_names = ['l_wooded', 'l_open', 'l_mountain-heath', 'l_urban']
               
        # normalise site data
        for SID in list(self.site_data.keys()):
            # incoming radiation                        
            self.site_data[SID].loc[:, 'LWIN'] = (self.site_data[SID].loc[:, 'LWIN'] - nm.lwin_mu) / nm.lwin_sd
            # first log transform SWIN
            #self.site_data[SID].loc[:, 'SWIN'] = np.log(1. + self.site_data[SID].loc[:, 'SWIN'])
            # then standardise
            #self.site_data[SID].loc[:, 'SWIN'] = (self.site_data[SID].loc[:, 'SWIN'] - nm.logswin_mu) / nm.logswin_sd
            self.site_data[SID].loc[:, 'SWIN'] = self.site_data[SID].loc[:, 'SWIN'] / nm.swin_norm
            
            # air pressure
            self.site_data[SID].loc[:, ['PA']] = (self.site_data[SID].loc[:, ['PA']] - nm.p_mu) / nm.p_sd
            
            # relative humidity            
            self.site_data[SID].loc[:, ['RH']] = (self.site_data[SID].loc[:, ['RH']] - nm.rh_mu) / nm.rh_sd
            
            # temperature
            #print(self.site_data[SID].loc[:, ['TA']].mean()) check T is in Kelvin?
            self.site_data[SID].loc[:, ['TA']] = (self.site_data[SID].loc[:, ['TA']] - 273.15 - nm.temp_mu) / nm.temp_sd
            
            # wind speed            
            self.site_data[SID].loc[:, ['WS']] = (self.site_data[SID].loc[:, ['WS']] - nm.ws_mu) / nm.ws_sd

        # resample site data to daily and note number of data points present
        self.site_points_present = {}
        self.daily_site_data = {}
        for SID in list(self.site_data.keys()):
            self.site_points_present[SID] = self.site_data[SID].groupby(pd.Grouper(freq='D')).count()
            self.daily_site_data[SID] = self.site_data[SID].resample('1D').mean()

        # load list of binary batches
        self.bin_batches = {}
        self.bin_batches['train'] = []
        for year in dp.train_years:
            self.bin_batches['train'] += glob.glob(f'{binary_batch_path}/batches/*_{year}*.pkl')
        self.bin_batches['val'] = []
        for year in dp.val_years:
            self.bin_batches['val'] += glob.glob(f'{binary_batch_path}/batches/*_{year}*.pkl')
        self.bin_batches['test'] = []
        for year in dp.heldout_years:
            self.bin_batches['test'] += glob.glob(f'{binary_batch_path}/batches/*_{year}*.pkl')        
        
        # indices in a dim_h x dim_h grid
        X1 = np.where(np.ones((self.dim_h, self.dim_h)))
        self.X1 = np.hstack([X1[0][...,np.newaxis],
                             X1[1][...,np.newaxis]])# / (dg.dim_h-1)
        
        
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
        self.parent_pixels[timestep] = self.parent_pixels[timestep].assign_coords(
            {'time':self.era5_dat.time.values}
        )
        ys = self.parent_pixels[timestep].y.shape[0] - 1
        xs = self.parent_pixels[timestep].x.shape[0] - 1
        ts = self.parent_pixels[timestep].time.shape[0]
        for var in era5_vars:
            self.parent_pixels[timestep][var] = (['time', 'y', 'x'],
                                      np.ones((ts, ys, xs), dtype=np.float32)*np.nan)
            source = self.era5_dat[var].values
            self.parent_pixels[timestep][var] = (('time', 'y', 'x'), 
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
        
        return self.parent_pixels[timestep]

    def sample_xyt(self, batch_type='train', timestep='daily'):
        # choose a station and find its 1km pixel
        if batch_type=='train' or batch_type=='val':
            SID = np.random.choice(self.train_sites)
        elif batch_type=='test':
            SID = np.random.choice(self.train_sites + self.heldout_sites) 
        targ_site = self.site_metadata[self.site_metadata['SITE_ID']==SID]
        targ_loc = self.site_metadata[self.site_metadata['SITE_ID']==SID][['LATITUDE','LONGITUDE']]
        targ_yx = np.where(self.parent_pixels[timestep].pixel_id.values == targ_site['parent_pixel_id'].values)

        # grab a random dim_l x dim_l tile that contains that 1km pixel        
        ix = np.random.randint(max(0, targ_yx[1][0] - self.dim_l + 1),
                               min(self.parent_pixels[timestep].x.shape[0] - self.dim_l + 1, targ_yx[1][0] + 1))
        iy = np.random.randint(max(0, targ_yx[0][0] - self.dim_l + 1),
                               min(self.parent_pixels[timestep].y.shape[0] - self.dim_l + 1, targ_yx[0][0] + 1))
        it = np.random.randint(0, 24)
        return ix, iy, it

    def get_input_data(self, var, ix=None, iy=None, it=0,
                       return_intermediates=False, timestep='daily'):        
        if (ix is None) and (iy is None):
            # for rectangular grid over whole space
            if timestep=='daily':
                subdat = self.parent_pixels[timestep]
            else:
                subdat = self.parent_pixels[timestep].isel(time=it)
        else:
            if timestep=='daily':
                subdat = self.parent_pixels[timestep].isel(y=range(iy, iy+self.dim_l),
                                                           x=range(ix, ix+self.dim_l))
            else:
                subdat = self.parent_pixels[timestep].isel(time=it,
                                                           y=range(iy, iy+self.dim_l),
                                                           x=range(ix, ix+self.dim_l))
        
        if timestep=='daily':
            timestamp = self.td
        else:
            timestamp = pd.to_datetime(
                self.parent_pixels[timestep].time.values[it], utc=True
            )        
        
        if timestep!='daily':
            self.sp = SolarPosition(timestamp, timezone=0) # utc
            self.sp.calc_solar_angles(self.chess_grid.lat, self.chess_grid.lon)
            if var=='SWIN':
                self.solar_illum_map = self.calculate_illumination_map(self.sp)
                self.shade_map = self.chess_grid.lat.copy()
                self.shade_map.values.fill(1)
                self.shade_map.values[self.chess_grid.landfrac.values==1] = -(self.shading_array[it,:] - 1)
        
        # subset hi-res fields to the chosen coarse tile
        x_inds = np.intersect1d(np.where(self.chess_grid.x.values < int(subdat.x.max().values) + self.res//2),
                                np.where(self.chess_grid.x.values > int(subdat.x.min().values) - self.res//2))
        y_inds = np.intersect1d(np.where(self.chess_grid.y.values < int(subdat.y.max().values) + self.res//2),
                                np.where(self.chess_grid.y.values > int(subdat.y.min().values) - self.res//2))
        sub_chess = self.chess_grid.isel(y=y_inds, x=x_inds)
        sub_topog = self.height_grid.isel(y=y_inds, x=x_inds)
        
        if var=='RH' or var=='LWIN' or var=='SWIN':
            if timestep=='daily':
                sub_met_dat = self.chess_dat.isel(y=y_inds, x=x_inds)
            else:
                sub_met_dat = self.met_fine.isel(y=y_inds, x=x_inds, time=it)
                
        if var=='SWIN' or var=='LWIN':
            if timestep=='daily':
                sub_sol_azimuth = self.sol_azi.isel(y=y_inds, x=x_inds)
                sub_sol_altitude = self.sol_alt.isel(y=y_inds, x=x_inds)
                sub_sun_hours = self.sun_hrs.isel(y=y_inds, x=x_inds)                
            else:
                sub_sol_azimuth = self.sp.solar_azimuth_angle.isel(y=y_inds, x=x_inds)
                sub_sol_altitude = self.sp.solar_elevation.isel(y=y_inds, x=x_inds)
                sub_sun_hours = sub_sol_altitude.copy()
                sub_sun_hours.values = (sub_sun_hours.values > 0).astype(np.int32)                
                # smooth edges to account for time resolution inaccuracy
                sub_sun_hours.values = gaussian_filter(sub_sun_hours.values, sigma=15)                

        if var=='SWIN':
            if timestep=='daily':                
                sub_illum = self.sol_ilum.isel(y=y_inds, x=x_inds)
                sub_shade = self.av_shade_map.isel(y=y_inds, x=x_inds)
            else:
                sub_illum = self.solar_illum_map.isel(y=y_inds, x=x_inds)
                sub_shade = self.shade_map.isel(y=y_inds, x=x_inds)

        # normalise hi res data
        sub_topog['aspect'].values = sub_topog['aspect'].values / 360. - 0.5 # so goes between -0.5 and 0.5
        for vv in self.elev_vars:
            if vv=='aspect': continue
            sub_topog[vv].values = (sub_topog[vv].values - nm.s_means.loc[vv]) / nm.s_stds.loc[vv]
        
        lat_grid = sub_chess.lat.values / nm.lat_norm
        lon_grid = sub_chess.lon.values / nm.lon_norm

        # create tensors with batch index and channel dim last
        coarse_input = torch.stack([torch.from_numpy(
            subdat[self.var_name_map.coarse.loc[var]].values).to(torch.float32)], dim=-1)
        # therefore coarse scale variable order (in cosmos parlance) is
        self.coarse_variable_order = [self.var_name_map['fine'].loc[var]]

        # get static inputs at the high resolution
        fine_input = torch.from_numpy(sub_chess['landfrac'].values).to(torch.float32)[...,None]
        use_elev_vars = list(set(self.elev_vars) & set(self.targ_var_depends[var]))
        if len(use_elev_vars)>0:
            fine_input = torch.cat(
                [fine_input,
                torch.stack([torch.from_numpy(sub_topog[vv].values).to(torch.float32) 
                                        for vv in use_elev_vars
                            ], dim=-1),
                ], dim=-1
            )
        
        # initial fine scale variable order is
        self.fine_variable_order = ['landfrac'] + use_elev_vars
        
        if 'illumination_map' in self.targ_var_depends[var]:
            # add illumination and terrain shading maps
            fine_input = torch.cat([fine_input,
                torch.from_numpy(sub_illum.values).to(torch.float32)[...,None],
                torch.from_numpy(sub_shade.values).to(torch.float32)[...,None],
                ], dim = -1)
            self.fine_variable_order += ['illumination_map', 'shade_map']
        
        if 'solar_azimuth' in self.targ_var_depends[var]:
            # add azimuthal angle          
            sub_sol_azimuth = np.deg2rad(sub_sol_azimuth.values)  
            fine_input = torch.cat([fine_input,
                torch.from_numpy(sub_sol_azimuth).to(torch.float32)[...,None]
                ], dim = -1)
            self.fine_variable_order += ['solar_azimuth']
        
        if 'solar_altitude' in self.targ_var_depends[var]:
            # add solar elevation angle
            sub_sol_altitude = np.deg2rad(sub_sol_altitude.values)
            fine_input = torch.cat([fine_input,                
                torch.from_numpy(sub_sol_altitude).to(torch.float32)[...,None]
                ], dim = -1)
            self.fine_variable_order += ['solar_altitude']
        
        if 'sunlight_hours' in self.targ_var_depends[var]:
            # turn into fraction of time period in sunlight
            if timestep=='daily':
                sub_sun_hours = sub_sun_hours.values / 24.
            else:
                sub_sun_hours = sub_sun_hours.values
            fine_input = torch.cat([fine_input,                
                torch.from_numpy(sub_sun_hours).to(torch.float32)[...,None]
                ], dim = -1)
            self.fine_variable_order += ['frac_time_sunlight']
            
        if 'land_cover' in self.targ_var_depends[var]:
            # add land cover classes (y axis inverted compared to other grids)
            sub_lcm = self.land_cover[:, self.land_cover.shape[1]-1-y_inds, :][:, :, x_inds]
            fine_input = torch.cat([fine_input,                            
                torch.from_numpy(np.transpose(sub_lcm, (1,2,0))).to(torch.float32)
                ], dim = -1)
            self.fine_variable_order += self.lcm_names
        
        if 'TA' in self.targ_var_depends[var]:            
            if timestep=='daily': vname = self.var_name_map.loc['TA'].chess
            else: vname = self.var_name_map.loc['TA'].coarse
            fine_input = torch.cat([
                fine_input,
                torch.from_numpy(sub_met_dat[vname].values).to(torch.float32)[...,None]
                ], dim = -1)
            self.fine_variable_order += ['TA']
        
        if 'PA' in self.targ_var_depends[var]:
            if timestep=='daily': vname = self.var_name_map.loc['PA'].chess
            else: vname = self.var_name_map.loc['PA'].coarse
            fine_input = torch.cat([
                fine_input,
                torch.from_numpy(sub_met_dat[vname].values).to(torch.float32)[...,None]
                ], dim = -1)
            self.fine_variable_order += ['PA']
        
        if 'RH' in self.targ_var_depends[var]:            
            if timestep=='daily': vname = self.var_name_map.loc['RH'].chess
            else: vname = self.var_name_map.loc['RH'].coarse
            fine_input = torch.cat([
                fine_input,
                torch.from_numpy(sub_met_dat[vname].values).to(torch.float32)[...,None]
                ], dim = -1)
            self.fine_variable_order += ['RH']
            
        # add lat/lon (always last two finescale inputs
        fine_input = torch.cat([fine_input,                            
                            torch.from_numpy(lat_grid).to(torch.float32)[...,None],
                            torch.from_numpy(lon_grid).to(torch.float32)[...,None]
                            ], dim = -1)
        self.fine_variable_order += ['lat', 'lon']

        if return_intermediates:
            return (coarse_input, fine_input, subdat, x_inds, y_inds, 
                    lat_grid, lon_grid, timestamp)
        else:
            return coarse_input, fine_input, lat_grid, lon_grid

    def get_station_targets(self, subdat, x_inds, y_inds, timestamp, 
                            var, latlon_grid, batch_type='train',
                            trim_edge_sites=True, context_frac=None,
                            timestep='daily'):
        # get station targets within the dim_l x dim_l tile,
        # ignoring stations on edge pixels       
        if trim_edge_sites:
            parents = subdat.pixel_id.values[1:-1,1:-1].flatten()
        else:
            parents = subdat.pixel_id.values.flatten()
        if batch_type=='train' or batch_type=='val':
            contained_sites = self.site_metadata[
                (self.site_metadata['SITE_ID'].isin(self.train_sites)) & 
                (self.site_metadata['parent_pixel_id'].isin(parents))
            ]
        elif batch_type=='test':
            contained_sites = self.site_metadata[
                (self.site_metadata['SITE_ID'].isin(self.train_sites + self.heldout_sites)) &
                (self.site_metadata['parent_pixel_id'].isin(parents))
            ]
        
        # find location of contained sites in local subset and pull out data
        contained_sites = contained_sites.set_index('SITE_ID')
        sites_npts = contained_sites[['SITE_NAME', 'LATITUDE', 'LONGITUDE',
                                      'chess_y', 'chess_x', 'parent_pixel_id']].copy()
        contained_sites['sub_x'] = -1
        contained_sites['sub_y'] = -1        
        for vv in self.var_name_map.fine:            
            contained_sites[vv] = -1
            sites_npts[vv] = 0
        for sid in contained_sites.index:            
            this_x = np.where(x_inds == contained_sites.loc[sid,'chess_x'])[0]
            this_y = np.where(y_inds == contained_sites.loc[sid,'chess_y'])[0]
            contained_sites.loc[sid,'sub_x'] = int(this_x)
            contained_sites.loc[sid,'sub_y'] = int(this_y)
            try:
                if timestep=='daily':
                    this_dat = self.daily_site_data[sid].loc[timestamp, :]
                else:
                    this_dat = self.site_data[sid].loc[timestamp, :]
            except:
                this_dat = pd.Series(np.nan, self.var_name_map.fine)
            try:
                numpoints = self.site_points_present[sid].loc[self.td, :]
            except:
                numpoints = pd.Series(0, self.var_name_map.fine)  
            for vv in self.var_name_map.fine:                
                contained_sites.loc[sid, vv] = this_dat[vv]
                sites_npts.loc[sid, vv] = numpoints[vv]
        
        # define station targets
        keep_vars = ['sub_x', 'sub_y'] + [var] + \
            ['LATITUDE', 'LONGITUDE'] #list(self.var_name_map.fine)
        if 'elev' in self.fine_variable_order:
                keep_vars.append('ALTITUDE')
        station_targets = contained_sites[keep_vars]
        station_npts = sites_npts[[var]]
        
        # trim sites with no data
        keepsites = station_targets[[var]].dropna(how='all').index
        station_targets = station_targets.loc[keepsites]
        station_npts = station_npts.loc[keepsites]
        
        if timestep=='daily':
            # trim sites with fewer than 24 readings for daily data
            keepsites2 = station_npts[station_npts[var]==24].index
            station_targets = station_targets.loc[keepsites2]
            station_npts = station_npts.loc[keepsites2]
            
            if len(keepsites2)==0:
                # no sites with 24 readings, return null batch
                if 'elev' in self.fine_variable_order:
                    val_dense_vecs = np.zeros((1, 5, 0), dtype=np.float32) # no null tag
                else:
                    val_dense_vecs = np.zeros((1, 4, 0), dtype=np.float32) # no null tag                
                YX_locs = np.zeros((0,2), dtype=np.float32)
                return ({'context':station_targets,
                         'target':station_targets},
                         {'context':station_npts,
                         'target':station_npts},
                         val_dense_vecs, YX_locs)
                
        # create context/target splits (randomising context fraction)
        if context_frac is None:
            context_frac = np.random.uniform(0, 0.9)
        context, targets = context_target_split(
            station_targets.index,
            context_frac=context_frac,
            random_state=np.random.randint(1000)
        )
        
        # create value/density pairs (+ altitude) vector for context points (+1 for null tag)
        if 'elev' in self.fine_variable_order:
            val_dense_vecs = np.zeros((1, 5, len(context)), dtype=np.float32) # no null tag
        else:
            val_dense_vecs = np.zeros((1, 4, len(context)), dtype=np.float32) # no null tag
        for i, sid in enumerate(context):
            vardat = station_targets.loc[sid]
            thisval = vardat[var]
            if not np.isnan(thisval):
                val_dense_vecs[0,0,i] = thisval
                val_dense_vecs[0,1,i] = 1.
                jj = 2
                if 'elev' in self.fine_variable_order:
                    val_dense_vecs[0,jj,i] = (vardat.ALTITUDE - nm.s_means.elev) / nm.s_stds.elev
                    jj += 1
                    # might be worth adding aspect/slope from the gridded elevation model
                    # to site data where appropriate?
                val_dense_vecs[0,jj  ,i] = vardat.LATITUDE / nm.lat_norm
                val_dense_vecs[0,jj+1,i] = vardat.LONGITUDE / nm.lon_norm
        
        # create location array to append (lat/lon)
        #X0 = station_targets.loc[context][['LATITUDE', 'LONGITUDE']]
        #X0 /= np.array([[nm.lat_norm, nm.lon_norm]]) # normalise
        #X0 = np.vstack([X0, np.array([[-99., -99.]])]) # add null tag
        
        # create location array to append (y, x)
        # this doesn't account for the site being offset from the
        # centre of the grid cell, though.
        #X0 = station_targets.loc[context][['sub_y', 'sub_x']]
        #X0 = np.array(X0) / (self.dim_h-1)
        #X0 = np.vstack([X0, np.array([[-1, -1]])]) # add null tag
        ''' since we are doing optional attention (i.e. checking
        whether we have sites as context) we likely don't need the 
        null tag anymore
        '''
       
        if len(context)>0:
            ## find off-grid YX locations of stations 
            ## (as floating point indices for positional encoding)
            lats  = station_targets.loc[context].LATITUDE / nm.lat_norm
            lons  = station_targets.loc[context].LONGITUDE / nm.lon_norm
            lls = pd.concat([lats, lons], axis=1).values
            latlon_grid = np.reshape(latlon_grid, (2, len(x_inds)*len(y_inds))).T #self.dim_h*self.dim_h
            
            neigh = NearestNeighbors(n_neighbors=4)
            neigh.fit(latlon_grid)
            dists, inds = neigh.kneighbors(X=lls)
            
            weights = 1 - (dists / dists.sum(axis=1, keepdims=True))
            
            if len(x_inds)==self.dim_h and len(y_inds)==self.dim_h:
                X1 = self.X1
            else:
                X1 = np.where(np.ones((len(y_inds), len(x_inds))))
                X1 = np.hstack([X1[0][...,np.newaxis],
                                X1[1][...,np.newaxis]])
            YX_locs = np.stack([(X1[inds][s] * weights[s][...,None]).sum(axis=0) / weights[s].sum() for s in range(weights.shape[0])])
        else:
            YX_locs = np.zeros((0,2), dtype=np.float32)
        
        if False:
            # if we want to scale or mask the attention by distance:
            nbrs = NearestNeighbors(n_neighbors = X0.shape[0], algorithm='ball_tree').fit(X0)
            distances, indices = nbrs.kneighbors(self.X1)
            # these are ordered by nearest rather than in order of X0, so 
            # reorder by indices vectors
            distances = np.take_along_axis(distances, np.argsort(indices, axis=1), axis=1)
        
        # return station data/npts, value/density array and locations
        return ({'context':station_targets.loc[context],
                 'target':station_targets.loc[targets]},
                 {'context':station_npts.loc[context],
                 'target':station_npts.loc[targets]},
                 val_dense_vecs, YX_locs)

    def get_constraints(self, x_inds, y_inds, it, var):
        if var=='TA':
            constraint = self.t_1km_elev.isel(time=it, y=y_inds, x=x_inds)
            constraint = (constraint - 273.15 - nm.temp_mu) / nm.temp_sd            
        if var=='PA':
            constraint = self.p_1km_elev.isel(time=it, y=y_inds, x=x_inds)
            constraint = (constraint - nm.p_mu) / nm.p_sd
        if var=='SWIN':
            ## Shortwave radiation, time of day dependent           
            self.Sw_1km = partition_interp_solar_radiation(
                #self.swin_1km_interp[it,:,:], # this is unnormalised by design
                self.parent_pixels['hourly'][self.var_name_map.loc['SWIN'].coarse][it,:,:] * nm.swin_norm, # de-nornmalise!
                self.chess_grid,
                self.sp,
                self.parent_pixels['hourly'][self.var_name_map.loc['PA'].coarse][it,:,:] * nm.p_sd + nm.p_mu, # de-nornmalise!
                self.p_1km_elev[it,:,:],
                self.height_grid,
                self.shading_array[it,:],
                self.scale                
            )            
            constraint = self.Sw_1km.isel(y=y_inds, x=x_inds)
            constraint = constraint / nm.swin_norm
        if var=='LWIN':
            constraint = self.Lw_1km.isel(time=it, y=y_inds, x=x_inds)
            constraint = (constraint - nm.lwin_mu) / nm.lwin_sd
        if var=='WS':
            constraint = self.ws_1km_interp.isel(time=it, y=y_inds, x=x_inds)
            constraint = (constraint - nm.ws_mu) / nm.ws_sd
        if var=='RH':
            constraint = self.rh_1km_interp.isel(time=it, y=y_inds, x=x_inds)
            constraint = (constraint - nm.rh_mu) / nm.rh_sd
        return constraint.values

    def get_sample(self, var, batch_type='train', context_frac=None, 
                   timestep='daily', sample_xyt=True, ix=None, iy=None,
                   it=0, return_constraints=True):

        if sample_xyt:
            ## sample a dim_l x dim_l tile
            ix, iy, it = self.sample_xyt(batch_type=batch_type, timestep=timestep)
        
        ## load input data
        (coarse_input, fine_input, subdat,
            x_inds, y_inds, lat_grid, lon_grid, timestamp) = self.get_input_data(
            var, ix, iy, it, return_intermediates=True, timestep=timestep
        )
        
        ## get station targets within the dim_l x dim_l tile        
        (station_targets, station_npts, 
            station_data, context_locations) = self.get_station_targets(
            subdat, x_inds, y_inds, timestamp, var,
            np.stack([lat_grid, lon_grid], axis=0),
            batch_type=batch_type,
            context_frac=context_frac,            
            timestep=timestep            
        )
        
        constraints = None
        if return_constraints:
            if timestep=='daily':
                ## grab subgrid chess constraints        
                constraints = self.chess_dat.isel(x=x_inds, y=y_inds)[self.chess_var].values            
            else:
                ## use physically reasoned constraints
                constraints = self.get_constraints(x_inds, y_inds, it, var)
            constraints = torch.from_numpy(constraints.astype(np.float32))[...,None]
        
        ## capture sample description
        sample_meta = {
            'timestamp':self.td,
            'x_inds':x_inds,
            'y_inds':y_inds,
            't_ind':it,
            'coarse_xl':ix,
            'coarse_yt':iy,
            'timestep':timestep,
            'fine_var_order':self.fine_variable_order            
        }
                
        return {'in_coarse':coarse_input,
                'in_fine':fine_input,
                'station_data':station_data,
                'station_metadata':station_targets,
                'constraints':constraints,
                'context_locs':context_locations,
                'station_num_obs':station_npts,
                'sample_meta':sample_meta
                }

    def prepare_constraints(self, var):
        '''
        This needs to be run before ERA5 data is re-normalised!
        '''
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # ignore warnings thrown by interp_like() which seem to be needless
            tn = self.var_name_map.loc['TA'].coarse
            if var=='TA' or var=='PA' or var=='SWIN' or var=='LWIN':
                ## Air Temp
                # reduce to sea level with lapse rate of -0.0065 K/m
                lapse_val = -0.0065
                self.parent_pixels['hourly']['t_sealevel'] = (
                    ['time', 'y', 'x'],
                    self.parent_pixels['hourly'][tn].values
                )
                self.parent_pixels['hourly']['t_sealevel'] = (
                    self.parent_pixels['hourly'].t_sealevel - 
                    self.parent_pixels['hourly'].elev * lapse_val
                )

                # bicubic spline to interpolate from 25km to 1km... cubic doesn't work here!
                t_sealevel_interp = self.parent_pixels['hourly']['t_sealevel'].interp_like(
                    self.chess_grid, method='linear')
                t_sealevel_interp = t_sealevel_interp.transpose('time', 'y', 'x')

                # adjust to the 1km elevation using same lapse rate
                self.t_1km_elev = t_sealevel_interp + self.height_grid['elev'] * lapse_val
                self.t_1km_elev = self.t_1km_elev.transpose('time', 'y', 'x')
                
                # rescale to counteract errors in physical downscaling assumptions
                self.t_1km_elev = scale_by_coarse_data(
                    self.t_1km_elev.copy(),
                    self.parent_pixels['hourly'][tn],
                    self.chess_grid,
                    self.scale
                )
            
            if var=='PA' or var=='SWIN':
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
                
                # rescale to counteract errors in physical downscaling assumptions
                self.p_1km_elev = scale_by_coarse_data(
                    self.p_1km_elev.copy(),
                    self.parent_pixels['hourly'][self.var_name_map.loc['PA'].coarse]/100., # Pa -> hPa
                    self.chess_grid,
                    self.scale
                )
            
            if var=='RH' or var=='LWIN':
                ## Relative Humidity
                # Assumed constant with respect to elevation, so can be simply 
                # interpolated from 25km to 1km using a bicubic spline.
                self.rh_1km_interp = self.parent_pixels['hourly'][
                    self.var_name_map.loc['RH'].coarse].interp_like(
                        self.chess_grid, method='linear')
                self.rh_1km_interp = self.rh_1km_interp.transpose('time', 'y', 'x')
            
                # rescale to counteract errors in physical downscaling assumptions
                self.rh_1km_interp = scale_by_coarse_data(
                    self.rh_1km_interp.copy(),
                    self.parent_pixels['hourly'][self.var_name_map.loc['RH'].coarse],
                    self.chess_grid,
                    self.scale
                )
            
            if var=='SWIN':
                ## Shortwave radiation. Simply interpolate this before normalisation
                ## for use in partitioning direct/diffuse
                #self.swin_1km_interp = self.parent_pixels['hourly']['msdwswrf'].interp_like(
                #    self.chess_grid, method='linear')
                #self.swin_1km_interp = self.swin_1km_interp.transpose('time', 'y', 'x')
                pass
                
            if var=='LWIN':
                ## Saturated and Actual vapour pressure
                # using Buck equation                
                es_25km = 6.1121 * np.exp(
                    (18.678 - (self.parent_pixels['hourly'][tn] - 273.15) / 234.5) *
                    ((self.parent_pixels['hourly'][tn] - 273.15) / 
                    (257.14 + (self.parent_pixels['hourly'][tn] - 273.15)))
                )
                es_1km = 6.1121 * np.exp(
                    (18.678 - (self.t_1km_elev - 273.15) / 234.5) *
                    ((self.t_1km_elev - 273.15) / (257.14 + (self.t_1km_elev - 273.15)))
                )
                # from RH
                ea_25km = (self.parent_pixels['hourly'][
                    self.var_name_map.loc['RH'].coarse] / 100) * es_25km
                ea_1km = (self.rh_1km_interp/100) * es_1km                
                
                ## Longwave radiation and Emissivity
                # method from Real-time and retrospective forcing in the North American Land
                #   Data Assimilation System (NLDAS) project, Brian A. Cosgrove,
                # which borrows model from
                # Satterlund, Donald R., An improved equation for estimating long‐wave radiation from the atmosphere
                # and I have taken calibrated constants from  M. Li, Y. Jiang, C.F.M. Coimbra
                #   On the determination of atmospheric longwave irradiance under all-sky conditions
                #   Sol. Energy, 144 (2017), pp. 40-48
                emms_25km = 1.02 * (1 - np.exp(-ea_25km*(
                    self.parent_pixels['hourly']['t2m']/1564.94)))
                emms_25km_interp = emms_25km.interp_like(self.chess_grid, method='linear')  
                emms_1km = 1.02 * (1 - np.exp(-ea_1km*(self.t_1km_elev/1564.94)))
                emms_ratio = emms_1km / emms_25km_interp
                
                Lw_25km_interp = self.parent_pixels['hourly'][
                    self.var_name_map.loc['LWIN'].coarse].interp_like(
                        self.chess_grid, method='linear')  
                self.Lw_1km = emms_ratio * Lw_25km_interp
                self.Lw_1km = self.Lw_1km.transpose('time', 'y', 'x')
                # there is "supposed" to be a ratio of temperatures between downscaled/nondownscaled
                # of form (T_dwns / T_raw)^4 but this might enforce an elevation dependence 
                # on the longwave radiation which we might not want?
                
                # rescale to counteract errors in physical downscaling assumptions
                self.Lw_1km = scale_by_coarse_data(
                    self.Lw_1km.copy(),
                    self.parent_pixels['hourly'][self.var_name_map.loc['LWIN'].coarse],
                    self.chess_grid,
                    self.scale
                )
                            
            if var=='WS':
                ## Wind speed. Simply interpolate this....            
                self.ws_1km_interp = self.parent_pixels['hourly']['ws'].interp_like(
                    self.chess_grid, method='linear')                
                
                # add a weighted version of the chess wind for spatial structure
                chess_day_wind = self.chess_dat[self.var_name_map.loc['WS'].chess]
                self.ws_1km_interp = (chess_day_wind + 2 * self.ws_1km_interp)/3.
                self.ws_1km_interp = self.ws_1km_interp.transpose('time', 'y', 'x')
                # rescale to counteract errors in physical downscaling assumptions
                self.ws_1km_interp = scale_by_coarse_data(
                    self.ws_1km_interp.copy(),
                    self.parent_pixels['hourly'][self.var_name_map.loc['WS'].coarse],
                    self.chess_grid,
                    self.scale
                )

    def prepare_era5_pixels(self, var, batch_tsteps='daily', constraints=True):
        if batch_tsteps=='daily':
            ## average over the day
            self.parent_pixels['daily'] = self.parent_pixels['hourly'].mean('time')
            self.parent_pixels['hourly'] = None
            tstep = 'daily'
        elif batch_tsteps=='hourly' or batch_tsteps=='mix':
            tstep = 'hourly'
            
        ## convert wind vectors and dewpoint temp to wind speed and relative humidity        
        self.parent_pixels[tstep]['rh'] = relhum_from_dewpoint(
            self.parent_pixels[tstep]['t2m'] - 273.15,
            self.parent_pixels[tstep]['d2m'] - 273.15
        )
        self.parent_pixels[tstep]['ws'] = (np.sqrt(
            np.square(self.parent_pixels[tstep]['v10']) +
            np.square(self.parent_pixels[tstep]['u10']))
        )        
        
        if batch_tsteps!='daily':
            if constraints:
                # this requires particular units - do before changing them!
                self.prepare_constraints(var)
        if var=='SWIN':            
            # terrain shading used in hourly AND daily
            doy = self.td.day_of_year - 1 # zero indexed
            self.shading_array = np.load(
                binary_batch_path + f'/terrain_shading/shading_mask_day_{doy}.npy')
        
        ## change units
        self.parent_pixels[tstep]['t2m'].values -= 273.15 # K -> Celsius 
        self.parent_pixels[tstep]['d2m'].values -= 273.15 # K -> Celsius
        self.parent_pixels[tstep]['sp'].values /= 100.    # Pa -> hPa
        
        self.parent_pixels[tstep] = self.parent_pixels[tstep].drop(['u10', 'v10', 'd2m'])
        
        ## normalise ERA5        
        self.parent_pixels[tstep]['t2m'].values = (self.parent_pixels[tstep]['t2m'].values - nm.temp_mu) / nm.temp_sd        
        self.parent_pixels[tstep]['sp'].values = (self.parent_pixels[tstep]['sp'].values - nm.p_mu) / nm.p_sd        
        self.parent_pixels[tstep]['msdwlwrf'].values = (self.parent_pixels[tstep]['msdwlwrf'].values - nm.lwin_mu) / nm.lwin_sd        
        #self.parent_pixels[tstep]['msdwswrf'].values = np.log(1. + self.parent_pixels[tstep]['msdwswrf'].values) # log(1 + swin)
        #self.parent_pixels[tstep]['msdwswrf'].values = (self.parent_pixels[tstep]['msdwswrf'].values - nm.logswin_mu) / nm.logswin_sd
        self.parent_pixels[tstep]['msdwswrf'].values = self.parent_pixels[tstep]['msdwswrf'].values / nm.swin_norm
        self.parent_pixels[tstep]['ws'].values = (self.parent_pixels[tstep]['ws'].values - nm.ws_mu) / nm.ws_sd        
        self.parent_pixels[tstep]['rh'].values = (self.parent_pixels[tstep]['rh'].values - nm.rh_mu) / nm.rh_sd
        
        ## drop variables we don't need
        self.era5_var = self.var_name_map.loc[var].coarse
        to_drop = list(self.parent_pixels[tstep].keys())
        to_drop.remove(self.era5_var)
        to_drop.remove('pixel_id')
        if var=='RH':
            for exvar in ['TA', 'PA']:
                to_drop.remove(self.var_name_map.loc[exvar].coarse)
        if var=='LWIN':
            for exvar in ['RH']:
                to_drop.remove(self.var_name_map.loc[exvar].coarse)
        if var=='SWIN':
            for exvar in ['PA']:
                to_drop.remove(self.var_name_map.loc[exvar].coarse)
        self.parent_pixels[tstep] = self.parent_pixels[tstep].drop(to_drop)
        
        if batch_tsteps=='mix':
            # do time-averaging for daily AFTER processing if mixed batch            
            self.parent_pixels['daily'] = self.parent_pixels['hourly'].mean('time')

    def prepare_fine_met_inputs(self, var, batch_tsteps):
        if var=='RH':
            extra_met_vars = ['TA', 'PA']
        if var=='LWIN':
            extra_met_vars = ['RH']
        if var=='SWIN':
            extra_met_vars = ['PA']
        if var in ['RH', 'LWIN', 'SWIN']:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if batch_tsteps!='hourly': # cover daily and mix
                    ## interpolate ERA5 extra vars to 1km and replace
                    ## NaN sea pixels of chess vars with ERA5 to use as fine input
                    era5_met = (self.parent_pixels['daily'].copy()
                        .drop(['pixel_id', self.var_name_map.loc[var].coarse]))
                    era5_met = era5_met.interp_like(self.chess_grid, method='linear')                    
                    era5_met = (era5_met
                        .interpolate_na(dim='x', method='nearest', fill_value='extrapolate')
                        .interpolate_na(dim='y', method='nearest', fill_value='extrapolate')
                    )
                    chess_sea_mask = np.isnan(self.chess_dat[self.chess_var].values)
                    for xtrvar in extra_met_vars:
                        self.chess_dat[self.var_name_map.loc[xtrvar].chess].values[chess_sea_mask] =\
                            era5_met[self.var_name_map.loc[xtrvar].coarse].values[chess_sea_mask]
                if batch_tsteps!='daily': # covers hourly and mix
                    ## just use interpolated ERA5 extra vars as input...
                    self.met_fine = (self.parent_pixels['hourly'].copy()
                        .drop(['pixel_id', self.var_name_map.loc[var].coarse]))
                    self.met_fine = self.met_fine.interp_like(self.chess_grid, method='linear')                    
                    self.met_fine = (self.met_fine
                        .interpolate_na(dim='x', method='nearest', fill_value='extrapolate')
                        .interpolate_na(dim='y', method='nearest', fill_value='extrapolate')
                    )
                ## then as fine input we use self.met_fine for hourly samples
                ## and self.chess_dat for daily samples
        else:
            pass
    
    def calculate_illumination_map(self, sp):
        mask = sp.solar_elevation >= 0
        solar_illum = self.chess_grid.landfrac.copy()
        solar_illum.values[:,:] = 0
        solar_illum.values[mask] = (
            np.cos(np.deg2rad(sp.solar_zenith_angle.values[mask])) * 
            np.cos(np.deg2rad(self.height_grid.slope.values[mask])) + 
            np.sin(np.deg2rad(sp.solar_zenith_angle.values[mask])) * 
            np.sin(np.deg2rad(self.height_grid.slope.values[mask])) *
            np.cos(np.deg2rad(sp.solar_azimuth_angle.values[mask] - self.height_grid.aspect.values[mask]))
        )
        return solar_illum
    
    def calculate_day_av_solar_vars(self, var, batch_tsteps):
        if (('sunlight_hours' in self.targ_var_depends[var] or 
            'solar_azimuth' in self.targ_var_depends[var] or 
            'solar_altitude' in self.targ_var_depends[var]) and 
            batch_tsteps!='hourly'):
            self.sol_azi = self.chess_grid.lat.copy()
            self.sol_alt = self.chess_grid.lat.copy()
            self.sun_hrs = self.chess_grid.lat.copy()
            self.sol_azi.values.fill(0)
            self.sol_alt.values.fill(0)
            self.sun_hrs.values.fill(0)
            if var=='SWIN':
                self.sol_ilum = self.chess_grid.lat.copy()
                self.av_shade_map = self.chess_grid.lat.copy()
                self.sol_ilum.values.fill(0)
                self.av_shade_map.values.fill(1)
                sun_hours = []
            for tt in range(0,24):
                sp = SolarPosition(self.td + datetime.timedelta(hours=tt), timezone=0) # utc
                sp.calc_solar_angles(self.chess_grid.lat, self.chess_grid.lon)
                dayhour = (sp.solar_elevation.values > 0).astype(np.int32)
                # only include daytime hours in the angle/illum averages
                self.sun_hrs.values += dayhour
                self.sol_azi.values += sp.solar_azimuth_angle.values * dayhour
                self.sol_alt.values += sp.solar_elevation.values * dayhour
                if var=='SWIN':
                    illum_map = self.calculate_illumination_map(sp)
                    self.sol_ilum.values += illum_map.values * dayhour
                    if np.mean((sp.solar_elevation.values > 0).astype(np.int32)) > 0.1:
                        sun_hours.append(tt)
            
            # this introduces artefacts due to the hour timestep
            # creating non-smooth edges between sunlight hours
            self.sol_azi.values /= self.sun_hrs.values
            self.sol_alt.values /= self.sun_hrs.values
            # smooth them!            
            self.sol_azi.values = gaussian_filter(self.sol_azi.values, sigma=15)
            self.sol_alt.values = gaussian_filter(self.sol_alt.values, sigma=15)
            self.sun_hrs.values = gaussian_filter(self.sun_hrs.values, sigma=15)
            if var=='SWIN':
                # we don't want a smoothed illumination map, though...
                self.sol_ilum.values /= self.sun_hrs.values
                self.av_shade_map.values[self.chess_grid.landfrac.values==1] = \
                    -(np.mean(self.shading_array[sun_hours,:], axis=0) - 1)
        else:
            pass
    
    def get_chess_batch(self, var, batch_size=1, batch_type='train',
                        load_binary_batch=True, context_frac=None,
                        p_hourly=0):
        self.parent_pixels = {}
        if load_binary_batch is False:
            self.parent_pixels['hourly'] = self.read_parent_pixel_day(batch_type=batch_type)            
            self.td = pd.to_datetime(self.parent_pixels[timestep].time[0].values, utc=True)
        else:
            got_batch = False
            while got_batch==False:
                try:
                    with Timeout(10):        
                        bfn = self.bin_batches[batch_type][np.random.randint(0, len(self.bin_batches[batch_type]))]
                        self.parent_pixels['hourly'] = pd.read_pickle(bfn)
                        got_batch = True
                except Timeout.Timeout:
                    got_batch = False
            try:
                self.td = pd.to_datetime(bfn.split('_')[-1].replace('.pkl',''), format='%Y%m%d', utc=True)
            except:
                self.td = pd.to_datetime(self.parent_pixels['hourly'].time[0].values, utc=True)
        
        ## work out the timesteps of the batch        
        batch_timesteps = np.random.choice(['daily', 'hourly'], batch_size,
                                           p=[1-p_hourly, p_hourly])
        if ('daily' in batch_timesteps) and ('hourly' in batch_timesteps):
            batch_tsteps = 'mix'
        elif batch_timesteps[0]=='daily':
            batch_tsteps = 'daily'
        else:
            batch_tsteps = 'hourly'

        if batch_tsteps!='hourly' or var=='WS':
            ## load chess data (don't normalise in case using in constraint preparation)
            self.chess_var = self.var_name_map.loc[var].chess
            self.chess_dat = load_process_chess(self.td.year, self.td.month, self.td.day,
                                                var=self.chess_var, normalise=False)

        ## process and normalise ERA5 parent_pixels
        self.prepare_era5_pixels(var, batch_tsteps=batch_tsteps)

        if batch_tsteps!='hourly' or var=='WS':
            ## now normalise if we have loaded chess data
            self.chess_dat = normalise_chess_data(self.chess_dat.copy(),
                                                  self.chess_var)

        ## get additional fine scale met vars for input
        self.prepare_fine_met_inputs(var, batch_tsteps)
        
        if batch_tsteps!='hourly': # cover daily and mix
            ## pool to scale and fill in the parent pixels
            chess_pooled = pooling(self.chess_dat[self.chess_var].values,
                                   (self.scale, self.scale),
                                   method='mean')
            chess_mask = ~np.isnan(chess_pooled)
            self.parent_pixels['daily'][self.era5_var].values[chess_mask] = chess_pooled[chess_mask]   
                        
        # solar values for daily timestep
        self.calculate_day_av_solar_vars(var, batch_tsteps)
        
        coarse_inputs = []
        fine_inputs = []
        station_targets = []
        station_data = []
        station_num_obs = []
        constraint_targets = []        
        context_locations = []
        batch_metadata = []
        for b in range(batch_size):
            # generate batch from parent pixels
            sample = self.get_sample(var, batch_type=batch_type,
                                     context_frac=context_frac,
                                     timestep=batch_timesteps[b])
           
            # append to storage
            coarse_inputs.append(sample['in_coarse'])            
            fine_inputs.append(sample['in_fine'])
            station_targets.append(sample['station_metadata'])
            constraint_targets.append(sample['constraints'])
            station_data.append(sample['station_data'])
            station_num_obs.append(sample['station_num_obs'])
            context_locations.append(sample['context_locs'].T)
            batch_metadata.append(sample['sample_meta'])

        coarse_inputs = torch.stack(coarse_inputs, dim=0)
        fine_inputs = torch.stack(fine_inputs, dim=0)
        constraint_targets = torch.stack(constraint_targets, dim=0)        
        station_data = [torch.from_numpy(a).type(torch.float32) for a in station_data]
        context_locations = [torch.from_numpy(a).type(torch.float32) for a in context_locations]
                
        return {'coarse_inputs':coarse_inputs.permute(0,3,1,2),
                'fine_inputs':fine_inputs.permute(0,3,1,2),
                'constraints':constraint_targets.permute(0,3,1,2),
                'station_targets':station_targets,
                'station_data':station_data,
                'station_num_obs':station_num_obs,
                'context_locations':context_locations,                
                'batch_metadata':batch_metadata
                }    

    def get_all_space(self, var, batch_type='train', load_binary_batch=True,
                      context_frac=None, date_string=None, it=None,
                      timestep='hourly', tile=False, min_overlap=1):
                          
        self.parent_pixels = {}
        if date_string is None:            
            if load_binary_batch is False:
                self.parent_pixels['hourly'] = self.read_parent_pixel_day(batch_type=batch_type)            
                self.td = pd.to_datetime(self.parent_pixels[timestep].time[0].values)
            else:
                got_batch = False
                while got_batch==False:
                    try:
                        with Timeout(10):        
                            bfn = self.bin_batches[batch_type][np.random.randint(0, len(self.bin_batches[batch_type]))]
                            self.parent_pixels['hourly'] = pd.read_pickle(bfn)
                            got_batch = True
                    except Timeout.Timeout:
                        got_batch = False
                try:
                    self.td = pd.to_datetime(bfn.split('_')[-1].replace('.pkl',''), format='%Y%m%d', utc=True)
                except:
                    self.td = pd.to_datetime(self.parent_pixels['hourly'].time[0].values, utc=True)
        else:
            if load_binary_batch is False:
                self.parent_pixels['hourly'] = self.read_parent_pixel_day(batch_type=batch_type)                
            else:
                bfn = f'{binary_batch_path}/batches/era5_bng_25km_pixels_{date_string}.pkl'                
                self.parent_pixels['hourly'] = pd.read_pickle(bfn)                
            self.td = pd.to_datetime(date_string, format='%Y%m%d', utc=True)        
        
        if timestep!='hourly' or var=='WS':
            ## load chess data (don't normalise in case using in constraint preparation)
            self.chess_var = self.var_name_map.loc[var].chess
            self.chess_dat = load_process_chess(self.td.year, self.td.month, self.td.day,
                                                var=self.chess_var, normalise=False)

        ## process and normalise ERA5 parent_pixels
        self.prepare_era5_pixels(var, batch_tsteps=timestep, constraints=False)

        if timestep!='hourly' or var=='WS':
            ## now normalise if we have loaded chess data
            self.chess_dat = normalise_chess_data(self.chess_dat.copy(),
                                                  self.chess_var)

        ## get additional fine scale met vars for input
        self.prepare_fine_met_inputs(var, timestep)
        
        if timestep!='hourly': # cover daily and mix
            ## pool to scale and fill in the parent pixels
            chess_pooled = pooling(self.chess_dat[self.chess_var].values,
                                   (self.scale, self.scale),
                                   method='mean')
            chess_mask = ~np.isnan(chess_pooled)
            self.parent_pixels['daily'][self.era5_var].values[chess_mask] = chess_pooled[chess_mask]   
                        
        # solar values for daily timestep
        self.calculate_day_av_solar_vars(var, timestep)
        
        if timestep=='hourly':
            if it is None:
                # choose a random time slice
                it = np.random.randint(0, 24)
        
        # allow us to iterate over a full day
        if type(it)==int: its = [it]
        else: its = it.copy()
        del(it)
        
        ixs = []
        iys = []
        coarse_inputs = []
        fine_inputs = []
        station_targets = []
        station_data = []
        station_num_obs = []
        constraint_targets = []        
        context_locations = []
        batch_metadata = []
        for it in its:
            if tile==False:
                # take one big (rectangular) tile
                ix = None
                iy = None
                
                sample = self.get_sample(var, batch_type=batch_type,
                                         context_frac=context_frac, 
                                         timestep=timestep,
                                         sample_xyt=False,
                                         ix=ix, iy=iy, it=it,
                                         return_constraints=False)
                
                # append to storage
                coarse_inputs.append(sample['in_coarse'])            
                fine_inputs.append(sample['in_fine'])
                station_targets.append(sample['station_metadata'])
                constraint_targets.append(sample['constraints'])
                station_data.append(sample['station_data'])
                station_num_obs.append(sample['station_num_obs'])
                context_locations.append(sample['context_locs'].T)
                batch_metadata.append(sample['sample_meta'])
            else:
                # divide space up into overlapping dim_l x dim_l tiles
                ixs = define_tile_start_inds(self.parent_pixels[timestep].x.shape[0],
                                             self.dim_l, min_overlap)
                iys = define_tile_start_inds(self.parent_pixels[timestep].y.shape[0],
                                             self.dim_l, min_overlap)
                #yx_tiles = []
                for ix in ixs:
                    for iy in iys:
                        sample = self.get_sample(var, batch_type=batch_type,
                                                 context_frac=context_frac, 
                                                 timestep=timestep,
                                                 sample_xyt=False,
                                                 ix=ix, iy=iy, it=it,
                                                 return_constraints=False)
                                                                     
                        # append to storage
                        coarse_inputs.append(sample['in_coarse'])            
                        fine_inputs.append(sample['in_fine'])
                        station_targets.append(sample['station_metadata'])
                        constraint_targets.append(sample['constraints'])
                        station_data.append(sample['station_data'])
                        station_num_obs.append(sample['station_num_obs'])
                        context_locations.append(sample['context_locs'].T)
                        batch_metadata.append(sample['sample_meta'])
                        #yx_tiles.append((iy, ix))

        coarse_inputs = torch.stack(coarse_inputs, dim=0)
        fine_inputs = torch.stack(fine_inputs, dim=0)        
        station_data = [torch.from_numpy(a).type(torch.float32) for a in station_data]
        context_locations = [torch.from_numpy(a).type(torch.float32) for a in context_locations]
                
        return {'coarse_inputs':coarse_inputs.permute(0,3,1,2),
                'fine_inputs':fine_inputs.permute(0,3,1,2),                
                'station_targets':station_targets,
                'station_data':station_data,
                'station_num_obs':station_num_obs,
                'context_locations':context_locations,                
                'batch_metadata':batch_metadata,
                'ixs':ixs, 'iys':iys
               }


if __name__=="__main__":
    if False:
        pass
        dg = data_generator()
        
        batch_size = 1
        batch_type = 'train'
        load_binary_batch = True
        p_hourly = 1.0
        var = 'PA'
                
        batch = dg.get_chess_batch(var,
                                   batch_size=batch_size,
                                   batch_type=batch_type,
                                   load_binary_batch=load_binary_batch,
                                   p_hourly=p_hourly)
        
        
        fig, ax = plt.subplots(1,2)
        ax[0].imshow(batch['coarse_inputs'][0,0,:,:].numpy())
        ax[1].imshow(batch['constraints'][0,0,:,:].numpy())
        plt.show()
        
        grid_av = pooling(batch['constraints'][0,0,:,:].numpy(), (25,25), method='mean')
        fig, ax = plt.subplots(1,2)
        ax[0].imshow(batch['coarse_inputs'][0,0,:,:].numpy())
        ax[1].imshow(grid_av)
        plt.show()
                
        b = Batch(batch, var_list=var)

        

        # p_rescale = scale_by_coarse_data(dg.p_1km_elev,
                                        # (dg.parent_pixels['hourly'].sp * nm.p_sd) + nm.p_mu,
                                        # dg.chess_grid, 25)
        # p_sub = (p_rescale.isel(y=batch['batch_metadata'][0]['y_inds'],
                               # x=batch['batch_metadata'][0]['x_inds'],
                               # time=batch['batch_metadata'][0]['t_ind']) - nm.p_mu) / nm.p_sd
        # grid_av2 = pooling(p_sub.values, (25,25), method='mean')
       
        # fig, ax = plt.subplots(1,3)
        # ax[0].imshow(batch['coarse_inputs'][0,0,:,:].numpy())
        # ax[1].imshow(batch['constraints'][0,0,:,:].numpy())
        # ax[2].imshow(p_sub.values)
        # plt.show()
        
        # fig, ax = plt.subplots(1,3)
        # ax[0].imshow(batch['coarse_inputs'][0,0,:,:].numpy())
        # ax[1].imshow(grid_av)
        # ax[2].imshow(grid_av2)
        # plt.show()
        
        
        #####
        if False:
            dg.parent_pixels = {}
            if load_binary_batch is False:
                dg.parent_pixels['hourly'] = dg.read_parent_pixel_day(batch_type=batch_type)            
                dg.td = pd.to_datetime(dg.parent_pixels[timestep].time[0].values)
            else:
                bfn = dg.bin_batches[batch_type][np.random.randint(0, len(dg.bin_batches[batch_type]))]            
                dg.parent_pixels['hourly'] = pd.read_pickle(bfn)
                try:
                    dg.td = pd.to_datetime(bfn.split('_')[-1].replace('.pkl',''), utc=True)
                except:
                    dg.td = pd.to_datetime(dg.parent_pixels['hourly'].time[0].values, utc=True)
            
            ## work out the timesteps of the batch        
            batch_timesteps = np.random.choice(['daily', 'hourly'], batch_size,
                                               p=[1-p_hourly, p_hourly])        
            if ('daily' in batch_timesteps) and ('hourly' in batch_timesteps):
                batch_tsteps = 'mix'
            elif batch_timesteps[0]=='daily':
                batch_tsteps = 'daily'
            else:
                batch_tsteps = 'hourly'

            ## process and normalise ERA5 parent_pixels
            dg.prepare_era5_pixels(var, batch_tsteps=batch_tsteps)

            if batch_tsteps!='hourly':
                ## load chess data
                dg.chess_var = dg.var_name_map.loc[var].chess
                dg.chess_dat = load_process_chess(dg.td.year, dg.td.month, dg.td.day,
                                                    var=dg.chess_var)

            ## get additional fine scale met vars for input
            dg.prepare_fine_met_inputs(var, batch_tsteps)
            
            if batch_tsteps!='hourly': # cover daily and mix
                ## pool to scale and fill in the parent pixels
                chess_pooled = pooling(dg.chess_dat[dg.chess_var].values,
                                       (dg.scale, dg.scale),
                                       method='mean')
                chess_mask = ~np.isnan(chess_pooled)
                dg.parent_pixels['daily'][dg.era5_var].values[chess_mask] = chess_pooled[chess_mask]   
                            
            # solar values for daily timestep
            dg.calculate_day_av_solar_vars(var, batch_tsteps)
            
            # # generate batch from parent pixels
            # sample = dg.get_sample(var, batch_type=batch_type,
                                     # context_frac=context_frac,
                                     # timestep=batch_timesteps[b])
            timestep = batch_timesteps[b]
            ## sample a dim_l x dim_l tile
            ix, iy, it = dg.sample_xyt(batch_type=batch_type, timestep=timestep)
            
            it = batch['batch_metadata'][0]['t_ind']
            
            ## load input data
            (coarse_input, fine_input, subdat,
                x_inds, y_inds, lat_grid, lon_grid, timestamp) = dg.get_input_data(
                var, ix, iy, it, return_intermediates=True, timestep=timestep
            )

            swin_coarse = dg.parent_pixels['hourly'][dg.var_name_map.loc['SWIN'].coarse][it,:,:] * nm.swin_norm
            fine_grid = dg.chess_grid
            sp = dg.sp
            press_coarse = dg.parent_pixels['hourly'][dg.var_name_map.loc['PA'].coarse][it,:,:] * nm.p_sd + nm.p_mu
            press_fine = dg.p_1km_elev[it,:,:]
            height_grid = dg.height_grid
            shading_array = dg.shading_array[it,:]
            
            timestamp = pd.to_datetime(
                dg.parent_pixels['hourly'].time.values[it], utc=True
            )        
            sp = SolarPosition(timestamp, timezone=0) # utc
            sp.calc_solar_angles(dg.chess_grid.lat, dg.chess_grid.lon)
            
            
            #if np.all(swin_coarse.values==0):
                # short cut for night time
                #return fine_grid.landfrac * 0
            
            # from Proposal of a regressive model for the hourly diffuse
            # solar radiation under all sky conditions, J.A. Ruiz-Arias, 2010
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ## calculate diffuse fraction
                I0 = 1361.5 # solar constant, W m-2    
                sw_in = swin_coarse.interp_like(fine_grid, method='linear')
                
                mask = sp.solar_elevation >= 0
                        
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
                    np.sin(np.deg2rad(height_grid.slope.values[mask])) *
                    np.cos(np.deg2rad(sp.solar_azimuth_angle.values[mask] - height_grid.aspect.values[mask]))
                )
                # we replace the azimuthal/aspect contribution with our shade mask:
                shade_map = (fine_grid.landfrac.copy()).astype(np.uint8)
                shade_map.values[:,:] = 1
                shade_map.values[fine_grid.landfrac.values==1] = -(shading_array-1) # invert shade to zero
                
                # or daily from average
                #shade_map = (fine_grid.landfrac.copy())
                #shade_map.values[:,:] = 1
                #shade_map.values[fine_grid.landfrac.values==1] = -(np.mean(shading_array, axis=0)-1) # invert shade to zero                
                
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
                
                # renorm
                pooled = pooling(Sw_1km.values, (25,25), method='mean')
                sw_c_coarse = swin_coarse.copy()
                sw_c_coarse.values = pooled
                grid_ratio = swin_coarse / sw_c_coarse
                grid_ratio_interp = grid_ratio.interp_like(fine_grid, method='linear')
                Sw_1km = Sw_1km * grid_ratio_interp
