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

from downscaling.timeout import Timeout
from downscaling.data_classes.met_data import ERA5Data
from downscaling.data_classes.cosmos_data import CosmosMetaData, CosmosData
from downscaling.utils import *
from downscaling.params import normalisation as nm
from downscaling.params import data_pars as dp
from downscaling.solar_position import SolarPosition

EPS = 1e-10

hj_base = '/gws/nopw/j04/hydro_jules/data/uk/'
hj_ancil_fldr = hj_base + '/ancillaries/'
era5_fldr = hj_base + '/driving_data/era5/28km_grid/'
nz_base = '/gws/nopw/j04/ceh_generic/netzero/'
nz_train_path = nz_base + '/downscaling/training_data/'
#precip_fldr = '/badc/ukmo-hadobs/data/insitu/MOHC/HadOBS/HadUK-Grid/v1.1.0.0/1km/rainfall/day/latest/'
precip_fldr = '/gws/nopw/j04/ceh_generic/netzero/downscaling/ceh-gear/'
home_data_dir = '/home/users/doran/data_dump/'
midas_fldr = home_data_dir + '/MetOffice/midas_data/'
ea_fldr = home_data_dir + '/EA_rain_gauges/'
chessmet_dir = hj_base + '/driving_data/chess/chess-met/daily/'

def load_process_chess(year, month, day, chess_var, normalise=True):
    if type(chess_var)==str: chess_var = [chess_var]
    if 'rlds' in chess_var:
        chess_var.append('huss')
    if 'huss' in chess_var:
        chess_var.append('psurf')
        chess_var.append('tas')
    if 'rsds' in chess_var:
        chess_var.append('psurf')
    chess_var = list(set(chess_var))
    
    ## load
    st = chessmet_dir + f'/chess-met_'
    en = f'_gb_1km_daily_{year}{zeropad_strint(month)}*.nc'    
    fnames = [glob.glob(st+v+en)[0] for v in chess_var]
    chess_dat = xr.open_mfdataset(fnames)
    chess_dat = chess_dat.isel(time=day-1)

    ## rescale
    if ('huss' in chess_var) or ('rlds' in chess_var):
        # specific humidity to RH, requires psurf in Pa and T in K
        chess_dat.huss.values = (0.263 * chess_dat.psurf.values * chess_dat.huss.values * 
            np.exp((-17.67 * (chess_dat.tas.values - 273.16)) /
                    (chess_dat.tas.values - 29.65)))
    if ('tas' in chess_var) or ('huss' in chess_var):
         # K to C
        chess_dat.tas.values = chess_dat.tas.values - 273.15
    if ('psurf' in chess_var) or ('huss' in chess_var) or ('rsds' in chess_var):
        # Pa to hPa
        chess_dat.psurf.values = chess_dat.psurf.values / 100.
    if normalise:
        return normalise_chess_data(chess_dat, chess_var)
    else:
        return chess_dat

def normalise_chess_data(chess_dat, chess_var):
    ## normalise
    if ('rlds' in chess_var) or chess_var is None:
        # incoming longwave radiation
        chess_dat.rlds.values = (chess_dat.rlds.values - nm.lwin_mu) / nm.lwin_sd
    if ('rsds' in chess_var) or chess_var is None:
        # incoming shortwave radiation
        #chess_dat.rsds.values = np.log(1. + chess_dat.rsds.values)
        #chess_dat.rsds.values = (chess_dat.rsds.values - nm.logswin_mu) / nm.logswin_sd
        chess_dat.rsds.values = chess_dat.rsds.values / nm.swin_norm
    if ('psurf' in chess_var) or ('huss' in chess_var) or ('rsds' in chess_var) or chess_var is None:
        # air pressure
        chess_dat.psurf.values = (chess_dat.psurf.values - nm.p_mu) / nm.p_sd
    if ('huss' in chess_var) or ('rlds' in chess_var) or chess_var is None:
        # relative humidity            
        chess_dat.huss.values = (chess_dat.huss.values - nm.rh_mu) / nm.rh_sd
    if ('tas' in chess_var) or ('huss' in chess_var) or chess_var is None:
        # temperature
        chess_dat.tas.values = (chess_dat.tas.values - nm.temp_mu) / nm.temp_sd
    if ('sfcWind' in chess_var) or chess_var is None:
        # wind speed            
        chess_dat.sfcWind.values = (chess_dat.sfcWind.values - nm.ws_mu) / nm.ws_sd
    if ('dtr' in chess_var):
        # daily temperature range
        chess_dat.dtr.values = chess_dat.dtr.values / nm.temp_sd
    return chess_dat

def create_chess_pred(sample_metadata, datgen, var, pred):
    ## load
    chess_dat = load_process_chess(sample_metadata[0]['timestamp'].year,
                                   sample_metadata[0]['timestamp'].month,
                                   sample_metadata[0]['timestamp'].day,
                                   datgen.var_name_map.loc[var].chess)    
    ## create "chess-pred"
    pred3 = np.zeros(pred.shape)
    for bb in range(len(sample_metadata)):    
        subdat = chess_dat.isel(y=sample_metadata[bb]['y_inds'],
                                x=sample_metadata[bb]['x_inds'])
        for j, cv in enumerate(datgen.coarse_variable_order):
            vv = datgen.var_name_map[datgen.var_name_map['fine']==cv]['chess'].values[0]
            pred3[bb,j,:,:] = subdat[vv]
    return pred3

def read_one_cosmos_site_met(SID, missing_val = -9999.0):
    data = CosmosData(SID)    
    data.read_subhourly()
    data.preprocess_all(missing_val, 'DATE_TIME')
    return data

def circular_mean(x):
    ''' calculate wind components
    Wind direction is generally reported by the direction from which the wind originates. For example, a north or northerly wind blows from the north to the south
    Wind direction increases clockwise such that:
        - a northerly wind is 0°, an easterly wind is 90°,
        - a southerly wind is 180°, and
        - a westerly wind is 270°.
    It is still easy to compute the wind components, u and v,
    given the meteorological wind angle. Let Φ be the meteorological
    wind direction angle, then the following equations can be applied:

    u = - W sin Φ   # wind speed blowing east
    v = - W cos Φ   # wind speed blowing north
    W = sqrt(u^2 + v^2)
    '''
    return np.arctan2(np.sum(np.sin(np.deg2rad(x))),
                      np.sum(np.cos(np.deg2rad(x))))

def get_50m_DTM_elev(site_meta):
    from convertbng.util import convert_bng
    import rasterio
    eastings, northings = convert_bng(site_meta.LONGITUDE, site_meta.LATITUDE)
    ii = np.where(site_meta.SITE_ID == '01386_scilly-st-marys-airport')[0][0]
    eastings[ii] = 91712
    northings[ii] = 10496
    
    coords = np.column_stack([eastings, northings])
    path_to_dtm = '/home/users/doran/data_dump/DTM/DTMGEN_UK.tif'
    with rasterio.open(path_to_dtm) as src:
        data_at_points = np.array([p[0] for p in src.sample(coords)])
    non_cosmos_inds = np.where(site_meta.SITE_ID.str.contains('EA_'))[0]
    site_meta['ALTITUDE'].iloc[non_cosmos_inds] = data_at_points[non_cosmos_inds]
    (site_meta[site_meta['SITE_ID'].str.contains('EA_')]
        [['SITE_ID', 'LONGITUDE', 'LATITUDE', 'ALTITUDE']]
        .to_csv(ea_fldr + '/site_elevation.csv')
    )
    return site_meta

def provide_cosmos_met_data(metadata, met_vars, sites=None, missing_val = -9999.0, forcenew=False):
    Path(home_data_dir+'/met_pickles/').mkdir(exist_ok=True, parents=True)
    fname = home_data_dir+'/met_pickles/cosmos_site_met.pkl'
    if forcenew is False and Path(fname).is_file():
        with open(fname, 'rb') as fo:
            metdat = pickle.load(fo)
        met_data = metdat.pop('data')        
    else:        
        met_data = {}
        if sites is None: sites = metadata.site['SITE_ID']
        # remove wind components and insert wind direction
        new_met_vars = np.hstack([np.setdiff1d(met_vars, ['UX', 'VY']), ['WD']])
        for SID in sites:            
            data = read_one_cosmos_site_met(SID, missing_val = missing_val)    
            data = data.subhourly[new_met_vars]
            # check that this aggregation labelling matches ERA5 hourly meaning!
            thisdat = pd.merge(
                pd.merge(
                    (data.loc[:,np.setdiff1d(new_met_vars, ['PRECIP', 'WD'])]
                        .resample('1H', label='right', closed='right').mean()
                    ),
                    (data.loc[:,['PRECIP']]
                        .resample('1H', label='right', closed='right').sum()
                    ), on='DATE_TIME', how='left'
                ), (data.loc[:,['WD']].reset_index()
                        .resample('1H', label='right', closed='right', on='DATE_TIME').apply(circular_mean)
                    ), on='DATE_TIME', how='left'
            )
            thisdat['UX'] = - thisdat.WS * np.sin(thisdat.WD)
            thisdat['VY'] = - thisdat.WS * np.cos(thisdat.WD)
            met_data[SID] = thisdat
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
        # remove wind components and insert wind direction
        new_met_vars = np.hstack([np.setdiff1d(met_vars, ['UX', 'VY']), ['WD']])
        for SID in sites:
            if Path(midas_fldr + f'/{SID}.parquet').exists():
                data = pd.read_parquet(midas_fldr + f'/{SID}.parquet')
                try:
                    thisdat = data[new_met_vars]
                    thisdat['UX'] = - thisdat.WS * np.sin(np.deg2rad(thisdat.WD))
                    thisdat['VY'] = - thisdat.WS * np.cos(np.deg2rad(thisdat.WD))
                    met_data[SID] = thisdat
                except:
                    print(f'Met vars missing from {SID}')
        metdat = dict(data=met_data)
        with open(fname, 'wb') as fs:
            pickle.dump(metdat, fs)
    return met_data

def load_midas_elevation():
    # load midas site elevation
    midas_meta = pd.read_csv('/badc/ukmo-midas/metadata/SRCE/SRCE.DATA.COMMAS_REMOVED')
    mid_cols = ['SRC_ID', 'SRC_NAME','HIGH_PRCN_LAT',# - Latitude to 0.001 deg
                'HIGH_PRCN_LON',# - Longitude to 0.001 deg
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
    fldr = hj_base+'/soil_moisture_map/ancillaries/land_cover_map/2015/data/'
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
    grid_ratio = coarse_data / (c_coarse + EPS)
    grid_ratio_interp = interp_to_grid(grid_ratio, fine_grid)
    grid_ratio_interp = reflect_pad_nans(grid_ratio_interp.load().copy())     
    return fine_data * grid_ratio_interp

def interp_to_grid(data, grid, coords=['lat', 'lon']):
    data_interp = data.interp_like(grid, method='linear')
    for coord in coords:
        data_interp[coord] = grid[coord]
    return data_interp

def calculate_illumination_map(sp, grid, height_grid):
    solar_illum = grid.landfrac.copy()
    solar_illum.values = (
        np.cos(np.deg2rad(sp.solar_zenith_angle.values)) * 
        np.cos(np.deg2rad(height_grid.slope.values)) + 
        np.sin(np.deg2rad(sp.solar_zenith_angle.values)) * 
        np.sin(np.deg2rad(height_grid.slope.values)) *
        np.cos(np.deg2rad(sp.solar_azimuth_angle.values - height_grid.aspect.values))
    )
    return solar_illum


# swin_coarse = dg.parent_pixels['hourly'][dg.var_name_map.loc['SWIN'].coarse][it,:,:] * nm.swin_norm # de-nornmalise!
# fine_grid = dg.fine_grid
# sp = dg.sp
# press_coarse = dg.parent_pixels['hourly'][dg.var_name_map.loc['PA'].coarse][it,:,:] * nm.p_sd + nm.p_mu # de-nornmalise
# press_fine = dg.p_1km_elev[it,:,:]
# height_grid = dg.height_grid
# shading_array = dg.shading_array[it,:]
# sea_shading_array = dg.sea_shading_array[it,:]
# scale = dg.scale
# sky_view_factor = dg.skyview_map

def partition_interp_solar_radiation(swin_coarse, fine_grid, sp,
                                     press_coarse, press_fine,
                                     height_grid, shade_map,
                                     scale, sky_view_factor):
    # A Physically Based Atmospheric Variables Downscaling Technique,
    # TASNUVA ROUF et al. (2019)
    if np.all(swin_coarse.values==0):
        # short cut for night time
        return fine_grid.landfrac * 0
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ## calculate diffuse fraction
        I0 = 1361.5 # solar constant, W m-2            
        sw_in = interp_to_grid(swin_coarse, fine_grid, coords=['lat', 'lon'])        
        sw_in = reflect_pad_nans(sw_in.load().copy())        
                
        '''
        this method of partitioning does not work for solar elevation
        <~ +- 6 degrees!
        What do we do around sunrise and sunset...
        '''
        # the clearness index, ratio of cloud top solar to surface incident solar rad
        kt = fine_grid.landfrac.copy()
        kt.values = sw_in.values / (I0 * np.cos(np.deg2rad(sp.solar_zenith_angle.values)))
        kt = kt.clip(min=0, max=1)
        ma = (90 - sp.solar_zenith_angle.values) <= 6
        kt_pert = fine_grid.landfrac.copy()
        kt_pert.values[:,:] = 1
        kt_pert.values[ma] = (90 - sp.solar_zenith_angle.values)[ma] / 6.
        kt_pert = kt_pert.clip(min=0, max=1)
        kt.values = kt.values * kt_pert.values
        
        # diffuse fraction
        # from Proposal of a regressive model for the hourly diffuse
        # solar radiation under all sky conditions, J.A. Ruiz-Arias, 2010
        k = 0.952 - 1.041 * np.exp(-np.exp(2.300 - 4.702*kt))
        
        ## partitioning into direct and diffuse
        S_dir = sw_in * (1 - k)
        S_dif = sw_in * k
        
        ### adjustments to SW components
        ## diffuse
        S_dif.values *= sky_view_factor.values
        
        ## direct
        # terrain illumination/shading
        cos_solar_illum = calculate_illumination_map(sp, fine_grid, height_grid)

        # add shade map        
        # shade_map = (fine_grid.landfrac.copy()).astype(np.float32)        
        # shade_map.values[fine_grid.landfrac.values>0] = -(shading_array-1) # invert shade to zero
        # shade_map.values[fine_grid.landfrac.values==0] = -(sea_shading_array-1) # invert shade to zero
        
        # broadband attenuation
        press_coarse.load()
        p_interp = interp_to_grid(press_coarse, fine_grid, coords=['lat', 'lon'])        
        p_interp = reflect_pad_nans(p_interp.load().copy())
        
        mask = sp.solar_zenith_angle <= 90
        broadband_attenuation = fine_grid.landfrac.copy()
        broadband_attenuation.values[:,:] = 0
        broadband_attenuation.values[mask] = - (
            (np.log(I0 * np.cos(np.deg2rad(sp.solar_zenith_angle.values[mask])) + EPS) - 
                np.log(S_dir.values[mask] + EPS)) / p_interp.values[mask]
        )
        
        press_fine = reflect_pad_nans(press_fine.load().copy())
        S_dir.values *= shade_map.values * cos_solar_illum.values * np.exp(broadband_attenuation.values * (press_fine.values - p_interp.values))        
        
        ## reflected
        # Modeling Topographic Solar Radiation Using GOES Data, R. Dubayah and S. Loechel
        terrain_config_factor = 0.5*(1 + np.cos(np.deg2rad(height_grid.slope.values))) - sky_view_factor
        albedo = 0.23 # chess PE assumption across the UK... bad! satellite data?
        
        S_dir = S_dir.clip(min=0)
        S_dif = S_dif.clip(min=0)
        S_ref = albedo * terrain_config_factor * (S_dir + (1-sky_view_factor) * S_dif)
        
        # final SW 1km
        Sw_1km = S_dir + S_dif + S_ref
        
        # renorm to match the input coarse SWIN to counteract errors in partitioning
        Sw_1km = scale_by_coarse_data(Sw_1km.copy(), swin_coarse, fine_grid, scale)        
        return Sw_1km

def reflect_pad_nans(data, ysize=1200, xsize=600):
    if len(data.values.shape)==2:
        nx_l = len(np.where(np.isnan(data[ysize//2, :xsize//2]))[0])
        nx_r = len(np.where(np.isnan(data[ysize//2, xsize//2:]))[0])
        
        ny_b = len(np.where(np.isnan(data[:ysize//2, xsize//2]))[0])
        ny_t = len(np.where(np.isnan(data[ysize//2:, xsize//2]))[0])
        
        # reflect
        if nx_l>0:
            data.values[:, :nx_l] = data.values[:, nx_l:2*nx_l][:, ::-1] # ny_b:-ny_t
        if nx_r>0:
            data.values[:, -nx_r:] = data.values[:, -2*nx_r:-nx_r][:, ::-1] # ny_b:-ny_t
        if ny_b>0:
            data.values[:ny_b, :] = data.values[ny_b:2*ny_b, :][::-1, :] # nx_l:-nx_r
        if ny_t>0:
            data.values[-ny_t:, :] = data.values[-2*ny_t:-ny_t, :][::-1, :] # nx_l:-nx_r
        
        # corners
        if ny_b>0 and nx_l>0:
            data.values[:ny_b, :nx_l] = np.nanmean(data.values[:2*ny_b, :2*nx_l])
        if ny_t>0 and nx_l>0:
            data.values[-ny_t:, :nx_l] = np.nanmean(data.values[-2*ny_t:, :2*nx_l])
        if ny_b>0 and nx_r>0:
            data.values[:ny_b:, -nx_r:] = np.nanmean(data.values[:2*ny_b:, -2*nx_r:])
        if ny_t>0 and nx_r>0:
            data.values[-ny_t:, -nx_r:] = np.nanmean(data.values[-2*ny_t:, -2*nx_r:])
    else:
        nx_l = len(np.where(np.isnan(data[0, ysize//2, :xsize//2]))[0])
        nx_r = len(np.where(np.isnan(data[0, ysize//2, xsize//2:]))[0])
        
        ny_b = len(np.where(np.isnan(data[0, :ysize//2, xsize//2]))[0])
        ny_t = len(np.where(np.isnan(data[0, ysize//2:, xsize//2]))[0])
        
        # reflect
        data.values[:, :, :nx_l] = data.values[:, :, nx_l:2*nx_l][:, :, ::-1]
        data.values[:, :, -nx_r:] = data.values[:, :, -2*nx_r:-nx_r][:, :, ::-1]
        data.values[:, :ny_b, :] = data.values[:, ny_b:2*ny_b, :][:, ::-1, :]
        data.values[:, -ny_t:, :] = data.values[:, -2*ny_t:-ny_t, :][:, ::-1, :]
        
        # corners
        if ny_b>0 and nx_l>0:
            data.values[:, :ny_b, :nx_l] = (np.nanmean(data.values[:, :2*ny_b, :2*nx_l], axis=(1,2))[...,None, None] + 
                np.zeros((data.values.shape[0], ny_b, nx_l)))
        if ny_t>0 and nx_l>0:
            data.values[:, -ny_t:, :nx_l] = (np.nanmean(data.values[:, -2*ny_t:, :2*nx_l], axis=(1,2))[...,None, None] + 
                np.zeros((data.values.shape[0], ny_t, nx_l)))
        if ny_b>0 and nx_r>0:
            data.values[:, :ny_b:, -nx_r:] = (np.nanmean(data.values[:, :2*ny_b:, -2*nx_r:], axis=(1,2))[...,None, None] + 
                np.zeros((data.values.shape[0], ny_b, nx_r)))
        if ny_t>0 and nx_r>0:
            data.values[:, -ny_t:, -nx_r:] = (np.nanmean(data.values[:, -2*ny_t:, -2*nx_r:], axis=(1,2))[...,None, None] + 
                np.zeros((data.values.shape[0], ny_t, nx_r)))
    return data
    
class Batch:
    def __init__(self, batch, masks=None, constraints=True,
                 var_list=None, device=None):
        if type(var_list)==str:
            var_list = [var_list]
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.raw_station_dict = batch['station_targets']
            self.batch_metadata = batch['batch_metadata']
            
            self.coarse_inputs = batch['coarse_inputs'].to(device)
            self.fine_inputs = batch['fine_inputs'].to(device)
            
            if var_list[0]=='WS':
                self.context_station_dict = tensorise_station_targets(
                    [s['context'] for s in batch['station_targets']],
                    device=device,
                    var_list=['UX', 'VY']
                )
                self.target_station_dict = tensorise_station_targets(
                    [s['target'] for s in batch['station_targets']],
                    device=device,
                    var_list=['UX', 'VY']
                )
            else:
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

            self.context_data = [a.to(device) for a in batch['station_data']]
            self.context_locs = [a.to(device) for a in batch['context_locations']]
            
            # number of hourly observations aggregated to get daily avg
            self.context_num_obs = [s['context'].values.astype(np.int32).flatten()
                for s in batch['station_num_obs']]
            self.target_num_obs = [s['target'].values.astype(np.int32).flatten()
                for s in batch['station_num_obs']]
            
            self.n_gridpts = batch['fine_inputs'].shape[-1] * batch['fine_inputs'].shape[-2]
            self.n_batch = batch['coarse_inputs'].shape[0]
        except:
            print('Creating empty batch')
            self.raw_station_dict = None
            self.batch_metadata = None
            self.coarse_inputs = None
            self.fine_inputs = None
            self.context_station_dict = None
            self.target_station_dict = None
            self.constraint_targets = None
            self.context_data = None
            self.context_locs = None
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
        self.context_data = [m.clone() for m in other_batch.context_data]
        self.context_locs = [m.clone() for m in other_batch.context_locs]
        self.context_num_obs = [m.copy() for m in other_batch.context_num_obs]
        self.target_num_obs = [m.copy() for m in other_batch.target_num_obs]



class data_generator():
    def __init__(self, train_sites=None, heldout_sites=None, load_site_data=True, precip_sites=False):
        
        # load 1km bng grid        
        #self.chess_grid = xr.open_dataset(hj_ancil_fldr+'/chess_lat_lon.nc')
        #self.chess_grid = self.fine_grid.load() # force load data
        
        self.met_cls = ERA5Data()
        
        self.wgs84_epsg = 4326
        self.bng_epsg = 27700
        self.res = dp.res # resolution of lo-res image in m
        self.scale = dp.scale # downscaling factor
        self.dim_l = dp.dim_l # size of lo-res image in 28km pixels (size of subest of UK to train on each sample)
        self.dim_h = self.dim_l*dp.scale # size of hi-res image in 1km pixels
        
        # load parent pixel ids on reprojected/regridded ERA5 BNG 28km cells
        self.coarse_grid = xr.open_dataset(home_data_dir + '/bng_grids/bng_28km_pixel_ids.nc')
        self.coarse_grid = self.coarse_grid.load()
        # fill elev nans as zeros (the sea)
        self.coarse_grid.elev.values[np.isnan(self.coarse_grid.elev.values)] = 0
        
        # load child pixels on 1km BNG grid labelled by parent pixel IDs
        self.fine_grid = xr.open_dataset(home_data_dir + '/bng_grids/bng_1km_28km_parent_pixel_ids.nc')
        self.fine_grid = self.fine_grid.load()
        
        ## load hi res static data        
        #self.height_grid = xr.open_dataset(hj_ancil_fldr+'/uk_ihdtm_topography+topoindex_1km.nc')
        self.height_grid = xr.open_dataset(home_data_dir + '/height_map/topography_bng_1km.nc')
        self.height_grid = self.height_grid.drop(['topi','stdtopi', 'fdepth','area']).load()        
        self.elev_vars = ['elev', 'stdev', 'slope', 'aspect']
        
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
        
        # load land cover for wind speed / long wave radiation model
        self.land_cover = load_landcover()
        # invert (due to rasterio loading 'correct' way up'
        self.land_cover = self.land_cover[:,::-1,:]
        # and trim to size (this assumed land cover map is bigger!)
        self.land_cover = self.land_cover[:,:len(self.fine_grid.y),:len(self.fine_grid.x)]
        self.lcm_names = ['l_wooded', 'l_open', 'l_mountain-heath', 'l_urban']        
        
        # load sky view factor map
        self.skyview_map = xr.load_dataset(nz_train_path + '/sky_view_factor.nc')
        self.skyview_map = self.skyview_map.sky_view_factor     
    
        # define ERA5 (coarse) and COSMOS (fine) variables to load
        var_name_map = dict(
            name   = ['Air_Temp', 'Pressure', 'Short_Wave_Rad_In',
                      'Long_Wave_Rad_In', 'Wind_Speed', 'Relative_Humidity',
                      'Precipitation', 'Wind_Speed_X', 'Wind_Speed_Y'],
            coarse = ['t2m', 'sp', 'msdwswrf', 'msdwlwrf', 'ws', 'rh',
                      'mtpr', 'u10', 'v10'],
            fine   = ['TA', 'PA', 'SWIN', 'LWIN', 'WS', 'RH',
                      'PRECIP', 'UX', 'VY']
        )        
        self.var_name_map = pd.DataFrame(var_name_map)
        self.var_name_map = self.var_name_map.assign(
            chess = np.array(['tas', 'psurf', 'rsds', 'rlds',
                              'sfcWind', 'huss', 'precip', 'NA', 'NA'])
            # huss is originally specific humidity but we transform it to RH
        )
        self.var_name_map.index = self.var_name_map.fine
        
        # define model inputs for each variable
        self.fine_variable_order = []
        self.coarse_variable_order = []
        self.context_variable_order = []

        self.rh_extra_met_vars = ['TA', 'PA']
        self.lwin_extra_met_vars = ['RH', 'TA']
        self.swin_extra_met_vars = ['PA']
        self.precip_extra_met_vars = ['UX', 'VY']
        
        self.targ_var_depends = {}
        self.targ_var_depends['TA'] = ['landfrac', 'elev']
        self.targ_var_depends['PA'] = ['landfrac', 'elev']
        self.targ_var_depends['SWIN'] = ['landfrac', 'elev',
                                         'illumination_map',                                         
                                         'solar_altitude',
                                         'cloud_cover'] + self.swin_extra_met_vars
        self.targ_var_depends['LWIN'] = ['landfrac',
                                         'sky_view_factor',
                                         'cloud_cover'] + self.lwin_extra_met_vars
        self.targ_var_depends['WS'] = ['landfrac', 'elev', 'stdev',
                                       'aspect', 'slope', 'land_cover']
        self.targ_var_depends['RH'] = ['landfrac'] + self.rh_extra_met_vars
        self.targ_var_depends['PRECIP'] = ['landfrac', 'elev', 'cloud_cover',
                                           'aspect'] + self.precip_extra_met_vars        
        
        ## deal with site data
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
        
        if load_site_data:
            # load site data
            self.site_data = provide_cosmos_met_data(self.site_metadata,
                                                     self.var_name_map['fine'].values,
                                                     forcenew=False)
        self.site_metadata = self.site_metadata.site

        #if load_site_data:
        midas_data = provide_midas_met_data(midas_metadata,
                                            self.var_name_map['fine'].values,
                                            forcenew=False)

        # rescale midas air temperature to Kelvin to match COSMOS and add to self.site_dat
        for sid in midas_data.keys():
            if load_site_data:
                midas_data[sid]['TA'] = midas_data[sid]['TA'] + 273.15
                self.site_data[sid] = midas_data[sid]
            self.site_metadata = pd.concat([
                self.site_metadata, 
                midas_metadata[midas_metadata['SITE_ID']==sid]], axis=0)
        self.site_metadata = self.site_metadata.reset_index().drop('index', axis=1)
        
        # numericise site elevation
        self.site_metadata['ALTITUDE'] = self.site_metadata.ALTITUDE.astype(np.float32)

        # for each site find the 1km chess tile it sits within
        cosmos_chess_y = []
        cosmos_chess_x = []
        coarse_parent_pixel_id = []
        for i in self.site_metadata.index:
            this_dat = self.site_metadata.loc[i]
            ccyx = find_chess_tile(this_dat['LATITUDE'], this_dat['LONGITUDE'],
                                   self.fine_grid)
            cosmos_chess_y.append(ccyx[0][0])
            cosmos_chess_x.append(ccyx[1][0])
            try:
                coarse_parent_pixel_id.append(
                    int(self.fine_grid.era5_nbr[ccyx[0][0], ccyx[1][0]].values)
                )
            except:
                coarse_parent_pixel_id.append(np.nan)

        # add to site metadata but filter out missing vals        
        self.site_metadata = self.site_metadata.assign(chess_y = cosmos_chess_y,
                                                       chess_x = cosmos_chess_x,
                                                       parent_pixel_id = coarse_parent_pixel_id)
        missing_sites = self.site_metadata[self.site_metadata['parent_pixel_id'].isna()]['SITE_ID']
        self.site_metadata = self.site_metadata[~self.site_metadata['parent_pixel_id'].isna()]
        
        if load_site_data:
            for sid in missing_sites: 
                self.site_data.pop(sid)
                midas_data[sid] = None
            del(midas_data)                
                                                
        # remove sites for which we don't have static data (ireland, shetland)
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
        shetland_sites = self.site_metadata[self.site_metadata['LATITUDE']>59.5]
        if load_site_data:
            for sid in ireland_sites.SITE_ID:
                self.site_data[sid] = None
                self.site_data.pop(sid)
            for sid in shetland_sites.SITE_ID:
                self.site_data[sid] = None
                self.site_data.pop(sid)
        self.site_metadata = self.site_metadata[~self.site_metadata['SITE_ID'].isin(ireland_sites['SITE_ID'])]
        self.site_metadata = self.site_metadata[~self.site_metadata['SITE_ID'].isin(shetland_sites['SITE_ID'])]
        
        # trim down columns
        self.site_metadata = self.site_metadata[
            ['SITE_NAME', 'SITE_ID', 'LATITUDE', 'LONGITUDE', 'ALTITUDE',
             'chess_y','chess_x', 'parent_pixel_id']
         ]
                
        # define train, val and test sets
        self.train_years = dp.train_years
        self.val_years = dp.val_years
        self.heldout_years = dp.heldout_years
        if train_sites is None:
            train_sites, heldout_sites = site_splits(use_sites=list(self.site_metadata.SITE_ID),
                                                     holdoutfrac=0.1, random_state=22)        
        self.train_sites = train_sites
        self.heldout_sites = heldout_sites
        
        if load_site_data:
            # normalise site data
            for SID in list(self.site_data.keys()):
                # incoming radiation                        
                self.site_data[SID].loc[:, 'LWIN'] = (self.site_data[SID].loc[:, 'LWIN'] - nm.lwin_mu) / nm.lwin_sd
                
                self.site_data[SID].loc[:, 'SWIN'] = self.site_data[SID].loc[:, 'SWIN'] / nm.swin_norm
                self.site_data[SID].loc[:, 'SWIN'] = self.site_data[SID].loc[:, 'SWIN'].clip(lower=0) # get rid of negative swin
                
                # air pressure
                self.site_data[SID].loc[:, ['PA']] = (self.site_data[SID].loc[:, ['PA']] - nm.p_mu) / nm.p_sd
                
                # relative humidity            
                self.site_data[SID].loc[:, ['RH']] = (self.site_data[SID].loc[:, ['RH']] - nm.rh_mu) / nm.rh_sd
                
                # temperature
                #print(self.site_data[SID].loc[:, ['TA']].mean()) check T is in Kelvin?
                self.site_data[SID].loc[:, ['TA']] = (self.site_data[SID].loc[:, ['TA']] - 273.15 - nm.temp_mu) / nm.temp_sd
                
                # wind speed            
                self.site_data[SID].loc[:, ['WS']] = (self.site_data[SID].loc[:, ['WS']] - nm.ws_mu) / nm.ws_sd
                # these can be negative due to directionality!
                # careful with normalising them differently to WS and then attempting 
                # to calculate loss on sqrt(UX^2 + VY^2) =?= WS
                self.site_data[SID].loc[:, ['UX']] = (self.site_data[SID].loc[:, ['UX']]) / nm.ws_sd
                self.site_data[SID].loc[:, ['VY']] = (self.site_data[SID].loc[:, ['VY']]) / nm.ws_sd

                # precipitation
                self.site_data[SID].loc[:, 'PRECIP'] = self.site_data[SID].loc[:, 'PRECIP'] / nm.precip_norm
                self.site_data[SID].loc[:, 'PRECIP'] = self.site_data[SID].loc[:, 'PRECIP'].clip(lower=0) # get rid of negative precip                

            # resample site data to daily and note number of data points present
            self.site_points_present = {}
            self.daily_site_data = {}
            for SID in list(self.site_data.keys()):
                self.site_points_present[SID] = self.site_data[SID].groupby(pd.Grouper(freq='D')).count()
                self.daily_site_data[SID] = self.site_data[SID].resample('1D').mean()
        
        # indices in a dim_h x dim_h grid
        X1 = np.where(np.ones((self.dim_h, self.dim_h)))
        self.X1 = np.hstack([X1[0][...,np.newaxis],
                             X1[1][...,np.newaxis]])        

    def load_EA_rain_gauge_data(self, load_site_data=True, train_sites=None):
        # load EA (SEPA?, NRW?) site data for precipitation downscaling
        ea_site_meta = (pd.read_csv(ea_fldr + 'sites_info.csv')
            .drop(['ELEVATION', 'PATH', 'START_DATE', 'END_DATE'], axis=1)
        )
        ea_site_elev = pd.read_csv(ea_fldr + '/site_elevation.csv')
        
        # for each site find the 1km chess tile it sits within
        # and grab 1km elevation (better to do 50m elevation...)
        ea_chess_y = []
        ea_chess_x = []
        ea_coarse_parent_pixel_id = []
        for i in ea_site_meta.index:
            this_dat = ea_site_meta.loc[i]
            ccyx = find_chess_tile(this_dat['LATITUDE'], this_dat['LONGITUDE'],
                                   self.fine_grid)
            ea_chess_y.append(ccyx[0][0])
            ea_chess_x.append(ccyx[1][0])
            try:
                ea_coarse_parent_pixel_id.append(
                    int(self.fine_grid.era5_nbr[ccyx[0][0], ccyx[1][0]].values)
                )
            except:
                ea_coarse_parent_pixel_id.append(np.nan)

        # add to site metadata but filter out missing vals            
        ea_site_meta = ea_site_meta.assign(chess_y = ea_chess_y,
                                           chess_x = ea_chess_x,
                                           parent_pixel_id = ea_coarse_parent_pixel_id,
                                           )                                           
        ea_missing_sites = ea_site_meta[ea_site_meta['parent_pixel_id'].isna()]['SITE_ID']            
        ea_site_meta = ea_site_meta[~ea_site_meta['parent_pixel_id'].isna()]            
        
        if load_site_data:
            ea_site_data = {}
            ea_missing_sites2 = []
            for sid in ea_site_meta.SITE_ID:
                if Path(ea_fldr + f'/data_hourly/{sid}.parquet').exists():
                    thisdat = pd.read_parquet(ea_fldr + f'/data_hourly/{sid}.parquet')
                    try:
                        thisdat = thisdat.rename({'PRECIPITATION': 'PRECIP'}, axis=1)
                    except:
                        pass
                    for vv in np.setdiff1d(self.var_name_map['fine'].values, 'PRECIP'):
                        thisdat[vv] = np.nan
                    ea_site_data['EA_'+str(sid)] = thisdat
                else:
                    ea_missing_sites2.append(sid)
            ea_site_meta = ea_site_meta.loc[~ea_site_meta.SITE_ID.isin(ea_missing_sites2)]
            
        ea_site_meta['SITE_ID'] = 'EA_' + ea_site_meta.SITE_ID.values.astype(str).astype(object)
        ea_site_meta = ea_site_meta.merge(ea_site_elev, on=['SITE_ID', 'LATITUDE', 'LONGITUDE'])
        ea_site_meta['ALTITUDE'] = ea_site_meta.ALTITUDE.astype(np.float32)
        ea_site_meta = ea_site_meta.loc[~(abs(ea_site_elev.ALTITUDE) > 2000)]
            
        # trim down columns
        ea_site_meta = ea_site_meta[
            ['SITE_ID', 'LATITUDE', 'LONGITUDE', 'ALTITUDE',
             'chess_y','chess_x', 'parent_pixel_id']
         ]
            
        # remove sites for which we don't have static data (ireland, shetland)
        lat_up = 55.3
        lat_down = 53.0
        lon_right = -5.4
        lon_left = -8.3
        ireland_sites = ea_site_meta[
            ((ea_site_meta['LATITUDE']>lat_down) & 
             (ea_site_meta['LATITUDE']<lat_up) &
             (ea_site_meta['LONGITUDE']>lon_left) &
             (ea_site_meta['LONGITUDE']<lon_right))
        ]
        shetland_sites = ea_site_meta[ea_site_meta['LATITUDE']>59.5]
        if load_site_data:
            for sid in ireland_sites.SITE_ID:
                ea_site_data[sid] = None
                ea_site_data.pop(sid)
            for sid in shetland_sites.SITE_ID:
                ea_site_data[sid] = None
                ea_site_data.pop(sid)
        ea_site_meta = ea_site_meta[~ea_site_meta['SITE_ID'].isin(ireland_sites['SITE_ID'])]
        ea_site_meta = ea_site_meta[~ea_site_meta['SITE_ID'].isin(shetland_sites['SITE_ID'])]
        
        # define train, val and test sets
        if train_sites is None:
            train_sites, heldout_sites = site_splits(use_sites=list(ea_site_meta.SITE_ID),
                                                     holdoutfrac=0.1, random_state=22)        
        self.train_sites += train_sites
        self.heldout_sites += heldout_sites
        
        if load_site_data:
            # normalise site data
            for SID in list(ea_site_data.keys()):
                # precipitation
                ea_site_data[SID].loc[:, 'PRECIP'] = ea_site_data[SID].loc[:, 'PRECIP'] / nm.precip_norm
                ea_site_data[SID].loc[:, 'PRECIP'] = ea_site_data[SID].loc[:, 'PRECIP'].clip(lower=0) # get rid of negative precip                

            # resample site data to daily and note number of data points present            
            for SID in list(ea_site_data.keys()):
                self.site_points_present[SID] = ea_site_data[SID].groupby(pd.Grouper(freq='D')).count()
            
        # add to master meta and data
        self.site_metadata = pd.concat([self.site_metadata, ea_site_meta], axis=0)
        if load_site_data:
            for sid in list(ea_site_data.keys()):
                self.site_data[sid] = ea_site_data.pop(sid)
            
            # trim suspect PRECIP values at all sites
            # the 60-minute rainfall record in the UK is 92mm (from wikipedia)
            precip_hour_thresh = 93 / nm.precip_norm
            for sid in self.site_data.keys():
                self.site_data[sid][self.site_data[sid].PRECIP >= precip_hour_thresh] = np.nan

    def find_1km_pixel(self, lat, lon):
        dist_diff = np.sqrt(np.square(self.fine_grid.lat.values - lat) +
                            np.square(self.fine_grid.lon.values - lon))
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
            self.td = pd.to_datetime(self.td, utc=True)
            date_string = f'{self.td.year}{zeropad_strint(self.td.month)}{zeropad_strint(self.td.day)}'
        
        # load ERA5 data for that date and trim lat/lon
        era5_vars = self.var_name_map[(self.var_name_map['coarse']!='ws') &
                                      (self.var_name_map['coarse']!='rh')]['coarse']
        era5_vars = list(era5_vars) + ['d2m']
                
        era5_filelist = [f'{era5_fldr}/{var}/era5_{date_string}_{var}.nc' for var in era5_vars]
        
        era5_dat = xr.open_mfdataset(era5_filelist)
        
        # if False:
            # #self.era5_dat = xr.open_mfdataset(era5_filelist)
            # # because these are already on the BNG at 1km, find are averages of 25x25 squares        
            # self.parent_pixels[timestep] = self.parent_pixels[timestep].assign_coords(
                # {'time':self.era5_dat.time.values}
            # )
            # ys = self.parent_pixels[timestep].y.shape[0] - 1
            # xs = self.parent_pixels[timestep].x.shape[0] - 1
            # ts = self.parent_pixels[timestep].time.shape[0]
            # for var in era5_vars:
                # self.parent_pixels[timestep][var] = (['time', 'y', 'x'],
                                          # np.ones((ts, ys, xs), dtype=np.float32)*np.nan)
                # source = self.era5_dat[var].values
                # self.parent_pixels[timestep][var] = (('time', 'y', 'x'), 
                    # skimage.measure.block_reduce(
                        # source, (1, self.scale, self.scale), np.mean
                    # )[:ts,:ys,:xs] # trim right/bottom edges
                # )
            # del(source)
            # del(self.era5_dat)
        
        ## if loading raw lat/lon projection
        # reproject and regrid onto 28km BNG
        # self.era5_dat = self.era5_dat.rio.write_crs(rasterio.crs.CRS.from_epsg(self.wgs84_epsg))
        # self.era5_dat = self.era5_dat.rio.reproject(f"EPSG:{self.bng_epsg}")
        # self.era5_dat = self.era5_dat.interp_like(self.coarsegrid)
        
        return era5_dat

    def sample_xyt(self, batch_type='train', timestep='daily',
                   SID=None, parent_pixel_id=None, it=None):
        if SID is None and parent_pixel_id is None:
            # choose a station and find its 1km pixel
            if batch_type=='train' or batch_type=='val':
                SID = np.random.choice(self.train_sites)
            elif batch_type=='test':
                SID = np.random.choice(self.train_sites + self.heldout_sites)        
            targ_site = self.site_metadata[self.site_metadata['SITE_ID']==SID]            
            targ_yx = np.where(self.coarse_grid.pixel_id.values == targ_site['parent_pixel_id'].values)
        else:
            # use the provided site id or parent pixel id
            if SID is None:
                targ_yx = np.where(self.coarse_grid.pixel_id.values == parent_pixel_id)
            else:
                targ_site = self.site_metadata[self.site_metadata['SITE_ID']==SID]            
                targ_yx = np.where(self.coarse_grid.pixel_id.values == targ_site['parent_pixel_id'].values)                
                
        # grab a random dim_l x dim_l tile that contains that 1km pixel
        # but check it has enough land pixels
        land_pixels = 0
        tries = 0
        while land_pixels < 0.3*(self.dim_l*self.dim_l) and tries<5:
            ix = np.random.randint(max(0, targ_yx[1][0] - self.dim_l + 1),
                                   min(self.coarse_grid.x.shape[0] - self.dim_l + 1, targ_yx[1][0] + 1))
            iy = np.random.randint(max(0, targ_yx[0][0] - self.dim_l + 1),
                                   min(self.coarse_grid.y.shape[0] - self.dim_l + 1, targ_yx[0][0] + 1))
            land_pixels = (self.coarse_grid.isel(y=range(iy, iy+self.dim_l),
                                                 x=range(ix, ix+self.dim_l))
                .landfrac.values.sum())
            tries += 1 

        # randomly select an hour of the day unless we provide one
        if it is None:
            it = np.random.randint(0, 24)
        return ix, iy, it

    def get_input_data(self, var, ix=None, iy=None, it=0, timestep='daily'):        
        if (ix is None) and (iy is None):
            # for rectangular grid over whole space
            if timestep=='daily':
                subdat = self.parent_pixels[timestep]
            else:
                subdat = self.parent_pixels[timestep].isel(time=it)
            subdat['pixel_id'] = self.coarse_grid.pixel_id
        else:
            if timestep=='daily':
                subdat = self.parent_pixels[timestep].isel(y=range(iy, iy+self.dim_l),
                                                           x=range(ix, ix+self.dim_l))
            else:
                subdat = self.parent_pixels[timestep].isel(time=it,
                                                           y=range(iy, iy+self.dim_l),
                                                           x=range(ix, ix+self.dim_l))
            subdat['pixel_id'] = self.coarse_grid.pixel_id.isel(y=range(iy, iy+self.dim_l),
                                                                x=range(ix, ix+self.dim_l))
        
        if timestep=='daily':
            timestamp = self.td
        else:
            timestamp = pd.to_datetime(
                self.parent_pixels[timestep].time.values[it], utc=True
            )
        
        if timestep!='daily':
            if 'SWIN' in var or 'LWIN' in var:
                # take the mid-point of the hour for the solar position calc
                self.sp = SolarPosition(timestamp - datetime.timedelta(minutes=30), timezone=0) # utc
                self.sp.calc_solar_angles(self.fine_grid.lat, self.fine_grid.lon)
            if 'SWIN' in var:
                self.solar_illum_map = calculate_illumination_map(self.sp, self.fine_grid, self.height_grid)
                #self.shade_map = self.fine_grid.lat.copy()                
                #self.shade_map.values[self.fine_grid.landfrac.values>0] = -(self.shading_array[it,:] - 1)
                #self.shade_map.values[self.fine_grid.landfrac.values==0] = -(self.sea_shading_array[it,:] - 1)                
                self.shade_map = self.shading_ds.isel(time=it)
        
        # subset hi-res fields to the chosen coarse tile
        x_inds = np.intersect1d(np.where(self.fine_grid.x.values < int(subdat.x.max().values) + self.res//2),
                                np.where(self.fine_grid.x.values > int(subdat.x.min().values) - self.res//2))
        y_inds = np.intersect1d(np.where(self.fine_grid.y.values < int(subdat.y.max().values) + self.res//2),
                                np.where(self.fine_grid.y.values > int(subdat.y.min().values) - self.res//2))
        sub_grid = self.fine_grid.isel(y=y_inds, x=x_inds)
        sub_topog = self.height_grid.isel(y=y_inds, x=x_inds)
        
        # those vars with extra met data as input
        if ('RH' in var) or ('LWIN' in var) or ('SWIN' in var) or ('PRECIP' in var):
            if timestep=='daily':
                sub_met_dat = self.chess_dat.isel(y=y_inds, x=x_inds)
            else:
                sub_met_dat = self.met_fine.isel(y=y_inds, x=x_inds, time=it)

        if ('LWIN' in var) or ('SWIN' in var) or ('PRECIP' in var):
            sub_cloud_cover = self.tcc_1km.isel(y=y_inds, x=x_inds)
            if timestep=='daily':
                sub_cloud_cover = sub_cloud_cover.mean('time').values
            else:
                sub_cloud_cover = sub_cloud_cover.isel(time=it).values
                
        if ('LWIN' in var) or ('SWIN' in var):
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
        
        if 'LWIN' in var:
            sub_skyview = self.skyview_map.isel(y=y_inds, x=x_inds)            

        if 'SWIN' in var:
            if timestep=='daily':                
                sub_illum = self.sol_ilum.isel(y=y_inds, x=x_inds)
                sub_shade = self.av_shade_map.isel(y=y_inds, x=x_inds)
            else:
                sub_illum = self.solar_illum_map.isel(y=y_inds, x=x_inds)
                sub_shade = self.shade_map.isel(y=y_inds, x=x_inds)

        ## normalise hi res data
        sub_topog['aspect'].values = sub_topog['aspect'].values / 360. - 0.5 # so goes between -0.5 and 0.5
        for vv in list(set(self.elev_vars) - set(['aspect'])):            
            sub_topog[vv].values = (sub_topog[vv].values - nm.s_means.loc[vv]) / nm.s_stds.loc[vv]
        
        lat_grid = sub_grid.lat.values / nm.lat_norm
        lon_grid = sub_grid.lon.values / nm.lon_norm

        # create tensors with batch index and channel dim last
        coarse_input = []
        if var == ['WS']: # two-channel input for wind components
            for v in ['u10', 'v10']:
                coarse_input.append(torch.from_numpy(
                    subdat[v].values).to(torch.float32)
                )
            self.coarse_variable_order = ['UX', 'VY']
        else:
            for v in var:
                coarse_input.append(torch.from_numpy(
                    subdat[self.var_name_map.coarse.loc[v]].values).to(torch.float32)
                )
            self.coarse_variable_order = var
        coarse_input = torch.stack(coarse_input, dim=-1)

        # get static inputs at the high resolution
        fine_input = torch.from_numpy(sub_grid['landfrac'].values).to(torch.float32)[...,None]
        use_elev_vars = []
        for v in var:
            use_elev_vars += list(set(self.elev_vars) & set(self.targ_var_depends[v]))
        use_elev_vars = list(set(use_elev_vars))
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
        
        if 'SWIN' in var:
            # add illumination and terrain shading maps
            fine_input = torch.cat([fine_input,
                torch.from_numpy(sub_illum.values).to(torch.float32)[...,None],
                torch.from_numpy(sub_shade.values).to(torch.float32)[...,None],
                ], dim = -1)
            self.fine_variable_order += ['illumination_map', 'shade_map']
        
            # add solar elevation angle
            sub_sol_altitude = np.deg2rad(sub_sol_altitude.values)
            fine_input = torch.cat([fine_input,                
                torch.from_numpy(sub_sol_altitude).to(torch.float32)[...,None]
                ], dim = -1)
            self.fine_variable_order += ['solar_altitude']
        
        if 'LWIN' in var:
            # add sky view factor
            fine_input = torch.cat([fine_input,
                torch.from_numpy(sub_skyview.values).to(torch.float32)[...,None],                
                ], dim = -1)
            self.fine_variable_order += ['sky_view_factor']
            
        # if 'solar_azimuth' in self.targ_var_depends[var]:
            # # add azimuthal angle          
            # sub_sol_azimuth = np.deg2rad(sub_sol_azimuth.values)  
            # fine_input = torch.cat([fine_input,
                # torch.from_numpy(sub_sol_azimuth).to(torch.float32)[...,None]
                # ], dim = -1)
            # self.fine_variable_order += ['solar_azimuth']
        
        if ('LWIN' in var) or ('SWIN' in var) or ('PRECIP' in var):           
            # add cloud cover            
            fine_input = torch.cat([fine_input,                
                torch.from_numpy(sub_cloud_cover).to(torch.float32)[...,None]
                ], dim = -1)
            self.fine_variable_order += ['cloud_cover']
            
        if ('WS' in var): # or ('LWIN' in var)
            # add land cover classes (y axis inverted compared to other grids)
            #sub_lcm = self.land_cover[:, self.land_cover.shape[1]-1-y_inds, :][:, :, x_inds]
            # though we already invert and trim in __init__()
            sub_lcm = self.land_cover[:, y_inds, :][:, :, x_inds]
            fine_input = torch.cat([fine_input,                            
                torch.from_numpy(np.transpose(sub_lcm, (1,2,0))).to(torch.float32)
                ], dim = -1)
            self.fine_variable_order += self.lcm_names
        
        if 'RH' in var:
            for exv in self.rh_extra_met_vars:
                if not exv in self.fine_variable_order: # don't double count
                    if timestep=='daily': vname = self.var_name_map.loc[exv].chess
                    else: vname = self.var_name_map.loc[exv].coarse
                    fine_input = torch.cat([
                        fine_input,
                        torch.from_numpy(sub_met_dat[vname].values).to(torch.float32)[...,None]
                        ], dim = -1)
                    self.fine_variable_order += [exv]
        
        if ('SWIN' in var):
            for exv in self.swin_extra_met_vars:
                if not exv in self.fine_variable_order: # don't double count            
                    if timestep=='daily': vname = self.var_name_map.loc[exv].chess
                    else: vname = self.var_name_map.loc[exv].coarse
                    fine_input = torch.cat([
                        fine_input,
                        torch.from_numpy(sub_met_dat[vname].values).to(torch.float32)[...,None]
                        ], dim = -1)
                    self.fine_variable_order += [exv]
        
        if 'LWIN' in var:
            for exv in self.lwin_extra_met_vars:
                if not exv in self.fine_variable_order: # don't double count            
                    if timestep=='daily': vname = self.var_name_map.loc[exv].chess
                    else: vname = self.var_name_map.loc[exv].coarse
                    fine_input = torch.cat([
                        fine_input,
                        torch.from_numpy(sub_met_dat[vname].values).to(torch.float32)[...,None]
                        ], dim = -1)
                    self.fine_variable_order += [exv]
            
        if 'PRECIP' in var:
            for exv in self.precip_extra_met_vars:
                if not exv in self.fine_variable_order: # don't double count            
                    if timestep=='daily': vname = self.var_name_map.loc[exv].chess
                    else: vname = self.var_name_map.loc[exv].coarse
                    fine_input = torch.cat([
                        fine_input,
                        torch.from_numpy(sub_met_dat[vname].values).to(torch.float32)[...,None]
                        ], dim = -1)
                    self.fine_variable_order += [exv]
        
        # add lat/lon (always last two finescale inputs
        fine_input = torch.cat([fine_input,                            
                            torch.from_numpy(lat_grid).to(torch.float32)[...,None],
                            torch.from_numpy(lon_grid).to(torch.float32)[...,None]
                            ], dim = -1)
        self.fine_variable_order += ['lat', 'lon']
        
        return (coarse_input, fine_input, subdat, x_inds, y_inds, 
                lat_grid, lon_grid, timestamp)

    def get_station_targets(self, subdat, x_inds, y_inds, timestamp, 
                            var, latlon_grid, fine_input, batch_type='train',
                            trim_edge_sites=True, context_frac=None,
                            timestep='hourly'):
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
        elif batch_type=='run':
            contained_sites = self.site_metadata.copy()
        
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
        if var[0]=='WS':
            keep_vars = ['sub_x', 'sub_y'] + ['UX', 'VY'] + ['LATITUDE', 'LONGITUDE', 'ALTITUDE']
            station_npts = sites_npts[['UX', 'VY']]
        else:
            keep_vars = ['sub_x', 'sub_y'] + var + ['LATITUDE', 'LONGITUDE', 'ALTITUDE']
            station_npts = sites_npts[var]
        station_targets = contained_sites[keep_vars]
        
        
        # trim sites with no data
        if var[0]=='WS':
            keepsites = station_targets[['UX', 'VY']].dropna().index
        else:
            keepsites = station_targets[var].dropna(how='all').index
        station_targets = station_targets.loc[keepsites]
        station_npts = station_npts.loc[keepsites]
        
        if timestep=='daily':
            # mask vars at sites with fewer than 24 readings for daily data
            keepsites2 = []
            for v in var:
                keepsites2 += list(station_npts[station_npts[v]==24].index)
                nansites_v = station_npts[station_npts[v]<24].index
                station_targets.loc[nansites_v, v] = np.nan
            keepsites2 = list(set(keepsites2))
            station_targets = station_targets.loc[keepsites2]
            station_npts = station_npts.loc[keepsites2]
            
            if len(keepsites2)==0:
                # no sites with 24 readings, return null batch                
                val_dense_vecs = np.zeros((1, 2*len(var) + 3, 0), dtype=np.float32) # no null tag
                YX_locs = np.zeros((0,2), dtype=np.float32)
                return ({'context':station_targets, 'target':station_targets},
                         {'context':station_npts, 'target':station_npts},
                         val_dense_vecs, YX_locs)
                
        # create context/target splits (randomising context fraction)
        if context_frac is None:
            context_frac = np.random.uniform(0, 0.9)
        context, targets = context_target_split(
            station_targets.index,
            context_frac=context_frac,
            random_state=np.random.randint(1000)
        )
        
        # create value/density pairs (+ altitude) vector for context points
        if var[0]=='WS':
            self.context_variable_order = ['var_value_u', 'var_value_v', 'elev', 'lat', 'lon']
        else:
            self.context_variable_order = ['var_value', 'elev', 'lat', 'lon']
            
        if var[0]=='SWIN':
            val_dense_vecs = np.zeros((1, 7, len(context)), dtype=np.float32) # add illum map, shade map, cloud cover
            cloud_vec = []
            shade_vec = []
            illum_vec = []
            self.context_variable_order = self.context_variable_order + ['cloud_cover', 'shade_map', 'illumination_map']
        elif var[0]=='LWIN' or var[0]=='PRECIP':
            val_dense_vecs = np.zeros((1, 5, len(context)), dtype=np.float32)
            cloud_vec = []
            self.context_variable_order = self.context_variable_order + ['cloud_cover']
        elif var[0]=='WS':
            val_dense_vecs = np.zeros((1, 2*len(var) + 3, len(context)), dtype=np.float32)
        else:
            val_dense_vecs = np.zeros((1, len(var) + 3, len(context)), dtype=np.float32)
        
        for i, sid in enumerate(context):
            vardat = station_targets.loc[sid]
            if var[0]=='WS':
                val_dense_vecs[0,0,i] = vardat['UX']
                val_dense_vecs[0,1,i] = vardat['VY']
                jj = 2
            else:
                val_dense_vecs[0,0,i] = vardat[var[0]]
                jj = 1
            
            val_dense_vecs[0,jj  ,i] = (vardat.ALTITUDE - nm.s_means.elev) / nm.s_stds.elev
            val_dense_vecs[0,jj+1,i] = vardat.LATITUDE / nm.lat_norm
            val_dense_vecs[0,jj+2,i] = vardat.LONGITUDE / nm.lon_norm
            
            if var[0]=='SWIN' or var[0]=='LWIN' or var[0]=='PRECIP':
                val_dense_vecs[0,jj+3,i] = fine_input[int(vardat.sub_y), int(vardat.sub_x), self.fine_variable_order.index('cloud_cover')]
                cloud_vec.append(val_dense_vecs[0,jj+3,i])
                             
            if var[0]=='SWIN':
                
                val_dense_vecs[0,jj+4,i] = fine_input[int(vardat.sub_y), int(vardat.sub_x), self.fine_variable_order.index('shade_map')]
                shade_vec.append(val_dense_vecs[0,jj+4,i])
                
                val_dense_vecs[0,jj+5,i] = fine_input[int(vardat.sub_y), int(vardat.sub_x), self.fine_variable_order.index('illumination_map')]
                illum_vec.append(val_dense_vecs[0,jj+5,i])                

        if var[0]=='SWIN' or var[0]=='LWIN' or var[0]=='PRECIP':
            station_targets = station_targets.assign(cloud_cover = np.nan)
            station_targets.loc[context, 'cloud_cover'] = cloud_vec
        if var[0]=='SWIN':
            station_targets = station_targets.assign(shade_map = np.nan)
            station_targets.loc[context, 'shade_map'] = shade_vec
            station_targets = station_targets.assign(illumination_map = np.nan)
            station_targets.loc[context, 'illumination_map'] = illum_vec
            
        if False:
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
        constraint = {}
        if 'TA' in var:
            c = self.t_1km_elev.isel(time=it, y=y_inds, x=x_inds)
            c.values = (c.values - 273.15 - nm.temp_mu) / nm.temp_sd
            constraint['TA'] = c.values.copy()
        if 'PA' in var:
            c = self.p_1km_elev.isel(time=it, y=y_inds, x=x_inds)
            c.values = (c.values - nm.p_mu) / nm.p_sd
            constraint['PA'] = c.values.copy()
        if 'SWIN' in var:
            ## Shortwave radiation, time of day dependent
            self.Sw_1km = partition_interp_solar_radiation(
                self.parent_pixels['hourly'][self.var_name_map.loc['SWIN'].coarse][it,:,:] * nm.swin_norm, # de-nornmalise!
                self.fine_grid,
                self.sp,
                self.parent_pixels['hourly'][self.var_name_map.loc['PA'].coarse][it,:,:] * nm.p_sd + nm.p_mu, # de-nornmalise!
                self.p_1km_elev[it,:,:],
                self.height_grid,
                self.shading_ds.isel(time=it),                
                self.scale,
                self.skyview_map
            )
            self.Sw_1km = self.Sw_1km.clip(min=0) # get rid of negative swin
            
            c = self.Sw_1km.isel(y=y_inds, x=x_inds)
            c.values = c.values / nm.swin_norm
            constraint['SWIN'] = c.values.copy()
        if 'LWIN' in var:
            c = self.Lw_1km.isel(time=it, y=y_inds, x=x_inds)
            c.values = (c.values - nm.lwin_mu) / nm.lwin_sd
            constraint['LWIN'] = c.values.copy()
        if 'WS' in var:
            '''
            Grab both wind components here and stack them on channel dim
            '''
            # c = self.ws_1km_interp.isel(time=it, y=y_inds, x=x_inds)
            # c.values = (c.values - nm.ws_mu) / nm.ws_sd
            # constraint['WS'] = c.values.copy()
            c_x = self.ux_1km_interp.isel(time=it, y=y_inds, x=x_inds)            
            c_x.values = c_x.values / nm.ws_sd
            c_y = self.vy_1km_interp.isel(time=it, y=y_inds, x=x_inds)
            c_y.values = c_y.values / nm.ws_sd
            constraint['UX'] = c_x.values.copy()
            constraint['VY'] = c_y.values.copy()
        if 'RH' in var:
            c = self.rh_1km_interp.isel(time=it, y=y_inds, x=x_inds)
            c.values = (c.values - nm.rh_mu) / nm.rh_sd
            constraint['RH'] = c.values.copy()
        if 'PRECIP' in var:
            ## LOAD CEH-GEAR data for the date/time, or do this prior
            ## to share dataset between same-day timepoints            
            fine_sub = self.fine_grid.isel(y=y_inds, x=x_inds)         
            c = (self.gear.sel(y=fine_sub.y, x=fine_sub.x)
                .isel(time=it)
                .rainfall_amount
            )
            # infill sea NaNs with ERA5 precip interp, might need to transform precip interp to mm
            sea_mask = np.isnan(c.values)
            c.values[sea_mask] = (self.precip_1km_interp
                .isel(y=y_inds, x=x_inds, time=it)
                .values[sea_mask]
            ) * 3600 # kg m2 s-1 to mm
            c.values = c.values / nm.precip_norm
            c = c.clip(min=0) # get rid of negative rain
            constraint['PRECIP'] = c.values.copy()
                        
        return [torch.from_numpy(constraint[v]).to(torch.float32) for v in constraint.keys()]

    def get_sample(self, var, batch_type='train',
                   context_frac=None, 
                   timestep='hourly',
                   sample_xyt=True,
                   ix=None, iy=None, it=None,
                   return_constraints=True,
                   SID=None, parent_pixel_id=None):

        if sample_xyt:
            ## sample a dim_l x dim_l tile
            ix, iy, it = self.sample_xyt(batch_type=batch_type,
                                         timestep=timestep,
                                         SID=SID,
                                         parent_pixel_id=parent_pixel_id,
                                         it=it)
        
        if it is None: it = 0 # hack for daily
        
        ## load input data
        (coarse_input, fine_input, subdat,
            x_inds, y_inds, lat_grid, lon_grid, timestamp) = self.get_input_data(
            var, ix, iy, it, timestep=timestep
        )
        
        ## get station targets within the dim_l x dim_l tile        
        (station_targets, station_npts, 
            station_data, context_locations) = self.get_station_targets(
            subdat, x_inds, y_inds, timestamp, var,
            np.stack([lat_grid, lon_grid], axis=0),
            fine_input,
            batch_type=batch_type,
            context_frac=context_frac,            
            timestep=timestep            
        )
        
        constraints = []
        if return_constraints:
            if timestep=='daily':
                ## grab subgrid chess constraints                
                for v in list(self.var_name_map.loc[var].chess):
                    constraints.append(torch.from_numpy(
                        self.chess_dat.isel(x=x_inds, y=y_inds)[v].values)
                        .to(torch.float32)
                    )
            else:
                ## use physically reasoned constraints
                constraints = self.get_constraints(x_inds, y_inds, it, var)               
        else:
            constraints.append(torch.zeros((0,0)).to(torch.float32))
        constraints = torch.stack(constraints, dim=-1)
        
        ## capture sample description
        sample_meta = {
            'timestamp':self.td,
            'x_inds':x_inds,
            'y_inds':y_inds,
            't_ind':it,
            'coarse_xl':ix,
            'coarse_yt':iy,
            'timestep':timestep,
            'fine_var_order':self.fine_variable_order,
            'coarse_var_order':self.coarse_variable_order,
            'context_var_order':self.context_variable_order
        }
                
        return {
            'in_coarse':coarse_input,
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
            if ('TA' in var) or ('PA' in var) or ('SWIN' in var) or ('LWIN' in var) or ('RH' in var):
                ## Air Temp
                # reduce to sea level with lapse rate of -0.0065 K/m
                lapse_val = -0.0065
                self.parent_pixels['hourly']['t_sealevel'] = (
                    ['time', 'y', 'x'],
                    self.parent_pixels['hourly'][tn].values
                )
                self.parent_pixels['hourly']['t_sealevel'] = (
                    self.parent_pixels['hourly'].t_sealevel - 
                    self.coarse_grid.elev * lapse_val
                )

                # interpolate from 28km to 1km
                t_sealevel_interp = interp_to_grid(self.parent_pixels['hourly']['t_sealevel'], self.fine_grid, coords=['lat', 'lon'])                
                t_sealevel_interp = t_sealevel_interp.transpose('time', 'y', 'x')

                # adjust to the 1km elevation using same lapse rate
                self.t_1km_elev = t_sealevel_interp + self.height_grid['elev'] * lapse_val
                self.t_1km_elev = self.t_1km_elev.transpose('time', 'y', 'x')
                self.t_1km_elev.load()
                self.t_1km_elev = reflect_pad_nans(self.t_1km_elev.copy())
                
                # rescale to counteract errors in physical downscaling assumptions
                self.t_1km_elev = scale_by_coarse_data(
                    self.t_1km_elev.copy(),
                    self.parent_pixels['hourly'][tn],
                    self.fine_grid,
                    self.scale
                )
            
            if ('PA' in var) or ('SWIN' in var) or ('RH' in var):
                ## Air Pressure:
                # integral of hypsometric equation using the 1km Air Temp?
                T_av = 0.5*(t_sealevel_interp + self.t_1km_elev) # K
                p_1 = 1013 # hPa, standard sea level pressure value
                R = 287 # J/kg·K = kg.m2/s2/kg.K = m2/s2.K  # specific gas constant of dry air
                g = 9.81 # m/s2 
                self.p_1km_elev = p_1 * np.exp(-g * self.height_grid['elev'] / (R * T_av))
                self.p_1km_elev = self.p_1km_elev.transpose('time', 'y', 'x')
                self.p_1km_elev.load()
                self.p_1km_elev = reflect_pad_nans(self.p_1km_elev.copy())
                del(T_av)
                del(t_sealevel_interp)
                
                # rescale to counteract errors in physical downscaling assumptions
                self.p_1km_elev = scale_by_coarse_data(
                    self.p_1km_elev.copy(),
                    self.parent_pixels['hourly'][self.var_name_map.loc['PA'].coarse]/100., # Pa -> hPa
                    self.fine_grid,
                    self.scale
                )
            
            if ('RH' in var) or ('LWIN' in var):
                ## Relative Humidity
                # Assumed constant with respect to elevation, so can be simply 
                # interpolated from 28km to 1km using a bicubic spline.
                self.rh_1km_interp = interp_to_grid(
                    self.parent_pixels['hourly'][self.var_name_map.loc['RH'].coarse],
                    self.fine_grid, coords=['lat', 'lon'])                
                self.rh_1km_interp = self.rh_1km_interp.transpose('time', 'y', 'x')
                self.rh_1km_interp.load()
                self.rh_1km_interp = reflect_pad_nans(self.rh_1km_interp.copy())
                
                # rescale to counteract errors in physical downscaling assumptions
                self.rh_1km_interp = scale_by_coarse_data(
                    self.rh_1km_interp.copy(),
                    self.parent_pixels['hourly'][self.var_name_map.loc['RH'].coarse],
                    self.fine_grid,
                    self.scale
                )
            
            if 'SWIN' in var:
                ## Shortwave radiation, relies on time-specific
                ## shade map and solar position                
                pass
                
            if 'LWIN' in var:
                ## Saturated and Actual vapour pressure
                # using Buck equation                
                es_28km = 6.1121 * np.exp(
                    (18.678 - (self.parent_pixels['hourly'][tn] - 273.15) / 234.5) *
                    ((self.parent_pixels['hourly'][tn] - 273.15) / 
                    (257.14 + (self.parent_pixels['hourly'][tn] - 273.15)))
                )
                es_1km = 6.1121 * np.exp(
                    (18.678 - (self.t_1km_elev - 273.15) / 234.5) *
                    ((self.t_1km_elev - 273.15) / (257.14 + (self.t_1km_elev - 273.15)))
                )
                # from RH
                ea_28km = (self.parent_pixels['hourly'][
                    self.var_name_map.loc['RH'].coarse] / 100) * es_28km
                ea_1km = (self.rh_1km_interp/100) * es_1km                
                
                ## Longwave radiation and Emissivity
                # method from Real-time and retrospective forcing in the North American Land
                #   Data Assimilation System (NLDAS) project, Brian A. Cosgrove,
                # which borrows model from
                # Satterlund, Donald R., An improved equation for estimating long‐wave radiation from the atmosphere
                # and I have taken calibrated constants from  M. Li, Y. Jiang, C.F.M. Coimbra
                #   On the determination of atmospheric longwave irradiance under all-sky conditions
                #   Sol. Energy, 144 (2017), pp. 40-48
                emms_28km = 1.02 * (1 - np.exp(-ea_28km*(
                    self.parent_pixels['hourly'][self.var_name_map.loc['TA'].coarse]/1564.94)))                
                emms_28km_interp = interp_to_grid(emms_28km, self.fine_grid, coords=['lat', 'lon'])
                
                emms_1km = 1.02 * (1 - np.exp(-ea_1km*(self.t_1km_elev/1564.94)))
                emms_ratio = emms_1km / emms_28km_interp
                
                Lw_28km_interp = interp_to_grid(
                    self.parent_pixels['hourly'][self.var_name_map.loc['LWIN'].coarse],
                    self.fine_grid, coords=['lat', 'lon']
                )
                    
                use_temp_ratio = False
                if use_temp_ratio:
                    # if we want to account for the temperature dependence here
                    # (though really it should be higher up temperature 
                    # rather than surface temp?)                    
                    t_28km_interp = interp_to_grid(
                        self.parent_pixels['hourly'][self.var_name_map.loc['TA'].coarse],
                        self.fine_grid, coords=['lat', 'lon']
                    )
                    
                    T_ratio = self.t_1km_elev / t_28km_interp                
                    self.Lw_1km = emms_ratio * (T_ratio)**4 * Lw_28km_interp
                    del(t_28km_interp)
                    del(T_ratio)
                else:
                    self.Lw_1km = emms_ratio * Lw_28km_interp        

                self.Lw_1km = self.Lw_1km.transpose('time', 'y', 'x')
                self.Lw_1km.load()
                self.Lw_1km = reflect_pad_nans(self.Lw_1km.copy())
                
                # rescale to counteract errors in physical downscaling assumptions
                self.Lw_1km = scale_by_coarse_data(
                    self.Lw_1km.copy(),
                    self.parent_pixels['hourly'][self.var_name_map.loc['LWIN'].coarse],
                    self.fine_grid,
                    self.scale
                )
                            
            if 'WS' in var:
                ## Wind speed. Simply interpolate this....
                '''
                Interpolate both components, u10 and v10!
                '''         
                # self.ws_1km_interp = interp_to_grid(
                    # self.parent_pixels['hourly'][self.var_name_map.loc['WS'].coarse],
                    # self.fine_grid, coords=['lat', 'lon']
                # )
                # self.ws_1km_interp = self.ws_1km_interp.transpose('time', 'y', 'x')
                # self.ws_1km_interp.load()
                # self.ws_1km_interp = reflect_pad_nans(self.ws_1km_interp.copy())
                
                # # rescale to counteract errors in physical downscaling assumptions
                # self.ws_1km_interp = scale_by_coarse_data(
                    # self.ws_1km_interp.copy(),
                    # self.parent_pixels['hourly'][self.var_name_map.loc['WS'].coarse],
                    # self.fine_grid,
                    # self.scale
                # )
                
                self.ux_1km_interp = interp_to_grid(
                    self.parent_pixels['hourly'][self.var_name_map.loc['UX'].coarse],
                    self.fine_grid, coords=['lat', 'lon']
                )
                self.ux_1km_interp = self.ux_1km_interp.transpose('time', 'y', 'x')
                self.ux_1km_interp.load()
                self.ux_1km_interp = reflect_pad_nans(self.ux_1km_interp.copy())
                
                # rescale to counteract errors in physical downscaling assumptions
                self.ux_1km_interp = scale_by_coarse_data(
                    self.ux_1km_interp.copy(),
                    self.parent_pixels['hourly'][self.var_name_map.loc['UX'].coarse],
                    self.fine_grid,
                    self.scale
                )
                
                self.vy_1km_interp = interp_to_grid(
                    self.parent_pixels['hourly'][self.var_name_map.loc['VY'].coarse],
                    self.fine_grid, coords=['lat', 'lon']
                )
                self.vy_1km_interp = self.vy_1km_interp.transpose('time', 'y', 'x')
                self.vy_1km_interp.load()
                self.vy_1km_interp = reflect_pad_nans(self.vy_1km_interp.copy())
                
                # rescale to counteract errors in physical downscaling assumptions
                self.vy_1km_interp = scale_by_coarse_data(
                    self.vy_1km_interp.copy(),
                    self.parent_pixels['hourly'][self.var_name_map.loc['VY'].coarse],
                    self.fine_grid,
                    self.scale
                )
            
            if 'PRECIP' in var:
                # interpolate ERA5 to fill in the NaNs in 
                self.precip_1km_interp = interp_to_grid(
                    self.parent_pixels['hourly'][self.var_name_map.loc['PRECIP'].coarse],
                    self.fine_grid, coords=['lat', 'lon']
                )
                self.precip_1km_interp = self.precip_1km_interp.transpose('time', 'y', 'x')
                self.precip_1km_interp.load()
                self.precip_1km_interp = reflect_pad_nans(self.precip_1km_interp.copy())                
                
                # rescale to counteract errors in physical downscaling assumptions
                # Do we want this for the precip? might not make sense for GEAR field
                # self.precip_1km_interp = scale_by_coarse_data(
                    # self.precip_1km_interp.copy(),
                    # self.parent_pixels['hourly'][self.var_name_map.loc['PRECIP'].coarse],
                    # self.fine_grid,
                    # self.scale
                # )
                

    def prepare_era5_pixels(self, var, batch_tsteps='hourly', constraints=True):
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
        
        ## load other large grids that we only want to load once per batch
        if 'SWIN' in var:
            # terrain shading used in hourly AND daily
            doy = self.td.day_of_year - 1 # zero indexed
            # self.shading_array = np.load(
                # nz_train_path + f'/terrain_shading/shading_mask_day_{doy}_merged_to_7999.npy')
            # self.sea_shading_array = np.load(
                # nz_train_path + f'/terrain_shading/sea_shading_mask_day_{doy}_merged_to_7999.npy')
            self.shading_ds = xr.open_dataset(nz_train_path + f'./terrain_shading/shading_mask_{doy}.nc')
            self.shading_ds = self.shading_ds.shading
                
        if ('SWIN' in var) or ('LWIN' in var) or ('PRECIP' in var) :
            # cloud cover
            tcc = xr.open_dataset(era5_fldr + f'/tcc/era5_{self.td.year}{zeropad_strint(self.td.month)}{zeropad_strint(self.td.day)}_tcc.nc')
            self.tcc_1km = interp_to_grid(tcc.tcc, self.fine_grid, coords=['lat', 'lon'])
            self.tcc_1km = reflect_pad_nans(self.tcc_1km.copy())
        
        if ('PRECIP' in var):
            self.gear = xr.open_dataset(precip_fldr + f'/{self.td.year}/CEH-GEAR-1hr-v2_{self.td.year}{zeropad_strint(self.td.month)}.nc')
            # label centre of pixel
            self.gear['x'] = self.gear.x + 500
            self.gear['y'] = self.gear.y + 500
            # subset to domain
            self.gear = self.gear.sel(
                x = np.intersect1d(self.gear.x, self.fine_grid.x),
                y = np.intersect1d(self.gear.y, self.fine_grid.y)
            )
            # subset to day
            day_inds = np.where(pd.to_datetime(self.gear.time.values, utc=True).day == self.td.day)[0]
            self.gear = self.gear.isel(time = day_inds)
            
        ## change units
        self.parent_pixels[tstep]['t2m'].values -= 273.15 # K -> Celsius 
        self.parent_pixels[tstep]['d2m'].values -= 273.15 # K -> Celsius
        self.parent_pixels[tstep]['sp'].values /= 100.    # Pa -> hPa
        self.parent_pixels[tstep]['mtpr'].values *= 3600. # kg m2 s-1 -> mm in hour
        
        #self.parent_pixels[tstep] = self.parent_pixels[tstep].drop(['u10', 'v10', 'd2m'])
        self.parent_pixels[tstep] = self.parent_pixels[tstep].drop(['d2m'])
        
        ## normalise ERA5        
        self.parent_pixels[tstep]['t2m'].values = (self.parent_pixels[tstep]['t2m'].values - nm.temp_mu) / nm.temp_sd        
        self.parent_pixels[tstep]['sp'].values = (self.parent_pixels[tstep]['sp'].values - nm.p_mu) / nm.p_sd        
        self.parent_pixels[tstep]['msdwlwrf'].values = (self.parent_pixels[tstep]['msdwlwrf'].values - nm.lwin_mu) / nm.lwin_sd        
        self.parent_pixels[tstep]['msdwswrf'].values = self.parent_pixels[tstep]['msdwswrf'].values / nm.swin_norm
        self.parent_pixels[tstep]['ws'].values = (self.parent_pixels[tstep]['ws'].values - nm.ws_mu) / nm.ws_sd
        self.parent_pixels[tstep]['u10'].values = (self.parent_pixels[tstep]['u10'].values) / nm.ws_sd
        self.parent_pixels[tstep]['v10'].values = (self.parent_pixels[tstep]['v10'].values) / nm.ws_sd
        self.parent_pixels[tstep]['rh'].values = (self.parent_pixels[tstep]['rh'].values - nm.rh_mu) / nm.rh_sd
        self.parent_pixels[tstep]['mtpr'].values = self.parent_pixels[tstep]['mtpr'].values / nm.precip_norm
        
        self.parent_pixels[tstep]['msdwswrf'] = self.parent_pixels[tstep]['msdwswrf'].clip(min=0)
        self.parent_pixels[tstep]['mtpr'] = self.parent_pixels[tstep]['mtpr'].clip(min=0)
        
        ## drop variables we don't need
        self.era5_var = list(self.var_name_map.loc[var].coarse)
        to_drop = list(self.parent_pixels[tstep].keys())
        for v in self.era5_var: to_drop.remove(v)
        #to_drop.remove('pixel_id') # not in era5 data anymore, moved to self.coarse_grid
        if 'RH' in var:
            for exvar in self.rh_extra_met_vars:
                cvar = self.var_name_map.loc[exvar].coarse
                if cvar in to_drop:
                    to_drop.remove(cvar)
        if 'LWIN' in var:
            for exvar in self.lwin_extra_met_vars:
                cvar = self.var_name_map.loc[exvar].coarse
                if cvar in to_drop:
                    to_drop.remove(cvar)
        if 'SWIN' in var:
            for exvar in self.swin_extra_met_vars:
                cvar = self.var_name_map.loc[exvar].coarse
                if cvar in to_drop:
                    to_drop.remove(cvar)
        if 'PRECIP' in var:
            for exvar in self.precip_extra_met_vars:
                cvar = self.var_name_map.loc[exvar].coarse
                if cvar in to_drop:
                    to_drop.remove(cvar)
        if 'WS' in var:
            to_drop.remove('u10')
            to_drop.remove('v10')
            
        self.parent_pixels[tstep] = self.parent_pixels[tstep].drop(to_drop)
        
        # load data
        self.parent_pixels[tstep].load()
        
        if batch_tsteps=='mix':
            # do time-averaging for daily AFTER processing if mixed batch            
            self.parent_pixels['daily'] = self.parent_pixels['hourly'].mean('time')            

    def prepare_fine_met_inputs(self, var, batch_tsteps):
        fmi = False
        extra_met_vars = []
        if 'RH' in var:
            extra_met_vars += self.rh_extra_met_vars
            fmi = True
        if 'LWIN' in var:
            extra_met_vars += self.lwin_extra_met_vars
            fmi = True
        if 'SWIN' in var:
            extra_met_vars += self.swin_extra_met_vars
            fmi = True
        if 'PRECIP' in var:
            extra_met_vars += self.precip_extra_met_vars
            fmi = True
        extra_met_vars = list(set(extra_met_vars))
        if fmi:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if batch_tsteps!='hourly': # cover daily and mix
                    ## interpolate ERA5 extra vars to 1km and replace
                    ## NaN sea pixels of chess vars with ERA5 to use as fine input
                    era5_met = self.parent_pixels['daily'].copy()#.drop(['pixel_id'])                    
                    era5_met = interp_to_grid(era5_met, self.fine_grid, coords=['lat', 'lon'])
                    for vv in list(era5_met.keys()):
                        era5_met[vv] = reflect_pad_nans(era5_met[vv].copy())
                    chess_sea_mask = np.isnan(self.chess_dat[self.chess_var[0]].values)
                    for xtrvar in extra_met_vars:
                        self.chess_dat[self.var_name_map.loc[xtrvar].chess].values[chess_sea_mask] =\
                            era5_met[self.var_name_map.loc[xtrvar].coarse].values[chess_sea_mask]
                if batch_tsteps!='daily': # covers hourly and mix
                    ## use hi-res constraints for land pixels and replace
                    ## NaN sea pixels with ERA5 to use as fine input                    
                    self.met_fine = self.parent_pixels['hourly'].copy()#.drop(['pixel_id'])                    
                    self.met_fine = interp_to_grid(self.met_fine, self.fine_grid, coords=['lat', 'lon'])
                    for vv in list(self.met_fine.keys()):
                        self.met_fine[vv] = reflect_pad_nans(self.met_fine[vv].copy())
                    # replace land tiles with normalised hi res constraints
                    for xtrvar in extra_met_vars:
                        if xtrvar=='TA':
                            land_mask = ~np.isnan(self.t_1km_elev.values)
                            self.met_fine[self.var_name_map.loc[xtrvar].coarse].values[land_mask] =\
                                (self.t_1km_elev.values[land_mask] - 273.15 - nm.temp_mu) / nm.temp_sd                                
                        if xtrvar=='PA':
                            land_mask = ~np.isnan(self.p_1km_elev.values)
                            self.met_fine[self.var_name_map.loc[xtrvar].coarse].values[land_mask] =\
                                (self.p_1km_elev.values[land_mask] - nm.p_mu) / nm.p_sd
                ## then as fine input we use self.met_fine for hourly samples
                ## and self.chess_dat for daily samples
    
    def calculate_day_av_solar_vars(self, var):
        if (('SWIN' in var) or ('LWIN' in var)):
            self.sol_azi = self.fine_grid.lat.copy()
            self.sol_alt = self.fine_grid.lat.copy()
            self.sun_hrs = self.fine_grid.lat.copy()            
            self.sol_azi.values.fill(0)
            self.sol_alt.values.fill(0)
            self.sun_hrs.values.fill(0)
            if 'SWIN' in var:
                self.sol_ilum = self.fine_grid.lat.copy()
                self.av_shade_map = self.fine_grid.lat.copy()
                self.sol_ilum.values.fill(0)
                self.av_shade_map.values.fill(1)
                sun_hours = []
            for tt in range(0, 24):
                sp = SolarPosition(self.td + datetime.timedelta(hours=tt) - datetime.timedelta(minutes=30), timezone=0) # utc
                sp.calc_solar_angles(self.fine_grid.lat, self.fine_grid.lon)
                dayhour = (sp.solar_elevation.values > 0).astype(np.int32)
                # only include daytime hours in the angle/illum averages
                self.sun_hrs.values += dayhour
                self.sol_azi.values += sp.solar_azimuth_angle.values * dayhour
                self.sol_alt.values += sp.solar_elevation.values * dayhour
                if 'SWIN' in var:
                    illum_map = calculate_illumination_map(sp, self.fine_grid, self.height_grid)
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
            if 'SWIN' in var:
                # we don't want a smoothed illumination map, though...
                self.sol_ilum.values /= self.sun_hrs.values
                # self.av_shade_map.values[self.fine_grid.landfrac.values==1] = \
                    # -(np.mean(self.shading_array[sun_hours,:], axis=0) - 1)
                # self.av_shade_map.values[self.fine_grid.landfrac.values==0] = \
                    # -(np.mean(self.sea_shading_array[sun_hours,:], axis=0) - 1)
                self.av_shade_map.values = self.shading_array.isel(time=sun_hours).mean(dim='time').values
        else:
            pass
    
    def get_batch(self, var, batch_size=1, batch_type='train',
                  context_frac=None, p_hourly=1,
                  date=None, parent_pixel_ids=[], times=[]):
        ''' 
        when using date, parent_pixel_ids, times to select batches,
        date must be a datetime.date object
        parent_pixel_ids must be list of integers
        times must be a list of integers in {0,23}
        '''
        if not date is None: date_string = f'{date.year}{zeropad_strint(date.month)}{zeropad_strint(date.day)}'
        if parent_pixel_ids is None: parent_pixel_ids = []
        if times is None: times = []
        if type(parent_pixel_ids)==int: parent_pixel_ids = [parent_pixel_ids]
        if type(times)==int: times = [times]        
        if type(var)==str: var = [var]
        
        self.parent_pixels = {}
        if date is None:
            self.parent_pixels['hourly'] = self.read_parent_pixel_day(batch_type=batch_type)
        else:                
            self.parent_pixels['hourly'] = self.read_parent_pixel_day(batch_type=batch_type,
                                                                      date_string=date_string)
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

        if (batch_tsteps!='hourly'):
            ## load chess data (don't normalise in case using in constraint preparation)
            self.chess_var = list(self.var_name_map.loc[var].chess)
            self.chess_dat = load_process_chess(self.td.year, self.td.month, self.td.day,
                                                self.chess_var, normalise=False)

        ## process and normalise ERA5 parent_pixels
        self.prepare_era5_pixels(var, batch_tsteps=batch_tsteps)

        if (batch_tsteps!='hourly'):
            ## now normalise if we have loaded chess data
            self.chess_dat = normalise_chess_data(self.chess_dat.copy(),
                                                  self.chess_var)

        ## get additional fine scale met vars for input
        self.prepare_fine_met_inputs(var, batch_tsteps)
        
        if batch_tsteps!='hourly': # cover daily and mix
            # pool to scale and fill in the parent pixels
            for v in var:
                cv = self.var_name_map.loc[v].chess
                ev = self.var_name_map.loc[v].coarse
                chess_pooled = pooling(self.chess_dat[cv].values, (self.scale, self.scale), method='mean')
                chess_mask = ~np.isnan(chess_pooled)
                self.parent_pixels['daily'][ev].values[chess_mask] = chess_pooled[chess_mask]   
            
            # solar values for daily timestep
            self.calculate_day_av_solar_vars(var)
        
        coarse_inputs = []
        fine_inputs = []
        station_targets = []
        station_data = []
        station_num_obs = []
        constraint_targets = []
        context_locations = []
        batch_metadata = []
        for b in range(batch_size):
            # try to select the parent pixel id and hour
            if b <= (len(parent_pixel_ids)-1):
                ppid = parent_pixel_ids[b]
            else:
                ppid = None
            if b <= (len(times)-1): # assuming entire batch is hourly samples
                _it = times[b]
            else:
                _it = None
            
            # generate batch from parent pixels
            sample = self.get_sample(var,
                                     batch_type=batch_type,
                                     context_frac=context_frac,
                                     timestep=batch_timesteps[b],
                                     parent_pixel_id=ppid,
                                     it=_it)
           
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
        
        # output tensors as (B, C, Y, X)
        return {'coarse_inputs':coarse_inputs.permute(0,3,1,2),
                'fine_inputs':fine_inputs.permute(0,3,1,2),
                'constraints':constraint_targets.permute(0,3,1,2),
                'station_targets':station_targets,
                'station_data':station_data,
                'station_num_obs':station_num_obs,
                'context_locations':context_locations,          
                'batch_metadata':batch_metadata
                }    

    def get_prefetch_data(self, var, batch_type='train',                          
                          date=None, times=[]):
        ''' 
        when using date, parent_pixel_ids, times to select batches,
        date must be a datetime.date object
        parent_pixel_ids must be list of integers
        times must be a list of integers in {0,23}
        '''
        if not date is None: date_string = f'{date.year}{zeropad_strint(date.month)}{zeropad_strint(date.day)}'
        if times is None: times = []
        if type(times)==int: times = [times]        
        if type(var)==str: var = [var]
        
        self.parent_pixels = {}
        self.parent_pixels['hourly'] = self.read_parent_pixel_day(batch_type=batch_type,
                                                                  date_string=date_string)
        self.td = pd.to_datetime(self.parent_pixels['hourly'].time[0].values, utc=True)        
        batch_tsteps = 'hourly'

        ## process and normalise ERA5 parent_pixels
        self.prepare_era5_pixels(var, batch_tsteps=batch_tsteps)
        
        ## get additional fine scale met vars for input
        self.prepare_fine_met_inputs(var, batch_tsteps)
        
        met_fine_path = nz_train_path + '/met_fine_prefetch/'
        constraints_path = nz_train_path + '/constraints_prefetch/'
        Path(met_fine_path).mkdir(exist_ok=True, parents=True)
        Path(constraints_path).mkdir(exist_ok=True, parents=True)
        
        # output self.met_fine if it exists
        
        ## get full-map constraints
        for it in times:
            if 'TA' in var:
                c = self.t_1km_elev.isel(time=it)
                c.values = (c.values - 273.15 - nm.temp_mu) / nm.temp_sd
                c.name = var[0]
                c.to_netcdf(constraints_path + f'/{var[0]}_constraint_{date_string}_{it}.nc',
                            encoding = {var[0]: {"zlib": True, "complevel": 9, "dtype": "float32"}})
            if 'PA' in var:
                c = self.p_1km_elev.isel(time=it)
                c.values = (c.values - nm.p_mu) / nm.p_sd
                constraint['PA'] = c.values.copy()
            if 'SWIN' in var:
                ## Shortwave radiation, time of day dependent
                self.Sw_1km = partition_interp_solar_radiation(
                    self.parent_pixels['hourly'][self.var_name_map.loc['SWIN'].coarse][it,:,:] * nm.swin_norm, # de-nornmalise!
                    self.fine_grid,
                    self.sp,
                    self.parent_pixels['hourly'][self.var_name_map.loc['PA'].coarse][it,:,:] * nm.p_sd + nm.p_mu, # de-nornmalise!
                    self.p_1km_elev[it,:,:],
                    self.height_grid,
                    self.shading_ds.isel(time=it),                
                    self.scale,
                    self.skyview_map
                )
                self.Sw_1km = self.Sw_1km.clip(min=0) # get rid of negative swin
                
                c = self.Sw_1km
                c.values = c.values / nm.swin_norm
                constraint['SWIN'] = c.values.copy()
            if 'LWIN' in var:
                c = self.Lw_1km.isel(time=it)
                c.values = (c.values - nm.lwin_mu) / nm.lwin_sd
                constraint['LWIN'] = c.values.copy()
            if 'WS' in var:            
                c_x = self.ux_1km_interp.isel(time=it)
                c_x.values = c_x.values / nm.ws_sd
                c_y = self.vy_1km_interp.isel(time=it)
                c_y.values = c_y.values / nm.ws_sd
                constraint['UX'] = c_x.values.copy()
                constraint['VY'] = c_y.values.copy()
            if 'RH' in var:
                c = self.rh_1km_interp.isel(time=it)
                c.values = (c.values - nm.rh_mu) / nm.rh_sd
                constraint['RH'] = c.values.copy()
            if 'PRECIP' in var:
                ## LOAD CEH-GEAR data for the date/time, or do this prior
                ## to share dataset between same-day timepoints            
                fine_sub = self.fine_grid
                c = (self.gear.isel(time=it)
                    .rainfall_amount
                )
                # infill sea NaNs with ERA5 precip interp, might need to transform precip interp to mm
                sea_mask = np.isnan(c.values)
                c.values[sea_mask] = (self.precip_1km_interp
                    .isel(time=it)
                    .values[sea_mask]
                ) * 3600 # kg m2 s-1 to mm
                c.values = c.values / nm.precip_norm
                c = c.clip(min=0) # get rid of negative rain
                constraint['PRECIP'] = c.values.copy()        
        
    def get_all_space(self, var, batch_type='run',
                      context_frac=None, date_string=None, it=None,
                      timestep='hourly', tile=False, min_overlap=1,
                      return_constraints=False):
        
        if type(var)==str: var = [var]
        self.parent_pixels = {}
        
        if date_string is None:            
            self.parent_pixels['hourly'] = self.read_parent_pixel_day(batch_type=batch_type)            
            self.td = pd.to_datetime(self.parent_pixels[timestep].time[0].values, utc=True)
        else:
            self.parent_pixels['hourly'] = self.read_parent_pixel_day(batch_type=batch_type,
                                                                      date_string=date_string)               
            self.td = pd.to_datetime(date_string, format='%Y%m%d', utc=True)        
        
        if (timestep!='hourly'):
            ## load chess data (don't normalise in case using in constraint preparation)
            self.chess_var = list(self.var_name_map.loc[var].chess)
            self.chess_dat = load_process_chess(self.td.year, self.td.month, self.td.day,
                                                self.chess_var, normalise=False)

        ## process and normalise ERA5 parent_pixels
        self.prepare_era5_pixels(var, batch_tsteps=timestep)

        if (timestep!='hourly'):
            ## now normalise if we have loaded chess data
            self.chess_dat = normalise_chess_data(self.chess_dat.copy(),
                                                  self.chess_var)

        ## get additional fine scale met vars for input
        self.prepare_fine_met_inputs(var, timestep)
        
        if timestep!='hourly': # cover daily and mix
            # pool to scale and fill in the parent pixels
            for v in var:
                cv = self.var_name_map.loc[v].chess
                ev = self.var_name_map.loc[v].coarse
                chess_pooled = pooling(self.chess_dat[cv].values, (self.scale, self.scale), method='mean')
                chess_mask = ~np.isnan(chess_pooled)
                self.parent_pixels['daily'][ev].values[chess_mask] = chess_pooled[chess_mask]   
            
            # solar values for daily timestep
            self.calculate_day_av_solar_vars(var)
        
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
                
                sample = self.get_sample(var,
                                         batch_type=batch_type,
                                         context_frac=context_frac, 
                                         timestep=timestep,
                                         sample_xyt=False,
                                         ix=ix, iy=iy, it=it,
                                         return_constraints=return_constraints)
                
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
                        sample = self.get_sample(
                            var, batch_type=batch_type,
                            context_frac=context_frac, 
                            timestep=timestep,
                            sample_xyt=False,
                            ix=ix, iy=iy, it=it,
                            return_constraints=return_constraints
                        )
                                                 
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
        constraint_targets = torch.stack(constraint_targets, dim=0) 
        station_data = [torch.from_numpy(a).type(torch.float32) for a in station_data]
        context_locations = [torch.from_numpy(a).type(torch.float32) for a in context_locations]
        
        # output tensors as (B, C, Y, X)
        return {'coarse_inputs':coarse_inputs.permute(0,3,1,2),
                'fine_inputs':fine_inputs.permute(0,3,1,2),
                'constraints':constraint_targets.permute(0,3,1,2),       
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
        dg = data_generator(precip_sutes=True)
        
        batch_size = 3
        batch_type = 'train'
        p_hourly = 1.0
        #var = ['TA', 'PA', 'SWIN', 'LWIN', 'WS', 'RH', 'PRECIP']
        var = 'WS'
                
        batch = dg.get_batch(var,
                           batch_size=batch_size,
                           batch_type=batch_type,
                           p_hourly=p_hourly)            

        b = 4
        print(batch['batch_metadata'][b]['t_ind'])
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(batch['coarse_inputs'].cpu().numpy()[b,0,::-1,:])
        ax[1].imshow(batch['constraints'].cpu().numpy()[b,0,::-1,:])
        plt.show()
    
        #############################################
        #############################################
        ## create dictionaries of dates/times that contain 
        ## representative samples for a spread of quantikes
        ## for each variable so we can create balanced batches
        dg = data_generator(precip_sites=True)
        all_site_data = pd.DataFrame()
        submeta = dg.site_metadata[['SITE_ID', 'chess_y', 'chess_x', 'parent_pixel_id']]
        submeta.loc[:,'parent_pixel_id'] = submeta['parent_pixel_id'].astype(np.int32)
        sites = list(dg.site_data.keys())
        for sid in sites:
            site_dat = (dg.site_data.pop(sid)
                .dropna(how='all')
                .assign(SITE_ID=sid)
                .reset_index() # [['DATE_TIME', 'SITE_ID', 'PRECIP']]
            )
            site_dat = site_dat.merge(submeta, on='SITE_ID', how='left') 
            all_site_data = pd.concat([all_site_data, site_dat], axis=0)        
        
        # look at quantiles of variables
        # qs = {}
        # qlist = [0, 0.025, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 0.975, 1]
        # for qq in qlist:
            # qs[str(qq)] = all_site_data[list(dg.var_name_map.fine)].quantile(qq)            
        
        # define larger pixels to merge when defining dates/sites to pull from        
        big_pixels = pd.DataFrame()
        bp_i = 0
        for i in range(20):
            for j in range(20):
                bps = np.unique(dg.fine_grid.era5_nbr.values[(i*200):((i+1)*200), # arbitrary 200 big pixel
                                                             (j*200):((j+1)*200)])
                if len(bps)>0:
                    thisdf = pd.DataFrame({'parent_pixel_id':bps}).assign(big_pixel_id = bp_i)
                    big_pixels = pd.concat([big_pixels, thisdf], axis=0)
                    bp_i += 1
        big_pixels = big_pixels.dropna()
        dg.site_metadata = dg.site_metadata.merge(big_pixels, on='parent_pixel_id', how='left')
        
        outdir = nz_train_path + '/var_quantile_samples/'
        
        for v in ['TA', 'PA', 'LWIN', 'WS', 'RH']:
            TT = (all_site_data[['DATE_TIME', 'SITE_ID', 'parent_pixel_id', v]]
                .dropna()
                .sort_values(v)
                .reset_index(drop=True))
            l1 = TT.shape[0] // 40 # delta_q = 0.025
            l2 = TT.shape[0] // 20 # delta_q = 0.05
            l3 = TT.shape[0] // 10 # delta_q = 0.1
            l4 = 3* (TT.shape[0] // 10) # delta_q = 0.3
            lls = [l1, l1, l2, l3, l4, l4, l3, l2, l1, l1]
            l0 = 0
            for i in range(10):
                if i<9:
                    T_bin = TT.iloc[l0:(l0+lls[i])]
                else:
                    T_bin = TT.iloc[l0:]
                pd.to_pickle(T_bin[['DATE_TIME', 'parent_pixel_id']], outdir + f'/{v}_bin_{i}.pkl', compression={'method': 'gzip', 'compresslevel': 5})
                l0 += lls[i]
        
        v = 'SWIN'
        TT = (all_site_data[['DATE_TIME', 'SITE_ID', 'parent_pixel_id', v]]
            .dropna()
            .sort_values(v)
            .reset_index(drop=True))
        TT_e0 = TT[TT[v] == 0]
        TT = TT[TT[v] > 0]
        
        # cut down to manageable number of zeros
        TT_e0 = TT_e0.sample(int(1e6), replace=False)
        pd.to_pickle(TT_e0[['DATE_TIME', 'parent_pixel_id']], outdir + f'/{v}_bin_{0}.pkl', compression={'method': 'gzip', 'compresslevel': 5})
        l1 = TT.shape[0] // 40
        l2 = TT.shape[0] // 20
        l3 = TT.shape[0] // 10
        l4 = 3* (TT.shape[0] // 10)
        lls = [0, l2, l2, l3, l4, l4, l3, l2, l1, l1]
        l0 = 0
        for i in range(1, 10):
            if i<9:
                T_bin = TT.iloc[l0:(l0+lls[i])]
            else:
                T_bin = TT.iloc[l0:]
            pd.to_pickle(T_bin[['DATE_TIME', 'parent_pixel_id']], outdir + f'/{v}_bin_{i}.pkl',
                         compression={'method': 'gzip', 'compresslevel': 5})
            l0 += lls[i]
            
        v = 'PRECIP'
        TT = (all_site_data[['DATE_TIME', 'SITE_ID', 'parent_pixel_id', v]]
            .dropna()
            .sort_values(v)
            .reset_index(drop=True))
        TT_e0 = TT[TT[v] == 0]
        TT = TT[TT[v] > 0]
        
        # cut down to manageable number of zeros
        TT_e0 = TT_e0.sample(int(1e6), replace=False)
        pd.to_pickle(TT_e0[['DATE_TIME', 'parent_pixel_id']], outdir + f'/{v}_bin_{0}.pkl', compression={'method': 'gzip', 'compresslevel': 5})
        # define uneven quantile bins to capture extreme events
        l1 = TT.shape[0] // 40
        l2 = TT.shape[0] // 20
        l3 = TT.shape[0] // 10
        l4 = 3 * (TT.shape[0] // 10)
        lls = [0, l2, l2, l3, l4, l4, l3, l2, l1, l1]
        l0 = 0
        for i in range(1, 10):
            if i<9:
                T_bin = TT.iloc[l0:(l0+lls[i])]
            else:
                T_bin = TT.iloc[l0:]
            pd.to_pickle(T_bin[['DATE_TIME', 'parent_pixel_id']], outdir + f'/{v}_bin_{i}.pkl',
                         compression={'method': 'gzip', 'compresslevel': 5})
            l0 += lls[i]
        
        for vv in ['TA', 'PA', 'SWIN', 'LWIN', 'WS', 'RH', 'PRECIP']:
            for nbin in range(10):
                dat = pd.read_pickle(outdir + f'/{vv}_bin_{nbin}.pkl',
                                     compression={'method': 'gzip', 'compresslevel': 5})
                dat = dat.sort_values('DATE_TIME')
                pd.to_pickle(dat[dat['DATE_TIME'].dt.year.isin(dp.train_years)],
                             outdir + f'/{vv}_bin_{nbin}_train.pkl',
                             compression={'method': 'gzip', 'compresslevel': 5})
                pd.to_pickle(dat[dat['DATE_TIME'].dt.year.isin(dp.val_years)],
                             outdir + f'/{vv}_bin_{nbin}_val.pkl',
                             compression={'method': 'gzip', 'compresslevel': 5})
                pd.to_pickle(dat[dat['DATE_TIME'].dt.year.isin(dp.heldout_years)],
                             outdir + f'/{vv}_bin_{nbin}_test.pkl',
                             compression={'method': 'gzip', 'compresslevel': 5})
        
