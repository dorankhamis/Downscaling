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
from pathlib import Path

from data_classes.met_data import ERA5Data
from data_classes.cosmos_data import CosmosMetaData, CosmosData
from utils import zeropad_strint, find_chess_tile
from params import normalisation as nm

#era5_fldr = '/gws/nopw/j04/hydro_jules/data/uk/driving_data/era5/hourly_single_levels/'
era5_fldr = '/gws/nopw/j04/hydro_jules/data/uk/driving_data/era5/bn_grid/'
precip_fldr = '/badc/ukmo-hadobs/data/insitu/MOHC/HadOBS/HadUK-Grid/v1.1.0.0/1km/rainfall/day/latest/'
midas_fldr = '/home/users/doran/data_dump/MetOffice/midas_data/'
data_dir = '/home/users/doran/data_dump/'
binary_batch_path = '/gws/nopw/j04/hydro_jules/data/uk/downscaling/training_data/'

def read_one_cosmos_site_met(SID, missing_val=-9999.0):
    data = CosmosData(SID)    
    data.read_subhourly()
    data.preprocess_all(missing_val, 'DATE_TIME')
    return data

def provide_cosmos_met_data(metadata, met_vars, sites=None, missing_val=-9999.0, forcenew=False):
    Path(data_dir+'/met_pickles/').mkdir(exist_ok=True, parents=True)
    fname = data_dir+'/met_pickles/cosmos_site_met.pkl'
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
    Path(data_dir+'/met_pickles/').mkdir(exist_ok=True, parents=True)
    fname = data_dir+'/met_pickles/midas_site_met.pkl'
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

def generate_random_date(year_range):
    start_date = pd.to_datetime(f'{np.min(year_range)}0101')
    end_date = pd.to_datetime(f'{np.max(year_range)+1}0101')
    days_between_dates = (end_date - start_date).days        
    random_number_of_days = np.random.randint(0, days_between_dates)
    random_date = start_date + datetime.timedelta(days=random_number_of_days)
    return random_date

def calculate_1km_pixels_in_25km_cells():
    dat = xr.open_dataset(era5_fldr+'/t2m/era5_20111014_t2m.nc')
    chess_grid = xr.open_dataset('/home/users/doran/data_dump/chess/chess_lat_lon.nc')

    # fig, ax = plt.subplots(1,2)
    # dat.t2m[0,:,:].plot(ax=ax[0])
    # chess_grid.landfrac.plot(ax=ax[1])
    # plt.show()

    # cut down large era5 spatial range to the chess grid extent
    latmin = float(chess_grid.lat.min())
    latmax = float(chess_grid.lat.max())
    lonmin = float(chess_grid.lon.min())
    lonmax = float(chess_grid.lon.max())

    # fig, ax = plt.subplots(1,2)
    # dat.loc[dict(longitude = dat.longitude[(dat.longitude < lonmax) & (dat.longitude > lonmin)],
                 # latitude = dat.latitude[(dat.latitude < latmax) & (dat.latitude > latmin)])].t2m[5,:,:].plot(ax=ax[0])
    # chess_grid.landfrac.plot(ax=ax[1])
    # plt.show()

    dat = dat.loc[dict(longitude = dat.longitude[(dat.longitude < lonmax) & (dat.longitude > lonmin)],
                       latitude = dat.latitude[(dat.latitude < latmax) & (dat.latitude > latmin)])]

    # reproject onto BNG
    wgs84_epsg = 4326
    bng_epsg = 27700
    dat = dat.rio.write_crs(rasterio.crs.CRS.from_epsg(wgs84_epsg))
    dat_bng = dat.rio.reproject(f"EPSG:{bng_epsg}") # too slow!

    # generate a chess grid at the same 25km resolution as the raw ERA5
    res = 25000
    ynew = np.array(range(res//2, res * (int(chess_grid.y.max()) // res) + res//2, res))
    xnew = np.array(range(res//2, res * (int(chess_grid.x.max()) // res) + res//2, res))
    dat_bng_chess = dat_bng.interp(y=ynew, x=xnew)

    # fig, ax = plt.subplots(2,2)
    # dat.t2m[5,:,:].plot(ax=ax[0,0])
    # dat_bng.t2m[5,:,:].plot(ax=ax[0,1])
    # dat_bng_chess.t2m[5,:,:].plot(ax=ax[1,0])
    # chess_grid.landfrac.plot(ax=ax[1,1])
    # plt.show()

    # given a 25km pixel from the era5 lat lon grid, which 1km chess pixels lie inside?
    # does this have a unique solution or is there judgement calls about edge pixels?
    # the netCDF version of ERA5 is on a regular lat/lon grid and should be thought of as 
    # point values at (lat, lon) (or as centroids of tiles, though the interpolation from GRIB to netCDF
    # does not preserve area so this is slightly inaccurate).  

    ## on re-gridded era5 we can do this via (h*(y//h) + h//2, w*(x//w) + w//2)
    # to calculate the centroid of the parent coarse pixel
    exs, eys = np.meshgrid(dat_bng_chess.x.values, dat_bng_chess.y.values)
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
    dat_bng_chess['pixel_id'] = (['y', 'x'],  pixel_ids)

    cxs, cys = np.meshgrid(res * (chess_grid.x.values // res) + res//2,
                           res * (chess_grid.y.values // res) + res//2)
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
    chess_grid.drop(['landfrac','lat','lon']).to_netcdf('/home/users/doran/data_dump/chess/chess_1km_25km_parent_pixel_ids.nc')    
    dat_bng_chess.drop(['t2m','time']).to_netcdf('/home/users/doran/data_dump/chess/chess_25km_pixel_ids.nc')
    # new_cmap = rand_cmap(era_xy.shape[0], ctype='bright')
    # fig, ax = plt.subplots(1,2)
    # dat_bng_chess.pixel_id.plot(ax=ax[0], cmap=new_cmap)
    # chess_grid.era5_nbr.plot(ax=ax[1], cmap=new_cmap)
    # plt.show()

    ## then, knowing which 25km cell each 1km pixel relates to, we can enforce
    ## approximate conservation of cell averages. Also, the above only
    ## needs to be done once and then the arrays re-used.

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

def tensorise_station_targets(station_targets, max_coord=99, device=None):
    # converts list of dataframes to [xy, value, mask] tensor list
    outputs = {'coords_yx':[], 'values':[], 'var_present':[]}
    
    for b in range(len(station_targets)):
        outputs['coords_yx'].append(
            torch.from_numpy(station_targets[b][['sub_y', 'sub_x']].to_numpy()).to(torch.long) #(torch.float32)
        )
       
        outputs['values'].append(
            torch.from_numpy(station_targets[b].iloc[:,2:].to_numpy()).to(torch.float32)
        )
        
        outputs['var_present'].append(
            torch.from_numpy(~station_targets[b].iloc[:,2:].isna().to_numpy()).to(torch.bool)
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

def relhum_from_dewpoint(air_temp, dewpoint_temp):
    return 100 * (np.exp((17.625 * dewpoint_temp)/(243.04 + dewpoint_temp)) / 
                    np.exp((17.625 * air_temp)/(243.04 + air_temp)))

class Batch:    
    def __init__(self, batch, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")        
        self.coarse_inputs = batch[0].to(device)
        self.fine_inputs = batch[1].to(device)
        self.station_dict = tensorise_station_targets(batch[2], device=device)
        self.constraint_targets = batch[3].to(device)

class data_generator():
    def __init__(self, train_years, val_years, heldout_years,
                 train_sites=None, heldout_sites=None,
                 dim_l=4, res=25000, scale=25): # perhaps can boost to dim_l=8 or higher on orchid?
        
        # load 1km chess grid
        self.chess_grid = xr.open_dataset('/home/users/doran/data_dump/chess/chess_lat_lon.nc')        
        self.latmin = float(self.chess_grid.lat.min())
        self.latmax = float(self.chess_grid.lat.max())
        self.lonmin = float(self.chess_grid.lon.min())
        self.lonmax = float(self.chess_grid.lon.max())
        
        self.met_cls = ERA5Data()
        
        self.wgs84_epsg = 4326
        self.bng_epsg = 27700
        self.res = res # resolution of lo-res image in m
        self.scale = scale # downscaling factor
        self.dim_l = dim_l # size of lo-res image in 25km pixels (size of subest of UK to train on each sample)
        self.dim_h = self.dim_l*scale # size of hi-res image in 1km pixels
        
        # create a coarse res chess grid        
        self.y_chess_25k = np.array(range(res//2, res * (int(self.chess_grid.y.max()) // res) + res//2, res))
        self.x_chess_25k = np.array(range(res//2, res * (int(self.chess_grid.x.max()) // res) + res//2, res))
        
        # load parent pixel ids on reprojected/regridded ERA5 25km cells
        self.parent_pixels = xr.open_dataset('/home/users/doran/data_dump/chess/chess_25km_pixel_ids.nc')
        
        # load child pixels on 1km chess grid labelled by parent pixel IDs
        self.child_parent_map = xr.open_dataset('/home/users/doran/data_dump/chess/chess_1km_25km_parent_pixel_ids.nc')    
    
        # define ERA5 (coarse) and COSMOS (fine) variables to load
        var_name_map = dict(
            name   = ['Air_Temp', 'Pressure', 'Short_Wave_Rad_In', 'Long_Wave_Rad_In', 'Wind_Speed', 'Relative_Humidity'],
            coarse = ['t2m'     , 'sp'      , 'msdwswrf'         , 'msdwlwrf'        , 'ws'        , 'rh'],
            fine   = ['TA'      , 'PA'      , 'SWIN'             , 'LWIN'            , 'WS'        , 'RH']
        )
        self.var_name_map = pd.DataFrame(var_name_map)
                        
        # load site metadata/locations
        self.site_metadata = CosmosMetaData()
        midas_metadata = pd.read_csv('~/data_dump/MetOffice/midas_site_locations.csv')

        # load site data
        self.site_data = provide_cosmos_met_data(self.site_metadata, self.var_name_map['fine'], forcenew=False)
        self.site_metadata = self.site_metadata.site

        #midas_vars = ['air_temperature', 'stn_pres', 'wind_speed', 'rltv_hum']
        midas_data = provide_midas_met_data(midas_metadata, self.var_name_map['fine'], forcenew=False)
        # inserted missing SWIN/LWIN columns (only total rad available at midas sites)

        # rescale midas air temperature to Kelvin to match COSMOS and add to self.site_dat
        for sid in midas_data.keys():
            midas_data[sid]['TA'] = midas_data[sid]['TA'] + 273.15
            self.site_data[sid] = midas_data[sid]
            self.site_metadata = pd.concat([self.site_metadata, 
                                          midas_metadata[midas_metadata['SITE_ID']==sid]], axis=0)
        self.site_metadata = self.site_metadata.reset_index().drop('index', axis=1)

        # for each site find the 1km chess tile it sits within
        cosmos_chess_y = []
        cosmos_chess_x = []
        coarse_parent_pixel_id = []
        for i in self.site_metadata.index:
            this_dat = self.site_metadata.loc[i]
            ccyx = find_chess_tile(this_dat['LATITUDE'], this_dat['LONGITUDE'], self.chess_grid)
            cosmos_chess_y.append(ccyx[0][0])
            cosmos_chess_x.append(ccyx[1][0])
            try:
                coarse_parent_pixel_id.append(int(self.child_parent_map.era5_nbr[ccyx[0][0], ccyx[1][0]].values))
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

        self.train_years = train_years
        self.val_years = val_years
        self.heldout_years = heldout_years
        if train_sites is None:
            train_sites, heldout_sites = site_splits(use_sites=list(self.site_metadata.SITE_ID),
                                                     holdoutfrac=0.1, random_state=22)        
        self.train_sites = train_sites
        self.heldout_sites = heldout_sites

        # load hi res static data        
        self.height_grid = xr.open_dataset('~/data_dump/height_map/uk_ihdtm_topography+topoindex_1km.nc')
        self.elev_vars = ['elev', 'stdev', 'slope', 'aspect', 'topi', 'stdtopi', 'fdepth']
        # easting/northing of SW corner of grid box, so redefine x,y
        self.height_grid = self.height_grid.assign_coords({'x':self.chess_grid.x.values,
                                                           'y':self.chess_grid.y.values})
        self.height_grid.eastings.values = self.height_grid.eastings.values + 500
        self.height_grid.northings.values = self.height_grid.northings.values + 500
        # now we are labelling the tile centroid
        
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
        
        # # load normalisation dict from soil moisture        
        # smfolder = '/home/users/doran/projects/soil_moisture/'
        # smmodel = 'sm_rescale_tdts_selective_cosmos'
        # with open(smfolder+f'/logs/{smmodel}/normalisations.pkl', 'rb') as fo:
            # self.normalisations = pickle.load(fo)
        # 
        #rad_norm = self.normalisations['rad_norm']                
        # lwin_mu = 330.
        # lwin_sd = 35.
        # self.normalisations['lwin_mu'] = lwin_mu
        # self.normalisations['lwin_sd'] = lwin_sd
        # logswin_mu = 5.
        # logswin_sd = 2.
        # self.normalisations['logswin_mu'] = logswin_mu
        # self.normalisations['logswin_sd'] = logswin_sd
        # temp_mu = self.normalisations['temp_mu']
        # temp_sd = self.normalisations['temp_sd']
        # p_mu = self.normalisations['p_mu']
        # p_sd = self.normalisations['p_sd']
        # #rh_norm = self.normalisations['rh_norm']
        # rh_mu = 85.
        # rh_sd = 12.
        # self.normalisations['rh_mu'] = rh_mu
        # self.normalisations['rh_sd'] = rh_sd
        # ws_mu = self.normalisations['ws_mu']
        # ws_sd = self.normalisations['ws_sd']
        
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
            self.site_data[SID].loc[:, ['RH']] = (self.site_data[SID].loc[:, ['RH']] - nm.rh_mu) / nm.rh_mu
            
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
        ys = self.parent_pixels.y.shape[0]
        xs = self.parent_pixels.x.shape[0]
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
                   
    def get_batch(self, batch_size=1, batch_type='train', load_binary_batch=True):
        if load_binary_batch is False:
            self.parent_pixels = self.read_parent_pixel_day(batch_type=batch_type)
        else:
            bfn = self.bin_batches[batch_type][np.random.randint(0, len(self.bin_batches[batch_type]))]            
            self.parent_pixels = pd.read_pickle(bfn)
        
        # extract timestamp
        self.td = pd.to_datetime(self.parent_pixels.time[0].values)
        
        with warnings.catch_warnings():            
            warnings.simplefilter("ignore")
            # ignore warnings thrown by interp_like() which seem to be needless
            
            ### constraints for training help 
            ## Air Temp
            # reduce to sea level with lapse rate of -0.006 K/m
            lapse_val = -0.0065
            self.parent_pixels['t_sealevel'] = (['time', 'y', 'x'],
                                               self.parent_pixels['t2m'].values)
            self.parent_pixels['t_sealevel'] = self.parent_pixels['t_sealevel'] - self.parent_pixels['elev'] * lapse_val

            # bicubic spline to interpolate from 25km to 1km... cubic doesn't work here!
            t_sealevel_interp = self.parent_pixels['t_sealevel'].interp_like(self.chess_grid, method='linear')

            # adjust to the 1km elevation using same lapse rate
            self.t_1km_elev = t_sealevel_interp + self.height_grid['elev'] * lapse_val

            ## Air Pressure:
            # integral of hypsometric equation using the 1km Air Temp?
            T_av = 0.5*(t_sealevel_interp + self.t_1km_elev) # K
            p_1 = 1013 # hPa, standard sea level pressure value
            R = 287 # J/kg·K = kg.m2/s2/kg.K = m2/s2.K  # specific gas constant of dry air
            g = 9.81 # m/s2 
            self.p_1km_elev = p_1 * np.exp(-g * self.height_grid['elev'] / (R * T_av))
            del(T_av)
            del(t_sealevel_interp)

            ## Relative Humidity
            # Assumed constant with respect to elevation, so can be simply 
            # interpolated from 25km to 1km using a bicubic spline.
            t2m_C = self.parent_pixels['t2m'] - 273.15 # K -> Celsius 
            d2m_C = self.parent_pixels['d2m'] - 273.15 # K -> Celsius
            rh_25km = relhum_from_dewpoint(t2m_C, d2m_C)
            # rh_25km = 100 * (np.exp((17.625 * d2m_C)/(243.04 + d2m_C)) / 
                # np.exp((17.625 * t2m_C)/(243.04 + t2m_C)))
            self.rh_1km_interp = rh_25km.interp_like(self.chess_grid, method='linear')
            del(rh_25km)
                
        # load the correct hi-res daily [or hourly] precipitation field
        precip_file = glob.glob(precip_fldr + f'rainfall_hadukgrid_uk_1km_day_{self.td.year}{zeropad_strint(self.td.month)}01*.nc')[0]
        precip_dat = xr.open_dataset(precip_file) # total in mm
        self.precip_dat = precip_dat.loc[dict(
            projection_y_coordinate = precip_dat.projection_y_coordinate[
                (precip_dat.projection_y_coordinate <= float(self.chess_grid.y.max().values)) & 
                (precip_dat.projection_y_coordinate >= float(self.chess_grid.y.min().values))],
            projection_x_coordinate = precip_dat.projection_x_coordinate[
                (precip_dat.projection_x_coordinate <= float(self.chess_grid.x.max().values)) & 
                (precip_dat.projection_x_coordinate >= float(self.chess_grid.x.min().values))]
        )]
        
        coarse_inputs = []
        fine_inputs = []
        station_targets = []
        constraint_targets = []
        for b in range(batch_size):
            coarse_input, fine_input, station_target, constraints = self.get_sample(batch_type=batch_type)            
            coarse_inputs.append(coarse_input)
            fine_inputs.append(fine_input)
            station_targets.append(station_target)
            constraint_targets.append(constraints)
        coarse_inputs = torch.stack(coarse_inputs, dim=0)
        fine_inputs = torch.stack(fine_inputs, dim=0)
        constraint_targets = torch.stack(constraint_targets, dim=0)        
        
        # replace NaNs (usually the ocean) with random noise? Or padding?        
        indices = torch.where(torch.isnan(coarse_inputs))
        coarse_inputs[indices] = torch.rand(indices[0].size())*2 - 1        
                
        indices = torch.where(torch.isnan(fine_inputs))
        fine_inputs[indices] = torch.rand(indices[0].size())*2 - 1        
               
        #return coarse_inputs, fine_inputs, station_targets, constraint_targets
        return (coarse_inputs.permute(0,3,1,2), fine_inputs.permute(0,3,1,2),
                station_targets, constraint_targets.permute(0,3,1,2))


    def get_sample(self, batch_type='train'):
        # reuse a single ERA5 file/date and just grab different times and spatial chunks
        # for each batch to cut down on loading/reprojecting time! 
        
        # choose a station and find its 1km pixel
        if batch_type=='train' or batch_type=='val':
            SID = np.random.choice(self.train_sites)
        elif batch_type=='test':
            SID = np.random.choice(self.train_sites + self.heldout_sites) 
        targ_site = self.site_metadata[self.site_metadata['SITE_ID']==SID]
        targ_loc = self.site_metadata[self.site_metadata['SITE_ID']==SID][['LATITUDE','LONGITUDE']]
        targ_yx = np.where(self.parent_pixels.pixel_id.values == targ_site['parent_pixel_id'].values)

        #print(SID)
        #print(targ_yx)

        # grab a random dim_l x dim_l tile that contains that 1km pixel
        ix = np.random.randint(max(0, targ_yx[1][0] - self.dim_l + 1),
                               min(self.parent_pixels.x.shape[0] - self.dim_l + 1, targ_yx[1][0] + 1))
        iy = np.random.randint(max(0, targ_yx[0][0] - self.dim_l + 1),
                               min(self.parent_pixels.y.shape[0] - self.dim_l + 1, targ_yx[0][0] + 1))
        it = np.random.randint(0, 24)
        subdat = self.parent_pixels.isel(time=it, y=range(iy, iy+self.dim_l), x=range(ix, ix+self.dim_l))
        
        timestamp = pd.to_datetime(self.parent_pixels.time.values[it], utc=True)        
        doy = timestamp.day_of_year
        year_sin = np.sin(doy / 365. * 2*np.pi - np.pi/2.)
        year_cos = np.cos(doy / 365. * 2*np.pi - np.pi/2.)
        hour = timestamp.hour
        hour_sin = np.sin(hour / 24 * 2*np.pi - 3*np.pi/4.)
        hour_cos = np.cos(hour / 24 * 2*np.pi - 3*np.pi/4.)
                 

        # find any other stations that lie inside the chosen tile and their pixels
        parents = subdat.pixel_id.values.flatten()
        if batch_type=='train' or batch_type=='val':
            contained_sites = self.site_metadata[(self.site_metadata['SITE_ID'].isin(self.train_sites)) & 
                                                 (self.site_metadata['parent_pixel_id'].isin(parents))]
        elif batch_type=='test':
            contained_sites = self.site_metadata[(self.site_metadata['SITE_ID'].isin(self.train_sites + self.heldout_sites)) &
                                                 (self.site_metadata['parent_pixel_id'].isin(parents))]
        # then we use all these sites as targets for the loss calculation
        # (if they have data for the particular variable on the particular day)        

        # subset hi-res fields to the chosen coarse tile
        x_inds = np.intersect1d(np.where(self.chess_grid.x.values < int(subdat.x.max().values) + self.res//2),
                                np.where(self.chess_grid.x.values > int(subdat.x.min().values) - self.res//2))
        y_inds = np.intersect1d(np.where(self.chess_grid.y.values < int(subdat.y.max().values) + self.res//2),
                                np.where(self.chess_grid.y.values > int(subdat.y.min().values) - self.res//2))
        sub_chess = self.chess_grid.isel(y=y_inds, x=x_inds)
        sub_topog = self.height_grid.isel(y=y_inds, x=x_inds)
        sub_precip = self.precip_dat.isel(time=timestamp.day-1, # due to accumulating label? Check this
                                          projection_y_coordinate=y_inds,
                                          projection_x_coordinate=x_inds)
                                          
        # also subset the constraint fields
        sub_temp_constraint = self.t_1km_elev.isel(time=it, y=y_inds, x=x_inds)
        sub_pres_constraint = self.p_1km_elev.isel(time=it, y=y_inds, x=x_inds)
        sub_relh_constraint = self.rh_1km_interp.isel(time=it, y=y_inds, x=x_inds)

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

        # convert wind vectors and dewpoint temp to wind speed and relative humidity
        subdat['ws'] = np.sqrt(np.square(subdat['v10']) + np.square(subdat['u10']))
        subdat['t2m'] -= 273.15 # K -> Celsius 
        subdat['d2m'] -= 273.15 # K -> Celsius
        subdat['sp'] /= 100.    # Pa -> hPa
        # RH calc from https://www.omnicalculator.com/physics/relative-humidity
        subdat['rh'] = relhum_from_dewpoint(subdat['t2m'], subdat['d2m'])
        # subdat['rh'] = 100 * (np.exp((17.625 * subdat['d2m'])/(243.04 + subdat['d2m'])) / 
            # np.exp((17.625 * subdat['t2m'])/(243.04 + subdat['t2m'])))
        
        subdat = subdat.drop(['u10','v10', 'd2m'])
        # subdat['msdwlwrf'][subdat['msdwlwrf']<0] = 0
        # subdat['msdwswrf'][subdat['msdwswrf']<0] = 0

        ## normalise data using norm dict        
        # lwin_mu = self.normalisations['lwin_mu']
        # lwin_sd = self.normalisations['lwin_sd']
        # logswin_mu = self.normalisations['logswin_mu']
        # logswin_sd = self.normalisations['logswin_sd']
        # temp_mu = self.normalisations['temp_mu']
        # temp_sd = self.normalisations['temp_sd']
        # p_mu = self.normalisations['p_mu']
        # p_sd = self.normalisations['p_sd']        
        # rh_mu = self.normalisations['rh_mu']        
        # rh_sd = self.normalisations['rh_sd']
        # ws_mu = self.normalisations['ws_mu']
        # ws_sd = self.normalisations['ws_sd']
        
        # met data
        subdat['t2m'] = (subdat['t2m'] - nm.temp_mu) / nm.temp_sd
        subdat['sp'] = (subdat['sp'] - nm.p_mu) / nm.p_sd
        subdat['msdwlwrf'] = (subdat['msdwlwrf'] - nm.lwin_mu) / nm.lwin_sd # standardise!
        subdat['msdwswrf'] = np.log(1. + subdat['msdwswrf']) # log(1 + swin)
        subdat['msdwswrf'] = (subdat['msdwswrf'] - nm.logswin_mu) / nm.logswin_sd # standardise!        
        subdat['ws'] = (subdat['ws'] - nm.ws_mu) / nm.ws_sd
        subdat['rh'] = (subdat['rh'] - nm.rh_mu) / nm.rh_sd # standardise!

        # constraints
        sub_temp_constraint = (sub_temp_constraint - 273.15 - nm.temp_mu) / nm.temp_sd
        sub_pres_constraint = (sub_pres_constraint - nm.p_mu) / nm.p_sd
        sub_relh_constraint = (sub_relh_constraint - nm.rh_mu) / nm.rh_sd

        # hi res data
        # precip_norm = self.normalisations['precip_norm']
        sub_precip['rainfall'] = sub_precip['rainfall'] / nm.precip_norm # Should we log transform this too?
        sub_topog['aspect'] = sub_topog['aspect'] / 360. - 0.5 # so goes between -0.5 and 0.5
        for var in self.elev_vars:
            if var=='aspect': continue
            sub_topog[var] = (sub_topog[var] - nm.s_means.loc[var]) / nm.s_stds.loc[var]

        ## create tensors with batch index and channel dim last
        coarse_input = torch.zeros((subdat['t2m'].shape[0],subdat['t2m'].shape[1], 0)) # (Y, X, C)
        for var in self.var_name_map.coarse:            
            coarse_input = torch.cat(
                (coarse_input, 
                 torch.from_numpy(subdat[var].values).to(torch.float32)[...,None]
                ), dim = -1)
        # therefore coarse scale variable order (in cosmos parlance) is
        self.coarse_variable_order = list(self.var_name_map['fine'])

        # get static inputs at the high resolution
        landfrac = sub_chess.landfrac.values     
        fine_input = torch.from_numpy(landfrac).to(torch.float32)[...,None] # (Y, X, C)
        for var in self.elev_vars:
            fine_input = torch.cat(
                (fine_input, 
                 torch.from_numpy(sub_topog[var].values).to(torch.float32)[...,None]
                ), dim = -1)
        # and join on precipitation
        fine_input = torch.cat(
            (fine_input, 
             torch.from_numpy(sub_precip['rainfall'].values).to(torch.float32)[...,None]
            ), dim = -1)
        
        # add time signals
        landfrac.fill(year_sin)
        ysin = torch.from_numpy(landfrac).to(torch.float32)[...,None]
        landfrac.fill(year_cos)
        ycos = torch.from_numpy(landfrac).to(torch.float32)[...,None]
        landfrac.fill(hour_sin)
        hsin = torch.from_numpy(landfrac).to(torch.float32)[...,None]
        landfrac.fill(hour_cos)
        hcos = torch.from_numpy(landfrac).to(torch.float32)[...,None]
        fine_input = torch.cat([fine_input, ysin, ycos, hsin, hcos], dim = -1)
                    
        # therefore fine scale variable order is
        self.fine_variable_order = (['landfrac'] + self.elev_vars + 
            ['rainfall', 'year_sin', 'year_cos', 'hour_sin', 'hour_cos'])
        
        # define station targets
        station_targets = contained_sites[
            ['sub_x', 'sub_y'] + list(self.var_name_map.fine)
        ]

        # combine constraints into tensor        
        constraints = torch.from_numpy(sub_temp_constraint.values).to(torch.float32)[...,None] # (Y, X, C)
        constraints = torch.cat(
                (constraints, 
                 torch.from_numpy(sub_pres_constraint.values).to(torch.float32)[...,None]
                ), dim = -1)
        constraints = torch.cat(
                (constraints, 
                 torch.from_numpy(sub_relh_constraint.values).to(torch.float32)[...,None]
                ), dim = -1)
        # therefore constraint variable order is
        self.constraint_variable_order = ['TA', 'PA', 'RH']
        
        return coarse_input, fine_input, station_targets, constraints
