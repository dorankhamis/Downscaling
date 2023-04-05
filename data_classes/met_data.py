""" Class(es) to load and process meteorological driving data.
Might not need at COSMOS sites as point measurements could work in first
instance. But moving to 1km grids, will definitely need.

raw driving data in
/gws/nopw/j04/hydro_jules/data/uk
"""
import netCDF4 as nc
import xarray as xr
import pandas as pd
import numpy as np
import pathlib
import glob

from soil_moisture.utils import (days_in_month, indexify_datetime,
                                 find_chess_tile, find_tile_bounds,
                                 zeropad_strint, data_dir, nonzero_mean,
                                 resample_to_res, resample_to_res_split, q_func)

fldr1 = data_dir + '/chess/chess_met/daily/'
fldr2 = '/gws/nopw/j04/hydro_jules/data/uk/driving_data/era5/bn_grid/'
#fldr3 = '/gws/nopw/j04/hydro_jules/data/uk/driving_data/era5/hourly_single_levels/'
fldr4 = data_dir + '/ERA5/bng_1km_npy/'
#fldr4 = '/gws/nopw/j04/hydro_jules/data/uk/soil_moisture_map/UKV/....'

haduk_raw = '/badc/ukmo-hadobs/data/insitu/MOHC/HadOBS/HadUK-Grid/v1.1.0.0/1km/rainfall/day/latest/'
fldr_base = '/gws/nopw/j04/hydro_jules/data/uk/soil_moisture_map/'
fldr_ukv = fldr_base + '/driving_data/UKV/bn_grid/'
fldr_haduk = fldr_base + '/driving_data/HadUK-Grid/bn_grid/'
fldr_era5 = data_dir + '/ERA5/bng_1km_npy/'
fldr_ancil = fldr_base + '/ancillaries/'

def process_date(date, doy=False):
    year = date.year
    month = zeropad_strint(date.month)
    day = zeropad_strint(date.day)
    dayofyear = zeropad_strint(date.day_of_year, num=2)
    if doy is True:        
        fpick = f'{year}{dayofyear}'
    else:
        fpick = f'{year}{month}{day}'
    return fpick, (int(year), int(month), int(day))

def process_date_range(start_date, end_date, freq='D', doy=False):
    drange = pd.date_range(start_date, end_date, freq=freq)
    fpicks = []
    year_month_day = []
    for d in drange:        
        f, d_tup = process_date(d, doy=doy)
        fpicks.append(f)
        year_month_day.append(d_tup)
    unique_files = np.unique(fpicks)
    return unique_files, year_month_day

def transform_psl_to_pa(P_sl, T, h):
    L_b = -0.0065 # temperature lapse rate, K/m
    Rstar = 8.3144598 # gas constant
    g0 = 9.80665 # gravity
    M = 0.0289644 # molar mass of air
    h_sl = 0
    # h = elevation at pixel
    # T = temperature at elevation h in Kelvin
    # this assumes we know sea level temperature
    #P = P_sl * ((T_sl + (h - h_sl)*L_b) / T_sl)**(-g0*M)/(Rstar*L_b)
    # so rewrite knowing surface temperature at height h
    return P_sl * (T / (T - (h - h_sl)*L_b))**((-g0*M)/(Rstar*L_b))


class HadUKData():
    def __init__(self):
        pass
        
    def grab_grid(self, year, month):
        precip_file = glob.glob(haduk_raw + f'rainfall_hadukgrid_uk_1km_day_{year}{zeropad_strint(month)}01*.nc')[0]
        return xr.open_dataset(precip_file) # total in mm


class UKVData():
    def __init__(self):
        self.var_names = ['TA', 'PA', 'PRECIP', 'LWIN', 'SWIN', 'RH', 'WS']
        self.out_names = ['TA', 'PA', 'PRECIP', 'LWIN', 'SWIN', 'RH', 'WS']
        self.map_era5_names = {'TA':'t2m', 'PA':'sp', 'PRECIP':'mtpr', 'WS':'ws',
                               'LWIN':'msdwlwrf', 'SWIN':'msdwswrf', 'RH':'rh'}        
        self.chess_flat_idx_map = pd.read_csv(fldr_ancil+'/chess_yx_to_flat_id.csv')
                          
    def grab_pixel_from_binary(self, start, end, chess_ys, chess_xs, elev=[0]):
        ## process time slice
        uniq_files, ymd = process_date_range(start, end, freq='D')
        
        ## find the index of the flattened .npy files that 
        ## corresponds to the chesstile_yx indices
        idx_df = (self.chess_flat_idx_map
            .merge(pd.DataFrame({'y':chess_ys,
                                 'x':chess_xs},
                                 index=list(range(len(chess_xs)))),
                   on=['y', 'x'], how='right'))
        
        ''' If file missing load up naive ERA5 pixel instead!
        This requires outputting the ERA5 binaries from the UKV 
        missing data list that Richard sent'''
                
        out_df = pd.DataFrame()
        idxs = idx_df.id.values
        for i, doy in enumerate(uniq_files):
            year = doy[:4]
            thisout = {}
            thisout['DATE_TIME'] = pd.to_datetime(doy, format='%Y%m%d', utc=True)
            for v in np.setdiff1d(self.out_names, ['PA']):
                if v=='PRECIP':
                    try:
                        psum = np.load(fldr_haduk + f'/{v}/{year}/{doy}_{v}_sum.npy')
                    except:
                        ve5 = self.map_era5_names[v]
                        psum = np.load(fldr_era5 + f'/{ve5}/{year}/{doy}_{ve5}_sum.npy')
                                              
                    psum = psum[idxs]
                    thisout[v+'_sum'] = psum
                    #pmax = np.load(fldr4 + f'/{v}/{year}/{doy}_{v}_max.npy')
                    #pmax = pmax[idx_df.id.values]
                    #thisout[v+'_max'] = pmax
                elif v=='SWIN':
                    try:
                        pmean = np.load(fldr_ukv + f'/{v}/{year}/{doy}_{v}_mean.npy')
                        pmax = np.load(fldr_ukv + f'/{v}/{year}/{doy}_{v}_max.npy')
                        pdaymean = np.load(fldr_ukv + f'/{v}/{year}/{doy}_{v}_daymean.npy')
                    except:
                        ve5 = self.map_era5_names[v]
                        pmean = np.load(fldr_era5 + f'/{ve5}/{year}/{doy}_{ve5}_mean.npy')
                        pmax = np.load(fldr_era5 + f'/{ve5}/{year}/{doy}_{ve5}_max.npy')
                        pdaymean = np.load(fldr_era5 + f'/{ve5}/{year}/{doy}_{ve5}_daymean.npy')
                        
                    pmean = pmean[idxs]
                    pmax = pmax[idxs]
                    pdaymean = pdaymean[idxs]
                    thisout[v+'_mean'] = pmean
                    thisout[v+'_daymean'] = pdaymean
                    thisout[v+'_max'] = pmax                   
                else:                        
                    try:
                        pmean = np.load(fldr_ukv + f'/{v}/{year}/{doy}_{v}_mean.npy')
                        pmax = np.load(fldr_ukv + f'/{v}/{year}/{doy}_{v}_max.npy')
                        pmin = np.load(fldr_ukv + f'/{v}/{year}/{doy}_{v}_min.npy')
                    except:
                        ve5 = self.map_era5_names[v]
                        pmean = np.load(fldr_era5 + f'/{ve5}/{year}/{doy}_{ve5}_mean.npy')
                        pmax = np.load(fldr_era5 + f'/{ve5}/{year}/{doy}_{ve5}_max.npy')
                        pmin = np.load(fldr_era5 + f'/{ve5}/{year}/{doy}_{ve5}_min.npy')
                        
                    pmean = pmean[idxs]
                    pmax = pmax[idxs]
                    pmin = pmin[idxs]
                    thisout[v+'_mean'] = pmean
                    thisout[v+'_min'] = pmin
                    thisout[v+'_max'] = pmax
                    
                    if v=='TA':
                        # also transform sea level pressure to surface pressure
                        try:
                            pslmean = np.load(fldr_ukv + f'/PSL/{year}/{doy}_PSL_mean.npy')
                            pslmax = np.load(fldr_ukv + f'/PSL/{year}/{doy}_PSL_max.npy')
                            pslmin = np.load(fldr_ukv + f'/PSL/{year}/{doy}_PSL_min.npy')
                            pslmean = pslmean[idxs]
                            pslmax = pslmax[idxs]
                            pslmin = pslmin[idxs]
                            pamean = transform_psl_to_pa(pslmean, pmean + 273.15, elev)
                            pamin = transform_psl_to_pa(pslmin, pmin + 273.15, elev)
                            pamax = transform_psl_to_pa(pslmax, pmax + 273.15, elev)
                        except:
                            ve5 = self.map_era5_names['PA']
                            pamean = np.load(fldr_era5 + f'/{ve5}/{year}/{doy}_{ve5}_mean.npy')
                            pamax = np.load(fldr_era5 + f'/{ve5}/{year}/{doy}_{ve5}_max.npy')
                            pamin = np.load(fldr_era5 + f'/{ve5}/{year}/{doy}_{ve5}_min.npy')                        
                            pamean = pamean[idxs]
                            pamax = pamax[idxs]
                            pamin = pamin[idxs]
                                                        
                        thisout['PA_mean'] = pamean
                        thisout['PA_min'] = pamin
                        thisout['PA_max'] = pamax
                        
            # create dataframe
            this_df = (pd.DataFrame(thisout, index=idxs)
                .set_index('DATE_TIME')
                .assign(bng_pixel_id = idxs, y = idx_df.y.values, x = idx_df.x.values))
            out_df = pd.concat([out_df, this_df], axis=0)
        
        ## sort out column types
        out_df = out_df.astype(np.float32)
        out_df[['bng_pixel_id', 'y', 'x']] = out_df[['bng_pixel_id', 'y', 'x']].astype(np.int32)        
        return out_df


class ERA5Data():
    def __init__(self):
        self.var_names = ['t2m', 'u10', 'v10', 'sp', 'mtpr', 'msdwlwrf', 'msdwswrf', 'd2m']
        self.proc_var_names = ['t2m', 'sp', 'mtpr', 'msdwlwrf', 'msdwswrf', 'ws', 'rh']
        self.var_long_names = ['2m temperature',
                           '10m U wind component',
                           '10m V wind component',
                           'surface pressure',
                           'mean total precipitation rate',
                           'mean surface downward long-wave radiation flux',
                           'mean surface downward short-wave radiation flux',
                           '2m dewpoint temperature']
        self.map_names = {'t2m':'TA', 'sp':'PA', 'mtpr':'PRECIP',
                          'msdwlwrf':'LWIN', 'msdwswrf':'SWIN'}
        self.out_names = ['TA', 'PA', 'PRECIP', 'LWIN', 'SWIN', 'RH', 'WS']
        self.chess_flat_idx_map = pd.read_csv(fldr_ancil+'/chess_yx_to_flat_id.csv')
                          

    def grab_pixel_from_binary(self, chesstile_yx, start, end):
        ## process time slice
        uniq_files, ymd = process_date_range(start, end, freq='D')
        
        ## find the index of the flattened .npy files that 
        ## corresponds to the chesstile_yx indices
        idx_df = (self.chess_flat_idx_map
            .merge(pd.DataFrame({'y':chesstile_yx[0][0],
                                 'x':chesstile_yx[1][0]}, index=[0]),
                                 on=['y', 'x'], how='right'))
        
        # add procesed rh, ws names to name map
        map_names = self.map_names
        map_names['rh'] = 'RH'
        map_names['ws'] = 'WS'
        
        out_df = pd.DataFrame()
        for i, doy in enumerate(uniq_files):
            year = doy[:4]
            thisout = {}
            thisout['DATE_TIME'] = pd.to_datetime(doy, format='%Y%m%d', utc=True)
            for v in self.proc_var_names:
                if v=='mtpr':
                    psum = np.load(fldr4 + f'/{v}/{year}/{doy}_{v}_sum.npy')
                    psum = psum[idx_df.id.values]
                    pq99 = np.load(fldr4 + f'/{v}/{year}/{doy}_{v}_max.npy')
                    pq99 = pq99[idx_df.id.values]
                    thisout[map_names[v]+'_sum'] = psum
                    thisout[map_names[v]+'_max'] = pq99
                elif v=='msdwswrf':
                    pmean = np.load(fldr4 + f'/{v}/{year}/{doy}_{v}_mean.npy')
                    pmean = pmean[idx_df.id.values]
                    pq99 = np.load(fldr4 + f'/{v}/{year}/{doy}_{v}_max.npy')
                    pq99 = pq99[idx_df.id.values]
                    pdaymean = np.load(fldr4 + f'/{v}/{year}/{doy}_{v}_daymean.npy')
                    pdaymean = pdaymean[idx_df.id.values]
                    thisout[map_names[v]+'_mean'] = pmean
                    thisout[map_names[v]+'_daymean'] = pdaymean
                    thisout[map_names[v]+'_max'] = pq99
                else:
                    pmean = np.load(fldr4 + f'/{v}/{year}/{doy}_{v}_mean.npy')
                    pmean = pmean[idx_df.id.values]
                    pq99 = np.load(fldr4 + f'/{v}/{year}/{doy}_{v}_max.npy')
                    pq99 = pq99[idx_df.id.values]
                    pq01 = np.load(fldr4 + f'/{v}/{year}/{doy}_{v}_min.npy')
                    pq01 = pq01[idx_df.id.values]
                    thisout[map_names[v]+'_mean'] = pmean
                    thisout[map_names[v]+'_min'] = pq01
                    thisout[map_names[v]+'_max'] = pq99
            
            # create dataframe
            this_df = pd.DataFrame(thisout, index=[0]).set_index('DATE_TIME')
            out_df = pd.concat([out_df, this_df], axis=0)
        return out_df

    def grab_grid_slice(self, lat, lon, start, end, latlon_ref,
                        chesstile_yx=None, tile_bounds=None):
        ## process time slice
        uniq_files, ymd = process_date_range(start, end, freq='D')
                
        ## process grid box
        if chesstile_yx is None:
            chesstile_yx = find_chess_tile(lat, lon, latlon_ref)
        if tile_bounds is None:
            tile_bounds = find_tile_bounds(latlon_ref, chesstile_yx)

        out = dict(DATE_TIME=[])        
        for v in self.var_names:   
            out[v] = []
        
        for i, doy in enumerate(uniq_files):
            for v in self.var_names:
                infile = fldr2+f'/{v}/era5_{doy}_{v}.nc'
                #infile = fldr3+f'/{v}/era5_{doy}_{v}.nc'
                if not pathlib.Path(infile).exists(): continue
                dat = xr.open_dataset(infile)
                # testing regridding on raw files
                #dat = dat.interp(longitude=latlon_ref.lon, latitude=latlon_ref.lat)
                # subset to lat lon
                sbst = dat[v][:, chesstile_yx[0][0], chesstile_yx[1][0]].load()
                out[v] += list(sbst.data)
                # add date time once only per variable sweep
                if len(out['DATE_TIME'])<max([len(x) for x in out.values()]):
                    out['DATE_TIME'] += pd.to_datetime(list(sbst.time.data), utc=True)
        out = pd.DataFrame(out).set_index('DATE_TIME')        
        
        ## do rescaling, renaming and calculate necessary statistics
        out['WS'] = np.sqrt(out['u10']**2 + out['v10']**2)
        out = out.rename(self.map_names, axis=1).drop(['u10','v10'], axis=1)
        out['TA'] -= 273.15 # K -> Celsius 
        out['d2m'] -= 273.15 # K -> Celsius
        out['PA'] /= 100. # Pa -> hPa
        out['PRECIP'] *= 60.*60. # mean kg m-2 s-1 -> total mm/hour                
        ## RH calc from https://www.omnicalculator.com/physics/relative-humidity        
        out['RH'] = 100 * (np.exp((17.625 * out['d2m'])/(243.04 + out['d2m'])) / 
            np.exp((17.625 * out['TA'])/(243.04 + out['TA'])))
        out = out.drop('d2m', axis=1)
        out['SWIN'][out['SWIN']<0] = 0
        out['LWIN'][out['LWIN']<0] = 0
        
        ## do aggregation
        sum_vars = ['PRECIP']
        av_vars = ['SWIN', 'LWIN', 'RH', 'WS', 'PA', 'TA']
        day_vars = ['SWIN']
        res = '1D'
        
        dnorm_av = out[out.columns & av_vars].resample(res).mean()
        dnorm_sum = out[out.columns & sum_vars].resample(res).sum()
                
        # do soft maximum and minimum over the 24 hours        
        dmax = out[out.columns & (sum_vars + av_vars)].resample(res).apply(q_func(0.99))
        dmax.columns = dmax.columns + '_q99'
        # exclude rainfall and day-only vars from the global soft min
        dmin = out[out.columns & np.setdiff1d(av_vars, day_vars)].resample(res).apply(q_func(0.01))        
        dmin.columns = dmin.columns + '_q01'        
        # add day separations
        swin_daymean = out[out.columns & day_vars].resample(res).apply(nonzero_mean)        
        swin_daymean.columns = swin_daymean.columns + '_daymean'        
        # collect            
        out = (dnorm_av.merge(dnorm_sum, how='left', on='DATE_TIME')
            .merge(dmax, how='left', on='DATE_TIME')
            .merge(dmin, how='left', on='DATE_TIME')                
            .merge(swin_daymean, how='left', on='DATE_TIME'))
        out = out.reindex(sorted(out.columns), axis=1)    
        
        #out['LATITUDE'] = lat
        #out['LONGITUDE'] = lon
        #lat_gridbox_centre = float(latlon_ref.lat[chesstile_yx[0][0],chesstile_yx[1][0]].data)
        #lon_gridbox_centre = float(latlon_ref.lon[chesstile_yx[0][0],chesstile_yx[1][0]].data)
        #out['LATITUDE_gridbox'] = lat_gridbox_centre
        #out['LONGITUDE_gridbox'] = lon_gridbox_centre
        return out, chesstile_yx, tile_bounds

    '''
    def grab_multi_grids(self, start, end, latlon_ref, lats=None, lons=None, area_mask=None):
        ## process time slice        
        uniq_files, ymd = process_date_range(start, end, freq='D')
        
        if area_mask is None: # implying lats/lons is not None        
            yx_inds = []
            for i in range(len(lats)):
                chesstile_yx = find_chess_tile(lats[i], lons[i], latlon_ref)
                yx_inds.append(chesstile_yx)        
        
        infiles = [fldr2+f'/{v}/era5_{doy}_{v}.nc' for doy in uniq_files for v in self.var_names]
        
        # load one joined dataset
        dat = xr.open_mfdataset(infiles)
        
        # do all rescaling / processing globally
        dat['t2m'] -= 273.15 # K -> Celsius 
        dat['d2m'] -= 273.15 # K -> Celsius
        dat['sp'] /= 100. # Pa -> hPa
        dat['mtpr'] *= 60.*60. # mean kg m-2 s-1 -> total mm/hour
        dat['WS'] = np.sqrt(dat['u10']**2 + dat['v10']**2)
        dat['RH'] = 100 * (np.exp((17.625 * dat['d2m'])/(243.04 + dat['d2m'])) / 
            np.exp((17.625 * dat['t2m'])/(243.04 + dat['t2m'])))
        dat = dat.rename(self.map_names).drop(['u10', 'v10', 'd2m'])
        
        ## TOO LARGE!
        ## best way would be to calculate the daily statistics on the fly
        ## and not output the hourly. Even that might be too large to do
        ## the whole UK land mask. Might need to subdivide the mask into 
        ## "regions" and then run each region forward in time separately.
        sum_vars = ['PRECIP']
        av_vars = ['SWIN', 'LWIN', 'RH', 'WS', 'Tsoil', 'PA', 'TA']
        daynight_vars = ['SWIN'] # 'TA'
        colnames = []
        num_feats = len(sum_vars)*2 + len(av_vars)*3 + len(daynight_vars)*2
        num_locs = np.where(area_mask)[0].shape[0]
        lats = latlon_ref.lat.data[area_mask]
        lons = latlon_ref.lon.data[area_mask]
        out_arr = np.zeros((len(uniq_files)*num_locs, num_feats+3), dtype=object)
        for k, doy in enumerate(uniq_files):
            out_arr[k*num_locs:((k+1)*num_locs), 0] = pd.to_datetime(doy, utc=True, format='%Y%m%d')
            out_arr[k*num_locs:((k+1)*num_locs), 1] = lats
            out_arr[k*num_locs:((k+1)*num_locs), 2] = lons
        colnames += ['DATE_TIME', 'LATITUDE', 'LONGITUDE']
        ix = 3
        for j, v in enumerate(self.out_names):
            if v in sum_vars:
                # do "standard" 24 hour sum
                subdat = dat[v].resample(time='1D').sum(dim='time')
                for k in range(len(uniq_files)):
                    out_arr[k*num_locs:((k+1)*num_locs), ix] =  subdat[k,:,:].data[area_mask].compute()
                colnames.append(v)
                ix+=1
                # do soft maximum
                subdat = dat[v].resample(time='1D').quantile(0.975, dim='time')
                for k in range(len(uniq_files)):
                    out_arr[k*num_locs:((k+1)*num_locs), ix] =  subdat[k,:,:].data[area_mask].compute()
                colnames.append(v+'_q975')
                ix+=1
            elif v in daynight_vars:
                # do "standard" 24 hour mean
                subdat = dat[v].resample(time='1D').mean(dim='time')
                for k in range(len(uniq_files)):
                    out_arr[k*num_locs:((k+1)*num_locs), ix] =  subdat[k,:,:].data[area_mask].compute()
                colnames.append(v)
                ix+=1
                # do nonzero daytime mean
                subdat = dat[v].where(dat[v]>0).resample(time='1D').mean()
                for k in range(len(uniq_files)):
                    out_arr[k*num_locs:((k+1)*num_locs), ix] =  subdat[k,:,:].data[area_mask].compute()
                colnames.append(v+'_daymean')
                ix+=1
            else:
                # do "standard" 24 hour mean or sum
                subdat = dat[v].resample(time='1D').mean(dim='time')
                for k in range(len(uniq_files)):
                    out_arr[k*num_locs:((k+1)*num_locs), ix] =  subdat[k,:,:].data[area_mask].compute()
                colnames.append(v)
                ix+=1
                # do soft maximum
                subdat = dat[v].resample(time='1D').quantile(0.975, dim='time')
                for k in range(len(uniq_files)):
                    out_arr[k*num_locs:((k+1)*num_locs), ix] =  subdat[k,:,:].data[area_mask].compute()
                colnames.append(v+'_q975')
                ix+=1
                # and soft minimum
                subdat = dat[v].resample(time='1D').quantile(0.025, dim='time')
                for k in range(len(uniq_files)):
                    out_arr[k*num_locs:((k+1)*num_locs), ix] =  subdat[k,:,:].data[area_mask].compute()
                colnames.append(v+'_q025')
                ix+=1

    out_df = pd.DataFrame(out_arr, 
        columns=colnames]
    )
    return out_df
    '''




class ChessMetData():

    def __init__(self, use_vars=None):
        # 1km        
        self.var_names = ['dtr', 'huss', 'precip', 'psurf', 
                          'rlds', 'rsds', 'sfcWind', 'tas']        
        self.use_vars = use_vars

    def set_use_vars(self, use_vars):
        self.use_vars = use_vars

    def grab_grid_slice(self, lat, lon, start, end, latlon_ref, chesstile_yx=None):
        ## process year, month and day of start and end    
        fpick_s, year_s, month_s, day_s = process_date(start)
        fpick_e, year_e, month_e, day_e = process_date(end)        
        
        ## change to using process_date_range for any number of files
        onefile = fpick_s==fpick_e

        if chesstile_yx is None:
            chesstile_yx = find_chess_tile(lat, lon, latlon_ref)        

        ## load up vars and subset
        out_dat = {}
        for var in self.use_vars:
            # load file(s) for var
            fnm = f'{year_s}/chess-met_{var}_gb_1km_daily_{fpick_s}.nc'
            dat = xr.open_dataset(fldr + fnm)
            northing_bnds = dat.y_bnds[chesstile_yx[0]].values.squeeze()
            easting_bnds = dat.x_bnds[chesstile_yx[1]].values.squeeze()   
            dat = dat[var][:, chesstile_yx[0], chesstile_yx[1]] # [t,y,x]     

            if not onefile:
                fnm = f'{year_e}/chess-met_{var}_gb_1km_daily_{fpick_e}.nc'
                dat_e = xr.open_dataset(fldr + fnm)
                dat_e = dat_e[var][:, chesstile_yx[0], chesstile_yx[1]] # [t,y,x]
                dat = xr.concat([dat, dat_e], dim="time")

            # subset to desired time period
            dat = dat.sel(time=slice(start, end))
            out_dat[var] = dat.values.squeeze()
            out_dat['DATE_TIME'] = dat.time.squeeze()
            
          
        out_df = pd.DataFrame(out_dat)
        out_df = indexify_datetime(out_df, 'DATE_TIME')
        ## convert from mean flux [kg m-2 s-1] to total [mm day-1]
        # 'comment': 'The precipitation flux on a given day refers to the 
        #             mean precipitation in the 24 hours between 09:00 UTC
        #             on that day and 09:00 UTC on the following day.'
        secs_per_day = 60*60*24
        out_df = out_df.assign(precip=lambda x: x.precip * secs_per_day)        
        return out_df, chesstile_yx, tile_bounds

    def process_date(self, date):
        year = date.year
        month = date.month
        day = date.day
        # find month end day for file name
        if month==2 and year%4==0:
            # leap year
            day_me = '29'
        else:
            day_me = days_in_month()[month-1]
        month = zeropad_strint(month)
        day = zeropad_strint(day)
        fpick = f'{year}{month}01-{year}{month}{day_me}'
        return fpick, year, month, day

    def process_date_range(self, start_date, end_date, freq='D'):
        drange = pd.date_range(start_date, end_date, freq=freq)
        fpicks = []
        year_month_day = []
        for d in drange:
            f, y, m, d = process_date(d)
            fpicks.append(f)
            year_month_day.append((y,m,d))
        unique_files = np.unique(fpicks)
        return unique_files, year_month_day    
