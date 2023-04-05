dg = data_generator([2015, 2016, 2017, 2018], [2019, 2020], [2021],                     
                     dim_l=4, res=25000, scale=25)

'''
ideally want to refactor this so that when we save a binary output
we don't save the static and station data associated with it
as this should be quick to retrieve and will reduce the storage 
associated with training/validation batches
'''

# def get_batch()
batch_type = 'train'
if batch_type=='train':
    dg.td = generate_random_date(dg.train_years)
elif batch_type=='val':
    dg.td = generate_random_date(dg.val_years)
elif batch_type=='test':
    dg.td = generate_random_date(dg.heldout_years)
date_string = f'{dg.td.year}{zeropad_strint(dg.td.month)}{zeropad_strint(dg.td.day)}'

## load ERA5 data for that date and trim lat/lon
era5_vars = dg.var_name_map[(dg.var_name_map['coarse']!='ws') &
                              (dg.var_name_map['coarse']!='rh')]['coarse']
era5_vars = list(era5_vars) + ['u10', 'v10', 'd2m']
        
era5_filelist = [f'{era5_fldr}/{var}/era5_{date_string}_{var}.nc' for var in era5_vars]
dg.era5_dat = xr.open_mfdataset(era5_filelist)

# because these are already on the BNG at 1km, find averages of 25x25 squares
dg.parent_pixels = dg.parent_pixels.assign_coords(
    {'time':dg.era5_dat.time.values}
)
ys = dg.parent_pixels.y.shape[0]
xs = dg.parent_pixels.x.shape[0]
ts = dg.parent_pixels.time.shape[0]
for var in era5_vars:
    dg.parent_pixels[var] = (['time', 'y', 'x'],
                              np.ones((ts, ys, xs), dtype=np.float32)*np.nan)
    source = dg.era5_dat[var].values
    dg.parent_pixels[var] = (('time', 'y', 'x'), 
        skimage.measure.block_reduce(
            source, (1, dg.scale, dg.scale), np.mean
        )[:ts,:ys,:xs] # trim right/bottom edges
    )
del(source)
del(dg.era5_dat)

## if loading raw lat/lon projection
# constrain to BNG lat/lon limits
# dg.era5_dat = dg.era5_dat.loc[dict(
    # longitude = dg.era5_dat.longitude[
        # (dg.era5_dat.longitude < dg.lonmax) & (dg.era5_dat.longitude > dg.lonmin)],
    # latitude = dg.era5_dat.latitude[
        # (dg.era5_dat.latitude < dg.latmax) & (dg.era5_dat.latitude > dg.latmin)]
# )]
# reproject and regrid onto 25km BNG
# dg.era5_dat = dg.era5_dat.rio.write_crs(rasterio.crs.CRS.from_epsg(dg.wgs84_epsg))
# dg.era5_dat = dg.era5_dat.rio.reproject(f"EPSG:{dg.bng_epsg}")
# dg.era5_dat = dg.era5_dat.interp(y=dg.y_chess_25k, x=dg.x_chess_25k)

## now save parent pixels as a "pickled batch" 
pd.to_pickle(dg.parent_pixels, './data/testbatch.pkl')

# load the parent pixels batch and retrieve the datetime stamp
dg.parent_pixels = pd.read_pickle('./data/testbatch.pkl')
dg.td = pd.to_datetime(dg.parent_pixels.time[0].values)

### constraints for training help 
## this could be moved to sample creation from a loaded binary batch?
if False:
    ## Air Temp
    # reduce to sea level with lapse rate of -0.006 K/m
    lapse_val = -0.0065
    dg.parent_pixels['t_sealevel'] = (['time', 'y', 'x'],
                                       dg.parent_pixels['t2m'].values)
    dg.parent_pixels['t_sealevel'] = dg.parent_pixels['t_sealevel'] - dg.parent_pixels['elev'] * lapse_val

    # bicubic spline to interpolate from 25km to 1km... cubic doesn't work here!
    t_sealevel_interp = dg.parent_pixels['t_sealevel'].interp_like(dg.chess_grid, method='linear')

    # adjust to the 1km elevation using same lapse rate
    dg.t_1km_elev = t_sealevel_interp + dg.height_grid['elev'] * lapse_val

    ## Air Pressure:
    # integral of hypsometric equation using the 1km Air Temp?
    T_av = 0.5*(t_sealevel_interp + dg.t_1km_elev) # K
    p_1 = 1013 # hPa, standard sea level pressure value
    R = 287 # J/kg·K = kg.m2/s2/kg.K = m2/s2.K  # specific gas constant of dry air
    g = 9.81 # m/s2 
    dg.p_1km_elev = p_1 * np.exp(-g * dg.height_grid['elev'] / (R * T_av))
    del(T_av)
    del(t_sealevel_interp)

    ## Relative Humidity
    # Assumed constant with respect to elevation, so can be simply 
    # interpolated from 25km to 1km using a bicubic spline.
    t2m_C = dg.parent_pixels['t2m'] - 273.15 # K -> Celsius 
    d2m_C = dg.parent_pixels['d2m'] - 273.15 # K -> Celsius
    rh_25km = 100 * (np.exp((17.625 * d2m_C)/(243.04 + d2m_C)) / 
        np.exp((17.625 * t2m_C)/(243.04 + t2m_C)))
    dg.rh_1km_interp = rh_25km.interp_like(dg.chess_grid, method='linear')
    del(rh_25km)

# load the correct hi-res daily [or hourly] precipitation field
## this could be moved to sample creation from a loaded binary batch?
if False:    
    precip_file = glob.glob(precip_fldr + f'rainfall_hadukgrid_uk_1km_day_{dg.td.year}{zeropad_strint(dg.td.month)}01*.nc')[0]
    precip_dat = xr.open_dataset(precip_file) # total in mm
    dg.precip_dat = precip_dat.loc[dict(
        projection_y_coordinate = precip_dat.projection_y_coordinate[
            (precip_dat.projection_y_coordinate <= float(dg.chess_grid.y.max().values)) & 
            (precip_dat.projection_y_coordinate >= float(dg.chess_grid.y.min().values))],
        projection_x_coordinate = precip_dat.projection_x_coordinate[
            (precip_dat.projection_x_coordinate <= float(dg.chess_grid.x.max().values)) & 
            (precip_dat.projection_x_coordinate >= float(dg.chess_grid.x.min().values))]
    )]

# construct batch from individual samples
## this could be moved to sample creation from a loaded binary batch?
if False:
    coarse_inputs = []
    fine_inputs = []
    station_targets = []
    constraint_targets = []
    for b in range(batch_size):
        coarse_input, fine_input, station_target, constraints = dg.get_sample(batch_type=batch_type)            
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

##############################

##############################


#def get_sample(self, batch_type='train'):
# reuse a single ERA5 file/date and just grab different times and spatial chunks
# for each batch to cut down on loading/reprojecting time! 

# choose a station and find its 1km pixel
if batch_type=='train' or batch_type=='val':
    SID = np.random.choice(dg.train_sites)
elif batch_type=='test':
    SID = np.random.choice(dg.train_sites + dg.heldout_sites) 
targ_site = dg.site_metadata[dg.site_metadata['SITE_ID']==SID]
targ_loc = dg.site_metadata[dg.site_metadata['SITE_ID']==SID][['LATITUDE','LONGITUDE']]
targ_yx = np.where(dg.parent_pixels.pixel_id.values == targ_site['parent_pixel_id'].values)

#print(SID)
#print(targ_yx)

# grab a random dim_l x dim_l tile that contains that 1km pixel
ix = np.random.randint(max(0, targ_yx[1][0] - dg.dim_l + 1),
                       min(dg.parent_pixels.x.shape[0] - dg.dim_l + 1, targ_yx[1][0] + 1))
iy = np.random.randint(max(0, targ_yx[0][0] - dg.dim_l + 1),
                       min(dg.parent_pixels.y.shape[0] - dg.dim_l + 1, targ_yx[0][0] + 1))
it = np.random.randint(0, 24)
subdat = dg.parent_pixels.isel(time=it, y=range(iy, iy+dg.dim_l), x=range(ix, ix+dg.dim_l))
timestamp = pd.to_datetime(dg.parent_pixels.time.values[it], utc=True)
print("NOTE: Should add timeofday/timeofyear signals to static data!")

# find any other stations that lie inside the chosen tile and their pixels
parents = subdat.pixel_id.values.flatten()
if batch_type=='train' or batch_type=='val':
    contained_sites = dg.site_metadata[(dg.site_metadata['SITE_ID'].isin(dg.train_sites)) & 
                                         (dg.site_metadata['parent_pixel_id'].isin(parents))]
elif batch_type=='test':
    contained_sites = dg.site_metadata[(dg.site_metadata['SITE_ID'].isin(dg.train_sites + dg.heldout_sites)) &
                                         (dg.site_metadata['parent_pixel_id'].isin(parents))]
# then we use all these sites as targets for the loss calculation
# (if they have data for the particular variable on the particular day)        

# subset hi-res fields to the chosen coarse tile
x_inds = np.intersect1d(np.where(dg.chess_grid.x.values < int(subdat.x.max().values) + dg.res//2),
                        np.where(dg.chess_grid.x.values > int(subdat.x.min().values) - dg.res//2))
y_inds = np.intersect1d(np.where(dg.chess_grid.y.values < int(subdat.y.max().values) + dg.res//2),
                        np.where(dg.chess_grid.y.values > int(subdat.y.min().values) - dg.res//2))
sub_chess = dg.chess_grid.isel(y=y_inds, x=x_inds)
sub_topog = dg.height_grid.isel(y=y_inds, x=x_inds)
sub_precip = dg.precip_dat.isel(time=timestamp.day-1, # due to accumulating label? Check this
                                  projection_y_coordinate=y_inds,
                                  projection_x_coordinate=x_inds)
                                  
# also subset the constraint fields
sub_temp_constraint = dg.t_1km_elev.isel(time=it, y=y_inds, x=x_inds)
sub_pres_constraint = dg.p_1km_elev.isel(time=it, y=y_inds, x=x_inds)
sub_relh_constraint = dg.rh_1km_interp.isel(time=it, y=y_inds, x=x_inds)

# find location of contained sites in local subset and pull out data
contained_sites = contained_sites.set_index('SITE_ID')
contained_sites['sub_x'] = -1
contained_sites['sub_y'] = -1
for var in dg.var_name_map.fine:
    if var=='TD': continue
    contained_sites[var] = -1
for sid in contained_sites.index:
    this_x = np.where(x_inds == contained_sites.loc[sid,'chess_x'])[0]
    this_y = np.where(y_inds == contained_sites.loc[sid,'chess_y'])[0]
    contained_sites.loc[sid,'sub_x'] = int(this_x)
    contained_sites.loc[sid,'sub_y'] = int(this_y)
    try:
        this_dat = dg.site_data[sid].loc[timestamp, :]
    except:
        this_dat = pd.Series(np.nan, dg.var_name_map.fine)            
    for var in dg.var_name_map.fine:
        if var=='TD': continue                
        contained_sites.loc[sid,var] = this_dat[var]

# convert wind vectors and dewpoint temp to wind speed and relative humidity
subdat['ws'] = np.sqrt(np.square(subdat['v10']) + np.square(subdat['u10']))
subdat['t2m'] -= 273.15 # K -> Celsius 
subdat['d2m'] -= 273.15 # K -> Celsius
subdat['sp'] /= 100.    # Pa -> hPa
# RH calc from https://www.omnicalculator.com/physics/relative-humidity        
subdat['rh'] = 100 * (np.exp((17.625 * subdat['d2m'])/(243.04 + subdat['d2m'])) / 
    np.exp((17.625 * subdat['t2m'])/(243.04 + subdat['t2m'])))
subdat = subdat.drop(['u10','v10', 'd2m'])
# subdat['msdwlwrf'][subdat['msdwlwrf']<0] = 0
# subdat['msdwswrf'][subdat['msdwswrf']<0] = 0


## normalise data using norm dict
# met data
rad_norm = dg.normalisations['rad_norm']
temp_mu = dg.normalisations['temp_mu']
temp_sd = dg.normalisations['temp_sd']
p_mu = dg.normalisations['p_mu']
p_sd = dg.normalisations['p_sd']
rh_norm = dg.normalisations['rh_norm']
ws_mu = dg.normalisations['ws_mu']
ws_sd = dg.normalisations['ws_sd']

subdat['t2m'] = (subdat['t2m'] - temp_mu) / temp_sd
subdat['sp'] = (subdat['sp'] - p_mu) / p_sd
subdat['msdwswrf'] = subdat['msdwswrf'] / rad_norm
subdat['msdwlwrf'] = subdat['msdwlwrf'] / rad_norm
subdat['ws'] = (subdat['ws'] - ws_mu) / ws_sd
subdat['rh'] = subdat['rh'] / rh_norm

# constraints
sub_temp_constraint = (sub_temp_constraint - 273.15 - temp_mu) / temp_sd
sub_pres_constraint = (sub_pres_constraint - p_mu) / p_sd
sub_relh_constraint = sub_relh_constraint / rh_norm

# hi res data
precip_norm = dg.normalisations['precip_norm']
sub_precip['rainfall'] = sub_precip['rainfall'] / precip_norm
sub_topog['aspect'] = sub_topog['aspect'] / 360. - 0.5
for var in dg.elev_vars:
    if var=='aspect': continue
    sub_topog[var] = (
        (sub_topog[var] - dg.normalisations['s_means'].loc[var]) / 
        dg.normalisations['s_stds'].loc[var]
    )

# create tensors with batch index and channel dim last
coarse_input = torch.zeros((subdat['t2m'].shape[0],subdat['t2m'].shape[1], 0)) # (Y, X, C)
for var in dg.var_name_map.coarse:            
    coarse_input = torch.cat(
        (coarse_input, 
         torch.from_numpy(subdat[var].values).to(torch.float32)[...,None]
        ), dim = -1)
# therefore coarse scale variable order (in cosmos parlance) is
dg.coarse_variable_order = list(dg.var_name_map['fine'])

# get static inputs at the high resolution
# how do we deal with NaNs on the sea?
fine_input = torch.from_numpy(sub_chess.landfrac.values).to(torch.float32)[...,None] # (Y, X, C)
for var in dg.elev_vars:
    fine_input = torch.cat(
        (fine_input, 
         torch.from_numpy(sub_topog[var].values).to(torch.float32)[...,None]
        ), dim = -1)
# and join on precipitation
fine_input = torch.cat(
    (fine_input, 
     torch.from_numpy(sub_precip['rainfall'].values).to(torch.float32)[...,None]
    ), dim = -1)
# therefore fine scale variable order is
dg.fine_variable_order = ['landfrac'] + dg.elev_vars + ['rainfall']

# define station targets
station_targets = contained_sites[
    ['sub_x', 'sub_y'] + list(dg.var_name_map.fine)
]

# combine constraints into tensor
# check the ".value" with not variable name in these!
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
dg.constraint_variable_order = ['TA', 'PA', 'RH']

# the target for conserving area averages is just
# [i*dg.scale:((i+1)dg.scale)] across the 200x200 output (200==dg.dim_l*dg.scale)
# and matching with each pixel of the 8x8 coarse input (8==dg.dim_l)
return coarse_input, fine_input, station_targets, constraints

