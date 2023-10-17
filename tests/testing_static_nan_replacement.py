if load_binary_batch is False:
    dg.parent_pixels = dg.read_parent_pixel_day(batch_type=batch_type)
else:
    bfn = dg.bin_batches[batch_type][np.random.randint(0, len(dg.bin_batches[batch_type]))]            
    dg.parent_pixels = pd.read_pickle(bfn)

# extract timestamp
dg.td = pd.to_datetime(dg.parent_pixels.time[0].values)

# calculate constraint grids
dg.prepare_constraints()
        
# load the correct hi-res daily [or hourly] precipitation field
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
# subset to correct day
dg.precip_dat = dg.precip_dat.isel(time=dg.td.day-1).rainfall.load()
# get the corresponding interpolated mtpr (PRECIP) from ERA5:
era5_precip = xr.open_dataset(f'/gws/nopw/j04/hydro_jules/data/uk/driving_data/era5/bn_grid/mtpr/era5_{dg.td.year}{zeropad_strint(dg.td.month)}{zeropad_strint(dg.td.day)}_mtpr.nc')
era5_precip = era5_precip.mtpr.sum(dim='time')*60*60 # convert from hourly kg/m2/s to daily total mm
'''TODO''' # infill preip sea nans from era5 interp precip! '''TODO'''
indices = np.where(np.isnan(dg.precip_dat.values))
dg.precip_dat.values[indices] = era5_precip.values[indices]


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

indices = torch.where(torch.isnan(fine_inputs))
## stepping through dg.fine_variable_order to treat NaNs correctly
# 0: land frac is fine as 0,1
pass
# 1: elev, sea NaNs should be elev==0
var = 'elev'
ii = np.where(np.array(dg.fine_variable_order)==var)[0][0]
indices = torch.where(torch.isnan(fine_inputs[:,:,:,ii]))
fine_inputs[:,:,:,ii][indices] = (0 - nm.s_means.loc[var]) / nm.s_stds.loc[var]
# 2: stdev, sea NaNs should be stdev==0
var = 'stdev'
ii = np.where(np.array(dg.fine_variable_order)==var)[0][0]
indices = torch.where(torch.isnan(fine_inputs[:,:,:,ii]))
fine_inputs[:,:,:,ii][indices] = (0 - nm.s_means.loc[var]) / nm.s_stds.loc[var]
# 3: slope, sea NaNs should be slope==0
var = 'slope'
ii = np.where(np.array(dg.fine_variable_order)==var)[0][0]
indices = torch.where(torch.isnan(fine_inputs[:,:,:,ii]))
fine_inputs[:,:,:,ii][indices] = (0 - nm.s_means.loc[var]) / nm.s_stds.loc[var]
# 4: aspect, sea NaNs are tricky, aspect is "straight up", stick with uniform noise [-0.5, 0.5]
#asp_test = dg.height_grid.aspect/360. - 0.5
var = 'aspect'
ii = np.where(np.array(dg.fine_variable_order)==var)[0][0]
indices = torch.where(torch.isnan(fine_inputs[:,:,:,ii]))
fine_inputs[:,:,:,ii][indices] = torch.rand(indices[0].size()) - 0.5
# 5: topi, topi is defined as log(catchment area/ slope)
# REMOVE THIS VARIABLE!
# 6: stdtopi, stdtopi is defined by catchment area/slope
# REMOVE THIS VARIABLE!
# 7: fdepth, soil saturated conductivity decay
# REMOVE THIS VARIABLE!
# 8: rainfall, sea NaNs are very hard. We could simply interpolate the 
# ERA5 PRECIP to 1km and input over the sea pixels?
# 9, 10, 11, 12: year/hour sin and cosine
pass
# 13, 14: lat lon grid?
pass
