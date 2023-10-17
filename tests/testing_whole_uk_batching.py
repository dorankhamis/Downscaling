#def get_batch(self, batch_size=1, batch_type='train', load_binary_batch=True):
from setupdata import *
load_binary_batch = True
batch_type = 'train'
if load_binary_batch is False:
    datgen.parent_pixels = datgen.read_parent_pixel_day(batch_type=batch_type)
else:
    bfn = datgen.bin_batches[batch_type][np.random.randint(0, len(datgen.bin_batches[batch_type]))]            
    datgen.parent_pixels = pd.read_pickle(bfn)

# extract timestamp
datgen.td = pd.to_datetime(datgen.parent_pixels.time[0].values)

with warnings.catch_warnings():            
    warnings.simplefilter("ignore")
    # ignore warnings thrown by interp_like() which seem to be needless
    
    ### constraints for training help 
    ## Air Temp
    # reduce to sea level with lapse rate of -0.006 K/m
    lapse_val = -0.0065
    datgen.parent_pixels['t_sealevel'] = (['time', 'y', 'x'],
                                       datgen.parent_pixels['t2m'].values)
    datgen.parent_pixels['t_sealevel'] = datgen.parent_pixels['t_sealevel'] - datgen.parent_pixels['elev'] * lapse_val

    # bicubic spline to interpolate from 25km to 1km... cubic doesn't work here!
    t_sealevel_interp = datgen.parent_pixels['t_sealevel'].interp_like(datgen.chess_grid, method='linear')

    # adjust to the 1km elevation using same lapse rate
    datgen.t_1km_elev = t_sealevel_interp + datgen.height_grid['elev'] * lapse_val

    ## Air Pressure:
    # integral of hypsometric equation using the 1km Air Temp?
    T_av = 0.5*(t_sealevel_interp + datgen.t_1km_elev) # K
    p_1 = 1013 # hPa, standard sea level pressure value
    R = 287 # J/kg·K = kg.m2/s2/kg.K = m2/s2.K  # specific gas constant of dry air
    g = 9.81 # m/s2 
    datgen.p_1km_elev = p_1 * np.exp(-g * datgen.height_grid['elev'] / (R * T_av))
    del(T_av)
    del(t_sealevel_interp)

    ## Relative Humidity
    # Assumed constant with respect to elevation, so can be simply 
    # interpolated from 25km to 1km using a bicubic spline.
    t2m_C = datgen.parent_pixels['t2m'] - 273.15 # K -> Celsius 
    d2m_C = datgen.parent_pixels['d2m'] - 273.15 # K -> Celsius
    rh_25km = relhum_from_dewpoint(t2m_C, d2m_C)
    # rh_25km = 100 * (np.exp((17.625 * d2m_C)/(243.04 + d2m_C)) / 
        # np.exp((17.625 * t2m_C)/(243.04 + t2m_C)))
    datgen.rh_1km_interp = rh_25km.interp_like(datgen.chess_grid, method='linear')
    del(rh_25km)
        
# load the correct hi-res daily [or hourly] precipitation field
precip_file = glob.glob(precip_fldr + f'rainfall_hadukgrid_uk_1km_day_{datgen.td.year}{zeropad_strint(datgen.td.month)}01*.nc')[0]
precip_dat = xr.open_dataset(precip_file) # total in mm
datgen.precip_dat = precip_dat.loc[dict(
    projection_y_coordinate = precip_dat.projection_y_coordinate[
        (precip_dat.projection_y_coordinate <= float(datgen.chess_grid.y.max().values)) & 
        (precip_dat.projection_y_coordinate >= float(datgen.chess_grid.y.min().values))],
    projection_x_coordinate = precip_dat.projection_x_coordinate[
        (precip_dat.projection_x_coordinate <= float(datgen.chess_grid.x.max().values)) & 
        (precip_dat.projection_x_coordinate >= float(datgen.chess_grid.x.min().values))]
)]


# choose a time slice
it = np.random.randint(0, 24)

# Divide the UK up into dim_l x dim_l tiles
def define_tile_start_inds(full_length, tile_size):
    n_tiles = full_length // tile_size
    pixels_leftover = full_length % tile_size
    tile_starts = list(range(0, full_length, tile_size))[:n_tiles]
    if pixels_leftover>0: tile_starts.append(full_length - tile_size)
    return tile_starts

ixs = define_tile_start_inds(datgen.parent_pixels.x.shape[0], datgen.dim_l)
iys = define_tile_start_inds(datgen.parent_pixels.y.shape[0], datgen.dim_l)
coarse_inputs = []
fine_inputs = []
for ix in ixs:
    for iy in iys:
        coarse_input, fine_input = datgen.get_input_data(ix, iy, it)
        coarse_inputs.append(coarse_input)
        fine_inputs.append(fine_input)

coarse_inputs = torch.stack(coarse_inputs, dim=0)
fine_inputs = torch.stack(fine_inputs, dim=0)

# replace NaNs (usually the ocean) with random noise? Or padding?        
indices = torch.where(torch.isnan(coarse_inputs))
coarse_inputs[indices] = torch.rand(indices[0].size())*2 - 1
indices = torch.where(torch.isnan(fine_inputs))
fine_inputs[indices] = torch.rand(indices[0].size())*2 - 1
       
#return coarse_inputs, fine_inputs, station_targets, constraint_targets
return (coarse_inputs.permute(0,3,1,2), fine_inputs.permute(0,3,1,2),
        station_targets, constraint_targets.permute(0,3,1,2))
