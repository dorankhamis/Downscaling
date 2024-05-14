from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics.pairwise import rbf_kernel
batch_type = "run"
date_string = "20140101"
it = 10
timestep = "hourly"
context_frac = 0.7
from downscaling.params2 import normalisation as nm

if type(var)==str: var = [var]
dg.parent_pixels = {}

if date_string is None:            
    dg.parent_pixels['hourly'] = dg.read_parent_pixel_day(batch_type=batch_type)            
    dg.td = pd.to_datetime(dg.parent_pixels[timestep].time[0].values, utc=True)
else:
    dg.parent_pixels['hourly'] = dg.read_parent_pixel_day(batch_type=batch_type,
                                                              date_string=date_string)               
    dg.td = pd.to_datetime(date_string, format='%Y%m%d', utc=True)        

## process and normalise ERA5 parent_pixels
dg.prepare_era5_pixels(var, batch_tsteps=timestep)

## get additional fine scale met vars for input
dg.prepare_fine_met_inputs(var, timestep)

ix = None
iy = None
## trim down the full region
subdat, x_inds, y_inds, timestamp = dg.get_subset_region(
    it, var=var, ix=ix, iy=iy, timestep=timestep
)
        
## use physically reasoned constraints
constraints = dg.get_constraints(x_inds, y_inds, it, var)
constraints = torch.stack(constraints, dim=-1)

## get station targets within the dim_l x dim_l tile        
(station_targets, station_npts, 
    station_data, context_locations) = dg.get_station_targets(
    subdat, x_inds, y_inds, timestamp, var,            
    batch_type=batch_type,
    context_frac=context_frac,            
    timestep=timestep            
)

## load input data
(coarse_input, fine_input, lat_grid, lon_grid) = dg.get_input_data(
    var, subdat, x_inds, y_inds, it, timestep=timestep
)


station_targets = station_targets['context']


###################
baseline_density = 0.25

X1 = np.where(np.ones((constraints.shape[0], constraints.shape[1])))
X1 = np.hstack([X1[0][...,np.newaxis],
                X1[1][...,np.newaxis]])
X1 = X1
kernel = RBF(25)
thisvar = ['UX', 'VY'] if var[0]=='WS' else var
interleaved_out = []
for j, vv in enumerate(thisvar):
    density = np.zeros((1, constraints.shape[0]*constraints.shape[1]), dtype=np.float32)
    value = np.zeros((1, constraints.shape[0]*constraints.shape[1]), dtype=np.float32)
    for i in range(station_targets.shape[0]):
        this_density = kernel(station_targets[['sub_y', 'sub_x']].iloc[i].values, X1)
        this_value = station_targets.iloc[i][vv]
        density += this_density
        value += this_value*this_density
        
    density = density[0,:].reshape(constraints.shape[0], constraints.shape[1])
    value = value[0,:].reshape(constraints.shape[0], constraints.shape[1])
                
    # combining era5 interp constraint with point obs using baseline min density for constraint            
    density = density + baseline_density
    value = value + constraints.numpy()[:,:,j] * baseline_density
    
    value = value / density
    density = density / density.max()
    
    plt.imshow(value[::-1,:])
    plt.show()
    plt.imshow(density[::-1,:])
    plt.show()
    
    interleaved_out.append(torch.from_numpy(value))
    interleaved_out.append(torch.from_numpy(density))

dist_vd = torch.stack(interleaved_out, dim=-1)
    

# examining joined density/val for x,y,elev
xy_norm = 50.
X1 = np.where(np.ones((constraints.shape[0], constraints.shape[1])))
X1 = np.hstack([X1[0][...,np.newaxis],
                X1[1][...,np.newaxis]])
X2 = fine_input.numpy().reshape((fine_input.shape[0]*fine_input.shape[1],1))
X3 = np.hstack([X1 / xy_norm, X2])

kernel1 = RBF(50)
kernel2 = RBF(0.5)
kernel3 = RBF(1)

i = 1
site_dat = station_targets.iloc[i].copy()
site_dat[['sub_y', 'sub_x']] = site_dat[['sub_y', 'sub_x']] / xy_norm
site_dat['ALTITUDE'] = (site_dat['ALTITUDE'] - nm.s_means.elev) / nm.s_stds.elev

density1 = kernel1(site_dat[['sub_y', 'sub_x']].values * xy_norm, X1)
value1 = site_dat[vv] * density1
density2 = kernel2(site_dat[['ALTITUDE']].values, X2)
value2 = site_dat[vv] * density2
density3 = kernel3(site_dat[['sub_y', 'sub_x', 'ALTITUDE']].values, X3)
value3 = site_dat[vv] * density3

density1 = density1[0,:].reshape(constraints.shape[0], constraints.shape[1])
value1 = value1[0,:].reshape(constraints.shape[0], constraints.shape[1])
density2 = density2[0,:].reshape(constraints.shape[0], constraints.shape[1])
value2 = value2[0,:].reshape(constraints.shape[0], constraints.shape[1])
density3 = density3[0,:].reshape(constraints.shape[0], constraints.shape[1])
value3 = value3[0,:].reshape(constraints.shape[0], constraints.shape[1])

# or combine 1 and 2
density4 = density1 * density2
density4 = density4 / density4.max()
value4 = site_dat[vv] * density4

fig, ax = plt.subplots(1,4)
ax[0].imshow(value1[::-1,:])
ax[1].imshow(value2[::-1,:])
ax[2].imshow(value3[::-1,:])
ax[3].imshow(value4[::-1,:])
plt.show()

fig, ax = plt.subplots(1,4)
ax[0].imshow(density1[::-1,:])
ax[1].imshow(density2[::-1,:])
ax[2].imshow(density3[::-1,:])
ax[3].imshow(density4[::-1,:])
plt.show()


# doing for all sites and joining to constraint with baseline density
X1 = np.where(np.ones((constraints.shape[0], constraints.shape[1])))
X1 = np.hstack([X1[0][...,np.newaxis],
                X1[1][...,np.newaxis]])
X2 = fine_input.numpy().reshape((fine_input.shape[0]*fine_input.shape[1],1))
#X3 = np.hstack([X1 / xy_norm, X2])
kernel1 = RBF(50)
kernel2 = RBF(0.3)
thisvar = ['UX', 'VY'] if var[0]=='WS' else var
interleaved_out = []
for j, vv in enumerate(thisvar):
    density = np.zeros((1, constraints.shape[0]*constraints.shape[1]), dtype=np.float32)
    value = np.zeros((1, constraints.shape[0]*constraints.shape[1]), dtype=np.float32)
    for i in range(station_targets.shape[0]):
        site_dat = station_targets.iloc[i].copy()
        site_dat[['sub_y', 'sub_x']] = site_dat[['sub_y', 'sub_x']]
        site_dat['ALTITUDE'] = (site_dat['ALTITUDE'] - nm.s_means.elev) / nm.s_stds.elev
        
        density1 = kernel1(site_dat[['sub_y', 'sub_x']].values, X1)        
        density2 = kernel2(site_dat[['ALTITUDE']].values, X2)
        this_density = density1 * density2
        this_density = this_density / this_density.max()
        this_value = site_dat[vv]
                
        density += this_density
        value += this_value * this_density
        
    density = density[0,:].reshape(constraints.shape[0], constraints.shape[1])
    value = value[0,:].reshape(constraints.shape[0], constraints.shape[1])
                
    # combining era5 interp constraint with point obs using baseline min density for constraint            
    density = density + baseline_density
    value = value + constraints.numpy()[:,:,j] * baseline_density
    
    value = value / density # do this before normalising the density!
    density = density / density.max()
    
    fig, ax = plt.subplots(1,3)
    ax[0].imshow(value[::-1,:])
    ax[1].imshow(constraints.numpy()[::-1,:,j])
    ax[2].imshow(density[::-1,:])
    plt.show()
    
    interleaved_out.append(torch.from_numpy(value))
    interleaved_out.append(torch.from_numpy(density))
elev_vd = torch.stack(interleaved_out, dim=-1)    

'''
Might need to use as input all of the raw constraint, the value merged field and the density field
'''    

## looking at SWIN and terrain shading
print(dg.fine_variable_order)
X1 = np.where(np.ones((constraints.shape[0], constraints.shape[1])))
X1 = np.hstack([X1[0][...,np.newaxis],
                X1[1][...,np.newaxis]])
X2 = fine_input[:,:,dg.fine_variable_order.index("shade_map")].numpy().reshape((fine_input.shape[0]*fine_input.shape[1],1))
X3 = fine_input[:,:,dg.fine_variable_order.index("solar_altitude")].numpy().reshape((fine_input.shape[0]*fine_input.shape[1],1))
X4 = fine_input[:,:,dg.fine_variable_order.index("cloud_cover")].numpy().reshape((fine_input.shape[0]*fine_input.shape[1],1))

kernel1 = RBF(50)
kernel2 = RBF(0.1)
kernel3 = RBF(0.1)
kernel4 = RBF(0.25)

i = 1
site_dat = station_targets.iloc[i].copy()
site_dat[['sub_y', 'sub_x']] = site_dat[['sub_y', 'sub_x']]
site_shading = fine_input[int(site_dat.sub_y), int(site_dat.sub_x), dg.fine_variable_order.index("shade_map")]
site_solarelev = fine_input[int(site_dat.sub_y), int(site_dat.sub_x), dg.fine_variable_order.index("solar_altitude")]
site_cloud = fine_input[int(site_dat.sub_y), int(site_dat.sub_x), dg.fine_variable_order.index("cloud_cover")]

density1 = kernel1(site_dat[['sub_y', 'sub_x']].values, X1)
density2 = kernel2(site_shading, X2)
density3 = kernel3(site_solarelev, X3)
density4 = kernel4(site_cloud, X4)

fig, ax = plt.subplots(1,4)
ax[0].imshow(density1[0,:].reshape(constraints.shape[0], constraints.shape[1])[::-1,:])
ax[1].imshow(density2[0,:].reshape(constraints.shape[0], constraints.shape[1])[::-1,:])
ax[2].imshow(density3[0,:].reshape(constraints.shape[0], constraints.shape[1])[::-1,:])
ax[3].imshow(density4[0,:].reshape(constraints.shape[0], constraints.shape[1])[::-1,:])
plt.show()

this_density = density1 * density2 * density3 * density4
this_density = this_density / this_density.max()
this_value = site_dat[var[0]]
    
density = this_density
value = this_value * this_density

plt.imshow(density[0,:].reshape(constraints.shape[0], constraints.shape[1])[::-1,:])
plt.show()

# doing for all sites and joining to constraint with baseline density
X1 = np.where(np.ones((constraints.shape[0], constraints.shape[1])))
X1 = np.hstack([X1[0][...,np.newaxis],
                X1[1][...,np.newaxis]])
X2 = fine_input[:,:,dg.fine_variable_order.index("shade_map")].numpy().reshape((fine_input.shape[0]*fine_input.shape[1],1))
X3 = fine_input[:,:,dg.fine_variable_order.index("solar_altitude")].numpy().reshape((fine_input.shape[0]*fine_input.shape[1],1))
X4 = fine_input[:,:,dg.fine_variable_order.index("cloud_cover")].numpy().reshape((fine_input.shape[0]*fine_input.shape[1],1))

kernel1 = RBF(50)
kernel2 = RBF(0.25)
kernel3 = RBF(0.25)
kernel4 = RBF(0.25)
thisvar = ['UX', 'VY'] if var[0]=='WS' else var
interleaved_out = []
for j, vv in enumerate(thisvar):
    density = np.zeros((1, constraints.shape[0]*constraints.shape[1]), dtype=np.float32)
    value = np.zeros((1, constraints.shape[0]*constraints.shape[1]), dtype=np.float32)
    for i in range(station_targets.shape[0]):
        site_dat = station_targets.iloc[i].copy()
        site_dat[['sub_y', 'sub_x']] = site_dat[['sub_y', 'sub_x']]
        site_shading = fine_input[int(site_dat.sub_y), int(site_dat.sub_x), dg.fine_variable_order.index("shade_map")]
        site_solarelev = fine_input[int(site_dat.sub_y), int(site_dat.sub_x), dg.fine_variable_order.index("solar_altitude")]
        site_cloud = fine_input[int(site_dat.sub_y), int(site_dat.sub_x), dg.fine_variable_order.index("cloud_cover")]

        density1 = kernel1(site_dat[['sub_y', 'sub_x']].values, X1)
        density2 = kernel2(site_shading, X2)
        density3 = kernel3(site_solarelev, X3)
        density4 = kernel4(site_cloud, X4)
        this_density = density1 * density2 * density3 * density4
        this_density = this_density / this_density.max()
        this_value = site_dat[vv]
                
        density += this_density
        value += this_value * this_density
        
    density = density[0,:].reshape(constraints.shape[0], constraints.shape[1])
    value = value[0,:].reshape(constraints.shape[0], constraints.shape[1])
                
    # combining era5 interp constraint with point obs using baseline min density for constraint            
    density = density + baseline_density
    value = value + constraints.numpy()[:,:,j] * baseline_density
    
    value = value / density # do this before normalising the density!
    density = density / density.max()
    
    fig, ax = plt.subplots(1,3)
    ax[0].imshow(value[::-1,:])
    ax[1].imshow(constraints.numpy()[::-1,:,j])
    ax[2].imshow(density[::-1,:])
    plt.show()
    
    interleaved_out.append(torch.from_numpy(value))
    interleaved_out.append(torch.from_numpy(density))
elev_vd = torch.stack(interleaved_out, dim=-1)    



# LWIN
print(dg.fine_variable_order)
X1 = np.where(np.ones((constraints.shape[0], constraints.shape[1])))
X1 = np.hstack([X1[0][...,np.newaxis],
                X1[1][...,np.newaxis]])
X2 = fine_input[:,:,dg.fine_variable_order.index("TA")].numpy().reshape((fine_input.shape[0]*fine_input.shape[1],1))
X3 = fine_input[:,:,dg.fine_variable_order.index("cloud_cover")].numpy().reshape((fine_input.shape[0]*fine_input.shape[1],1))

kernel1 = RBF(40)
kernel2 = RBF(0.1)
kernel3 = RBF(0.15)
thisvar = ['UX', 'VY'] if var[0]=='WS' else var
interleaved_out = []
for j, vv in enumerate(thisvar):
    density = np.zeros((1, constraints.shape[0]*constraints.shape[1]), dtype=np.float32)
    value = np.zeros((1, constraints.shape[0]*constraints.shape[1]), dtype=np.float32)
    for i in range(station_targets.shape[0]):
        site_dat = station_targets.iloc[i].copy()
        site_dat[['sub_y', 'sub_x']] = site_dat[['sub_y', 'sub_x']]
        site_TA = fine_input[int(site_dat.sub_y), int(site_dat.sub_x), dg.fine_variable_order.index("TA")]
        site_cloud = fine_input[int(site_dat.sub_y), int(site_dat.sub_x), dg.fine_variable_order.index("cloud_cover")]

        density1 = kernel1(site_dat[['sub_y', 'sub_x']].values, X1)
        density2 = kernel2(site_TA, X2)
        density3 = kernel3(site_cloud, X3)        
        this_density = density1 * density2 * density3
        this_density = this_density / this_density.max()
        this_value = site_dat[vv]
                
        density += this_density
        value += this_value * this_density
        
    density = density[0,:].reshape(constraints.shape[0], constraints.shape[1])
    value = value[0,:].reshape(constraints.shape[0], constraints.shape[1])
                
    # combining era5 interp constraint with point obs using baseline min density for constraint            
    density = density + baseline_density
    value = value + constraints.numpy()[:,:,j] * baseline_density
    
    value = value / density # do this before normalising the density!
    density = density / density.max()
    
    fig, ax = plt.subplots(1,3)
    ax[0].imshow(value[::-1,:])
    ax[1].imshow(constraints.numpy()[::-1,:,j])
    ax[2].imshow(density[::-1,:])
    plt.show()
   
# RH
print(dg.fine_variable_order)
X1 = np.where(np.ones((constraints.shape[0], constraints.shape[1])))
X1 = np.hstack([X1[0][...,np.newaxis],
                X1[1][...,np.newaxis]])
X2 = fine_input[:,:,dg.fine_variable_order.index("TA")].numpy().reshape((fine_input.shape[0]*fine_input.shape[1],1))
X3 = fine_input[:,:,dg.fine_variable_order.index("PA")].numpy().reshape((fine_input.shape[0]*fine_input.shape[1],1))
X2 = np.hstack([X2, X3])

kernel1 = RBF(40)
kernel2 = RBF(1)
thisvar = ['UX', 'VY'] if var[0]=='WS' else var
interleaved_out = []
for j, vv in enumerate(thisvar):
    density = np.zeros((1, constraints.shape[0]*constraints.shape[1]), dtype=np.float32)
    value = np.zeros((1, constraints.shape[0]*constraints.shape[1]), dtype=np.float32)
    for i in range(station_targets.shape[0]):
        site_dat = station_targets.iloc[i].copy()
        site_dat[['sub_y', 'sub_x']] = site_dat[['sub_y', 'sub_x']]
        site_TA = fine_input[int(site_dat.sub_y), int(site_dat.sub_x), dg.fine_variable_order.index("TA")]
        site_PA = fine_input[int(site_dat.sub_y), int(site_dat.sub_x), dg.fine_variable_order.index("PA")]        

        density1 = kernel1(site_dat[['sub_y', 'sub_x']].values, X1)
        density2 = kernel2(np.hstack([site_TA, site_PA]), X2)
        this_density = density1 * density2
        this_density = this_density / this_density.max()
        this_value = site_dat[vv]
                
        density += this_density
        value += this_value * this_density
        
    density = density[0,:].reshape(constraints.shape[0], constraints.shape[1])
    value = value[0,:].reshape(constraints.shape[0], constraints.shape[1])
                
    # combining era5 interp constraint with point obs using baseline min density for constraint            
    density = density + baseline_density
    value = value + constraints.numpy()[:,:,j] * baseline_density
    
    value = value / density # do this before normalising the density!
    density = density / density.max()
    
    fig, ax = plt.subplots(1,3)
    ax[0].imshow(value[::-1,:])
    ax[1].imshow(constraints.numpy()[::-1,:,j])
    ax[2].imshow(density[::-1,:])
    plt.show()
   
# WS
print(dg.fine_variable_order)
X1 = np.where(np.ones((constraints.shape[0], constraints.shape[1])))
X1 = np.hstack([X1[0][...,np.newaxis],
                X1[1][...,np.newaxis]])
X2 = fine_input[:,:,dg.fine_variable_order.index("elev")].numpy().reshape((fine_input.shape[0]*fine_input.shape[1],1))

# X3a = fine_input[:,:,dg.fine_variable_order.index('l_wooded')].numpy().reshape((fine_input.shape[0]*fine_input.shape[1],1))
# X3b = fine_input[:,:,dg.fine_variable_order.index('l_open')].numpy().reshape((fine_input.shape[0]*fine_input.shape[1],1))
# X3c = fine_input[:,:,dg.fine_variable_order.index('l_mountain-heath')].numpy().reshape((fine_input.shape[0]*fine_input.shape[1],1))
# X3d = fine_input[:,:,dg.fine_variable_order.index('l_urban')].numpy().reshape((fine_input.shape[0]*fine_input.shape[1],1))
# X3 = np.hstack([X3a, X3b, X3c, X3d])

kernel1 = RBF(40)
kernel2 = RBF(1.5)
# kernel3 = RBF(0.25)
thisvar = ['UX', 'VY'] if var[0]=='WS' else var
interleaved_out = []
for j, vv in enumerate(thisvar):
    density = np.zeros((1, constraints.shape[0]*constraints.shape[1]), dtype=np.float32)
    value = np.zeros((1, constraints.shape[0]*constraints.shape[1]), dtype=np.float32)
    for i in range(station_targets.shape[0]):
        site_dat = station_targets.iloc[i].copy()
        site_dat[['sub_y', 'sub_x']] = site_dat[['sub_y', 'sub_x']]
        site_dat['ALTITUDE'] = (site_dat['ALTITUDE'] - nm.s_means.elev) / nm.s_stds.elev
        # site_wood = fine_input[int(site_dat.sub_y), int(site_dat.sub_x), dg.fine_variable_order.index('l_wooded')]
        # site_open = fine_input[int(site_dat.sub_y), int(site_dat.sub_x), dg.fine_variable_order.index('l_open')]
        # site_mtn = fine_input[int(site_dat.sub_y), int(site_dat.sub_x), dg.fine_variable_order.index('l_mountain-heath')]
        # site_urb = fine_input[int(site_dat.sub_y), int(site_dat.sub_x), dg.fine_variable_order.index('l_urban')]        
        
        density1 = kernel1(site_dat[['sub_y', 'sub_x']].values, X1)
        density2 = kernel2(site_dat.ALTITUDE, X2)
        #density3 = kernel2(np.hstack([site_wood, site_open, site_mtn, site_urb]), X3)
        this_density = density1 * density2
        this_density = this_density / this_density.max()
        this_value = site_dat[vv]
                
        density += this_density
        value += this_value * this_density
        
    density = density[0,:].reshape(constraints.shape[0], constraints.shape[1])
    value = value[0,:].reshape(constraints.shape[0], constraints.shape[1])
                
    # combining era5 interp constraint with point obs using baseline min density for constraint            
    density = density + baseline_density
    value = value + constraints.numpy()[:,:,j] * baseline_density
    
    value = value / density # do this before normalising the density!
    density = density / density.max()
    
    fig, ax = plt.subplots(1,3)
    ax[0].imshow(value[::-1,:])
    ax[1].imshow(constraints.numpy()[::-1,:,j])
    ax[2].imshow(density[::-1,:])
    plt.show()
   
   


