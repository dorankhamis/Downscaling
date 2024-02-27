from types import SimpleNamespace
import pandas as pd

data_pars = SimpleNamespace( 
    train_years = [2010, 2011, 2012, 2013, 2014], #[2015, 2016, 2017],
    val_years = [2015, 2016], #[2018, 2019],
    heldout_years = [2017, 2018, 2019, 2020, 2021], #[2020, 2021],
    dim_l = 8, #8, # number of large pixels (dim_l x dim_l)
    scale = 28, # downsampling factor, 28km -> 1km, approx 0.25 degrees
    res = 28000 # coarse resolution in metres 
)

model_pars = SimpleNamespace(
    filters = [12, 24, 48, 96],
    dropout_rate = 0.0,    
    attn_heads = 2,
    ds_cross_attn = [8, 12, 16, 20, 24],
    scale_factor = 3,
    pe = False, # positional encoding
    
    in_channels =      {'TA':1, 'PA':1, 'SWIN':1, 'LWIN':1, 'WS':2 , 'RH':1, 'PRECIP':1},
    output_channels =  {'TA':1, 'PA':1, 'SWIN':1, 'LWIN':1, 'WS':2 , 'RH':1, 'PRECIP':1},
    hires_fields =     {'TA':4, 'PA':4, 'SWIN':9, 'LWIN':7, 'WS':11, 'RH':5, 'PRECIP':8},
    context_channels = {'TA':4, 'PA':4, 'SWIN':7, 'LWIN':5, 'WS':5 , 'RH':4, 'PRECIP':5},
    
    coarse_variable_order = {
        'TA':['TA'],
        'PA':['PA'],
        'SWIN':['SWIN'],
        'LWIN':['LWIN'],
        'WS':['UX', 'VY'],
        'RH':['RH'],
        'PRECIP':['PRECIP']
    },
    fine_variable_order = {
        'TA':['landfrac', 'elev', 'lat', 'lon'],
        'PA':['landfrac', 'elev', 'lat', 'lon'],
        'SWIN':['landfrac', 'elev', 'illumination_map', 'shade_map', 'solar_altitude', 'cloud_cover', 'PA', 'lat', 'lon'],
        'LWIN':['landfrac', 'sky_view_factor', 'cloud_cover', 'RH', 'TA', 'lat', 'lon'],
        'WS':['landfrac', 'stdev', 'aspect', 'slope', 'elev', 'l_wooded', 'l_open', 'l_mountain-heath', 'l_urban', 'lat', 'lon'],
        'RH':['landfrac', 'TA', 'PA', 'lat', 'lon'],
        'PRECIP':['landfrac', 'aspect', 'elev', 'cloud_cover', 'UX', 'VY', 'lat', 'lon']
    },
    context_variable_order = {
        'TA':['var_value', 'elev', 'lat', 'lon'],
        'PA':['var_value', 'elev', 'lat', 'lon'],
        'SWIN':['var_value', 'elev', 'lat', 'lon', 'cloud_cover', 'shade_map', 'illumination_map'],
        'LWIN': ['var_value', 'elev', 'lat', 'lon', 'cloud_cover'],
        'WS':['var_value_u', 'var_value_v', 'elev', 'lat', 'lon'],
        'RH':['var_value', 'elev', 'lat', 'lon'],
        'PRECIP':['var_value', 'elev', 'lat', 'lon', 'cloud_cover']
    },
    
    ## attn params
    soft_masks = True,
    pixel_pass_masks = True,
    binary_masks = False,
    
    diminish_model = "polynomial", # ["gaussian", "polynomial"]
    dist_lim = {'TA':80, 'PA':80, 'SWIN':80, 'LWIN':80, 'WS':80, 'RH':80, 'PRECIP':10},
    poly_exp = {'TA':4,  'PA':4,  'SWIN':4,  'LWIN':4,  'WS':4,  'RH':4,  'PRECIP':6},
    
    dist_pixpass = {'TA':125, 'PA':125, 'SWIN':125, 'LWIN':125, 'WS':125, 'RH':125, 'PRECIP':25},
    pass_exp =     {'TA':1.5, 'PA':1.5, 'SWIN':1.5, 'LWIN':1.5, 'WS':1.5, 'RH':1.5, 'PRECIP':3.},
    
    dist_lim_far = None, # used in binary masks
    attn_eps = None # used when diminish_model=="gaussian"
)

train_pars = SimpleNamespace(    
    batch_size = 5,
    warmup_epochs = 2, # num epochs with just daily batches
    increase_epochs = 5, # num epochs with p_hourly increasing
    cooldown_epochs = 600, # num epochs with constant max p_hourly
    p_hourly_max = 1,
    lr = 1e-4, # 5e-5,
    gamma = 0.99, # learning rate reduction    
    sigma_context_stations = {'TA':0.02, 'PA':0.02, 'SWIN':0.02, 'LWIN':0.02, 'WS':0.02, 'RH':0.02, 'PRECIP':0.02}, # those station values used as input
    sigma_target_stations =  {'TA':0.025, 'PA':0.025, 'SWIN':0.025, 'LWIN':0.025, 'WS':0.025, 'RH':0.025, 'PRECIP':0.025}, # those station values that only occur in the loss
    sigma_constraints =      {'TA':0.07,  'PA':0.07,  'SWIN':0.07,  'LWIN':0.07,  'WS':0.07,  'RH':0.07,  'PRECIP':0.07},
    sigma_gridavg =          {'TA':0.05,  'PA':0.05,  'SWIN':0.05,  'LWIN':0.05,  'WS':0.05,  'RH':0.05,  'PRECIP':0.035},
    #sigma_localcont = 0.06, # unused
    train_len = 400,
    val_len = 200,    
    use_unseen_sites = True,
    #attn_mask_sizes = [3, 5, 7, 9], # unused
    #null_batch_prob = 0.25 # unused
    #ensemble_size = None, # unused
)
train_pars.max_epochs = (train_pars.warmup_epochs + 
    train_pars.increase_epochs + 
    train_pars.cooldown_epochs
)

normalisation = SimpleNamespace(
    precip_norm = 10., # mm/hr
    lwin_mu = 330.,
    lwin_sd = 35.,
    swin_norm = 500.,    
    temp_mu = 10.,
    temp_sd = 10.,
    p_mu = 1013.,
    p_sd = 25.,
    rh_mu = 85.,
    rh_sd = 12.,
    ws_mu = 4.,
    ws_sd = 2.,
    lat_norm = 90.,
    lon_norm = 180.,
    s_means = pd.Series({'elev':134.620572,
                         'fdepth':0.079323,
                         'slope':3.368099,
                         'stdev':12.924280,
                         'stdtopi':1.882920,
                         'topi':5.003433}),
    s_stds = pd.Series({'elev':117.351676,
                         'fdepth':0.058454,
                         'slope':2.962117,
                         'stdev':11.500340,
                         'stdtopi':0.542261,
                         'topi':1.100768})
)
