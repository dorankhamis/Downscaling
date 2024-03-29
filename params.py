from types import SimpleNamespace
import pandas as pd

data_pars = SimpleNamespace( 
    train_years = [2010, 2011, 2012, 2013, 2014], #[2015, 2016, 2017],
    val_years = [2015, 2016], #[2018, 2019],
    heldout_years = [2017, 2018, 2019, 2020, 2021], #[2020, 2021],
    dim_l = 8, #8, # number of large pixels (dim_l x dim_l)
    scale = 28, # downsampling factor, 28km -> 1km, approx 0.25 degrees
    res = 28000 # coarse resolution in metres
    #use_precip = True
)

model_pars = SimpleNamespace(
    filters = [12, 24, 48, 96],
    dropout_rate = 0.02,
    ## current params
    attn_heads = 2, #4
    ds_cross_attn = [8, 12, 16, 20, 24],  

    scale_factor = 3,
    pe = False, # positional encoding
    ## attn params
    dist_lim = 60, # mean==80 but could load var specific vals?
    dist_lim_far = 90, # dist_lim + 50
    attn_eps = 1e-6, # used when diminish_model=="gaussian"
    poly_exp = 3., # 6 used when diminish_model=="polynomial"
    diminish_model = "polynomial", # ["gaussian", "polynomial"]
    dist_pixpass = 150,
    pass_exp = 1.,
    soft_masks = True,
    pixel_pass_masks = True,
    binary_masks = False
)

train_pars = SimpleNamespace(
    ensemble_size = None, # unused
    batch_size = 5,
    warmup_epochs = 2, # num epochs with just daily batches
    increase_epochs = 5, # num epochs with p_hourly increasing
    cooldown_epochs = 600, # num epochs with constant max p_hourly
    p_hourly_max = 1,
    lr = 1e-4, # 5e-5,
    gamma = 0.99, # learning rate reduction    
    sigma_context_stations = 0.015, # those station values used as input
    sigma_target_stations = 0.02, # those station values that only occur in the loss
    sigma_constraints = 0.07,
    sigma_gridavg = 0.06,
    #sigma_localcont = 0.06, # unused
    train_len = 400,
    val_len = 200,
    attn_mask_sizes = [3, 5, 7, 9], # unused
    use_unseen_sites = True,
    null_batch_prob = 0.25 # unused
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
