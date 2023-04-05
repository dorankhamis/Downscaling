from types import SimpleNamespace
import pandas as pd

data_pars = SimpleNamespace(    
    train_years = [2015, 2016, 2017, 2018],
    val_years = [2019, 2020],
    heldout_years = [2021],
    dim_l = 4, # number of large pixels (dim_l x dim_l)
    scale = 25, # downsampling factor, 25km -> 1km
    res = 25000 # coarse resolution in metres
)

model_pars = SimpleNamespace(
    filters = [256, 128, 64],
    dropout_rate = 0.1,
    input_channels = 6,
    hires_fields = 13,
    output_channels = 6,
    latent_variables = 6
)

train_pars = SimpleNamespace(
    ensemble_size = None,
    batch_size = 4,
    warmup_epochs = 2,
    max_epochs = 50,
    lr = 1e-4,
    gamma = 0.99, # learning rate reduction
    sigma_stations = 0.05,
    sigma_constraints = 0.1,
    sigma_gridavg = 0.1,
    train_len = 500,
    val_len = 250
)

normalisation = SimpleNamespace(
    precip_norm = 100.,
    lwin_mu = 330.,
    lwin_sd = 35.,
    logswin_mu = 5., # for SWIN having been transformed by log(1+SWIN)
    logswin_sd = 2., # for SWIN having been transformed by log(1+SWIN)
    temp_mu = 10.,
    temp_sd = 10.,
    p_mu = 1013.,
    p_sd = 25.,
    rh_mu = 85.,
    rh_sd = 12.,
    ws_mu = 4.,
    ws_sd = 2., 
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
