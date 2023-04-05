model = MetVAE(input_channels=input_channels,
               hires_fields=hires_fields,
               output_channels=output_channels,
               latent_variables=latent_variables,
               filters=filters,
               dropout_rate=dropout_rate,
               scale=scale)
model.to(device)

## create optimizer, schedulers and loss function
optimizer = Adam(model.parameters(), lr=lr)
LR_scheduler = ExponentialLR(optimizer, gamma=gamma)
KLW_scheduler = frange_cycle_linear(max_epochs, start=0.0, stop=1.0, n_cycle=3, ratio=0.6)
KLW_scheduler = np.hstack([np.zeros(warmup_epochs), KLW_scheduler])
#KLW_scheduler *= kl_weight_max
# loglikelihood = make_loss_func(sigma_stations=sigma_stations,
                               # sigma_constraints=sigma_constraints,
                               # sigma_gridavg=sigma_gridavg)
def loglik_step(pred, coarse_inputs, station_dict, constraints):
    station_pixel_L = loglik_of_station_pixels(pred, station_dict, sigma=sigma_stations)
    phys_constraints_L = physical_constraints(pred, constraints, sigma=sigma_constraints)
    grid_avg_L = preserve_coarse_grid_average(pred, coarse_inputs, sigma=sigma_gridavg)
    return station_pixel_L, phys_constraints_L, grid_avg_L  


model.train()
batch = datgen.get_batch(batch_size=batch_size,
                         batch_type='train',
                         load_binary_batch=True)
batch = Batch(batch, device=device)
        
# run data through model
vae_normal_dists, KL = model.encode(batch.coarse_inputs, batch.fine_inputs, calc_kl=True)
pred = model.decode(vae_normal_dists, batch.fine_inputs)


    hires_at_lores = model.encoder.coarsen(batch.fine_inputs)
    joined_fields = torch.cat([batch.coarse_inputs, hires_at_lores], dim=1)
    joined_fields = model.encoder.resblock1(joined_fields)
    joined_fields = model.encoder.resblock2(joined_fields)
    vae_latents = model.encoder.create_vae_latents(joined_fields)
    
    # priors on variational params
    # all standard normals as have z-scored all inputs
    ones_vec = torch.ones_like(vae_latents[:,0,:,:])
    priors = []
    vae_normal_dists = []
    from torch.distributions.normal import Normal
    for i in range(vae_latents.shape[1]//2):
        # variable ordering
        # ['TA', 'PA', 'SWIN', 'LWIN', 'WS', 'RH']
        priors.append(Normal(0*ones_vec, ones_vec))
        vae_normal_dists.append(Normal(vae_latents[:,2*i,:,:],
                                       vae_latents[:,2*i+1,:,:].exp()+1e-5))

# calculate log likelihood
station_pixel_L, phys_constraints_L, grid_avg_L = loglik_step(
    pred,
    batch.coarse_inputs,
    batch.station_dict,
    batch.constraint_targets
)
loglik = station_pixel_L + phys_constraints_L + grid_avg_L

# calculate ELBO
ELBO = loglik - KL*kl_weight
mean_neg_ELBO = -ELBO / float(len(batch.station_dict['values']))

# propagate derivatives
mean_neg_ELBO.backward()
optimizer.step()
optimizer.zero_grad()
