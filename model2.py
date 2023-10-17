import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions import kl

from layers import (ResnetBlocWithAttn, CoarsenField, RefineField,
                    find_num_pools, Attention1D, SublayerConnection,
                    PositionalEncodingOffGrid, PositionalEncoding2D,
                    GridPointAttn)

class SimpleDownscaler(nn.Module):
    def __init__(self,
                 input_channels=1,
                 hires_fields=3,
                 output_channels=1,
                 context_channels=2,
                 filters=[6, 12, 24, 48],
                 dropout_rate=0.1,
                 scale=28,
                 scale_factor=3,
                 attn_heads=2,
                 ds_cross_attn=[6, 10, 14, 18, 24],
                 pe=False):
        super().__init__()
        self.nups = find_num_pools(scale, factor=scale_factor)
        self.scale = scale
        self.scale_factor = scale_factor

        self.grid_point_attn_0 = GridPointAttn(
            input_channels + hires_fields,
            ds_cross_attn[0],
            context_channels,
            attn_heads,
            dropout_rate,
            pe=pe
        )

        self.ups = nn.Upsample(scale_factor=scale_factor, mode='bilinear')
        cur_size = ds_cross_attn[0]
        process = []
        grdpnt_attn = []
        for i in range(self.nups+1):
            process.append(
                nn.Sequential(
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(cur_size,
                              filters[i],
                              kernel_size=3, stride=1, padding=0),
                    nn.GELU()
                )
            )
            grdpnt_attn.append(
                GridPointAttn(filters[i] + hires_fields,
                              ds_cross_attn[i+1],
                              context_channels,
                              attn_heads,
                              dropout_rate,
                              pe=pe)
            )
            cur_size += ds_cross_attn[i+1]
                    
        self.process = nn.ModuleList(process)
        self.grid_point_attn = nn.ModuleList(grdpnt_attn)

        self.pre_output = nn.Conv2d(cur_size,
                                    ds_cross_attn[self.nups+1]//2,
                                    kernel_size=1, stride=1, padding=0)
        self.act = nn.GELU()

        self.gen_output = nn.Conv2d(ds_cross_attn[self.nups+1]//2, 1,
                                    kernel_size=1, stride=1, padding=0)  
        
    def forward(self, coarse_inputs, fine_inputs,
                context_data, context_locs,
                context_masks=[None, None, None, None, None],
                context_soft_masks=[None, None, None, None, None],
                pixel_passer=[None, None, None, None, None]):
        # find intermediate scales
        coarse_size = coarse_inputs.shape[-1]
        hires_size = fine_inputs.shape[-1]
        sizes = [coarse_size * (self.scale_factor**i) for i in range(1, self.nups+1)]
        final_scale = coarse_size * self.scale / sizes[-1]

        sf = nn.functional.interpolate(fine_inputs, scale_factor=1./self.scale, mode='bilinear')
        x = torch.cat([coarse_inputs, sf], dim=1)
        x = self.grid_point_attn_0(x, context_data, context_locs, hires_size,
                                   mask=context_masks[0],
                                   softmask=context_soft_masks[0],
                                   pixel_passer=pixel_passer[0])

        for i in range(self.nups):
            # upscale
            x_res = self.ups(x)            
            x = self.process[i](x_res)
            
            # join coarsened static fields
            fct = sizes[i] / hires_size
            sf = nn.functional.interpolate(fine_inputs,
                                           scale_factor=fct,
                                           mode='bilinear')
            x = torch.cat([x, sf], dim=1)    
            
            # attn
            x = self.grid_point_attn[i](x, context_data, context_locs, hires_size,
                                        mask=context_masks[i+1],
                                        softmask=context_soft_masks[i+1],
                                        pixel_passer=pixel_passer[i+1])
            
            # residual
            x = torch.cat([x, x_res], dim=1)

        # final upscale
        fct = hires_size / sizes[-1] # == final_scale
        x_res = nn.functional.interpolate(x, scale_factor=fct,
                                          mode='bilinear')        
        x = self.process[i+1](x_res)

        # join raw static fields
        x = torch.cat([x, fine_inputs], dim=1)

        # attn
        x = self.grid_point_attn[i+1](x, context_data, context_locs, hires_size,
                                      mask=context_masks[i+2],
                                      softmask=context_soft_masks[i+2],
                                      pixel_passer=pixel_passer[i+2])

        # residual
        x = torch.cat([x, x_res], dim=1)

        # output
        x = self.act(self.pre_output(x))
        output = self.gen_output(x)
        return output


class Encoder(nn.Module):
    def __init__(self,
                 input_channels=6,
                 hires_fields=10,
                 latent_variables=6,
                 filters=[128, 64, 32, 16],
                 dropout_rate=0.1,
                 scale=25,
                 scale_factor=3):
        super().__init__()
        # embed inputs
        self.embed = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(input_channels, 3*input_channels,
                      kernel_size=3, stride=1, padding=0)
        )
        
        # lo-res feature map from hi-res static fields
        self.coarsen = CoarsenField(hires_fields, filters[::-1],
                                    filters[1], scale=scale, simple=True)

        # redidual blocks             
        self.resblock1 = ResnetBlocWithAttn(3*input_channels + filters[1],
                                            filters[0],
                                            norm_groups=1,
                                            dropout=dropout_rate,
                                            with_attn=False)

        # refine once
        self.refine = RefineField(filters[0], filters, filters[1],
                                  scale=scale_factor)
        
        # create latent (mu, sigma) pairs
        self.create_vae_latents = nn.Conv2d(filters[1], 2*latent_variables, kernel_size=1)
        
    def forward(self, coarse_inputs, fine_inputs, calc_kl=False, mask=None):        
        hires_at_lores = self.coarsen(fine_inputs)
        coarse_embedded = self.embed(coarse_inputs)
        joined_fields = torch.cat([coarse_embedded, hires_at_lores], dim=1)
        
        joined_fields = self.resblock1(joined_fields, mask=mask)
        joined_fields = self.refine(joined_fields)
        vae_latents = self.create_vae_latents(joined_fields)
        
        # priors on variational params
        # all standard normals as have z-scored all inputs
        ones_vec = torch.ones_like(vae_latents[:,0,:,:])
        priors = []
        vae_normal_dists = []
        for i in range(vae_latents.shape[1]//2):
            # variable ordering
            # ['TA', 'PA', 'SWIN', 'LWIN', 'WS', 'RH']
            priors.append(Normal(0*ones_vec, ones_vec))
            vae_normal_dists.append(Normal(vae_latents[:,2*i,:,:],
                                           vae_latents[:,2*i+1,:,:].exp() + 1e-6))
        
        if calc_kl:
            KL = kl.kl_divergence(vae_normal_dists[0], priors[0])
            if len(priors)>1:
                for i in range(1, len(priors)):
                    KL += kl.kl_divergence(vae_normal_dists[i], priors[i])
        else:
            KL = 0
        
        return vae_normal_dists, KL


class Decoder(nn.Module):
    def __init__(self,
                 hires_fields=9,
                 output_channels=6,
                 context_channels=2,
                 latent_variables=6,
                 filters=[128, 64, 32, 16],
                 dropout_rate=0.1,
                 scale=25,
                 scale_factor=3,
                 attn_heads=2,
                 d_cross_attn=64):
        super().__init__()
                        
        " we divide the scale=25 upsample into n 3x upsamples and 1 fine tune upsample "
        " for each upsample block we run: resnet with/without attention, RefineField, "
        " and append CoarsenField(fine_inputs), keeping track of the required scale "
        self.pre_refines = 1
        self.scale = scale# / (scale_factor**self.pre_refines) # these happened in the encoder
        n = find_num_pools(self.scale/scale_factor, factor=scale_factor)
        self.scale_factor = scale_factor        
        self.resnet_dict = nn.ModuleDict({})
        self.refine = nn.ModuleList([])
        self.coarsen = nn.ModuleList([])
        self.joiner = nn.ModuleList([]) # embed refined field + injected hi res to useable dim
        for i in range(n):
            if i==0:
                ''' adding attention at i==0 as we may have a pre_refine==1 '''
                self.resnet_dict.add_module(
                    f'first_{i}', ResnetBlocWithAttn(
                        latent_variables, filters[i], norm_groups=2,
                        dropout=dropout_rate, with_attn=True)
                )
                self.resnet_dict.add_module(    
                    f'second_{i}', ResnetBlocWithAttn(
                        filters[i], filters[i], norm_groups=filters[i]//8,
                        dropout=dropout_rate, with_attn=False)
                )
                self.refine.append(RefineField(filters[i], filters, filters[i],
                                               scale=self.scale_factor))
                self.coarsen.append(CoarsenField(
                    hires_fields, filters[::-1], filters[i]//4,
                    scale=self.scale/(self.scale_factor**(i + 1 + self.pre_refines)),
                    simple=True)
                )
            else:
                self.resnet_dict.add_module(
                    f'first_{i}', ResnetBlocWithAttn(filters[i-1], filters[i], norm_groups=2,
                                               dropout=dropout_rate, with_attn=True)
                )
                self.resnet_dict.add_module(    
                    f'second_{i}', ResnetBlocWithAttn(filters[i], filters[i], norm_groups=filters[i]//8,
                                                dropout=dropout_rate, with_attn=False)
                )
                self.refine.append(RefineField(filters[i], filters, filters[i],
                                               scale=self.scale_factor))
                self.coarsen.append(CoarsenField(
                    hires_fields, filters[::-1], filters[i]//4,
                    scale=self.scale/(self.scale_factor**(i + 1 + self.pre_refines)),
                    simple=True)
                )
            self.joiner.append(nn.Conv2d(filters[i]//4 + filters[i], filters[i], kernel_size=1))
        
        # interpolate remainder
        self.resblock1 = ResnetBlocWithAttn(filters[i], filters[i]//4 * 3, norm_groups=4,
                                            dropout=dropout_rate, with_attn=True)
        self.resblock2 = ResnetBlocWithAttn(filters[i]//4 * 3, filters[i]//4 * 3, norm_groups=4,
                                            dropout=dropout_rate, with_attn=False)
        self.post_interp_conv = nn.Sequential(nn.ReflectionPad2d(1),
            nn.Conv2d(filters[i]//4 * 3, filters[i]//4 * 3,
                      kernel_size=3, stride=1, padding=0)            
        )
        self.embed_finescale = nn.Conv2d(hires_fields, (3*filters[i])//16, kernel_size=1)
        
        # cross-attention with context points        
        self.embed_crs_atn = nn.Conv2d(filters[i]//4 * 3 + (3*filters[i])//16,
                                       d_cross_attn-2, kernel_size=1)
        self.embed_context = nn.Linear(context_channels, d_cross_attn-2)
        self.sublayer = SublayerConnection(d_cross_attn, dropout_rate)
        self.cross_attn = Attention1D(attn_heads, d_cross_attn, dropout=dropout_rate)
        
        # output        
        self.resblock_final = ResnetBlocWithAttn(d_cross_attn, filters[-1], norm_groups=2,
                                                 dropout=dropout_rate, with_attn=False)
        self.gen_output = nn.Conv2d(filters[-1], output_channels, kernel_size=1)
        
    def forward(self, vae_normal_dists, fine_inputs, 
                context_data, context_locs,
                masks=[None, None, None, None], context_mask=None):
        # do reparam trick sampling
        sampled_vars = []
        for i in range(len(vae_normal_dists)):
            sampled_vars.append(vae_normal_dists[i].rsample())
        
        # think of these as compressed versions of the 
        # 100x100 variables that we want at output.
        # We now uncompress and combine with the hi-res static fields
        x = torch.stack(sampled_vars, dim=1)
        
        ## Upsampling from dim_l to dim_h
        # in 3x upsamples
        for i in range(len(self.refine)):            
            x = self.resnet_dict[f'first_{i}'](x, mask=masks[i+self.pre_refines])
            x = self.resnet_dict[f'second_{i}'](x, mask=masks[i+self.pre_refines])            
            x = self.refine[i](x)
            hi = self.coarsen[i](fine_inputs)
            x = self.joiner[i](torch.cat([x, hi], dim=1))     

        # the remainder
        x = self.resblock1(x, mask=masks[i + 1 + self.pre_refines])
        x = self.resblock2(x, mask=masks[i + 1 + self.pre_refines])
        remaining_scale = self.scale / (self.scale_factor**(len(self.refine)+self.pre_refines))
        x = nn.functional.interpolate(x, scale_factor=remaining_scale, mode='bilinear')
        x = self.post_interp_conv(x)

        # inject native res static data
        x = torch.cat([x, self.embed_finescale(fine_inputs)], dim=1)
        
        ## cross-attention with context points
        '''
        Would ideally want to encode something about the "path" from
        the home point to the point of attention, i.e. in the shortest
        route through a network between the two points, what nodes
        do we pass through? Mountains, etc? 
        '''
        # embed to attention dimension
        x = self.embed_crs_atn(x)
        context = self.embed_context(context_data.transpose(-2,-1)).transpose(-2,-1)
        
        # attach locations to front of vectors
        raw_H = x.shape[2]
        raw_W = x.shape[3]
        x = torch.reshape(x, (x.shape[0], x.shape[1], raw_H*raw_W))
        
        # location array for hi-res grid (for cross-attention with context points)
        # X1 = np.where(np.ones((self.dim_h, self.dim_h)))
        # self.X1 = np.hstack([X1[0][...,np.newaxis], X1[1][...,np.newaxis]]) / (self.dim_h-1)
        grid_locs = torch.reshape(fine_inputs[:,-2:,:,:],
                                  (x.shape[0],  2, raw_H*raw_W))                
        x = torch.cat([grid_locs, x], dim=1)        
        context = torch.cat([context_locs, context], dim=1)
                
        # calculate cross-attention
        x = x.transpose(-2,-1)
        x = self.sublayer(x, 
            lambda x: self.cross_attn(x,
                                      context.transpose(-2,-1),
                                      context.transpose(-2,-1),
                                      context_mask)
        )
        x = torch.reshape(x.transpose(-2,-1), (x.shape[0], x.shape[-1], raw_H, raw_W))
        
        # process outputs
        x = self.resblock_final(x, mask=None)        
        output = self.gen_output(x)
        return output


class MetVAE(nn.Module):
    def __init__(self,
                 input_channels=6,
                 hires_fields=10,
                 output_channels=6,
                 context_channels=2, # depends on number of vars and additional ancillaries?
                 latent_variables=6,
                 filters_enc=[128, 64, 32],
                 filters_dec=[64, 32, 16],
                 dropout_rate=0.1,
                 scale=25,
                 scale_factor=3,
                 attn_heads=2,
                 d_cross_attn=12):
        super().__init__()

        self.encoder = Encoder(input_channels=input_channels,
                               hires_fields=hires_fields,                 
                               latent_variables=latent_variables,
                               filters=filters_enc,                               
                               dropout_rate=dropout_rate,
                               scale=scale,
                               scale_factor=scale_factor)        

        self.decoder = Decoder(hires_fields=hires_fields,
                               output_channels=output_channels,
                               latent_variables=latent_variables,
                               filters=filters_dec,
                               dropout_rate=dropout_rate,
                               scale=scale,
                               context_channels=context_channels,
                               scale_factor=scale_factor,
                               attn_heads=attn_heads,
                               d_cross_attn=d_cross_attn)

    def encode(self, coarse_inputs, fine_inputs, calc_kl=False, mask=None):
        vae_normal_dists, KL = self.encoder(coarse_inputs, fine_inputs,
                                            calc_kl=calc_kl, mask=mask)
        if calc_kl:
            return vae_normal_dists, KL.sum()
        else:
            return vae_normal_dists

    def decode(self, vae_normal_dists, fine_inputs, context_data,
               context_locs, masks=[None, None, None, None],
               context_mask=None):
        pred = self.decoder(vae_normal_dists, fine_inputs, context_data,
                            context_locs, masks=masks,
                            context_mask=context_mask)
        return pred

    def predict(self, coarse_inputs, fine_inputs, context_data,
                context_locs, masks=[None, None, None, None],
                context_mask=None):
        vae_normal_dists = self.encode(coarse_inputs, fine_inputs,
                                       calc_kl=False, mask=None)
        
        pred = self.decode(vae_normal_dists, fine_inputs, context_data,
                           context_locs, masks=masks,
                           context_mask=context_mask)
        return pred

    def forward(self, coarse_inputs, fine_inputs, context_data,
                context_locs, masks=[None, None, None, None],
                context_mask=None):
        vae_normal_dists, KL = self.encode(coarse_inputs, fine_inputs,
                                           calc_kl=True, mask=None)
        
        pred = self.decode(vae_normal_dists, fine_inputs, context_data,
                           context_locs, masks=masks,
                           context_mask=context_mask)
        # ELBO_i = log_lik - KL.sum()
        return pred, KL.sum()

if __name__=="__main__":
    
    if False:
        from setupdata2 import data_generator, Batch
        from params import model_pars, data_pars, train_pars
        from utils import make_mask_list_from_filters
        dg = data_generator()
        batch_size = 3
        batch_type = 'train'
        load_binary_batch = True
        var = 'TA'
        
        masks = make_mask_list_from_filters(dg.dim_l, dg.dim_l, dg.scale,
                                    scale_factor=3,
                                    batch_size=batch_size,
                                    filter_sizes=[3, 5, 7])
        batch = dg.get_chess_batch(var, batch_size=batch_size,
                                   batch_type=batch_type,
                                   load_binary_batch=load_binary_batch)
        batch = Batch(batch, masks, var_list=var)
        
        encoder = Encoder(input_channels=1,
                hires_fields=len(dg.targ_var_depends[var])+2,
                latent_variables=6,
                filters=[128, 64, 32],
                dropout_rate=0.1,
                scale=25)
        
        vae_normal_dists, KL = encoder(batch.coarse_inputs, batch.fine_inputs, calc_kl=True, mask=None)
        
        decoder = Decoder(hires_fields=len(dg.targ_var_depends[var])+2,
                output_channels=1,
                context_channels=3, # value, density, elevation (per variable)
                latent_variables=6,
                filters=[64, 32, 16],
                dropout_rate=0.1,
                scale=25,
                scale_factor=3,
                attn_heads=2,
                d_cross_attn=12)

        pred = decoder(vae_normal_dists, batch.fine_inputs, batch.context_data,
                       batch.context_locs, masks=batch.masks,
                       context_mask=batch.context_mask)

        def loglik_of_station_pixels(pred, station_dict, station_num_obs, sigma=0.01):
            station_loglik = torch.zeros((), dtype=torch.float32).to(pred.device)
            n_elements = 0
            for b in range(pred.shape[0]):
                yx = station_dict['coords_yx'][b]
                for i in range(yx.shape[0]):
                    mask = station_dict['var_present'][b][i,:] # variable mask
                    vals_true = station_dict['values'][b][i,:][mask]
                    vals_pred = pred[b,:,yx[i,0],yx[i,1]][mask]
                    s_mult = sigma_growth[station_num_obs[b][i]-1]
                    station_loglik += normal_loglikelihood(vals_pred, vals_true, sigma*s_mult).sum()
                    n_elements += mask.sum()
            if n_elements==0:
                return station_loglik
            else:
                return station_loglik / n_elements.to(torch.float32)

