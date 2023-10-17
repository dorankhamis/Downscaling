import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions import kl

from layers import (ResnetBlocWithAttn, CoarsenField, RefineField,
                    find_num_pools, Attention1D, SublayerConnection)

class Encoder(nn.Module):
    def __init__(self,
                 input_channels=6,
                 hires_fields=10,
                 latent_variables=6,
                 filters=[128, 64, 32],
                 dropout_rate=0.1,
                 scale=25):
        super().__init__()
        
        # lo-res feature map from hi-res static fields
        self.coarsen = CoarsenField(hires_fields, filters[::-1], filters[0], scale=scale)

        # redidual blocks             
        self.resblock1 = ResnetBlocWithAttn(input_channels + filters[0],
                                            filters[0],
                                            norm_groups=1,
                                            dropout=dropout_rate,
                                            with_attn=False)
        self.resblock2 = ResnetBlocWithAttn(filters[0],
                                            filters[0],
                                            norm_groups=filters[0]//8,
                                            dropout=dropout_rate,
                                            with_attn=False)
        
        # create latent (mu, sigma) pairs
        self.create_vae_latents = nn.Conv2d(filters[0], 2*latent_variables, kernel_size=1)
        
    def forward(self, coarse_inputs, fine_inputs, calc_kl=False, mask=None):        
        hires_at_lores = self.coarsen(fine_inputs)
        joined_fields = torch.cat([coarse_inputs, hires_at_lores], dim=1)
        
        joined_fields = self.resblock1(joined_fields, mask=mask)
        joined_fields = self.resblock2(joined_fields, mask=mask)
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
                 context_channels=12,
                 latent_variables=6,
                 filters=[256, 128, 64],
                 dropout_rate=0.1,
                 scale=25,
                 scale_factor=3,
                 attn_heads=2,
                 d_cross_attn=64):
        super().__init__()
                        
        " we divide the scale=25 upsample into n 3x upsamples and 1 fine tune upsample "
        " for each upsample block we run: resnet with/without attention, RefineField, "
        " and append CoarsenField(fine_inputs), keeping track of the required scale "
        n = find_num_pools(scale, factor=scale_factor)
        self.scale = scale
        self.scale_factor = scale_factor
        self.resnet_dict = nn.ModuleDict({})
        self.refine = nn.ModuleList([])
        self.coarsen = nn.ModuleList([])
        self.joiner = nn.ModuleList([]) # embed refined field + injected hi res to useable dim
        for i in range(n):
            if i==0:       
                self.resnet_dict.add_module(
                    f'first_{i}', ResnetBlocWithAttn(latent_variables, filters[i], norm_groups=1,
                                                     dropout=dropout_rate, with_attn=False)
                )
                self.resnet_dict.add_module(
                    f'second_{i}', ResnetBlocWithAttn(filters[i], filters[i], norm_groups=filters[i]//8,
                                                      dropout=dropout_rate, with_attn=False)
                )
                self.refine.append(RefineField(filters[i], filters, filters[i], scale=self.scale_factor))                
                self.coarsen.append(CoarsenField(hires_fields, filters[::-1], filters[i]//4,
                                                 scale=scale/(self.scale_factor**(i+1))))
            else:                
                self.resnet_dict.add_module(
                    f'first_{i}', ResnetBlocWithAttn(filters[i-1], filters[i], norm_groups=2,
                                               dropout=dropout_rate, with_attn=True)
                )
                self.resnet_dict.add_module(    
                    f'second_{i}', ResnetBlocWithAttn(filters[i], filters[i], norm_groups=filters[i]//8,
                                                dropout=dropout_rate, with_attn=False)
                )
                self.refine.append(RefineField(filters[i], filters, filters[i], scale=self.scale_factor))
                self.coarsen.append(CoarsenField(hires_fields, filters[::-1], filters[i]//4,
                                                 scale=scale/(self.scale_factor**(i+1))))
            self.joiner.append(nn.Conv2d(filters[i]//4 + filters[i], filters[i], kernel_size=1))
        
        # interpolate remainder
        self.resblock1 = ResnetBlocWithAttn(filters[i], filters[i]//4 * 3, norm_groups=4,
                                            dropout=dropout_rate, with_attn=True)
        self.resblock2 = ResnetBlocWithAttn(filters[i]//4 * 3, filters[i]//4 * 3, norm_groups=4,
                                            dropout=dropout_rate, with_attn=False)
        self.post_interp_conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(filters[i]//4 * 3, filters[i]//4 * 3,
                      kernel_size=3, stride=1, padding=0)            
        )
        self.embed_finescale = nn.Conv2d(hires_fields, (3*filters[i])//16, kernel_size=1)
        
        # cross-attention with context points        
        self.embed_crs_atn = nn.Conv2d(filters[i]//4 * 3 + (3*filters[i])//16, d_cross_attn-2, kernel_size=1)
        self.embed_context = nn.Linear(context_channels, d_cross_attn-2)
        self.sublayer = SublayerConnection(d_cross_attn, dropout_rate)
        self.cross_attn = Attention1D(attn_heads, d_cross_attn, dropout=dropout_rate)
        
        # output        
        self.resblock_final = ResnetBlocWithAttn(d_cross_attn, filters[-1], norm_groups=2,
                                                 dropout=dropout_rate, with_attn=False)
        self.gen_output = nn.Conv2d(filters[-1], output_channels, kernel_size=1)
        
    def forward(self, vae_normal_dists, fine_inputs, 
                context_data, context_locs, grid_locs,
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
            x = self.resnet_dict[f'first_{i}'](x, mask=masks[i])
            x = self.resnet_dict[f'second_{i}'](x, mask=masks[i])            
            x = self.refine[i](x)
            hi = self.coarsen[i](fine_inputs)
            x = self.joiner[i](torch.cat([x, hi], dim=1))     

        # the remainder
        x = self.resblock1(x, mask=masks[i+1])
        x = self.resblock2(x, mask=masks[i+1])
        remaining_scale = self.scale / (self.scale_factor**len(self.refine))
        x = nn.functional.interpolate(x, scale_factor=remaining_scale, mode='bilinear')
        x = self.post_interp_conv(x)

        # inject native res static data
        x = torch.cat([x, self.embed_finescale(fine_inputs)], dim=1)
        
        # embed to cross-attention dimension
        x = self.embed_crs_atn(x)
        context = self.embed_context(context_data.transpose(-2,-1)).transpose(-2,-1)
        
        # attach locations to front of vectors
        raw_H = x.shape[2]
        raw_W = x.shape[3]
        x = torch.reshape(x, (x.shape[0], x.shape[1], raw_H*raw_W))
        x = torch.cat([grid_locs, x], dim=1)
        
        context = torch.cat([context_locs, context], dim=1)
                
        # calculate cross attention with context points
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
                 context_channels=12,
                 latent_variables=12,
                 filters=[128, 64, 32],                 
                 dropout_rate=0.1,
                 scale=25,
                 attn_heads=2,
                 d_cross_attn=64):
        super().__init__()

        self.encoder = Encoder(input_channels=input_channels,
                               hires_fields=hires_fields,                 
                               latent_variables=latent_variables,
                               filters=filters,                               
                               dropout_rate=dropout_rate,
                               scale=scale)        

        self.decoder = Decoder(hires_fields=hires_fields,
                               output_channels=output_channels,
                               latent_variables=latent_variables,
                               filters=filters,
                               dropout_rate=dropout_rate,
                               scale=scale)

    def encode(self, coarse_inputs, fine_inputs, calc_kl=False, mask=None):
        vae_normal_dists, KL = self.encoder(coarse_inputs, fine_inputs,
                                            calc_kl=calc_kl, mask=mask)
        if calc_kl:
            return vae_normal_dists, KL.sum()
        else:
            return vae_normal_dists

    def decode(self, vae_normal_dists, fine_inputs, context_data,
               context_locs, grid_locs, masks=[None, None, None, None],
               context_mask=None):
        pred = self.decoder(vae_normal_dists, fine_inputs, context_data,
                            context_locs, grid_locs, masks=masks,
                            context_mask=context_mask)
        return pred

    def predict(self, coarse_inputs, fine_inputs, context_data,
                context_locs, grid_locs, masks=[None, None, None, None],
                context_mask=None):
        vae_normal_dists = self.encode(coarse_inputs, fine_inputs,
                                       calc_kl=False, mask=None)
        
        pred = self.decode(vae_normal_dists, fine_inputs, context_data,
                           context_locs, grid_locs, masks=masks,
                           context_mask=context_mask)
        return pred

    def forward(self, coarse_inputs, fine_inputs, context_data,
                context_locs, grid_locs, masks=[None, None, None, None],
                context_mask=None):
        vae_normal_dists, KL = self.encode(coarse_inputs, fine_inputs,
                                           calc_kl=True, mask=None)
        
        pred = self.decode(vae_normal_dists, fine_inputs, context_data,
                           context_locs, grid_locs, masks=masks,
                           context_mask=context_mask)
        # ELBO_i = log_lik - KL.sum()
        return pred, KL.sum()

if __name__=="__main__":
    
    if False:
        from setupdata import data_generator
        from params import model_pars, data_pars, train_pars
        
        model = MetVAE(input_channels=model_pars.input_channels,
                       hires_fields=10,
                       output_channels=model_pars.output_channels,
                       latent_variables=24,
                       filters=model_pars.filters,
                       dropout_rate=model_pars.dropout_rate,
                       scale=data_pars.scale)
        model.eval()
        
        datgen = data_generator()
        batch = datgen.get_batch(batch_size=train_pars.batch_size,
                                 batch_type='train',
                                 load_binary_batch=True)
        station_targets = batch[2]
        batch = Batch(batch)
        
        hires_at_lores = model.encoder.coarsen(batch.fine_inputs)
        joined_fields = torch.cat([batch.coarse_inputs, hires_at_lores], dim=1)    
        x = model.encoder.resblock1.res_block(joined_fields)
        x = model.encoder.resblock1.pe(x)
        x = model.encoder.resblock1.attn(x, mask=mask)

