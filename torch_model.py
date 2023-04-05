import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions import kl

from layers import ResnetBlocWithAttn, CoarsenFields

class Encoder(nn.Module):
    def __init__(self,
                 input_channels=9,
                 hires_fields=2,                 
                 latent_variables=1,
                 filters=[256, 128, 64],                 
                 dropout_rate=0.1,
                 scale=25):
        super().__init__()
        ## lo-res feature map from hi-res static fields
        self.coarsen = CoarsenFields(hires_fields, filters, scale=scale)

        ## redidual blocks       
        self.resblock1 = ResnetBlocWithAttn(input_channels+filters[0], filters[0], norm_groups=2,
                                            dropout=dropout_rate, with_attn=True)
        self.resblock2 = ResnetBlocWithAttn(filters[0], filters[0], norm_groups=32,
                                            dropout=dropout_rate, with_attn=False)
        
        ## create latent vars
        self.create_vae_latents = nn.Conv2d(filters[0], 2*latent_variables, kernel_size=1)
        
    def forward(self, coarse_inputs, fine_inputs, calc_kl=False):        
        hires_at_lores = self.coarsen(fine_inputs)
        joined_fields = torch.cat([coarse_inputs, hires_at_lores], dim=1)
        joined_fields = self.resblock1(joined_fields)
        joined_fields = self.resblock2(joined_fields)
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
                                           vae_latents[:,2*i+1,:,:].exp()+1e-5))
        
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
                 filters=[256, 128, 64],                 
                 dropout_rate=0.1,
                 scale=25):
        super().__init__()
        
        # upscale block 1
        self.resblock1 = ResnetBlocWithAttn(output_channels, filters[0], norm_groups=2,
                                            dropout=dropout_rate, with_attn=True)
        self.resblock2 = ResnetBlocWithAttn(filters[0], filters[0], norm_groups=32,
                                            dropout=dropout_rate, with_attn=False)
        self.conv_a = nn.Conv2d(filters[0], scale*8, kernel_size=1, stride=1, padding="same")

        # injecting coarsened static fields
        self.intermediate_coarsener = CoarsenFields(hires_fields, filters, scale=int(np.sqrt(scale)))
        self.conv_b = nn.Conv2d(filters[0], hires_fields*2, kernel_size=1, stride=1, padding="same")

        # upscale block 2
        self.resblock3 = ResnetBlocWithAttn(8 + hires_fields*2, filters[1], norm_groups=2,
                                            dropout=dropout_rate, with_attn=True)
        self.resblock4 = ResnetBlocWithAttn(filters[1], filters[1], norm_groups=16,
                                            dropout=dropout_rate, with_attn=False)        
        self.conv_c = nn.Conv2d(filters[1], scale*4, kernel_size=1, stride=1, padding="same")
        
        # output
        self.resblock5 = ResnetBlocWithAttn(4 + hires_fields, filters[2], norm_groups=1,
                                            dropout=dropout_rate, with_attn=False)
        self.resblock6 = ResnetBlocWithAttn(filters[2], filters[2], norm_groups=8,
                                            dropout=dropout_rate, with_attn=False)
        self.gen_output = nn.Conv2d(filters[2], output_channels, kernel_size=1)
        self.scale = scale
        
    def forward(self, vae_normal_dists, fine_inputs):           
        # do reparam trick sampling
        sampled_vars = []
        for i in range(len(vae_normal_dists)):
            sampled_vars.append(vae_normal_dists[i].rsample())
        
        # think of these as compressed versions of the 
        # 100x100 variables that we want at output.
        # We now uncompress and combine with the hi-res static fields
        x = torch.stack(sampled_vars, dim=1)
        
        ## Upsampling from (4,4) to (100,100) with depth_to_space in two blocks
        # pass through residual blocks, mixing the variables!
        x = self.resblock1(x)
        x = self.resblock2(x)
        # depth to space with block size sqrt(scale) == sqrt(25)
        x = self.conv_a(x) # prepare channels
        x = nn.functional.pixel_shuffle(x, int(np.sqrt(self.scale)))

        ## inject coarsened static data
        feat_map = self.intermediate_coarsener(fine_inputs)
        feat_map = self.conv_b(feat_map)
        x = torch.cat([x, feat_map], dim=1)

        # pass through residual blocks, mixing the variables!
        x = self.resblock3(x)
        x = self.resblock4(x)
        # depth to space with block size sqrt(scale) == sqrt(25)
        x = self.conv_c(x) # prepare channels
        x = nn.functional.pixel_shuffle(x, int(np.sqrt(self.scale)))

        ## inject raw static data
        x = torch.cat([x, fine_inputs], dim=1)

        # outputs
        x = self.resblock5(x)
        x = self.resblock6(x)        
        output = self.gen_output(x)
        return output


class MetVAE(nn.Module):
    def __init__(self,
                 input_channels=6,
                 hires_fields=9,
                 output_channels=6,
                 latent_variables=6,
                 filters=[256, 128, 64],                 
                 dropout_rate=0.1,
                 scale=25):
        super().__init__()

        self.encoder = Encoder(input_channels=input_channels,
                               hires_fields=hires_fields,                 
                               latent_variables=latent_variables,
                               filters=filters,                               
                               dropout_rate=dropout_rate,
                               scale=scale)        

        self.decoder = Decoder(hires_fields=hires_fields,
                               output_channels=output_channels,
                               filters=filters,
                               dropout_rate=dropout_rate,
                               scale=scale)

    def encode(self, coarse_inputs, fine_inputs, calc_kl=False):
        # two step transposition to keep Y, X ordering
        #coarse_inputs = torch.transpose(torch.transpose(coarse_inputs, -2, -1), -3, -2)
        #fine_inputs = torch.transpose(torch.transpose(fine_inputs, -2, -1), -3, -2)
        vae_normal_dists, KL = self.encoder(coarse_inputs, fine_inputs, calc_kl=calc_kl)
        if calc_kl:
            return vae_normal_dists, KL.sum()
        else:
            return vae_normal_dists

    def decode(self, vae_normal_dists, fine_inputs):
        #fine_inputs = torch.transpose(torch.transpose(fine_inputs, -2, -1), -3, -2)
        pred = self.decoder(vae_normal_dists, fine_inputs)
        return pred

    def predict(self, coarse_inputs, fine_inputs):
        vae_normal_dists = self.encode(coarse_inputs, fine_inputs, calc_kl=False)
        pred = self.decode(vae_normal_dists, fine_inputs)
        return pred

    def forward(self, coarse_inputs, fine_inputs):
        vae_normal_dists, KL = self.encode(coarse_inputs, fine_inputs, calc_kl=True)
        pred = self.decode(vae_normal_dists, fine_inputs)
        # ELBO_i = log_lik - KL.sum()
        '''
        So now I need to recast the loss functions as log likelihoods!
        Might be tricky with the combo of station pixels, constraint maps and other loss parts....                
        But then I will truly be minimizing the ELBO....
        '''
        return pred, KL.sum()
