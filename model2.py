import torch
import torch.nn as nn
#from torch.distributions.normal import Normal
#from torch.distributions import kl

from layers import find_num_pools, GridPointAttn

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

        self.gen_output = nn.Conv2d(ds_cross_attn[self.nups+1]//2, output_channels,
                                    kernel_size=1, stride=1, padding=0)  

    # coarse_inputs=batch.coarse_inputs
    # fine_inputs=batch.fine_inputs
    # context_data=batch.context_data
    # context_locs=batch.context_locs            
    # context_masks=masks['context_masks']
    # context_soft_masks=masks['context_soft_masks']
    # pixel_passers=masks['pixel_passers']
    def forward(self, coarse_inputs, fine_inputs,
                context_data, context_locs,
                context_masks=[None, None, None, None, None],
                context_soft_masks=[None, None, None, None, None],
                pixel_passers=[None, None, None, None, None]):
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
                                   pixel_passer=pixel_passers[0])

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
                                        pixel_passer=pixel_passers[i+1])
            
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
                                      pixel_passer=pixel_passers[i+2])

        # residual
        x = torch.cat([x, x_res], dim=1)

        # output
        x = self.act(self.pre_output(x))
        output = self.gen_output(x)
        return output


