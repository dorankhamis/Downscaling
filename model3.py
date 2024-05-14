import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import find_num_pools, GridPointAttn

class ConvDownscaler(nn.Module):
    def __init__(self,
                 input_channels=1,
                 hires_fields=3,
                 output_channels=1,                 
                 filters=[6, 12, 24, 48],
                 dropout_rate=0.1,
                 scale=28,
                 scale_factor=3,
                 final_relu=False):
        super().__init__()
        self.nups = find_num_pools(scale, factor=scale_factor)
        self.scale = scale
        self.scale_factor = scale_factor
        self.final_relu = final_relu

        self.coarse_process = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(input_channels + hires_fields,
                      filters[0],
                      kernel_size=3, stride=1, padding=0),
            nn.GELU()
        )
        self.ups = nn.Upsample(scale_factor=scale_factor, mode='bilinear')
        cur_size = filters[0] + input_channels # sum from residual
        process = []        
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
            cur_size += (filters[i] + hires_fields) # sum from residual
        self.process = nn.ModuleList(process)        

        self.pre_output = nn.Conv2d(cur_size,
                                    filters[1],
                                    kernel_size=1, stride=1, padding=0)
        self.act = nn.GELU()

        self.gen_output = nn.Conv2d(filters[1], output_channels,
                                    kernel_size=1, stride=1, padding=0)  

    def forward(self, coarse_inputs, fine_inputs, constraint_inds=None):
        # find intermediate scales
        coarse_size = coarse_inputs.shape[-1]
        hires_size = fine_inputs.shape[-1]
        sizes = [coarse_size * (self.scale_factor**i) for i in range(1, self.nups+1)]
        final_scale = coarse_size * self.scale / sizes[-1]
        
        # processing at coarsest resolution to share large scale infomation
        x_res = coarse_inputs
        sf = nn.functional.interpolate(fine_inputs, scale_factor=1./self.scale, mode='bilinear')
        x = torch.cat([coarse_inputs, sf], dim=1)
        x = self.coarse_process(x)
        # residual
        x = torch.cat([x, x_res], dim=1)        
        
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
            
            # residual
            x = torch.cat([x, x_res], dim=1)

        # final upscale
        fct = hires_size / sizes[-1] # == final_scale
        x_res = nn.functional.interpolate(x, scale_factor=fct, mode='bilinear')        
        x = self.process[i+1](x_res)

        # join raw static fields
        x = torch.cat([x, fine_inputs], dim=1)

        # residual
        x = torch.cat([x, x_res], dim=1)

        # output
        x = self.act(self.pre_output(x))
        output = self.gen_output(x)
        if self.final_relu: return F.relu(output)
        else: return output

class Resolver(nn.Module):
    def __init__(self,
                 hires_fields=3,
                 output_channels=1,
                 filters=[6, 12],
                 dropout_rate=0.1,
                 final_relu=False):
        super().__init__()        
        self.final_relu = final_relu
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.c1 = nn.Conv2d(hires_fields, filters[0],
                            kernel_size=1, stride=1, padding=0)
        self.c2 = nn.Conv2d(filters[0], filters[1],
                            kernel_size=1, stride=1, padding=0)
        self.c3 = nn.Conv2d(filters[1], output_channels,
                            kernel_size=1, stride=1, padding=0)

    def forward(self, coarse_inputs, fine_inputs, constraint_inds):
        # output
        x = self.dropout(self.act(self.c1(fine_inputs)))
        x = self.dropout(self.act(self.c2(x)))
        resolver = self.c3(x)
        if type(constraint_inds)==int:
            output = resolver * fine_inputs[:,constraint_inds:(constraint_inds+1),:,:]
        else:
            output = resolver * fine_inputs[:,constraint_inds,:,:]
        if self.final_relu: return F.relu(output)
        else: return output


