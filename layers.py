import math
import torch
import torch.nn.functional as F
import copy

from torch import nn
from inspect import isfunction
    
from utils import find_num_pools, decide_scale_factor

## some code from https://github.com/Janspiry/Palette-Image-to-Image-Diffusion-Models/blob/main/models/sr3_modules/unet.py
EPS = 1e-30

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])  
        
def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

class LayerNorm(nn.Module):
    "Construct a layernorm module."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2.to(x.device) * (x - mean) / (std + self.eps) + self.b_2.to(x.device)

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class GridPointAttn(nn.Module):
    def __init__(self, d_input, d_cross_attn, context_channels,
                 attn_heads, dropout_rate, pe=True):
        super(GridPointAttn, self).__init__()
        self.pe_bool = pe
        self.embed_grid = nn.Conv2d(d_input, d_cross_attn,
                                    kernel_size=1, stride=1, padding=0)
        self.embed_context = nn.Linear(context_channels, d_cross_attn)
        self.sublayer = SublayerConnection(d_cross_attn, dropout_rate)
        if self.pe_bool:
            self.pe_2d = PositionalEncoding2D(d_cross_attn)
            self.pe_offgrid = PositionalEncodingOffGrid(d_cross_attn)
        self.cross_attn = Attention1D(attn_heads, d_cross_attn,
                                      dropout=dropout_rate)
                                     
    def forward(self, x, context_data, context_locs, raw_size,
                mask=None, softmask=None, pixel_passer=None):
        if mask is None: mask = [None] * len(context_data)
        if softmask is None: softmask = [None] * len(context_data)
        if pixel_passer is None: pixel_passer = [None] * len(context_data)
        
        x = self.embed_grid(x)
        attn_out = []
        raw_H = x.shape[2]
        raw_W = x.shape[3]
        if self.pe_bool:
            # do positional encoding
            x = x + self.pe_2d(x.permute(0,2,3,1)).permute(0,3,1,2)

        for b in range(len(context_data)):
            if context_data[b].shape[-1]==0: # 0 as we have removed null tags
                attn_out.append(x[b])
            else:      
                # take batch slice and reshape to 1D vector list
                x_b = x[b:(b+1)]
                x_b = torch.reshape(x_b, (x_b.shape[0], x_b.shape[1], raw_H*raw_W))
                
                # embed context data and do off-grid positional encoding 
                context_b = self.embed_context(context_data[b].transpose(-2,-1)).transpose(-2,-1)
                if self.pe_bool:
                    context_b = context_b + self.pe_offgrid(context_locs[b]/(raw_size - 1) * (raw_H - 1))
                                        
                # calculate cross-attention
                x_b = x_b.transpose(-2,-1)
                x_b = self.sublayer(x_b, 
                    lambda x_b: self.cross_attn(x_b,
                                                context_b.transpose(-2,-1),
                                                context_b.transpose(-2,-1),
                                                mask[b],
                                                softmask[b],
                                                pixel_passer[b])
                )
                x_b = torch.reshape(x_b.transpose(-2,-1),
                                    (x_b.shape[0], x_b.shape[-1], raw_H, raw_W))
                attn_out.append(x_b[0, :, :, :]) # remove batch dim
                
        # process outputs
        return torch.stack(attn_out, dim=0)


class CoarsenField(nn.Module):
    def __init__(self, hires_fields, filters, outsize,
                 scale=25, simple=False, pool='max'):
        super(CoarsenField, self).__init__()
        " filters are used sequentially from the zeroth element "
        self.scale = scale
        n = find_num_pools(scale, max_num=5, factor=3)
        self.npools = min(n, len(filters))      
        if pool=='max':
            self.nn_pool = nn.MaxPool2d(3)
        else:
            self.nn_pool = nn.AvgPool2d(3)
            
        coarsen_list = []
        for i in range(self.npools):
            if i==0:
                if simple:
                    coarsen_list.append(nn.Sequential(
                        nn.Conv2d(hires_fields, filters[i], kernel_size=3,
                                  stride=1, padding="same"),
                        nn.ReLU(),                        
                        self.nn_pool)
                    )

                else:
                    coarsen_list.append(nn.Sequential(
                        nn.Conv2d(hires_fields, filters[i], kernel_size=3,
                                  stride=1, padding="same"),
                        nn.ReLU(),                    
                        nn.Conv2d(filters[i], filters[i], kernel_size=3,
                                  stride=1, padding="same"),
                        self.nn_pool)
                    )
            else:
                if simple:
                    coarsen_list.append(nn.Sequential(
                        nn.Conv2d(filters[i-1], filters[i], kernel_size=3,
                                  stride=1, padding="same"),
                        nn.ReLU(),                        
                        self.nn_pool)
                    )                    
                else:                    
                    coarsen_list.append(nn.Sequential(
                        nn.Conv2d(filters[i-1], filters[i], kernel_size=3,
                                  stride=1, padding="same"),
                        nn.ReLU(),
                        nn.Conv2d(filters[i], filters[i], kernel_size=3,
                                  stride=1, padding="same"),
                        self.nn_pool)
                    )
        self.pool = nn.ModuleList(coarsen_list)
        
        if self.npools==0:
            self.embed = nn.Conv2d(hires_fields, filters[0], kernel_size=3,
                                   stride=1, padding="same")
        
        last_filt = min(max(1,self.npools), len(filters)-1)
        self.pre_interp_conv = nn.Conv2d(filters[max(0, self.npools-1)],
                                         filters[last_filt], kernel_size=1,
                                         stride=1, padding="same")
        self.post_interp_conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(filters[last_filt], outsize,
                      kernel_size=3, stride=1, padding=0)
        )        
        
    def forward(self, fine_inputs):
        # coarsening using pooling
        x = fine_inputs        
        if self.npools>0:            
            for i in range(len(self.pool)):
                x = self.pool[i](x)
        else:            
            x = self.embed(x)
            
        # final correction
        x = self.pre_interp_conv(x)
        scale_factor = decide_scale_factor(fine_inputs.shape[-2:], x.shape[-2:], self.scale)
        x = nn.functional.interpolate(x, scale_factor=scale_factor, mode='bilinear')
        x = self.post_interp_conv(x)        
        return x

class RefineField(nn.Module):
    def __init__(self, lores_fields, filters, outsize, scale=25, scale_factor=2):
        super(RefineField, self).__init__()        
        " filters are used sequentially from the zeroth element "
        self.scale = scale
        self.scale_factor = scale_factor
        n = find_num_pools(scale, max_num=5, factor=self.scale_factor)        
        self.nups = min(n, len(filters))
        self.upsample = nn.Upsample(scale_factor=self.scale_factor, mode='bilinear') 
        
        pre_refine_list = []
        post_refine_list = []        
        for i in range(self.nups):
            if i==0:
                pre_refine_list.append(
                    nn.Conv2d(lores_fields, filters[i], kernel_size=3, # kernel_size=1?
                              stride=1, padding="same")
                )
            else:
                pre_refine_list.append(
                    nn.Conv2d(filters[i-1], filters[i], kernel_size=3, # kernel_size=1?
                              stride=1, padding="same")                    
                )
            post_refine_list.append(nn.Sequential(
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(filters[i], filters[i],
                              kernel_size=3, stride=1, padding=0),
                    nn.ReLU()
                )
            )
        self.pre_ups = nn.ModuleList(pre_refine_list)
        self.post_ups = nn.ModuleList(post_refine_list)
        
        if self.nups==0:
            self.embed = nn.Conv2d(lores_fields, filters[0], kernel_size=3, # kernel_size=1?
                                   stride=1, padding="same")
        
        last_filt = min(max(1, self.nups), len(filters)-1)
        self.pre_interp_conv = nn.Conv2d(filters[max(0,self.nups-1)],
                                         filters[last_filt], kernel_size=3,  # kernel_size=1?
                                         stride=1, padding="same")
        self.post_interp_conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(filters[last_filt], outsize,
                      kernel_size=3, stride=1, padding=0)            
        )
        
    def forward(self, coarse_input):
        # refining using interpolation
        x = coarse_input
        if self.nups>0:
            for i in range(len(self.pre_ups)):
                x = self.pre_ups[i](x) # prepare for upsampling                
                x = self.upsample(x)
                x = self.post_ups[i](x) # post-process upsampling
        else:
            x = self.embed(x) 
        
        # final correction
        x = self.pre_interp_conv(x)
        scale_factor = self.scale / (self.scale_factor**self.nups)
        x = nn.functional.interpolate(x, scale_factor=scale_factor, mode='bilinear')
        x = self.post_interp_conv(x)
        return x


def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)

    
class PositionalEncoding2D(nn.Module):
    # source: https://github.com/tatp22/multidim-positional-encoding/blob/master/positional_encodings/torch_encodings.py
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding2D, self).__init__()
        self.org_channels = channels
        channels = int(math.ceil(channels / 4) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.cached_penc = None

    def forward(self, tensor):
        """
        :param tensor: A 4d tensor of size (batch_size, x, y, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, ch)
        """
        if len(tensor.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, y, x, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        emb_y = get_emb(sin_inp_y).unsqueeze(1)
        emb_x = get_emb(sin_inp_x)
        emb = torch.zeros((y, x, self.channels * 2), device=tensor.device).type(
            tensor.type()
        )
        emb[:, :, : self.channels] = emb_y
        emb[:, :, self.channels : 2 * self.channels] = emb_x

        self.cached_penc = emb[None, :, :, :orig_ch].repeat(tensor.shape[0], 1, 1, 1)
        return self.cached_penc # then add this to input tensor
        

class PositionalEncodingOffGrid(nn.Module):
    # source: https://github.com/tatp22/multidim-positional-encoding/blob/master/positional_encodings/torch_encodings.py
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncodingOffGrid, self).__init__()
        self.org_channels = channels
        channels = int(math.ceil(channels / 4) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, YX):
        self.cached_penc = None
        _, num = YX.shape
        sin_inp_y = torch.einsum("i,j->ij", YX[0,:], self.inv_freq)
        sin_inp_x = torch.einsum("i,j->ij", YX[1,:], self.inv_freq)
        
        emb_y = get_emb(sin_inp_y).unsqueeze(1)
        emb_x = get_emb(sin_inp_x)
        emb = torch.zeros((num, num, self.channels * 2), device=YX.device).type(YX.type())
        emb[:, :, : self.channels] = emb_y
        emb[:, :, self.channels : 2 * self.channels] = emb_x
        
        return torch.diagonal(emb[:,:,:self.org_channels], dim1=0, dim2=1)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0):
        super(Block, self).__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(dim, dim_out, 3, padding=1)
        )

    def forward(self, x):
        return self.block(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, dropout=0, norm_groups=32):
        super(ResnetBlock, self).__init__()        
        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x):
        b, c, h, w = x.shape
        h = self.block1(x)        
        h = self.block2(h)
        return h + self.res_conv(x)


class SelfAttention2D(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=32):
        super(SelfAttention2D, self).__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, input, mask=None):
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head

        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)  # bhdyx

        attn = torch.einsum("bnchw, bncyx -> bnhwyx", query, key).contiguous() / math.sqrt(channel)
        if mask is not None:
            # mask would apply here, attn.masked_fill(mask == 0, -1e9)
            mask = mask.unsqueeze(1) # spread across all heads
            attn = attn.masked_fill(mask == 0, -1e9)
        
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, height, width)

        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))

        return out + input


class ResnetBlocWithAttn(nn.Module):
    def __init__(self, dim, dim_out, *, norm_groups=32, dropout=0, with_attn=False):
        super(ResnetBlocWithAttn, self).__init__()
        self.with_attn = with_attn        
        self.res_block = ResnetBlock(
            dim, dim_out, norm_groups=norm_groups, dropout=dropout)
        if with_attn:
            self.pe = PositionalEncoding2D(dim_out)
            self.attn = SelfAttention2D(dim_out, norm_groups=norm_groups)
            #self.sublayer = SublayerConnection(dim, dropout)

    def forward(self, x, mask=None):
        x = self.res_block(x)
        if(self.with_attn):        
            # (B, C, Y, X) -> (B, Y, X, C) -> PE -> (B, C, Y, X)
            x = self.pe(x.permute(0,2,3,1)).permute(0,3,1,2)
            x = self.attn(x, mask=mask)
        return x

def attention(query, key, value, mask=None, softmask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    
    if softmask is not None:
        #if scores.max() > 70:
        #    print('potential exp overflow in attn score softmasking')        
        #scores = torch.log(torch.exp(scores) * softmask)
        scores = scores + torch.log(softmask + EPS)
    elif mask is not None:        
        scores = scores.masked_fill(mask == 0, -1e9)
    
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn
    
class Attention1D(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(Attention1D, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)        
        
    def forward(self, query, key, value, mask=None, softmask=None, pixel_passer=None):
        # Same mask applied to all h heads.
        if softmask is not None:
            softmask = softmask.unsqueeze(1)
        elif mask is not None:            
            mask = mask.unsqueeze(1)
            
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value,
                                 mask=mask, 
                                 dropout=self.dropout,
                                 softmask=softmask)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
             
        if pixel_passer is not None:
            return self.linears[-1](x) * torch.reshape(pixel_passer, (pixel_passer.shape[0], pixel_passer.shape[1]*pixel_passer.shape[2])).unsqueeze(-1)
        else:
            return self.linears[-1](x)
