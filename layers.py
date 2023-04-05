import math
import torch
from torch import nn
from inspect import isfunction

class CoarsenFields(nn.Module):
    def __init__(self, hires_fields, filters, scale=25):
        super().__init__()
        self.feat_map_conv = nn.Conv2d(hires_fields, filters[2], kernel_size=3, stride=1, padding="same")
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(filters[2]*scale*scale, 2*filters[0], kernel_size=1, stride=1, padding="valid")
        self.conv2 = nn.Conv2d(2*filters[0], filters[0], kernel_size=1, stride=1, padding="valid")
        self.scale = scale
        #self.conv3 = nn.Conv2d(filters[0], filters[0], kernel_size=1, stride=1, padding="valid")

    def forward(self, fine_input):
        # pre convolution to create feature map
        feat_map = self.feat_map_conv(fine_input)
        feat_map = self.relu(feat_map)
        # check ordering of channels, unshuffle works like
        # (*, C, H x r, W x r) -> (*, C x r^2, H, W)
        feat_map = torch.nn.functional.pixel_unshuffle(feat_map, self.scale)
        
        # pass through pixel-wise convolutions and activations
        feat_map = self.conv1(feat_map)
        feat_map = self.relu(feat_map)
        
        feat_map = self.conv2(feat_map)
        feat_map = self.relu(feat_map)        
        return feat_map

# PositionalEncoding Source https://github.com/lmnt-com/wavegrad/blob/master/src/wavegrad/model.py
class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, y):
        count = self.dim // 2
        step = torch.arange(count, dtype=y.dtype, device=y.device) / count
        encoding = noise_level.unsqueeze(1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        encoding = torch.cat([torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))

class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


# building block modules

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0):
        super().__init__()
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
        super().__init__()        
        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x):
        b, c, h, w = x.shape
        h = self.block1(x)        
        h = self.block2(h)
        return h + self.res_conv(x)

class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=32):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, input):
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head

        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)  # bhdyx

        attn = torch.einsum("bnchw, bncyx -> bnhwyx", query, key).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, height, width)

        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))

        return out + input

class ResnetBlocWithAttn(nn.Module):
    def __init__(self, dim, dim_out, *, norm_groups=32, dropout=0, with_attn=False):
        super().__init__()
        self.with_attn = with_attn
        self.res_block = ResnetBlock(
            dim, dim_out, norm_groups=norm_groups, dropout=dropout)
        if with_attn:
            self.attn = SelfAttention(dim_out, norm_groups=norm_groups)

    def forward(self, x):
        x = self.res_block(x)
        if(self.with_attn):
            x = self.attn(x)
        return x
        
def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d
