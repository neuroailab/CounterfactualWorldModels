from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from torchvision.models import vgg16
from timm.data.constants import (IMAGENET_DEFAULT_MEAN,
                                 IMAGENET_DEFAULT_STD)
from einops import rearrange
import time

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 400, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        **kwargs
    }


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement 
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None, flash_attention=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.flash_attention = flash_attention

        if self.flash_attention:
            from flash_attn.flash_attention import FlashAttention
            self.fa = FlashAttention(softmax_scale=1, attention_dropout=attn_drop)

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask=None):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale

        t1 = time.time()
        if attn_mask is not None:
            assert not self.flash_attention, "Flash attention does not support attn_mask. Consider using casual mask or torch 2.0"
        if self.flash_attention:
            qkv = torch.stack([q, k, v], 0)
            qkv = qkv.permute(1, 3, 0, 2, 4)  # .shape # (B, S, 3, H, D)
            x, _ = self.fa(qkv)
            x = x.permute(0, 2, 1, 3)
        else:
            attn = (q @ k.transpose(-2, -1))
            if attn_mask is not None:
                attn = attn.masked_fill(attn_mask==1, float('-inf'))
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v)
        t2 = time.time()
        # print("attention %s flash attention: %0.6f" % (['WITHOUT', 'WITH'][int(self.flash_attention)],
        #                                                (t2-t1)))

        x = x.transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None, in_dim=None, flash_attention=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim, flash_attention=flash_attention)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if (init_values or 0) > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, attn_mask=None):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x), attn_mask=attn_mask))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), attn_mask=attn_mask))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=(16, 16), in_chans=3, embed_dim=768, num_frames=16, tubelet_size=2):
        super().__init__()
        img_size = to_2tuple(img_size)

        self.tubelet_size = int(tubelet_size)
        if num_frames is not None:
            self.num_frames = int(num_frames)
            self.num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * (num_frames // self.tubelet_size)
        else:
            self.num_frames = None
            self.num_patches = None
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(in_channels=in_chans, out_channels=embed_dim, 
                            kernel_size = (self.tubelet_size,  patch_size[0],patch_size[1]), 
                            stride=(self.tubelet_size,  patch_size[0],  patch_size[1]))

    def forward(self, x, **kwargs):
        B, C, T, H, W = x.shape
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        assert (H % self.proj.weight.size(-2) == 0) and (W % self.proj.weight.size(-1) == 0), \
            f"Input image size({H},{W}) must be divisible by patch size " + \
            f"({self.proj.weight.size(-2)},{self.proj.weight.size(-1)})"



        # Conv3D isn't implemented in mps
        if x.device.type == 'mps':
            t, h, w = self.proj.weight.shape[-3:]
            x_slices = [_x.reshape(B, C * t, H, W) for _x in torch.split(x, [t]*(T // t), dim=2)]
            weights = self.proj.weight.view(-1, self.proj.weight.size(1) * t, h, w)
            x_out = [F.conv2d(_x, weight=weights, bias=self.proj.bias, stride=(h, w)) for _x in x_slices]
            x = torch.stack(x_out, 2).flatten(2).transpose(1, 2)
        else:
            x = self.proj(x).flatten(2).transpose(1, 2)
        return x
    
# sin-cos position encoding
# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py#L31
def get_sinusoid_encoding_table(positions,
                                d_hid,
                                apply_sinusoid=True): 
    ''' Sinusoid position encoding table ''' 
    # TODO: make it with torch instead of numpy 
    def get_position_angle_vec(position): 
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)] 

    if isinstance(positions, int):
        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(positions)])
    else:
        assert hasattr(positions, '__len__')
        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in positions])
    if apply_sinusoid:
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i 
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i+1 

    return torch.FloatTensor(sinusoid_table).unsqueeze(0) 

def make_reconstruction_videos(ori_imgs,
                               bool_masked_pos,
                               outputs,
                               tubelet_size,
                               patch_size):
    
    height = width = 224//8

    img_squeeze = rearrange(ori_imgs,
                                'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c',
                                p0=tubelet_size,
                                p1=patch_size[0],
                                p2=patch_size[0])
    img_norm = img_squeeze
        
    img_patch = rearrange(img_norm, 'b n p c -> b n (p c)')

    outputs = rearrange(outputs,'b p c -> (b p) c').to(torch.float32)

    
    img_patch[bool_masked_pos] = outputs 

    #make mask
    mask = torch.ones_like(img_patch)
    mask[bool_masked_pos] = 0
    mask = rearrange(mask, 'b n (p c) -> b n p c', c=3)
    mask = rearrange(mask,
                         'b (t h w) (p0 p1 p2) c -> b c (t p0) (h p1) (w p2) ',
                         p0=tubelet_size,
                         p1=patch_size[0],
                         p2=patch_size[1],
                         h=height,
                         w=width)

    #save reconstruction video
    rec_imgs = rearrange(img_patch, 'b n (p c) -> b n p c', c=3)
    # Notice: To visualize the reconstruction video, we add the predict and the original mean and var of each patch.
    rec_imgs = rearrange(rec_imgs,
                             'b (t h w) (p0 p1 p2) c -> b c (t p0) (h p1) (w p2)',
                             p0=tubelet_size,
                             p1=patch_size[0],
                             p2=patch_size[1],
                             h=height,
                             w=width)


    return rec_imgs
