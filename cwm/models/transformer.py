from functools import partial
import copy
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import cwm.models.patches as patches
import cwm.models.masking as masking
from cwm.models.utils import (activation_func,
                              normalization_func,
                              num_parameters)

from einops import rearrange
from torch import einsum

def _no_grad_trunc_normal_(x, mean, std, a, b):
    
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        x.uniform_(2 * l - 1, 2 * u - 1)
        x.erfinv_()
        x.mul_(std * math.sqrt(2.))
        x.add_(mean)
        x.clamp_(min=a, max=b)
        return x

def trunc_normal_(x, mean=0.0, std=1.0):
    _no_grad_trunc_normal_(x, mean, std, a=-std, b=std)
        
def pos_embedding(positions, hidden_dim, device='cuda'):

    if isinstance(positions, int):
        positions = torch.arange(positions).float().to(device)
    elif isinstance(positions, torch.Tensor):
        positions = positions.clone().detach().float().requires_grad_(True).to(device)
    else:
        assert hasattr(positions, '__len__')
        positions = torch.tensor(positions, dtype=torch.float).to(device)
    freqs = torch.arange(hidden_dim).float().to(device)
    # freqs = torch.pow(10000, 2 * (freqs // 2) / hidden_dim)
    freqs = torch.pow(10000, 2 * (torch.div(freqs, 2, rounding_mode='trunc')) / hidden_dim)    
    out = positions[:,None] / freqs[None,:]
    out[:, 0::2] = torch.sin(out[:, 0::2]) # dim 2i
    out[:, 1::2] = torch.cos(out[:, 1::2]) # dim 2i+1
    return out.unsqueeze(0)

def drop_path(x, drop_prob: float=0.0, training: bool=False, scale_by_keep: bool=True):
    """Based on timm drop path"""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class Mlp(nn.Module):
    def __init__(self, in_dim, out_dim=None, hidden_dim=None, activation='gelu', dropout_prob=0.0):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim or self.in_dim
        if hidden_dim is None:
            self.hidden_dim = self.in_dim
        else:
            self.hidden_dim = hidden_dim
        if not hasattr(self.hidden_dim, '__len__'):
            self.hidden_dim = [self.hidden_dim]

        self.activation = activation            
        self.layers = self._make_layers()
        self.dropout = nn.Dropout(dropout_prob)

    def _make_layers(self):
        dim_now = self.in_dim
        dims = self.hidden_dim + [self.out_dim]
        n_layers = len(dims)
        layers = []
        for idx, out_dim in enumerate(dims):
            layers.append(nn.Linear(dim_now, out_dim))
            if idx < (n_layers - 1):
                layers.append(activation_func(self.activation))
            dim_now = out_dim
        return nn.Sequential(*layers)
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.dropout(x)
        return x

class Attention(nn.Module):
    def __init__(self,
                 in_dim,
                 num_heads=8,
                 head_dim=None,
                 out_dim=None,
                 qkv_bias=False,
                 qk_scale=None,
                 attention_dropout_prob=0.0,
                 projection_dropout_prob=0.0,
                 flash_attention=False,
                 **kwargs):
        super().__init__()
        self.in_dim = in_dim
        self.num_heads = self.H = num_heads
        if out_dim is None:
            self.head_dim = head_dim or (self.in_dim // self.num_heads)
            self.out_dim = self.head_dim * self.num_heads
        else:
            self.out_dim = out_dim
            self.head_dim = head_dim or (self.out_dim // self.num_heads)
        self.scale = qk_scale or (self.head_dim ** -0.5)

        self.qkv = nn.Linear(self.in_dim, self.head_dim * self.num_heads * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(self.head_dim * self.num_heads))
            self.v_bias = nn.Parameter(torch.zeros(self.head_dim * self.num_heads))
        else:
            self.q_bias = self.v_bias = None

        self.flash_attention = flash_attention
        if self.flash_attention:
            from flash_attn.flash_attention import FlashAttention
            self.fa = FlashAttention(softmax_scale=1, attention_dropout=attention_dropout_prob)

        self.attn_drop = nn.Dropout(attention_dropout_prob)
        self.projection = nn.Linear(self.head_dim * self.num_heads, self.out_dim)
        self.proj_drop = nn.Dropout(projection_dropout_prob)

    def forward(self, x, return_attention=False):

        B,N,C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat([self.q_bias,
                                  torch.zeros_like(self.v_bias, requires_grad=False),
                                  self.v_bias], 0)

        qkv = F.linear(x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B,N,3,self.num_heads,self.head_dim).permute(2,0,3,1,4) # [3,B,H,N,D]
        q, k, v = list(qkv)
        q = q * self.scale
        t1 = time.time()
        if self.flash_attention:
            print("using flash attention (custom)")
            qkv = torch.stack([q,k,v], 0)
            qkv = qkv.permute(1,3,0,2,4)
            x, _ = self.fa(qkv)
            x = x.permute(0, 2, 1, 3)
        else:
            attn = (q @ k.transpose(-2, -1)).softmax(-1) # [B,H,N,N]
            if return_attention:
                return attn
            attn = self.attn_drop(attn)
            x = (attn @ v)
        t2 = time.time()
        print("custom attention %s flash attention: %0.6f" % (['WITHOUT', 'WITH'][int(self.flash_attention)],
                                                       (t2-t1)))
            
        x = x.transpose(1, 2).reshape(B,N,self.head_dim*self.num_heads) # [B,N,H*D]
        x = self.projection(x)
        x = self.proj_drop(x)

        return x

class UnidirectionalCrossAttention(nn.Module):
    """Attention that passes info from src to target stream"""
    def __init__(self,
                 in_dim,
                 num_heads,                 
                 in_dim_src=None,
                 head_dim=None,
                 out_dim=None,
                 qkv_bias=False,
                 qk_scale=None,
                 attention_dropout_prob=0,
                 projection_dropout_prob=0,
                 **kwargs):
        super().__init__()
        self.in_dim = in_dim
        self.in_dim_src = in_dim_src or self.in_dim
        self.num_heads = self.H = num_heads
        self.head_dim = head_dim or (self.in_dim // self.num_heads)
        self.out_dim = out_dim or self.in_dim
        self.scale = qk_scale or (self.head_dim ** -0.5)

        self.qv = nn.Linear(self.in_dim_src,
                            self.head_dim * self.num_heads * 2,
                            bias=False)
        self.k = nn.Linear(self.in_dim, self.head_dim * self.num_heads,
                           bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(self.out_dim))
            self.v_bias = nn.Parameter(torch.zeros(self.out_dim))
        else:
            self.q_bias = self.v_bias = None

        self.attn_drop = nn.Dropout(attention_dropout_prob)
        self.projection = nn.Linear(self.head_dim * self.num_heads,
                                    self.out_dim)
        self.proj_drop = nn.Dropout(projection_dropout_prob)

    def forward(self, x, src=None):

        ## x and src may be concatenated along last dimension
        if src is None:
            x, src = x.split([self.in_dim, self.in_dim_src], -1)

        B,N,C = x.shape
        B,M,D = src.shape
        qv_bias = None

        if self.q_bias is not None:
            qv_bias = torch.cat([self.q_bias,
                                 self.v_bias], 0)
        qv = F.linear(src, weight=self.qv.weight, bias=qv_bias)
        qv = qv.reshape(B,M,2,self.num_heads,self.head_dim).permute(2,0,3,1,4) # [2,B,H,M,D]
        q,v = list(qv)

        k = self.k(x).view(B,N,self.num_heads,self.head_dim)
        k = k.permute(0,2,3,1) # [B,H,D,N]        
        k = k * self.scale
        
        attn = (q @ k).transpose(-2,-1).softmax(-1) # [B,H,N,M]
        attn = self.attn_drop(attn)
        y = (attn @ v).transpose(1,2).reshape(B,N,self.head_dim*self.num_heads) # [B,N,H*D]
        y = self.projection(y)
        y = self.proj_drop(y)
        
        return (y, None)

class BidirectionalCrossAttention(nn.Module):
    """Exchange information between two sets of input tokens"""
    def __init__(self,
                 in_dim,
                 num_heads,
                 shared_similarity=False,
                 in_dim_src=None,
                 head_dim=None,
                 out_dim=None,
                 out_dim_src=None,
                 qkv_bias=False,
                 qk_scale=None,
                 attention_dropout_prob=0,
                 projection_dropout_prob=0,
                 flash_attention=False):
        super().__init__()
        self.in_dim = in_dim
        self.in_dim_src = in_dim_src or self.in_dim
        self.num_heads = self.H = num_heads
        self.head_dim = head_dim or (self.in_dim // self.num_heads)
        self.out_dim = out_dim or self.in_dim
        self.out_dim_src = out_dim_src or self.in_dim_src
        self.scale = qk_scale or (self.head_dim ** -0.5)

        ## whether to share a similarity matrix
        self.shared_similarity = shared_similarity

        ## need qkv embeddings for both streams
        self.qk = nn.Linear(self.in_dim, self.D * 2, bias=False)
        self.qk_src = nn.Linear(self.in_dim_src, self.D * 2, bias=False)
        self.v = nn.Linear(self.in_dim, self.D, bias=False)
        self.v_src = nn.Linear(self.in_dim_src, self.D, bias=False)

        if qkv_bias:
            self.qkv_bias = torch.cat([self._build_bias() for _ in range(3)], 0)
            self.qkv_bias_src = torch.cat([self._build_bias() for _ in range(3)], 0)            
        else:
            self.qkv_bias = self.qkv_bias_src = None

        self.flash_attention = flash_attention
        if self.flash_attention:
            from flash_attn.modules.mha import FlashCrossAttention
            self.fa = FlashCrossAttention(softmax_scale=1,
                                          attention_dropout=attention_dropout_prob)
            
        self.attn_drop = nn.Dropout(attention_dropout_prob)

        ## Linear outputs
        self.projection = nn.Linear(self.D, self.out_dim)
        self.projection_src = nn.Linear(self.D, self.out_dim_src)

        self.proj_drop = nn.Dropout(projection_dropout_prob)

    @property
    def D(self):
        """Inner dim"""
        return self.num_heads * self.head_dim

    def _build_bias(self):
        return nn.Parameter(torch.zeros(self.D))

    def forward(self, x, src=None):
        
        ## x and src may be concatenated along last dimension
        concat_output = False
        if src is None:
            x, src = x.split([self.in_dim, self.in_dim_src], -1)
            concat_output = True

        B,N,C = x.shape
        B,M,C_src = src.shape
        qkv_bias = qkv_bias_src = None

        if self.qkv_bias is not None:
            qkv_bias = self.qkv_bias
            qkv_bias_src = self.qkv_bias_src
        else:
            qkv_bias = qkv_bias_src = torch.zeros([3*self.D], requires_grad=False).to(x.device).to(x.dtype)

        ## linear embeddings of the base and src streams
        qk = F.linear(x, weight=self.qk.weight, bias=qkv_bias[:2*self.D]) # [B,N,H*Dh*2]
        qk_src = F.linear(src, weight=self.qk_src.weight, bias=qkv_bias_src[:2*self.D]) # [B,M,H*Dh*2]
        v = F.linear(x, weight=self.v.weight, bias=qkv_bias[-self.D:]) # [B,N,H*Dh]
        v_src = F.linear(src, weight=self.v_src.weight, bias=qkv_bias_src[-self.D:]) # [B,M,H*Dh]

        ## dimensional gymnastics and attention ops, similarity is not shared
        _rearr = lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.H)
        qk, qk_src, v, v_src = map(_rearr, (qk, qk_src, v, v_src))

        if self.flash_attention:
            print(qk.shape, v.shape, qk_src.shape, v_src.shape)
            qk = qk * self.scale
            qk_src = qk_src * self.scale
            y = self.fa(
                q=v.permute(0, 2, 1, 3), # [B, N, H, D]
                kv=qk_src.view(*qk_src.shape[:3], -1, 2).permute(0, 2, 4, 1, 3))
            print("ca out", y.shape)
            raise NotImplementedError("Trying to do flash cross attention")
        else:

            if self.shared_similarity:
                sim = einsum('bhnd,bhmd->bhnm', qk * self.scale, qk_src) # [B,H,N,M]
                attn = sim.softmax(-1)
                attn_src = sim.transpose(-2,-1).softmax(-1)
            else:
                attn = einsum('bhnd,bhmd->bhnm',
                              qk[...,0:self.head_dim] * self.scale,
                              qk_src[...,0:self.head_dim]
                ).softmax(-1) # [B,H,N,M]
                attn_src = einsum('bhnd,bhmd->bhmn',
                                  qk[...,self.head_dim:] * self.scale,
                                  qk_src[...,self.head_dim:]
                ).softmax(-1) # [B,H,M,N]
            attn, attn_src = self.attn_drop(attn), self.attn_drop(attn_src)

            ## compute updates and project to output dim
            _rearr = lambda t: rearrange(t, 'b h n d -> b n (h d)')
            y = _rearr(attn @ v_src)
            y_src = _rearr(attn_src @ v)

        y = self.proj_drop(self.projection(y))
        y_src = self.proj_drop(self.projection_src(y_src))

        if concat_output:
            return torch.cat([y, y_src], -1)
        return (y, y_src)

class TransformerBlock(nn.Module):

    def __init__(self,
                 in_dim,
                 num_heads=8,
                 head_dim=None,
                 out_dim=None,
                 mlp_ratio=4.0,
                 qkv_bias=False,
                 qk_scale=None,
                 dropout_prob=0.0,
                 attention_dropout_prob=0.0,
                 drop_path_prob=0.0,
                 init_values=None,
                 activation='gelu',
                 normalization={'func': 'layer', 'eps': 1e-6},
                 attention_func=Attention,
                 **kwargs):
        
        super().__init__()
        self.in_dim = in_dim
        self.norm1 = normalization_func(normalization, in_dim)
        self.attention = attention_func(
            in_dim=in_dim,
            head_dim=head_dim,            
            out_dim=out_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attention_dropout_prob=attention_dropout_prob,
            projection_dropout_prob=dropout_prob,
            **kwargs)
        self.out_dim = self.attention.out_dim

        if self.in_dim == self.out_dim:
            self.shortcut = nn.Identity(inplace=True)
        else:
            self.shortcut = nn.Linear(self.in_dim, self.out_dim, bias=False)
        
        self.drop_path = DropPath(drop_path_prob) if (drop_path_prob > 0.0) else nn.Identity()
        self.norm2 = normalization_func(normalization, self.out_dim)
        if mlp_ratio > 0.0:
            self.mlp = Mlp(self.out_dim, hidden_dim=[int(self.out_dim*mlp_ratio)], out_dim=None,
                           activation=activation, dropout_prob=dropout_prob)
        else:
            self.mlp = nn.Identity()

        if (init_values or 0) > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((self.out_dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((self.out_dim)), requires_grad=True)
        else:
            self.gamma_1 = self.gamma_2 = None

    def forward(self, x):
        if self.gamma_1 is None:
            x = self.shortcut(x) + self.drop_path(self.attention(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = self.shortcut(x) + self.drop_path(self.gamma_1 * self.attention(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x

class CrossAttentionTransformerBlock(nn.Module):
    """A transformer layer that includes cross attention between two streams"""
    default_attention_func = BidirectionalCrossAttention
    def __init__(self,
                 in_dim,
                 num_heads,
                 in_dim_src=None,
                 head_dim=None,
                 out_dim=None,
                 out_dim_src=None,
                 mlp_ratio=4.0,
                 drop_path_prob=0.0,
                 init_values=None,                 
                 activation='gelu',
                 normalization={'func': 'layer', 'eps': 1e-6},                 
                 attention_func=default_attention_func,
                 with_self_attention=True,                 
                 shared_similarity=False,
                 **kwargs):

        super().__init__()
        self.in_dim = in_dim
        self.in_dim_src = in_dim_src or self.in_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        self.normalization = normalization
        self.norm1, self.norm1_src = self._build_norms(self.in_dim, self.in_dim_src, do_norm=with_self_attention)
        self.norm1_cross, self.norm1_src_cross = self._build_norms(self.in_dim, self.in_dim_src)

        ## first update to trg and src streams is optionally via self-attention
        func = Attention if with_self_attention else None
        self.self_attention = nn.ModuleDict([
            ('trg', self._build_attention(func, self.in_dim, out_dim, **kwargs)),
            ('src', self._build_attention(func, self.in_dim_src, out_dim_src, **kwargs))
        ])
        self.out_dim = getattr(self.self_attention['trg'], 'out_dim', None) \
            or (out_dim or self.in_dim)
        self.out_dim_src = getattr(self.self_attention['src'], 'out_dim', None) \
            or (out_dim_src or self.in_dim_src)
        
        self.shortcut = nn.ModuleDict([
            ('trg', nn.Identity(inplace=True) if self.in_dim == self.out_dim \
             else nn.Linear(self.in_dim, self.out_dim, bias=False)),
            ('src', nn.Identity(inplace=True) if self.in_dim_src == self.out_dim_src \
             else nn.Linear(self.in_dim_src, self.out_dim_src, bias=False))
        ])
        self.drop_path = DropPath(drop_path_prob) if (drop_path_prob > 0.0) else nn.Identity()
        self.norm2, self.norm2_src = self._build_norms(self.out_dim, self.out_dim_src)

        ## cross attention
        self.cross_attention = self._build_attention(attention_func,
                                                     in_dim=self.in_dim,
                                                     out_dim=self.out_dim,
                                                     in_dim_src=self.in_dim_src,
                                                     out_dim_src=self.out_dim_src,
                                                     shared_similarity=shared_similarity,
                                                     **kwargs)

        ## mlp
        self.activation = activation
        self.mlp_ratio = mlp_ratio
        dropout_prob = kwargs.get('dropout_prob', 0)
        self.mlp = nn.ModuleDict([
            ('trg', self._build_mlp(self.out_dim, dropout_prob)),
            ('src', self._build_mlp(self.out_dim_src, dropout_prob))
        ])

        ## init
        if (init_values or 0) > 0:
            _init = lambda dim: self._build_init_values(dim, init_values)
            self.gamma_1, self.gamma_1_cross, self.gamma_1_src, self.gamma_1_src_cross = \
                map(_init, (self.out_dim, self.out_dim, self.out_dim_src, self.out_dim_src))
            self.gamma_2, self.gamma_2_src = _init(self.out_dim), _init(self.out_dim_src)
        else:
            self.gamma_1 = self.gamma_1_cross = self.gamma_1_src = self.gamma_1_src_cross = 1.0
            self.gamma_2 = self.gamma_2_src = 1.0

        if not with_self_attention:
            self.gamma_1 = self.gamma_1_src = 0.0
        
    def _build_norms(self, dim, dim_src, do_norm=True):
        if not do_norm:
            return (nn.Identity(inplace=True), nn.Identity(inplace=True))
        return (
            normalization_func(self.normalization, dim),
            normalization_func(self.normalization, dim_src)
        )
        
    def _build_attention(self, func, in_dim, out_dim, **kwargs):
        if func is None:
            return nn.Identity(inplace=True)
        return func(in_dim=in_dim,
                    num_heads=self.num_heads,
                    head_dim=self.head_dim,
                    out_dim=out_dim,
                    **kwargs)

    def _build_mlp(self, in_dim, dropout_prob=0):
        if not (self.mlp_ratio > 0.0):
            return nn.Identity()
        return Mlp(in_dim, hidden_dim=[int(in_dim*self.mlp_ratio)], out_dim=None,
                   activation=self.activation, dropout_prob=dropout_prob)

    def _build_init_values(self, dim, val):
        return nn.Parameter(val * torch.ones((dim)), requires_grad=True)

    def _get_cross_attention(self, x, src):
        self._passthrough_src = False
        y, y_src = self.cross_attention(
            self.norm1_cross(x),
            self.norm1_src_cross(src)
        )
        if y_src is None:
            self._passthrough_src = True
        return {'trg': y, 'src': y_src}

    def forward(self, x, src=None):

        concat_output = False
        if src is None:
            x, src = x.split([self.in_dim, self.in_dim_src], -1)
            concat_output = True

        ## self-attention and cross-attention effects have same input, but are just added
        cross_effects = self._get_cross_attention(x, src)
        x = self.shortcut['trg'](x) + \
            self.drop_path(self.gamma_1 * self.self_attention['trg'](self.norm1(x))) + \
            self.drop_path(self.gamma_1_cross * cross_effects['trg'])
        if not self._passthrough_src:
            src = self.shortcut['src'](src) + \
                self.drop_path(self.gamma_1_src * self.self_attention['src'](self.norm1_src(src))) + \
                self.drop_path(self.gamma_1_src_cross * cross_effects['src'])

        ## mlp to output
        x = x + self.drop_path(self.gamma_2 * self.mlp['trg'](self.norm2(x)))
        if not self._passthrough_src:
            src = src + self.drop_path(self.gamma_2_src * self.mlp['src'](self.norm2_src(src)))

        if concat_output:
            return torch.cat([x, src], -1)
        return (x, src)
