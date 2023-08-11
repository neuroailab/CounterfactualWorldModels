import os
import copy
import math
import numpy as np
from typing import Tuple, List, Optional, Union, Callable, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from functools import partial

from cwm.models.VideoMAE.vmae import (PretrainVisionTransformerDecoder,
                                      trunc_normal_)
from cwm.models.VideoMAE.conjoined_vmae import PaddedVisionTransformer

from cwm.models.VideoMAE.utils import (
    Block,
    _cfg,
    ImagePatchEmbed,
    get_sinusoid_encoding_table
)

_LayerNorm = partial(nn.LayerNorm, eps=1e-6)
TwoTuple = Tuple[int, int]

def _int_to_two_tuple(val):
    if isinstance(val, tuple) and len(val) == 2:
        return val
    assert isinstance(val, int), type(Val)
    return (val, val)

class ChannelMaeDecoder(nn.Module):
    """A stack of transformer layers that optionally returns only the last N tokens"""
    def __init__(
            self,
            embed_dim: int = 384,
            num_classes: int = 0,
            depth: int = 4,
            num_heads: int = 6,
            mlp_ratio: float = 4.0,
            qkv_bias: bool = True,
            qk_scale: Optional[float] = None,
            drop_rate: Optional[float] = None,
            attn_drop_rate: Optional[float] = None,
            drop_path_rate: Optional[float] = None,
            norm_layer: Callable = _LayerNorm,
            block_func: nn.Module = Block,
            block_kwargs: Dict = {},
            init_values: Optional[float] = None
    ) -> None:
        
        super().__init__()
        self.embed_dim = embed_dim

        # build transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate or 0.0, depth)]
        self.blocks = nn.ModuleList(
            [
                block_func(
                    dim=self.embed_dim,
                    in_dim=self.embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=(drop_rate or 0.0),
                    attn_drop=(attn_drop_rate or 0.0),
                    drop_path=dpr[idx],
                    norm_layer=norm_layer,
                    init_values=init_values,
                    **block_kwargs
                )
                for idx in range(depth)
            ]
        )

        # norm and head layers
        self.norm = norm_layer(self.embed_dim)
        self.head = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def get_last_tokens(self, x, return_token_num):
        if return_token_num > 0:
            return self.head(self.norm(x[:,-return_token_num:]))
        elif return_token_num == 0:
            return self.head(self.norm(x))[:,x.size(1):]
        else:
            return self.head(self.norm(x))

    def forward(self, x, return_token_num: int = -1):
        for blk in self.blocks:
            x = blk(x)

        return self.get_last_tokens(x, return_token_num)
        
class ChannelMaeEncoder(ChannelMaeDecoder):
    """An encoder for a MAE that treats groups of channels as different 'frames'"""
    def __init__(
            self,
            image_size: Union[int, TwoTuple] = (224, 224),
            patch_size: TwoTuple = (32, 32),
            in_channels: int = 3,
            channel_partition: Optional[Tuple[int]] = None,
            concat_base_channels: Optional[Union[Tuple[int, ...], List[int]]] = None,
            num_classes: int = 0,
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.0,
            qkv_bias: bool = True,
            qk_scale: Optional[float] = None,
            drop_rate: Optional[float] = None,
            attn_drop_rate: Optional[float] = None,
            drop_path_rate: Optional[float] = None,
            norm_layer: Callable = _LayerNorm,
            block_func: nn.Module = Block,
            block_kwargs: Dict = {},
            init_values: Optional[float] = None,
            use_learnable_pos_emb: bool = False
    ) -> None:

        # initialize Transformer blocks using parent class ("Decoder")
        super(ChannelMaeEncoder, self).__init__(
            embed_dim=embed_dim,
            num_classes=num_classes,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            block_func=block_func,
            block_kwargs=block_kwargs,
            init_values=init_values
        )

        # input size and tokenization parameters
        self.image_size = _int_to_two_tuple(image_size)
        self.in_channels = self.num_channels = in_channels
        self.patch_size = _int_to_two_tuple(patch_size)
        self.ph, self.pw = self.patch_size
        self.embed_dim = embed_dim        

        # for properties of parent class that involve time dimension        
        self.pt = 1
        self.num_frames = 1

        # how to group channels into 'frames'
        if channel_partition is None:
            self.channel_partition = (1, ) * self.num_channels
        else:
            assert sum(channel_partition) == self.num_channels
            self.channel_partition = channel_partition

        # whether to concatenate any channels onto every group
        self.concat_base_channels = concat_base_channels or []

        # create the patch embedding layers
        self._build_patch_embedding_layers()

        # create positional embedding, treating channel groups as 'frames'
        self.pos_embed = self._init_pos_embed(use_learnable_pos_emb)
        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=0.02)

        # init
        self.apply(self._init_weights)

    @property
    def num_base_channels(self):
        return sum(self.concat_base_channels)

    @property
    def num_channel_groups(self):
        return len(self.channel_partition)

    @property
    def num_patches(self):
        return sum(
            [patch_embed_group.get_num_patches(self.image_size) for patch_embed_group in self.patch_embed]
        )

    @property
    def mask_size(self):
        return (self.num_channel_groups, self.image_size[0] // self.ph, self.image_size[1] // self.pw)
        

    def _build_patch_embedding_layers(self) -> None:

        self.patch_embed = nn.ModuleList(
            [
                ImagePatchEmbed(
                    patch_size=self.patch_size,
                    embed_dim=self.embed_dim,
                    in_channels=num_channels_in_group + self.num_base_channels
                )
                for num_channels_in_group in self.channel_partition
            ]
        )

    def _init_pos_embed(self, use_learnable_pos_emb: bool = False) -> torch.Tensor:

        if use_learnable_pos_emb:
            return nn.Parameter(torch.zeros(1, self.num_patches, self.embed_dim))

        return get_sinusoid_encoding_table(self.num_patches, self.embed_dim)

    def tokenize(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Tokenize by applying each patch embedding layer to the respective channel group"""
        
        assert len(x.shape) == 4, f"input must be a [batch, channels, height, width] image: {x.shape}"
        xs = torch.split(x, self.channel_partition, dim=1)

        # concat the base channels if there are any
        if len(self.concat_base_channels):
            x_base = torch.index_select(
                x, dim=1, index=torch.tensor(self.concat_base_channels, device=x.device, dtype=torch.long)
            )
            xs = [torch.cat([x, x_base], dim=1) for x in xs]

        # apply separate patch embedding to each group
        x = torch.cat(
            [
                patch_embed_group(xs[group_idx])
                for group_idx, patch_embed_group in enumerate(self.patch_embed)
            ],
            dim=1
        )

        # add pos embed
        pos_embed = self.pos_embed.type_as(x).to(x.device).clone().detach()
        x = x + pos_embed

        return (x, mask)

    def forward_features(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:

        try:
            B, C, H, W = x.shape
        except:
            assert (len(x.shape) == 5) and (x.shape[2] == 1), x.shape
            x = x.squeeze(2)
            B, C, H, W = x.shape
        assert C == self.num_channels, (C, self.num_channels)

        x, mask = self.tokenize(x, mask)
        B, _, D = x.shape

        x_vis = x[~mask].reshape(B, -1, D)

        for blk in self.blocks:
            x_vis = blk(x_vis)

        x_vis = self.norm(x_vis)
        return x_vis

    def forward(self, x, mask = None, *args, **kwargs):
        if mask is None:
            mask = torch.zeros(self.mask_size).view(1, -1).repeat(x.size(0), 1).to(x.device).bool()
        x = self.forward_features(x, mask)
        x = self.head(x)
        return x



    
                    
                    
        
                 
        
            

