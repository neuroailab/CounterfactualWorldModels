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

from cwm.models.patches import Patchify
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

    def reset_classifier(self, num_classes):
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

        # patchifier for computing labels
        self.patchifier = Patchify(
            self.patch_size,
            temporal_dim=2,
            squeeze_channel_dim=False
        )

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

class ChannelMae(nn.Module):
    """A ChannelMaeEncoder, ChannelMaeDecoder, and channel heads"""
    def __init__(
            self,
            encoder_params: Dict = {},
            decoder_params: Dict = {},
            head_params: Optional[Dict] = None,
            preprocessor: Optional[Callable] = None,
            use_flash_attention: bool = False
    ) -> None:

        super().__init__()

        # preprocessing to the input
        self.preprocess = preprocessor() if (preprocessor is not None) else nn.Identity()

        # encoder and decoder
        self.encoder_params = copy.deepcopy(encoder_params)
        self.decoder_params = copy.deepcopy(decoder_params)
        self.head_params = copy.deepcopy(head_params) if head_params is not None else None

        enc_block_kwargs = self.encoder_params.get('block_kwargs', {})
        dec_block_kwargs = self.decoder_params.get('block_kwargs', {})
        enc_block_kwargs.update({'flash_attention': use_flash_attention})
        dec_block_kwargs.update({'flash_attention': use_flash_attention})        

        if self.head_params is not None:
            head_block_kwargs = self.head_params.get('block_kwargs', {})
            head_block_kwargs.update({'flash_attention': use_flash_attention})


        self.encoder = self._build_encoder(params=self.encoder_params)
        if self.decoder_params.get('depth', 4) > 0:
            self.decoder = self._build_decoder(params=self.decoder_params)
            self.encoder_to_decoder = nn.Linear(self.encoder.embed_dim, self.decoder.embed_dim, bias=False)
        else:
            self.decoder, self.encoder_to_decoder = None

        # channel heads (decoders for each channel group)
        self.channel_heads = self._build_channel_heads(params=self.head_params)

        # mask token and positional embedding
        self._init_mask_token()
        self.pos_embed = get_sinusoid_encoding_table(self.encoder.num_patches, self.decoder.embed_dim)

        # init weights
        self.apply(self._init_weights)

    def _init_mask_token(self) -> None:
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.decoder.embed_dim))
        trunc_normal_(self.mask_token, std=0.02)        

    def _build_encoder(self, params: Dict = {}) -> nn.Module:
        return ChannelMaeEncoder(**params)

    def _build_decoder(self, params: Dict = {}) -> nn.Module:
        return ChannelMaeDecoder(**params)

    def _build_channel_heads(self, params: Optional[Dict] = None) -> nn.ModuleList:
        """
        Different channels are decoded with different heads.
        If no params are passed, these heads are just linear layers.
        """
        num_classes_per_head = [
            math.prod(self.patch_size) * in_chans for in_chans in self.encoder.channel_partition
        ]

        embed_dim = self.decoder.embed_dim if self.decoder is not None else self.encoder.embed_dim

        # linear layers
        if params is None:
            return nn.ModuleList(
                [
                    nn.Linear(
                        in_features=embed_dim,
                        out_features=num_classes,
                        bias=True
                        )
                    for num_classes in num_classes_per_head
                ]
            )

        # else channel heads are ChannelMaeDecoders
        channel_heads = nn.ModuleList(
            [
                nn.Sequential(
                    # linear decoder to head
                    nn.Linear(
                        in_features=embed_dim,
                        out_features=params.get('embed_dim', embed_dim),
                        bias=False
                    ),
                    # transformer
                    self._build_decoder(params)
                )
                for _ in range(self.num_channel_groups)
            ]
        )
        for group_idx, num_classes in enumerate(num_classes_per_head):
            channel_heads[group_idx][1].reset_classifier(num_classes=num_classes)

        return channel_heads

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, x: torch.Tensor, squeeze_channel_dim: bool = False):

        x = self.encoder.patchifier(x)
        if squeeze_channel_dim:
            x = x.view(*x.shape[:2], self.patch_dim * x.shape[-1])
        return x

    def _apply_channel_heads(
            self,
            x: torch.Tensor,
            mask: torch.Tensor,
            pos_embed: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        Apply the channel heads to tokens after the decoder stage.
        Only return the masked tokens.
        """

        B, _, C = pos_embed.shape

        # figure out which tokens belong to which channel group
        ps = torch.split(pos_embed, self.token_channel_group_splits, dim=1)
        ms = torch.split(mask, self.token_channel_group_splits, dim=1)
        vis_ns = [
            ps[group_idx][~ms[group_idx]].reshape(B, -1, C).shape[1]
            for group_idx in range(self.num_channel_groups)
        ]

        masked_ns = [
            ps[group_idx][ms[group_idx]].reshape(B, -1, C).shape[1]
            for group_idx in range(self.num_channel_groups)
        ]

        # separate the visible from the masked tokens
        x_vis, x_masked = torch.split(x, (sum(vis_ns), sum(masked_ns)), dim=1)
        xs_vis = torch.split(x_vis, vis_ns, dim=1)
        xs_masked = torch.split(x_masked, masked_ns, dim=1)

        # if heads are just linear layers
        if isinstance(self.channel_heads[0], nn.Linear):
            return [
                self.channel_heads[group_idx](xs_masked[group_idx])
                for group_idx in range(self.num_channel_groups)
            ]

        # if heads are transformer Decoders
        return [
            self.channel_heads[group_idx][1](
                self.channel_heads[group_idx][0](
                    torch.cat([xs_vis[group_idx], xs_masked[group_idx]], dim=1)
                ),
                return_token_num=masked_ns[group_idx]
            )            
            for group_idx in range(self.num_channel_groups)
        ]

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        """Pass in an image tensor, mask it, and get the predicted masked patches for each channel group"""

        x = self.preprocess(x)

        # encoder runs patch embed, then the encoder blocks on unmasked tokens
        x_vis = self.encoder(x, mask)
        if self.decoder is None:
            return x_vis

        # embed in decoder dimension
        x_vis = self.encoder_to_decoder(x_vis)
        B, _, C = x_vis.shape

        # apply pos embed to decoder inputs
        decoder_pos_embed = self.pos_embed.type_as(x).to(x.device).clone().detach()

        # get and apply the positional embedding for the decoder via the mask
        mask = mask.unsqueeze(dim=-1).repeat(1, 1, C)
        pos_embed_vis = decoder_pos_embed[~mask].reshape(B, -1, C)
        pos_embed_mask = decoder_pos_embed[mask].reshape(B, -1, C)

        # create full decoder input, with mask tokens
        x_vis = x_vis + pos_embed_vis
        mask_token_w_pos_emb = self.mask_token + pos_embed_mask
        x_full = torch.cat([x_vis, mask_token_w_pos_emb], dim=1)

        # run the decoder, keeping all the tokens
        x = self.decoder(x_full, return_token_num=-1)

        # split the decoder output into tokens per channel, then apply output heads
        ys = self._apply_channel_heads(x, mask, decoder_pos_embed)

        return ys

    def compute_labels(
            self,
            x: torch.Tensor,
            mask: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        Get the masked patches from each channel group of the input image x.

        Outputs a List[torch.Tensor] of shape [batch_size, num_masked_patches, patch_dim] for each group.
        """
        # convert input image to patches
        x = self.patchify(x, squeeze_channel_dim=False)

        mask_list = torch.split(mask, self.token_channel_group_splits, dim=1)
        channel_inds = self.channel_group_start_inds

        labels_list = []
        for idx, group_mask in enumerate(mask_list):
            group_dim = self.channel_partition[idx]
            group_labels = x[...,channel_inds[idx]:channel_inds[idx+1]]
            group_labels = group_labels[group_mask].reshape(
                x.size(0), -1, self.patch_dim * group_dim
            )
            labels_list.append(group_labels)

        return labels_list

    def compute_train_loss(
            self,
            x: torch.Tensor,
            mask: torch.Tensor,
            loss_fn: Optional[Callable] = None
    ) -> torch.Tensor:
        """
        Get the predictions and labels from each channel group, and apply loss_fn. Defaults to MSE
        """
        group_preds = self.forward(x, mask)
        with torch.no_grad():
            group_labels = self.compute_labels(x, mask)

        loss = 0.0
        if loss_fn is None:
            loss_fn = nn.MSELoss()

        # skip groups that have no masked tokens
        for idx, pred in enumerate(group_preds):
            loss += loss_fn(pred, group_labels[idx]) if pred.size(1) > 0 else 0.0

        return loss

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'mask_token'}

    @property
    def patch_size(self):
        return self.encoder.patch_size

    @property
    def patch_dim(self):
        return math.prod(self.patch_size)

    @property
    def mask_size(self):
        return self.encoder.mask_size

    @property
    def image_size(self):
        return self.encoder.image_size

    @property
    def num_channel_groups(self):
        return self.encoder.num_channel_groups

    @property
    def num_tokens_per_channel_group(self):
        return int(self.num_patches // self.num_channel_groups)

    @property
    def token_channel_group_splits(self):
        return (self.num_tokens_per_channel_group, ) * self.num_channel_groups

    @property
    def channel_partition(self):
        return self.encoder.channel_partition

    @property
    def channel_group_start_inds(self):
        return [0] + [
            i.item() for i in torch.cumsum(torch.tensor(self.channel_partition), dim=0)
        ]

    @property
    def num_channels(self):
        return sum(self.channel_partition)

    @property
    def num_patches(self):
        return self.encoder.num_patches

    @property
    def num_tokens(self):
        return self.encoder.num_patches

    @property
    def encoder_depth(self):
        return self.encoder.get_num_layers()

    @property
    def decoder_depth(self):
        return self.encoder.get_num_layers()


class SoftChannelMaeEncoder(ChannelMaeEncoder):
    """
    Encoder that pads the hard mask with a `soft_mask` up to a determined number of tokens.
    """
    def __init__(self, min_num_padding_tokens: int = 0, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.min_num_padding_tokens = min_num_padding_tokens
        
    def forward(
            self,
            x: torch.Tensor,
            mask: torch.Tensor,
            padding_token: torch.Tensor,
            soft_mask: Optional[torch.Tensor] = None,
            min_num_padding_tokens: Optional[int] = None
    ) -> torch.Tensor:

        if min_num_padding_tokens is None:
            min_num_padding_tokens = self.min_num_padding_tokens

        return

class SoftChannelMae(ChannelMae):
    """
    Variant of ChannelMae that can train and inference on variable number of visible tokens.

    Besides the usual boolean mask that determines which patches / tokens are visible,
    also accepts two new new args, `soft_mask` and `num_decode_tokens`.

    The former is a soft mask that can fully reveal (value 0.0) or fully mask (value 1.0)
    input tokens.
    
    The latter determines how many of the tokens interpolated with `soft_mask` will be
    concatenated onto the visible inputs and therefore decoded to predicted values. During training,
    this number may be << the number of masked tokens to promote computational efficiency.

    Args:
        - x : input image tensor [B, C, H, W] <torch.float>
        - mask : hard masking tensor [B, num_tokens] <torch.bool>
        - soft_mask : soft masking tensor [B, num_tokens] <torch.float>
        - num_decode_tokens : int or None that determines how many tokens are concatenated to x_vis
    """
    

class SoftInputChannelMae(SoftChannelMae):
    """
    Like SoftChannelMae, except the tokens that are predicted are an entirely new set of
    `mask_tokens` (analogous to those in the hard ChannelMae). The soft_masked tokens are
    not directly decoded; they merely provide a differentiable route for providing variable
    numbers of visible tokens to the model.
    """
    pass


   
    


        

        

    
                    
                    
        
                 
        
            

