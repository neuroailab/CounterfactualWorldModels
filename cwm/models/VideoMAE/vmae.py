import os
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from functools import partial

from timm.models.registry import register_model
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from timm.data.constants import (IMAGENET_DEFAULT_MEAN,
                                 IMAGENET_DEFAULT_STD)

from .utils import (Block,
                    _cfg,
                    PatchEmbed,
                    get_sinusoid_encoding_table
                    )

def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)

def _imagenet_unnormalize(x):
    return generator.imagenet_unnormalize(x, temporal_dim=2)

class PretrainVisionTransformerEncoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=(16, 16), in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, tubelet_size=2,
                 use_learnable_pos_emb=False, num_frames=16, embed_per_frame=False, spacetime_separable_pos_embed=False, block_func=Block, block_kwargs={}):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_size = (tubelet_size,) + patch_size
        self.pt, self.ph, self.pw = self.patch_size

        self._embed_per_frame = embed_per_frame
        if not self._embed_per_frame:
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,tubelet_size=tubelet_size,num_frames=num_frames)
            num_patches = self.patch_embed.num_patches
        elif self._embed_per_frame:
            assert (num_frames % tubelet_size) == 0
            num_embeddings = (num_frames // tubelet_size)
            self.patch_embed = nn.ModuleList([
                PatchEmbed(
                    img_size=img_size, patch_size=patch_size,
                    in_chans=in_chans, embed_dim=embed_dim,
                    tubelet_size=tubelet_size, num_frames=tubelet_size)
                for _ in range(num_embeddings)])
            num_patches = self.patch_embed[0].num_patches * num_embeddings

        self.image_size = img_size
        self.num_patches = num_patches
        self.num_frames = num_frames
        print("NUM PATCHES IN ENCODER", self.num_patches)

        # TODO: Add the cls token
        if num_patches is None:
            self.pos_embed = None
        elif use_learnable_pos_emb:
            self._learnable_pos_embed = True
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        else:
            # sine-cosine positional embeddings
            self._learnable_pos_embed = False
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            block_func(
                dim=embed_dim, in_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, **block_kwargs)
            for i in range(depth)])
        self.norm =  norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        self.apply(self._init_weights)

    def _set_pos_embed(self, dim=None):
        if dim is None:
            dim = self.embed_dim
        if self.pos_embed is None:
            self.pos_embed = get_sinusoid_encoding_table(
                self.num_patches, dim)


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

    def _get_pos_embed(self):
        return self.pos_embed

    def forward_block(self, x, idx):
        return self.blocks[idx](x)

    def tokenize(self, x, mask=None):

        if not self._embed_per_frame:
            x = self.patch_embed(x)
        elif self._embed_per_frame:
            x = torch.cat([
                self.patch_embed[i](
                    x[:,:,(i*self.pt):((i+1)*self.pt)])
                for i in range(len(self.patch_embed))], 1)
            
        pos_embed = self._get_pos_embed().type_as(x).to(x.device).clone()
        if not self._learnable_pos_embed:
            pos_embed = pos_embed.detach()
        x = x + pos_embed
        return (x, mask)

    def tokenize_and_mask(self, x, mask):

        x, mask = self.tokenize(x, mask)
        B, _, C = x.shape
        x_vis = x[~mask].reshape(B, -1, C)
        return x_vis

    def forward_features(self, x, mask):
        _, _, T, _, _ = x.shape
        if not self._embed_per_frame:
            x = self.patch_embed(x)
        elif self._embed_per_frame:
            x = torch.cat([
                self.patch_embed[i](
                    x[:,:,(i*self.pt):((i+1)*self.pt)])
                for i in range(len(self.patch_embed))], 1)

        pos_embed = self._get_pos_embed().type_as(x).to(x.device).clone()
        if not self._learnable_pos_embed:
            pos_embed = pos_embed.detach()
        x = x + pos_embed
        B, _, C = x.shape
        x_vis = x[~mask].reshape(B, -1, C) # ~mask means visible

        for blk in self.blocks:
            x_vis = blk(x_vis)

        x_vis = self.norm(x_vis)
        return x_vis

    def _set_inputs(self, *args, **kwargs):
        pass

    def forward(self, x, mask, *args, **kwargs):
        self._set_inputs(x, mask, *args, **kwargs)
        x = self.forward_features(x, mask)
        x = self.head(x)
        return x

class PretrainVisionTransformerDecoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, patch_size=(16, 16), num_classes=768, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, num_patches=196, tubelet_size=2, block_func=Block, block_kwargs={}
                 ):
        super().__init__()


        self.num_classes = num_classes

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_size = patch_size

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            block_func(
                dim=embed_dim, in_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, **block_kwargs)
            for i in range(depth)])
        self.norm =  norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

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

    def forward_block(self, x, idx):
        return self.blocks[idx](x)

    def get_last_tokens(self, x, return_token_num):
        if return_token_num > 0:
            return self.head(self.norm(x[:,-return_token_num:]))
        elif return_token_num == 0:
            return self.head(self.norm(x))[:,x.size(1):]
        else:
            return self.head(self.norm(x))

    def forward(self, x, return_token_num):
        for blk in self.blocks:
            x = blk(x)

        if return_token_num > 0:
            x = self.head(self.norm(x[:, -return_token_num:])) # only return the mask tokens predict pixels
        else:
            x = self.head(self.norm(x))

        return x

class PretrainVisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    default_input_kwargs = {'unnormalize': True}
    def __init__(self,
                 img_size=224, 
                 patch_size=(16, 16),
                 main_input=None,
                 main_input_kwargs=default_input_kwargs,
                 encoder_func=PretrainVisionTransformerEncoder,
                 encoder_in_chans=3, 
                 encoder_num_classes=0, 
                 encoder_embed_dim=768, 
                 encoder_depth=12,
                 encoder_num_heads=12,
                 encoder_block_func=Block,
                 encoder_block_kwargs={},
                 decoder_num_classes=None, # For pretraining this parameter isn't relevant but must be set according to tube&patch size
                 decoder_embed_dim=512, 
                 decoder_depth=8,
                 decoder_num_heads=8,
                 decoder_block_func=Block,
                 decoder_block_kwargs={},
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=None,
                 num_frames=16,
                 drop_rate=0., 
                 attn_drop_rate=0.,
                 drop_path_rate=0., 
                 norm_layer=nn.LayerNorm, 
                 init_values=0.,
                 use_learnable_pos_emb=False,
                 spacetime_separable_pos_embed=False,
                 tubelet_size=2,
                 num_classes=0, # avoid the error from create_fn in timm
                 in_chans=0, # avoid the error from create_fn in timm
                 embed_per_frame=False,
                 flow_model_ckpt=None,
                 flow_frames=None,
                 random_input=False,
                 use_flash_attention=False,
                 **kwargs
                 ):
        super().__init__()
        if main_input is not None:
            self.get_main_input = self._build_stream_input(main_input, **main_input_kwargs)
            assert num_frames == self.get_main_input.get_num_frames(), (num_frames,
                                                                        self.get_main_input.get_num_frames())
            assert encoder_in_chans == self.get_main_input.num_channels
        else:
            self.get_main_input = None

        encoder_block_kwargs.update({'flash_attention': use_flash_attention})
        decoder_block_kwargs.update({'flash_attention': use_flash_attention})

        self.encoder = encoder_func(
            img_size=img_size, 
            patch_size=patch_size, 
            in_chans=encoder_in_chans, 
            num_classes=encoder_num_classes, 
            embed_dim=encoder_embed_dim, 
            depth=encoder_depth,
            num_heads=encoder_num_heads, 
            mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, 
            norm_layer=norm_layer, 
            init_values=init_values,
            tubelet_size=tubelet_size,
            use_learnable_pos_emb=use_learnable_pos_emb,
            spacetime_separable_pos_embed=spacetime_separable_pos_embed,
            num_frames=num_frames,
            embed_per_frame=embed_per_frame,
            block_func=encoder_block_func,
            block_kwargs=encoder_block_kwargs,
            **kwargs)

        if decoder_depth > 0:
            self.decoder = PretrainVisionTransformerDecoder(
                patch_size=patch_size,
                num_patches=self.encoder.num_patches,
                num_classes= 3*tubelet_size*(patch_size[0]*patch_size[1]) if decoder_num_classes is None else decoder_num_classes,
                embed_dim=decoder_embed_dim,
                depth=decoder_depth,
                num_heads=decoder_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rate,
                norm_layer=norm_layer,
                init_values=init_values,
                tubelet_size=tubelet_size,
                block_func=decoder_block_func,
                block_kwargs=decoder_block_kwargs)

            self.encoder_to_decoder = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=False)
        else:
            self.decoder = None
            self.encoder_to_decoder = None

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self._learnable_pos_embed = False
        self._spacetime_separable_pos_embed = spacetime_separable_pos_embed
        self.timestamps = None
        self.encoder.timestamps = None
        self.pos_embed = get_sinusoid_encoding_table(self.encoder.num_patches, decoder_embed_dim)

        if self._spacetime_separable_pos_embed:
            self.pos_embed_encoder = nn.Linear(2*decoder_embed_dim, decoder_embed_dim)

        trunc_normal_(self.mask_token, std=.02)

        self.num_frames = num_frames
        self.num_patches = self.encoder.num_patches
        if self.num_frames is not None:
            self.num_patches_per_frame = self.num_patches // self.num_frames
        else:
            self.num_patches_per_frame = self.num_patches
        self.patch_size = self.encoder.patch_size
        if isinstance(img_size, int):
            self.image_size = (img_size, img_size)
        else:
            assert hasattr(img_size, '__len__'), img_size
            self.image_size = img_size
            
    @property
    def mask_size(self):
        return (self.num_frames // self.patch_size[0],
                self.image_size[-2] // self.patch_size[-2],
                self.image_size[-1] // self.patch_size[-1])

    def _build_stream_input(self, func, temporal_dim=2, **kwargs):

        if isinstance(func, str):
            try:
                stream_input = preproc.get_preprocessor(func, temporal_dim=temporal_dim, **kwargs)
            except KeyError:
                stream_input = getattr(preproc, func, None)(temporald_dim=temporal_dim, **kwargs)
        elif isinstance(func, (partial, nn.Module)):
            stream_input = func(temporal_dim=temporal_dim, **kwargs)
        else:
            raise ValueError("%s is not a valid stream input function or name" % func)
            
        return stream_input            

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
        return {'pos_embed', 'cls_token', 'mask_token'}

    def _get_spacetime_separable_pos_embed(self, dim=None):
        if dim is None:
            dim = self.embed_dim

        pos_embed = []
        B, T = self.timestamps.shape
        N = self.num_patches_per_frame
        dev = self.timestamps.device
        for b in range(B):
            positions = list(range(self.num_patches_per_frame))
            pos_embed_s = transformer.pos_embedding(positions, dim, self.device).squeeze(0) # [N,dim]
            pos_embed_t = transformer.pos_embedding(self.timestamps[b], dim, self.device).squeeze(0) # [T,dim]
            pos_embed_b = torch.cat([
                pos_embed_s[None].repeat(T,1,1).view(self.num_patches, dim).to(dev),
                pos_embed_t[:,None].repeat(1,N,1).view(self.num_patches,dim).to(dev)
            ], -1)
            pos_embed.append(pos_embed_b)

        pos_embed = torch.stack(pos_embed, 0)
        return self.pos_embed_encoder(pos_embed)    

    def _set_pos_embed(self, dim=None):
        if dim is None:
            dim = self.embed_dim
        if self.timestamps is None:
            positions = self.num_patches
            self.pos_embed = transformer.pos_embedding(
                positions, dim, self.device)
        elif self._spacetime_separable_pos_embed:
            self.pos_embed = self._get_spacetime_separable_pos_embed(dim)
            # print("spactime separable decoder", self.pos_embed.shape)            
        else:
            pos_embeds = []
            for b in range(self.timestamps.size(0)):
                positions = []                
                for t in range(self.timestamps.size(1)):
                    offset = self.num_patches_per_frame*self.timestamps[b,t].item()
                    positions.extend([
                        p + offset for p in range(self.num_patches_per_frame)])
                    
                pos_embeds.append(transformer.pos_embedding(
                    positions, dim, self.device))
            self.pos_embed = torch.cat(pos_embeds, 0)

    def get_input(self, x, mask, timestamps=None, *args, **kwargs):
        B,_,T = x.shape[:3]
        if self.get_main_input is None:
            return (x, mask)
        
        if timestamps is None:
            timestamps = torch.arange(T)[None].expand(B,-1).float().to(x.device)
        else:
            assert list(timestamps.shape) == [B,T], timestamps.shape


        
        ## preprocess
        x = self.get_main_input(x, timestamps=timestamps, *args, **kwargs)
        assert (mask.size(-1) % T) == 0, mask.shape
        mask = mask.view(B,T,mask.size(-1)//T)
        mask = self.get_main_input.get_output_frames(mask, temporal_dim=1).reshape(B,-1)
        return (x, mask)

    def get_masked_targets(self,
                           x,
                           mask,
                           patch_size=None,
                           preproc_func=None,
                           postproc_func=None,
                           apply_mask=True):
        if preproc_func is not None:
            x = preproc_func(x)
        if patch_size is None:
            patch_size = self.patch_size

        x = rearrange(x, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c',
                      p0=patch_size[0],
                      p1=patch_size[1],
                      p2=patch_size[2])
        if postproc_func is not None:
            x = postproc_func(x)


        targets = x[mask] if apply_mask else x[torch.ones_like(mask)]
        if targets.size(1) > 0:
            targets = targets.reshape(mask.size(0),
                                      -1,
                                      np.prod(list(x.shape[2:])))
        else:
            targets = targets.reshape(mask.size(0),
                                      0,
                                      np.prod(list(x.shape[2:])))

        return targets

    def get_train_target(self,
                         target,
                         model_inputs,
                         temporal_dim=2,
                         apply_mask=True):
        if len(target.shape) == 4:
            target = target.unsqueeze(temporal_dim)
        inp_t = self.get_input(**model_inputs)
        if target.size(temporal_dim) != inp_t[0].size(temporal_dim):
            if target.size(temporal_dim) == 1:
                target = torch.cat([target]*inp_t[0].size(temporal_dim),
                                   temporal_dim)
            else:
                raise ValueError("target has shape %s" % target.shape)

        train_target = self.get_masked_targets(
            x=target,
            mask=inp_t[1],
            patch_size=None,
            preproc_func=None,
            postproc_func=None,
            apply_mask=apply_mask)
        return train_target

    def forward(self, x, mask, timestamps=None, *args, **kwargs):
        _, _, T, _, _ = x.shape
        self.device = x.device
        x, mask = self.get_input(x, mask, timestamps=timestamps, *args, **kwargs)

        x_vis = self.encoder(x, mask, timestamps=timestamps, *args, **kwargs) # [B, N_vis, C_e]
        if self.decoder is None:
            return x_vis
        x_vis = self.encoder_to_decoder(x_vis) # [B, N_vis, C_d]
        B, N, C = x_vis.shape
        # we don't unshuffle the correct visible token order, 
        # but shuffle the pos embedding accorddingly.
        if self.encoder.timestamps is not None:
            self.timestamps = self.encoder.timestamps
            self._set_pos_embed(dim=self.decoder.embed_dim)
        expand_pos_embed = self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        pos_emd_vis = expand_pos_embed[~mask].reshape(B, -1, C)
        pos_emd_mask = expand_pos_embed[mask].reshape(B, -1, C)
        x_full = torch.cat([x_vis + pos_emd_vis, self.mask_token + pos_emd_mask], dim=1) # [B, N, C_d]
        x = self.decoder(x_full, pos_emd_mask.shape[1]) # [B, N_mask, 3 * 16 * 16]

        return x

#### DEFAULT LARGE AND BASE MODELS
def pretrain_videomae_large_224_scaffold(**kwargs):
    model = PretrainVisionTransformer(
        img_size=224,
        encoder_embed_dim=1024, 
        encoder_depth=24, 
        encoder_num_heads=16,
        encoder_num_classes=0,
        decoder_embed_dim=512,
        decoder_num_heads=8,
        decoder_depth=12,        
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    model.default_cfg = _cfg()
    return model

def pretrain_videomae_base_224_scaffold(**kwargs):
    model = PretrainVisionTransformer(
        img_size=224,
        encoder_embed_dim=768, 
        encoder_depth=12, 
        encoder_num_heads=12,
        encoder_num_classes=0,
        decoder_embed_dim=384,
        decoder_num_heads=6,
        decoder_depth=4,        
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    model.default_cfg = _cfg()
    return model

def base_16x16patch_2frames_1tube(**kwargs):
    model = pretrain_videomae_base_224_scaffold(
        patch_size=(16, 16),
        num_frames=2,
        tubelet_size=1,
        **kwargs)
    return model

def base_8x8patch_2frames_1tube(**kwargs):
    model = pretrain_videomae_base_224_scaffold(
        patch_size=(8, 8),
        num_frames=2,
        tubelet_size=1,
        **kwargs)
    return model

def large_4x4patch_2frames_1tube(**kwargs):
    model = pretrain_videomae_large_224_scaffold(
        patch_size=(4, 4),
        num_frames=2,
        tubelet_size=1,
        **kwargs)
    return model

