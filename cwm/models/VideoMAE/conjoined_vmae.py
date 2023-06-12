import os
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from functools import partial

import cwm.models.preprocessor as preproc
from cwm.models.transformer import (CrossAttentionTransformerBlock,
                                    pos_embedding)

from cwm.models.VideoMAE.vmae import (PretrainVisionTransformerEncoder,
                                      PretrainVisionTransformerDecoder,
                                      PretrainVisionTransformer,
                                      trunc_normal_,
                                      _imagenet_unnormalize,
                                      _cfg)
from cwm.models.utils import num_parameters

class PaddedVisionTransformer(PretrainVisionTransformer):
    """Allow batches of inputs with a mixed number of visible patches by padding encoder tokens"""
    PRINT_PADDING = False
    def __init__(self,
                 min_padding_tokens=0,
                 max_padding_tokens=16, # must be >= gap between most and fewest visible patches in batch
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        ## pad visible mask by up to at most this many learnable null tokens
        self.min_padding_tokens = min_padding_tokens
        self.max_padding_tokens = max_padding_tokens
        self.null_token_enc = nn.Parameter(torch.zeros(1, 1, self.encoder.embed_dim), requires_grad=True)
        self.null_token_dec = nn.Parameter(torch.zeros(1, 1, self.decoder.embed_dim), requires_grad=True)
        
        trunc_normal_(self.null_token_enc, std=0.02)
        trunc_normal_(self.null_token_dec, std=0.02)

        self._reset_padding_mask()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'mask_token', 'null_token_enc', 'null_token_dec'}

    def _set_padding_mask(self, mask, device=None):
        """Set the padding mask that determines number of padding tokens to concat onto inputs"""

        if device is not None:
            self.device = device
        else:
            device = self.device

        with torch.no_grad():
            if self.PRINT_PADDING:
                print("mask", mask.dtype, mask.shape, torch.sum((~(mask.bool())).int(), -1), mask.device)
            self._num_visible = torch.sum((~mask).flatten(1).int(), -1, keepdim=True)
            _min_num_vis = torch.min(self._num_visible)
            _max_num_vis = torch.max(self._num_visible)                

            num_padding_per_ex = _max_num_vis - self._num_visible + self.min_padding_tokens
            padding_mask = torch.arange(
                self.max_padding_tokens,
                device="cpu")[None].expand(mask.size(0), -1) < num_padding_per_ex.to("cpu")

            null_padding_mask = torch.cat([
                torch.ones((mask.size(0), 1), dtype=torch.bool, device="cpu"),
                torch.zeros((mask.size(0), self.max_padding_tokens - 1), dtype=torch.bool, device="cpu")
            ], -1)

            any_visible = torch.sum(self._num_visible.float(), dim=(0,1), keepdim=True) > 0
            any_visible = any_visible.expand(padding_mask.size(0), padding_mask.size(-1))
            if self.PRINT_PADDING:
                print("num visible", self._num_visible.shape, self._num_visible[...,0])
                print("num padding per ex", num_padding_per_ex.shape, num_padding_per_ex[...,0])
                print("any visible", any_visible.shape, torch.sum(any_visible, -1))
                print("padding mask before cond", torch.sum(padding_mask, -1))
            padding_mask = torch.where(
                any_visible.to(padding_mask.device), padding_mask, null_padding_mask)

            _min_num_vis = torch.maximum(_min_num_vis, torch.ones_like(_min_num_vis))
            _max_num_vis = torch.maximum(_max_num_vis, torch.ones_like(_max_num_vis))
            
            padding_mask = ~padding_mask.clone().detach()
            min_masked = mask.size(1) - _max_num_vis - self.min_padding_tokens            

            if self.PRINT_PADDING:
                print("visible padding", padding_mask.shape, torch.sum(~padding_mask, -1))
            self.padding_mask = padding_mask.to(device, non_blocking=True)
            self._num_visible = self._num_visible.to(device, non_blocking=True)
            self._min_num_vis = _min_num_vis.to(device, non_blocking=True)
            self._max_num_vis = _max_num_vis.to(device, non_blocking=True)

            ## create the full input mask
            if mask.device == padding_mask.device:
                full_input_mask = torch.cat([mask, padding_mask], -1).contiguous().detach()
                if self.PRINT_PADDING:
                    print("create full input mask on device, with visible",
                          mask.device, full_input_mask.shape,
                          torch.sum(~full_input_mask, -1))
                self.full_input_mask = full_input_mask.to(device, non_blocking=True)
                null_mask = torch.cat([torch.zeros_like(mask[:,:min_masked]), padding_mask], -1)
                self.null_mask = null_mask.contiguous().detach().to(device, non_blocking=True)

            else:
                if self.PRINT_PADDING:
                    print("create full input mask on cpu", padding_mask.device)
                self.full_input_mask = torch.cat([mask, padding_mask.to(device)], -1).clone().detach()
                self.null_mask = torch.cat([torch.zeros_like(mask[:,:min_masked]),
                                            padding_mask.to(device)], -1).clone().detach()

            if self.PRINT_PADDING:
                print("null mask", self.null_mask.shape, torch.sum(self.null_mask, -1))

    def _reset_padding_mask(self):
        self.padding_mask = None
        self.full_input_mask = None
        self.null_mask = None
        self._num_visible = None
        self._min_num_vis = self._max_num_vis = None    
        
    def pad_and_mask_input(self, x, mask):

        assert self.padding_mask is not None
        mask_padded = self.full_input_mask
        x, mask = self.encoder.tokenize(x, mask)
        x_padded = torch.cat([
            x, self.null_token_enc.expand(x.size(0), self.max_padding_tokens, -1).to(x.device)], 1)
        B, _, C = x.shape
        x_vis = x_padded[~mask_padded].reshape(B, -1, C)
        return x_vis

    def _encode(self, x, mask, *args, **kwargs):

        self.encoder._set_inputs(x, mask, *args, **kwargs)
        self._set_padding_mask(mask)
        mask_padded = self.full_input_mask
        x, mask = self.encoder.tokenize(x, mask)
        x_padded = torch.cat([
            x, self.null_token_enc.expand(x.size(0), self.max_padding_tokens, -1).to(x.device)], 1)
        B, _, C = x.shape
        x_vis = x_padded[~mask_padded].reshape(B, -1, C)
        
        for blk in self.encoder.blocks:
            x_vis = blk(x_vis)

        x_vis = self.encoder.norm(x_vis)
        x_vis = self.encoder.head(x_vis)
        return x_vis

    def _pad_pos_embed(self, x, mask):
        B, _, C = x.shape
        expand_pos_embed = self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        expand_pos_embed = torch.cat([
            expand_pos_embed,
            self.null_token_dec.expand(B, self.max_padding_tokens, -1).to(x.device)
        ], 1)
        mask_padded = self.full_input_mask

        pos_emd_vis = expand_pos_embed[~mask_padded].reshape(B, -1, C)
        pos_emd_mask = expand_pos_embed[mask_padded].reshape(B, -1, C)
        return pos_emd_vis, pos_emd_mask, mask_padded    
    
    def get_masked_targets(self,
                           x,
                           mask,
                           patch_size=None,
                           preproc_func=_imagenet_unnormalize,
                           postproc_func=None):
        if preproc_func is not None:
            x = preproc_func(x)
        if patch_size is None:
            patch_size = self.patch_size
        x = rearrange(x, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c',
                      p0=patch_size[0], p1=patch_size[1], p2=patch_size[2])
        if postproc_func is not None:
            x = postproc_func(x)

        x_padded = torch.cat([
            x,
            torch.zeros(x.size(0), self.max_padding_tokens, x.size(2), x.size(3)).to(x.dtype).to(x.device)
        ], 1)
        mask_padded = self.full_input_mask
        return x_padded[mask_padded].reshape(mask.size(0), -1, np.prod(list(x.shape[2:])))

    def forward(self, x, mask, *args, reset_padding_mask=True, **kwargs):
        if reset_padding_mask:
            self._reset_padding_mask()
        self.device = x.device
        x, mask = self.get_input(x, mask, *args, **kwargs)

        ## split this function into tokenizing, padding, masking, encoding, removing padding
        x_vis = self._encode(x, mask, *args, **kwargs)
        x_vis = self.encoder_to_decoder(x_vis) # [B, N_vis, C_d]

        B, N, C = x_vis.shape
        
        ## pad the positional embeddng and mask accordingly
        pos_emd_vis, pos_emd_mask, mask_padded = self._pad_pos_embed(x_vis, mask)

        x_full = torch.cat([x_vis + pos_emd_vis, self.mask_token + pos_emd_mask], dim=1) # [B, N, C_d]
        x = self.decoder(x_full, pos_emd_mask.shape[1]) # [B, N_mask, 3 * 16 * 16]

        ## replace outputs at padding tokens with zeros so they don't contribute to loss
        x = x * ((~self.null_mask)[...,None].to(x))

        return x

class ConjoinedPretrainVisionTransformer(nn.Module):
    """Transformer that splits (or creates) inputs in parallel streams, which are conjoined"""
    debug_mode = False
    default_cross_block_kwargs = {
        'num_heads': 4,
        'mlp_ratio': 2.0,
        'shared_similarity': False,
        'with_self_attention': False,
    }
    default_model_kwargs = {
        'encoder_func': PretrainVisionTransformerEncoder,
        'tubelet_size': 1,
    }
    default_input_kwargs = {'unnormalize': True}
    def __init__(self,
                 img_size=224,
                 patch_size=(8,8),
                 context_img_size=None,
                 context_patch_size=(8,8),
                 num_frames=None,
                 main_input='rgb02',
                 main_input_kwargs=default_input_kwargs,
                 context_input='flow01',
                 context_input_kwargs=default_input_kwargs,
                 main_model_func=PretrainVisionTransformer,
                 main_model_kwargs=default_model_kwargs,
                 context_model_func=PretrainVisionTransformer,
                 context_model_kwargs=default_model_kwargs,                 
                 conjoin_encoder_layers=[(0, 0), (-1, -1)],
                 conjoin_decoder_layers=[(0, 0)],
                 conjoin_func=CrossAttentionTransformerBlock,                                  
                 encoder_cross_block_kwargs=default_cross_block_kwargs,
                 decoder_cross_block_kwargs=default_cross_block_kwargs,
                 output_main=True,
                 output_context=False,
                 context_mask_func=None,
                 context_mask_kwargs={},
                 decode_main=True,                 
                 decode_context=True,
                 use_flash_attention=False,
                 *args,
                 **kwargs):
        super().__init__()



        ## build the input configs
        self.get_main_input = self._build_stream_input(main_input, **main_input_kwargs)
        self.get_context_input = self._build_stream_input(context_input, **context_input_kwargs)

        ## figure out input shapes for each transformer stream
        num_frames_main = self.get_main_input.get_num_frames()
        num_frames_context = self.get_context_input.get_num_frames()
        self.num_frames = num_frames or ((num_frames_main or 0)) # + (num_frames_context or 0))

        ## build the main streams
        main_kwargs = copy.deepcopy(kwargs)
        main_kwargs.update(main_model_kwargs)
        main_kwargs['use_flash_attention'] = use_flash_attention

        main_kwargs['encoder_in_chans'] = self.get_main_input.num_channels or 3
        if main_kwargs.get('decoder_num_classes', None) is None:
            num_out_chans = main_kwargs.get('tubelet_size', 1) * np.prod(patch_size)
            main_kwargs['decoder_num_classes'] = main_kwargs['encoder_in_chans'] * num_out_chans
            
        context_kwargs = copy.deepcopy(kwargs)
        context_kwargs.update(context_model_kwargs)
        context_kwargs['use_flash_attention'] = use_flash_attention

        context_kwargs['encoder_in_chans'] = self.get_context_input.num_channels or 3
        if context_kwargs.get('decoder_num_classes', None) is None:
            num_out_chans = context_kwargs.get('tubelet_size', 1) * np.prod((context_patch_size or patch_size))        
            context_kwargs['decoder_num_classes'] = context_kwargs['encoder_in_chans'] * num_out_chans

        self.main_stream = main_model_func(img_size=img_size,
                                           patch_size=patch_size,
                                           num_frames=num_frames_main,
                                           **main_kwargs)
        context_img_size = (context_img_size or img_size)

        self.context_stream = context_model_func(img_size=context_img_size,
                                                 patch_size=context_patch_size,
                                                 num_frames=num_frames_context,
                                                 **context_kwargs)

        print("%d main frames: %s\n%d context frames: %s\n%d total frames" %\
            (self.main_stream.num_frames or 0, self.get_main_input.frames_list,
             self.context_stream.num_frames or 0, self.get_context_input.frames_list,
             self.num_frames))

        ## how to hook up the two streams
        self._conjoin_func = conjoin_func

        encoder_cross_block_kwargs['flash_attention'] = False
        decoder_cross_block_kwargs['flash_attention'] = False
        
        self._encoder_cross_block_kwargs = copy.deepcopy(encoder_cross_block_kwargs)
        self._decoder_cross_block_kwargs = copy.deepcopy(decoder_cross_block_kwargs)
        if (conjoin_encoder_layers == True):
            conjoin_encoder_layers = list(range(min(self.main_stream.encoder.get_num_layers(),
                                                    self.context_stream.encoder.get_num_layers())))
        elif conjoin_encoder_layers in [False, None]:
            conjoin_encoder_layers = []
        if (conjoin_decoder_layers == True):
            conjoin_decoder_layers = list(range(min(self.main_stream.decoder.get_num_layers(),
                                                    self.context_stream.decoder.get_num_layers())))
        elif conjoin_decoder_layers in [False, None]:
            conjoin_decoder_layers = []
        self._build_conjoining_attention_blocks(conjoin_encoder_layers,
                                                conjoin_decoder_layers)
        self._set_decoder_outputs(output_main, output_context)

        print("Parameter Breakdown:\n" +\
              "Main stream: %d\nContext stream: %d\nConjoining: %d\nTotal: %d" %\
              (num_parameters(self.main_stream), num_parameters(self.context_stream),
               num_parameters(self) - num_parameters(self.main_stream) - num_parameters(self.context_stream),
               num_parameters(self)))

        if context_mask_func is not None:
            context_video_size = (self.context_stream.num_frames,) + self.context_stream.image_size[-2:]
            self.context_mask_generator = context_mask_func(
                input_size=[c // self.context_stream.patch_size[i]
                            for i,c in enumerate(context_video_size)],
                always_batch=True,
                **context_mask_kwargs)
        else:
            self.context_mask_generator = None

        self._decode_main = decode_main
        self._decode_context = decode_context

        self._context_input = context_input
        if self._context_input in ['campose',]:
            self.context_mask_generator = lambda x: torch.zeros([x.shape[0], self.num_frames], dtype=torch.bool)
        
    def __getattr__(self, key):
        try:
            return super(ConjoinedPretrainVisionTransformer, self).__getattr__(key)
        except:
            attr = getattr(self.main_stream, key, None)
            if attr is None:
                raise AttributeError("no attr %s in the module or the main transformer stream" % key)
            return attr

    @property
    def mask_size(self):
        return (self.num_frames // self.main_stream.patch_size[0],
                self.main_stream.image_size[-2] // self.main_stream.patch_size[-2],
                self.main_stream.image_size[-1] // self.main_stream.patch_size[-1])

    def _build_stream_input(self, func, temporal_dim=2, **kwargs):

        if isinstance(func, str):
            try:
                stream_input = preproc.get_preprocessor(func, temporal_dim=temporal_dim, **kwargs)
            except KeyError:
                stream_input = getattr(preproc, func, None)(temporal_dim=temporal_dim, **kwargs)
        elif isinstance(func, (partial, nn.Module)):
            stream_input = func(temporal_dim=temporal_dim, **kwargs)
        else:
            raise ValueError("%s is not a valid stream input function or name" % func)

        return stream_input

    def _build_conjoining_block(self, layer_pair, encoder=True):
        main_idx, context_idx = layer_pair
        main_block = getattr(self.main_stream, 'encoder' if encoder else 'decoder').blocks[main_idx]
        context_block = getattr(self.context_stream, 'encoder' if encoder else 'decoder').blocks[context_idx]
        kwargs = self._encoder_cross_block_kwargs if encoder else self._decoder_cross_block_kwargs

        def _in_out_dim(block):
            try:
                in_dim = block.attn.qkv.in_features
                out_dim = block.mlp.fc2.out_features
            except:
                in_dim, out_dim = block.in_dim, block.out_dim
            return (in_dim, out_dim)

        in_dim, out_dim = _in_out_dim(main_block)
        in_dim_src, out_dim_src = _in_out_dim(context_block)

        cross_block = self._conjoin_func(
            in_dim=in_dim,
            out_dim=out_dim,
            in_dim_src=in_dim_src,
            out_dim_src=out_dim_src,
            **kwargs)

        return cross_block

    def _build_conjoining_attention_blocks(self, enc_layers, dec_layers):
        num_layers_me = self.main_stream.encoder.get_num_layers()
        num_layers_md = self.main_stream.decoder.get_num_layers()
        num_layers_ce = self.context_stream.encoder.get_num_layers()
        num_layers_cd = self.context_stream.decoder.get_num_layers()

        enc_keys = []
        for pair in enc_layers:
            if not hasattr(pair, '__len__'):
                pair = (pair, pair)
            enc_keys.append((pair[0] % num_layers_me, pair[1] % num_layers_ce))

        dec_keys = []
        for pair in dec_layers:
            if not hasattr(pair, '__len__'):
                pair = (pair, pair)
            dec_keys.append((pair[0] % num_layers_md, pair[1] % num_layers_cd))

        self.encoder_conjoining_blocks = nn.ModuleDict([
            ("{}-{}".format(*key), self._build_conjoining_block(key, encoder=True))
            for key in enc_keys
        ])

        self.decoder_conjoining_blocks = nn.ModuleDict([
            ("{}-{}".format(*key), self._build_conjoining_block(key, encoder=False))
            for key in dec_keys
        ])

    def get_stream_inputs(self,
                          x,
                          mask,
                          timestamps=None,
                          x_context=None,
                          mask_context=None):
        
        B,_,T = x.shape[:3]
        if timestamps is None:
            timestamps = torch.arange(self.num_frames)[None].expand(B,-1).float().to(x.device)
        else:
            assert list(timestamps.shape) == [B,self.num_frames], (timestamps.shape, [B,self.num_frames])

        ## get the (not necessarily visual) inputs for the two streams
        x_m, x_c = (self.get_main_input(x, timestamps=timestamps),
                    self.get_context_input(x_context if x_context is not None else x,
                                           timestamps=timestamps))

        if self.debug_mode:
            print("main stream inp", x_m.shape, x_m.amin().item(), x_m.amax().item())
            print("context stream inp", x_c.shape, x_c.amin().item(), x_c.amax().item())

        ## figure out which frames to take for the masks and timestamps
        ## rule is to get the frames_list and the num_frames from each input
        ## and take frames_list[-num_frames:]
        if self.get_main_input.num_frames:
            ts_m = self.get_main_input.get_output_frames(timestamps, temporal_dim=1)
        else:
            ts_m = timestamps
        if self.get_context_input.num_frames:
            ts_c = self.get_context_input.get_output_frames(timestamps, temporal_dim=1)
        else:
            ts_c = timestamps

        ## for the masks, need to first reshape it to expose temporal dim
        assert (mask.size(-1) % T) == 0, mask.shape
        mask = mask.view(B,T,mask.size(-1)//T)
        mask_m = self.get_main_input.get_output_frames(mask, temporal_dim=1).reshape(B,-1)

        if mask_context is None and self.context_mask_generator is None:
            mask_c = self.get_context_input.get_output_frames(mask, temporal_dim=1).reshape(B,-1)
        elif mask_context is None:
            mask_c = self.context_mask_generator(x_c).to(x_c.device)
        else:
            if self.get_context_input.num_frames in [0,None]:
                ## timeframes for two inputs are not shared
                mask_c = mask_context
            else:
                ## get the specified output frames
                mask_c = mask_context.view(B,T,mask_context.size(-1)//T)
                mask_c = self.get_context_input.get_output_frames(mask_c, temporal_dim=1).reshape(B,-1)
                
        return (
            (x_m, mask_m, ts_m),
            (x_c, mask_c, ts_c)
        )


    def conjoined_encode(self, main_args, context_args):
        x, mask, ts = main_args
        x_c, mask_c, ts_c = context_args
        self.main_stream.device = x.device
        self.context_stream.device = x_c.device

        ## set the inputs for each
        self.main_stream.encoder._set_inputs(x, mask, ts)
        self.context_stream.encoder._set_inputs(x_c, mask_c, ts_c)

        ## tokenize each stream
        if self.debug_mode:
            print("pre tokenize", x.shape, x_c.shape, mask.shape, mask_c.shape)
        x = self.main_stream.encoder.tokenize_and_mask(x, mask)
        x_c = self.context_stream.encoder.tokenize_and_mask(x_c, mask_c)

        if self.debug_mode:        
            print("post tokenize", x.shape, x_c.shape)

        ## now run encoder blocks up to a point where there's crosstalk between streams
        ## then evaluate the cross-attention block
        i,j = 0,0
        for pair in self.encoder_conjoining_blocks.keys():
            pi, pj = (int(p) for p in pair.split('-'))

            ## cross-attention comes before self-attention
            while i < pi:
                x = self.main_stream.encoder.forward_block(x, i)
                if self.debug_mode:                        
                    print("main %d" % i)
                i += 1
            while j < pj:
                x_c = self.context_stream.encoder.forward_block(x_c, j)
                if self.debug_mode:
                    print("ctx %d" % j)
                j += 1

            x, x_c = self.encoder_conjoining_blocks[pair](x, x_c)
            if self.debug_mode:            
                print("did cross attention %s" % pair)

        ## finish streams
        for _i in range(i, self.main_stream.encoder.get_num_layers()):
            x = self.main_stream.encoder.forward_block(x, _i)
            if self.debug_mode:
                print("main %d" % _i)
        for _j in range(j, self.context_stream.encoder.get_num_layers()):
            x_c = self.context_stream.encoder.forward_block(x_c, _j)
            if self.debug_mode:
                print("ctx %d" % _j)

        x = self.main_stream.encoder.norm(x)
        x_c = self.context_stream.encoder.norm(x_c)
        return (x, x_c)

    def forward_encoder_blocks(self, x, x_c):
        i,j = 0,0
        for pair in self.encoder_conjoining_blocks.keys():
            pi, pj = (int(p) for p in pair.split('-'))

            ## cross-attention comes before self-attention
            while i < pi:
                x = self.main_stream.encoder.forward_block(x, i)
                if self.debug_mode:                        
                    print("main %d" % i)
                i += 1
            while j < pj:
                x_c = self.context_stream.encoder.forward_block(x_c, j)
                if self.debug_mode:
                    print("ctx %d" % j)
                j += 1

            x, x_c = self.encoder_conjoining_blocks[pair](x, x_c)
            if self.debug_mode:            
                print("did cross attention %s" % pair)

        ## finish streams
        for _i in range(i, self.main_stream.encoder.get_num_layers()):
            x = self.main_stream.encoder.forward_block(x, _i)
            if self.debug_mode:
                print("main %d" % _i)
        for _j in range(j, self.context_stream.encoder.get_num_layers()):
            x_c = self.context_stream.encoder.forward_block(x_c, _j)
            if self.debug_mode:
                print("ctx %d" % _j)

        x = self.main_stream.encoder.norm(x)
        x_c = self.context_stream.encoder.norm(x_c)
        return (x, x_c)

    def _set_decoder_inputs(self, x, x_c):
        self.B,self.N,self.C = x.shape
        self.B,self.M,self.D = x_c.shape

        if self.main_stream.encoder.timestamps is not None:
            self.timestamps = self.main_stream.encoder.timestamps
            self.main_stream._set_pos_embed(self.main_stream.decoder.embed_dim)
        if self.context_stream.encoder.timestamps is not None:
            self.timestamps_context = self.context_stream.encoder.timestamps
            self.context_stream._set_pos_embed(self.context_stream.decoder.embed_dim)

    def _set_decoder_outputs(self, output_main=None, output_context=None):
        if output_main is not None:
            self._output_main = output_main
        if output_context is not None:
            self._output_context = output_context

    def _concat_excess_tokens(self, mask):
        if getattr(self.context_stream.encoder, '_concat_dummy_token', False):
            mask = torch.cat([
                mask,
                torch.zeros((mask.size(0), 1)).to(mask.device).bool()
            ], -1)
            if self.context_stream.pos_embed.size(1) < mask.size(-1):
                self.context_stream.pos_embed = torch.cat([
                    self.context_stream.pos_embed,
                    pos_embedding([self.context_stream.encoder.num_tokens],
                                              self.context_stream.pos_embed.size(-1),
                                              self.device)
                ], -2)

        return mask
            
    def conjoined_decode(self, main_args, context_args):

        x, mask = main_args
        x_c, mask_c = context_args

        ## check if extra visible tokens were added by the encoder (e.g. in case of fully blank imu)
        N_vis_c = x_c.size(-2)
        mask_c = self._concat_excess_tokens(mask_c)

        ## get pos embeddings
        _detach = lambda p: p.expand(self.B,-1,-1).type_as(x).to(x.device).clone().detach()
        _separate = lambda p,m,dim: (p[~m].reshape(self.B,-1,dim), p[m].reshape(self.B,-1,dim))
        pos_vis, pos_mask = _separate(_detach(self.main_stream.pos_embed), mask, self.C)
        pos_vis_c, pos_mask_c = _separate(_detach(self.context_stream.pos_embed), mask_c, self.D)

        ## add mask tokens with positional information
        if self._decode_main:
            x = torch.cat([x + pos_vis, self.main_stream.mask_token + pos_mask], 1)
        else:
            x = x + pos_vis
            
        if self._decode_context:
            x_c = torch.cat([x_c + pos_vis_c, self.context_stream.mask_token + pos_mask_c], 1)
        else: ## just keep the visible tokens
            x_c = x_c + pos_vis_c

        if self.debug_mode:
            print("DECODER")
            print("positional embeddings", pos_vis.shape, pos_mask.shape, pos_vis_c.shape, pos_mask_c.shape)
            print("full embeddings", x.shape, x_c.shape)

        ## now run decoder blocks up to a point where there's crosstalk between streams
        ## then evaluate the cross-attention block
        i,j = 0,0
        for pair in self.decoder_conjoining_blocks.keys():
            pi, pj = (int(p) for p in pair.split('-'))

            ## cross-attention comes after self-attention
            while i <= pi:
                x = self.main_stream.decoder.forward_block(x, i)
                if self.debug_mode:                
                    print("main %d" % i)
                i += 1
            while j <= pj:
                x_c = self.context_stream.decoder.forward_block(x_c, j)
                if self.debug_mode:                
                    print("ctx %d" % j)
                j += 1

            x, x_c = self.decoder_conjoining_blocks[pair](x, x_c)
            if self.debug_mode:
                print("did cross attention %s" % pair)

        for _i in range(i, self.main_stream.decoder.get_num_layers()):
            x = self.main_stream.decoder.forward_block(x, _i)
            if self.debug_mode:            
                print("main %d" % _i)
        for _j in range(j, self.context_stream.decoder.get_num_layers()):
            x_c = self.context_stream.decoder.forward_block(x_c, _j)
            if self.debug_mode:            
                print("ctx %d" % _j)

        ## now get the masked tokens from each stream
        if self._output_main and self._decode_main:
            x = self.main_stream.decoder.get_last_tokens(x, pos_mask.size(1))
        if self._output_context and self._decode_context:
            x_c = self.context_stream.decoder.get_last_tokens(x_c, pos_mask_c.size(1))

        if self._output_main and self._output_context:
            return (x, x_c)
        elif self._output_main:
            return x
        elif self._output_context:
            return x_c
        else:
            return (x, x_c) # return all the tokens from both streams

    def forward_decoder_blocks(self, x, x_c):
        ## now run decoder blocks up to a point where there's crosstalk between streams
        ## then evaluate the cross-attention block
        i,j = 0,0
        for pair in self.decoder_conjoining_blocks.keys():
            pi, pj = (int(p) for p in pair.split('-'))

            ## cross-attention comes after self-attention
            while i <= pi:
                x = self.main_stream.decoder.forward_block(x, i)
                if self.debug_mode:                
                    print("main %d" % i)
                i += 1
            while j <= pj:
                x_c = self.context_stream.decoder.forward_block(x_c, j)
                if self.debug_mode:                
                    print("ctx %d" % j)
                j += 1

            x, x_c = self.decoder_conjoining_blocks[pair](x, x_c)
            if self.debug_mode:
                print("did cross attention %s" % pair)

        for _i in range(i, self.main_stream.decoder.get_num_layers()):
            x = self.main_stream.decoder.forward_block(x, _i)
            if self.debug_mode:            
                print("main %d" % _i)
        for _j in range(j, self.context_stream.decoder.get_num_layers()):
            x_c = self.context_stream.decoder.forward_block(x_c, _j)
            if self.debug_mode:            
                print("ctx %d" % _j)

        return (x, x_c)                
        
    def get_current_inputs(self, x, mask, *args, **kwargs):

        inp_m, inp_c = self.get_stream_inputs(x, mask, *args, **kwargs)
        if self._output_main and self._output_context:
            return (inp_m, inp_c)
        elif self._output_main:
            return (inp_m,)
        elif self._output_context:
            return (inp_c,)
        else:
            return (inp_m, inp_c)

    def pad_and_mask_target(self, x, mask, output_main=True):

        ## get the padding mask
        if output_main:
            full_mask = getattr(self.main_stream, 'full_input_mask', None)
        else:
            full_mask = getattr(self.context_stream, 'full_input_mask', None)

        if full_mask is None:
            return x[mask]

        num_pad = full_mask.size(1) - mask.size(1)
        shape = x.shape
        x_padded = torch.cat([
            x,
            torch.zeros(shape[0], num_pad, shape[2], shape[3]).to(x.dtype).to(x.device)
        ], 1)
        
        if PRINT_PADDING:
            print("x_padded", x_padded.shape, "full mask", full_mask.shape, "full mask sum", torch.sum(full_mask, -1))
        
        return x_padded[full_mask]

    def get_masked_targets(self,
                           x,
                           mask,
                           patch_size=None,
                           preproc_func=_imagenet_unnormalize,
                           postproc_func=None,
                           output_main=True):

        if preproc_func is not None:
            x = preproc_func(x)
        if patch_size is None:
            patch_size = self.patch_size # from main stream
        x = rearrange(x, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c',
                      p0=patch_size[0],
                      p1=patch_size[1],
                      p2=patch_size[2])
        if postproc_func is not None:
            x = postproc_func(x)

        ## output is [B,num_masked_patches,p0*p1*p2*c]
        targets = self.pad_and_mask_target(x, mask, output_main=output_main)

        if PRINT_PADDING:
            print("x", x.shape, "targets", targets.shape, "mask", mask.shape, mask.sum(-1))
        
        if targets.size(1) > 0:
            targets = targets.reshape(mask.size(0), -1, np.prod(list(x.shape[2:])))
        else:
            targets = targets.reshape(mask.size(0), 0, np.prod(list(x.shape[2:])))

        if PRINT_PADDING:
            print("targets reshaped", targets.shape, torch.sum(targets.abs().sum((-2,-1)) > 0))
        return targets

    def get_train_target(self, target, model_inputs, temporal_dim=2, output_main=None):
        if output_main is None:
            output_main = self._output_main
        if len(target.shape) == 4:
            target = target.unsqueeze(temporal_dim)
        inp_t = self.get_current_inputs(**model_inputs)[0]
        train_target = self.get_masked_targets(
            x=target,
            mask=inp_t[1],
            preproc_func=None,
            postproc_func=None,
            output_main=output_main
        )
        return train_target

    def get_masked_input(self, x, mask,
                         output_main=True,
                         output_context=False,
                         preproc_func_main=None,
                         postproc_func_main=None,
                         preproc_func_context=None,
                         postproc_func_context=None,
                         *args, **kwargs):
        self._output_main = self._output_context = True
        inp_m, inp_c = self.get_current_inputs(x, mask, *args, **kwargs)
        out_m = out_c = None
        if output_main:
            out_m = self.get_masked_targets(
                inp_m[0],
                inp_m[1],
                preproc_func=preproc_func_main,
                postproc_func=postproc_func_main)
        if output_context:
            out_c = self.get_masked_targets(
                inp_c[0],
                inp_c[1],
                preproc_func=preproc_func_context,
                postproc_func=postproc_func_context)

        if (out_m is not None) and (out_c is not None):
            return (out_m, out_c)
        elif out_m is not None:
            return out_m
        elif out_c is not None:
            return out_c
        else:
            return None
        

    def get_masked_imu(self,
                       imu,
                       mask,
                       preproc_func=None,
                       postproc_func=None):
        return self.get_masked_targets(imu[...,None,None],
                                       mask,
                                       patch_size=self.context_stream.patch_size,
                                       preproc_func=preproc_func,
                                       postproc_func=postproc_func,
                                       output_main=False)
            
    def forward(self,
                x,
                mask,
                timestamps=None,
                x_context=None,
                mask_context=None,
                output_main=None,
                output_context=None,
                *args,
                **kwargs):

        ## split the inputs into two streams
        self.main_stream.device = x.device
        if x_context is not None:
            self.context_stream.device = x_context.device
        else:
            self.context_stream.device = self.main_stream.device
        main_args, context_args = self.get_stream_inputs(
            x, mask, timestamps,
            x_context=(x_context if x_context is not None else x),
            mask_context=mask_context
        )

        x_vis, x_vis_context = self.conjoined_encode(main_args, context_args)
        x_vis, x_vis_context = (self.main_stream.encoder_to_decoder(x_vis),
                                self.context_stream.encoder_to_decoder(x_vis_context))

        ## set params for decoder and decode masked positions
        self._set_decoder_inputs(x_vis, x_vis_context)
        self._set_decoder_outputs(output_main, output_context)        
        _mask, _mask_context = main_args[1], context_args[1]
        x_decoded = self.conjoined_decode(
            main_args=(x_vis, _mask), # pass masks
            context_args=(x_vis_context, _mask_context))
        
        return x_decoded

class ConjoinedPaddedVisionTransformer(ConjoinedPretrainVisionTransformer):

    def __init__(self,
                 main_model_func=PaddedVisionTransformer,
                 context_model_func=PaddedVisionTransformer,
                 *args, **kwargs):
        super().__init__(main_model_func=main_model_func,
                         context_model_func=context_model_func,
                         *args, **kwargs)

    @property
    def _main_padded(self):
        return hasattr(self.main_stream, 'padding_mask')
    @property
    def _context_padded(self):
        return hasattr(self.context_stream, 'padding_mask')

    def _reset_padding_mask(self):
        self.main_stream._reset_padding_mask()
        self.context_stream._reset_padding_mask()

    def _set_padding_mask(self, mask, mask_context):
        self.main_stream.device = mask.device
        self.context_stream.device = mask_context.device        
        if self._main_padded:
            self.main_stream._set_padding_mask(mask)
        if self._context_padded:
            self.context_stream._set_padding_mask(mask_context)

    def conjoined_encode(self, main_args, context_args):
        x, mask, ts = main_args
        x_c, mask_c, ts_c = context_args

        ## set the inputs for each
        self.main_stream.encoder._set_inputs(x, mask, ts)
        self.context_stream.encoder._set_inputs(x_c, mask_c, ts_c)

        ## tokenize each stream with padding
        self.main_stream.device = mask.device
        self.context_stream.device = mask_c.device
        
        if self.debug_mode:
            print("pre tokenize", x.shape, x_c.shape)

        if self._main_padded:
            if self.main_stream.padding_mask is None:
                self.main_stream._set_padding_mask(mask)
            x = self.main_stream.pad_and_mask_input(x, mask)
        else:
            x = self.main_stream.encoder.tokenize_and_mask(x, mask)

        if self._context_padded:
            if self.context_stream.padding_mask is None:
                self.context_stream._set_padding_mask(mask_c)
            x_c = self.context_stream.pad_and_mask_input(x_c, mask_c)
        else:
            x_c = self.context_stream.encoder.tokenize_and_mask(x_c, mask_c)

        if self.debug_mode:        
            print("post tokenize", x.shape, x_c.shape)

        return self.forward_encoder_blocks(x, x_c)

    def conjoined_decode(self, main_args, context_args):
        x, mask = main_args
        x_c, mask_c = context_args

        ## get pos embeddings
        _detach = lambda p: p.expand(self.B,-1,-1).type_as(x).to(x.device).clone().detach()
        _separate = lambda p,m,dim: (p[~m].reshape(self.B,-1,dim), p[m].reshape(self.B,-1,dim))
        if self._main_padded:
            pos_vis, pos_mask, _ = self.main_stream._pad_pos_embed(x, mask)
        else:
            pos_vis, pos_mask = _separate(_detach(self.main_stream.pos_embed), mask, self.C)

        if self._context_padded:
            pos_vis_c, pos_mask_c, _ = self.context_stream._pad_pos_embed(x_c, mask_c)
        else:
            pos_vis_c, pos_mask_c = _separate(_detach(self.context_stream.pos_embed), mask_c, self.D)

        if self._decode_main:
            x = torch.cat([x + pos_vis, self.main_stream.mask_token + pos_mask], 1)
        else:
            x = x + pos_vis
            
        if self._decode_context:
            x_c = torch.cat([x_c + pos_vis_c, self.context_stream.mask_token + pos_mask_c], 1)
        else:
            x_c = x_c + pos_vis_c

        if self.debug_mode:
            print("DECODER")
            print("positional embeddings", pos_vis.shape, pos_mask.shape, pos_vis_c.shape, pos_mask_c.shape)
            print("full embeddings", x.shape, x_c.shape)

        ## do the blocks
        x, x_c = self.forward_decoder_blocks(x, x_c)

        ## now get the masked tokens from each stream
        if self._decode_main:
            x = self.main_stream.decoder.get_last_tokens(x, pos_mask.size(1))
        if self._decode_context:
            x_c = self.context_stream.decoder.get_last_tokens(x_c, pos_mask_c.size(1))

        if self.debug_mode:
            print("masked main output", x.shape)
            print("masked context output", x_c.shape)

        ## replace outputs at padding tokens with zeros so they don't contribute to loss
        if self._main_padded and self._decode_main:
            x = x * ((~self.main_stream.null_mask)[...,None].to(x))
        
        if self._context_padded and self._decode_context:
            x_c = x_c * ((~self.context_stream.null_mask)[...,None].to(x_c))

        if self._output_main and self._output_context:
            return (x, x_c)
        elif self._output_main:
            return x
        elif self._output_context:
            return x_c
        else:
            return (x, x_c) # return all the tokens from both streams

class ImuEncoder(PretrainVisionTransformerEncoder):
    """Encoder for IMU data, which has shape [B,D,L] where D is num channels and L is sequence_length"""
    default_num_imu_channels = 6
    def __init__(self,
                 img_size=None,
                 patch_size=None,
                 sequence_length=200,
                 num_frames=None,
                 tubelet_size=8,
                 in_chans=default_num_imu_channels,
                 use_learnable_pos_emb=False,
                 frame_gap=None,
                 concat_dummy_token=True,
                 use_campose=False,
                 campose_in_chans=16,
                 *args,
                 **kwargs):
        ### treat inputs as having a spatial dimension of (1,1) and a temporal dimension of L
        super(ImuEncoder, self).__init__(img_size=(1,1),
                                         patch_size=(1,1),
                                         tubelet_size=tubelet_size,
                                         use_learnable_pos_emb=False,
                                         in_chans=in_chans if not use_campose else campose_in_chans,
                                         num_frames=sequence_length,
                                         *args,
                                         **kwargs)
        self.in_dim = in_chans if not use_campose else campose_in_chans
        self.sequence_length = sequence_length
        self.num_tokens = self.num_patches
        self.num_frames = 0
        self.frame_gap = frame_gap
        self.timestamps = None
        self._concat_dummy_token = concat_dummy_token

        self._learnable_pos_embed = use_learnable_pos_emb

        if self._learnable_pos_embed:
            assert self.num_tokens is not None
            self.pos_embed = nn.Parameter(torch.zeros(1,
                                                      self.num_tokens + int(concat_dummy_token),
                                                      self.embed_dim), requires_grad=True)
            trunc_normal_(self.pos_embed, std=0.02)
        else:
            self.pos_embed = None

        if self._concat_dummy_token:
            self.dummy_token = nn.Parameter(torch.zeros((1, self.in_dim, self.patch_size[0], 1, 1)),
                                            requires_grad=True)
            trunc_normal_(self.dummy_token, std=0.02)

    @property
    def shape(self):
        return (self.sequence_length, 1, 1)

    def _set_timestamps(self, x, timestamps):
        num_frames = x.size(2)
        if timestamps is None:
            self.timestamps = torch.arange(num_frames).to(x).to(x.device)
        elif isinstance(timestamps, torch.Tensor):
            self.timestamps = timestamps.to(x.device).to(x.dtype)
        else:
            self.timestamps = torch.tensor(timestamps, dtype=x.dtype).to(x.device)
            self.timestamps = self.timestamps - self.timestamps[0]

        if self.frame_gap is not None:
            self.timestamps = self.timestamps / self.frame_gap

    def _set_pos_embed(self, imu=None, dim=None):
        if imu is None:
            pass
        if dim is None:
            dim = self.embed_dim
            
        if self._learnable_pos_embed:
            assert self.pos_embed is not None
            assert self.pos_embed.size(-2) == imu.size(2)
        elif self.pos_embed is None and (self.num_tokens is None):
            L = imu.size(2) // self.patch_size[0]
            self.pos_embed = pos_embedding(L, dim, self.device)
        elif self.pos_embed is None:
            self.pos_embed = pos_embedding(self.num_tokens + int(self._concat_dummy_token),
                                                       dim, self.device)

    def _set_inputs(self, imu, mask, timestamps=None, *args, **kwargs):
        self.device = imu.device
        self._set_timestamps(imu, timestamps=timestamps)
        self._set_pos_embed(imu, dim=None)

    def _get_dummy_token(self, B, device):
        return self.dummy_token.expand(B,-1,-1,-1,-1).to(device)

    def _null_input(self, B, device, sequence_length=0):
        return torch.cat([
            torch.zeros(B, self.in_dim, sequence_length, 1, 1).float().to(device),
            self._get_dummy_token(B, device)
        ], 2)

    def tokenize(self, imu, mask):
        if imu is None and self.num_tokens is None:
            # single token that will be masked
            imu = self._null_input(mask.size(0), mask.device, self.patch_size[0])
            mask = torch.zeros((mask.size(0), 1)).bool().to(mask.device)
        elif imu is None:
            # fully masked input
            imu = self._null_input(mask.size(0), mask.device)
            mask = torch.ones((mask.size(0),
                               self.num_tokens + int(self._concat_dummy_token))).bool().to(mask.device)
            mask[:,-1] = 0
        elif self._concat_dummy_token:
            imu = torch.cat([imu, self._get_dummy_token(imu.size(0), imu.device)], 2)
            mask = torch.cat([mask, torch.zeros_like(mask[:,-1:])], -1)

        return super().tokenize(imu, mask)

    def tokenize_and_mask(self, imu, mask):
        """Last token is always zero and is always unmasked"""
        return super().tokenize_and_mask(imu, mask)

    def forward(self, imu, mask, *args, **kwargs):
        """Last token is always zero and is always unmasked"""
        if imu is None and self.num_tokens is None:
            # single token that will be masked
            imu = self._null_input(mask.size(0), mask.device, self.patch_size[0])
            mask = torch.zeros((mask.size(0), 1)).bool().to(mask.device)
        elif imu is None:
            # fully masked input
            imu = self._null_input(mask.size(0), mask.device, self.sequence_length)
            mask = torch.ones((mask.size(0),
                               self.num_tokens + int(self._concat_dummy_token))).bool().to(mask.device)
            mask[:,-1] = 0
        elif self._concat_dummy_token:
            imu = torch.cat([imu, self._get_dummy_token(imu.size(0), imu.device)], 2)
            mask = torch.cat([mask, torch.zeros_like(mask[:,-1:])], -1)
            
        return super().forward(imu, mask, *args, **kwargs)


"""Define models that take in RGB and Flow to predict IMU, or RGB and IMU to predict RGB"""
def conjoined_full_videomae_base_224_scaffold(**kwargs):
    model = ConjoinedPretrainVisionTransformer(
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

def conjoined_padded_videomae_base_224_scaffold(**kwargs):
    model = ConjoinedPaddedVisionTransformer(
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

rgb_encoder_kwargs = copy.deepcopy(ConjoinedPretrainVisionTransformer.default_model_kwargs)
rgb_encoder_kwargs.update({
    'encoder_func': PretrainVisionTransformerEncoder,
    'decoder_num_classes': None
})

rgb_padded_encoder_kwargs = copy.deepcopy(rgb_encoder_kwargs)
rgb_padded_encoder_kwargs.update({'min_padding_tokens': 0,
                                  'max_padding_tokens': 16})

rgb_4x4_padded_encoder_kwargs = copy.deepcopy(rgb_padded_encoder_kwargs)
rgb_4x4_padded_encoder_kwargs.update({'max_padding_tokens': 64})

imu_encoder_kwargs = copy.deepcopy(ConjoinedPretrainVisionTransformer.default_model_kwargs)
imu_encoder_kwargs.update({
    'encoder_func': ImuEncoder,
    'spacetime_separable_pos_embed': True,
    'encoder_embed_dim': 384,
    'decoder_embed_dim': 192
})

imu400_encoder_kwargs = copy.deepcopy(imu_encoder_kwargs)
imu400_encoder_kwargs.update({
    'sequence_length': 400,
    'tubelet_size': 16,
    'decoder_num_classes': 6 * 16
})

imu400_padded_encoder_kwargs = copy.deepcopy(imu400_encoder_kwargs)
imu400_padded_encoder_kwargs.update({'min_padding_tokens': 0,
                                     'max_padding_tokens': 25,
                                     'concat_dummy_token': False})

def imu400_8x8patch_2frames_1tube_flowbackrgb01(**kwargs):
    model = conjoined_full_videomae_base_224_scaffold(
        num_frames=2,
        main_input='flowback_rgb01',
        context_input='imu',
        main_model_kwargs=rgb_encoder_kwargs,
        context_model_kwargs=imu400_encoder_kwargs,
        conjoin_encoder_layers=[0,-1],
        conjoin_decoder_layers=True,
        **kwargs)
    return model
                                      
def imu400_base_4x4patch_2frames_1tube(**kwargs):
    model = conjoined_padded_videomae_base_224_scaffold(
        patch_size=(4,4),
        main_model_func=PaddedVisionTransformer,
        main_model_kwargs=rgb_4x4_padded_encoder_kwargs,
        main_input='rgb01',
        main_input_kwargs={'unnormalize': False},
        context_model_func=PaddedVisionTransformer,
        context_model_kwargs=imu400_padded_encoder_kwargs,
        context_input='imu',
        conjoin_encoder_layers=range(0,12,3),
        conjoin_decoder_layers=True,
        **kwargs)
    return model

