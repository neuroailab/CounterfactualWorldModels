from functools import partial
import copy, os, sys, time
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from einops import rearrange
from torch import einsum

import cwm.models.patches as patches
import cwm.models.masking as masking
from cwm.models.raft.raft_model import load_raft_model
from cwm.models.utils import (imagenet_normalize,
                              imagenet_unnormalize)

class Preprocessor(nn.Module):
    """Base Class for doing something to a transformer input prior to tokenization"""
    num_channels = None
    def __init__(self,
                 frames_list=None,
                 temporal_dim=2,
                 channel_dim=None,
                 preproc_func=None,
                 preproc_kwargs={},
                 num_frames=None,
                 num_channels=None,
                 stack=False,
                 *args,
                 **kwargs):
        super().__init__()
        self.set_frames_list(frames_list)
        self.temporal_dim = temporal_dim
        if channel_dim is None:
            self.channel_dim = 1 if (self.t_dim == 2) else 2
        else:
            self.channel_dim = channel_dim
            assert self.c_dim != self.t_dim, (self.c_dim, self.t_dim)
        self.set_preproc_func(preproc_func, **preproc_kwargs)
        self.num_frames = num_frames
        if num_channels is not None:
            self.num_channels = num_channels
        self.stack = stack

    @property
    def t_dim(self):
        return self.temporal_dim
    @property
    def c_dim(self):
        return self.channel_dim

    def set_frames_list(self, frames_list):
        if frames_list is None:
            frames_list = None
        elif isinstance(frames_list, int):
            frames_list = [frames_list, frames_list + 1]
        elif not isinstance(frames_list, list):
            frames_list = list(frames_list)

        self.frames_list = frames_list
        if self.frames_list is not None:
            self.num_input_frames = len(frames_list)

    def set_preproc_func(self, func, **kwargs):
        self._preproc_kwargs = copy.deepcopy(kwargs)
        if func is None:
            self.preproc_func = lambda x: x
        else:
            self.preproc_func = lambda x: func(x, **self._preproc_kwargs)

    def get_input_frames(self, x):
        frame_tensor = torch.tensor(self.frames_list).long().to(x.device)
        return torch.index_select(x, dim=self.temporal_dim, index=frame_tensor)

    def get_num_frames(self):
        if self.stack:
            return 1
        if self.num_frames is None:
            if self.frames_list is not None:
                return len(self.frames_list)
            return None
        return self.num_frames

    def get_num_channels(self, x):

        if self.c_dim not in range(len(x.shape)):
            return 0
        return x.shape[self.c_dim]

    def _preproc(self, x, *args, **kwargs):
        try:
            return self.preproc_func(x, *args, **kwargs)
        except:
            return self.preproc_func(x)

    def set_input_dims(self, x):
        self.num_input_channels = self.get_num_channels(x)
        self.T = x.shape[self.t_dim]
        if self.frames_list is None:
            self.frames_list = list(range(self.T))
            self.num_input_frames = self.T
        self.frames_list = [fr % self.T for fr in self.frames_list]

    def set_output_dims(self, x):

        if self.num_channels is None:
            self.num_channels = self.get_num_channels(x)
        else:
            assert self.num_channels == self.get_num_channels(x), (self.num_channels, self.get_num_channels(x))

        if self.num_frames is None:
            self.num_frames = x.shape[self.t_dim]
        else:
            assert self.num_frames == x.shape[self.t_dim]

    def get_output_frames(self, y, temporal_dim=None):
        """Get the output frames from a tensor y"""
        frame_tensor = torch.tensor(self.frames_list[-self.num_frames:]).long().to(y.device)
        if temporal_dim is None:
            temporal_dim = self.t_dim
        return torch.index_select(y, dim=temporal_dim, index=frame_tensor)

    def forward(self, x, *args, **kwargs):
        self.set_input_dims(x)
        x = self.get_input_frames(x)
        x = self._preproc(x, *args, **kwargs)

        if self.stack:
            x = x.view(x.shape[0], x.shape[1] * x.shape[2], 1 , *x.shape[-2:])
            self.num_frames = 1
            self.num_input_frames = 1
            self.num_channels = x.shape[1]

        self.set_output_dims(x)
        return x

class Noise(Preprocessor):
    def __init__(self,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.set_preproc_func(self.noise)

    def noise(self, x):
        return torch.rand_like(x)

class ImagenetNormalize(Preprocessor):
    num_channels = 3
    def __init__(self,
                 unnormalize=False,
                 *args,
                 **kwargs):

        super().__init__(*args, **kwargs)
        func = imagenet_normalize if not unnormalize else imagenet_unnormalize
        self.set_preproc_func(func, temporal_dim=self.t_dim)

class FirstAndTargetRGB(Preprocessor):
    num_channels = 3
    def __init__(self,
                 *args,
                 **kwargs):
        super().__init__(frames_list=[0,-1],
                         *args,
                         **kwargs)


class IMU(Preprocessor):
    num_frames = None
    num_channels = 6
    def __init__(self,
                 sequence_length=None,
                 frames_list=None,
                 temporal_dim=2,
                 channel_dim=None,
                 preproc_func=None,
                 preproc_kwargs={},
                 *args,
                 **kwargs):
        super().__init__(frames_list=frames_list,
                         temporal_dim=temporal_dim,
                         channel_dim=channel_dim,
                         preproc_func=preproc_func,
                         preproc_kwargs=preproc_kwargs)
        self.num_frames = None
        self.sequence_length = sequence_length

    def set_output_dims(self, x):
        super().set_output_dims(x)
        self.num_frames = None
        sequence_length = x.shape[self.t_dim]
        if self.sequence_length is not None:
            assert self.sequence_length == sequence_length, (self.sequence_length, sequence_length)

    def get_sequence_length(self):
        return self.sequence_length

    def forward(self, imu=None, timestamps=None, *args, **kwargs):
        if imu is None:
            return None
        imu = imu.unsqueeze(-1).unsqueeze(-1) # now [B,(L,D),1,1]
        self.set_input_dims(imu)
        imu = self._preproc(imu, *args, **kwargs)
        self.set_output_dims(imu)        
        return imu

class FramePairFlow(Preprocessor):
    num_channels = 2
    default_raft_ckpt = '../checkpoints/raft_checkpoints/raft-large.pth'
    def __init__(self,
                 iters=24,
                 backward=False,
                 unnormalize_rgb=True,
                 normalize_flow=True,
                 concat_backward=False,
                 concat_rgb=False,
                 flow_model_ckpt=default_raft_ckpt,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.load_flow_model(flow_model_ckpt)
        self.unnorm_rgb = (lambda x: imagenet_unnormalize(x, self.t_dim)) if unnormalize_rgb else (lambda x:x)
        self.norm_flow = self._normalize_flow if normalize_flow else lambda x:x
        self._concat_backward = concat_backward
        self._concat_rgb = concat_rgb
        func_list = [lambda x: self.get_flow(x, iters=iters, backward=backward)]
        if self._concat_backward:
            func_list.append(lambda x: self.get_flow(x, iters=iters, backward=(not backward)))
            self.num_channels += 2
        if self._concat_rgb:
            norm = (lambda x: imagenet_normalize(x, self.t_dim)) if unnormalize_rgb else (lambda x:x)
            func_list.append(lambda x: torch.index_select(
                norm(x), dim=self.t_dim, index=torch.tensor(self.frames_list[1:]).long().to(x.device)))
            self.num_channels += 3

        transforms_list = [self.unnorm_rgb,
                           lambda x: torch.cat([f(x) for f in func_list], self.c_dim),
                           self.norm_flow]

        get_flow_func = transforms.Compose(transforms_list)

        self.set_preproc_func(get_flow_func)

        if self.frames_list is not None:
            self.num_frames = self.num_input_frames - 1

    def get_num_frames(self):
        if self.num_frames is None:
            if self.frames_list is not None:
                return len(self.frames_list) - 1
            return None
        return self.num_frames            

    def load_flow_model(self, ckpt):
        self.flow_model = load_raft_model(ckpt).eval().requires_grad_(False)

    def _normalize_flow(self, flow):
        size = flow.shape[-2:]
        size = torch.as_tensor([size[1], size[0]]).to(flow)
        size = size.view(1,2,1,1,1)
        if self._concat_backward:
            size = torch.cat([size, size], 1)
        if self._concat_rgb:
            size = torch.cat([size, 2 * torch.ones((1,3,1,1,1)).to(flow)], 1)
        if self.c_dim == 2:
            size = size.transpose(self.t_dim, self.c_dim)
        flow = flow / (size / 2.0)
        return flow

    def get_flow(self, x, **kwargs):
        if self.t_dim == 2 and self.c_dim == 1:
            x = x.transpose(self.t_dim, self.c_dim)
            flow = self.flow_model(x, **kwargs).transpose(self.t_dim, self.c_dim)
        else:
            flow = self.flow_model(x, **kwargs)
        return flow

    def set_raft_kwargs(self, iters=12, backward=False):
        func = transforms.Compose([
            self.unnorm_rgb,
            lambda x: self.get_flow(x, iters=iters, backward=backward),
            self.normalize_flow
        ])
        self.set_preproc_func(func)
    

"""Some Preprocessors for use in ConjoinedVMAE"""
RGB02 = partial(
    Preprocessor,
    num_channels=3,
    frames_list=[0,-1])
RGB01 = partial(
    Preprocessor,
    num_channels=3,
    frames_list=[0,1]
)
RGB01Stack = partial(
    Preprocessor,
    num_channels=6,
    frames_list=[0,1],
    stack=True
)
RGB12 = partial(
    Preprocessor,
    num_channels=3,
    frames_list=[1,-1])
RGB012 = partial(
    Preprocessor,
    num_channels=3,
    frames_list=[0,1,-1])
RGB0 = partial(
    Preprocessor,
    num_channels=3,
    frames_list=[0])
RGB1 = partial(
    Preprocessor,
    num_channels=3,
    frames_list=[1])
Noise1 = partial(
    Noise,
    num_channels=3,
    frames_list=[1])
Flow01 = partial(
    FramePairFlow,
    frames_list=[0,1],
    unnormalize_rgb=True,
    normalize_flow=True)
FlowRGB01 = partial(
    FramePairFlow,
    frames_list=[0,1],
    unnormalize_rgb=True,
    normalize_flow=True,
    concat_rgb=True)
Flow = partial(
    FramePairFlow,
    unnormalize_rgb=True,
    normalize_flow=True)
FlowRGB = partial(
    FramePairFlow,
    unnormalize_rgb=True,
    normalize_flow=True,
    concat_rgb=True)
FlowBackRGB = partial(
    FramePairFlow,
    unnormalize_rgb=True,
    normalize_flow=True,
    concat_backward=True,
    concat_rgb=True)
FlowBack01 = partial(
    FramePairFlow,
    frames_list=[0,1],
    unnormalize_rgb=True,
    normalize_flow=True,
    concat_backward=True)
FlowBackRGB01 = partial(
    FramePairFlow,
    frames_list=[0,1],
    unnormalize_rgb=True,
    normalize_flow=True,
    concat_backward=True,
    concat_rgb=True)

def get_preprocessor(name, temporal_dim=2, unnormalize=True, **kwargs):
    if unnormalize and ('imu' not in name):
        kwargs['preproc_func'] = ImagenetNormalize(unnormalize=True,
                                                   temporal_dim=temporal_dim)
    func = {
        'rgb01': RGB01,
        'rgb02': RGB02,
        'rgb0': RGB0,
        'rgb1': RGB1,
        'noise1': Noise1,
        'flow01': Flow01,
        'flow_rgb01': FlowRGB01,
        'flow': Flow,
        'flow_rgb': FlowRGB,
        'flowback_rgb': FlowBackRGB,        
        'flowback01': FlowBack01,
        'flowback_rgb01': FlowBackRGB01,
        'rgb12': RGB12,
        'rgb012': RGB012,
        'imu': IMU,
        'rgb01stack': RGB01Stack,
    }[name]

    return func(temporal_dim=temporal_dim, **kwargs)
                 
