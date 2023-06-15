"""Modified from github.com/princeton-vl/RAFT"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cwm.models.raft.update import BasicUpdateBlock, SmallUpdateBlock
from cwm.models.raft.extractor import BasicEncoder, SmallEncoder
from cwm.models.raft.corr import CorrBlock, AlternateCorrBlock
from cwm.models.raft.utils import bilinear_sampler, coords_grid, upflow8

import argparse

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass

class Dummy:
    def __init__(self, enabled):
        pass
    def __enter__(self):
        pass
    def __exit__(self, *args):
        pass

def get_args(cmd=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--corr_levels', type=int, default=4)
    parser.add_argument('--corr_radius', type=int, default=4)
    parser.add_argument('--output_dim', type=int, default=None)
    parser.add_argument('--iters', type=int, default=None)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--mixed_precision', action='store_true')
    parser.add_argument('--small', action='store_true')
    parser.add_argument('--gpus', type=int, nargs='+', default=[0])

    if cmd is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(cmd)
    return args

default_raft_ckpt = '../../../checkpoints/raft_checkpoints/raft-large.pth'

def load_raft_model(load_path=default_raft_ckpt,
                    ignore_prefix=None,
                    multiframe=True,
                    scale_inputs=True,
                    output_dim=None,
                    **kwargs):

    if ((load_path is None) or (not os.path.exists(load_path))) and (output_dim is None):
        print("%s is not a valid raft checkpoint" % load_path)
        raise ValueError("You must download RAFT checkpoints with cwm/models/raft/download_raft_checkpoints.sh\n" + \
                         "Checkpoints will be downloaded to CounterfactualWorldModels/checkpoints/raft_checkpoints/")

    args = get_args("")
    for k,v in kwargs.items():
        args.__setattr__(k,v)
    args.multiframe = multiframe
    args.scale_inputs = scale_inputs
    args.output_dim = output_dim

    model = RAFT(args)

    if load_path is not None:
        weight_dict = torch.load(load_path, map_location=torch.device("cpu"))
        new_dict = dict()
        for k in weight_dict.keys():
            if 'module' in k:
                new_dict[k.replace('module.', '')] = weight_dict[k]
            else:
                new_dict[k] = weight_dict[k]

        if ignore_prefix is not None:
            new_dict_1 = dict()
            for k, v in new_dict.items():
                new_dict_1[k.replace(ignore_prefix, '')] = v
            new_dict = new_dict_1

        did_load = model.load_state_dict(new_dict, strict=False)
        print(did_load, type(model).__name__, load_path)
    else:
        print("created a new %s with %d parameters" % (
            type(model).__name__,
            sum([v.numel() for v in model.parameters()])))

    return model

def get_raft_flow(x, raft_model, iters=24, backward=False, t_dim=1):
    assert len(x.shape) == 5, x.shape
    assert x.shape[t_dim] >= 2, x.shape
    x = x * 255.0
    inds = torch.tensor([0,1]).to(x.device)
    x1, x2 = torch.index_select(x, t_dim, inds).unbind(t_dim)
    if backward:
        flow = raft_model(x2, x1, test_mode=True, iters=iters)[-1]
    else:
        flow = raft_model(x1, x2, test_mode=True, iters=iters)[-1]
        
    return flow

class RAFT(nn.Module):
    def __init__(self, args):
        super(RAFT, self).__init__()
        self.args = args
        self.multiframe = self.args.multiframe
        self.scale_inputs = self.args.scale_inputs

        if self.args.iters is not None:
            self.iters = self.args.iters

        if args.small:
            self.hidden_dim = hdim = 96
            self.context_dim = cdim = 64
            args.corr_levels = 4
            args.corr_radius = 3
        
        else:
            self.hidden_dim = hdim = 128
            self.context_dim = cdim = 128
            args.corr_levels = 4
            args.corr_radius = 4

        if 'dropout' not in self.args:
            self.args.dropout = 0

        if 'alternate_corr' not in self.args:
            self.args.alternate_corr = False

        # feature network, context network, and update block
        if args.small:
            self.fnet = SmallEncoder(output_dim=128, norm_fn='instance', dropout=args.dropout)        
            self.cnet = SmallEncoder(output_dim=hdim+cdim, norm_fn='none', dropout=args.dropout)
            self.update_block = SmallUpdateBlock(self.args, hidden_dim=hdim)

        else:
            self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout)        
            self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=args.dropout)
            self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)

        if self.args.output_dim is not None:
            in_dim = 96 if self.args.small else 128
            hid_dim = 192 if self.args.small else 256
            self.output_block = nn.Sequential(
                nn.Conv2d(in_dim, hid_dim, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hid_dim, self.args.output_dim, 1, padding=0)
            )
        else:
            self.output_block = None

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//8, W//8, device=img.device, dtype=img.dtype)
        coords1 = coords_grid(N, H//8, W//8, device=img.device, dtype=img.dtype)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, C, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, C, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, C, 8*H, 8*W)

    @property
    def iters(self):
        if getattr(self, '_iters', None) is None:
            return None
        return self._iters
    @iters.setter
    def iters(self, value=None):
        self._iters = value

    def _forward_two_images(
            self,
            image1, image2,
            iters=24, flow_init=None,
            upsample=True, test_mode=True, **kwargs):
        """ Estimate optical flow between pair of frames """
        if self.iters is not None:
            iters = self.iters

        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        decorator = autocast(enabled=True) if \
            (self.args.mixed_precision or (image1.dtype in [torch.float16, torch.bfloat16])) \
            else Dummy(enabled=False)
        with decorator:
            fmap1, fmap2 = self.fnet([image1, image2])        
        
        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        if self.args.alternate_corr:
            corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
        else:
            corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

        # run the context network
        # with autocast(enabled=self.args.mixed_precision):
        with decorator:
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1) # index correlation volume

            flow = coords1 - coords0
            # with autocast(enabled=self.args.mixed_precision):
            with decorator:
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # get the output
            if self.output_block is not None:
                out = self.output_block(net)
            else:
                out = coords1 - coords0

            # upsample outputs
            if up_mask is None:
                flow_up = upflow8(out)
            else:
                flow_up = self.upsample_flow(out, up_mask)
            
            flow_predictions.append(flow_up)

        if test_mode:
            return coords1 - coords0, flow_up
            
        return flow_predictions

    def forward(self, *args, **kwargs):
        
        if not self.multiframe:
            return self._forward_two_images(*args, **kwargs)
        x = (args[0] * 255.0) if self.scale_inputs else args[0]

        if len(x.shape) == 4:
            x = x.unqueeze(1)
        assert len(x.shape) == 5, x.shape

        ## single frame input gets repeated
        if x.size(1) == 1:
            x = x.repeat(1, 2, 1, 1, 1)
        assert x.shape[1] >= 2, x.shape
        num_frames = x.size(1)
        flows = []
        backward = kwargs.get('backward', False)
        for t in range(num_frames-1):
            x1, x2 = torch.index_select(
                x, 1, torch.tensor([t,t+1]).to(x.device)).unbind(1)
            _args = (x2, x1) if backward else (x1, x2)
            flow = self._forward_two_images(*_args, *args[1:], **kwargs)[-1]
            flows.insert(0, flow) if backward else flows.append(flow)

        return torch.stack(flows, 1)

