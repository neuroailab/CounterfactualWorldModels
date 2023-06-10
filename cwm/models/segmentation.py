import numpy as np
import copy
import torch
from torch import nn

from cwm.models.prediction import PredictorBasedGenerator
from cwm.models.raft.raft_model import (load_raft_model,
                                        RAFT,
                                        default_raft_ckpt)

from cwm.vis_utils import imshow
from cwm.data.utils import FlowToRgb

class FlowGenerator(PredictorBasedGenerator):
    """
    A wrapper for masked predictors that runs counterfactual movies through a pretrained RAFT model to predict flow.

    Sampling counterfactual optical flow is the basis for Spelke Object segmentation in static images.
    """
    def __init__(self,
                 *args,
                 flow_model=None,
                 flow_model_load_path=None,
                 flow_model_kwargs={},
                 raft_iters=24,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)
        self.set_flow_model(
            flow_model=flow_model,
            flow_model_load_path=flow_model_load_path,
            **flow_model_kwargs)
        self.set_raft_iters(raft_iters)
        self.flow2rgb = FlowToRgb()

    def set_flow_model(self,
                       flow_model=None,
                       flow_model_load_path=None,
                       **kwargs):
        if flow_model is None:
            flow_model = load_raft_model(
                load_path=flow_model_load_path,
                multiframe=True,
                scale_inputs=True,
                **kwargs)
        else:
            assert isinstance(flow_model, nn.Module)

        self.flow_model = flow_model.eval().requires_grad_(False)

    def set_raft_iters(self, iters=None):
        for m in self.modules():
            if isinstance(m, RAFT):
                print("set RAFT to %s iters" % str(iters))
                m.iters = iters

    def predict_flow(self,
                     vid,
                     backward=False,
                     iters=None,
                     **kwargs):
        if iters is not None:
            self.set_raft_iters(iters)
        flow = self.flow_model(
            vid,
            backward=backward,
            **kwargs).to(vid)
        return flow

    def flowshow(self,
                 flow,
                 set_max_speed=True,
                 frame=0,
                 **kwargs):
        if len(flow.shape) == 5:
            flow = flow[:,frame]
        else:
            assert len(flow.shape) == 4

        if set_max_speed:
            self.flow2rgb.max_speed = flow.square().sum(-3, True).sqrt().amax().item()
        flow_img = self.flow2rgb(flow)
        imshow(flow_img, **kwargs)

    def predict_video_and_flow(self,
                               x=None,
                               mask=None,
                               backward=False,
                               propagate_error=False,
                               **kwargs):
        if x is None:
            x = self.x
        if mask is None:
            mask = self.mask

        num_frames = x.size(1)
        dt = self.sequence_length
        x_pred = [x[:,0:1]]
        for t in range(num_frames-dt+1):
            x_pred.append(self.predict(x[:,t:t+dt], mask, frame=1, **kwargs))
        x_pred = torch.cat(x_pred, 1)

        if propagate_error:
            f_pred = self.predict_flow(x_pred, backward, **kwargs)
        else:
            f_pred = []
            for t in range(num_frames-dt+1):
                _x = torch.cat([x[:,t:t+1], x_pred[:,t+1:t+2], x[:,t+2:t+dt]], 1)
                f_pred.append(self.predict_flow(_x, backward, **kwargs))
            f_pred = torch.cat(f_pred, 1)
                        
        return (x_pred, f_pred)

    def predict_flow_per_sample(self, x, masks, x_context=None, mask_context=None, timestamps=None,
                                backward=False, **kwargs):
        S = masks.size(-1)
        x_preds = self.predict_per_sample(x, masks, x_context=x_context, mask_context=mask_context,
                                          timestamps=timestamps,
                                          frame=None, split_samples=False)
        flow = self.predict_flow(x_preds, backward, **kwargs)
        p_dims = tuple(range(2, len(flow.shape)+1))
        flow = flow.view(-1,S,*flow.shape[1:]).permute(0,*p_dims,1)
        return flow        
    
    def predict_video_and_flow_per_sample(self,
                                          x,
                                          masks,
                                          x_context=None,
                                          mask_context=None,
                                          timestamps=None,
                                          backward=False,
                                          **kwargs):

        assert len(masks.shape) == 3
        B,_,S = masks.shape
        if x_context is not None:
            if x_context.size(0) != B*S:
                x_context = self.sample_tile(x_context, S)
        if mask_context is not None:
            if mask_context.size(0) != B*S:
                mask_context = self.sample_tile(mask_context, S)
        if timestamps is not None:
            if timestamps.size(0) != B*S:
                timestamps = self.sample_tile(timestamps, S)

        ys = self.predict_per_sample(
            x,
            masks,
            x_context=x_context,
            mask_context=mask_context,
            timestamps=timestamps,
            frame=None,
            split_samples=False,
            **kwargs)

        flows = self.predict_flow(ys, backward)
        p_dims = tuple(range(2, len(flows.shape)+1))
        ys = ys.view(-1,S,*ys.shape[1:]).permute(0,*p_dims,1)
        flows = flows.view(-1,S,*flows.shape[1:]).permute(0,*p_dims,1)
        return (ys, flows)
        
                 
                 
