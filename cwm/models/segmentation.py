import numpy as np
import copy
import torch
from torch import nn
from einops import rearrange

import cwm.models.masking as masking
import cwm.models.perturbation as perturbation
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

    def reset_shifts(self):
        self.shifts = []

    def create_motion_counterfactuals(self,
                                      x,
                                      masks,
                                      active_patches=None,
                                      shifts=None,
                                      frame=1,
                                      num_samples=None,
                                      fix_passive=True,
                                      reset_shifts=False):
        """
        Create motion counterfactuals by applying shifts to active_patches and no shifts to masks
        """
        if (getattr(self, 'shifts', None) is None) or reset_shifts:
            self.reset_shifts()
            
        if len(masks.shape) == 2:
            assert num_samples is not None, "Choose how many samples to shift with arg num_samples"
            masks = masks.unsqueeze(-1).expand(-1, -1, num_samples)
        else:
            num_samples = masks.size(-1)
        
        if active_patches is None:
            active_patches = torch.ones_like(masks)
        elif len(active_patches.shape) == 2:
            active_patches = active_patches.unsqueeze(-1).expand(-1, -1, masks.size(-1))

        motion_points = ~active_patches
        B, N, S = masks.shape
        assert motion_points.size(-1) in [1, S]
        if motion_points.size(-1) == 1:
            motion_points = motion_points.expand(-1, -1, S)

        if fix_passive:
            x = self.make_static_movie(x[:,0:1], T=2)

        x = self.sample_tile(x, S)
        _rearr = lambda tensor: rearrange(tensor, 'b n s -> (b s) n')
        masks, motion_points, active_patches = map(_rearr, (masks, motion_points, active_patches))

        # set the shifts
        self.shifter.set_num_shifts(S)
        shifts = self.shifter._preprocess_shifts_sequence(shifts, is_mask_shift=True)

        # shift each example one by one
        BS = B * S
        x_shift, mask_shift = [], []
        for i in range(BS):
            x_shift_i, mask_shift_i = self.shifter(
                x[i:i+1],
                mask=torch.minimum(masks[i:i+1], active_patches[i:i+1]),
                perturbation_points=motion_points[i:i+1],
                mask_shift=shifts[i],
                frame=frame
            )
            x_shift.append(x_shift_i)
            mask_shift.append(mask_shift_i)

            self.shift = self.shifter.shift
            self.shift = [self.shift[0] // self.patch_size[-2],
                          self.shift[1] // self.patch_size[-1]]
            self.shifts.append(np.array(self.shift))

        x_shift = torch.cat(x_shift, 0)
        mask_shift = torch.cat(mask_shift, 0)
        mask_shift = self.mask_rectangularizer(mask_shift)

        return (x_shift, mask_shift)

    def predict_counterfactual_videos_and_flows(self,
                                                x,
                                                active_patches,
                                                passive_patches=None,
                                                shifts=None,
                                                num_samples=8,
                                                sample_batch_size=8,
                                                fix_passive=True,
                                                max_shift_fraction=None,
                                                frame=1,
                                                raft_iters=None,
                                                backward=False,
                                                **kwargs):
        """
        Computing counterfactual flows from moving active patches and fixing passive patches;
        Both active and passive patches are revealed to the masked predictor in frame 1.

        Args:
            x: an input movie or image
        """
        # preprocess the input to be a 2-frame movie, regardless of what it is
        if len(x.shape) == 3:
            x = x.unsqueeze(0).unsqueeze(1).expand(-1, 2, -1, -1, -1)
            fix_passive = True
        elif len(x.shape) == 4:
            x = x.unsqueeze(1).expand(-1, 2, -1, -1, -1)
            fix_passive = True
        elif len(x.shape) == 5 and x.size(1) == 1:
            x = x.expand(-1, 2, -1, -1, -1)
        assert len(x.shape) == 5, x.shape
        x = x[:,0:2]
        
        self.set_input(x)
        self.reset_shifts()

        # preprocess the patches and shifts so they all have the same number of samples
        if passive_patches is None:
            passive_patches = self.get_zeros_mask().unsqueeze(-1)
        elif len(passive_patches.shape) == 2:
            passive_patches = passive_patches.unsqueeze(-1)

        if len(active_patches.shape) == 2:
            active_patches = active_patches.unsqueeze(-1)

        S = max(active_patches.size(-1), passive_patches.size(-1))
        if (S == 1) and num_samples > 1:
            S = num_samples
            
        if shifts is None:
            self.shifter.set_num_shifts(S)
            if max_shift_fraction is not None:
                self.shifter.max_shift_fraction = max_shift_fraction
        else:
            self.shifter.set_num_shifts(len(shifts) if not hasattr(shifts, 'shape') else shifts.shape[-1])
        shifts = self.shifter._preprocess_shifts_sequence(shifts, is_mask_shift=True)
        num_samples = len(shifts)

        if (active_patches.size(-1) == 1) and (num_samples > 1):
            active_patches = active_patches.expand(-1, -1, num_samples)
        if (passive_patches.size(-1) == 1) and (num_samples > 1):
            passive_patches = passive_patches.expand(-1, -1, num_samples)

        assert active_patches.size(-1) == passive_patches.size(-1) == num_samples, \
            (active_patches.shape, passive_patches.shape, num_samples)

        x_mocos, masks_mocos = self.create_motion_counterfactuals(
            x,
            masks=passive_patches,
            active_patches=active_patches,
            shifts=shifts,
            num_samples=num_samples,
            fix_passive=fix_passive,
            reset_shifts=False
        )

        ## batch predict
        y_mocos = self.batch_predict_per_sample(
            x_mocos,
            masks=masks_mocos,
            frame=None,
            batch_size=(sample_batch_size or x_mocos.size(0)),
            sample_dim=0,
            **kwargs
        )
        flow_mocos = self.predict_flow(y_mocos, backward=backward, iters=raft_iters)
        return (y_mocos, flow_mocos)


        
        
            
        
        


        

        
        
        
                                     
        
