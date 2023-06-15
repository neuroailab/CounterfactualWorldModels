import numpy as np
import copy
import torch
from torch import nn
from einops import rearrange

import cwm.models.masking as masking
import cwm.models.perturbation as perturbation
from cwm.models.prediction import PredictorBasedGenerator
from cwm.models.sampling import (FlowSampleFilter,
                                 RotatedTableEnergyMaskingGenerator)
from cwm.models.raft.raft_model import (load_raft_model,
                                        RAFT,
                                        default_raft_ckpt)

from cwm.vis_utils import imshow
from cwm.data.utils import FlowToRgb
import cwm.models.utils as utils

imagenet_normalize = utils.imagenet_normalize
imagenet_unnormalize = utils.imagenet_unnormalize

class FlowGenerator(PredictorBasedGenerator):
    """
    A wrapper for masked predictors that runs counterfactual movies through a pretrained RAFT model to predict flow.

    Sampling counterfactual optical flow is the basis for Spelke Object segmentation in static images.
    """
    default_flow_filter_params = {
        'filter_methods': ['patch_magnitude', 'flow_area', 'num_corners'],
        'flow_magnitude_threshold': 5.0,
        'flow_area_threshold': 0.75,
        'num_corners_threshold': 2
    }

    default_patch_sampling_kwargs = {
        'energy_power': 1,
        'eps': 1e-16,
        'pool_mode': 'mean',
        'resize': False
    }
    
    def __init__(self,
                 *args,
                 flow_model=None,
                 flow_model_load_path=None,
                 flow_model_kwargs={},
                 flow_sample_filter=FlowSampleFilter(**default_flow_filter_params),
                 raft_iters=24,
                 patch_sampling_func=RotatedTableEnergyMaskingGenerator,
                 patch_sampling_kwargs=default_patch_sampling_kwargs,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)
        self.set_flow_model(
            flow_model=flow_model,
            flow_model_load_path=flow_model_load_path,
            **flow_model_kwargs)
        self.set_raft_iters(raft_iters)
        self.flow2rgb = FlowToRgb()

        # filter for flow samples
        self.flow_sample_filter = flow_sample_filter

        # submodule for sampling patches
        self._patch_sampling_func = patch_sampling_func
        self._patch_sampling_kwargs = copy.deepcopy(self.default_patch_sampling_kwargs)
        self._patch_sampling_kwargs.update(patch_sampling_kwargs)
        self.set_patch_sampler()

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

    def set_flow_sample_filter(self, params=None):
        if params is None:
            self.flow_sample_filter = None
        else:
            self.flow_sample_filter = FlowSampleFilter(**params)

    def set_patch_sampler(self, num_visible=1, mask_ratio=None, **kwargs):
        if (getattr(self, 'patch_sampler', None) is None) or len(kwargs.keys()):
            _kwargs = copy.deepcopy(self._patch_sampling_kwargs)
            _kwargs.update(kwargs)
            try:
                mask_shape = self.mask_shape
            except:
                mask_shape = self.predictor.mask_size
            self.patch_sampler = self._patch_sampling_func(
                input_size=mask_shape,
                mask_ratio=(mask_ratio or 0),
                seed=self.rng.randint(9999),
                always_batch=True,
                **_kwargs)

        if mask_ratio is not None:
            self.patch_sampler.mask_ratio = mask_ratio
        elif num_visible is not None:
            self.patch_sampler.num_visible = num_visible * self.patch_sampler.clumping_factor**2

    def sample_patches_from_energy(self, energy=None, num_samples=10, num_visible=1, beta=None, **kwargs):
        self.set_patch_sampler(num_visible, **kwargs)
        if num_visible == 0:
            return torch.stack([self.get_zeros_mask() for _ in range(num_samples)], -1)
        if energy is None:
            assert self.x is not None
            energy = torch.ones_like(self.x[:,0,0:1])
        energy = utils.boltzmann(energy, beta)
        torch.manual_seed(self.rng.randint(99999))
        masks = torch.stack([self.patch_sampler(energy) for _ in range(num_samples)], -1)
        return masks

    @staticmethod
    def batch_to_samples(flows, t=0, B=1):
        assert len(flows.shape) == 5, flows.shape
        return rearrange(flows[:,t], '(b s) c h w -> b c h w s', b=B)

    def _batch_to_samples(self, flows, t=0):
        assert self.x is not None
        if len(flows.shape) != 5:
            flows = flows.unsqueeze(1)
            t = 0
        return self.batch_to_samples(flows, t=t, B=self.x.size(0))

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

    def compute_flow_samples_magnitude(self, flows, normalize=True, dim=-4, eps=1e-2):
        flow_mags = flows.square().sum(dim, True).sqrt().to(flows.dtype)
        if normalize:
            flow_mags = flow_mags - flow_mags.amin((-3, -2), True)
            flow_mags = flow_mags / flow_mags.amax((-3, -2), True).clamp(min=eps)
        return flow_mags

    def compute_mean_motion_map(self,
                                flows,
                                normalize_per_sample=False,
                                normalize=True,
                                dim=-4,
                                eps=1e-2):
        if len(flows.shape) == 5:
            flow_mags = self.compute_flow_samples_magnitude(flows,
                                                            normalize=normalize_per_sample,
                                                            dim=dim,
                                                            eps=eps)
            motion_map = flow_mags.mean(-1)
        else: # just normalize the input distribution
            motion_map = flows
            normalize = True
            
        if normalize:
            motion_map = motion_map - motion_map.amin((-2, -1), True)
            motion_map = motion_map / motion_map.amax((-2, -1), True).clamp(min=eps)
        return motion_map
    
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

        self.shifter.set_shapes(x, mask=active_patches[...,0])
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

    def sample_counterfactual_motion_map(self,
                                         x,
                                         active_sampling_distribution=None,
                                         passive_sampling_distribution=None,
                                         active_patches=None,
                                         passive_patches=None,
                                         num_active_patches=1,
                                         num_passive_patches=0,
                                         num_samples=8,
                                         sample_batch_size=8,
                                         patch_sampling_kwargs={},
                                         do_filter=True,
                                         **kwargs):

        self.set_input(x)

        def _sample_patches(dist, num_visible):
            return self.sample_patches_from_energy(energy=dist,
                                                   num_samples=num_samples,
                                                   num_visible=num_visible,
                                                   **patch_sampling_kwargs)

        if active_patches is None:
            active_patches = _sample_patches(active_sampling_distribution, num_active_patches)
        if passive_patches is None:
            passive_patches = _sample_patches(passive_sampling_distribution, num_passive_patches)

        ys, flows = self.predict_counterfactual_videos_and_flows(
            x,
            active_patches=active_patches,
            passive_patches=passive_patches,
            num_samples=num_samples,
            sample_batch_size=sample_batch_size,
            fix_passive=True,
            **kwargs
        )
        flows = self._batch_to_samples(flows)
        # flows = rearrange(flows.squeeze(1), '(b s) c h w -> b c h w s', b=x.size(0))

        if (self.flow_sample_filter is not None) and do_filter:
            flows, filter_mask = self.flow_sample_filter(flows, active_patches)

        return (flows, active_patches, passive_patches)

    @staticmethod
    def compute_flow_corrs(flow_samples,
                           flow_samples_swap=None,
                           downsample=1,
                           take_top_k=None,
                           do_spearman=False,
                           distance_func=utils.ChannelMSE(dim=1),
                           thresh=None,
                           use_covariance=False,
                           eps=1e-12,
                           binarize=False,
                           normalize=False,
                           zscore=False,
                           range_thresh=None
    ):
        B,C,H,W,S = flow_samples.shape
        if S == 0:
            flow_samples = torch.zeros(list(flow_samples.shape)[:-1] + [1]).to(
                flow_samples.device).float()
            S = 1
        
        if flow_samples_swap is not None:
            assert list(flow_samples_swap.shape) == [B,C,H,W,S]
        if take_top_k is None:
            K = S
        else:
            K = take_top_k

        ds = downsample            
        def _ds(fs):
            return torch.nn.functional.avg_pool3d(fs[...,:K].permute(0,1,4,2,3),
                                                  (1,ds,ds),
                                                  stride=(1,ds,ds)).permute(0,1,3,4,2)
        
        flow_inp = _ds(flow_samples)
        if flow_samples_swap is not None:
            flow_inp = torch.cat([flow_inp, _ds(flow_samples_swap)], -1)

        flow_inp = distance_func(flow_inp, torch.zeros_like(flow_inp)).reshape(B,-1,flow_inp.size(-1))

        flow_corrs = []
        for b in range(B):
            if do_spearman:
                flow_inp_b = torch.argsort(flow_inp[b], -1).float()
            else:
                flow_inp_b = flow_inp[b]
                
            if (thresh is not None) and (binarize is False):
                flow_inp_b = flow_inp_b * (flow_inp_b > thresh).float()
            elif thresh is not None:
                flow_inp_b = (flow_inp_b > thresh).float()
            elif range_thresh is not None:
                flow_inp_b = flow_inp_b - flow_inp_b.amin(0, True)
                flow_range = flow_inp_b.amax(0, True)
                flow_inp_b = (flow_inp_b > (range_thresh * flow_range)).float()

            if normalize:
                flow_inp_b = flow_inp_b / flow_inp_b.amax(0, True).clamp(min=eps)
            if zscore:
                mn, std = flow_inp_b.mean(0), flow_inp_b.std(0).clamp(min=eps)
                flow_inp_b = (flow_inp_b - mn[None]) / std[None]
            
            flow_corrs_b = torch.cov(flow_inp_b) if use_covariance else torch.corrcoef(flow_inp_b)
            flow_corrs_b[torch.isnan(flow_corrs_b)] = 0

            flow_corrs.append(flow_corrs_b)

        flow_corrs = torch.stack(flow_corrs, 0)
        flow_corrs = flow_corrs.view(B,1,H//ds,W//ds,H//ds,W//ds)
        return flow_corrs    

class ImuGenerator(FlowGenerator):
    """
    Wrap predictors that input and output IMU data in addition to RGB video
    """
    def __init__(self,
                 *args,
                 head_mask_generator=None,
                 head_mask_ratio=0,
                 always_use_predicted=False,
                 require_none_missing=False,
                 **kwargs):
        super().__init__(*args, **kwargs)

        # IMU masked predictors in this work are always conjoined vmaes with a main RGB stream
        assert hasattr(self.predictor, 'context_stream')
        self._is_padded = hasattr(self.predictor.context_stream, 'padding_mask')
        self.num_head_tokens = self.predictor.context_stream.encoder.num_tokens

        # default mask generator for the visual input/output
        if self.mask_generator is None:
            self.mask_generator = masking.MaskingGenerator(
                input_size=self.predictor.mask_size,
                mask_ratio=0,
                always_batch=True,
                create_on_cpu=False)

        ## default mask generator for the head motion input/output
        if head_mask_generator is not None:
            self.head_mask_generator = head_mask_generator
        else:
            self.set_head_mask_generator()
            self.set_head_mask_params(mask_ratio=head_mask_ratio)

        self._always_use_predicted = always_use_predicted
        self._require_none_missing = require_none_missing
        self.missing_imu = None

    def set_head_mask_generator(self):
        self.head_mask_generator = masking.MissingDataImuMaskGenerator(
            input_size=(self.num_head_tokens),
            mask_ratio=0,
            full_mask_prob=0,
            full_vis_prob=0,
            truncation_mode='none',
            create_on_cpu=True)

    def set_head_mask_params(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self.head_mask_generator, k, v)

    def set_mode(self, mode='output'):
        if mode == 'output':
            self.set_head_mask_params(mask_ratio=1.0)
        elif mode == 'input':
            self.set_head_mask_params(mask_ratio=0.0)
        else:
            raise ValueError("%s is not a known mode" % mode)

    def input_mode(self):
        self.set_mode('input')
    def output_mode(self):
        self.set_mode('output')

    def get_imu_input(self,
                      inp_dict,
                      imu_mode='input',
                      missing_thresh=0.5,
                      device=None):

        if imu_mode is not None:
            self.set_mode(imu_mode)

        add_batch_dim = (len(inp_dict['imu'].shape) != 3)
        _unsq = lambda x: (x.unsqueeze(0) if add_batch_dim else x)

        if self.t_dim == 2:
            x = imagenet_unnormalize(_unsq(inp_dict['video']).transpose(1,2)).transpose(1,2)
        else:
            x = imagenet_unnormalize(_unsq(x))
        imu = _unsq(inp_dict['imu'])
        if self.t_dim == 2:
            imu = imu.transpose(1,2)
        missing_imu = _unsq(inp_dict['imu_missing_data'])
        missing_imu = missing_imu.view(missing_imu.size(0), self.num_head_tokens, -1)
        imu_mask = self.head_mask_generator(missing_imu.float().mean(-1) > missing_thresh)
        ts = _unsq(inp_dict['video_ts'])

        out_list = [x, imu, missing_imu, imu_mask, ts]
        if device is not None:
            out_list = [v.to(device) for v in out_list]
        return out_list

    def reshape_input(self, x, tubelet_size=None):
        if tubelet_size is None:
            tubelet_size = self.predictor.context_stream.patch_size[0]
        return rearrange(x, 'b c (t pt) -> b t (pt c)', pt=tubelet_size)

    def reshape_output(self, y, tubelet_size=None):
        if tubelet_size is None:
            tubelet_size = self.predictor.context_stream.patch_size[0]
        c = y.size(-1) // tubelet_size
        return rearrange(y, 'b t (pt c) -> b c (t pt)', c=c, pt=tubelet_size)

    def predict_imu(self, inp_dict, imu_mask_ratio=1, device=None, get_labels=True):

        self.set_head_mask_params(mask_ratio=imu_mask_ratio)

        x, imu, missing_imu, imu_mask, timestamps = self.get_imu_input(
            inp_dict, device=device, imu_mode=None)
        self.missing_imu = missing_imu
        self.mask = self.mask_generator(x).to(x.device)

        if imu_mask_ratio == 1:
            imu_mask = torch.ones_like(imu_mask)
        elif not self._is_padded:
            imu_mask = self.mask_rectangularizer(imu_mask)

        main_out, imu_out = self.predictor(
            x=x.transpose(1,2),
            mask=self.mask,
            timestamps=timestamps,
            x_context=imu,
            mask_context=imu_mask,
            output_main=True,
            output_context=True)

        imu_labels_orig = self.reshape_input(imu)
        
        ## no indexing necessary
        if imu_mask_ratio == 1 and (not self._is_padded):
            imu_labels = imu_labels_orig
            imu_pred = imu_out

        ## padded models require some indexing to get the non-padding tokens
        elif self._is_padded:
            imu_labels = self.predictor.get_masked_imu(imu, torch.ones_like(imu_mask))
            imu_pred = torch.zeros_like(imu_labels)
            null_mask = self.predictor.context_stream.null_mask

            imu_true = self.predictor.get_masked_imu(imu, ~imu_mask)
            imu_pred[null_mask] = imu_true[null_mask]
            imu_pred[~null_mask] = imu_out[~null_mask]

            _imu_pred = imu_pred[~null_mask]
            _imu_labels = imu_labels[~null_mask]

            imu_pred = torch.zeros(imu_out.size(0), self.num_head_tokens, _imu_pred.size(-1)).to(_imu_pred)
            imu_pred[imu_mask] = _imu_pred
            imu_pred[~imu_mask] = imu_labels_orig[~imu_mask]

            imu_labels = torch.zeros_like(imu_pred)
            imu_labels[imu_mask] = _imu_labels
            imu_labels[~imu_mask] = imu_labels_orig[~imu_mask]

            self.predictor.context_stream._reset_padding_mask()
            
        else: ## no padding involved
            imu_labels = imu_labels_orig
            imu_pred = torch.zeros_like(imu_labels)
            imu_true = self.predictor.get_masked_imu(imu, ~imu_mask)
            imu_pred[~imu_mask] = imu_true.view(-1, imu_pred.size(-1))
            imu_pred[imu_mask] = imu_out.view(-1, imu_pred.size(-1))

        if getattr(self.predictor, '_main_padded', False):
            self.predictor.main_stream._reset_padding_mask()

        if get_labels:
            return (imu_pred, imu_labels)
        return imu_pred

    @property
    def any_imu(self):
        if self.missing_imu is None:
            return None
        return ~(torch.amin(self.missing_imu, (-2,-1)).bool())

    @property
    def full_imu(self):
        if self.missing_imu is None:
            return None
        return ~(torch.amax(self.missing_imu, (-2,-1)).bool())

    def forward(self, inp_dict, imu_mask_ratio=1, device=None):
        """Get the predicted imu when it's not in the dataset and the true imu when it is (optionally)"""
        imu_pred, imu_labels = self.predict_imu(inp_dict,
                                                imu_mask_ratio=imu_mask_ratio,
                                                device=device,
                                                get_labels=True)

        ## get the predicted imu when it's not in the data
        if self._always_use_predicted:
            imu_out = imu_pred
        elif self._require_none_missing:
            imu_out = torch.where(self.full_imu[:,None,None], imu_labels, imu_pred)
        else:
            imu_out = torch.where(self.any_imu[:,None,None], imu_labels, imu_pred)

        ## update the missing data indicator
        if self._always_use_predicted:
            missing_imu = torch.zeros_like(self.missing_imu)
        else:
            missing_imu = torch.where(self.any_imu[:,None,None],
                                      self.missing_imu,
                                      torch.zeros_like(self.missing_imu))

        return (imu_out, missing_imu)

class ImuConditionedFlowGenerator(FlowGenerator):
    """
    A combined wrapper for two models:
        1. A model that predicts ~2 seconds of IMU data from a pair of frames
        2. A masked predictor conditioned on both patches of video and on ~2 seconds of IMU

    Note that model (1) is just a special case of a masked predictor that _predicts_ IMU,
    which is always masked as an input to the model.
    """
    default_imu_generator_kwargs = {
        'head_mask_ratio': 1
    }
    
    def __init__(self,
                 *args,
                 predictor,
                 head_motion_predictor,
                 head_motion_load_path=None,
                 head_motion_generator=ImuGenerator,
                 head_motion_kwargs=default_imu_generator_kwargs,
                 head_motion_mask_generator=None,
                 flow_model=None,
                 flow_model_load_path=None,
                 **kwargs):

        # init the main predictor model, which has imu conditioning
        super().__init__(*args,
                         predictor=predictor,
                         flow_model=flow_model,
                         flow_model_load_path=flow_model_load_path,
                         **kwargs)

        # init the imu predictor
        head_motion_kwargs = copy.deepcopy(head_motion_kwargs)
        self._update_head_motion_kwargs(head_motion_load_path, head_motion_kwargs)

        if not isinstance(head_motion_predictor, nn.Module):
            head_motion_predictor = head_motion_predictor()
        self.head_motion_generator = head_motion_generator(
            predictor=head_motion_predictor,
            mask_generator=head_motion_mask_generator,
            flow_model=self.flow_model,
            **head_motion_kwargs)

    def _update_head_motion_kwargs(self, load_path, kwargs):
        kwargs['imagenet_normalize_inputs'] = kwargs.get('imagenet_normalize_inputs',
                                                         self.imagenet_normalize_inputs)
        kwargs['temporal_dim'] = kwargs.get('temporal_dim', self.predictor.t_dim)
        kwargs['predictor_load_path'] = load_path

    @property
    def num_head_tokens(self):
        return self.head_motion_generator.num_head_tokens

    @property
    def head_tubelet_size(self):
        return self.head_motion_generator.predictor.context_stream.patch_size[0]

    @property
    def head_motion_channels(self):
        return getattr(self.head_motion_generator.predictor.get_context_input, 'num_channels', 6)

    def get_fake_head_motion(self, x):
        """Get a fake head motion input and mask, typical use case for input to head_motion_generator"""
        B = x.size(0)
        device = x.device
        head_motion = torch.zeros(
            (B, self.head_tubelet_size * self.num_head_tokens, self.head_motion_channels),
            device=device).to(x.dtype)
        head_mask = torch.ones(
            (B, self.num_head_tokens),
            device=device).bool()

        if self.head_motion_generator.t_dim == 2:
            head_motion = head_motion.transpose(self.head_motion_generator.t_dim,
                                                self.head_motion_generator.c_dim)
        return (head_motion, head_mask)

    def predict_imu_from_video(self, x, timestamps=None):
        fake_imu, imu_mask = self.get_fake_head_motion(x)
        mask = self.head_motion_generator.mask_generator(x).to(x.device)
        x = self.head_motion_generator._preprocess(x)
        
        imu_out = self.head_motion_generator.predictor(
            x,
            mask=mask,
            timestamps=timestamps,
            x_context=fake_imu,
            mask_context=imu_mask,
            output_main=False,
            output_context=True)

        if not self.head_motion_generator._is_padded:
            return imu_out

        imu_labels_orig = self.head_motion_generator.reshape_imu(fake_imu)
        imu_labels = self.head_motion_generator.predictor.get_masked_imu(
            fake_imu, imu_mask)
        imu_pred = torch.zeros_like(imu_labels)
        null_mask = self.head_motion_generator.predictor.context_stream.null_mask
        imu_true = self.head_motion_generator.predictor.get_masked_imu(fake_imu, ~imu_mask)
        imu_pred[null_mask] = imu_true[null_mask]
        imu_pred[~null_mask] = imu_out[~null_mask]
        _imu_pred = imu_pred[~null_mask]
        imu_pred = torch.zeros_like(imu_out.size(0),
                                    self.num_head_tokens,
                                    _imu_pred.size(-1)).to(_imu_pred)
        imu_pred[imu_mask] = _imu_pred
        imu_pred[~imu_mask] = imu_labels_orig[~imu_mask]

        self.head_motion_generator.predictor.context_stream._reset_padding_mask()        

        if getattr(self.head_motion_generator.predictor, '_main_padded', False):
            self.head_motion_generator.main_stream._reset_padding_mask()

        return imu_pred

    def get_static_imu(self, x=None, timestamps=None):
        if x is None:
            x = self.x
        _x = torch.tile(x[:,0:1], (1,x.size(1),1,1,1))
        return self.predict_imu_from_video(_x, timestamps=timestamps)

    def get_zeros_imu(self, x=None, timestamps=None):
        if x is None:
            x = self.x
        return torch.zeros_like(
            self.predict_imu_from_video(x, timestamps=timestamps))

    def predict_imu_video_and_flow(self,
                                   x,
                                   mask=None,
                                   timestamps=None,
                                   head_motion=None,                
                                   mask_head_motion=False,
                                   static_head_motion=False,
                                   return_flow=True,
                                   return_head_motion=False,
                                   *args, **kwargs):

        self.set_input(x)
        if mask is None:
            self.mask = self.generate_mask(x)
        else:
            self.mask = mask

        ## get head motion
        if head_motion is not None:
            h = head_motion
        elif static_head_motion:
            h = self.get_static_imu()
        else:
            h = self.predict_imu_from_video(x, timestamps=timestamps)

        if return_head_motion:
            return h

        # get head motion mask
        h_mask = torch.zeros(h.size(0), self.num_head_tokens).bool().to(h.device)
        if mask_head_motion:
            h_mask = ~h_mask

        # get the video and flow prediction
        y, flow = self.predict_video_and_flow(
            x,
            mask=self.mask,
            timestamps=timestamps,
            x_context=self.head_motion_generator.reshape_output(h),
            mask_context=h_mask,
            *args, **kwargs)

        self.reset_padding_masks()

        return (y, flow)

    def predict_counterfactual_videos_and_flows(self,
                                                x,
                                                *args,
                                                head_motion=None,
                                                timestamps=None,
                                                mask_head_motion=False,
                                                static_head_motion=True,
                                                **kwargs):
        self.set_input(x)
        h = self.predict_imu_video_and_flow(
            x,
            *args,
            head_motion=head_motion,
            static_head_motion=static_head_motion,
            return_head_motion=True,
            **kwargs
        )
        self.mask = None
        self.reset_padding_masks()
        h_mask = torch.zeros(h.size(0), self.num_head_tokens).bool().to(h.device)
        if mask_head_motion:
            h_mask = ~h_mask

        h = self.head_motion_generator.reshape_output(h)

        return super().predict_counterfactual_videos_and_flows(
            x,
            *args,
            timestamps=timestamps,
            x_context=h,
            mask_context=h_mask,
            **kwargs
        )
            

    def forward(self, *args, **kwargs):
        return self.predict_imu_video_and_flow(*args, **kwargs)
    
    
        
        
        

        
        
        
                                     
        
