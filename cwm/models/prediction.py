import numpy as np
import copy
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
import kornia
from einops import rearrange

from cwm.models.patches import Patchify
import cwm.models.utils as utils
import cwm.models.perturbation as perturbation
import cwm.models.masking as masking

from cwm.vis_utils import imshow

class PredictorBasedGenerator(nn.Module):
    """
    Base wrapper class for getting factual and counterfactual predictions
    from a pretrained model, assumed to have the functional signature of models in
    cwm/models/VideoMAE/vmae.py
    """
    def __init__(self,
                 predictor=None,
                 predictor_load_path=None,                 
                 error_func=nn.MSELoss(reduction='none'),
                 imagenet_normalize_inputs=False,
                 temporal_dim=2,
                 seed=0,
                 mask_generator=None,
                 raft_iters=None,
                 **kwargs
    ):
        super().__init__()

        self.set_predictor(predictor, predictor_load_path)
        self.error_func = error_func
        self.imagenet_normalize_inputs = imagenet_normalize_inputs
        self.set_temporal_dim(temporal_dim)
        # self.set_raft_iters(raft_iters)

        self.rng = np.random.RandomState(seed=seed)
        self.torch_rng = torch.manual_seed(seed)
        self.seed = seed

        self.mask_generator = mask_generator
        self.mask_rectangularizer = masking.RectangularizeMasks('min')

        self.x, self.mask, self.timestamps = None, None, None

    def set_predictor(self, net, predictor_load_path):
        self.predictor = net
        self.load_predictor(predictor_load_path, model=None)
        self.patchify = Patchify(self.patch_size, temporal_dim=1, squeeze_channel_dim=True)
        self.x = self.mask = self.inp_shape = None

    # def set_raft_iters(self, iters=None):
    #     for m in self.modules():
    #         if isinstance(m, RAFT):
    #             print("set RAFT to %s iters" % str(iters))
    #             m.iters = iters

    def load_predictor(self, load_path=None, model=None, map_location='cpu'):
        if (getattr(self, 'predictor', None) is None) and (model is None):
            raise ValueError("There is no predictor set for this generator and no model to load to")
        
        if load_path is None:
            print(
                ("No predictor weights were loaded in constructing the %s\n" +\
                "This is fine as long as you already loaded the weights!") % \
                type(self).__name__)
            if hasattr(self.predictor, '_predictor_load_path'):
                self._predictor_load_path = self.predictor._predictor_load_path
            return

        try:
            weights = torch.load(load_path, map_location=torch.device(map_location))
            if 'model' in weights.keys():
                weights = weights['model']
            if model is None:
                did_load = self.predictor.load_state_dict(weights)
                self._predictor_load_path = load_path                
            else:
                did_load = model.load_state_dict(weights)
            print(did_load, load_path)

        except FileExistsError as e:
            print(e)
            print("No weights were found at %s" % load_path)

    def generate_mask(self, x=None):
        assert self.mask_generator is not None
        if x is None:
            x = self.x
        mask = self.mask_generator(x).view(x.size(0), -1).to(x.device)
        return self.mask_rectangularizer(mask)

    def set_new_mask(self, x=None):
        if x is None:
            x = self.x
        self.mask = self.generate_mask(x)

    def reset_padding_masks(self):
        if hasattr(self.predictor, 'padding_mask') and not hasattr(self.predictor, 'main_stream'):
            self.predictor._reset_padding_mask()

        elif hasattr(self.predictor, 'main_stream'):
            if hasattr(self.predictor.main_stream, 'padding_mask'):
                self.predictor.main_stream._reset_padding_mask()
            if hasattr(self.predictor.context_stream, 'padding_mask'):
                self.predictor.context_stream._reset_padding_mask()

    @property
    def patch_size(self):
        if self.predictor is None:
            return None
        elif hasattr(self.predictor, 'patch_size'):
            return self.predictor.patch_size
        elif hasattr(self.predictor.encoder.patch_embed, 'proj'):
            return self.predictor.encoder.patch_embed.proj.kernel_size
        else:
            return None

    def _get_patch_size(self, p):
        if p is None:
            return None
        elif isinstance(p, int):
            return (1, p, p)
        elif len(p) == 2:
            return (1, p[0], p[1])
        else:
            assert len(p) == 3, p
            return (p[0], p[1], p[2])
        
    @property
    def image_size(self):
        if self.predictor is None:
            return None
        return self.predictor.image_size

    @property
    def sequence_length(self):
        if self.predictor is None:
            return None
        elif hasattr(self.predictor, 'sequence_length'):
            return self.predictor.sequence_length
        elif hasattr(self.predictor, 'num_frames'):
            return self.predictor.num_frames        
        else:
            return 2

    @property
    def mask_shape(self):
        if self.predictor is None:
            return None
        elif hasattr(self.predictor, 'mask_shape'):
            return self.predictor.mask_shape

        assert self.patch_size is not None
        pt, ph, pw = self.patch_size
        return (self.sequence_length // pt,
                self.inp_shape[-2] // ph,
                self.inp_shape[-1] // pw)

    @property
    def inp_mask_shape(self):
        return (self.x.shape[0], np.prod(self.mask_shape))

    def set_temporal_dim(self, t_dim=1):
        if t_dim == 1:
            self.predictor.t_dim = 1
            self.predictor.c_dim = 2
        elif t_dim == 2:
            self.predictor.c_dim = 1
            self.predictor.t_dim = 2
        else:
            raise ValueError("temporal_dim must be 1 or 2")

    @property
    def t_dim(self):
        if self.predictor is None:
            return None
        return self.predictor.t_dim

    @property
    def c_dim(self):
        if self.predictor is None:
            return None
        return self.predictor.c_dim

    def set_image_size(self, *args, **kwargs):
        assert self.predictor is not None, "Can't set the image size without a predictor"
        if hasattr(self.predictor, 'set_image_size'):
            self.predictor.set_image_size(*args, **kwargs)
        else:
            self.predictor.image_size = args[0]

    def get_zeros_mask(self, x=None, frame=-1):
        if x is None:
            x = self.x
        mask = torch.zeros(self.mask_shape, device=x.device, dtype=torch.bool)
        if frame is not None:
            mask[frame,...] = torch.ones_like(mask[frame,...])
        mask = mask.flatten()
        mask = mask.unsqueeze(0).expand(x.shape[0], -1)
        return mask

    def get_fully_visible_mask(self, x=None):
        if x is None:
            x = self.x
        return torch.zeros(self.mask_shape, device=x.device, dtype=torch.bool)

    def mask_complement(self, mask1, mask2, frame=-1):
        mask1 = self.get_mask_image(mask1)
        mask2 = self.get_mask_image(mask2)
        mask_diff = mask1 & (~mask2)        
        if frame is None:
            return (~mask_diff).view(mask_diff.shape[0], -1)
        else:
            frame = (frame % mask1.shape[1])
            return torch.cat([
                mask1[:,:frame],
                ~mask_diff[:,frame,None],
                mask1[:,(frame+1):]
            ], 1).view(mask_diff.shape[0], -1)

    def pred_patches_to_video(self, y, x, mask):
        """input at visible positions, preds at masked positions"""
        B, C = y.shape[0], y.shape[-1]
        self.patchify._check_shape(x)
        self.patchify.D = np.prod(self.patch_size)
        x = self.patchify(x).to(y.dtype)
        y_out = torch.zeros_like(x)
        x_vis = x[~mask]
        y_out[~mask] = x_vis.view(-1,C)
        try:
            y_out[mask] = y.view(-1,C)
        except:
            y_out[mask] = y.reshape(-1,C)

        return self.patchify(y_out, to_video=True)

    def get_masked_pred_patches(self, preds, mask,
                                invert=False, fill_value=None):
        _inp_shape = copy.deepcopy(self.inp_shape)
        shape = preds.shape
        self.inp_shape = shape
        mask_shape = (self.inp_shape[1],) + self.mask_shape[-2:]
        mask_vis = masking.upsample_masks(
            self.get_mask_image(
                mask, shape=mask_shape),
            shape[-2:]).to(preds)
        if invert:
            mask_vis = 1.0 - mask_vis
        self.inp_shape = _inp_shape
        out = preds * mask_vis.unsqueeze(2)
        if isinstance(fill_value, torch.Tensor):
            assert list(fill_value.shape) == list(out.shape)
            out = out + (1 - mask_vis.unsqueeze(2)) * fill_value
        elif fill_value is not None:
            fill_value = torch.tensor(fill_value).to(out.device).to(out)
            fill_value = fill_value.view(1,1,-1,1,1)
            out = out + (1 - mask_vis.unsqueeze(2)) * fill_value
        return out

    def patchify_energy_density(self, density, mode='min', beta=None):
        rank = len(density.shape)
        assert rank in [4,5], rank
        density = utils.boltzmann(density, beta=beta)
        
        if mode == 'mean':
            func = F.avg_pool3d if rank == 5 else F.avg_pool2d
        elif mode == 'max':
            func = F.max_pool3d if rank == 5 else F.max_pool2d
        elif mode == 'min':
            func = (lambda x, **kw: -F.max_pool3d(-x, **kw)) if rank == 5 \
                else (lambda x: -F.max_pool2d(-x, **kw))

        pooled_density = func(
            (density.transpose(1,2) if rank == 5 else density),
            kernel_size=self.patch_size, stride=self.patch_size)
        if rank == 5:
            pooled_density = pooled_density.squeeze(1)
        return pooled_density

    def _preprocess(self, x):

        if self.t_dim != 1:
            x = x.transpose(self.t_dim, self.c_dim)

        if self.imagenet_normalize_inputs:
            x = utils.imagenet_normalize(x, temporal_dim=self.t_dim)
            
        return x

    def _get_frames(self, x, frames=0):
        assert len(x.shape) == 5, x.shape
        frame_tensor = torch.tensor(frames).long().to(x.device)
        return torch.index_select(x, dim=1, index=frame_tensor)

    def _get_target(self, x):
        assert len(x.shape) == 5, x.shape
        assert x.shape[1] == 2, (x.shape[1])
        return self._get_frames(x, frames=[1])

    def _get_error(self, pred, gt, dim=-3, frame=None):
        T_pred = gt.shape[1]
        return self.error_func(
            pred[:,-T_pred:],
            gt
        ).sum(dim, True)

    def predict_error(x=None, mask=None, target=None, frame=None, dim=-3):
        if x is None:
            x = self.x
        if mask is None:
            mask = self.generate_mask(x)
        x_pred = self.predict(x, mask, frame=frame)
        if target is None:
            target = x
        if frame is None:
            target = target[:,frame].unsqueeze(1)
            
        error = self.error_func(x_pred, target).sum(dim, True)
        return error

    def get_nearby_patches(self, mask, radius=1, upsample=False, shape=None):
        mask = self.get_mask_image(mask, shape=shape)
        nearby_patches = masking.patches_adjacent_to_visible(
            mask, radius=radius, size=None)
        if upsample:
            nearby_patches = masking.upsample_masks(nearby_patches, size=self.inp_shape[-2:])
        return nearby_patches

    def _upsample_mask(self, mask):
        return masking.upsample_masks(
            mask.view(mask.size(0), -1, *self.mask_shape[-2:]).float(), self.inp_shape[-2:])

    def get_mask_image(self, mask, upsample=False, invert=False, shape=None):
        if shape is None:
            shape = self.mask_shape
        mask = mask.view(-1, *shape)
        if upsample:
            mask = self._upsample_mask(mask)
        if invert:
            mask = 1 - mask
        return mask

    def maskshow(self, mask, shape=None, ex=0, **kwargs):
        mask = self.get_mask_image(mask, shape=shape).transpose(0,1)
        imshow(mask[:,ex:ex+1], cmap='inferno', vmin=0, vmax=1, **kwargs)

    @staticmethod
    def invert_mask_frame(mask, size, frame=-1):
        shape = mask.shape
        mask = mask.view(shape[0], -1, *size)
        frame = frame % mask.size(1)
        mask = torch.cat([
            mask[:,:frame],
            ~mask[:,frame:frame+1],
            mask[:,(frame+1):]
        ], 1).view(*shape)
        return mask

    def _invert_mask(self, mask, frame=-1):
        return self.invert_mask_frame(mask, self.mask_shape[-2:], frame)

    def _sample_random_patches(self, batch_size=1, t_idx=None):
        patches = []
        for b_idx in range(batch_size):
            if t_idx is None:
                t_idx = self.mask_shape[0] - 1 # unmask in last frame
            h_idx = self.rng.randint(self.mask_shape[1])
            w_idx = self.rng.randint(self.mask_shape[2])
            patches.append([b_idx, t_idx, h_idx, w_idx])
        return patches

    def _sample_patches_from_dist(self, probs):
        patches = []
        for b in range(probs.shape[0]):
            b_idx = torch.tensor([b]).long().to(probs.device)
            t_idx = torch.tensor([self.mask_shape[0] - 1]).long().to(probs.device)
            hw_inds = utils.sample_image_inds_from_probs(
                F.relu(probs[b]), num_points=1).long() # [1,1,2]
            patches.append(torch.cat([b_idx, t_idx, hw_inds[0,0]], 0))
        return patches

    def predict(self, x=None, mask=None, frame=-1, reset_masks=True, *args, **kwargs):
        if x is None:
            x = self.x
        if mask is None:
            mask = self.generate_mask(x)

        self.set_image_size(x.shape[-2:])
        y = self.predictor(
            self._preprocess(x),
            mask if (x.size(0) == 1) else self.mask_rectangularizer(mask),
            *args, **kwargs)

        if hasattr(self.predictor, 'padding_mask'):
            if hasattr(self.predictor, 'main_stream'):
                y = y[:,:-self.predictor.main_stream.max_padding_tokens]
            else:
                y = y[:,:-self.predictor.max_padding_tokens]

        ## if y isn't a video, need to do some postprocessing
        if len(y.shape) != 5:
            if hasattr(self.predictor, 'main_stream'):
                _x, _mask = self.predictor.get_current_inputs(
                    self._preprocess(x), mask, *args, **kwargs)[0][:2]
                if self.t_dim == 2:
                    _x = _x.transpose(1,2)
            else:
                _x, _mask = x, mask

            y = self.pred_patches_to_video(y, _x, mask=_mask)
            if frame is not None:
                frame = frame % y.size(1)
                y = y[:,frame:frame+1]

        if reset_masks:
            self.reset_padding_masks()
            
        return y

    def predict_per_sample(self, x, masks, frame=-1, batch_size=None,
                           split_samples=True, *args, **kwargs):
        """Run predictions in parallel for S sample masks"""
        assert len(masks.shape) == 3, masks.shape
        S = masks.size(-1)
        if x is None:
            x = self.x
        B = x.size(0)
        BS = B*S

        ## tile along batch dimension
        try:
            x = x[:,None].expand(-1,S,-1,-1,-1,-1).view(BS,*x.shape[1:])
        except:
            x = x[:,None].expand(-1,S,-1,-1,-1,-1).reshape(BS,*x.shape[1:])

        try:
            masks = masks.transpose(1,2).view(BS,-1)
        except:
            masks = masks.transpose(1,2).reshape(BS,-1)

        y = self.predict(x=x, mask=masks, frame=frame, *args, **kwargs)
        if not split_samples:
            return y
        p_dims = tuple(range(2,len(y.shape)+1))
        y = y.view(B,S,*y.shape[1:]).permute(0,*p_dims,1)
        return  y

    def sample_tile(self, z, num_samples):
        S = num_samples
        rank = len(z.shape)
        return z[:,None].expand(-1,S,*([-1]*(rank-1))).reshape(-1,*z.shape[1:])

    def batch_predict_per_sample(self, x, masks, frame=-1, batch_size=None, **kwargs):
        S = masks.size(-1)
        if batch_size is None:
            batch_size = S
        else:
            batch_size = max(1, batch_size // x.size(0))
            
        ys = []
        for b in range(int(np.ceil(S / batch_size))):
            b0,b1 = b*batch_size, (b+1)*batch_size
            ys.append(
                self.predict_per_sample(
                    x,
                    masks=masks[...,b0:b1],
                    split_samples=True,
                    frame=frame,
                    **kwargs
                )
            )
            self.reset_padding_masks()
        return torch.cat(ys, -1)
            

    def predict_with_mask(self, mask, invert_mask=False, *args, **kwargs):
        assert self.x is not None
        if invert_mask:
            mask = ~mask
        return self.predict(self.x, mask.view(*self.inp_mask_shape), *args, **kwargs)

    def error_with_mask(self, mask, invert_mask=False, frame=-1, *args, **kwargs):
        x_pred = self.predict_with_mask(mask, invert_mask, *args, **kwargs)
        error = self._get_error(x_pred[:,frame].unsqueeze(1), self.x[:,frame].unsqueeze(1), dim=-3)
        return error

    def get_error_on_target_region(self,
                                   x, mask, target_mask,
                                   target=None, average_error=True, frame=-1,
                                   aggregate_over_patches=True, patch_size=None,
                                   **kwargs):

        if target is None:
            target = x
        if len(target_mask.shape) == 2:
            target_region = 1 - target_mask.view(x.shape[0], -1, *self.mask_shape[-2:]).to(x)
        else:
            target_region = 1 - target_mask.to(x)
        error = self._get_error(self.predict(x, mask, frame=frame, **kwargs), target)
        if not aggregate_over_patches:
            return error
        patch_size = patch_size or self.patch_size
        error = F.avg_pool3d(error.transpose(1,2), patch_size, stride=patch_size)
        error = (error.squeeze(1) * target_region.to(x))
        if not average_error:
            return error
        error = error.sum((1,2,3)) / target_region.sum((1,2,3)).clamp(min=1)
        return error
    
        
    def get_initial_mask(self, x):
        raise NotImplementedError("Need to specify how to get the initial mask")

    @staticmethod
    def unmask_one_patch(mask, idx=None, mask_shape=None, inplace=False, frame=0):
        """Unmask the patch at idx position."""
        shape = mask.shape
        if not inplace:
            mask = mask.clone()
            
        if mask_shape is None:
            assert len(shape) == 2, "If you don't pass a mask shape, it must be [B,N]"
            mask[:,idx] = torch.zeros_like(mask[:,0])
            return mask

        ## else need an index for each dimension of mask_shape
        if len(idx) == 2 and isinstance(idx, (list, tuple)):
            idx = [frame] + list(idx)
        assert (len(idx) == len(mask_shape)) or (len(idx) == (len(mask_shape) + 1)), (idx, mask_shape)
        idx_tensor = torch.tensor(idx, dtype=torch.long, device=mask.device)
        mask = mask.view(-1, *mask_shape)
        if idx_tensor.shape[0] == len(mask_shape):
            for b in range(shape[0]):
                _idx = torch.cat([
                    torch.tensor([b], device=idx_tensor.device, dtype=torch.long),
                    idx_tensor], 0)
                mask[list(_idx)] = 0
        elif idx_tensor.shape[0] == (1 + len(mask_shape)):
            mask[list(idx_tensor)] = torch.zeros_like(mask[list(idx_tensor)])
        mask = mask.view(*shape)
        return mask

    @staticmethod
    def patch_idx_list_from_mask(mask):
        assert len(mask.shape) == 4, mask.shape
        patch_idx_list = torch.where(~mask)
        patch_idx_list = [torch.stack([patch_idx_list[i][n] for i in range(4)], 0)
                          for n in range(len(patch_idx_list[0]))]
        return [list(p.cpu().numpy()) for p in patch_idx_list]

    @staticmethod
    def make_visible_from_patch_idx_list(
            mask, patch_idx_list, stride=1, b=0, t=-1):
        if len(patch_idx_list) == 0:
            return mask
        if not isinstance(patch_idx_list, torch.Tensor):
            patch_idx_list = torch.tensor(np.array(patch_idx_list), dtype=torch.long).to(mask.device)
        inds = torch.unbind(patch_idx_list, -1)
        inds_h = (inds[-2] // stride) % mask.size(-2)
        inds_w = (inds[-1] // stride) % mask.size(-1)
        if len(inds) == 2:
            inds_b = b * torch.ones_like(inds_h)
            inds_t = t * torch.ones_like(inds_b)
        elif len(inds) == 3:
            inds_b = b * torch.ones_like(inds_h)
            inds_t = inds[0]
        else:
            assert len(inds) == 4, len(inds)
            inds_b, inds_t = inds[0], inds[1]
            
        mask[inds_b, inds_t, inds_h, inds_w] = 0
        return mask

    def generate_mask_from_patch_idx_list(
            self, patch_idx_list, stride=None, b=0, frame=-1):
        assert self.x is not None
        m = self.get_mask_image(self.get_zeros_mask(frame=frame))
        if stride is None:
            stride = self.inp_shape[-1] // m.size(-1)
        m = self.make_visible_from_patch_idx_list(
            m, patch_idx_list, stride=stride, b=b, t=frame)
        return m.view(m.size(0), -1)

    def generate_cutout_mask(
            self, patch_idx_list, radius=1, stride=None, b=0, frame=-1):
        mask = self.generate_mask_from_patch_idx_list(
            patch_idx_list, stride=stride, b=b, frame=frame)
        mask = self.get_mask_image(mask)
        cutout = masking.patches_adjacent_to_visible(
            mask[:,frame:frame+1], radius=radius)
        cutout = torch.maximum(cutout, ~mask[:,frame:frame+1])
        mask[:,frame] = cutout[:,0]
        return mask.flatten(1)

    def shift_patches_and_mask(self, x, mask, max_shift_fraction=0.1, frame=-1, padding_mode='reflect'):
        H,W = x.shape[-2:]
        if len(x.shape) == 5:
            x = x[:,frame]
        max_shift = [int(max_shift_fraction * s) for s in [H,W]]
        random_shift = (
            self.rng.randint(-max_shift[0], max_shift[0]+1),
            self.rng.randint(-max_shift[1], max_shift[1]+1)
        )
        random_shift = (
            int((random_shift[0] // self.patch_size[-2]) * self.patch_size[-2]),
            int((random_shift[1] // self.patch_size[-1]) * self.patch_size[-1])
        )
        def _padding(p):
            sgn = np.sign(p)
            return (2*p,0) if (sgn > 0) else (0,-2*p)
        padding = _padding(random_shift[1]) + _padding(random_shift[0])
        x_shift = transforms.CenterCrop([H,W])(F.pad(x, padding, mode=padding_mode))

        mask = mask.view(x.shape[0], -1, H // self.patch_size[-2], W // self.patch_size[-1])
        if mask.size(1) > 1:
            mask = mask[:,frame].unsqueeze(1)
        mask_padding = \
            _padding(random_shift[1] // self.patch_size[-2]) + \
            _padding(random_shift[0] // self.patch_size[-1])
        mask_shift = transforms.CenterCrop(mask.shape[-2:])(
            F.pad(mask.float(), mask_padding, mode=padding_mode)).bool()

        return (x_shift, mask_shift)

    def get_frame_pairs(self, x, frame=None):
        """Get pairs of frames that will be inputs to the model."""
        assert len(x.shape) == 5, x.shape
        T = x.shape[1]
        self.num_frame_pairs = T - 1
        self.target_frame = frame if frame is not None else (T // 2)

        x_frames = torch.unbind(x, 1)
        x_pairs = [torch.stack([x_frames[t], x_frames[self.target_frame]], 1)
                   for t in range(T) if t != self.target_frame]
        return x_pairs

    def set_input(self, x, mask=None, make_mask=False, timestamps=None):
        shape = x.shape
        if len(shape) == 4:
            x = x.unsqueeze(1)
        else:
            assert len(shape) == 5, \
                "Input must be a movie of shape [B,T,C,H,W]" + \
                "or a single frame of shape [B,C,H,W]"
            
        self.inp_shape = x.shape
        self.x = x
        self.B = self.inp_shape[0]
        self.T = self.inp_shape[1]
        self.C = self.inp_shape[2]
        if mask is not None:
            self.mask = mask
        elif make_mask:
            assert self.mask_generator is not None, "You need to have a mask generator to set a new mask"
            self.set_new_mask(self.x)

        if timestamps is not None:
            self.timestamps = timestamps

    def get_static_input(self, x=None):
        if x is None:
            x = self.x
        return torch.tile(x[:,0:1], (1,x.size(1),1,1,1))

    def make_static_movie(self, x=None, T=None, frame=0):
        if x is None:
            x = self.x
        if T is None:
            T = getattr(self.predictor, 'num_frames', 2)
        if len(x.shape) == 4:
            x = x[:,None]
        assert len(x.shape) == 5, "x must be of shape [B,C,H,W] or [B,T,C,H,W], but is %s" % x.shape
        return torch.tile(x[:,frame%x.size(1),None], (1,T,1,1,1))

    def sample_random_masks(self, num_samples=10, num_visible=1, mask_ratio=None):
        assert self.mask_generator is not None
        _num_vis = self.mask_generator.num_visible
        if mask_ratio is None:
            self.mask_generator.num_visible = num_visible
        else:
            self.mask_generator.mask_ratio = mask_ratio

        if self.x is not None:
            x = self.x
        else:
            x = None
            
        masks = torch.stack([self.mask_generator(x) for _ in range(num_samples)], -1)
        if self.x is not None:
            masks = masks.to(x.device)
        self.mask_generator.num_visible = _num_vis
        return masks

    def predict_value_map(self, x, *args, **kwargs):
        assert len(x.shape) == 5, x.shape
        if self.value_predictor is None:
            return torch.ones_like(x[:,0:1,0:1])
        value = self.value_predictor(x, *args, **kwargs)
        return value

    def forward(self, x, mask=None, frame=None, *args, **kwargs):

        self.set_input(x, mask)
        if mask is None:
            self.mask = self.generate_mask(x)
        y = self.predict(self.x, self.mask, frame=frame, *args, **kwargs)
        return y

class MaeWrapper(PredictorBasedGenerator):
    """
    Wrapper for generating factual and counterfactual predictions from
    original MAE model class
    """

    @property
    def patch_size(self):
        if self.predictor is None:
            return None
        return (1,) + self.predictor.patch_embed.patch_size

    @property
    def mask_shape(self):
        if self.predictor is None:
            return None

        assert self.inp_shape is not None
        _, ph, pw = self.patch_size
        return (1, self.inp_shape[-2] // ph, self.inp_shape[-1] // pw)


    def set_predictor(self, net, predictor_load_path=None):
        self.predictor = net
        self.load_predictor(predictor_load_path, model=None)
        self.patchify = Patchify(
            patch_size=(1,)+net.patch_embed.patch_size,
            temporal_dim=1,
            squeeze_channel_dim=True
        )
        self.x = self.mask = self.inp_shape = None

    def imagenet_normalize(self, x):
        assert len(x.shape) == 4, x.shape
        assert x.size(1) == 3, x.shape
        m = torch.tensor(utils.IMAGENET_DEFAULT_MEAN)[None,:,None,None]
        s = torch.tensor(utils.IMAGENET_DEFAULT_STD)[None,:,None,None]
        return (x - m.to(x)) / s.to(x)

    def imagenet_unnormalize(self, x):
        assert len(x.shape) == 4, x.shape
        assert x.size(1) == 3, x.shape
        m = torch.tensor(utils.IMAGENET_DEFAULT_MEAN)[None,:,None,None]
        s = torch.tensor(utils.IMAGENET_DEFAULT_STD)[None,:,None,None]
        return (x * s.to(x)) + m.to(x)

    def predict(self, x=None, mask=None, frame=0, mask_ratio=0.9):
        if x is None:
            x = x

        is_video = False
        if len(x.shape) == 5:
            is_video = True
            x = x[:,frame]

        if self.imagenet_normalize_inputs:
            x = self.imagenet_normalize(x)

        _, y, mask = self.predictor(x, mask_ratio=mask_ratio, mask=mask)
        self.mask = mask
        y = self.predictor.unpatchify(y)


        if self.imagenet_normalize_inputs:
            y = self.imagenet_unnormalize(y)

        if is_video:
            y = y.unsqueeze(1)

        return y

