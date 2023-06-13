import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from cwm.models.patches import Patchify
from cwm.models.utils import sample_from_energy, coordinate_ims, boltzmann

def upsample_masks(masks, size, thresh=0.5):
    shape = masks.shape
    dtype = masks.dtype
    h,w = shape[-2:]
    H,W = size
    if (H == h) and (W == w):
        return masks
    elif (H < h) and (W < w):
        s = (h // H, w // W)
        return masks[...,::s[0],::s[1]]
        
    masks = masks.unsqueeze(-2).unsqueeze(-1)
    masks = masks.repeat(*([1] * (len(shape) - 2)), 1, H // h, 1, W // w)
    if ((H % h) == 0) and ((W % w) == 0):        
        masks = masks.view(*shape[:-2],H,W)
    else:
        _H = np.prod(masks.shape[-4:-2])
        _W = np.prod(masks.shape[-2:])
        masks = transforms.Resize(size)(masks.view(-1, 1, _H, _W)) > thresh
        masks = masks.view(*shape[:2], H, W).to(masks.dtype)
    return masks

def patch_distance_transform(masks, self_mask=True):
    """For each masked patch, L-inf distance to nearest visible patch"""
    B,T,H,W = masks.shape
    BT = B*T
    masks = masks.view(BT, H, W)
    dists = []
    for b in range(BT):
        _vis_inds = torch.where(~masks[b])
        N = _vis_inds[0].shape[0]
        if N == 0:
            dists.append(torch.zeros([H,W]).float().to(masks.device))
            continue
        vis_inds = torch.stack([torch.stack([_vis_inds[0][n], _vis_inds[1][n]], 0)
                                for n in range(N)], 0).float()
        dists_b = coordinate_ims(1, 0, [H,W], normalize=False).to(vis_inds.device)
        dists_b = (dists_b - vis_inds.view(N, 1, 1, 2))
        dists_b = dists_b / torch.tensor([(H-1)//2,(W-1)//2]).view(1,1,1,2).to(dists_b.device).float()
        dists_b = dists_b.abs().amax(-1).amin(0) # [H,W]
        if self_mask:
            dists_b[_vis_inds[0],_vis_inds[1]] = dists_b.amax()
        dists.append(dists_b)

    dists = torch.stack(dists, 0)
    dists = dists.view(B,T,H,W)
    return dists

def patches_adjacent_to_visible(masks, radius=1, size=None):
    if size is not None:
        masks = masks.view(-1, 1, *size)
    if radius is None:
        return masks
    H,W = masks.shape[-2:]
    dists = patch_distance_transform(masks)
    if radius != 0:
        r = (1 / ((min(H,W) - 1) // 2))
        adjacent = (dists <= (r * radius))
    else:
        rmax = dists.amax((-1,-2), keepdim=True)
        adjacent = (rmax - dists) / rmax.clip(min=1.0)
    return adjacent

def partition_masks(masks, num_samples=2, leave_one_out=False):
    B = masks.shape[0]
    S = num_samples
    masks = masks.view(B,-1)
    partitioned = [torch.ones_like(masks) for _ in range(S)]
    for b in range(B):
        vis_inds = torch.where(~masks[b])[0]
        vis_inds = vis_inds[torch.randperm(vis_inds.size(0))]
        if leave_one_out:
            for s in range(S):
                partitioned[s][b][vis_inds] = 0
                partitioned[s][b][vis_inds[s::S]] = 1
        else:
            for s in range(S):
                partitioned[s][b][vis_inds[s::S]] = 0
    return partitioned

class RectangularizeMasks(nn.Module):
    """Make sure all masks in a batch have same number of 1s and 0s"""
    def __init__(self, truncation_mode='min'):
        super().__init__()
        self._mode = truncation_mode
        assert self._mode in ['min', 'max', 'mean', 'full', 'none', None], (self._mode)

    def set_mode(self, mode):
        self._mode = mode

    def __call__(self, masks):

        if self._mode in ['none', None]:
            return masks
        
        assert isinstance(masks, torch.Tensor), type(masks)
        if self._mode == 'full':
            return torch.ones_like(masks)
            
        shape = masks.shape
        masks = masks.flatten(1)
        B,N = masks.shape
        num_masked = masks.float().sum(-1)
        M = {
            'min': torch.amin, 'max': torch.amax, 'mean': torch.mean
        }[self._mode](num_masked).long()
        
        num_changes = num_masked.long() - M

        for b in range(B):
            nc = num_changes[b]
            if nc > 0:
                inds = torch.where(masks[b])[0]
                inds = inds[torch.randperm(inds.size(0))[:nc].to(inds.device)]
                masks[b,inds] = 0
            elif nc < 0:
                inds = torch.where(~masks[b])[0]
                inds = inds[torch.randperm(inds.size(0))[:-nc].to(inds.device)]
                masks[b,inds] = 1
        if list(masks.shape) != list(shape):
            masks = masks.view(*shape)

        return masks


class UniformMaskingGenerator(object):
    def __init__(self, input_size, mask_ratio, seed=None, clumping_factor=1, randomize_num_visible=False):
        self.frames = None
        if len(input_size) == 3:
            self.frames, self.height, self.width = input_size
        elif len(input_size) == 2:
            self.height, self.width = input_size
        elif len(input_size) == 1 or isinstance(input_size, int):
            self.height = self.width = input_size

        self.clumping_factor = clumping_factor
        self.pad_h = self.height % self.c[0]
        self.pad_w = self.width % self.c[1]
        self.num_patches_per_frame = (self.height // self.c[0]) * (self.width // self.c[1])
        self.mask_ratio = mask_ratio

        self.rng = np.random.RandomState(seed=seed)
        self.randomize_num_visible = randomize_num_visible

    @property
    def num_masks_per_frame(self):
        if not hasattr(self, '_num_masks_per_frame'):
            self._num_masks_per_frame = int(self.mask_ratio * self.num_patches_per_frame)
        return self._num_masks_per_frame
    @num_masks_per_frame.setter
    def num_masks_per_frame(self, val):
        self._num_masks_per_frame = val
        self._mask_ratio = (val / self.num_patches_per_frame)
    @property
    def c(self):
        if isinstance(self.clumping_factor, int):
            return (self.clumping_factor, self.clumping_factor)
        else:
            return self.clumping_factor[:2]
        
    @property
    def mask_ratio(self):
        return self._mask_ratio
    @mask_ratio.setter
    def mask_ratio(self, val):
        self._mask_ratio = val
        self._num_masks_per_frame = int(self._mask_ratio * self.num_patches_per_frame)        
    @property
    def num_visible(self):
        return self.num_patches_per_frame - self.num_masks_per_frame
    @num_visible.setter
    def num_visible(self, val):
        self.num_masks_per_frame = self.num_patches_per_frame - val
        
    
    def __repr__(self):
        repr_str = "Mask: total patches per frame {}, mask patches per frame {}, mask ratio {}, random num num visible? {}".format(
            self.num_patches_per_frame, self.num_masks_per_frame, self.mask_ratio, self.randomize_num_visible
        )
        return repr_str

    def sample_mask_per_frame(self):
        num_masks = self.num_masks_per_frame
        if self.randomize_num_visible:
            num_masks = self.rng.randint(low=num_masks, high=(self.num_patches_per_frame+1))
        mask = np.hstack([
            np.zeros(self.num_patches_per_frame - num_masks),
            np.ones(num_masks)])
        self.rng.shuffle(mask)
        if max(*self.c) > 1:
            mask = mask.reshape(self.height // self.c[0],
                                1,
                                self.width // self.c[1],
                                1)
            mask = np.tile(mask, (1, self.c[0], 1, self.c[1]))
            mask = mask.reshape((self.height - self.pad_h, self.width - self.pad_w))
            _pad_h = self.rng.choice(range(self.pad_h+1))
            pad_h = (self.pad_h - _pad_h, _pad_h)
            _pad_w = self.rng.choice(range(self.pad_w+1))
            pad_w = (self.pad_w - _pad_w, _pad_w)
            mask = np.pad(mask,
                          (pad_h, pad_w),
                          constant_values=1
            ).reshape((self.height, self.width))
        return mask

    def __call__(self, num_frames=None):
        num_frames = (num_frames or self.frames) or 1
        masks = np.stack([self.sample_mask_per_frame() for _ in range(num_frames)]).flatten()
        return masks

class TubeMaskingGenerator(UniformMaskingGenerator):

    def __call__(self, num_frames=None):
        num_frames = (num_frames or self.frames) or 1
        masks = np.tile(self.sample_mask_per_frame(), (num_frames, 1)).flatten()
        return masks


class RotatedTableMaskingGenerator(TubeMaskingGenerator):

    def __init__(self, tube_length=None, *args, **kwargs):
        super(RotatedTableMaskingGenerator, self).__init__(*args, **kwargs)
        self.tube_length = tube_length

    def __call__(self, num_frames=None):
        num_frames = (num_frames or self.frames) or 2
        tube_length = self.tube_length or (num_frames - 1)
        table_thickness = num_frames - tube_length        
        assert tube_length < num_frames, (tube_length, num_frames)

        tubes = super().__call__(num_frames=tube_length)
        top = np.zeros(table_thickness * self.height * self.width).astype(tubes.dtype).flatten()
        masks = np.concatenate([top, tubes], 0)
        return masks

class PytorchMaskGeneratorWrapper(nn.Module):
    """Pytorch wrapper for numpy masking generators"""
    def __init__(self,
                 mask_generator=TubeMaskingGenerator,
                 *args, **kwargs):
        super().__init__()
        self.mask_generator = mask_generator(*args, **kwargs)

    @property
    def mask_ratio(self):
        return self.mask_generator.mask_ratio
    @mask_ratio.setter
    def mask_ratio(self, value):
        self.mask_generator.mask_ratio = value

    def forward(self, device='cuda', dtype_out=torch.bool, **kwargs):

        masks = self.mask_generator(**kwargs)
        masks = torch.tensor(masks).to(device).to(dtype_out)
        return masks

class MaskingGenerator(nn.Module):
    """Pytorch base class for masking generators"""
    def __init__(self,
                 input_size,
                 mask_ratio,
                 seed=0,
                 visible_frames=0,
                 clumping_factor=1,
                 randomize_num_visible=False,
                 create_on_cpu=True,
                 always_batch=False):
        super().__init__()
        self.frames = None
        if len(input_size) == 3:
            self.frames, self.height, self.width = input_size
        elif len(input_size) == 2:
            self.height, self.width = input_size
        elif len(input_size) == 1 or isinstance(input_size, int):
            self.height = self.width = input_size

        self.clumping_factor = clumping_factor
        self.pad_h = self.height % self.c[0]
        self.pad_w = self.width % self.c[1]
        self.num_patches_per_frame = (self.height // self.c[0]) * (self.width // self.c[1])        

        self.mask_ratio = mask_ratio
        self.visible_frames = visible_frames
        self.always_batch = always_batch
        self.create_on_cpu = create_on_cpu

        self.rng = np.random.RandomState(seed=seed)
        self._set_torch_seed(seed)

        self.randomize_num_visible = randomize_num_visible

    @property
    def num_masks_per_frame(self):
        if not hasattr(self, '_num_masks_per_frame'):
            self._num_masks_per_frame = int(self.mask_ratio * self.num_patches_per_frame)
        return self._num_masks_per_frame
    @num_masks_per_frame.setter
    def num_masks_per_frame(self, val):
        self._num_masks_per_frame = val
        self._mask_ratio = (val / self.num_patches_per_frame)
    @property
    def c(self):
        if isinstance(self.clumping_factor, int):
            return (self.clumping_factor, ) * 2
        else:
            return self.clumping_factor[:2]
    
    @property
    def mask_ratio(self):
        return self._mask_ratio
    @mask_ratio.setter
    def mask_ratio(self, val):
        self._mask_ratio = val
        self._num_masks_per_frame = int(self._mask_ratio * self.num_patches_per_frame)

    @property
    def num_visible(self):
        return self.num_patches_per_frame - self.num_masks_per_frame
    @num_visible.setter
    def num_visible(self, val):
        self.num_masks_per_frame = self.num_patches_per_frame - val

    def _set_torch_seed(self, seed):
        self.seed = seed
        torch.manual_seed(self.seed)

    def __repr__(self):
        repr_str = ("Class: {}\nMask: total patches per mask {},\n" +\
                    "mask patches per mask {}, visible patches per mask {}, mask ratio {:0.3f}\n" + \
                    "randomize num visible? {}").format(
            type(self).__name__, self.num_patches_per_frame,
                        self.num_masks_per_frame, self.num_visible, self.mask_ratio,
                        self.randomize_num_visible
                    )
        return repr_str

    def sample_mask_per_frame(self, *args, **kwargs):
        num_masks = self.num_masks_per_frame
        if self.randomize_num_visible:
            num_masks = self.rng.randint(low=num_masks, high=(self.num_patches_per_frame+1))
        
        mask = torch.cat([
            torch.zeros([self.num_patches_per_frame - num_masks]),
            torch.ones([num_masks])], 0).bool()
        inds = torch.randperm(mask.size(0)).long()
        mask = mask[inds]

        if max(*self.c) > 1:
            mask = mask.view(self.height // self.c[0],
                                1,
                                self.width // self.c[1],
                                1)
            mask = torch.tile(mask, (1, self.c[0], 1, self.c[1]))
            mask = mask.reshape(self.height - self.pad_h, self.width - self.pad_w)
            _pad_h = self.rng.choice(range(self.pad_h+1))
            pad_h = (self.pad_h - _pad_h, _pad_h)
            _pad_w = self.rng.choice(range(self.pad_w+1))
            pad_w = (self.pad_w - _pad_w, _pad_w)
            mask = F.pad(mask,
                         pad_w + pad_h,
                         mode='constant',
                         value=1)
            mask = mask.reshape(self.height, self.width)
        
        return mask

    

    def forward(self, x=None, num_frames=None):
        num_frames = (num_frames or self.frames) or 1
        if isinstance(x, torch.Tensor):
            batch_size = x.size(0)
            masks = torch.stack([
                torch.cat([self.sample_mask_per_frame() for _ in range(num_frames)], 0).flatten()
                for b in range(batch_size)], 0)
            if not self.create_on_cpu:
                masks = masks.to(x.device)
            if batch_size == 1 and not self.always_batch:
                masks = masks.squeeze(0)
        else:
            batch_size = 1
            masks = torch.cat([self.sample_mask_per_frame() for _ in range(num_frames)], 0).flatten()
            if self.always_batch:
                masks = masks[None]

        if self.visible_frames > 0:
            vis = torch.zeros((batch_size, 1, self.height, self.width), dtype=torch.bool)
            vis = vis.view(masks.shape).to(masks.device)
            masks = torch.cat(([vis]*self.visible_frames) + [masks], -1)
            
        return masks

class FullMaskGenerator(MaskingGenerator):
    """Outputs fully masked (or fully visible) with probability full_mask_prob, full_vis_prob"""
    def __init__(self,
                 full_mask_prob=0.2,
                 full_vis_prob=0.0,
                 always_batch=True,
                 full_mask_per_example=False,                 
                 *args,
                 **kwargs):
        super().__init__(always_batch=always_batch, *args, **kwargs)
        self.full_mask_prob = min(max(full_mask_prob, 0), 1)
        self.full_vis_prob = min(max(full_vis_prob, 0), 1)
        self.partial_prob = max(1 - self.full_mask_prob - self.full_vis_prob, 0)
        self._final_full_mask_prob = self.full_mask_prob / max(self.full_mask_prob + self.full_vis_prob, 1e-6)
        self._per_sample = full_mask_per_example        

    def forward(self, x=None, num_frames=None):
        masks = super().forward(x=x, num_frames=num_frames)
        if not self._per_sample:
            if self.rng.rand() < self.partial_prob:
                return masks
            elif self.rng.rand() < self._final_full_mask_prob:
                return torch.ones_like(masks)
            else:
                return torch.zeros_like(masks)
        elif self._per_sample:
            fully_masked = (torch.rand((masks.size(0), 1)).to(masks.device) < \
                            self.full_mask_prob).expand(-1,masks.size(-1))
            masks = torch.maximum(masks, fully_masked)
            return masks

class ImuFullMaskGenerator(FullMaskGenerator):
    def __init__(self,
                 input_size=10,
                 clumping_factor=1,
                 *args,
                 **kwargs):
        if not isinstance(input_size, int):
            input_size = int(np.prod(input_size))
        assert isinstance(input_size, int), "input size must be a single int for Imu mask, %s" % input_size
        super().__init__(input_size=(1,1,input_size),
                         clumping_factor=(1, clumping_factor),
                         *args,
                         **kwargs)

class MissingDataImuMaskGenerator(ImuFullMaskGenerator):
    def __init__(self,
                 truncation_mode='max',
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.rect = RectangularizeMasks(truncation_mode)

    def set_mode(self, mode):
        self.rect.set_mode(mode)

    @property
    def mode(self):
        return self.rect._mode

    def forward(self, missing=None):
        masks = super().forward(x=missing)
        if missing is None:
            return masks
        
        assert missing.dtype == torch.bool, missing.dtype                
        assert list(missing.shape) == list(masks.shape), (missing.shape, masks.shape)

        ## no missing data or ignore it        
        if self.mode in ['none', None]:
            return torch.maximum(masks, missing.to(masks.device))
        
        ## else make sure all masks have equal number of masked tokens, includding missing ones
        return self.rect(torch.maximum(masks, missing.to(masks.device)))        

class RotatedTableUniformMaskingGenerator(MaskingGenerator):

    def __init__(self,
                 input_size,
                 mask_ratio,
                 visible_frames=None,
                 context_mask_ratio=None,
                 seed=0,
                 clumping_factor=1,
                 always_batch=True,
                 randomize_num_visible=False,
                 full_mask_prob=0):

        assert len(input_size) == 3, input_size
        if visible_frames is None:
            visible_frames = input_size[0] - 1
        super().__init__(input_size=(input_size[0]-visible_frames, *input_size[1:]),
                         mask_ratio=mask_ratio,
                         visible_frames=visible_frames,
                         seed=seed,
                         clumping_factor=clumping_factor,
                         always_batch=always_batch,
                         randomize_num_visible=randomize_num_visible)
        self.visible_frames = visible_frames
        self.full_mask_prob = full_mask_prob

        if context_mask_ratio is not None:
            self.context_mask_ratio = context_mask_ratio
            self.vis_frame_sampler = MaskingGenerator(
                input_size=(1, self.height, self.width),
                mask_ratio=context_mask_ratio,
                visible_frames=0,
                clumping_factor=1,
                create_on_cpu=self.create_on_cpu,
                always_batch=self.always_batch)
        else:
            self.context_mask_ratio = 0
            self.vis_frame_sampler = None

    def __repr__(self):
        return super().__repr__() + "\nVisible frames: {} Context Mask Ratio: {}".format(
            self.visible_frames, self.context_mask_ratio)

    def get_visible_frame_masks(self, x=None):
        vis = torch.cat([self.vis_frame_sampler(x) for _ in range(self.visible_frames)], -1)
        return vis

    def vis_mask(self, x=None):
        B = 1
        if x is not None:
            B = x.size(0)
        vis = torch.zeros((B, 1, self.height, self.width), dtype=torch.bool)
        vis = vis.view(B,-1)
        vis = torch.cat([vis]*self.visible_frames, -1)
        return vis

    def forward(self, x=None, *args, **kwargs):
        masks = super().forward(x=x, *args, **kwargs)
        if self.full_mask_prob > 0:
            vis = self.vis_mask(x).to(masks.device)
            fully_masked = (torch.rand((masks.size(0), 1)).to(masks.device) < \
                            self.full_mask_prob).expand(-1, masks.size(-1) - vis.size(-1))
            masks = torch.maximum(masks, torch.cat([vis, fully_masked], -1))
        if self.vis_frame_sampler is not None:
            context_masks = self.get_visible_frame_masks(x=x)
            masks = masks.view(masks.size(0), self.frames, -1)[:,self.visible_frames:,:]
            masks = torch.cat([context_masks, masks.view(masks.size(0), -1)], -1)
        return masks

class ForwardBackwardRotatedTableMaskingGenerator(RotatedTableUniformMaskingGenerator):

    def __init__(self,
                 input_size,
                 mask_ratio,
                 context_mask_ratio=None,
                 visible_frames=None,
                 always_batch=True,
                 randomize_num_visible=False,
                 seed=0,
                 clumping_factor=1,
                 flip_prob=0.5,
                 exact_flip_ratio=False,
                 split_masked_patches=False,
                 *args, **kwargs):
        super().__init__(input_size=input_size,
                         mask_ratio=mask_ratio,
                         visible_frames=visible_frames,
                         always_batch=always_batch,
                         randomize_num_visible=randomize_num_visible,
                         seed=seed,
                         clumping_factor=clumping_factor,
                         *args, **kwargs)

        self.vis_frame_sampler = MaskingGenerator(
            input_size=(1, self.height, self.width),
            mask_ratio=context_mask_ratio or (1-self.mask_ratio),
            visible_frames=0,
            clumping_factor=clumping_factor,
            create_on_cpu=self.create_on_cpu,
            always_batch=self.always_batch)
        if context_mask_ratio is None:
            self.vis_frame_sampler.num_visible = self.num_patches_per_frame - \
                (self.num_visible if split_masked_patches else 0)

        self.flip_prob = flip_prob
        self.exact_flip_ratio = exact_flip_ratio


    def forward(self, x=None, *args, **kwargs):
        masks = super().forward(x=x, *args, **kwargs)
        B = masks.size(0)
        masks = masks.view(B,
                           -1,
                           self.num_patches_per_frame)[:,self.visible_frames:]
        masks = masks.view(B, -1) # masked frames
        vis_frames = self.get_visible_frame_masks(x=x).to(masks.device)

        forward = torch.cat([vis_frames, masks], -1)
        backward = torch.cat([masks, vis_frames], -1)
        cond = (torch.arange(B)[:,None] < (self.flip_prob * B)) if self.exact_flip_ratio \
            else (torch.rand(B, 1) < self.flip_prob)
            
        return torch.where(
            cond.to(masks.device), backward, forward)
        
class MixedMaskGenerator(nn.Module):
    """Combine multiple mask generators"""
    def __init__(self,
                 mask_generator_list,
                 mask_ratio_list=None):
        super().__init__()
        self.mask_generator_list = mask_generator_list
        self.mask_ratio_list = mask_ratio_list

    @property
    def mask_ratio_list(self):
        self._mask_ratio_list = [mg.mask_ratio for mg in self.mask_generator_list]
        return self._mask_ratio_list
    @mask_ratio_list.setter
    def mask_ratio_list(self, r_list):
        if r_list is None:
            return
        self._mask_ratio_list = r_list
        for i, ratio in enumerate(self._mask_ratio_list):
            setattr(self.mask_generator_list[i], 'mask_ratio', ratio)

    def forward(self, *args, **kwargs):

        masks = []
        for gen in self.mask_generator_list:
            masks.append(gen(*args, **kwargs))
        masks = torch.stack(masks, -1).amin(-1) # min over masks to unmask up to sum(ratios)
        return masks

class MixedClumpingMaskingGenerator(nn.Module):

    def __init__(self,
                 input_size,
                 mask_ratio,
                 clumping_factor=(1,2,4,8),
                 seed=0,
                 mask_generator_func=MaskingGenerator,
                 **kwargs):
        super().__init__()
        if isinstance(clumping_factor, int):
            clumping_factor = (clumping_factor,)

        self.clumping_factor = clumping_factor
        self.rng = np.random.RandomState(seed=seed)
        torch.manual_seed(seed)
        self.mask_generator = nn.ModuleList([
            mask_generator_func(input_size=input_size,
                                mask_ratio=mask_ratio,
                                seed=seed,
                                clumping_factor=c,
                                **kwargs)
            for c in self.clumping_factor])
        for mgen in self.mask_generator:
            mgen.always_batch = True
        
    def forward(self, x=None, **kwargs):
        if x is None:
            return self.rng.choice(self.mask_generator)(x, **kwargs)

        B = x.size(0)
        masks = []
        for b in range(B):
            mask_gen = self.rng.choice(self.mask_generator)
            masks.append(mask_gen(x[b:b+1], **kwargs))
        return torch.cat(masks, 0)

class RotatedTableMixedClumpingMaskingGenerator(MixedClumpingMaskingGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            mask_generator_func=RotatedTableUniformMaskingGenerator,
            **kwargs)


class ThreeFrameForwardBackwardMasking(nn.Module):
    """Pytorch base class for masking generators"""
    def __init__(self, input_size, mask_ratio, independent_samples=True, fully_visible=False, seed=0):
        super().__init__()
        self.frames = None
        if len(input_size) == 3:
            self.frames, self.height, self.width = input_size
            assert self.frames==3
        elif len(input_size) == 2:
            self.height, self.width = input_size
            self.frames=3
        elif len(input_size) == 1 or isinstance(input_size, int):
            self.height = self.width = input_size
            self.frames=3

        self.num_patches_per_frame = self.height * self.width
        self.mask_ratio = mask_ratio

        self.num_frames = self.frames

        self._set_torch_seed(seed)
        
        self.independent_samples = independent_samples
        self.fully_visible = fully_visible
        
    def _set_torch_seed(self, seed):
        self.seed = seed
        torch.manual_seed(self.seed)

    def sample_mask_per_frame(self, num_masks_this_frame):
        
        mask = torch.cat([
            torch.zeros([self.num_patches_per_frame - num_masks_this_frame]),
            torch.ones([num_masks_this_frame])], 0).bool()        
        inds = torch.randperm(mask.size(0)).long()        
        mask = mask[inds]
        return mask

    def forward(self, x=None):
       
        if isinstance(x, torch.Tensor):
            batch_size = x.size(0)
            if self.independent_samples is True:
                frame_1_mask_p = torch.rand(batch_size)
            else:
                frame_1_mask_p = torch.rand(1).repeat(batch_size)
                
            frame_1_num_masks = (frame_1_mask_p * self.num_patches_per_frame).to(torch.int)
            
                
            frame_2_num_masks = torch.ones_like(frame_1_num_masks)*int(self.mask_ratio * self.num_patches_per_frame)
            frame_3_num_masks = self.num_patches_per_frame-frame_1_num_masks

            if self.fully_visible:
                frame_1_num_masks = frame_3_num_masks = torch.zeros_like(frame_2_num_masks)
            

            frame_num_masks = [frame_1_num_masks, frame_2_num_masks, frame_3_num_masks]               
            masks = torch.stack([
                torch.cat([self.sample_mask_per_frame(frame_num_masks[frameidx][b]) for frameidx in range(self.num_frames)], 0).flatten()
                for b in range(batch_size)], 0)
            masks = masks.to(x.device)
            if batch_size == 1:
                masks = masks.squeeze(0)
        else:
            frame_1_mask_p = torch.rand(1)
            frame_2_mask_p = self.mask_ratio
            frame_3_mask_p = 1-frame_1_mask_p

            mask_ps = [frame_1_mask_p, frame_2_mask_p, frame_3_mask_p]             
            masks = torch.cat([self.sample_mask_per_frame(mask_ps[frameidx]) for frameidx in range(self.num_frames)], 0).flatten()
        return masks






class TwoFrameForwardBackwardMasking(nn.Module):
    """Pytorch base class for masking generators"""
    def __init__(self, input_size, mask_ratio, independent_samples=True, force_normal_rot_table=True, seed=0):
        super().__init__()
        self.frames = None
        if len(input_size) == 3:
            self.frames, self.height, self.width = input_size
            assert self.frames==2
        elif len(input_size) == 2:
            self.height, self.width = input_size
            self.frames=2
        elif len(input_size) == 1 or isinstance(input_size, int):
            self.height = self.width = input_size
            self.frames=2

        self.num_patches_per_frame = self.height * self.width
        self.mask_ratio = mask_ratio

        self.num_frames = self.frames

        self._set_torch_seed(seed)
        
        self.independent_samples = independent_samples
        self.force_normal_rot_table = force_normal_rot_table

        import importlib.util
        import sys, os
        spec = importlib.util.spec_from_file_location(
            "masking_generator",
            os.path.expanduser("~/BBNet/bbnet/models/VideoMAE-main/masking_generator.py"))
        self.mask_module = importlib.util.module_from_spec(spec)
        sys.modules["masking_generator"] = self.mask_module
        spec.loader.exec_module(self.mask_module)

        self.rotated_table_gen = self.mask_module.RotatedTableMaskingGenerator(
                input_size, mask_ratio, 1)
        
        self.time_reverse_rotated_table_gen = self.mask_module.TimeReverseRotatedTableMaskingGenerator(
                input_size, mask_ratio, 1)

        
        
    def _set_torch_seed(self, seed):
        self.seed = seed
        torch.manual_seed(self.seed)

    def sample_mask_per_frame(self, num_masks_this_frame):
        
        mask = torch.cat([
            torch.zeros([self.num_patches_per_frame - num_masks_this_frame]),
            torch.ones([num_masks_this_frame])], 0).bool()        
        inds = torch.randperm(mask.size(0)).long()        
        mask = mask[inds]
        return mask


    def forward(self, x=None):

        # With 50% chance, we want to use the forward-backward masking strategy. Otherwise, we want to return either the regular rotated table, or the time-reverse rotated table
        
       
        if isinstance(x, torch.Tensor):
            batch_size = x.size(0)
            if self.independent_samples is True:
                if self.force_normal_rot_table is False:
                    frame_1_mask_p = torch.rand(batch_size)
                
                    # Take care of 50% probability that it's either rot. table or time reversed
                    frame_1_mask_p[batch_size//2: batch_size//2+batch_size//4] = self.mask_ratio
                    frame_1_mask_p[batch_size//2+batch_size//4:] = 1-self.mask_ratio
                elif self.force_normal_rot_table is True:
                    frame_1_mask_p = torch.rand(batch_size)                    

                    frame_1_mask_p[:batch_size//2] = self.mask_ratio
                    frame_1_mask_p[batch_size//2:] = 1-self.mask_ratio
                    # print(frame_1_mask_p)
                    
            else:
                frame_1_mask_p = torch.rand(1).repeat(batch_size)
                
            frame_1_num_masks = (frame_1_mask_p * self.num_patches_per_frame).to(torch.int)

            frame_2_num_masks = self.num_patches_per_frame-frame_1_num_masks

            frame_num_masks = [frame_1_num_masks, frame_2_num_masks]

            masks = []
            
            for b in range(batch_size):                
                '''if np.random.uniform() < 0.5:
                    # We're going to do either rot. table or time-reverse
                    if np.random.uniform() < 0.5:
                        mask_gen = self.rotated_table_gen
                    else:
                        mask_gen = self.time_reverse_rotated_table_gen
                    b_masks = torch.tensor(mask_gen())'''
                
                b_masks = torch.cat([self.sample_mask_per_frame(frame_num_masks[frameidx][b]) for frameidx in range(self.num_frames)], 0).flatten()

                masks.append(b_masks)
                
            masks = torch.stack(masks, 0)
            masks = masks.to(x.device)
            if batch_size == 1:
                masks = masks.squeeze(0)

        else:

            frame_1_mask_p = torch.rand(1)
            frame_2_mask_p = 1-frame_1_mask_p

            mask_ps = [frame_1_mask_p, frame_2_mask_p]             
            masks = torch.cat([self.sample_mask_per_frame(mask_ps[frameidx]) for frameidx in range(self.num_frames)], 0).flatten()

        return masks    
