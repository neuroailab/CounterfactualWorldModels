import numpy as np
import copy
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
import kornia

from cwm.models.patches import Patchify
import cwm.models.utils as utils
import cwm.models.masking as masking

class PatchPerturbation(nn.Module):

    def __init__(self,
                 patch_size,
                 seed=0,
                 frame=None,
                 use_image_coordinates=True,
                 **kwargs):
        super().__init__()
        self.patch_size = patch_size
        self.patchify = Patchify(self.patch_size, temporal_dim=1, squeeze_channel_dim=True)
        self.frame = frame

        self.seed = seed
        self.rng = np.random.RandomState(seed=seed)
        self.torch_rng = torch.manual_seed(seed)

        self.use_image_coordinates = use_image_coordinates

    @property
    def T(self):
        return self.sequence_length

    @property
    def C(self):
        return self.num_channels

    @property
    def H(self):
        return self.image_size[0]

    @property
    def W(self):
        return self.image_size[1]

    @property
    def mask_shape(self):
        return (self.sequence_length // self.patch_size[0],
                self.image_size[0] // self.patch_size[1],
                self.image_size[1] // self.patch_size[2])

    @property
    def mask_image_size(self):
        return self.mask_shape[-2:]

    def _check_shapes(self, x, mask):
        if mask is not None:
            self.inp_mask_shape = mask.shape

    def set_shapes(self, x, mask):
        assert len(x.shape) == 5, x.shape
        self.inp_shape = x.shape
        self.B = self.inp_shape[0]        
        self.image_size = self.inp_shape[-2:]
        self.sequence_length = self.inp_shape[1]
        self.num_channels = self.inp_shape[2]

        self.num_patches = np.prod(self.mask_shape)

        self._check_shapes(x, mask)

    def reshape_mask_to_video(self, mask):
        mask = mask.view(self.B, -1, *self.mask_image_size)
        self.T_mask = mask.size(1)
        return mask

    def sample_random_patch(self, batch_size, frames=[0]):
        patch_idx_list = []
        if frames is None:
            frames = list(range(self.T))
        elif not isinstance(frames, (list, tuple)):
            frames = [frames]

        for b_idx in range(batch_size):
            t_idx = self.rng.choice(frames)
            h_idx = self.rng.randint(self.mask_image_size[0])
            w_idx = self.rng.randint(self.mask_image_size[1])
            patch_idx_list.append([b_idx, t_idx, h_idx, w_idx])
        return patch_idx_list

    def image_to_patch_inds(self, inds):
        return [inds[-i] // self.patch_size[-i] for i in range(1, len(inds)+1)]

    def perturb(self, x, mask, **kwargs):
        raise NotImplementedError("Do the perturbation")

    def forward(self, x, mask=None, perturbation_points=None, **kwargs):

        self.set_shapes(x, mask)
        mask = mask.clone()
        if perturbation_points is None:
            perturbation_mask = mask
        else: # remove the visible patches in common between mask and perturbation mask
            mask[perturbation_points] = 1
            perturbation_mask = torch.logical_not(perturbation_points)
        x_perturbed, mask_perturbed = self.perturb(x, perturbation_mask, **kwargs)

        if perturbation_points is not None:
            mask_perturbed = torch.minimum(mask, mask_perturbed)
        
        return x_perturbed, mask_perturbed

class NullPerturbation(PatchPerturbation):

    def perturb(self, x, mask, **kwargs):
        return (x, mask)

class MakeStatic(PatchPerturbation):
    """
    Make the visible patches in frames t > 0 identical to
    the spatially equivalent patches in frame t = 0
    """

    def _check_shapes(self, x, mask):
        assert self.T > 1

    def perturb(self, x, mask):
        m = self.reshape_mask_to_video(mask)[...,None]
        x = self.patchify(x).view(self.B, self.T, *self.mask_image_size, -1)

        ## assume all other frames are masked, so they won't be altered
        if self.T_mask != self.T:
            T_vis = self.T - self.T_mask
            m_vis = torch.ones((self.B, T_vis, *self.mask_image_size, 1)).to(mask).to(mask.device)
            m = torch.cat([m_vis, m[:,-1:]], 1)
        m = m.to(x.dtype)

        x_static = (1 - m) * x[:,0:1] + m * x
        x_static = x_static.view(self.B, self.num_patches, -1, self.C)
        x_static = self.patchify(x_static, to_video=True)

        ## original mask is unaltered
        return (x_static, mask)

class ShiftPatchesAndMask(PatchPerturbation):
    """
    Shift the visible patches and mask in a target frame by some 2D vector
    """

    def __init__(self,
                 patch_size,
                 max_shift_fraction=0.15,
                 padding_mode='constant',
                 allow_fractional_shifts=False,
                 **kwargs):
        super().__init__(patch_size, **kwargs)

        self.max_shift_fraction = max_shift_fraction
        self.padding_mode = padding_mode
        self.allow_fractional_shifts = allow_fractional_shifts

    def _check_shapes(self, x, mask):
        self.inp_mask_shape = mask.shape

    def get_random_shift(self):

        rect = (lambda s,p: int((s // p) * p)) if not self.allow_fractional_shifts else (lambda s,p: s)
        max_shift = [int(self.max_shift_fraction * s) for s in self.image_size]
        random_shift = (0,0)
        while sum(random_shift) == 0:
            random_shift = (
                rect(self.rng.randint(-max_shift[0], max_shift[0] + 1), self.patch_size[-2]),
                rect(self.rng.randint(-max_shift[1], max_shift[1] + 1), self.patch_size[-1])
            )
        return random_shift

    def _get_padding(self, shift):

        def _padding(p):
            sgn = np.sign(p)
            return (2*p,0) if (sgn > 0) else (0,-2*p)

        padding = _padding(shift[1]) + _padding(shift[0])
        if not self.allow_fractional_shifts:
            mask_padding = \
                _padding(shift[1] // self.patch_size[-2]) + \
                _padding(shift[0] // self.patch_size[-1])
        else:
            mask_padding = \
                _padding(np.sign(shift[1]) * (np.abs(shift[1]) // self.patch_size[-2])) + \
                _padding(np.sign(shift[0]) * (np.abs(shift[0]) // self.patch_size[-1]))
            
        return (padding, mask_padding)

    def perturb(self, x, mask, shift=None, mask_shift=None, frame=-1):

        frame = (frame % self.T)
        if shift is not None:
            assert len(shift) == 2, shift
            if not self.allow_fractional_shifts:
                assert (shift[0] % self.patch_size[-2]) == 0, shift
                assert (shift[1] % self.patch_size[-1]) == 0, shift
        elif mask_shift is not None:
            assert len(mask_shift) == 2, mask_shift
            shift = (mask_shift[0] * self.patch_size[-2], mask_shift[1] * self.patch_size[-1])
        else:
            shift = self.get_random_shift()
        # print("shifting images and masks by (%d, %d)" % (shift[0], shift[1]))
        self.shift = shift

        padding, mask_padding = self._get_padding(shift)

        x_shift = x[:,frame]
        x_shift = transforms.CenterCrop([self.H,self.W])(F.pad(x_shift, padding, mode=self.padding_mode, value=0))

        mask = self.reshape_mask_to_video(mask)
        m = mask[:,frame].unsqueeze(1) if (self.T_mask > 1) else mask

        mask_shift = transforms.CenterCrop(self.mask_image_size)(
            F.pad(m.to(x.dtype), mask_padding, mode=self.padding_mode, value=1)).bool()
        mask_shift = torch.cat([
            mask[:,:frame], mask_shift, mask[:,(frame+1):]], 1)

        ## only shift visible patches in target frame
        x_shift = torch.cat([
            x[:,:frame], x_shift.unsqueeze(1), x[:,(frame+1):]], 1)
        x = self.patchify(x).view(self.B, self.T, *self.mask_image_size, -1)
        x_shift = self.patchify(x_shift).view(self.B, self.T, *self.mask_image_size, -1)
        m = mask_shift[:,frame].unsqueeze(-1).to(x.dtype)
        x_shift = torch.cat([
            x_shift[:,:frame],
            (x_shift[:,frame] * (1 - m) + x[:,frame] * m).unsqueeze(1),
            x_shift[:,(frame+1):]
        ], 1)
        x_shift = self.patchify(x_shift.view(self.B, self.num_patches, -1, self.C), to_video=True)

        mask_shift = mask_shift.view(*self.inp_mask_shape)            
        
        return (x_shift, mask_shift)

class ShiftPatches(ShiftPatchesAndMask):
    """Only shift the patches"""

    def perturb(self, x, mask, shift=None, mask_shift=None, frame=-1):

        frame = (frame % self.T)
        if shift is not None:
            assert len(shift) == 2, shift
            assert (shift[0] % self.patch_size[-2]) == 0, shift
            assert (shift[1] % self.patch_size[-1]) == 0, shift
        elif mask_shift is not None:
            assert len(mask_shift) == 2, mask_shift
            shift = (mask_shift[0] * self.patch_size[-2], mask_shift[1] * self.patch_size[-1])
        else:
            shift = self.get_random_shift()

        padding, mask_padding = self._get_padding(shift)

        x_shift = x[:,frame]
        x_shift = transforms.CenterCrop([self.H,self.W])(F.pad(x_shift, padding, mode=self.padding_mode))

        m = self.reshape_mask_to_video(mask)

        ## only shift visible patches in target frame
        x_shift = torch.cat([
            x[:,:frame], x_shift.unsqueeze(1), x[:,(frame+1):]], 1)
        x = self.patchify(x).view(self.B, self.T, *self.mask_image_size, -1)
        x_shift = self.patchify(x_shift).view(self.B, self.T, *self.mask_image_size, -1)
        m = m[:,frame].unsqueeze(-1).to(x.dtype)
        x_shift = torch.cat([
            x_shift[:,:frame],
            (x_shift[:,frame] * (1 - m) + x[:,frame] * m).unsqueeze(1),
            x_shift[:,(frame+1):]
        ], 1)
        x_shift = self.patchify(x_shift.view(self.B, self.num_patches, -1, self.C), to_video=True)
        
        return (x_shift, mask)    

class MarkerShape:
    shapes = ['full', 'cross']
    def __init__(self, size):
        self.size = size

    def _cross(self):
        cross = np.zeros(self.size)
        _is_center = lambda k,s: (abs((k - ((s-1) / 2))) < 1.0)
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                if _is_center(i, self.size[0]) or _is_center(j, self.size[1]):
                    cross[i,j] = 1

        return cross
        
    def _get_shape(self, shape='full'):
        if shape == 'full':
            return np.ones(self.size)
        elif shape == 'cross':
            return self._cross()
            
    def __call__(self, shape='full'):
        if shape not in self.shapes:
            raise NotImplementedError(
                "%s not in set of shapes %s" % (shape, self.shapes))
        return self._get_shape(shape)

class AddMarkers(PatchPerturbation):
    """
    Add markers to visible patches in an input video
    """
    def __init__(self,
                 patch_size,
                 marker_shapes=['full'],
                 marker_color=[1,0,0],
                 **kwargs):
        super().__init__(patch_size, **kwargs)
        self.marker_shapes = marker_shapes
        self.marker_func = MarkerShape(size=self.patch_size[-2:])
        self.marker_color = marker_color

    def _check_shapes(self, x, mask):
        self.inp_mask_shape = mask.shape

    def _get_marker_color(self):

        ## random color
        if self.marker_color is None:
            return self.rng.random((3))
        elif isinstance(self.marker_color[0], (list, tuple, np.ndarray)):
            return np.array((self.marker_color)[self.rng.choice(range(len(self.marker_color)))])
        else:
            assert len(self.marker_color) == self.C
            return np.array(self.marker_color)

    def sample_marker(self, x_patch):
        shape = self.rng.choice(self.marker_shapes)
        marker_shape = self.marker_func(shape) # [ph, pw]
        marker_color = self._get_marker_color()
        marker = marker_shape[None] * marker_color[:,None,None]
        return torch.tensor(marker, device=x_patch.device, dtype=x_patch.dtype)

    def _patch_to_image_inds(self, inds):
        assert len(inds) >= 2, inds
        w_inds = [inds[-1]*self.patch_size[-1], (inds[-1]+1)*self.patch_size[-1]]
        h_inds = [inds[-2]*self.patch_size[-2], (inds[-2]+1)*self.patch_size[-2]]

        if len(inds) == 2:
            b_inds, t_inds = 0, 0
        if len(inds) == 3:
            b_inds, t_inds = inds[0], 0
        elif len(inds) == 4:
            b_inds, t_inds = inds[0], inds[1]

        return (b_inds, t_inds, h_inds, w_inds)

    def add_marker_to_patch(self, img, marker, patch_idx):

        b, t, h, w = self._patch_to_image_inds(patch_idx)
        m = (marker.sum(0, True) > 0).to(img.dtype)
        marker = m*marker + (1-m)*img[b,t,:,h[0]:h[1],w[0]:w[1]]
        img[b,t,:,h[0]:h[1],w[0]:w[1]] = marker
        return img

    def perturb(self, x, mask, patch_idx_list=None, frame=0, num_random_patches=0):
        orig_mask = mask.clone()
        mask = torch.ones((x.size(0), *self.mask_shape)).bool().to(x.device)

        if frame is not None:
            frame = (frame % self.T)        
            m = mask[:,frame].unsqueeze(1)
        else:
            m = mask

        ## if patch_idx is None, perturb everywhere visible in target frame
        if (patch_idx_list is None) and (not bool(num_random_patches)):
            patch_idx_list = torch.where(torch.logical_not(m))
            patch_idx_list = [torch.stack([patch_idx_list[i][n] for i in range(4)], 0)
                              for n in range(len(patch_idx_list[0]))]
        elif bool(num_random_patches):
            patch_idx_list = []
            for p in range(num_random_patches):
                patch_idx_list.extend(
                    self.sample_random_patch(x.shape[0],
                    frames=(0 if frame is not None else None)
                    ))

            patch_idx_list = [torch.tensor(p, device=x.device, dtype=torch.long)
                              for p in patch_idx_list]
        else:
            assert hasattr(patch_idx_list, '__len__')
            if not isinstance(patch_idx_list[0], (list, tuple, torch.Tensor, np.ndarray)):
                patch_idx_list = [patch_idx_list]

        assert all((len(patch_idx_list[n]) in (2,3,4) for n in range(len(patch_idx_list))))

        ## add markers to those patches
        x_marked = x[:,frame].unsqueeze(1).clone() if (frame is not None) else x.clone()
        for p, patch_idx in enumerate(patch_idx_list):
            marker = self.sample_marker(x)
            x_marked = self.add_marker_to_patch(x_marked, marker, patch_idx)

        if frame is not None:
            x_marked = torch.cat([x[:,:frame], x_marked, x[:,(frame+1):]], 1)

        ## unmask any perturbed patches
        for p in patch_idx_list:
            m[p[0],p[1],p[2],p[3]] = 0

        if frame is not None:
            mask_marked = torch.cat([
                mask[:,:frame], m, mask[:,(frame+1):]], 1)
        else:
            mask_marked = m
        mask_marked = mask_marked.view(-1,*self.mask_shape)
        stride = np.prod(self.mask_shape) / np.prod(orig_mask.shape[1:])
        if stride >= 1.0:
            s = int(np.sqrt(stride))
            mask_marked = mask_marked[:,:,::s,::s].view(self.inp_mask_shape)
        else:
            s = int(np.sqrt(int(1 / stride)))
            orig_size = (s * n for n in self.mask_image_size)
            mask_marked = masking.upsample_masks(
                mask_marked, orig_size).view(self.inp_mask_shape)

        mask_marked = torch.minimum(mask_marked, orig_mask)

        return (x_marked, mask_marked)

class ShuffleVisible(PatchPerturbation):
    """Shuffle the patches that are visible amongst themselves"""

    def perturb(self, x, mask, frame=-1):

        m = self.reshape_mask_to_video(mask)
        if frame is not None:
            frame = (frame % self.T)
            m = torch.flatten(m[:,frame], 1, -1)
        else:
            m = torch.flatten(m, 1, -1)

        ## convert to patches
        x = self.patchify(x).view(self.B, self.T, np.prod(self.mask_image_size), -1).clone()
        if frame is not None:
            x_shuffled = x[:,frame]
        else:
            x_shuffled = x.view(self.B, -1, x.size(-1))

        ## get inds to shuffle
        for b in range(m.shape[0]):
            p_inds = torch.where(~m[b])[-1]
            p_inds_shuffled = p_inds[torch.randperm(len(p_inds)).to(p_inds.device)]
            x_shuffled[b,...,p_inds,:] = x_shuffled[b,...,p_inds_shuffled,:]

        ## reconstruct
        x_shuffled = x.view(self.B, -1, np.prod(self.mask_image_size), x_shuffled.size(-1))
        if frame is not None:
            x_shuffled = torch.cat([x[:,:frame], x_shuffled, x[:,(frame+1):]], 1)
        x_shuffled = self.patchify(x_shuffled.view(self.B, self.num_patches, -1, self.C), to_video=True)

        return (x_shuffled, mask)

class ShuffleAll(PatchPerturbation):

    def perturb(self, x, mask, frame=-1):
        
        m = self.reshape_mask_to_video(mask)
        if frame is not None:
            frame = (frame % self.T)
            m = torch.flatten(m[:,frame], 1, -1)
        else:
            m = torch.flatten(m, 1, -1)

        ## convert to patches
        x = self.patchify(x).view(self.B, self.T, np.prod(self.mask_image_size), -1)
        if frame is not None:
            x_shuffled = x[:,frame]
        else:
            x_shuffled = x.view(self.B, -1, x.size(-1))

        ## get inds to shuffle
        for b in range(m.shape[0]):
            p_inds = torch.arange(0, x_shuffled.shape[1]).long().to(x_shuffled.device)
            p_inds_shuffled = torch.randperm(x_shuffled.shape[1]).to(x_shuffled.device)
            x_shuffled[b,...,p_inds,:] = x_shuffled[b,...,p_inds_shuffled,:].clone()

        ## reconstruct
        x_shuffled = x.view(self.B, -1, np.prod(self.mask_image_size), x_shuffled.size(-1))
        if frame is not None:
            m = m.unsqueeze(-1).unsqueeze(1).float()
            x_shuffled = torch.cat([
                x[:,:frame],
                x_shuffled*(1-m) + x[:,frame:frame+1]*m,
                x[:,(frame+1):]], 1)
        else:
            m = m.view(self.B, self.T, -1, 1).float()
            x_shuffled = x_shuffled*(1-m) + x*m
            
        x_shuffled = self.patchify(x_shuffled.view(self.B, self.num_patches, -1, self.C), to_video=True)

        return (x_shuffled, mask)

class ShuffleInvisible(PatchPerturbation):
    """Swap visible patches with random inivisble patches"""

    def perturb(self, x, mask, frame=-1):

        m = self.reshape_mask_to_video(mask)
        if frame is not None:
            frame = (frame % self.T)
            m = torch.flatten(m[:,frame], 1, -1)
        else:
            m = torch.flatten(m, 1, -1)

        ## convert to patches
        x = self.patchify(x).view(self.B, self.T, np.prod(self.mask_image_size), -1).clone()
        if frame is not None:
            x_shuffled = x[:,frame]
        else:
            x_shuffled = x.view(self.B, -1, x.size(-1))

        ## get inds to shuffle
        for b in range(m.shape[0]):
            vis_inds = torch.where(~m[b])[-1]
            invis_inds = torch.where(m[b])[-1]
            if (len(invis_inds) == 0) or (len(vis_inds)) == 0:
                continue

            ## make sure there are more invis inds than vis

            R = (len(vis_inds) // len(invis_inds)) + 1
            invis_inds_shuffled = torch.cat([
                invis_inds[torch.randperm(len(invis_inds)).to(invis_inds.device)]
                for r in range(R)], 0)
            x_shuffled[b,...,vis_inds,:] = x_shuffled[b,...,invis_inds_shuffled[:len(vis_inds)],:]

        ## reconstruct
        x_shuffled = x.view(self.B, -1, np.prod(self.mask_image_size), x_shuffled.size(-1))
        if frame is not None:
            x_shuffled = torch.cat([x[:,:frame], x_shuffled, x[:,(frame+1):]], 1)
        x_shuffled = self.patchify(x_shuffled.view(self.B, self.num_patches, -1, self.C), to_video=True)

        return (x_shuffled, mask)

class EnergySampleUnmask(PatchPerturbation):

    def __init__(self, num_visible=None, radius=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_visible = num_visible
        self.radius = radius
        self.mask_generator = None

    def _check_shapes(self, x, mask):
        self.inp_mask_shape = mask.shape

    def get_mask_generator(self):
        if self.mask_generator is None:
            self.mask_generator = masking.EnergyMaskingGenerator(
                input_size=self.mask_shape, mask_ratio=1.0, temperature=None)
        self.mask_generator.num_visible = self.num_visible
        return self.mask_generator

    def get_nearby_patches(self, mask):
        nearby_patches = masking.patches_adjacent_to_visible(
            mask, radius=self.radius, size=self.mask_shape[-2:])
        nearby_patches = masking.upsample_masks(nearby_patches, size=self.image_size)        
        return nearby_patches

    def perturb(self, x, mask, energy, frame=-1):
        assert list(x.shape[-2:]) == list(energy.shape[-2:]), (x.shape, energy.shape)
        mask = self.reshape_mask_to_video(mask)
        if frame is not None:
            frame = (frame % self.T)
            m = mask[:,frame:frame+1]
            energy = energy[:,frame:frame+1]
        else:
            m = mask

        if self.radius is not None:
            nearby_mask = self.get_nearby_patches(mask)
            energy = energy * nearby_mask

        if self.num_visible is not None:
            EG = self.get_mask_generator()
            new_mask = self.reshape_mask_to_video(EG(energy))
        else:
            energy = nn.AvgPool2d(self.patch_size[-2:], stride=self.patch_size[-2:])(energy[:,0])
            new_mask = torch.logical_not(utils.sample_per_pixel(energy).bool())

        if frame is not None:
            new_mask = torch.cat([mask[:,:frame], new_mask, mask[:,(frame+1):]], 1)

        new_mask = new_mask.view(*self.inp_mask_shape)
        return (x, new_mask)

class MultiShiftPatchesAndMask(ShiftPatchesAndMask):
    """Shift different patches by different amounts, in pixels."""
    
    def __init__(self,
                 patch_size,
                 max_shift_fraction=0.15,
                 padding_mode='constant',
                 allow_fractional_shifts=True,
                 **kwargs):

        super().__init__(patch_size=patch_size,
                         max_shift_fraction=max_shift_fraction,
                         padding_mode=padding_mode,
                         allow_fractional_shifts=allow_fractional_shifts,
                         **kwargs)
        self.reset_shifts()

    def reset_shifts(self):
        self.shifts = None
        self.num_shifts = 1

    def _check_shapes(self, x, mask_sequence):
        """figure out the shapes of the mask and the number of shifts"""
        if mask_sequence is None:
            self.num_shifts = 1
            return
        
        if isinstance(mask_sequence, (list, tuple)):
            assert all((len(m.shape) == 2 for m in mask_sequence)), mask_sequence[0].shape
            mask_sequence = torch.stack(mask_sequence, -1)
            
        if len(mask_sequence.shape) == 2:
            self.inp_mask_shape = mask_sequence.shape
            self.num_shifts = 1
        else:
            ## last dimension iterates the shifts
            assert len(mask_sequence.shape) == 3, mask_sequence.shape 
            self.inp_mask_shape = mask_sequence.shape[:-1]
            self.num_shifts = mask_sequence.size(-1)

    def _preprocess_masks_and_perturbations(self,
                                            mask_sequence,
                                            perturbation_points_sequence):
        
        if isinstance(mask_sequence, (list, tuple)):
            assert all((len(m.shape) == 2 for m in mask_sequence)), mask_sequence[0].shape
            mask_sequence = torch.stack(mask_sequence, -1)

        if len(mask_sequence.shape) == 2:
            mask_sequence = mask_sequence.unsqueeze(-1) # [B,N,1]

        m_seq = mask_sequence.clone()

        if perturbation_points_sequence is None:
            p_seq = m_seq
            self._has_base_mask = False
        else:
            p_seq = perturbation_points_sequence
            if isinstance(p_seq, (list, tuple)):
                p_seq = torch.stack(p_seq, -1)
            elif len(p_seq.shape) == 2:
                p_seq = p_seq.unsqueeze(-1)

            if m_seq.size(-1) < p_seq.size(-1):
                self.num_shifts = p_seq.size(-1)
                m_seq = m_seq.expand(1, 1, self.num_shifts)
            assert p_seq.shape == m_seq.shape, (p_seq.shape, m_seq.shape)
            assert p_seq.dtype == torch.bool
            m_seq[p_seq] = 1
            p_seq = torch.logical_not(p_seq) # true patches will be perturbed
            self._has_base_mask = True

        return (m_seq, p_seq)

    def _preprocess_shifts_sequence(self, shifts_sequence):
        if shifts_sequence is None:
            return [self.get_random_shift() for _ in range(self.num_shifts)]

        if hasattr(shifts_sequence, 'shape'):
            ## tensor
            assert len(shifts_sequence.shape) == 2, shifts_sequence.shape
            D,S = shifts_sequence.shape
            assert D == 2, D
            assert S in (self.num_shifts, 1), (S, self.num_shifts)
            if isinstance(shifts_sequence, torch.Tensor):
                shifts_sequence = [shifts_sequence[...,s].detach().cpu().numpy() for s in range(S)]
            else:
                shifts_sequence = [shifts_sequence[...,s] for s in range(S)]

        if isinstance(shifts_sequence, (list, tuple)):
            if not isinstance(shifts_sequence[0], (list, tuple)):
                shifts_sequence = [shifts_sequence]
            assert all((len(s) == 2 for s in shifts_sequence))

            ## all have same shift
            if len(shifts_sequence) == 1:
                return shifts_sequence * self.num_shifts

            else:
                assert len(shifts_sequence) == self.num_shifts

        return shifts_sequence
            
    def forward(self,
                x,
                mask_sequence,
                perturbation_points_sequence=None,
                shift_sequence=None,
                **kwargs):

        self.set_shapes(x, mask_sequence)
        if mask_sequence is None:
            return (x, mask_sequence)
        m_seq, p_seq = self._preprocess_masks_and_perturbations(
            mask_sequence,
            perturbation_points_sequence
        )

        s_seq = self._preprocess_shifts_sequence(shift_sequence)

        ## apply the shifts sequentially
        x_p = x
        mask_ps = []
        for s in range(self.num_shifts):
            x_p, mask_p = self.perturb(
                x_p,
                p_seq[...,s],
                mask_shift=None,
                shift=s_seq[s],
                **kwargs)
            if self._has_base_mask:
                mask_p = torch.minimum(m_seq[...,s], mask_p)
            mask_ps.append(mask_p)

        mask_ps = torch.stack(mask_ps, -1).amin(-1) # all the visible masks
        return (x_p, mask_ps)        
    
