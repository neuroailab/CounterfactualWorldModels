import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from cwm.models.utils import sample_from_energy
from cwm.models.masking import (MaskingGenerator,
                                upsample_masks)

class EnergySamplingMaskingGenerator(MaskingGenerator):
    """Sample unmasked patches that are high on motion energy"""
    
    def __init__(self,
                 input_size,
                 mask_ratio,
                 seed=0,
                 resize=True,
                 temperature=None,
                 clumping_factor=1,
                 pool_mode='mean',
                 eps=1e-9,
                 energy_power=1,
                 **kwargs
    ):

        
        super(EnergySamplingMaskingGenerator, self).__init__(
            input_size=input_size,
            mask_ratio=mask_ratio,
            clumping_factor=clumping_factor,
            seed=seed,
            **kwargs)
        self.motion_energy_density = nn.Identity(inplace=True)

        ## if self.resize, resize the input movie bilinearly before estimating energy
        self.resize = transforms.Resize((self.height, self.width)) if resize else (lambda x:x)
        self.pool_mode = pool_mode

        ## clumping factor controls how large the visible patches are
        self.cf = clumping_factor

        ## temperature controls how biased sampling is toward motion.
        ## temperature of 0 ==> uniform sampling
        ## temperature of inf ==> argmax(motion) sampling
        ## temperature of None ==> no rescaling of density
        self.temperature = temperature
        self.eps = eps
        self.energy_power = energy_power

    def boltzmann(self, x):
        x = x - x.amax((-2,-1), keepdim=True)
        return torch.exp(x * self.temperature)

    def _get_pool_func(self, k):
        if self.pool_mode == 'mean':
            return nn.AvgPool2d(k, stride=k)
        elif self.pool_mode == 'max':
            return nn.MaxPool2d(k, stride=k)
        elif self.pool_mode == 'min':
            return lambda x: -nn.MaxPool2d(k, stride=k)(-x)

    def sample_mask_per_frame(self, video):
        energy = self.motion_energy_density(video).view(-1, 1, *video.shape[-2:]) # [_BT,1,H,W]
        H,W = energy.shape[-2:]
        assert (H % self.height == 0) and (W % self.width == 0)
        if (H != self.height) or (W != self.width):
            pool_kernel_size = ((H * self.cf) // self.height, (W * self.cf) // self.width)
            pool_func = self._get_pool_func(pool_kernel_size)
            energy = pool_func(energy)
            
        if self.temperature is not None:
            energy = self.boltzmann(energy)

        num_points = (self.num_patches_per_frame - self.num_masks_per_frame) // (self.cf**2)
        if self.randomize_num_visible:
            num_points = self.rng.randint(low=0, high=(num_points + 1))
        visible = sample_from_energy(
            torch.pow(energy, self.energy_power),
            binarize=True,
            num_points=max(num_points, 1),
            eps=self.eps,
            normalize=True
        ) > 0.5
        if num_points == 0:
            visible = torch.zeros_like(visible)
        if self.cf > 1:
            visible = upsample_masks(visible, size=(self.height, self.width))
        mask = torch.logical_not(visible).flatten(1) # [_BT,N]
        return mask

    def forward(self, video, num_frames=None):
        shape = video.shape
        if len(shape) == 4:
            video = self.resize(video)
            video = video.unsqueeze(1)
        else:
            assert len(shape) == 5, shape
            video = torch.stack(list(map(self.resize, torch.unbind(video, 1)), 1))
        B = video.size(0)
        masks = self.sample_mask_per_frame(video)
        masks = masks.view(B, -1, masks.shape[-1]).flatten(1)
        if B == 1 and not self.always_batch:
            masks = masks.squeeze(0)

        if self.visible_frames > 0:
            vis = torch.zeros(
                (B, 1, self.height, self.width), dtype=torch.bool)
            vis = vis.view(masks.shape).to(masks.device)
            masks = torch.cat(([vis]*self.visible_frames) + [masks], -1)
            
        return masks

class RotatedTableEnergyMaskingGenerator(EnergySamplingMaskingGenerator):
    
    def __init__(self, input_size, mask_ratio, visible_frames=1, seed=0,
                 *args, **kwargs):

        super().__init__(
            input_size=(
                input_size[0]-visible_frames, *input_size[1:]),
            mask_ratio=mask_ratio,
            visible_frames=visible_frames,
            seed=seed,
            *args, **kwargs)
        self.visible_frames = visible_frames            

class FlowSampleFilter(nn.Module):
    """
    Filter out flow samples based on a list of filter methods
    - patch_magnitude: filter out samples if the flow magnitude at the selected patch is too small
    - flow_area: filter out samples if there is a large motion across the scene
    - num_corners: filter out samples if the flow covers more than 1 corner of the image

    @param filter_methods: list of filter methods
    @param flow_magnitude_threshold: threshold for patch_magnitude filter
    @param flow_area_threshold: threshold for flow_area filter
    @param num_corners_threshold: threshold for num_corners filter

    """

    ALL_FILTERS = ['patch_magnitude', 'flow_area', 'num_corners']
    def __init__(self,
                filter_methods=ALL_FILTERS,
                flow_magnitude_threshold=5.0,
                flow_area_threshold=0.75,
                num_corners_threshold=2):
        super(FlowSampleFilter, self).__init__()

        # filtering methods and hyperparameters
        self.filter_methods = filter_methods
        self.flow_magnitude_threshold = flow_magnitude_threshold
        self.flow_area_threshold = flow_area_threshold
        self.num_corners_threshold = num_corners_threshold

    def __repr__(self):
        return ("filtering by %s\nusing flow_magnitude_threshold %0.1f\n" + \
            "using flow_area_threshold %0.2f\n" + \
            "using num_corners_threshold %d") % \
            (self.filter_methods, self.flow_magnitude_threshold,
             self.flow_area_threshold, self.num_corners_threshold)

    def compute_flow_magnitude(self, flow_samples, active_patches=None):
        """
        Compute the flow magnitude over the entire image and at the selected patches of the second frame

        @param flow_samples: [B, 2, H, W, num_samples]
        @param active_patches: [B, num_patches, num_samples], note: num_patches is equal to the number of patches in two frames

        @return:
            flow_mag: flow magnitude of shape [B, H, W, num_samples]
            flow_mag_down: downsized flow magnitude of shape, [B, h, w, num_samples] where h = H // patch_size
            patch_flow_mag: average flow magnitude at selected patches in frame 2 [B, num_samples]
            active_second: indication of which patch in frame 2 is active, [B, num_samples, hw] (or None)
        """

        # Compute the flow magnitude map
        flow_mag = flow_samples.norm(dim=1, p=2)  # [B, H, W, num_samples]

        # Compute the average flow magnitude at the selected patches, if active_patches is not None
        if active_patches is not None:
            B, _, H, W, num_samples = flow_samples.shape

            _, num_patches, _ = active_patches.shape
            assert active_patches.shape[-1] == num_samples, (active_patches.shape, num_samples)

            # infer dimension of active patches map:
            assert H == W, "the inference of patch size assumes H == W"
            # num_patches is the number of patches in 2 frames, so we need to divide by 2
            h = w = int((num_patches / 2) ** 0.5)

            # get the active patches of the second frame and reshape
            active_second = 1 - active_patches[:, (h * w):, :].float()  # [B, hw, num_samples]
            active_second = active_second.permute(0, 2, 1)  # [B, num_samples, hw]

            # downsample flow magnitude map to the dimension of the active patches map
            flow_mag_down = F.interpolate(flow_mag.permute(0, 3, 1, 2), size=[h, w], mode='bilinear')  # [B, num_samples, h, w]
            flow_mag_down = flow_mag_down.flatten(2, 3)  # [B, num_samples, hw]

            # compute the mean flow magnitude at the selected patches
            patch_flow_mag = (flow_mag_down * active_second).sum(dim=-1) / (active_second.sum(-1) + 1e-12)  # [B, num_samples]

            return flow_mag, flow_mag_down, patch_flow_mag, active_second
        else:
            return flow_mag

    def filter_by_patch_magnitude(self, patch_flow_mag):
        """
        Filter out samples if the flow magnitude at the selected patch is too small (< self.flow_magnitude_threshold)
        @param patch_flow_mag: average flow magnitude at the selected patch of shape (B, S)
        @return: filter mask of shape (B, S), 1 for samples to be filtered out, 0 otherwise
        """
        assert self.flow_magnitude_threshold is not None
        return patch_flow_mag < self.flow_magnitude_threshold

    def filter_by_flow_area(self, flow_mag):
        """
        Filter out samples if there is a large motion across the scene (> self.flow_area_threshold)
        @param flow_mag: flow magnitude of shape (B, H, W, S)
        @return: boolean mask of shape (B, S), 1 for samples to be filtered out, 0 otherwise
        """
        assert self.flow_magnitude_threshold is not None
        assert self.flow_area_threshold is not None
        _, H, W, _ = flow_mag.shape
        flow_area = (flow_mag > self.flow_magnitude_threshold).flatten(1, 2).sum(1) / (H * W) # [B, num_samples]
        return flow_area > self.flow_area_threshold

    def filter_by_num_corners(self, flow_mag):
        """
        Filter out samples if the flow covers more than 1 corner of the image
        @param flow_mag: flow magnitude of shape (B, H, W, S)
        @return: boolean mask of shape (B, S), 1 for samples to be filtered out, 0 otherwise
        """
        assert self.flow_magnitude_threshold is not None
        # Binarize flow magnitude map
        flow_mag_binary = (flow_mag > self.flow_magnitude_threshold).float()

        # Get the four corners of the flow magnitude map
        top_l, top_r, bottom_l, bottom_r = flow_mag_binary[:, 0:1, 0:1], flow_mag_binary[:, 0:1,-1:], \
                                           flow_mag_binary[:, -1:, 0:1], flow_mag_binary[:, -1:,-1:]
        top_l = top_l.flatten(1, 2).max(1)[0]
        top_r = top_r.flatten(1, 2).max(1)[0]
        bottom_l = bottom_l.flatten(1, 2).max(1)[0]
        bottom_r = bottom_r.flatten(1, 2).max(1)[0]

        # Add up the 4 corners
        num_corners = top_l + top_r + bottom_l + bottom_r

        return num_corners >= self.num_corners_threshold


    def forward(self, flow_samples, active_patches):
        """
        @param flow_samples: flow samples of shape [B, 2, H, W, num_samples]
        @param active_patches: active patches of shape [B, num_patches, num_samples]

        @return: filtered flow samples of shape [B, 2, H, W, num_samples]. Flow sampled being filtered is set to zero
        @return: filtered mask of shape [B, num_samples]. 1 means the sample is filtered out
        """
        B, _, H, W, num_samples = flow_samples.shape

        # Compute flow magnitude maps and the average flow magnitude at active patches
        flow_mag, flow_mag_down, patch_flow_mag, _ = self.compute_flow_magnitude(flow_samples, active_patches)

        # Initialize the filter mask, 0 for keeping the sample, 1 for filtering out the sample
        filter_mask = torch.zeros(B, num_samples).to(flow_samples.device).bool()

        # Iterate through all filter methods and update the filter mask
        for method in self.filter_methods:
            if method == 'patch_magnitude':
                _filter_mask = self.filter_by_patch_magnitude(patch_flow_mag)  # [B, num_samples]
            elif method == 'flow_area':
                 _filter_mask = self.filter_by_flow_area(flow_mag)  # [B, num_samples]
            elif method == 'num_corners':
                _filter_mask = self.filter_by_num_corners(flow_mag)  # [B, num_samples]
            else:
                raise ValueError(f'Filter method must be one of {self.filter_methods}, but got {method}')

            filter_mask = filter_mask | _filter_mask # [B, num_samples]

        # Apply the filter mask and set the rejected flow_samples to be zero
        filter_mask = filter_mask.view(B, 1, 1, 1, num_samples).contiguous() # [B, 1, 1, 1, num_samples]
        filter_mask = filter_mask.expand_as(flow_samples) # [B, 2, H, W, num_samples]
        flow_samples[filter_mask] = 0. # [B, 2, H, W, num_samples]

        return flow_samples.contiguous(), filter_mask # [B, 2, H, W, num_samples], [B, num_samples]
