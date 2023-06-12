import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
