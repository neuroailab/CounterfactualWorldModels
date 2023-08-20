from functools import partial
import numpy as np
import torch
import torch.nn as nn

from cwm.models.VideoMAE.utils import (
    to_2tuple,
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD
)

def imagenet_normalize_image(x):
    assert len(x.shape) == 4 and x.size(1) == 3, x.shape
    mean = torch.tensor(IMAGENET_DEFAULT_MEAN).to(x).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_DEFAULT_STD).to(x).view(1, 3, 1, 1)
    return (x - mean) / std

class ImageNetNormalize(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return imagenet_normalize_image(x)

class ImagePatchEmbed(nn.Module):
    """Patch embedding layer for image inputs. Uses Conv2d. For videos, use PatchEmbed"""
    def __init__(
            self,
            patch_size: int = 16,
            in_channels: int = 3,
            embed_dim: int = 768
    ) -> None:
        
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        self.ph, self.pw = self.patch_size
        self.in_channels = self.num_channels = in_channels
        self.embed_dim = embed_dim

        # projection to tokens
        self.proj = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )

    def get_num_patches(self, image_size):
        H, W = image_size
        return (H // self.patch_size[0]) * (W // self.patch_size[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # handle videos with single frame
        try:
            B, C, H, W = x.shape
        except:
            assert (len(x.shape) == 5) and (x.shape[2] == 1), x.shape
            x = x.squeeze(2)
            B, C, H, W = x.shape

        assert (
            (H % self.patch_size[0] == 0)
            and (W % self.patch_size[1] == 0)
        ), (
            f"Patch size {self.patch_size} does not evenly divide image dimensions ({H}, {W})"
        )

        x = self.proj(x).flatten(2).transpose(1, 2) # [B, num_patches, embed_dim]
        return x
