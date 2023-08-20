import itertools
from dataclasses import dataclass
from typing import List, Optional, Union, Tuple

import torch
import torch.nn as nn

from cwm.models.masking import MaskingGenerator

@dataclass
class ChannelGroups:
    """
    Parameters for a group of channels that share a common mask ratio.

    Attributes:
        mask_ratio: the proportion of patches/tokens to be masked, in [0, 1]
        num_channels: the number of channels masked at this ratio
    """

    mask_ratio: float
    num_groups: float = 1

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        assert self.mask_ratio >= 0 and self.mask_ratio <= 1, (
            f"Mask fraction must be in [0.0, 1.0] but is {self.mask_ratio}"
        )
            

class ChannelGroupMaskingGenerator(MaskingGenerator):
    def __init__(
            self,
            height: int,
            width: int,
            mask_ratio: float,
            num_groups: int = 1,            
            seed: int = 0,
            clumping_factor: int = 1,
            randomize_num_visible: bool = False
    ) -> None:
        """
        Wrapper class for making MaskingGenerators (uniform) for individual channel groups
        """
        input_size = (num_groups, height, width)
        super(ChannelGroupMaskingGenerator, self).__init__(
            input_size,
            mask_ratio=mask_ratio,
            seed=seed,
            visible_frames=0,
            clumping_factor=clumping_factor,
            randomize_num_visible=randomize_num_visible,
            create_on_cpu=False,
            always_batch=True
        )

class MixedChannelGroupMasker:

    def __init__(
            self,
            height: int,
            width: int,
            channel_groups_list: List[ChannelGroups],
            seed: int = 0,
            clumping_factor: int = 1,
            randomize_num_visible: bool = False
    ) -> None:

        self.generators = [
            ChannelGroupMaskingGenerator(
                height=height, width=width, num_groups=groups.num_groups,
                mask_ratio=groups.mask_ratio,
                clumping_factor=clumping_factor,
                randomize_num_visible=randomize_num_visible,
                seed=seed
            )
            for groups in channel_groups_list
        ]

    def __repr__(self) -> str:
        repr_str = ""
        for i, gen in enumerate(self.generators):
            repr_i = gen.__repr__()
            repr_str += f"Channel Groups {i}\n{repr_i}\n"
        return repr_str

    def __call__(self, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        mask_groups = [[generator(x=x)] for generator in self.generators]
        all_masks = list(itertools.chain.from_iterable(mask_groups))
        return torch.cat(all_masks, dim=-1)

    @property
    def num_groups(self):
        return len(self.generators)
    
    def set_group_num_visible(self, num_visible: int, group: int = 0) -> None:
        self.generators[group].num_visible = num_visible

    def set_group_mask_ratio(self, mask_ratio: float, group: int = 0) -> None:
        self.generators[group].mask_ratio = mask_ratio

    def set_generator_num_visibles(self, num_visibles: Union[List[int], int]) -> None:

        if not hasattr(num_visibles, '__len__'):
            num_visibles = [num_visibles] * self.num_groups

        assert len(num_visibles) == self.num_groups
        for idx in range(self.num_groups):
            self.generators[idx].set_num_visible(num_visibles[idx], group=idx)
                

    def set_generator_ratios(self, mask_ratios: Union[List[float], float]) -> None:

        if not hasattr(mask_ratios, '__len__'):
            mask_ratios = [mask_ratios] * self.num_groups

        assert len(mask_ratios) == self.num_groups
        for idx in range(self.num_groups):
            self.generators[idx].set_mask_ratio(mask_ratios[idx], group=idx)
        
                

        
        

