import numpy as np
import matplotlib.pyplot as plt
from time import time
import copy
import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange

from cwm.models.segmentation import ImuConditionedFlowGenerator
from cwm.vis_utils import imshow

class MovabilityPredictor(ImuConditionedFlowGenerator):
    """
    Wrapper for head motion-conditioned predictor for predicting which
    parts of a scene are movable and what the consequences of moving them are.

    Note that if the predictor being wrapped is not conditioned on head motion,
    the methods in this class will still work but flow samples may be biased
    toward predicting counterfactual scenes with substantial camera motion.

    Movability is estimated by first sampling some patches, trying to move them,
    and seeing which parts of the scene have high counterfactual flow.

    This process is then iterated num_iters times by resampling patches from
    places that have high "movability", and also [optional] sampling "static"
    patches to try to isolate independent object motion.
    """
    VERBOSE = False
    
    def __init__(self,
                 *args,
                 initialize_from_keypoints=True,
                 iterate_from_keypoints=False,
                 keypoints_power=8,
                 movability_power=1,
                 num_initial_samples=16,
                 num_initial_active_patches=1,
                 num_initial_passive_patches=0,
                 num_samples_per_iteration=16,
                 num_active_patches_per_sample=1,
                 num_passive_patches_per_sample=1,
                 sample_passives_from_movable=False,
                 update_distribution_per_iteration=True,
                 num_iters=2,
                 sample_batch_size=4,
                 **kwargs):
        super().__init__(*args,
                         **kwargs)

        # using keypoints to sample
        self.initialize_from_keypoints = initialize_from_keypoints
        self.keypoints_power = keypoints_power
        self.keypoints_distribution = None

        # sampling parameters
        self.sample_batch_size = sample_batch_size
        self.movability_power = movability_power
        self.sample_passives_from_movable = sample_passives_from_movable

        # iteration parameters
        self.iterate_from_keypoints = iterate_from_keypoints
        self.num_initial_samples = num_initial_samples
        self.num_initial_active_patches = num_initial_active_patches
        self.num_initial_passive_patches = num_initial_passive_patches
        self.num_samples_per_iteration = num_samples_per_iteration
        self.num_active_patches_per_sample = num_active_patches_per_sample
        self.num_passive_patches_per_sample = num_passive_patches_per_sample
        self.num_iters = num_iters
        self.update_distribution_per_iteration = update_distribution_per_iteration

    def set_verbosity(self, is_verbose=True):
        self.VERBOSE = is_verbose

    def set_keypoints_distribution(self, x=None):
        
        if x is None:
            x = self.x
        assert x is not None

        if self.keypoint_predictor is not None:
            self.keypoints_distribution = self.predict_keypoints_distribution(
                x,
                power=self.keypoints_power
            )
        else:
            self.keypoints_distribution = None
        
    def sample_and_visualize_keypoints(self,
                                       x=None,
                                       sampled_keypoints=None,
                                       sampled_passive_patches=None,
                                       num_samples=32):

        if x is None:
            assert self.x is not None
            x = self.x

        if sampled_keypoints is None:
            self.set_keypoints_distribution(x)

            sampled_keypoints = self.sample_patches_from_energy(self.keypoints_distribution,
                                                                num_visible=1,
                                                                num_samples=num_samples)
        kps = sampled_keypoints.amin(-1)
        img = x if (x is not None) else self.x
        img = img.clone()

        alpha = self.get_masked_pred_patches(
            torch.zeros_like(x), kps, fill_value=[1,0,0])[:,:,0:1]

        red = torch.cat([alpha,
                         torch.zeros_like(alpha),
                         torch.zeros_like(alpha)
                         ], -3)

        img = img * (1 - alpha) + 0.5 * alpha * (red + img)

        if sampled_passive_patches is not None:
            passives = sampled_passive_patches.amin(-1)
            alpha = self.get_masked_pred_patches(
                torch.zeros_like(x), passives, fill_value=[0,0,1])[:,:,2:3]
            blue = torch.cat(2*[torch.zeros_like(alpha)] + [alpha], -3)
            img = img * (1 - alpha) + 0.5 * alpha * (blue + img)
        return (sampled_keypoints, img)

    def _sample_initial_motion_map(self,
                                   x,
                                   num_samples=None,
                                   sample_batch_size=None,
                                   do_filter=True,
                                   mask_head_motion=False,
                                   static_head_motion=True,
                                   normalize=True,
                                   patch_sampling_kwargs={},
                                   **kwargs):

        self.set_input(x)
        if self.initialize_from_keypoints:
            self.set_keypoints_distribution()
            sampling_dist = self.keypoints_distribution
            passive_dist = 1 - self.keypoints_distribution
        else:
            sampling_dist = None
            passive_dist = None

        flows, motion_patches, static_patches = self.sample_counterfactual_motion_map(
            x=self.x,
            active_sampling_distribution=sampling_dist,
            passive_sampling_distribution=passive_dist,
            num_active_patches=self.num_initial_active_patches,
            num_passive_patches=self.num_initial_passive_patches,
            num_samples=(num_samples or self.num_initial_samples),
            sample_batch_size=(sample_batch_size or self.sample_batch_size),
            do_filter=do_filter,
            mask_head_motion=mask_head_motion,
            static_head_motion=static_head_motion,
            patch_sampling_kwargs=patch_sampling_kwargs,
            **kwargs
        )

        motion_map = self.compute_mean_motion_map(flows,
                                                  normalize_per_sample=False,
                                                  normalize=normalize)

        return (motion_map, flows, motion_patches, static_patches)

    def _iterate_motion_map(self,
                            movability_distribution,
                            sample_passives_from_movable=True,
                            num_active_patches=None,
                            num_passive_patches=None,
                            num_samples=None,
                            sample_batch_size=None,
                            do_filter=True,
                            mask_head_motion=False,
                            static_head_motion=True,
                            patch_sampling_kwargs={},
                            normalize=True,
                            **kwargs):
        
        assert self.x is not None
        if movability_distribution is None:
            movability_distribution = torch.ones_like(self.x[:,0:1,0])
        movability_distribution = self.compute_mean_motion_map(movability_distribution)
        movability_distribution = movability_distribution ** self.movability_power

        if sample_passives_from_movable:
            passive_distribution = movability_distribution
        else:
            passive_distribution = (1 - movability_distribution).relu()

        if self.iterate_from_keypoints:
            self.set_keypoints_distribution(self.x)
            movability_distribution *= self.keypoints_distribution
            passive_distribution *= self.keypoints_distribution
        
        flows, motion_patches, static_patches = self.sample_counterfactual_motion_map(
            x=self.x,
            active_sampling_distribution=movability_distribution,
            passive_sampling_distribution=passive_distribution,
            num_active_patches=(num_active_patches or self.num_active_patches_per_sample),
            num_passive_patches=(num_passive_patches or self.num_passive_patches_per_sample),
            num_samples=(num_samples or self.num_samples_per_iteration),
            sample_batch_size=(sample_batch_size or self.sample_batch_size),
            do_filter=do_filter,
            mask_head_motion=mask_head_motion,
            static_head_motion=static_head_motion,
            patch_sampling_kwargs=patch_sampling_kwargs,
            **kwargs
        )

        motion_map = self.compute_mean_motion_map(flows,
                                                  normalize_per_sample=False,
                                                  normalize=normalize)
        
        return (motion_map, flows, motion_patches, static_patches)

    def reset_samples(self):
        self.movability_maps = []
        self.flow_samples_per_iter = []
        self.active_patches_per_iter = []
        self.passive_patches_per_iter = []

    def _update_results(self, results):
        movability, flows, active_patches, passive_patches = results
        self.movability_maps.append(movability)
        self.flow_samples_per_iter.append(flows)
        self.active_patches_per_iter.append(active_patches)
        self.passive_patches_per_iter.append(passive_patches)

    def visualize_iterations(self,
                             axes=None,
                             minimum_movability=False,
                             cmap='inferno'):
        self.fig = None        
        num_iters = len(self.movability_maps)
        if num_iters == 0:
            return

        if axes is None:
            self.fig, axes = plt.subplots(2, num_iters + 1, figsize=(4*(num_iters + 1), 8))

        vmax = torch.stack(self.movability_maps, -1).amax()
        for it in range(num_iters):
            _, img = self.sample_and_visualize_keypoints(
                x=self.x,
                sampled_keypoints=self.active_patches_per_iter[it],
                sampled_passive_patches=self.passive_patches_per_iter[it])
            imshow(img, t=1, ax=axes[0, it])
            imshow(self.movability_maps[it], ax=axes[1, it],
                   cmap=cmap,
                   vmin=0,
                   vmax=vmax)
            axes[0,it].set_title('iteration %d' % it, fontsize=20)
            for row in range(2):
                axes[row,it].set_xticks([])
                axes[row,it].set_yticks([])

        # last col is total
        _, img = self.sample_and_visualize_keypoints(
            x=self.x,
            sampled_keypoints=torch.cat(self.active_patches_per_iter, -1),
            sampled_passive_patches=torch.cat(self.passive_patches_per_iter, -1))
        total_movability = self.get_minimum_movability() if minimum_movability \
            else self.get_total_movability()
        imshow(img, t=1, ax=axes[0,-1])
        imshow(total_movability, ax=axes[1,-1], cmap=cmap, vmin=0, vmax=vmax)
        axes[0,-1].set_title('%s movability' % ['total', 'minimum'][int(minimum_movability)], fontsize=20)

        for row in range(2):
            axes[row,-1].set_xticks([])
            axes[row,-1].set_yticks([])
        axes[0,0].set_ylabel('selected motion patches', fontsize=18)
        axes[1,0].set_ylabel('relative movability', fontsize=18)

        plt.tight_layout()
        plt.show()

        self.axes = axes
        return

    def get_total_movability(self):
        if len(self.flow_samples_per_iter) == 0:
            return None
        all_flows = torch.cat(self.flow_samples_per_iter, -1)
        total_movability = self.compute_mean_motion_map(all_flows,
                                                        normalize_per_sample=False,
                                                        normalize=True)
        return total_movability

    def get_minimum_movability(self):
        if len(self.flow_samples_per_iter) == 0:
            return None

        mags = torch.stack([self.compute_mean_motion_map(fs) for fs in self.flow_samples_per_iter], -1)
        return mags.amin(-1)

    def forward(self,
                x,
                initial_active_patches=None,
                initial_passive_patches=None,
                initial_sampling_distribution=None,
                num_initial_samples=None,
                num_samples_per_iteration=None,
                sample_batch_size=None,
                num_iters=None,
                **kwargs):

        self.set_input(x)
        self.reset_samples()
        self.it = 0
        t0 = time()

        # create an initial movability map to sample from
        if initial_active_patches is None:
            results = self._sample_initial_motion_map(
                x=self.x,
                num_samples=num_initial_samples,
                sample_batch_size=sample_batch_size,
                **kwargs
            )
            self._update_results(results)
            if self.VERBOSE:
                t1 = time()
                print("Completed iter %d with %d samples in %0.3f s" % \
                      (self.it,
                       results[1].size(-1),
                       (t1 - t0)))
                t0 = time()

        else:
            raise NotImplementedError("pass initial patches")

        # iterate on this map
        for self.it in range(1, (num_iters or self.num_iters) + 1):
            if self.update_distribution_per_iteration:
                dist = self.get_total_movability()
            else:
                dist = self.movability_maps[-1]
                
            results = self._iterate_motion_map(
                dist,
                sample_passives_from_movable=self.sample_passives_from_movable,
                num_samples=num_samples_per_iteration,
                sample_batch_size=sample_batch_size,
                **kwargs
            )
            self._update_results(results)
            if self.VERBOSE:
                t1 = time()
                print("Completed iter %d with %d samples in %0.3f s" % \
                      (self.it,
                       results[1].size(-1),
                       (t1 - t0)))
                t0 = time()


        final_movability_map = self.movability_maps[-1]
        return final_movability_map


            
                
                
                
