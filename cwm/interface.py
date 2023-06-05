import numpy as np
import copy, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import matplotlib
import matplotlib.pyplot as plt

from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage

import cwm.models.prediction as prediction
import cwm.models.perturbation as perturbation
import cwm.models.utils as utils
import cwm.data.utils as data_utils

from cwm.vis_utils import imshow as vis_tensor
from cwm.models.masking import RectangularizeMasks


class Dummy():
    def __init__(self):
        pass
    def __enter__(self):
        pass
    def __exit__(self, type, value, traceback):
        pass

class CounterfactualPredictionInterface(object):
    """
    Uses a countefactual flow generator (optionally with head motion conditioning
    as backend for object segmentation.

    The user specifies:
     - a set of visible active patches that will be perturbed,
     - a set of visible passive patches that will be unperturbed or fixed in place (made static)
     - a perturbation to apply to the active patches
     - [optional] other contextual inputs to the backend predictor, such as head motion

    The backend predictor is then made to run a prediction given its selected inputs,
    and the resulting optical flow is used to segment the scene.
    """

    def __init__(self, axes, G,
                 x=None, model_kwargs={},
                 patch_selector=None,
                 size=[224,224],
                 bbox_corners=None,
                 frame=0,
                 device='cpu',
                 static=True,
                 max_speed=None,
                 max_shift=3,
                 sample_batch_size=8,
                 max_samples_per_batch=32,
                 normalize_flow_magnitude=False,
                 show_ticks=True,
                 show_error_diff=False,
                 active_color=[1,1,1],
                 passive_color=[0.25,0.25,0.25],
                 dtype=torch.float32,
                 seed=0):
        """
        arguments:
         - ax: the axis on which to plot generated flow
         - G: a generator object that has a 'get_counterfactual_flow' method.
         - x: the input image or movie on which to do the prediction and segmentation.
        """
        ## set up the backend
        assert hasattr(G, 'get_counterfactual_flow') or hasattr(G, 'get_counterfactual_prediction'), \
            "%s must have a method for generating counterfactual prediction given some inputs" % G
        self.G = G
        self.device = device
        self.frame = frame
        self.size = size
        self._static = static
        if bbox_corners is not None:
            assert hasattr(bbox_corners, '__len__')
            assert len(bbox_corners) == 2
            h1,w1 = bbox_corners[0]
            h2,w2 = bbox_corners[1]
            x = x[...,h1:h2,w1:w2]
        self.dtype = dtype
        self.decorator = Dummy() if (self.dtype != torch.float16) \
            else torch.cuda.amp.autocast(enabled=True)
        self.x = x
        self._model_kwargs = {k:v for k,v in model_kwargs.items()}
        self._reset_masks()

        ## running a batch of samples from a single init
        self.sample_batch_size = sample_batch_size
        self.max_samples_per_batch = max_samples_per_batch

        ## patch selection algorithm
        if patch_selector is not None:
            self.patch_selector = patch_selector
            self.patch_selector.visualization_mode = True

        ## set up the visualization
        self.flow_ax, self.seg_ax, self.corr_ax = (None, None, None)
        if hasattr(axes, '__len__'):
            _axes = []
            R = len(axes)
            if hasattr(axes[0], '__len__'):
                C = len(axes[0])
                for i in range(R):
                    for j in range(C):
                        _axes.append(axes[i][j])
            else:
                _axes = [axes[r] for r in range(R)]
                        
            self.ax = _axes[0]
            if len(_axes) > 1:
                self.corr_ax = _axes[1]
            if len(_axes) > 2:
                self.flow_ax = _axes[2]
            if len(_axes) > 3:
                self.seg_ax = _axes[3]
        else:
            self.ax = ax
        
        self._show_ticks = show_ticks
        H = self.size[0] if (size is not None) else 224
        self.text = self.ax.text(0, 1.1*H, "", va="bottom", ha="left")        
        self.connect()
        self.imshow(self.ax)

        self.max_speed = max_speed
        self._normalize_flow_magnitude = normalize_flow_magnitude
        self.flow2rgb = data_utils.FlowToRgb(max_speed=(self.max_speed or 10),
                                             from_image_coordinates=False,
                                             from_sampling_grid=True)

        self.max_shift = max_shift
        self.shift = self.press_loc = None
        self.do_drag = False
        self.seed = seed
        self.rng = np.random.RandomState(self.seed)
        torch.manual_seed(self.seed)

        # set the input and initialize
        self.G.set_input(self.get_input())                    
        self._show_flow = False
        self._active_color = active_color
        self._passive_color = passive_color

        ## for showing segments
        self.counterfactual_inputs = []
        self.preds_list, self.flow_samples_list, self._corrmat_inds_list, self.shifts = [], [], [], []
        self._show_error_diff = show_error_diff
        self._reset_flow_errors_list()
        self._flow_corrs = self._num_flow_samples = None
        self.seed = seed
        self.rng = np.random.RandomState(seed=seed)

    def set_sample_batch_size(self, v):
        self.sample_batch_size = sample_batch_size

    def connect(self):
        self.cidpush = self.ax.figure.canvas.mpl_connect('button_press_event', self.__call__)
        self.cidmove = self.ax.figure.canvas.mpl_connect('motion_notify_event', self.drag_to_set_shift)
        self.cidrelease = self.ax.figure.canvas.mpl_connect('button_release_event', self.on_release)

    def disconnect(self):
        self.ax.figure.mpl_disconnect(self.cidpush)

    def imshow(self, ax=None, img=None, txt=None, channels_first=True, **kwargs):
        if img is None and self.x is None:
            return

        if ax is None:
            return

        if img is None:
            img = self.x


            
        if len(img.shape) == 4:
            img = img[0]
        else:
            assert len(img.shape) == 3

        if channels_first:
            img = img.permute(1,2,0)

        if isinstance(img, torch.Tensor):
            img = img.float().detach().cpu().numpy()

        self._img = img            
        
        ax.imshow(img, **kwargs)
        if not self._show_ticks:
            ax.set_xticks([])
            ax.set_yticks([])
        if txt is not None:
            self.text.set_text(str(txt))

    def resize(self, x):
        if self.size is None:
            return x
        _resize = transforms.Resize(self.size)
        shape = x.shape[-3:]
        _img = x.view(-1, *shape)
        _img = _resize(_img).view(*x.shape[:-3], *_img.shape[-3:])
        return _img

    @property
    def x(self):
        if self._x is None:
            return None
        if len(self._x.shape) == 5:
            return self._x[:,(self.frame or 0)].to(self.dtype)
        elif len(self._x.shape) == 4:
            return self._x.to(self.dtype)
        elif len(self._x.shape) == 3:
            return self._x[None].to(self.dtype)
        else:
            raise ValueError("image must have rank in {3,4,5} but has shape %s" % self._x.shape)
    @x.setter
    def x(self, x):
        if x is None:
            self._x = None
            return
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x).to(self.device)

        x = (x / 255.) if (x.dtype == torch.uint8) else x
        self._x_ori = x.clone()
        self._shape = x.shape
        self._x = self.resize(x).to(self.dtype)

    def get_input(self):
        return self.G.get_static_input(self._x.to(self.dtype)) if self._static else self._x.to(self.dtype)

    def _get_flow(self, shift, static=True, *args, **kwargs):
        x = self.G.x
        if static:
            x = self.G.make_static_movie(x[:,0:1], T=2)

        kwargs.update(self._model_kwargs)
        if hasattr(self.G, 'get_counterfactual_flow'):
            assert hasattr(self.G, 'get_static_imu')
            y, flow = self.G.get_counterfactual_flow(x,
                                                     head_motion=self.G.get_static_imu(
                                                         x=self.G.x.to(self.dtype)),
                                                     mask_head_motion=False,
                                                     *args, **kwargs)
            return (y, flow)
        else:
            y = self.G.get_counterfactual_prediction(x,
                                                     active_patches=self.active_patches,
                                                     mask=self.passive_patches,
                                                     shift=shift)
                                                     
            return (y, None)
        
    def _get_patch_inds(self, event):
        if event.xdata is None or event.ydata is None:
            return None, None
        else:
            x,y = int(np.floor(event.xdata)), int(np.floor(event.ydata))
        txt = 'xdata=%d, ydata=%d, key=%s' % (x, y, event.key)
        i,j = y,x
        return [i,j]

    def _add_patch(self, i, j, mask=None, t=-1):
        if mask is None:
            mask = self.mask

        shape = mask.shape
        _i,_j = i//self.G.patch_size[-2], j//self.G.patch_size[-1]
        T,H,W = self.G.mask_shape
        N = H*W
        ind = (t % T) * N + _i * W + _j
        mask[0,ind] = ~(mask[0,ind])

        return mask

    def _reset_masks(self):
        self.G.set_input(self.get_input())
        self.mask = self.G.get_zeros_mask(self.get_input())
        self.active_patches = self.G.get_zeros_mask(self.get_input())
        self.passive_patches = self.G.get_zeros_mask(self.get_input())

    def _get_shift_color(self):
        if self.shift is None:
            return self._active_color
        y, x = list(np.array(self.shift) / self.max_shift)
        angle = np.arctan2(-y, x)
        speed = np.sqrt(x**2 + y**2)
        hue = (angle % (2 * np.pi)) / (2 * np.pi)
        sat = 1
        val = speed
        hsv = np.array([hue, sat, val])
        rgb = matplotlib.colors.hsv_to_rgb(hsv)
        if sum(rgb) == 0:
            rgb = np.array([0.1, 0.1, 0.1])
        return list(rgb)

    def _make_mask_img(self):
        img = self.G.get_masked_pred_patches(self.G.make_static_movie(self.G.x[:,0:1], T=2),
                                             self.active_patches,
                                             fill_value=self._get_shift_color())
        img = self.G.get_masked_pred_patches(img,
                                             self.passive_patches,
                                             fill_value=self._passive_color)
        self.masked_img = img
        return img[:,-1]

    def _store_current_patches(self):
        self._active_patches_last = self.active_patches.clone()
        self._passive_patches_last = self.passive_patches.clone()

    def _restore_last_patches(self):
        self.active_patches = self._active_patches_last.clone()
        self.passive_patches = self._passive_patches_last.clone()

    def sample_shift(self):
        shift = [0,0]
        assert self.max_shift > 0
        while sum([s**2 for s in shift]) == 0:
            shift = [self.rng.randint(-self.max_shift, self.max_shift+1),
                     self.rng.randint(-self.max_shift, self.max_shift+1)]

        return shift

    def drag_to_set_shift(self, event):
        if (self.press_loc is None) or (self.do_drag is False):
            return
        dx = event.xdata - self.press_loc[0]
        dy = event.ydata - self.press_loc[1]
        self.shift = np.array([dy, dx]) // np.array([self.G.patch_size[-2], self.G.patch_size[-1]])
        self.shift = np.maximum(np.minimum(self.shift, self.max_shift), -self.max_shift)
        self.shift = [int(s) for s in list(self.shift)]
        txt = 'shift by %s' % str(list(self.shift))
        self.text.set_text(txt)

    def on_release(self, event):
        self.press_loc = None
        self.do_drag = False
        if not self._show_flow:
            self.imshow(self.ax, self._make_mask_img(), self.txt)

    def _add_flow_to_samples_list(self, flow):
        self.flow_samples_list.append(flow)

    def _add_pred_to_samples_list(self, pred):
        self.preds_list.append(pred)

    def _reset_flow_samples_list(self):
        self._flow_samples_list_last = [f for f in self.flow_samples_list]
        self._preds_list_last = [y for y in self.preds_list]
        self.counterfactual_inputs = []
        self.flow_samples_list, self.preds_list = [], []
        self.shifts = []
        self._flow_corrs = None

    def _reset_flow_errors_list(self):
        mask = self.G.get_zeros_mask()
        with self.decorator:
            error = self.G.predict_error(x=self.G.x, mask=mask, frame=1)

        self._flow_errors = [error]

    def show_flow_error(self, flow_error):
        if self._show_error_diff:
            prev_flow_error = self._flow_errors[-1]
            self._flow_errors.append(flow_error.clone())            
            flow_error = prev_flow_error - flow_error
            vmin, vmax = flow_error.amin().item(), flow_error.amax().item()
        else:
            vmin, vmax = 0, flow_error.amax().item()

        self.imshow(img=flow_error[:,0],
                    ax=self.corr_ax,
                    cmap='RdBu_r',
                    vmin=vmin,
                    vmax=vmax)
        self.corr_ax.set_title('%s flow error | max=%0.1f' % \
                               (('diff' if self._show_error_diff else 'abs'), vmax),
                               fontsize=12)

    def _get_flow_mag(self, flow, normalize=True, dim=-3, eps=1e-2):
        flow_mag = flow.square().sum(dim).sqrt()
        if normalize:
            flow_mag = flow_mag - flow_mag.amin((-2,-1))
            flow_mag = flow_mag / flow_mag.amax((-2,-1)).clamp(min=eps)
        return flow_mag

    def show_last_segment(self, flow, ax=None, dim=-3):
        seg = self._get_flow_mag(flow, True)[:,0]
        img = self.get_input()[:,0] * seg
        if ax is None:
            ax = self.seg_ax
        self.imshow(ax=ax, img=img)
        
    def show_corrmat_segment(self, i=0, j=0, sample_inds=None, downsample=1):
        if len(self.flow_samples_list) == 0 or (self.corr_ax is None):
            return
        if sample_inds is None:
            sample_inds = range(len(self.flow_samples_list))
        samples = [self.flow_samples_list[i] for i in sample_inds]
        if len(samples) == 1:
            self.show_last_segment(samples[0], ax=self.corr_ax)
            return

        samples = torch.stack(samples, -1)[:,0]
        if (self._flow_corrs is None) or (self._num_flow_samples != samples.size(-1)):
            ## recompute
            self._flow_corrs = None
            self._flow_corrs = flow_generator.SingleObjectFlowBasedSelector.compute_flow_corrs(
                samples,
                downsample=downsample,
                use_covariance=True).relu()
            self._num_flow_samples = samples.size(-1)
        s = downsample or 1
        self.imshow(ax=self.corr_ax, img=self._flow_corrs[:,:,i//s,j//s])
        self.corr_ax.set_title('Covmat at [%d,%d] from %d flow samples' % (i, j, samples.size(-1)),
                               fontsize=10)

    def _get_corrmat(self,
                     num_samples=10,
                     num_active_patches=1,
                     num_passive_patches=1,
                     downsample=1,
                     use_covariance=True,
                     resample=False,
                     **kwargs):
        if self._flow_corrs is not None and not resample:
            return self._flow_corrs
        self._flow_corrs = None
        flow_samples = self.get_random_flow_samples(num_samples,
                                                    num_active_patches,
                                                    num_passive_patches,
                                                    **kwargs)
        self._flow_corrs = flow_generator.SingleObjectFlowBasedSelector.compute_flow_corrs(
            flow_samples,
            downsample=downsample,
            use_covariance=use_covariance,
            **kwargs).relu()
        self._num_flow_samples = flow_samples.size(-1)
        return self._flow_corrs
            
    def __call__(self, event):
        """Set the relevant patch inputs and then call the flow generator"""
        self.G.x = self.G.x.to(self.dtype)

        if self._show_flow and not str(event.key).upper() == 'CONTROL':
            self._show_flow = False
            self.G.set_input(self.get_input())
            if str(event.key).upper() == 'SHIFT':
                self._store_current_patches()
                self._reset_masks()

        i,j = self._get_patch_inds(event)
        if i is None or j is None:
            return
        txt = 'xdata=%d, ydata=%d, key=%s' % (j,i,event.key)
        self.txt = txt

        ## enable setting the shift by dragging
        self.do_drag = (str(event.key).upper() == 'D')
        self.press_loc = (event.xdata, event.ydata)
        if bool(event.dblclick) and self.do_drag:
            self.shift = self.press_loc = None
            self.do_drag = False
            self.imshow(self.ax, self._make_mask_img(), 'reset_shift')            
            return

        if (event.key is None) and not self.do_drag:
            self.active_patches = self._add_patch(
                i,j,mask=self.active_patches,t=((self.frame or 0)+1))
        elif (str(event.key).upper() == 'META') or ('RIGHT' in str(event.button).upper()):
            self.passive_patches = self._add_patch(
                i,j,mask=self.passive_patches, t=((self.frame or 0)+1))
        elif str(event.key).upper() == 'SHIFT':
            self._store_current_patches()
            self._reset_dmasks()
            self._reset_flow_samples_list()
            self._reset_flow_errors_list()
            self._corrmat_inds_list = []
        elif str(event.key).upper() == 'ALT':
            self._restore_last_patches()
            self.flow_samples_list = self._flow_samples_list_last
        elif (str(event.key).upper() == 'CONTROL') or (str(event.key).upper() == 'F'):
            self._show_flow = True
            shift = self.shift if self.shift is not None else self.sample_shift()
            self.shifts.append(shift)
            with self.decorator:
                y, flow = self._get_flow(
                    static=True,
                    active_patches=self.active_patches,
                    mask=self.passive_patches,
                    shift=shift,
                    fix_passive=False)
                self.out = [3,3]

                print("y", y.shape)
                self.y, self.flow = y, flow

            if flow is not None:
                if self._normalize_flow_magnitude:
                    self.flow2rgb.max_speed = flow.square().sum(-3).sqrt().amax()
                flow_rgb = self.flow2rgb(flow[:,0])
                self._add_flow_to_samples_list(flow)            

                self.imshow(self.flow_ax if self.flow_ax is not None else self.ax,
                            flow_rgb, txt='shift=%s, max flow=%0.1f' % (shift, self.flow2rgb.max_speed))

            self._add_pred_to_samples_list(y)
            self.counterfactual_inputs.append(self.masked_img.clone())
                
            if self.corr_ax is not None:
                self.imshow(self.corr_ax, y[:,-1])

            if flow is not None:
                self.show_last_segment(flow)
            self._store_current_patches()
        elif str(event.key).upper() == 'B': ## get a batch of flows
            assert hasattr(self.G, 'sample_flows_from_single_mask'), \
                "Your model wrapper must have a method 'sample_flows_from_single_mask'"
            with self.decorator:
                fs, actives, _ = self.G.sample_flows_from_single_mask(
                    x=self._x.to(self.dtype),
                    active_masks=self.active_patches,
                    passive_masks=self.passive_patches,
                    num_samples=self.sample_batch_size,
                    batch_size=self.max_samples_per_batch,                    
                    num_splits=1,
                    mask_head_motion=False,
                    static_head_motion=True,
                    **self._model_kwargs)
                fs_filter = getattr(self.G, 'flow_sample_filter')
                if fs_filter is not None:
                    fs, fs_mask = fs_filter(fs, actives)
                    num_filtered = fs_mask.amax((1,2,3)).sum().item()
                else:
                    num_filtered = 0
                    
                self.flow_samples_list.extend([f[:,None] for f in torch.unbind(fs, -1)])
                if self._normalize_flow_magnitude:
                    self.flow2rgb.max_speed = fs.square().sum(1).sqrt().amax()
                flow_rgbs = torch.stack([self.flow2rgb(flow) for flow in torch.unbind(fs, -1)],
                                        -1).sum(-1)
                self.imshow(ax=self.flow_ax,
                            img=flow_rgbs)

                fs_mag = fs.square().sum(1, True).sqrt().mean(-1)
                fs_mag = fs_mag - fs_mag.amin((-2,-1), True)
                fs_mag = fs_mag / fs_mag.amax((-2,-1), True).clamp(min=1e-3)
                img = self.get_input()[:,0] * fs_mag
                self.imshow(ax=self.seg_ax, img=img)
                txt = "filtered %d / %d samples" % (num_filtered, fs.size(-1))
                self.flow_ax.set_title(txt)                

        elif str(event.key).upper() == 'X': ## extract segment
            self._corrmat_inds_list.append([i,j])
            self.show_corrmat_segment(i, j, sample_inds=None)

        elif str(event.key).upper() == 'E': ## look at true flow and error
            mask = torch.minimum(self.active_patches, self.passive_patches)
            with self.decorator:
                error_dict = self.G.get_error_maps(x=self._x.to(self.dtype),
                                                   mask=mask,
                                                   head_motion=None,
                                                   mask_head_motion=False,
                                                   static_head_motion=False)
            if self.flow_ax is not None:
                self.G.flowshow(error_dict['flow_true'][:,0],
                                ax=self.flow_ax,
                                set_max_speed=True,
                                title='true flow',
                                fontsize=12)
            if self.seg_ax is not None:
                self.G.flowshow(error_dict['flow_pred'][:,0],
                                ax=self.seg_ax,
                                set_max_speed=False,
                                title='pred flow',
                                fontsize=12)

            self.show_flow_error(error_dict['flow_error'])
            self._show_flow = True
        elif str(event.key).upper() == 'T': ## run patch selection algorithm
            self.text.set_text('running patch selector...')
            with self.decorator:
                fs, actives, passives = self.patch_selector(
                    self._x[:,-1:].repeat(1,2,1,1,1).to(torch.float16),
                    make_static=True,
                    sample_batch_size=None,
                    init_actives=self.active_patches.clone(),
                    init_passives=self.passive_patches.clone()
                )
                self.flow_samples_list.extend([f[:,None] for f in torch.unbind(fs, -1)])
                affs, _, _  = self.patch_selector.compute_affinity_targets_from_samples(fs)

            _img = self.G.get_masked_pred_patches(self._x.clone(), actives.amin(-1), fill_value=[0,1,1])
            _img = self.G.get_masked_pred_patches(_img, passives[...,0], fill_value=[1,0,1])
            _img = self.G.get_masked_pred_patches(_img, self.active_patches, fill_value=[0,1,0])
            _img = self.G.get_masked_pred_patches(_img, self.passive_patches, fill_value=[1,0,0])
            self.imshow(img=_img[:,-1],
                        ax=self.corr_ax)
            self.corr_ax.set_title('iterated patches, blue=new')

            self.imshow(img=affs,
                        ax=self.flow_ax,
                        cmap='RdBu_r',
                        vmin=0,
                        vmax=1)

            _seg = self._x[:,-1] * affs
            self.imshow(img=_seg,
                        ax=self.seg_ax)
            self.seg_ax.set_title('%s segment' % type(self.patch_selector).__name__)
            if hasattr(self.G, 'flow_sample_filter'):
                txt = "filtered %d / %d samples" % (self.G.filter_masks.sum().item(), fs.size(-1))
                self.flow_ax.set_title(txt)
                                                    
                        

    def sample_random_patches(self, num_samples=10, num_visible=1):
        assert self.G.mask_generator is not None
        _num_vis = self.G.mask_generator.num_visible
        self.G.mask_generator.num_visible = num_visible
        masks = torch.stack([self.G.mask_generator() for _ in range(num_samples)], -1)
        if self.x is not None:
            masks = masks.to(self.x.device)
        self.G.mask_generator.num_visible = _num_vis
        return masks

    def get_random_flow_samples(self,
                                num_samples=10,
                                num_active_patches=1,
                                num_passive_patches=0,
                                **kwargs):
        
        active_patches = self.sample_random_patches(num_samples, num_active_patches)
        passive_patches = self.sample_random_patches(num_samples, num_passive_patches)

        h_static = self.G.get_static_imu()
        mask_context = torch.zeros(h_static.size(0), self.G.num_head_tokens).bool().to(h_static.device)
        h_static = self.G.head_motion_generator.reshape_output(h_static)
        kwargs.update(copy.deepcopy(self._model_kwargs))
        if 'timestamps' in kwargs.keys():
            kwargs['timestamps'] = torch.tile(kwargs['timestamps'], (num_samples, 1))

        self.G.reset_padding_masks()
        with self.decorator:
            _, flow_samples, _, _ = self.G.sample_counterfactual_flows_parallel(
                active_sampled=active_patches,
                passive_sampled=passive_patches,
                num_samples=num_samples,
                frame=1,
                get_original_flow=False,
                fix_passive=True,
                swap_active_passive=False,
                x_context=h_static.expand(num_samples, -1, -1),
                mask_context=mask_context.expand(num_samples, -1),
                **kwargs)
        return flow_samples

    def show_random_correlogram(self, i=0, j=0,
                                num_samples=10,
                                num_active_patches=1,
                                num_passive_patches=0,
                                resample=False,
                                batch_size=None,
                                **kwargs):
        if resample or (num_samples != self._num_flow_samples):
            self._flow_corrs, self._num_flow_samples = (None, None)
            batch_size = batch_size or num_samples
            self.flow_samples_list = []
            for b in range(num_samples // batch_size):
                flow_samples = self.get_random_flow_samples(batch_size,
                                                            num_active_patches,
                                                            num_passive_patches,
                                                            **kwargs)
                self.flow_samples_list.extend([f[:,None] for f in flow_samples.unbind(-1)])
                self.G.reset_padding_masks()


        self.show_corrmat_segment(i, j, sample_inds=None)

    def visualize_correlogram(self,
                              num_points=4,
                              inds_list=[],
                              use_stored_inds=True,
                              num_samples=10,
                              num_active_patches=1,
                              num_passive_patches=1,
                              resample=False,
                              overlay=False,
                              **kwargs):
        corrmat = self._get_corrmat(num_samples,
                                    num_active_patches,
                                    num_passive_patches,
                                    resample=resample,
                                    **kwargs)
        size = corrmat.shape[-4:-2]
        s = self.x.size(-2) // size[-2], self.x.size(-1) // size[-1]

        # choose the index points
        points = []
        for inds in inds_list[-num_points:]:
            points.append(inds)

        if use_stored_inds:
            remainder = num_points - len(points)
            if remainder > 0:
                points.extend(self._corrmat_inds_list[-remainder:])

        if len(points) < num_points:
            remainder = num_points - len(points)
            for k in range(remainder):
                points.append([
                    self.rng.randint(0, size[0]),
                    self.rng.randint(0, size[1])
                ])

        ## set up visualizer
        n_rows = max(2, num_points // 2)
        n_cols = 4

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 4))
        for idx, p in enumerate(points):
            row = idx // 2
            col = idx % 2
            corr_img = corrmat[:,:,p[0]//s[0],p[1]//s[1]]
            corr_img = corr_img - corr_img.amin((-2,-1))
            corr_img = corr_img / corr_img.amax((-2,-1)).clamp(min=1e-3)
            _img = self.G.get_masked_pred_patches(self.G.x,
                                                  self.G.generate_mask_from_patch_idx_list([p]),
                                                  fill_value=[1,0,1])[:,1]

            if overlay:
                vis_tensor(corr_img * _img, ax=axes[row, col*2])
            else:
                vis_tensor(_img, ax=axes[row, col*2])

            vis_tensor(corrmat[:,:,p[0]//s[0],p[1]//s[1]], ax=axes[row, col*2 + 1])
            axes[row,col*2].set_xticks([])
            axes[row,col*2].set_yticks([])
            axes[row,col*2+1].set_xticks([])
            axes[row,col*2+1].set_yticks([])

        plt.suptitle('/'.join(self.G._predictor_load_path.split('/')[-2:]), fontsize=16, va='bottom')
        plt.tight_layout()
        plt.show()
        return points
