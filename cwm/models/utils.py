import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torch.distributions.categorical import Categorical
from torch.distributions.bernoulli import Bernoulli

import kornia

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

def imagenet_normalize(x, temporal_dim=1):
    mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(x.device)[None,None,:,None,None].to(x)
    std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(x.device)[None,None,:,None,None].to(x)
    if temporal_dim == 2:
        mean = mean.transpose(1,2)
        std = std.transpose(1,2)
    return (x - mean) / std

def imagenet_unnormalize(x, temporal_dim=2):
    device = x.device
    mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(device)[None, None, :, None, None].to(x)
    std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(device)[None, None, :, None, None].to(x)
    if temporal_dim == 2:
        mean = mean.transpose(1,2)
        std = std.transpose(1,2)
    x = x*std + mean
    return x


def convex_upsample(x, mask, upsample_factor=8):
    U = upsample_factor
    B,_,H,W = x.shape
    mask = mask.view(B,1,-1,U,U,H,W)
    mask = torch.softmax(mask, dim=2)
    k = int(np.sqrt(mask.shape[2]))

    x = F.unfold(U * x, [k,k], padding=(k // 2)) # patches
    x = x.view(B, -1, k**2, 1, 1, H, W)
    x = torch.sum(x * mask, dim=2) # convex combination of pixels in local kernel
    x = x.permute(0,1,4,2,5,3) # [B,C,H,U,W,U]
    x = x.reshape(B,-1,H*U,W*U)

    return x

def boltzmann(x, beta=1, eps=1e-9):
    if beta is None:
        return x
    x = torch.exp(x * beta)
    return x / x.amax((-1,-2), keepdim=True).clamp(min=eps)

def spatial_moments_from_local_dist(local_dist, eps=1e-3, squeeze=True):
    """
    local_dist: [B,K,H,W]
    moments: [B,C,2,H,W] or [B,2,H,W] if C == 1 and squeeze == True
    """
    if len(local_dist.shape) == 4:
        local_dist = local_dist[:,None]
    B,C,K,H,W = local_dist.shape
    k = int(np.sqrt(K))    
    norm = local_dist.sum(-3, True).clamp(min=eps) # [B,C,1,H,W]

    grid = coordinate_ims(1, 1, [k,k], normalize=True).to(local_dist.device) # [1,1,k,k,2]
    local_dist = local_dist.permute(0,3,4,1,2).reshape(B*H*W,C,k,k)
    moments = torch.einsum('nchw,nchwd->ncd', local_dist, grid).view(B,H,W,C,2).permute(0,3,4,1,2)
    moments = moments / norm
    if (C == 1) and squeeze:
        return moments[:,0] # [B,2,H,W]
    return moments

def get_distribution_centroid(dist, eps=1e-9, normalize=False):

    B,T,C,H,W = dist.shape
    assert C == 1
    dist_sum = dist.sum((-2, -1), keepdim=True).clamp(min=eps)
    dist = dist / dist_sum

    grid = coordinate_ims(B, T, [H,W], normalize=normalize).to(dist.device)
    grid = grid.permute(0,1,4,2,3)
    centroid = (grid * dist).sum((-2,-1))
    return centroid

def coordinate_ims(batch_size, seq_length, imsize, normalize=True, dtype_out=torch.float32):
    static = False
    if seq_length == 0:
        static = True
        seq_length = 1
    B = batch_size
    T = seq_length
    H,W = imsize
    ones = torch.ones([B,H,W,1], dtype=dtype_out)
    if normalize:
        h = torch.divide(torch.arange(H).to(ones), torch.tensor(H-1, dtype=dtype_out))
        h = 2.0 * ((h.view(1, H, 1, 1) * ones) - 0.5)
        w = torch.divide(torch.arange(W).to(ones), torch.tensor(W-1, dtype=dtype_out))
        w = 2.0 * ((w.view(1, 1, W, 1) * ones) - 0.5)
    else:
        h = torch.arange(H).to(ones).view(1,H,1,1) * ones
        w = torch.arange(W).to(ones).view(1,1,W,1) * ones
    h = torch.stack([h]*T, 1)
    w = torch.stack([w]*T, 1)
    hw_ims = torch.cat([h,w], -1)
    if static:
        hw_ims = hw_ims[:,0]
    return hw_ims

def sample_image_inds_from_probs(probs, num_points,
                                 eps=1e-9, normalize=False, seed=0):

    B,H,W = probs.shape
    P = num_points
    N = H*W

    probs = probs.reshape(B,N)
    if normalize:
        probs = probs - probs.amin(-1, True)
    probs = F.relu(probs + eps)
    probs = probs / probs.to(probs.dtype).sum(dim=-1, keepdim=True).clamp(min=eps)
    dist = Categorical(probs=probs)
    indices = dist.sample([P]).permute(1,0).to(torch.long) # [B,P]

    indices_h = torch.minimum(torch.maximum(torch.div(indices, W, rounding_mode='floor'), torch.tensor(0)), torch.tensor(H-1))
    indices_w = torch.minimum(torch.maximum(torch.fmod(indices, W), torch.tensor(0)), torch.tensor(W-1))
    indices = torch.stack([indices_h, indices_w], dim=-1) # [B,P,2]
    return indices

def sample_from_energy(probs,
                       num_points=1,
                       num_samples=1,
                       binarize=False,
                       normalize=False,
                       eps=1e-9):

    shape = probs.shape
    if len(shape) == 5:
        B,T,_,H,W = shape
    elif len(shape) == 4:
        B,_,H,W = shape
        T = 1
        probs = probs[:,None]
    else:
        raise ValueError(probs.shape)

    assert probs.size(-3) == 1, probs.shape

    S = num_samples
    P = num_points

    probs = probs.unsqueeze(1).expand(-1,S,-1,-1,-1,-1)
    probs = probs.reshape(B*S*T,H,W)
    sample_inds = sample_image_inds_from_probs(
        probs, P, eps=eps, normalize=normalize)
    sample_energy = index_into_images(probs[:,None], sample_inds)
    if binarize:
        sample_energy = torch.ones_like(sample_energy)

    b_inds = torch.arange(B*S*T, dtype=torch.long)[:,None].expand(-1, P).to(sample_inds.device)
    inds = torch.stack([b_inds, sample_inds[...,0].to(torch.long), sample_inds[...,1].to(torch.long)], 0)
    inds = [ix.flatten() for ix in list(inds)]

    activated = torch.zeros_like(probs)
    activated[inds] = sample_energy.flatten()
    activated = activated.view(B*S, T, 1, H, W)

    if len(shape) == 4:
        activated = activated[:,0]
        
    return activated

def sample_per_pixel(probs, eps=1e-9):
    probs = F.relu(probs).clamp(max=1.0)
    dist = Bernoulli(probs=probs)
    sample = dist.sample()
    return sample

def index_into_images(images, indices, channels_last=False):
    """
    index into an image at P points to get its values

    images: [B,C,H,W]
    indices: [B,P,2]
    """
    assert indices.size(-1) == 2, indices.size()
    if channels_last:
        images = images.permute(0,3,1,2) # [B,C,H,W]
    B,C,H,W = images.shape
    _,P,_ = indices.shape
    inds_h, inds_w = list(indices.to(torch.long).permute(2,0,1)) # [B,P] each
    inds_b = torch.arange(B, dtype=torch.long).unsqueeze(-1).expand(-1,P).to(indices)
    inds = torch.stack([inds_b, inds_h, inds_w], 0).to(torch.long)
    values = images.permute(0,2,3,1)[list(inds)] # [B,P,C]
    return values

def soft_index(images, indices, scale_by_imsize=True):
    assert indices.shape[-1] == 2, indices.shape
    B,C,H,W = images.shape
    _,P,_ = indices.shape

    # h_inds, w_inds = indices.split([1,1], dim=-1)
    h_inds, w_inds = list(indices.permute(2,0,1))
    if scale_by_imsize:
        h_inds = (h_inds + 1.0) * torch.tensor(H).to(h_inds) * 0.5
        w_inds = (w_inds + 1.0) * torch.tensor(W).to(w_inds) * 0.5

    h_inds = torch.maximum(torch.minimum(h_inds, torch.tensor(H-1).to(h_inds)), torch.tensor(0.).to(h_inds))
    w_inds = torch.maximum(torch.minimum(w_inds, torch.tensor(W-1).to(w_inds)), torch.tensor(0.).to(w_inds))

    h_floor = torch.floor(h_inds)
    w_floor = torch.floor(w_inds)
    h_ceil = torch.ceil(h_inds)
    w_ceil = torch.ceil(w_inds)

    bot_right_weight = (h_inds - h_floor) * (w_inds - w_floor)
    bot_left_weight = (h_inds - h_floor) * (w_ceil - w_inds)
    top_right_weight = (h_ceil - h_inds) * (w_inds - w_floor)
    top_left_weight = (h_ceil - h_inds) * (w_ceil - w_inds)

    in_bounds = (bot_right_weight + bot_left_weight + top_right_weight + top_left_weight) > 0.95
    in_bounds = in_bounds.to(torch.float32)

    top_left_vals = index_into_images(images, torch.stack([h_floor, w_floor], -1))
    top_right_vals = index_into_images(images, torch.stack([h_floor, w_ceil], -1))
    bot_left_vals = index_into_images(images, torch.stack([h_ceil, w_floor], -1))
    bot_right_vals = index_into_images(images, torch.stack([h_ceil, w_ceil], -1))

    im_vals = top_left_vals * top_left_weight[...,None]
    im_vals += top_right_vals * top_right_weight[...,None]
    im_vals += bot_left_vals * bot_left_weight[...,None]
    im_vals += bot_right_vals * bot_right_weight[...,None]

    im_vals = im_vals.view(B,P,C)

    return im_vals

def get_local_neighbors(im, size=None, radius=3, invalid=-1, to_image=False):
    shape = im.shape
    if len(shape) == 2 and (size is not None):
        B,N = shape
        assert len(size) == 2, size
        H,W = size
        C = 1
        assert N == H*W, (N, H, W, H*W)
        im = im.view(B,1,H,W)
    elif len(shape) == 3 and (size is not None):
        B,C,N  = shape
        assert len(size) == 2, size
        H,W = size
        assert N == H*W, (N, H, W, H*W)
        im = im.view(B,C,H,W)
    else:
        assert len(shape) == 4 and ((size is None) or (size == list(shape[-2:]))), (shape, size)
        B,C,H,W = shape

    ## pad the input tensor
    local_k = 2*radius + 1
    out = F.pad(im, (radius, radius, radius, radius), "constant", invalid).float()
    out = F.unfold(out, (local_k, local_k)).to(im)
    if to_image:
        out = out.view(B,C,-1,H,W)
    else:
        out = out.view(B,C,-1,H*W)
    return out

def get_patches(x, radius=1):
    if radius == 0:
        return x
    shape = x.shape
    if len(shape) == 5:
        x = x.view(shape[0]*shape[1], *shape[2:])
    B,C,H,W = x.shape
    x = F.pad(x, (radius, radius, radius, radius))
    k = 2*radius + 1
    x = F.unfold(x, (k, k)).view(-1, C*(k**2), H, W)
    if len(shape) == 5:
        x = x.view(shape[0], shape[1], *x.shape[1:])
    return x

def spatial_moments_to_circular_target(moments,
                                       beta=10.0):
    circle = coordinate_ims(1, 0, [3,3], normalize=True).to(moments.device)
    circle = circle.permute(0,3,1,2).view(1,2,9,1,1)
    _n = lambda x: F.normalize(x, p=2, dim=1)
    dots = (_n(moments[:,:,None]) * _n(circle)).sum(1)
    if beta is None:
        eightway = F.one_hot(dots.argmax(1), num_classes=9)
        return eightway.permute(0,3,1,2).float()
    else:
        eightway = (dots * beta).softmax(dim=1)
        return eightway

def circular_target_to_spatial_moment(target):
    assert target.shape[1] == 8, target.shape
    clock = torch.tensor([
        [-1, -1], [0, -1], [0, 1],
        [0, -1], [0, 1],
        [1, -1], [1, 0], [1, 1]]).float().to(target.device)
    clock = clock.view(1, 8, 2, 1, 1)
    circular = (target[:,:,None] * clock).sum(1)
    return circular

def estimate_boundary_orientations(boundaries,
                                   energy,
                                   radius=3,
                                   to_circle=False,
                                   beta=10.0,
                                   eps=1e-3):
    B,_,H,W = boundaries.shape
    local_energy = get_local_neighbors( # [B,K,H,W]
        energy * (1 - boundaries),
        size=[H,W], radius=radius, invalid=0, to_image=True)[:,0]
    num_px = local_energy.sum(1, True)
    K,k = local_energy.shape[1], int(np.sqrt(local_energy.shape[1]))
    local_grid = coordinate_ims(1, 0, [k,k], normalize=True).to(energy.device)
    local_energy = local_energy.permute(0,2,3,1).reshape(B*H*W,1,k,k)
    orientations = (local_energy * local_grid.permute(0,3,1,2)).sum((-2,-1))
    orientations = orientations.view(B,H,W,2).permute(0,3,1,2)
    orientations = orientations / num_px.clamp(min=eps)
    if not to_circle:
        return orientations

    circle = coordinate_ims(1, 0, [3,3], normalize=True).to(energy.device)
    circle = circle.permute(0,3,1,2).view(1,2,9,1,1)
    _n = lambda x: F.normalize(x, p=2, dim=1)
    dots = (_n(orientations[:,:,None]) * _n(circle)).sum(1)
    if beta is None:
        eightway = F.one_hot(dots.argmax(1), num_classes=9)
        return eightway.permute(0,3,1,2).float()
    else:
        eightway = (dots * beta).softmax(dim=1)
        return eightway


def compute_local_effects(source, adj_local):
    """
    Splat effects from each source point multiplied by local affinity matrix

    source: [B,C,H,W]
    adj_local: [B,K,H,W]
    """
    B,D,H,W = source.shape
    K,_H,_W = adj_local.shape[-3:]
    assert (H == _H) and (W == _W), (source.shape, adj_local.shape)

    ## patch size
    k = int(np.sqrt(K))
    assert k**2 == K, "Local adjacency must be square, but K = %d" % K
    r = (k - 1) // 2

    # local_effects = source.view(-1,C,1,H,W) * adj_local.view(-1,1,K,H,W)
    if len(source.shape) == 4 and len(adj_local.shape) == 4:
        C = None
        local_effects = source[:,:,None] * adj_local[:,None] # [B,D,K,H,W]
        try:
            local_effects = local_effects.view(B,D*K,H*W)
        except:
            local_effects = local_effects.reshape(B,D*K,H*W)
    elif len(source.shape) == 4 and len(adj_local.shape) == 5:
        C = adj_local.shape[1]
        local_effects = source[:,:,None,None] * adj_local[:,None]
        local_effects = local_effects.view(B,D*C*K,H*W)
    elif len(source.shape) == 5 and len(adj_local.shape) == 5:
        raise NotImplementedError("Both source and messages have channel dims")

    fold = nn.Fold(output_size=[H,W], kernel_size=[k,k], padding=[r,r])
    local_effects = fold(local_effects) # [B,D*C,H,W]

    return local_effects

def local_average(values, excluded, radius=1):

    B,C,H,W = values.shape
    neighbors = get_local_neighbors(
        values * (1 - excluded), radius=radius, invalid=0, to_image=True) # [B,C,K,H,W]
    norm = get_local_neighbors(
        1 - excluded, radius=radius, invalid=0, to_image=True).sum(-3).clamp(min=1)
    return neighbors.sum(-3) / norm

def get_mask_boundaries(masks, Shift=None):
    B,K,H,W = masks.shape
    masks = (masks > 0.5).float()
    if Shift is None:
        Shift = ConvLocalShifts(local_radius=1, dilation=1)
    if 'cuda' in masks.device.type:
        Shift.cuda()
    shifted, _ = Shift(masks)
    boundaries = (shifted != shifted[...,Shift.null_idx,None]).float().amax(-1)
    boundaries *= masks
    return (boundaries, Shift)

def video_to_images(video, dim=1):
    return torch.unbind(video, dim)

def video_to_frame_pairs(video, frame=None, dim=1):
    images = torch.unbind(video, dim)
    num_frames = len(images)
    frame = frame or (num_frames - 1) // 2
    target_ts = [t for t in range(num_frames) if t != frame]

    pairs = [
        torch.stack([images[frame], images[t]], dim=dim)
        for t in target_ts]
    return pairs

class VideoToFramePairs(nn.Module):
    def __init__(self, frame=None, dim=1):
        super().__init__()
        self.frame = frame
        self.dim = dim
    def forward(self, video):
        return video_to_frame_pairs(
            video, frame=self.frame, dim=self.dim)

class Normalize(nn.Module):
    def __init__(self, dim=1, p=2):
        super().__init__()
        self.dim = dim
        self.p = p

    def forward(self, x):
        return F.normalize(x, p=self.p, dim=self.dim)

class GlobalAvgPool2d(nn.Module):
    def forward(self, x):
        kernel = list(x.shape[-2:])
        global_pool = nn.AvgPool2d(kernel)
        return global_pool(x)

class GlobalMaxPool2d(nn.Module):
    def forward(self, x):
        kernel = list(x.shape[-2:])
        global_pool = nn.MaxPool2d(kernel)
        return global_pool(x)

def normalize_input(x, norm=255.):
    if (x.dtype == torch.uint8) or (x.amax() > 1.0):
        return x.float() / norm
    else:
        return x.float()

class NormalizeInput(nn.Module):
    def __init__(self, norm=255.):
        super().__init__()
        self.norm = norm
    def forward(self, x):
        return normalize_input(x, self.norm)

def channel_mse(x, y, dim=-3):
    err = torch.sqrt((x-y).square().mean(dim, True).float())
    err = err.to(x.dtype)
    return err

class ChannelMSE(nn.Module):
    def __init__(self, dim=-3):
        super().__init__()
        self.dim = dim
    def forward(self, x, y, dim=None):
        dim = dim or self.dim
        return channel_mse(x, y, dim)

def channel_l1error(x, y, dim=-3):
    return (x-y).abs().mean(dim, True)

class ChannelL1Error(nn.Module):
    def __init__(self, dim=-3):
        super().__init__()
        self.dim = dim
    def forward(self, x, y, dim=None):
        dim = dim or self.dim
        return channel_l1error(x, y, dim)

class ChannelL2Error(nn.Module):
    def __init__(self, dim=-3):
        super().__init__()
        self.dim = dim
    def forward(self, x, y, dim=None):
        dim = dim or self.dim
        return (x - y).square().mean(dim, True)

def max_delta_error(x, y, dim=-3, backward=False):
    sign = -1 if backward else 1
    return F.relu(sign * (x - y)).abs().amax(dim=dim, keepdim=True)

class MaxDeltaError(nn.Module):
    def __init__(self, dim=-3, backward=False):
        super().__init__()
        self.dim = dim
        self.backward = backward
    def forward(self, x, y, dim=None):
        dim = dim or self.dim
        return max_delta_error(x, y, dim=dim, backward=self.backward)
        
class ConvLocalShifts(nn.Module):

    def __init__(self,
                 local_radius,
                 invert_order=True,
                 pad_value=0,
                 dilation=1,
                 disocclusion_thresh=0.9):
        super().__init__()
        self.local_radius = self.r = local_radius
        self.k = 2*self.r + 1
        self.K = self.k**2
        self.null_idx = (self.K // 2)
        self.kernel = None
        self.dilation = dilation

        # make shifts look like active motion rather htan passive head motion
        self.invert_order = invert_order
        self.pad_value = pad_value

        # disocclusion
        self.disocclusion_thresh = disocclusion_thresh

    def _preprocess(self, img):
        shape = list(img.shape)
        self._temporal = False
        if len(shape) == 5:
            self._temporal = True
            B,T,C,H,W = shape
            img = img.view(B*T, *shape[2:])
        elif len(shape) == 4:
            T = 1
            B,C,H,W = shape
        elif len(shape) == 3:
            B = T = 1
            C,H,W = shape
            img = img[None]
        else:
            raise ValueError("input doesn't have a valid shape: %s" % shape)

        self.shape = shape
        self.B, self.T, self.C, self.H, self.W = B, T, C, H, W
        return img

    def _set_shift_kernel(self):
        if self.kernel is not None:
            return
        kernel = torch.arange(self.K).long()
        if self.invert_order:
            kernel = torch.flip(kernel, (0,))
        self.kernel = F.one_hot(kernel, num_classes=self.K).view(self.K,1,self.k,self.k).float()

    def _index_kernel(self, kernel, idx=None):
        if idx is None:
            return kernel
        if not isinstance(idx, torch.Tensor):
            idx = torch.tensor(idx, device=kernel.device)
            
        if idx.shape[-1] == 1:
            assert idx < self.K, "No kernel for shifting by %d" % idx
            return kernel[idx:idx+1]
        elif idx.shape[-1] == 2:
            _idx = idx + self.r
            _idx = _idx[...,0] * self.k + _idx[...,1]
            return kernel[_idx:_idx+1]
        else:
            raise ValueError("can't index into kernel of shape %s with idx=%s" % kernel.shape, idx)

    def _get_shifts(self, x, idx=None):
        self._set_shift_kernel()
        C = x.shape[-3]
        kernel = self._index_kernel(self.kernel.to(x.dtype).to(x.device), idx=idx)        
        kernel = kernel.repeat(C, 1, 1, 1)
        x = F.conv2d(x, kernel,
                     groups=C,
                     dilation=self.dilation,
                     padding='same')
        x = x.view(self.B*self.T, C, (1 if (idx is not None) else self.K), self.H, self.W)
        return x.permute(0,1,3,4,2)

    def forward(self, img, idx=None):
        img = self._preprocess(img)
        img = self._get_shifts(img, idx=idx)
        mask = self._get_shifts(
            torch.ones((self.B*self.T, 1, self.H, self.W)).to(img.device).to(img.dtype), idx=idx)

        return (
            img.view(*(self.shape + ([self.K] if (idx is None) else []))),
            mask.view(*(self.shape[:-3] + [1] + self.shape[-2:] + ([-1] if (idx is None) else [])))
        )

class ConsistentTarget(nn.Module):

    def __init__(self, thresh=0.5):
        super().__init__()
        self.thresh = thresh
        
    def forward(self, x_list):
        assert len(x_list) >= 2, len(x_list)
        if x_list[0].dtype in [torch.bool, torch.int32, torch.long]:
            target = sum([x.float() for x in x_list]) / len(x_list)
            if self.thresh is not None:
                target = (target >= self.thresh).float()
        else:
            raise NotImplementedError("consistency for real valued")

        return target

def l2_loss(x, y):
    return (x - y).square()

def l1_loss(x, y):
    return (x - y).abs()

class CharbonnierLoss(nn.Module):

    def __init__(self, eps=1e-3, alpha=0.5):
        super().__init__()
        self.eps = eps
        self.alpha = alpha

    def forward(self, x, y):
        return torch.pow((x - y).square() + self.eps**2, self.alpha).sum(-3, True)
    
class MaskedPerPixelLoss(nn.Module):
    def __init__(self, loss_fn=l2_loss):
        super().__init__()
        self.per_px_loss_fn = loss_fn

    def forward(self, logits, labels, mask):
        if mask is None:
            mask = torch.ones_like(labels[...,0:1,:,:])
        num_px = mask.detach().sum((-2,-1)).clamp(min=1)
        loss = self.per_px_loss_fn(logits, labels) * mask.detach()
        loss = loss.sum((-2,-1)) / num_px
        loss = loss.mean()
        return loss

MaskedL1Loss = MaskedPerPixelLoss(l1_loss)
MaskedL2Loss = MaskedPerPixelLoss(l2_loss)
MaskedCharbLoss = MaskedPerPixelLoss(CharbonnierLoss())


class MaskedBCELoss(nn.Module):
    def __init__(self, with_logits=False):
        super().__init__()
        if with_logits:
            self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        else:
            self.loss_fn = nn.BCELoss(reduction='none')

    def forward(self, logits, labels, mask):
        num_valid = mask.sum((-3,-2,-1)).clamp(min=1).detach()
        loss = self.loss_fn(logits, labels) * mask.detach()
        loss = loss.sum((-3,-2,-1)) / num_valid
        return loss

def weighted_softmax(x, mask, dim=-1, eps=1e-12):
    maxes = x.amax(dim, True)
    x_exp = torch.exp(x - maxes)
    x_exp_sum = (x_exp * mask).sum(dim, True) + eps
    return (x_exp / x_exp_sum) * mask
    
class MaskedKLDivLoss(nn.Module):
    def __init__(self, dim=-1, eps=1e-9):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, logits, labels, mask):
        B,K,H,W = logits.shape
        N = H*W
        logits = logits.view(B,K,N).transpose(1,2) # [B,N,K]
        labels = labels.view(B,K,N).transpose(1,2)
        mask = mask.view(B,K,N).transpose(1,2)

        logits = weighted_softmax(logits, mask, dim=self.dim, eps=self.eps)
        logits = logits.clamp(min=self.eps).log()
        labels = (labels * mask) / (labels * mask).sum(self.dim, True).clamp(min=self.eps)
        
        loss = F.kl_div(logits, labels, reduction='none') * mask
        loss = loss.sum(-1) # [B,N]

        num_valid = (mask.sum(-1) > 0).float().sum(1).clamp(min=1)
        loss = loss.sum(1) / num_valid
        return loss

class MaskedSequenceLoss(nn.Module):
    def __init__(self, gamma=0.8, loss_func=MaskedL1Loss):
        super().__init__()
        self.gamma = gamma
        self.loss_func = loss_func

    def forward(self, logits, labels, mask):
        if not isinstance(logits, (list, tuple)):
            logits = [logits]
        n_preds = len(logits)
        loss = 0.0
        for it in range(n_preds):
            it_loss = self.loss_func(logits[it], labels, mask)
            it_weight = self.gamma**(n_preds - it - 1)
            loss += it_loss * it_weight
        return loss

def confidence_thresh_samples(x, value_thresh=0.0, confidence_thresh=0.5, dim=-1):
    if isinstance(x, (list, tuple)):
        x = torch.stack(x, dim=dim)
    if value_thresh is not None:
        x = (x > value_thresh).float()
    else:
        x = x.float()
    return (x.mean(dim=dim) >= confidence_thresh)

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

class ConfidenceThresh(nn.Module):
    def __init__(self,
                 confidence_thresh=0.5,
                 value_thresh=0.0,
                 to_float=True,
                 dim=-1
    ):
        super().__init__()
        self.confidence_thresh = confidence_thresh
        self.value_thresh = value_thresh
        self.to_float = to_float
        self.dim = dim
        
    def forward(self, x):
        x = confidence_thresh_samples(
            x, dim=self.dim,
            value_thresh=self.value_thresh,
            confidence_thresh=self.confidence_thresh)

        return x.float() if self.to_float else x

if __name__ == '__main__':

    logits = torch.tensor(
        [1,1,1,0,0,0,0.5,0.5,0.5]).cuda().float().view(1,9,1,1) * 10
    labels = torch.tensor(
        [1,1,1,0,0,0,1,1,1]).cuda().float().view(1,9,1,1)
    mask = torch.tensor(
        [1,1,1,1,1,1,0,0,0]).cuda().float().view(1,9,1,1)
    
    loss_fn = MaskedKLDivLoss().cuda()
    loss = loss_fn(logits, labels, mask)
    print(loss)
