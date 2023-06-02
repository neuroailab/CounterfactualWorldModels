import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import matplotlib
import matplotlib.pyplot as plt

from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage

def imshow(ims,
           ax=None,
           ex=0,
           t=0,
           vmin=None,
           vmax=None,
           title=None,
           cmap=None,
           fontsize=20):
    if ax is None:
        fig, ax = plt.subplots(1,1)
    with torch.no_grad():
        if len(ims.shape) == 5:
            ims = ims[ex]
        elif len(ims.shape) == 3:
            ims = ims[None]
            t = 0
        im = ims[t].float().cpu().numpy().transpose((1,2,0))
    if (vmin is not None) and (vmax is not None):
        im =ax.imshow(im, vmin=vmin, vmax=vmax, cmap=(cmap or 'viridis'))
    else:
        im =ax.imshow(im)

    if title is not None:
        ax.set_title(title, fontsize=fontsize)

    return (im, ax)

