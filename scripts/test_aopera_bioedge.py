# %%

import matplotlib.pyplot as plt
import numpy as np

from aopera.ffwfs import compute_bioedge_masks

# %% Parameters

nx = 512  # number of pixels for simulation
grey_width = 10.0  # half-width of the grey area in pixels
grey_length = None  # half-length of the grey area in pixels (None for full mask)

# %% Compute the BioEdge masks

import cupy as cp

masks = compute_bioedge_masks(nx, grey_width, grey_length)

fig, axes = plt.subplots(2, 2)
axes_flat = axes.flatten()
for index, ax in enumerate(axes_flat):
    mask = masks[index].get() if isinstance(masks[index], cp.ndarray) else masks[index]
    im = ax.imshow(mask, cmap="gray")
    ax.set_title(f"Mask {index + 1}")
