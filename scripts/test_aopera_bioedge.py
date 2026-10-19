# %%

import matplotlib.pyplot as plt
import numpy as np

from aopera.ffwfs import compute_bioedge_masks

# %% Parameters

nx = 10  # number of pixels for simulation
grey_width = 6.0  # half-width of the grey area in pixels
grey_length = None  # half-length of the grey area in pixels (None for full mask)

# %% Compute the BioEdge masks

masks = compute_bioedge_masks(nx, grey_width, grey_length)

try:
    import cupy as cp

    masks = [mask.get() for mask in masks]  # Convert to Numpy arrays
except:
    pass  # If CuPy is not available, continue with NumPy arrays

fig, axes = plt.subplots(2, 2, constrained_layout=True)
axes_flat = axes.flatten()
for index, ax in enumerate(axes_flat):
    mask = masks[index]
    im = ax.imshow(mask**2, cmap="gray")
    ax.set_title(f"Mask {index + 1}")

# %% Check

total_transmission = np.sum([np.sum(mask**2) for mask in masks]) / (nx**2)
print(f"Total transmission: {total_transmission:.2f}")

# %%
