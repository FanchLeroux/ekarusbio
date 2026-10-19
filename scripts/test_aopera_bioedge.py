# %%

import matplotlib.pyplot as plt
import numpy as np

from aopera.ffwfs import compute_bioedge_masks

# %% Parameters

nx = 10  # number of pixels for simulation
grey_width = 6.0  # half-width of the grey area in pixels
grey_length = None  # half-length of the grey area in pixels (None for full mask)

markersize = 8

# %% Compute the BioEdge masks

masks = compute_bioedge_masks(nx, grey_width, grey_length)

try:
    import cupy as cp

    masks = [mask.get() for mask in masks]  # Convert to Numpy arrays
except:
    pass  # If CuPy is not available, continue with NumPy arrays

# %% Check

fig, axes = plt.subplots(2, 2, constrained_layout=True)
axes_flat = axes.flatten()
for index, ax in enumerate(axes_flat):
    mask = masks[index]
    im = ax.imshow(mask**2, cmap="gray")
    ax.set_title(f"Mask {index + 1}")

plt.figure()
plt.plot(masks[0][nx // 2, :], "o", label="left_1_right_0")
plt.plot(masks[1][nx // 2, :], "o", label="left_0_right_1")
plt.plot(masks[2][:, nx // 2], "+", markersize=markersize, label="top_0_bottom_1")
plt.plot(masks[3][:, nx // 2], "+", markersize=markersize, label="top_1_bottom_0")
plt.xlabel("Pixel")
plt.ylabel("Field transmission")
plt.legend()

plt.figure()
plt.plot(masks[0][nx // 2, :] ** 2, "o", label="left_1_right_0")
plt.plot(masks[1][nx // 2, :] ** 2, "o", label="left_0_right_1")
plt.plot(masks[2][:, nx // 2] ** 2, "+", markersize=markersize, label="top_0_bottom_1")
plt.plot(masks[3][:, nx // 2] ** 2, "+", markersize=markersize, label="top_1_bottom_0")
plt.xlabel("Pixel")
plt.ylabel("Intensity transmission")
plt.legend()

transmissions = [np.sum(mask**2) / (nx**2) for mask in masks]
print(f"Transmission left_1_right_0: {transmissions[0]:.2f}")
print(f"Transmission left_0_right_1: {transmissions[1]:.2f}")
print(f"Transmission top_0_bottom_1: {transmissions[2]:.2f}")
print(f"Transmission top_1_bottom_0: {transmissions[3]:.2f}\n")

total_transmission = np.sum([np.sum(mask**2) for mask in masks]) / (nx**2)
print(f"Total transmission: {total_transmission:.2f}")

plt.show()

# %%

from aopera.ffwfs import ffwfs_sensitivity, modulation

modulation_path = modulation(nx, 0.0, 1.0).get()

psf = np.abs(np.fft.fftshift(np.fft.fft2((np.ones((nx, nx)))))) ** 2

sensitivity = ffwfs_sensitivity(
    masks[0],
    modulation_path,
    psf,
    psf,
).get()
