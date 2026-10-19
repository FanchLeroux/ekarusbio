# %%

import matplotlib.pyplot as plt
import numpy as np

from aopera.backend import set_backend

set_backend(mode="cpu")

from aopera.ffwfs import compute_bioedge_masks

# %% Parameters

nx = 64  # number of pixels for simulation
grey_width = 12.0  # half-width of the grey area in pixels
grey_length = None  # half-length of the grey area in pixels (None for full mask)

markersize = 8

# %% Compute the Bi-O edge masks

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

from aopera.otfpsf import otf_diffraction, otf2psf

samp = 2

otf_diff = otf_diffraction(nx, occ=0, samp=samp)
psf_diff = otf2psf(otf_diff)

psf_calib = psf_diff
psf_current = psf_diff

modulation_path = modulation(nx, 0.0, samp).get()

sensitivity_left_1_right_0 = ffwfs_sensitivity(
    masks[0],
    modulation_path,
    psf_calib,
    psf_current,
).get()
sensitivity_left_0_right_1 = ffwfs_sensitivity(
    masks[1],
    modulation_path,
    psf_calib,
    psf_current,
).get()
sensitivity_top_1_bottom_0 = ffwfs_sensitivity(
    masks[2],
    modulation_path,
    psf_calib,
    psf_current,
).get()
sensitivity_top_0_bottom_1 = ffwfs_sensitivity(
    masks[3],
    modulation_path,
    psf_calib,
    psf_current,
).get()

sensitivity_total = (
    sensitivity_left_1_right_0
    + sensitivity_left_0_right_1
    + sensitivity_top_1_bottom_0
    + sensitivity_top_0_bottom_1
)

plt.figure()
plt.imshow(sensitivity_total, cmap="hot")

fig, axes = plt.subplots(1, 2, constrained_layout=True)
axes[0].plot(sensitivity_total[nx // 2, nx // 2 :], label="Horizontal")
axes[0].plot(sensitivity_total[nx // 2 :, nx // 2], linestyle=":", label="Vertical")
axes[0].plot(
    [sensitivity_total[nx // 2 + k, nx // 2 + k] for k in range(nx // 2)],
    label="Diagonal",
)
axes[0].axhline(1, color="k", linestyle="--")
axes[0].axhline(2.0**0.5, color="k", linestyle="-")
axes[0].set_xlabel("Pixel")
axes[0].set_ylabel("Sensitivity")
axes[0].legend()
axes[0].set_title("Bi-O edge")

axes[1].set_xlabel("Pixel")
axes[1].set_ylabel("Sensitivity")
axes[1].legend()
