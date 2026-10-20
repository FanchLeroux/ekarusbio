# %%

import matplotlib.pyplot as plt
import numpy as np

from aopera.backend import set_backend

set_backend(mode="cpu")

from aopera.ffwfs import (
    ffwfs_sensitivity,
    modulation,
    compute_bioedge_masks,
    pyramid_mask,
)
from aopera.otfpsf import otf_diffraction, otf2psf

# %% Parameters

nx = 2048  # number of pixels for simulation
samp = 8  # sampling [pixels per lambda/D]
grey_width = 8.0  # width of the grey area in lambda/D
grey_length = None  # length of the grey area in pixels (None for full mask)

markersize = 8

# %% Compute the Bi-O edge masks

masks = compute_bioedge_masks(
    nx, samp * grey_width, samp * grey_length if grey_length is not None else None
)

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

# %% Bi-O edge sensitivity

otf_diff = otf_diffraction(nx, occ=0, samp=samp)
psf_diff = otf2psf(otf_diff)

psf_calib = psf_diff
psf_current = psf_diff

modulation_path_bioedge = modulation(nx, 0.0, samp).get()

s_ron_left_1_right_0 = ffwfs_sensitivity(
    masks[0],
    modulation_path_bioedge,
    psf_calib,
    psf_current,
).get()
s_ron_left_0_right_1 = ffwfs_sensitivity(
    masks[1],
    modulation_path_bioedge,
    psf_calib,
    psf_current,
).get()
s_ron_top_1_bottom_0 = ffwfs_sensitivity(
    masks[2],
    modulation_path_bioedge,
    psf_calib,
    psf_current,
).get()
s_ron_top_0_bottom_1 = ffwfs_sensitivity(
    masks[3],
    modulation_path_bioedge,
    psf_calib,
    psf_current,
).get()

s_ron_bioedge = (
    s_ron_left_1_right_0**2
    + s_ron_left_0_right_1**2
    + s_ron_top_1_bottom_0**2
    + s_ron_top_0_bottom_1**2
) ** 0.5

s_ph_bioedge = s_ron_bioedge * np.sqrt(4)  # Flux is splilled in 4 quadrants

# %% Modulated pyramid sensitivity

mask_pywfs = pyramid_mask(nx, 4)
modulation_path_pyramid = modulation(nx, grey_width / 2, samp).get()

s_ron_pyramid = ffwfs_sensitivity(
    mask_pywfs, modulation_path_pyramid, psf_calib, psf_current
).get()

s_ph_pyramid = s_ron_pyramid * np.sqrt(4)  # Flux is splilled in 4 quadrants

# %% Plot sensitivity

freq_diagonal_cut = np.arange(s_ron_bioedge.shape[0] // 2) / samp * np.sqrt(2)
freq_cartesian_cut = np.arange(s_ron_bioedge.shape[0] // 2) / samp
x_lim = 0.25 * freq_diagonal_cut[-1]

plt.figure()
plt.imshow(s_ron_bioedge, cmap="hot")
plt.title("Bi-O edge sensitivity")

plt.figure()
plt.imshow(s_ron_pyramid, cmap="hot")
plt.title("Pyramid sensitivity")

fig, axes = plt.subplots(1, 2, figsize=(7, 0.5 * 7), constrained_layout=True)
axes[0].plot(freq_cartesian_cut, s_ron_bioedge[nx // 2, nx // 2 :], label="Horizontal")
axes[0].plot(
    freq_diagonal_cut,
    np.diag(s_ron_bioedge)[nx // 2 :],
    label="Diagonal",
)
axes[0].axhline(1, color="k", linestyle="--")
axes[0].axhline(2.0**0.5, color="k", linestyle="-")
axes[0].axvline(grey_width / 2, color="grey", linestyle=":")
axes[0].set_xlabel("Spatial frequency [D/lambda]")
axes[0].set_ylabel(r"$S_{ron}$")
axes[0].legend()
axes[0].set_title("Bi-O edge")
axes[0].set_xlim(-0.1, x_lim)

axes[1].plot(freq_cartesian_cut, s_ron_pyramid[nx // 2, nx // 2 :], label="Horizontal")
axes[1].plot(
    freq_diagonal_cut,
    np.diag(s_ron_pyramid)[nx // 2 :],
    label="Diagonal",
)
axes[1].axhline(1, color="k", linestyle="--")
axes[1].axhline(2.0**0.5 / 2, color="k", linestyle=":")
axes[1].axvline(grey_width / 2, color="grey", linestyle=":")
axes[1].set_xlabel("Spatial frequency [D/lambda]")
axes[1].set_ylabel(r"$S_{ron}$")
axes[1].legend()
axes[1].set_title("Pyramid")
axes[1].set_xlim(-0.1, x_lim)

fig, axes = plt.subplots(1, 2, figsize=(7, 0.5 * 7), constrained_layout=True)
axes[0].plot(freq_cartesian_cut, s_ph_bioedge[nx // 2, nx // 2 :], label="Horizontal")
axes[0].plot(
    freq_diagonal_cut,
    np.diag(s_ph_bioedge)[nx // 2 :],
    label="Diagonal",
)
axes[0].axhline(1, color="k", linestyle="--")
axes[0].axhline(2.0**0.5, color="k", linestyle="-")
axes[0].axvline(grey_width / 2, color="grey", linestyle=":")
axes[0].set_xlabel("Spatial frequency [D/lambda]")
axes[0].set_ylabel(r"$S_{ph}$")
axes[0].legend()
axes[0].set_title("Bi-O edge")
axes[0].set_xlim(-0.1, x_lim)

axes[1].plot(freq_cartesian_cut, s_ph_pyramid[nx // 2, nx // 2 :], label="Horizontal")
axes[1].plot(
    freq_diagonal_cut,
    np.diag(s_ph_pyramid)[nx // 2 :],
    label="Diagonal",
)
axes[1].axhline(1, color="k", linestyle="--")
axes[1].axhline(2.0**0.5 / 2, color="k", linestyle=":")
axes[1].axvline(grey_width / 2, color="grey", linestyle=":")
axes[1].set_xlabel("Spatial frequency [D/lambda]")
axes[1].set_ylabel(r"$S_{ph}$")
axes[1].legend()
axes[1].set_title("Pyramid")
axes[1].set_xlim(-0.1, x_lim)

# %% Pyramid and Bi-O edge sensitivity comparison

# readout noise
fig, ax = plt.subplots(figsize=(7, 0.75 * 7), constrained_layout=True)

# Bi-O edge
ax.plot(
    freq_cartesian_cut,
    s_ron_bioedge[nx // 2, nx // 2 :],
    label="Bi-O edge — Horizontal",
)
ax.plot(
    freq_diagonal_cut,
    np.diag(s_ron_bioedge)[nx // 2 :],
    label="Bi-O edge — Diagonal",
)

# Pyramid
ax.plot(
    freq_cartesian_cut,
    s_ron_pyramid[nx // 2, nx // 2 :],
    label="Pyramid — Horizontal",
)
ax.plot(
    freq_diagonal_cut,
    np.diag(s_ron_pyramid)[nx // 2 :],
    label="Pyramid — Diagonal",
)

ax.axvline(
    grey_width / 2,
    color="grey",
    linestyle=":",
    label=r"Modulation / grey area width",
)

ax.set_xlabel("Spatial frequency [D/lambda]")
ax.set_ylabel(r"$S_{ron}$")
ax.set_title("Sensitivity comparison: Bi-O edge vs Pyramid")
ax.set_xlim(-0.1, x_lim)
ax.legend(loc="lower right")

# photon noise
fig, ax = plt.subplots(figsize=(7, 0.75 * 7), constrained_layout=True)

# Bi-O edge
ax.plot(
    freq_cartesian_cut,
    s_ph_bioedge[nx // 2, nx // 2 :],
    label="Bi-O edge — Horizontal",
)
ax.plot(
    freq_diagonal_cut,
    np.diag(s_ph_bioedge)[nx // 2 :],
    label="Bi-O edge — Diagonal",
)

# Pyramid
ax.plot(
    freq_cartesian_cut,
    s_ph_pyramid[nx // 2, nx // 2 :],
    label="Pyramid — Horizontal",
)
ax.plot(
    freq_diagonal_cut,
    np.diag(s_ph_pyramid)[nx // 2 :],
    label="Pyramid — Diagonal",
)

# Reference levels
ax.axhline(
    1,
    color="k",
    linestyle="--",
    label="Theory: pyramid and Bi-O edge - along edges (1)",
)
ax.axhline(
    2.0**0.5,
    color="k",
    linestyle="-",
    label=r"Theory: Bi-O edge - diagonal ($\sqrt{2}$)",
)
ax.axhline(
    2.0**0.5 / 2,
    color="k",
    linestyle="-.",
    label=r"Theory: pyramid - diagonal ($\sqrt{2}/2$)",
)

ax.axvline(
    grey_width / 2,
    color="grey",
    linestyle=":",
    label=r"Modulation / grey area width",
)

ax.set_xlabel("Spatial frequency [D/lambda]")
ax.set_ylabel(r"$S_{ph}$")
ax.set_title("Sensitivity comparison: Bi-O edge vs Pyramid")
ax.set_xlim(-0.1, x_lim)
ax.legend(loc="lower right")

# %%
