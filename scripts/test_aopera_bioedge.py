# %%

import matplotlib.pyplot as plt
import numpy as np

from aopera.ffwfs import (
    ffwfs_sensitivity,
    modulation,
    compute_bioedge_masks,
    pyramid_mask,
)
from aopera.otfpsf import otf_diffraction, otf2psf

# %% Functions definitions


def compute_amplitude_pyramid_mask(nx, n_faces, theta_ref=0):
    """
    Compute the complex amplitudes of the pyramidal bioedge_masks.

    Parameters
    ----------
    nx : number of pixels.
    n_faces : number of faces for the pyramidal mask.
    theta_ref : float, optional
        Reference angle for the first face of the pyramid. Default is 0.

    Returns
    -------
    bioedge_masks : list of complex arrays
        List of complex amplitudes for each face of the pyramidal mask.
        Each mask is a 2D array of shape (nx, nx) with binary values (0 or sqrt(1/n_faces)).
    """

    center = nx // 2 - 0.5
    xx, yy = np.mgrid[0:nx, 0:nx] - center
    theta = np.mod(np.arctan2(xx, yy) + np.pi - theta_ref, 2 * np.pi)
    theta_face = 2 * np.pi / n_faces
    bioedge_masks = []

    for k in range(0, n_faces):
        # Tesselation
        mask = ((k * theta_face) <= theta) * (theta < ((k + 1) * theta_face))
        mask = mask.astype(float)
        bioedge_masks.append(mask)

    return bioedge_masks


# %% Parameters

nx = 1024  # number of pixels for simulation
samp = 4  # sampling [pixels per lambda/D]
grey_width = 5.0  # width of the grey area in lambda/D
modulation_radius = grey_width / 2  # modulation amplitude in lambda/D
grey_length = None  # length of the grey area in pixels (None for full mask)

markersize = 8

# %% Compute the Bi-O edge bioedge_masks

bioedge_masks = compute_bioedge_masks(
    nx, samp * grey_width, samp * grey_length if grey_length is not None else None
)

try:
    import cupy as cp

    bioedge_masks = [mask.get() for mask in bioedge_masks]  # Convert to Numpy arrays
except:
    pass  # If CuPy is not available, continue with NumPy arrays

# %% Check

fig, axes = plt.subplots(2, 2, constrained_layout=True)
axes_flat = axes.flatten()
for index, ax in enumerate(axes_flat):
    mask = bioedge_masks[index]
    im = ax.imshow(mask**2, cmap="gray")
    ax.set_title(f"Mask {index + 1}")

plt.figure()
plt.plot(bioedge_masks[0][nx // 2, :], "o", label="top_left")
plt.plot(bioedge_masks[1][nx // 2, :], "o", label="left_0_right_1")
plt.plot(
    bioedge_masks[2][:, nx // 2], "+", markersize=markersize, label="top_0_bottom_1"
)
plt.plot(
    bioedge_masks[3][:, nx // 2], "+", markersize=markersize, label="top_1_bottom_0"
)
plt.xlabel("Pixel")
plt.ylabel("Field transmission")
plt.legend()

plt.figure()
plt.plot(bioedge_masks[0][nx // 2, :] ** 2, "o", label="top_left")
plt.plot(bioedge_masks[1][nx // 2, :] ** 2, "o", label="left_0_right_1")
plt.plot(
    bioedge_masks[2][:, nx // 2] ** 2,
    "+",
    markersize=markersize,
    label="top_0_bottom_1",
)
plt.plot(
    bioedge_masks[3][:, nx // 2] ** 2,
    "+",
    markersize=markersize,
    label="top_1_bottom_0",
)
plt.xlabel("Pixel")
plt.ylabel("Intensity transmission")
plt.legend()

transmissions = [np.sum(mask**2) / (nx**2) for mask in bioedge_masks]
print(f"Transmission top_left: {transmissions[0]:.2f}")
print(f"Transmission left_0_right_1: {transmissions[1]:.2f}")
print(f"Transmission top_0_bottom_1: {transmissions[2]:.2f}")
print(f"Transmission top_1_bottom_0: {transmissions[3]:.2f}\n")

total_transmission = np.sum([np.sum(mask**2) for mask in bioedge_masks]) / (nx**2)
print(f"Total transmission: {total_transmission:.2f}")

plt.show()

# %% Bi-O edge sensitivity

otf_diff = otf_diffraction(nx, occ=0, samp=samp)
psf_diff = otf2psf(otf_diff)

psf_calib = psf_diff
psf_current = psf_diff

modulation_path_bioedge = modulation(nx, 0.0, samp).get()

s_ron_top_left = ffwfs_sensitivity(
    bioedge_masks[0],
    modulation_path_bioedge,
    psf_calib,
    psf_current,
).get()
s_ron_left_0_right_1 = ffwfs_sensitivity(
    bioedge_masks[1],
    modulation_path_bioedge,
    psf_calib,
    psf_current,
).get()
s_ron_top_1_bottom_0 = ffwfs_sensitivity(
    bioedge_masks[2],
    modulation_path_bioedge,
    psf_calib,
    psf_current,
).get()
s_ron_top_0_bottom_1 = ffwfs_sensitivity(
    bioedge_masks[3],
    modulation_path_bioedge,
    psf_calib,
    psf_current,
).get()

s_ron_bioedge = (
    s_ron_top_left**2
    + s_ron_left_0_right_1**2
    + s_ron_top_1_bottom_0**2
    + s_ron_top_0_bottom_1**2
) ** 0.5

s_ph_bioedge = s_ron_bioedge * np.sqrt(4)  # Flux is splilled in 4 quadrants

# %% Modulated pyramid sensitivity

mask_pywfs = pyramid_mask(nx, 4)
modulation_path_pyramid = modulation(nx, modulation_radius, samp).get()

s_ron_pyramid = ffwfs_sensitivity(
    mask_pywfs, modulation_path_pyramid, psf_calib, psf_current
).get()

s_ph_pyramid = s_ron_pyramid * np.sqrt(4)  # Flux is splilled in 4 quadrants

# %% Amplitude pyramid

pyramid_masks = np.asarray(compute_amplitude_pyramid_mask(nx=nx, n_faces=4))

print(f"Transmission top left: {np.sum(pyramid_masks[0]**2)/nx**2:.2f}")
print(f"Transmission top right: {np.sum(pyramid_masks[1]**2)/nx**2:.2f}")
print(f"Transmission bottom right: {np.sum(pyramid_masks[2]**2)/nx**2:.2f}")
print(f"Transmission bottom left: {np.sum(pyramid_masks[3]**2)/nx**2:.2f}\n")
print(f"Total transmission: {np.sum(pyramid_masks**2)/nx**2:.2f}")

plt.figure()
plt.imshow(pyramid_masks.sum(axis=0), cmap="gray")

s_ron_top_left = ffwfs_sensitivity(
    pyramid_masks[0],
    modulation_path_pyramid,
    psf_calib,
    psf_current,
).get()
s_ron_top_right = ffwfs_sensitivity(
    pyramid_masks[1],
    modulation_path_pyramid,
    psf_calib,
    psf_current,
).get()
s_ron_bottom_right = ffwfs_sensitivity(
    pyramid_masks[2],
    modulation_path_pyramid,
    psf_calib,
    psf_current,
).get()
s_ron_bottom_left = ffwfs_sensitivity(
    pyramid_masks[3],
    modulation_path_pyramid,
    psf_calib,
    psf_current,
).get()

s_ron_amplitude_pyramid = (
    s_ron_top_left**2
    + s_ron_top_right**2
    + s_ron_bottom_right**2
    + s_ron_bottom_left**2
) ** 0.5

s_ph_amplitude_pyramid = s_ron_amplitude_pyramid * np.sqrt(
    4
)  # Flux is splilled in 4 quadrants

# %% Plot sensitivity maps

plt.figure()
plt.imshow(s_ron_bioedge, cmap="hot")
plt.title("Bi-O edge sensitivity")

plt.figure()
plt.imshow(s_ron_pyramid, cmap="hot")
plt.title("Phase pyramid sensitivity")

plt.figure()
plt.imshow(s_ron_pyramid, cmap="hot")
plt.title("Amplitude pyramid sensitivity")

# %% Plot sensitivity curves

freq_diagonal_cut = np.arange(s_ron_bioedge.shape[0] // 2) / samp * np.sqrt(2)
freq_cartesian_cut = np.arange(s_ron_bioedge.shape[0] // 2) / samp
x_lim = 0.25 * freq_diagonal_cut[-1]

# readout noise
fig, axes = plt.subplots(1, 2, figsize=(7, 0.5 * 7), constrained_layout=True)

# bioedge
axes[0].plot(freq_cartesian_cut, s_ron_bioedge[nx // 2, nx // 2 :], label="Horizontal")
axes[0].plot(
    freq_diagonal_cut,
    np.diag(s_ron_bioedge)[nx // 2 :],
    label="Diagonal",
)
axes[0].axvline(
    grey_width / 2,
    color="grey",
    linestyle=":",
    label=r"Grey area width",
)
axes[0].set_xlabel("Spatial frequency [D/lambda]")
axes[0].set_ylabel(r"$S_{ron}$")
axes[0].legend()
axes[0].set_title("Bi-O edge")
axes[0].set_xlim(-0.1, x_lim)

# pyramid
axes[1].plot(freq_cartesian_cut, s_ron_pyramid[nx // 2, nx // 2 :], label="Horizontal")
axes[1].plot(
    freq_diagonal_cut,
    np.diag(s_ron_pyramid)[nx // 2 :],
    label="Diagonal",
)
axes[1].plot(
    freq_cartesian_cut,
    s_ron_amplitude_pyramid[nx // 2, nx // 2 :],
    linestyle=":",
    color="red",
    label="Amplitude pyramid",
)
axes[1].plot(
    freq_diagonal_cut,
    np.diag(s_ron_amplitude_pyramid)[nx // 2 :],
    linestyle=":",
    color="red",
)
axes[1].axvline(
    grey_width / 2,
    color="grey",
    linestyle=":",
    label=r"Modulation radius",
)
axes[1].set_xlabel("Spatial frequency [D/lambda]")
axes[1].set_ylabel(r"$S_{ron}$")
axes[1].legend()
axes[1].set_title("Pyramid")
axes[1].set_xlim(-0.1, x_lim)

# photon noise
fig, axes = plt.subplots(1, 2, figsize=(10, 0.5 * 10), constrained_layout=True)

# bioedge
axes[0].plot(freq_cartesian_cut, s_ph_bioedge[nx // 2, nx // 2 :], label="Horizontal")
axes[0].plot(
    freq_diagonal_cut,
    np.diag(s_ph_bioedge)[nx // 2 :],
    label="Diagonal",
)
axes[0].axhline(
    1, color="k", linestyle="--", label="Theory: Bi-O edge - along edges (1)"
)
axes[0].axhline(
    2.0**0.5,
    color="k",
    linestyle="-",
    label=r"Theory: Bi-O edge - diagonal ($\sqrt{2}$)",
)
axes[0].axvline(
    grey_width / 2,
    color="grey",
    linestyle=":",
    label=r"Half grey width",
)
axes[0].set_xlabel("Spatial frequency [D/lambda]")
axes[0].set_ylabel(r"$S_{ph}$")
axes[0].legend()
axes[0].set_title("Bi-O edge")
axes[0].set_xlim(-0.1, x_lim)
axes[0].set_ylim(0, 1.47)

# pyramid
axes[1].plot(freq_cartesian_cut, s_ph_pyramid[nx // 2, nx // 2 :], label="Horizontal")
axes[1].plot(
    freq_diagonal_cut,
    np.diag(s_ph_pyramid)[nx // 2 :],
    label="Diagonal",
)
axes[1].plot(
    freq_cartesian_cut,
    s_ph_amplitude_pyramid[nx // 2, nx // 2 :],
    linestyle=":",
    color="red",
    label="Amplitude pyramid",
)
axes[1].plot(
    freq_diagonal_cut,
    np.diag(s_ph_amplitude_pyramid)[nx // 2 :],
    linestyle=":",
    color="red",
)
axes[1].axhline(
    1,
    color="k",
    linestyle="-",
    label="Theory: modulated pyramid - along edges (1)",
)
axes[1].axhline(
    2.0**0.5 / 2,
    color="k",
    linestyle="--",
    label=r"Theory: modulated pyramid - diagonal ($\sqrt{2}/2$)",
)
axes[1].axvline(
    modulation_radius, color="grey", linestyle=":", label=r"Modulation radius"
)
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
    label=r"Modulation / half grey area width",
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
    2.0**0.5,
    color="k",
    linestyle="-",
    label=r"Theory: Bi-O edge - diagonal ($\sqrt{2}$)",
)
ax.axhline(
    1,
    color="k",
    linestyle="-.",
    label="Theory: modulated pyramid and Bi-O edge - along edges (1)",
)
ax.axhline(
    2.0**0.5 / 2,
    color="k",
    linestyle="--",
    label=r"Theory: modulated pyramid - diagonal ($\sqrt{2}/2$)",
)

ax.axvline(
    grey_width / 2,
    color="grey",
    linestyle=":",
    label=r"Modulation / half grey area width",
)

ax.set_xlabel("Spatial frequency [D/lambda]")
ax.set_ylabel(r"$S_{ph}$")
ax.set_title("Sensitivity comparison: Bi-O edge vs Pyramid")
ax.set_xlim(-0.1, x_lim)
ax.legend(loc="lower right")

# %%
