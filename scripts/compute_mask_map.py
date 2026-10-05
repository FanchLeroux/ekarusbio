# %%

import pathlib

import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits

fig_dir = pathlib.Path(__file__).parent

from ekarusbio.config import Config

config = Config()
fig_dir = config.root_dir / "outputs"

# %% parameters

hwp_pixel_pitch = 10e-6  # [m]
grey_width = 240e-6  # [m]
mask_extent_x = 1e-2  # [m]
mask_extent_y = 0.5e-2  # [m]
separation = 3.5e-3  # [m] distance between the centers of the two grey regions

# %% compute transmission map for one output polarization direction (i.e. one pupil)

mask = np.ones(
    (round(mask_extent_y / hwp_pixel_pitch), round(mask_extent_x / hwp_pixel_pitch))
)
filters_frontier = mask.shape[1] // 2
center_vertical_filter = (
    filters_frontier - round(separation / (2 * hwp_pixel_pitch)),
    mask.shape[0] // 2,
)  # [pixel] (x, y)
center_horizontal_filter = (
    filters_frontier + round(separation / (2 * hwp_pixel_pitch)),
    mask.shape[0] // 2,
)  # [pixel] (x, y)
grey_width_pixels = round(grey_width / hwp_pixel_pitch)

# transmission vertical filter
mask[
    :,
    : center_vertical_filter[0] - grey_width_pixels // 2,
] = 1
mask[
    :,
    center_vertical_filter[0]
    - grey_width_pixels // 2 : center_vertical_filter[0]
    + grey_width_pixels // 2,
] = np.linspace(1, 0, grey_width_pixels)
mask[
    :,
    center_vertical_filter[0] + grey_width_pixels // 2 : filters_frontier,
] = 0

# transmission horizontal filter
mask[: center_horizontal_filter[1] - grey_width_pixels // 2, filters_frontier:] = 1
mask[
    center_horizontal_filter[1]
    - grey_width_pixels // 2 : center_horizontal_filter[1]
    + grey_width_pixels // 2,
    filters_frontier:,
] = np.linspace(1, 0, grey_width_pixels).reshape(-1, 1)
mask[center_horizontal_filter[1] + grey_width_pixels // 2 :, filters_frontier:] = 0

# %% compute fast axis orientations map

fast_axis_orientations = 0.5 * np.arccos(np.sqrt(mask))
fast_axis_orientations = np.rad2deg(fast_axis_orientations)  # [deg]
fast_axis_orientations[
    :,
    :filters_frontier,
] += 45  # adapt to first Bi-O edge mask prototype fast axis map

# %% visualize the fast axis orientations map

x_extent_mm = mask_extent_x * 1e3
y_extent_mm = mask_extent_y * 1e3

fig_fast_axis_orientations = plt.figure(figsize=(3.45, 0.7 * 3.45))
plt.imshow(
    fast_axis_orientations,
    cmap="gray",
    extent=[
        -x_extent_mm / 2,
        x_extent_mm / 2,
        -y_extent_mm / 2,
        y_extent_mm / 2,
    ],
)
plt.axis("equal")
plt.xlabel("x [mm]")
plt.ylabel("y [mm]")
plt.title("Bi-O edge fast axis orientations map [deg]")

fig_fast_axis_orientations.savefig(
    fig_dir / "fast_axis_orientations.pdf", dpi=300, bbox_inches="tight"
)

# %% import Bi-O edge mask first prototype fast axis orientations map

data_dir = config.root_dir / "data" / "bioedge_mask_prototype_1"
filename = "FAST_AXIS_EXTENDED_F62_SEP1.5arcsec_LAM770nm_GFW_5LAMoD_res10mic.fits"

fast_axis_orientations_map_prototype_1 = fits.getdata(data_dir / filename)

# %% visualize the difference between the computed and the imported fast axis orientations maps

plt.figure()
plt.imshow(
    fast_axis_orientations - fast_axis_orientations_map_prototype_1,
    cmap="RdBu",
    vmin=-5,
    vmax=5,
)

# %%

n_px_extra = 10

gw_vertical = fast_axis_orientations[
    fast_axis_orientations_map_prototype_1.shape[0] // 2,
    center_vertical_filter[0]
    - grey_width_pixels // 2
    - n_px_extra : center_vertical_filter[0]
    + grey_width_pixels // 2
    + n_px_extra,
]
gw_horizontal = fast_axis_orientations[
    center_horizontal_filter[1]
    - grey_width_pixels // 2
    - n_px_extra : center_horizontal_filter[1]
    + grey_width_pixels // 2
    + n_px_extra,
    filters_frontier + fast_axis_orientations_map_prototype_1.shape[0] // 4,
]

gw_vertical_prototype_1 = fast_axis_orientations_map_prototype_1[
    fast_axis_orientations_map_prototype_1.shape[0] // 2,
    center_vertical_filter[0]
    - grey_width_pixels // 2
    - n_px_extra : center_vertical_filter[0]
    + grey_width_pixels // 2
    + n_px_extra,
]

gw_horizontal_prototype_1 = fast_axis_orientations_map_prototype_1[
    center_horizontal_filter[1]
    - grey_width_pixels // 2
    - n_px_extra : center_horizontal_filter[1]
    + grey_width_pixels // 2
    + n_px_extra,
    filters_frontier + fast_axis_orientations_map_prototype_1.shape[0] // 4,
]

fig, axs = plt.subplots(1, 2, constrained_layout=True)
axs[0].plot(gw_vertical, label="Computed")
axs[0].plot(gw_vertical_prototype_1, label="Imported")
axs[0].axhline(y=0, color="k", linestyle=":", label="0 deg")
axs[0].axhline(y=45, color="k", linestyle="--", label="45 deg")
axs[0].axhline(y=90, color="k", linestyle="-.", label="90 deg")
axs[0].set_ylim(-5, 95)
axs[0].set_xlabel("Pixel index")
axs[0].set_ylabel("Fast axis orientation [deg]")
axs[0].set_title("Vertical grey region")
axs[0].legend(loc="lower right")
axs[1].plot(gw_horizontal, label="Computed")
axs[1].plot(gw_horizontal_prototype_1, label="Imported")
axs[1].axhline(y=0, color="k", linestyle=":", label="0 deg")
axs[1].axhline(y=45, color="k", linestyle="--", label="45 deg")
axs[1].axhline(y=90, color="k", linestyle="-.", label="90 deg")
axs[1].set_ylim(-5, 95)
axs[1].set_xlabel("Pixel index")
axs[1].set_ylabel("Fast axis orientation [deg]")
axs[1].set_title("Horizontal grey region")
axs[1].legend(loc="lower right")

# %%

plt.show()
