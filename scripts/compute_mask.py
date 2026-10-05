# %%

import pathlib

import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits

fig_dir = pathlib.Path(__file__).parent

from ekarusbio.config import Config

config = Config()
fig_dir = config.root_dir / "outputs"

# %% functions definitions


# %% parameters

hwp_pixel_pitch = 10e-6  # [m]
grey_width = 240e-6  # [m]
mask_extent_x = 1e-2  # [m]
mask_extent_y = 0.5e-2  # [m]
separation = 3.6e-3  # [m] distance between the centers of the two grey regions

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
] = np.linspace(1, 0, grey_width_pixels + 1)[
    :-1
]  # adapt to first Bi-O edge mask prototype 1 fast axis map
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
] = np.linspace(1, 0, grey_width_pixels + 1)[:-1].reshape(
    -1, 1
)  # adapt to first Bi-O edge mask prototype 1 fast axis map
mask[center_horizontal_filter[1] + grey_width_pixels // 2 :, filters_frontier:] = 0

# %% compute fast axis orientations map

fast_axis_orientation_map_computed = 0.5 * np.arccos(np.sqrt(mask))
fast_axis_orientation_map_computed = np.rad2deg(
    fast_axis_orientation_map_computed
)  # [deg]
fast_axis_orientation_map_computed[
    :,
    :filters_frontier,
] += 45  # adapt to first Bi-O edge mask prototype 1 fast axis map

# %% save computed fast axis orientations map

fast_axis_orientations_filename = (
    f"bioedge_fast_axis_map__"
    f"pitch_{hwp_pixel_pitch * 1e6:.0f}um__"
    f"grey_width_{grey_width * 1e6:.0f}um__"
    f"separation_{separation * 1e3:.1f}mm__"
    f"extent_{mask_extent_x * 1e3:.0f}x{mask_extent_y * 1e3:.0f}_mm".replace(".", "p")
    + ".fits"
)

fits.writeto(
    fig_dir / fast_axis_orientations_filename,
    fast_axis_orientation_map_computed,
    overwrite=True,
)

# %% visualize the fast axis orientations map

x_extent_mm = mask_extent_x * 1e3
y_extent_mm = mask_extent_y * 1e3

fig_fast_axis_orientations = plt.figure(figsize=(3.45, 0.7 * 3.45))
plt.imshow(
    fast_axis_orientation_map_computed,
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

# %% import Bi-O edge mask first prototype fast axis orientations map

data_dir = config.root_dir / "data" / "bioedge_mask_prototype_1"
filename = "FAST_AXIS_EXTENDED_F62_SEP1.5arcsec_LAM770nm_GFW_5LAMoD_res10mic.fits"

fast_axis_orientations_map_prototype_1 = fits.getdata(data_dir / filename)

# %% visualize the difference between the computed and the imported fast axis orientations maps

plt.figure()
plt.imshow(
    np.abs(fast_axis_orientation_map_computed - fast_axis_orientations_map_prototype_1),
    cmap="RdBu",
    vmin=-5,
    vmax=5,
)
plt.title(
    "Difference between computed and prototype 1\nfast axis orientations maps [deg]"
)
plt.xlabel("x [pixel]")
plt.ylabel("y [pixel]")
plt.colorbar(label="Difference [deg]", shrink=0.56, pad=0.04)

# %%

n_px_extra = 10

gw_vertical = fast_axis_orientation_map_computed[
    fast_axis_orientations_map_prototype_1.shape[0] // 2,
    center_vertical_filter[0]
    - grey_width_pixels // 2
    - n_px_extra : center_vertical_filter[0]
    + grey_width_pixels // 2
    + n_px_extra,
]
gw_horizontal = fast_axis_orientation_map_computed[
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

fig = plt.figure(figsize=(config.width_double_column, 1.2 * config.width_double_column))

gs = fig.add_gridspec(
    2,
    2,
    hspace=0.3,
    wspace=0.2,
)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, :])
ax1.plot(gw_vertical, label="computed")
ax1.plot(gw_vertical_prototype_1, label="prototype 1", linestyle="-.")
ax1.axhline(y=0, color="k", linestyle=":", label="0 deg")
ax1.axhline(y=45, color="k", linestyle="--", label="45 deg")
ax1.axhline(y=90, color="k", linestyle="-.", label="90 deg")
ax1.set_ylim(-5, 95)
ax1.set_xlabel("Pixel index")
ax1.set_ylabel("Fast axis orientation [deg]")
ax1.set_title("Vertical grey region")
ax1.legend(loc="lower right", bbox_to_anchor=(1.0, 0.05), fontsize=10)
ax2.plot(gw_horizontal, label="computed")
ax2.plot(gw_horizontal_prototype_1, label="prototype 1", linestyle="-.")
ax2.axhline(y=0, color="k", linestyle=":", label="0 deg")
ax2.axhline(y=45, color="k", linestyle="--", label="45 deg")
ax2.axhline(y=90, color="k", linestyle="-.", label="90 deg")
ax2.set_ylim(-5, 95)
ax2.set_xlabel("Pixel index")
ax2.set_ylabel("Fast axis orientation [deg]")
ax2.set_title("Horizontal grey region")
ax2.legend(loc="lower right", bbox_to_anchor=(1.0, 0.51), fontsize=10)
im = ax3.imshow(
    fast_axis_orientation_map_computed,
    cmap="gray",
    extent=[
        -x_extent_mm / 2,
        x_extent_mm / 2,
        -y_extent_mm / 2,
        y_extent_mm / 2,
    ],
)
ax3.set_title(
    f"Computed fast axis orientations map\npixel pitch = {hwp_pixel_pitch*1e6:.1f} µm"
)
ax3.set_xlabel("x [mm]")
ax3.set_ylabel("y [mm]")

fig.colorbar(
    im,
    ax=ax3,
    label="Fast axis orientation [deg]",
    ticks=[0, 45, 90],
    shrink=0.79,
    pad=0.04,
)


# %%

plt.show()
