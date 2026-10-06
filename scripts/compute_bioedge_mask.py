# %%

import pathlib

import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits

root_dir = pathlib.Path(__file__).parent

# %% functions definitions


def compute_fast_axis_orientation_map(
    hwp_pixel_pitch, grey_width, mask_extent_x, mask_extent_y, separation
):
    """
    Compute the fast axis orientation map for a Bi-O edge mask.

    Parameters
    ----------
    hwp_pixel_pitch : float
        Pixel pitch of the HWP in meters.
    grey_width : float
        Width of the grey region in meters.
    mask_extent_x : float
        Extent of the mask in the x-direction in meters.
    mask_extent_y : float
        Extent of the mask in the y-direction in meters.
    separation : float
        Distance between the centers of the two grey regions in meters.

    Returns
    -------
    fast_axis_orientation_map : np.ndarray
        Computed fast axis orientation map in degrees.
    """

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
    ]  # adapt to Bi-O edge mask prototype 1 fast axis map
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
    )  # adapt to Bi-O edge mask prototype 1 fast axis map
    mask[center_horizontal_filter[1] + grey_width_pixels // 2 :, filters_frontier:] = 0

    # compute fast axis orientations map
    fast_axis_orientation_map = 0.5 * np.arccos(np.sqrt(mask))
    fast_axis_orientation_map = np.rad2deg(fast_axis_orientation_map)  # [deg]
    fast_axis_orientation_map[
        :,
        :filters_frontier,
    ] += 45  # adapt to Bi-O edge mask prototype 1 fast axis map

    return fast_axis_orientation_map


# %% parameters - Bi-O edge mask prototype 1

hwp_pixel_pitch = 10e-6  # [m]
grey_width = 240e-6  # [m]
mask_extent_x = 1e-2  # [m]
mask_extent_y = 0.5e-2  # [m]
separation = 3.6e-3  # [m] distance between the centers of the two grey regions

# %% compute Bi-O edge mask fast axis map

fast_axis_orientation_map_computed = compute_fast_axis_orientation_map(
    hwp_pixel_pitch, grey_width, mask_extent_x, mask_extent_y, separation
)

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
    root_dir / fast_axis_orientations_filename,
    fast_axis_orientation_map_computed,
    overwrite=True,
)

# %% import Bi-O edge mask first prototype fast axis orientations map

filename = "FAST_AXIS_EXTENDED_F62_SEP1.5arcsec_LAM770nm_GFW_5LAMoD_res10mic.fits"

fast_axis_orientation_map_prototype_1 = fits.getdata(root_dir / filename).astype(
    np.float64
)

# %% visualize the difference between the computed and the imported fast axis orientations maps

plt.figure()
plt.imshow(
    np.abs(fast_axis_orientation_map_computed - fast_axis_orientation_map_prototype_1),
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

# %% compute Bi-O edge mask prototype 2 (pitch = 5 µm)

# parameters
hwp_pixel_pitch_2 = 5e-6  # [m]

# %% compute Bi-O edge mask fast axis map

fast_axis_orientation_map_computed_2 = compute_fast_axis_orientation_map(
    hwp_pixel_pitch_2, grey_width, mask_extent_x, mask_extent_y, separation
)

print(
    f"Fast axis orientation map 2 shape: {fast_axis_orientation_map_computed_2.shape}"
)

x_extent_mm = mask_extent_x * 1e3
y_extent_mm = mask_extent_y * 1e3

# %% save computed fast axis orientations map

fast_axis_orientations_filename = (
    f"bioedge_fast_axis_map__"
    f"pitch_{hwp_pixel_pitch_2 * 1e6:.0f}um__"
    f"grey_width_{grey_width * 1e6:.0f}um__"
    f"separation_{separation * 1e3:.1f}mm__"
    f"extent_{mask_extent_x * 1e3:.0f}x{mask_extent_y * 1e3:.0f}_mm".replace(".", "p")
    + ".fits"
)

fits.writeto(
    root_dir / fast_axis_orientations_filename,
    fast_axis_orientation_map_computed_2,
    overwrite=True,
)

# %% plot grey width profiles and computed fast axis orientation map

filters_frontier = fast_axis_orientation_map_computed.shape[1] // 2
center_vertical_filter = (
    filters_frontier - round(separation / (2 * hwp_pixel_pitch)),
    fast_axis_orientation_map_computed.shape[0] // 2,
)  # [pixel] (x, y)
center_horizontal_filter = (
    filters_frontier + round(separation / (2 * hwp_pixel_pitch)),
    fast_axis_orientation_map_computed.shape[0] // 2,
)  # [pixel] (x, y)
grey_width_pixels = round(grey_width / hwp_pixel_pitch)

n_px_extra = 10

gw_vertical = fast_axis_orientation_map_computed[
    fast_axis_orientation_map_prototype_1.shape[0] // 2,
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
    filters_frontier + fast_axis_orientation_map_prototype_1.shape[0] // 4,
]

gw_vertical_prototype_1 = fast_axis_orientation_map_prototype_1[
    fast_axis_orientation_map_prototype_1.shape[0] // 2,
    center_vertical_filter[0]
    - grey_width_pixels // 2
    - n_px_extra : center_vertical_filter[0]
    + grey_width_pixels // 2
    + n_px_extra,
]

gw_horizontal_prototype_1 = fast_axis_orientation_map_prototype_1[
    center_horizontal_filter[1]
    - grey_width_pixels // 2
    - n_px_extra : center_horizontal_filter[1]
    + grey_width_pixels // 2
    + n_px_extra,
    filters_frontier + fast_axis_orientation_map_prototype_1.shape[0] // 4,
]

x_extent_mm = mask_extent_x * 1e3
y_extent_mm = mask_extent_y * 1e3

fig = plt.figure(figsize=(7, 1.2 * 7), constrained_layout=True)

gs = fig.add_gridspec(
    2,
    2,
    hspace=0.1,
    wspace=0.0,
)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, :])
ax1.plot(
    np.arange(-len(gw_vertical) / 2, len(gw_vertical) / 2) * hwp_pixel_pitch * 1e6,
    gw_vertical,
    label="computed",
)
ax1.plot(
    np.arange(-len(gw_vertical) / 2, len(gw_vertical) / 2) * hwp_pixel_pitch * 1e6,
    gw_vertical_prototype_1,
    label="prototype 1",
    linestyle="-.",
)
ax1.axhline(y=0, color="k", linestyle=":", label="0 deg")
ax1.axhline(y=45, color="k", linestyle="--", label="45 deg")
ax1.axhline(y=90, color="k", linestyle="-.", label="90 deg")
ax1.axvline(
    x=-120,
    color="gray",
    linestyle=":",
)
ax1.axvline(
    x=120,
    color="gray",
    linestyle=":",
)
ax1.set_ylim(-5, 95)
ax1.set_xlabel(r"[$\mu$m]")
ax1.set_ylabel("Fast axis orientation [deg]")
ax1.set_title("Vertical grey region")
ax1.legend(loc="lower right", bbox_to_anchor=(1.0, 0.05), fontsize=10)

ax2.plot(
    np.arange(-len(gw_horizontal) / 2, len(gw_horizontal) / 2) * hwp_pixel_pitch * 1e6,
    gw_horizontal,
)
ax2.plot(
    np.arange(-len(gw_horizontal) / 2, len(gw_horizontal) / 2) * hwp_pixel_pitch * 1e6,
    gw_horizontal_prototype_1,
    linestyle="-.",
)
ax2.axhline(y=0, color="k", linestyle=":")
ax2.axhline(y=45, color="k", linestyle="--")
ax2.axhline(y=90, color="k", linestyle="-.")
ax2.axvline(
    x=-120,
    color="gray",
    linestyle=":",
    label="240 µm grey width",
)
ax2.axvline(
    x=120,
    color="gray",
    linestyle=":",
)
ax2.set_ylim(-5, 95)
ax2.set_xlabel(r"[$\mu$m]")
ax2.set_ylabel("Fast axis orientation [deg]")
ax2.set_title("Horizontal grey region")
ax2.legend(loc="lower right", bbox_to_anchor=(1.0, 0.7), fontsize=10)

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
    f"Computed fast axis orientations map\n"
    f"pixel pitch = {hwp_pixel_pitch * 1e6:.1f} µm\n"
)

ax3.set_xlabel("x [mm]")
ax3.set_ylabel("y [mm]")

ny, nx = fast_axis_orientation_map_computed.shape

# Pixel-center <-> physical-coordinate conversions
dx = x_extent_mm / nx
dy = y_extent_mm / ny


def x_mm_to_px(x):
    return (x + x_extent_mm / 2) / dx - 0.5


def x_px_to_mm(px):
    return (px + 0.5) * dx - x_extent_mm / 2


def y_mm_to_px(y):
    return (y + y_extent_mm / 2) / dy - 0.5


def y_px_to_mm(py):
    return (py + 0.5) * dy - y_extent_mm / 2


# Pixel-index axes
ax_top = ax3.secondary_xaxis(
    "top",
    functions=(x_mm_to_px, x_px_to_mm),
)

ax_right = ax3.secondary_yaxis(
    "right",
    functions=(y_mm_to_px, y_px_to_mm),
)

ax_top.set_xlabel("x [pixel]")
ax_right.set_ylabel("y [pixel]")

# Make the pixel axes show integer pixel indices
ax_top.set_xticks(np.linspace(0, nx - 1, 5, dtype=int))
ax_right.set_yticks(np.linspace(0, ny - 1, 5, dtype=int))

fig.colorbar(
    im,
    ax=ax3,
    label="Fast axis orientation [deg]",
    ticks=[0, 45, 90],
    shrink=0.79,
    pad=0.04,
)

# %% plot grey width profiles and computed fast axis orientation map - prototype 2

filters_frontier_2 = fast_axis_orientation_map_computed_2.shape[1] // 2
center_vertical_filter_2 = (
    filters_frontier_2 - round(separation / (2 * hwp_pixel_pitch_2)),
    fast_axis_orientation_map_computed_2.shape[0] // 2,
)  # [pixel] (x, y)
center_horizontal_filter_2 = (
    filters_frontier_2 + round(separation / (2 * hwp_pixel_pitch_2)),
    fast_axis_orientation_map_computed_2.shape[0] // 2,
)  # [pixel] (x, y)
grey_width_pixels_2 = round(grey_width / hwp_pixel_pitch_2)

gw_vertical_2 = fast_axis_orientation_map_computed_2[
    fast_axis_orientation_map_prototype_1.shape[0] // 2,
    center_vertical_filter_2[0]
    - grey_width_pixels_2 // 2
    - n_px_extra : center_vertical_filter_2[0]
    + grey_width_pixels_2 // 2
    + n_px_extra,
]
gw_horizontal_2 = fast_axis_orientation_map_computed_2[
    center_horizontal_filter_2[1]
    - grey_width_pixels_2 // 2
    - n_px_extra : center_horizontal_filter_2[1]
    + grey_width_pixels_2 // 2
    + n_px_extra,
    filters_frontier_2 + fast_axis_orientation_map_prototype_1.shape[0] // 4,
]

fig_2 = plt.figure(figsize=(7, 1.2 * 7), constrained_layout=True)

gs = fig_2.add_gridspec(
    2,
    2,
    hspace=0.1,
    wspace=0.0,
)

ax1 = fig_2.add_subplot(gs[0, 0])
ax2 = fig_2.add_subplot(gs[0, 1])
ax3 = fig_2.add_subplot(gs[1, :])
ax1.plot(
    np.arange(-len(gw_vertical_2) / 2, len(gw_vertical_2) / 2)
    * hwp_pixel_pitch_2
    * 1e6,
    gw_vertical_2,
    label="computed",
)
ax1.axhline(y=0, color="k", linestyle=":", label="0 deg")
ax1.axhline(y=45, color="k", linestyle="--", label="45 deg")
ax1.axhline(y=90, color="k", linestyle="-.", label="90 deg")
ax1.axvline(
    x=-120,
    color="gray",
    linestyle=":",
)
ax1.axvline(
    x=120,
    color="gray",
    linestyle=":",
)
ax1.set_ylim(-5, 95)
ax1.set_xlabel(r"[$\mu$m]")
ax1.set_ylabel("Fast axis orientation [deg]")
ax1.set_title("Vertical grey region")
ax1.legend(loc="lower right", bbox_to_anchor=(1.0, 0.05), fontsize=10)

ax2.plot(
    np.arange(-len(gw_horizontal_2) / 2, len(gw_horizontal_2) / 2)
    * hwp_pixel_pitch_2
    * 1e6,
    gw_horizontal_2,
)

ax2.axhline(y=0, color="k", linestyle=":")
ax2.axhline(y=45, color="k", linestyle="--")
ax2.axhline(y=90, color="k", linestyle="-.")
ax2.axvline(
    x=-120,
    color="gray",
    linestyle=":",
    label="240 µm grey width",
)
ax2.axvline(
    x=120,
    color="gray",
    linestyle=":",
)
ax2.set_ylim(-5, 95)
ax2.set_xlabel(r"[$\mu$m]")
ax2.set_ylabel("Fast axis orientation [deg]")
ax2.set_title("Horizontal grey region")
ax2.legend(loc="lower right", bbox_to_anchor=(1.0, 0.7), fontsize=10)

im_2 = ax3.imshow(
    fast_axis_orientation_map_computed_2,
    cmap="gray",
    extent=[
        -x_extent_mm / 2,
        x_extent_mm / 2,
        -y_extent_mm / 2,
        y_extent_mm / 2,
    ],
)

ax3.set_title(
    f"Computed fast axis orientations map\n"
    f"pixel pitch = {hwp_pixel_pitch_2 * 1e6:.1f} µm\n"
)

ax3.set_xlabel("x [mm]")
ax3.set_ylabel("y [mm]")

ny_2, nx_2 = fast_axis_orientation_map_computed_2.shape

# Pixel-center <-> physical-coordinate conversions
dx_2 = x_extent_mm / nx_2
dy_2 = y_extent_mm / ny_2


def x_mm_to_px_2(x):
    return (x + x_extent_mm / 2) / dx_2 - 0.5


def x_px_to_mm_2(px):
    return (px + 0.5) * dx_2 - x_extent_mm / 2


def y_mm_to_px_2(y):
    return (y + y_extent_mm / 2) / dy_2 - 0.5


def y_px_to_mm_2(py):
    return (py + 0.5) * dy_2 - y_extent_mm / 2


# Pixel-index axes
ax_top = ax3.secondary_xaxis(
    "top",
    functions=(x_mm_to_px_2, x_px_to_mm_2),
)

ax_right = ax3.secondary_yaxis(
    "right",
    functions=(y_mm_to_px_2, y_px_to_mm_2),
)

ax_top.set_xlabel("x [pixel]")
ax_right.set_ylabel("y [pixel]")

# Make the pixel axes show integer pixel indices
ax_top.set_xticks(np.linspace(0, nx_2 - 1, 5, dtype=int))
ax_right.set_yticks(np.linspace(0, ny_2 - 1, 5, dtype=int))

fig_2.colorbar(
    im_2,
    ax=ax3,
    label="Fast axis orientation [deg]",
    ticks=[0, 45, 90],
    shrink=0.79,
    pad=0.04,
)


# %%

plt.show()

# %%
