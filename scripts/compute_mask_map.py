# %%

import pathlib

import numpy as np
import matplotlib.pyplot as plt

fig_dir = pathlib.Path(__file__).parent

from ekarusbio.config import Config

config = Config()
fig_dir = config.root_dir / "outputs"

# %% parameters

hwp_pixel_pitch = 5e-6  # [m]
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
] = 0
mask[
    :,
    center_vertical_filter[0]
    - grey_width_pixels // 2 : center_vertical_filter[0]
    + grey_width_pixels // 2,
] = np.linspace(0, 1, grey_width_pixels)
mask[
    :,
    center_vertical_filter[0] + grey_width_pixels // 2 : filters_frontier,
] = 1

# transmission horizontal filter
mask[: center_horizontal_filter[1] - grey_width_pixels // 2, filters_frontier:] = 0
mask[
    center_horizontal_filter[1]
    - grey_width_pixels // 2 : center_horizontal_filter[1]
    + grey_width_pixels // 2,
    filters_frontier:,
] = np.linspace(0, 1, grey_width_pixels).reshape(-1, 1)
mask[center_horizontal_filter[1] + grey_width_pixels // 2 :, filters_frontier:] = 1

# %% compute fast axis orientations map

fast_axis_orientations = 0.5 * np.arccos(np.sqrt(mask))
fast_axis_orientations = np.rad2deg(fast_axis_orientations)  # [deg]

# %% visualize the mask map

x_extent_mm = mask_extent_x * 1e3
y_extent_mm = mask_extent_y * 1e3

fig_transmission_map = plt.figure(figsize=(3.45, 0.7 * 3.45))
plt.imshow(
    mask,
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
plt.title("Bi-O edge transmission map")
# plt.xticks(
#     np.arange(
#         -mask_extent_x * 1e3 / 2,
#         mask_extent_x * 1e3 / 2 + 1,
#         1,
#     )
# )
# plt.yticks(
#     np.arange(
#         -mask_extent_y * 1e3 / 2,
#         mask_extent_y * 1e3 / 2 + 1,
#         1,
#     )
# )

fig_transmission_map.savefig(
    fig_dir / "transmission_map.pdf", dpi=300, bbox_inches="tight"
)

# %% visualize the fast axis orientations map

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
# plt.xticks(
#     np.arange(
#         -mask_extent_x * 1e3 / 2,
#         mask_extent_x * 1e3 / 2 + 1,
#         1,
#     )
# )
# plt.yticks(
#     np.arange(
#         -mask_extent_y * 1e3 / 2,
#         mask_extent_y * 1e3 / 2 + 1,
#         1,
#     )
# )

fig_fast_axis_orientations.savefig(
    fig_dir / "fast_axis_orientations.pdf", dpi=300, bbox_inches="tight"
)

plt.show()

# %%
