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
grey_width = 1.0  # [m]
mask_extent_x = 1e-2  # [m]
mask_extent_y = 0.5e-2  # [m]

# %% compute fast axis orientatins map

mask = np.ones(
    (int(mask_extent_y / hwp_pixel_pitch), int(mask_extent_x / hwp_pixel_pitch))
)

# %% visualize the mask map

plt.figure(figsize=(3.45, 0.7 * 3.45))
plt.imshow(mask, cmap="gray")
plt.axis("equal")
