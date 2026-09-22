# %%

import pathlib

import numpy as np
import matplotlib.pyplot as plt

fig_dir = pathlib.Path(__file__).parent

from ekarusbio.config import Config

config = Config()
fig_dir = config.root_dir / "outputs"

# %% parameters

hwp_pixel_pitch = 0.5e-6  # [m]
grey_width = 1.0  # [m]
