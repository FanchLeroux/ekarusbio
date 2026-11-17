# %% aopera backend

import aopera.backend as bk

backend_name = (
    "gpu"  # "cpu" or "gpu" or "auto" (auto will use GPU if available, otherwise CPU)
)
bk.set_backend(backend_name)

# %% imports

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from aopera.turbulence import Atmosphere
from aopera.control import RTC
from aopera.otfpsf import Pupil
from aopera.ffwfs import BioedgeWFS
from aopera.photometry import SourceScience, SourceWFS, Band
from aopera.utils import rad2arcsec, circavg, circsum, print_std
from aopera.simulation import simulation
from aopera.variance import var_temporal, var_fitting

from ekarusbio.config import Config

config = Config()

# temp

from aopera.otfpsf import otf_diffraction, otf2psf
from aopera.backend import to_cpu

# %% parameters

nx = 1024  # number of pixels for simulation
samp = 4  # sampling [pixels per lambda/D]

# Apply correction to convolutive model output for low frequencies: If True, sensitivity
# is modelled as a linear ramp inside the grey area.
apply_low_order_correction = True

# %%

otf_diff = otf_diffraction(nx, occ=0, samp=samp, return_cpu=False)
psf_diff = otf2psf(otf_diff, return_cpu=False)

psf_calib = psf_diff
psf_current = psf_diff

# %%

bioedge = BioedgeWFS.from_file(config.root_dir / "inputs" / "bioedge_ekarus.ini")
print(bioedge)

s_ron = bioedge.sensitivity(
    samp, psf_calib, psf_current, apply_low_order_correction=apply_low_order_correction
)

s_ph = s_ron * np.sqrt(bioedge.n_masks)

plt.figure()
plt.imshow(to_cpu(s_ph), cmap="inferno")
plt.title("Bi-O edge sensitivity")
plt.colorbar(label=r"$S_{ron}$")

# %% Plot sensitivity curves

freq_diagonal_cut = np.arange(s_ron.shape[0] // 2) / samp * np.sqrt(2)
freq_cartesian_cut = np.arange(s_ron.shape[0] // 2) / samp
x_lim = 0.25 * freq_diagonal_cut[-1]


fig, axes = plt.subplots(1, 2, figsize=(7, 0.5 * 7), constrained_layout=True)

# readout noise
axes[0].plot(
    to_cpu(freq_cartesian_cut),
    to_cpu(s_ron[nx // 2, nx // 2 :]),
    label="Horizontal",
)
axes[0].plot(
    to_cpu(freq_diagonal_cut),
    to_cpu(np.diag(s_ron)[nx // 2 :]),
    label="Diagonal",
)
axes[0].axvline(
    bioedge.grey_width / 2,
    color="grey",
    linestyle=":",
    label=r"Grey area width",
)
axes[0].set_xlabel("Spatial frequency [D/lambda]")
axes[0].set_ylabel(r"$S_{ron}$")
axes[0].legend()
axes[0].set_title("Bi-O edge\nSensitivity to readout noise")
axes[0].set_xlim(-0.1, x_lim)

# photon noise
axes[1].plot(
    to_cpu(freq_cartesian_cut),
    to_cpu(s_ph[nx // 2, nx // 2 :]),
    label="Horizontal",
)
axes[1].plot(
    to_cpu(freq_diagonal_cut),
    to_cpu(np.diag(s_ph)[nx // 2 :]),
    label="Diagonal",
)
axes[1].axhline(1, color="k", linestyle="--", label="Theory - along edges (1)")
axes[1].axhline(
    2.0**0.5,
    color="k",
    linestyle="-",
    label=r"Theory - diagonal ($\sqrt{2}$)",
)
axes[1].axvline(
    bioedge.grey_width / 2,
    color="grey",
    linestyle=":",
    label=r"Half grey width",
)
axes[1].set_xlabel("Spatial frequency [D/lambda]")
axes[1].set_ylabel(r"$S_{ph}$")
axes[1].legend()
axes[1].set_title("Bi-O edge\nSensitivity to photon noise")
axes[1].set_xlim(-0.1, x_lim)
axes[1].set_ylim(0, 1.47)
