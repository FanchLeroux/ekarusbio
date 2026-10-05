# %% imports

from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt

from ekarusbio.config import Config
from ekarusbio import (
    oopao_config,
)  # choose between GPU and CPU as well as float precision (float32/float64) for OOPAO

from OOPAO.Telescope import Telescope
from OOPAO.Atmosphere import Atmosphere
from OOPAO.Source import Source
from OOPAO.DeformableMirror import DeformableMirror
from OOPAO.BioEdge import BioEdge
from OOPAO.calibration.InteractionMatrix import InteractionMatrix

from ekarusbio.pattern import get_circular_pupil
from ekarusbio.modal_bases.KL_basis import compute_KL_basis
from ekarusbio.sensitivity import compute_photon_noise_sensitivity
from ekarusbio.closed_loop import close_the_loop
from ekarusbio.miscellaneous import pad_array, crop_array

# %% outputs directory

config = Config()
fig_dir = config.root_dir / "outputs"

# %% simulation parameters

# ---------------------- NGS ---------------------- #

# phot.R4 = [0.670e-6, 0.300e-6, 7.66e12]
wavelength = 670e-9  # [m] wavelength of the guide star
optical_band = "R4"  # optical band of the guide star
magnitude = 10  # magnitude of the guide star

# ------------------ ATMOSPHERE ----------------- #

r0 = 0.05  # [m] value of r0 at 500 nm
external_scale = 30  # [m] value of L0 in the visibile
fractional_r0 = [0.45, 0.1, 0.1, 0.25, 0.1]  # Cn2 profile (percentage)
wind_speed = [5, 4, 8, 10, 2]  # [m.s-1] wind speed of layers
wind_direction = [0, 72, 144, 216, 288]  # [degrees] wind direction of layers
altitude = [0, 1000, 5000, 10000, 12000]  # [m] altitude of layers

# ------------------- TELESCOPE ------------------ #

diameter = 2  # [m] telescope diameter
n_subaperture = 41  # number of WFS subaperture along the telescope diameter
n_pixel_per_subaperture = (
    8  # [pixel] sampling of the WFS subapertures in telescope pupil space
)
resolution = (
    n_subaperture * n_pixel_per_subaperture
)  # resolution of the telescope driven by the WFS
pupil_oversampling_factor: int = (
    10  # oversampling the pupil then bin it to avoid edge effects
)
# ------------------------ DM ---------------------- #

n_actuator = 24  # number of actuators

# ----------------------- WFS ---------------------- #

grey_width = 7.96 / 2  # [lambda/D] half grey width. Computed at 670 nm for F/# = 45
n_pix_separation = 10  # [pixel] separation ratio between the pupils
light_threshold = (
    0.3 if grey_width > 0.0 else 0
)  # light threshold to select the valid pixels
detector_photon_noise = True
detector_read_out_noise = 0.0  # e- RMS

# -------------------- CALIBRATION - MODAL BASIS ---------------- #

modal_basis = "KL"
stroke_rad = 0.01  # [rad]
stroke = stroke_rad * wavelength / (2 * np.pi)  # [nm]
single_pass = False  # push-pull or push only for the calibration

# -------------------- LOOP ----------------------- #

loop_gain = 0.4
n_iter = 200
delay = 2

# %% Build objects

# % -----------------------    TELESCOPE   -----------------------------

# create the Telescope object
tel = Telescope(
    resolution=resolution,  # [pixel] resolution of
    # the telescope
    diameter=diameter,
)  # [m] telescope diameter
pupil_oversampled = get_circular_pupil(tel.resolution * pupil_oversampling_factor)
pupil_binned = pupil_oversampled.reshape(
    tel.resolution,
    pupil_oversampling_factor,
    tel.resolution,
    pupil_oversampling_factor,
).mean(axis=(1, 3))

# %% -----------------------     NGS   ----------------------------------

# create the Natural Guide Star object
ngs = Source(
    optBand=optical_band,  # Source optical band
    # (see photometry.py)
    magnitude=magnitude,
)  # Source Magnitude

# % -----------------------    ATMOSPHERE   ----------------------------

# coupling telescope and source is mandatory to generate Atmosphere object
ngs * tel

# create the Atmosphere object
atm = Atmosphere(
    telescope=tel,  # Telescope
    r0=r0,  # Fried Parameter [m]
    L0=external_scale,  # Outer Scale [m]
    # Cn2 Profile (percentage)
    fractionalR0=fractional_r0,
    windSpeed=wind_speed,  # [m.s-1] wind speed of layers
    # [degrees] wind direction
    windDirection=wind_direction,
    # of layers
    altitude=altitude,
)  # [m] altitude of layers

# % -------------------------     DM   ----------------------------------

dm = DeformableMirror(tel, nSubap=n_actuator)

# %% ----------------------- Bi-O edge ---------------------------- #

bioedge = BioEdge(
    nSubap=n_subaperture,
    telescope=tel,
    modulation=0.0,
    grey_width=grey_width,
    lightRatio=light_threshold,
    n_pix_separation=n_pix_separation,
    postProcessing="fullFrame",
)

# %% ------------------------- MODAL BASIS -------------------------------

m2c, c_phi = compute_KL_basis(tel, atm, dm, return_covariance=True)
ngs**tel  # reset

# %% extract calibration basis

influence_functions = dm.modes
calibration_basis = influence_functions @ m2c
calibration_basis = tel.pupil.reshape(-1, 1) * calibration_basis  # apply pupil mask

# %% -------------------------   Modal  DM   ----------------------------------

calibration_modal_dm = DeformableMirror(tel, nSubap=n_actuator, modes=calibration_basis)

# %% calibration

calib = InteractionMatrix(
    ngs,
    tel,
    calibration_modal_dm,
    bioedge,
    M2C=np.diag(np.ones(calibration_modal_dm.nValidAct)),
    stroke=stroke,
    single_pass=single_pass,
    noise="off",
    display=True,
)

interaction_matrix = calib.D

print(
    f"Interaction matrix shape: {interaction_matrix.shape}\n"
    f"Interaction matrix rank: {np.linalg.matrix_rank(interaction_matrix)}"
)

# %% sensitivity analysis - allows low order mode cutoff identification

interaction_matrix_rad_normalized = calib.D * wavelength / (2 * np.pi)
reference_intensities = bioedge.referenceSignal

photon_noise_sensitivity = compute_photon_noise_sensitivity(
    interaction_matrix_rad_normalized, reference_intensities
)

fig_sensitivity, ax_sensitivity = plt.subplots()
ax_sensitivity.plot(photon_noise_sensitivity)
ax_sensitivity.axhline(y=2**0.5, color="k", linestyle="--", label=r"$\sqrt{2}$")
ax_sensitivity.set_xlabel("# KL mode")
ax_sensitivity.set_ylabel(r"$S_{ph}$")
ax_sensitivity.legend(loc="lower right")

# %% choose number of controlled modes

n_modes = dm.nValidAct - 100  # number of controlled modes

# %% compute classic lse reconstructor

reconstructor_lse = np.linalg.pinv(
    interaction_matrix[:, :n_modes]
)  # unweighted LSE reconstructor
reconstructor_lse = np.concatenate(
    (
        reconstructor_lse,
        np.zeros((interaction_matrix.shape[1] - n_modes, reconstructor_lse.shape[1])),
    ),
    axis=0,
)  # pad the reconstructor with zeros to match the number of WFS signals

# %% compute reconstructor with truncated SVD weightened by the phase covariance matrix

# L = np.linalg.cholesky(c_phi)
L = np.diag(np.diag(c_phi) ** 0.5)


A = interaction_matrix @ L

U, s, Vh = np.linalg.svd(A, full_matrices=False)

U_k = U[:, :n_modes]
s_k = s[:n_modes]
Vh_k = Vh[:n_modes, :]

reconstructor_lse = L @ Vh_k.T / s_k @ U_k.T

# %% SEED

seed = 12  # seed for atmosphere computation

# %% Close the loop - LSE - SR

(
    total_lse,
    residual_lse,
    strehl_lse,
    dm_coefs_lse,
    turbulence_phase_screens_lse,
    residual_phase_screens_lse,
    wfs_frames_lse,
    wfs_signals_lse,
    short_exposure_psf_lse,
) = close_the_loop(
    tel,
    ngs,
    atm,
    calibration_modal_dm,
    bioedge,
    reconstructor_lse,
    loop_gain,
    n_iter,
    delay=delay,
    photon_noise=detector_photon_noise,
    read_out_noise=detector_read_out_noise,
    polc=False,
    interaction_matrix=interaction_matrix,
    seed=seed,
    save_telemetry=True,
    save_psf=True,
)

# post processing
long_exposure_psf_lse = np.sum(short_exposure_psf_lse[:, :, 100:], axis=2)

# plots

# noise propagation
plt.figure()
plt.plot(np.diag(reconstructor_lse @ reconstructor_lse.T) / bioedge.nSignal)
plt.yscale("log")
plt.title("modal uniform noise propagation")
plt.xlabel("# modes")

# residuals
plt.figure()
plt.plot(total_lse, label="total_lse")
plt.plot(residual_lse, label="residual_lse")
plt.xlabel("loop iteration")
plt.ylabel("residual phase RMS [nm]")
plt.title("Closed Loop residuals")
plt.legend()
plt.savefig(fig_dir / "residuals.png", bbox_inches="tight")

# strehls
plt.figure()
plt.plot(strehl_lse, label="strehl_lse")
plt.ylabel("strehl ratio")
plt.title("Closed Loop strehls")
plt.legend()
plt.savefig(fig_dir / "strehls.png", bbox_inches="tight")

# long exposure PSF
plt.figure()
plt.imshow(
    crop_array(np.log(long_exposure_psf_lse), 100),
    norm="linear",
    cmap="inferno",
)
plt.title(f"long_exposure_psf_lse\nBi-O edge - {n_modes} controlled modes")

plt.show()

# %%
