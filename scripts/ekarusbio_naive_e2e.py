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
from OOPAO.Pyramid import Pyramid
from OOPAO.BioEdge import BioEdge
from OOPAO.calibration.InteractionMatrix import InteractionMatrix

from ekarusbio.pattern import get_circular_pupil
from ekarusbio.modal_bases.KL_basis import compute_KL_basis
from ekarusbio.sensitivity import compute_photon_noise_sensitivity
from ekarusbio.analytical_budget import (
    compute_fitting,
    compute_temporal,
    compute_readout_noise,
    compute_photon_noise,
)
from ekarusbio.closed_loop import close_the_loop
from ekarusbio.miscellaneous import pad_array, crop_array

# %% functions definitions


def compute_weighted_svd_reconstructor(interaction_matrix, c_phi, n_controlled_modes):
    """
    Compute the weighted SVD reconstructor.

    Parameters
    ----------
    interaction_matrix : np.ndarray
        Interaction matrix.
    c_phi : np.ndarray
        Phase covariance matrix.
    n_controlled_modes : int
        Number of controlled modes.

    Returns
    -------
    np.ndarray
        Weighted SVD reconstructor.
    """
    L = np.diag(np.diag(c_phi) ** 0.5)
    A = interaction_matrix @ L

    U, s, Vh = np.linalg.svd(A, full_matrices=False)

    U_k = U[:, :n_controlled_modes]
    s_k = s[:n_controlled_modes]
    Vh_k = Vh[:n_controlled_modes, :]

    reconstructor = L @ Vh_k.T / s_k @ U_k.T

    return reconstructor


# %% outputs directory

config = Config()
fig_dir = config.root_dir / "outputs"

# %% simulation parameters

# ------------------ ATMOSPHERE ----------------- #

r0 = 0.1  # [m] value of r0 at 500 nm
external_scale = 30  # [m] value of L0 in the visibile
fractional_r0 = [0.45, 0.1, 0.1, 0.25, 0.1]  # Cn2 profile (percentage)
wind_speed = [5, 4, 8, 10, 2]  # [m.s-1] wind speed of layers
wind_direction = [0, 72, 144, 216, 288]  # [degrees] wind direction of layers
altitude = [0, 1000, 5000, 10000, 12000]  # [m] altitude of layers

# ------------------- TELESCOPE ------------------ #

diameter = 1.8  # [m] telescope diameter
n_subaperture = 41  # number of WFS subaperture along the telescope diameter
n_pixel_per_subaperture = (
    4  # [pixel] sampling of the WFS subapertures in telescope pupil space
)
resolution = (
    n_subaperture * n_pixel_per_subaperture
)  # resolution of the telescope driven by the WFS
central_obstruction_ratio = 0.318  # ratio of the central obscuration

# ------------------------ DM ---------------------- #

n_actuator = 24  # number of actuators

# ----------------------- WFS ---------------------- #

# Bi-O edge
grey_width = (
    7.96 / 2
)  # [lambda/D] Bi-O edge half grey width. Computed at 670 nm for F/# = 45
polarization_leakage_factor = 0.05  # polarization leakage factor for the Bi-O edge

# pyramid
modulation = grey_width  # [lambda/D] modulation radius

# both
n_pix_separation = 10  # [pixel] separation ratio between the pupils
light_threshold = (
    0.3 if grey_width > 0.0 else 0
)  # light threshold to select the valid pixels
detector_photon_noise = True
detector_read_out_noise = 0.0  # e- RMS


# -------------------- CALIBRATION - MODAL BASIS ---------------- #

modal_basis = "KL"
stroke_rad = 0.01  # [rad]
single_pass = False  # push-pull or push only for the calibration

# -------------------- LOOP ----------------------- #

loop_integrator_gain = 0.7
loop_frequency = 1000  # [Hz]
loop_delay = 1  # [frame]
n_controlled_modes = int(
    np.pi * (n_actuator / 2) ** 2 * (1 - central_obstruction_ratio**2)
)
n_iter = 200

# ---------------------- NGS ---------------------- #

# phot.R4 = [0.670e-6, 0.300e-6, 7.66e12]
optical_band = "R4"  # optical band of the guide star
n_photons_per_controlled_mode = 2


# %% Build objects

# % -----------------------     NGS   ----------------------------------

# create the Natural Guide Star object
ngs = Source(
    optBand=optical_band,  # Source optical band
    # (see photometry.py)
    magnitude=0,  # arbitrary. magnitude will be updated later based on n_photons_per_controlled_mode
)  # Source Magnitude
wavelength = ngs.wavelength  # [m] wavelength of the guide star

# % -----------------------    TELESCOPE   -----------------------------

# create the Telescope object
tel = Telescope(
    resolution=resolution,  # [pixel] resolution of the telescope
    diameter=diameter,  # [m] telescope diameter
    samplingTime=1 / loop_frequency,  # [s] sampling time of the telescope
    centralObstruction=central_obstruction_ratio,  # ratio of the central obscuration
)

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

dm = DeformableMirror(tel, nSubap=n_actuator - 1)

# % ----------------------- Bi-O edge ---------------------------- #

bioedge = BioEdge(
    nSubap=n_subaperture,
    telescope=tel,
    modulation=0.0,
    grey_width=grey_width,
    lightRatio=light_threshold,
    postProcessing="fullFrame",
    polarization_leakage_factor=polarization_leakage_factor,
)

pyramid = Pyramid(
    nSubap=n_subaperture,
    telescope=tel,
    modulation=modulation,
    lightRatio=light_threshold,
    postProcessing="fullFrame",
)

# % --------------------- # photons ------------------------------ #

telescope_surface = tel.pupil.sum() * tel.pixelSize * tel.pixelSize
n_measurement_points = int(np.sum(bioedge.validSignal) / 4)

ngs.nPhoton = (
    n_photons_per_controlled_mode
    * loop_frequency
    / (telescope_surface / n_measurement_points)
)  # nPhoton = # photons per s per m2

ngs**tel * dm * bioedge  # coupling telescope, DM and WFS is mandatory to compute the number of photons per measurement point

print(
    f"# photons per measurement points: {bioedge.cam.frame.sum() / n_measurement_points}"
)

# %% ------------------------- MODAL BASIS -------------------------------

m2c, c_phi = compute_KL_basis(tel, atm, dm, return_covariance=True)
ngs**tel  # reset

# %% extract calibration basis

influence_functions = dm.modes
calibration_basis = influence_functions @ m2c
calibration_basis = tel.pupil.reshape(-1, 1) * calibration_basis  # apply pupil mask

# %% -------------------------   Modal  DM   ----------------------------------

modal_dm = DeformableMirror(tel, nSubap=n_actuator - 1, modes=calibration_basis)

plt.figure()
plt.plot(modal_dm.modes[tel.pupil.reshape(-1)].std(axis=0))
plt.title("modal basis std")
plt.xlabel("# KL mode")
plt.ylabel("Standard deviation")
plt.ylim(0.999, 1.001)
plt.show()

# %% calibration

stroke_nm = stroke_rad * wavelength / (2 * np.pi)  # [nm]

calib_bioedge = InteractionMatrix(
    ngs,
    tel,
    modal_dm,
    bioedge,
    M2C=np.diag(np.ones(modal_dm.nValidAct)),
    stroke=stroke_nm,
    single_pass=single_pass,
    noise="off",
    display=True,
)
interaction_matrix_bioedge = calib_bioedge.D

calib_pyramid = InteractionMatrix(
    ngs,
    tel,
    modal_dm,
    pyramid,
    M2C=np.diag(np.ones(modal_dm.nValidAct)),
    stroke=stroke_nm,
    single_pass=single_pass,
    noise="off",
    display=True,
)
interaction_matrix_pyramid = calib_pyramid.D

print(
    f"Interaction matrix bioedge shape: {interaction_matrix_bioedge.shape}\n"
    f"Interaction matrix bioedge rank: {np.linalg.matrix_rank(interaction_matrix_bioedge)}\n"
    f"Interaction matrix pyramid shape: {interaction_matrix_pyramid.shape}\n"
    f"Interaction matrix pyramid rank: {np.linalg.matrix_rank(interaction_matrix_pyramid)}"
)

# %% sensitivity analysis - sanity check and allows low order mode cutoff identification

interaction_matrix_bioedge_rad_normalized_bioedge = (
    calib_bioedge.D * wavelength / (2 * np.pi)
)
reference_intensities_bioedge = bioedge.referenceSignal

photon_noise_sensitivity_bioedge = compute_photon_noise_sensitivity(
    interaction_matrix_bioedge_rad_normalized_bioedge, reference_intensities_bioedge
)
interaction_matrix_pyramid_rad_normalized_pyramid = (
    calib_pyramid.D * wavelength / (2 * np.pi)
)
reference_intensities_pyramid = pyramid.referenceSignal

photon_noise_sensitivity_pyramid = compute_photon_noise_sensitivity(
    interaction_matrix_pyramid_rad_normalized_pyramid, reference_intensities_pyramid
)

fig_sensitivity_bioedge, ax_sensitivity_bioedge = plt.subplots()
ax_sensitivity_bioedge.plot(photon_noise_sensitivity_bioedge, label="bioedge")
ax_sensitivity_bioedge.plot(photon_noise_sensitivity_pyramid, label="pyramid")
ax_sensitivity_bioedge.axhline(y=2**0.5, color="k", linestyle="--", label=r"$\sqrt{2}$")
ax_sensitivity_bioedge.set_xlabel("# KL mode")
ax_sensitivity_bioedge.set_ylabel(r"$S_{ph}$")

ax_sensitivity_bioedge.legend(loc="lower right")

# %% compute reconstructor with truncated SVD weightened by the phase covariance matrix

reconstructor_lse_bioedge = compute_weighted_svd_reconstructor(
    interaction_matrix_bioedge, c_phi, n_controlled_modes
)
reconstructor_lse_pyramid = compute_weighted_svd_reconstructor(
    interaction_matrix_pyramid, c_phi, n_controlled_modes
)

# %% compute classic LSE reconstructor

reconstructor_lse_bioedge = np.pad(
    np.linalg.pinv(interaction_matrix_bioedge[:, :n_controlled_modes]),
    ((0, interaction_matrix_bioedge.shape[1] - n_controlled_modes), (0, 0)),
)

reconstructor_lse_pyramid = np.pad(
    np.linalg.pinv(interaction_matrix_pyramid[:, :n_controlled_modes]),
    ((0, interaction_matrix_pyramid.shape[1] - n_controlled_modes), (0, 0)),
)

# %% Analytical error budget

n_act = np.ceil(2 * (modal_dm.modes.shape[1] / np.pi) ** 0.5)
actuator_pitch = tel.D / n_act
r0_at_wavelength = r0 * (wavelength / 500e-9) ** (
    6 / 5
)  # [m] Fried parameter at the wavelength of the guide star
reconstructor_lse_bioedge_rad_normalized = (
    reconstructor_lse_bioedge * (2 * np.pi) / wavelength
)
reconstructor_lse_pyramid_rad_normalized = (
    reconstructor_lse_pyramid * (2 * np.pi) / wavelength
)
n_photons_per_frame = (
    ngs.nPhoton * telescope_surface / loop_frequency
)  # [photon/frame] number of photons per frame

fitting_error = compute_fitting(r0_at_wavelength, actuator_pitch)

temporal_error = compute_temporal(
    loop_frequency,
    loop_delay / loop_frequency,
    loop_integrator_gain,
    n_controlled_modes,
    tel.D,
    np.mean(wind_speed),
    r0_at_wavelength,
)

readout_noise_error_bioedge = compute_readout_noise(
    n_photons_per_frame,
    reconstructor_lse_bioedge_rad_normalized,
    detector_read_out_noise,
)

photon_noise_error_bioedge = (
    compute_photon_noise(
        n_photons_per_frame,
        reconstructor_lse_bioedge_rad_normalized,
        reference_intensities_bioedge,
    )
    if detector_photon_noise
    else 0.0
)

readout_noise_error_pyramid = compute_readout_noise(
    n_photons_per_frame,
    reconstructor_lse_pyramid_rad_normalized,
    detector_read_out_noise,
)

photon_noise_error_pyramid = (
    compute_photon_noise(
        n_photons_per_frame,
        reconstructor_lse_pyramid_rad_normalized,
        reference_intensities_pyramid,
    )
    if detector_photon_noise
    else 0.0
)

strehl_analytical_bioedge = np.exp(
    -(
        fitting_error
        + temporal_error
        + readout_noise_error_bioedge
        + photon_noise_error_bioedge
    )
)
residual_phase_std_analytical_bioedge = (
    fitting_error
    + temporal_error
    + readout_noise_error_bioedge
    + photon_noise_error_bioedge
) ** 0.5  # [rad] residual phase std

strehl_analytical_pyramid = np.exp(
    -(
        fitting_error
        + temporal_error
        + readout_noise_error_pyramid
        + photon_noise_error_pyramid
    )
)

residual_phase_std_analytical_pyramid = (
    fitting_error
    + temporal_error
    + readout_noise_error_pyramid
    + photon_noise_error_pyramid
) ** 0.5  # [rad] residual phase std

print(
    f"Fitting error: {fitting_error:.3e} rad^2 RMS\n"
    f"Temporal error: {temporal_error:.3e} rad^2 RMS\n"
    f"Readout noise error bioedge: {readout_noise_error_bioedge:.3e} rad^2 RMS\n"
    f"Photon noise error bioedge: {photon_noise_error_bioedge:.3e} rad^2 RMS\n"
    f"Readout noise error pyramid: {readout_noise_error_pyramid:.3e} rad^2 RMS\n"
    f"Photon noise error pyramid: {photon_noise_error_pyramid:.3e} rad^2 RMS\n"
    f"Residual phase std bioedge: {residual_phase_std_analytical_bioedge:.3e} rad RMS\n"
    f"Residual phase std pyramid: {residual_phase_std_analytical_pyramid:.3e} rad RMS\n"
    f"Analytical Strehl ratio bioedge: {strehl_analytical_bioedge:.3f}\n"
    f"Analytical Strehl ratio pyramid: {strehl_analytical_pyramid:.3f}"
)

# %% Close the loop - LSE

seed = 1  # seed for atmosphere computation

(
    total_lse_bioedge,
    residual_lse_bioedge,
    strehl_lse_bioedge,
    dm_coefs_lse_bioedge,
    turbulence_phase_screens_lse_bioedge,
    residual_phase_screens_lse_bioedge,
    wfs_frames_lse_bioedge,
    wfs_signals_lse_bioedge,
    short_exposure_psf_lse_bioedge,
) = close_the_loop(
    tel,
    ngs,
    atm,
    modal_dm,
    bioedge,
    reconstructor_lse_bioedge,
    loop_integrator_gain,
    n_iter,
    delay=loop_delay,
    photon_noise=detector_photon_noise,
    read_out_noise=detector_read_out_noise,
    polc=False,
    seed=seed,
    save_telemetry=True,
    save_psf=True,
)

(
    total_lse_pyramid,
    residual_lse_pyramid,
    strehl_lse_pyramid,
    dm_coefs_lse_pyramid,
    turbulence_phase_screens_lse_pyramid,
    residual_phase_screens_lse_pyramid,
    wfs_frames_lse_pyramid,
    wfs_signals_lse_pyramid,
    short_exposure_psf_lse_pyramid,
) = close_the_loop(
    tel,
    ngs,
    atm,
    modal_dm,
    pyramid,
    reconstructor_lse_pyramid,
    loop_integrator_gain,
    n_iter,
    delay=loop_delay,
    photon_noise=detector_photon_noise,
    read_out_noise=detector_read_out_noise,
    polc=False,
    seed=seed,
    save_telemetry=True,
    save_psf=True,
)

# post processing
long_exposure_psf_lse_bioedge = np.sum(
    short_exposure_psf_lse_bioedge[:, :, 100:], axis=2
)
long_exposure_psf_lse_pyramid = np.sum(
    short_exposure_psf_lse_pyramid[:, :, 100:], axis=2
)

# %% plots

# noise propagation
plt.figure()
plt.plot(
    np.diag(reconstructor_lse_bioedge @ reconstructor_lse_bioedge.T),
    label="bioedge",
)
plt.plot(
    np.diag(reconstructor_lse_pyramid @ reconstructor_lse_pyramid.T),
    label="pyramid",
)
plt.yscale("log")
plt.title("modal uniform noise propagation")
plt.xlabel("# modes")
plt.legend()

# residuals
plt.figure()
plt.plot(
    residual_lse_bioedge * 1e-9 * 2 * np.pi / wavelength, label="residual_lse_bioedge"
)
plt.plot(
    residual_lse_pyramid * 1e-9 * 2 * np.pi / wavelength, label="residual_lse_pyramid"
)
plt.plot(total_lse_bioedge * 1e-9 * 2 * np.pi / wavelength, label="total_lse")
plt.axhline(
    y=residual_phase_std_analytical_bioedge,
    color="k",
    linestyle="--",
    label=f"analytical residual phase std bioedge\n{residual_phase_std_analytical_bioedge:.3e} rad RMS",
)
plt.xlabel("loop iteration")
plt.ylabel("residual phase RMS [rad]")
plt.title("Closed Loop residuals")
plt.legend()
plt.savefig(fig_dir / "residuals.png", bbox_inches="tight")

# strehls
plt.figure()
plt.plot(strehl_lse_bioedge, label="strehl_lse_bioedge")
plt.plot(strehl_lse_pyramid, label="strehl_lse_pyramid")
plt.axhline(
    y=strehl_analytical_bioedge,
    color="k",
    linestyle="--",
    label="analytical strehl ratio bioedge\n(fitting + temporal + readout noise + photon noise)",
)
plt.ylabel("Strehl ratio")
plt.title("Closed Loop strehls")
plt.legend()
plt.savefig(fig_dir / "strehls.png", bbox_inches="tight")

# long exposure PSF
plt.figure()
plt.imshow(
    crop_array(np.log(long_exposure_psf_lse_bioedge), 100),
    norm="linear",
    cmap="inferno",
)
plt.title(
    f"long_exposure_psf_lse_bioedge \nBi-O edge - {n_controlled_modes} controlled modes"
)

plt.figure()
plt.imshow(
    crop_array(np.log(long_exposure_psf_lse_pyramid), 100),
    norm="linear",
    cmap="inferno",
)
plt.title(
    f"long_exposure_psf_lse_pyramid \nPyramid - {n_controlled_modes} controlled modes"
)

plt.show()

# %%
