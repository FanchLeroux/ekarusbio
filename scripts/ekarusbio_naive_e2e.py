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
from ekarusbio.analytical_budget import (
    compute_fitting,
    compute_temporal,
    compute_readout_noise,
    compute_photon_noise,
)
from ekarusbio.closed_loop import close_the_loop
from ekarusbio.miscellaneous import pad_array, crop_array

# %% outputs directory

config = Config()
fig_dir = config.root_dir / "outputs"

# %% simulation parameters

# ---------------------- NGS ---------------------- #

# phot.R4 = [0.670e-6, 0.300e-6, 7.66e12]
optical_band = "R4"  # optical band of the guide star
n_photons_per_measurment_point = 10

# ------------------ ATMOSPHERE ----------------- #

r0 = 0.1  # [m] value of r0 at 500 nm
external_scale = 30  # [m] value of L0 in the visibile
fractional_r0 = [0.45, 0.1, 0.1, 0.25, 0.1]  # Cn2 profile (percentage)
wind_speed = [5, 4, 8, 10, 2]  # [m.s-1] wind speed of layers
wind_direction = [0, 72, 144, 216, 288]  # [degrees] wind direction of layers
altitude = [0, 1000, 5000, 10000, 12000]  # [m] altitude of layers

# ------------------- TELESCOPE ------------------ #

diameter = 2  # [m] telescope diameter
n_subaperture = 41  # number of WFS subaperture along the telescope diameter
n_pixel_per_subaperture = (
    4  # [pixel] sampling of the WFS subapertures in telescope pupil space
)
resolution = (
    n_subaperture * n_pixel_per_subaperture
)  # resolution of the telescope driven by the WFS
central_obstruction_ratio = 0.3  # ratio of the central obscuration
# ------------------------ DM ---------------------- #

n_actuator = 24  # number of actuators

# ----------------------- WFS ---------------------- #

grey_width = 7.96 / 2  # [lambda/D] half grey width. Computed at 670 nm for F/# = 45
n_pix_separation = 10  # [pixel] separation ratio between the pupils
light_threshold = (
    0.3 if grey_width > 0.0 else 0
)  # light threshold to select the valid pixels
detector_photon_noise = False
detector_read_out_noise = 2 * n_photons_per_measurment_point / 4  # e- RMS

# -------------------- CALIBRATION - MODAL BASIS ---------------- #

modal_basis = "KL"
stroke_rad = 0.01  # [rad]
single_pass = False  # push-pull or push only for the calibration

# -------------------- LOOP ----------------------- #

loop_integrator_gain = 0.4
loop_frequency = 1000  # [Hz]
loop_delay = 2  # [frame]
n_iter = 200

# %% Build objects

# % -----------------------     NGS   ----------------------------------

# create the Natural Guide Star object
ngs = Source(
    optBand=optical_band,  # Source optical band
    # (see photometry.py)
    magnitude=0,  # arbitrary. magnitude will be updated later based on n_photons_per_measurment_point
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
    n_pix_separation=n_pix_separation,
    postProcessing="fullFrame",
)

# % --------------------- # photons ------------------------------ #

telescope_surface = tel.pupil.sum() * tel.pixelSize * tel.pixelSize
n_measurement_points = int(np.sum(bioedge.validSignal) / 4)

ngs.nPhoton = (
    n_photons_per_measurment_point
    * loop_frequency
    / (telescope_surface / n_measurement_points)
)  # nPhoton = # photons per s per m2

ngs**tel * dm * bioedge  # propagate the source through the system

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
calib = InteractionMatrix(
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

interaction_matrix = calib.D

print(
    f"Interaction matrix shape: {interaction_matrix.shape}\n"
    f"Interaction matrix rank: {np.linalg.matrix_rank(interaction_matrix)}"
)

# %% sensitivity analysis - sanity check and allows low order mode cutoff identification

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

n_controlled_modes = int(
    modal_dm.modes.shape[1] * (1 - central_obstruction_ratio**2)
)  # number of controlled modes. We assume more measurement points than dm actuators accross the pupil. The number of controlled modes is then equal to the number of actuators accross the pupil times the number of actuators accross the pupil times the ratio of the unobstructed area over the total area.

# %% compute classic lse reconstructor

reconstructor_lse = np.linalg.pinv(
    interaction_matrix[:, :n_controlled_modes]
)  # unweighted LSE reconstructor
reconstructor_lse = np.concatenate(
    (
        reconstructor_lse,
        np.zeros(
            (
                interaction_matrix.shape[1] - n_controlled_modes,
                reconstructor_lse.shape[1],
            )
        ),
    ),
    axis=0,
)  # pad the reconstructor with zeros to match the number of WFS signals

# %% compute reconstructor with truncated SVD weightened by the phase covariance matrix

# L = np.linalg.cholesky(c_phi)
L = np.diag(np.diag(c_phi) ** 0.5)


A = interaction_matrix @ L

U, s, Vh = np.linalg.svd(A, full_matrices=False)

U_k = U[:, :n_controlled_modes]
s_k = s[:n_controlled_modes]
Vh_k = Vh[:n_controlled_modes, :]

reconstructor_lse = L @ Vh_k.T / s_k @ U_k.T

# %% Analytical error budget

n_act = np.ceil(2 * (modal_dm.modes.shape[1] / np.pi) ** 0.5)
actuator_pitch = tel.D / n_act
r0_at_wavelength = r0 * (wavelength / 500e-9) ** (
    6 / 5
)  # [m] Fried parameter at the wavelength of the guide star
reconstructor_lse_rad_normalized = reconstructor_lse * (2 * np.pi) / wavelength
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

readout_noise_error = compute_readout_noise(
    n_photons_per_frame, reconstructor_lse_rad_normalized, detector_read_out_noise
)

photon_noise_error = (
    compute_photon_noise(
        n_photons_per_frame, reconstructor_lse_rad_normalized, reference_intensities
    )
    if detector_photon_noise
    else 0.0
)

strehl_analytical = np.exp(
    -(fitting_error + temporal_error + readout_noise_error + photon_noise_error)
)
residual_phase_std_analytical = (
    fitting_error + temporal_error + readout_noise_error + photon_noise_error
) ** 0.5  # [rad] residual phase std

print(
    f"Fitting error: {fitting_error:.3e} rad^2 RMS\n"
    f"Temporal error: {temporal_error:.3e} rad^2 RMS\n"
    f"Readout noise error: {readout_noise_error:.3e} rad^2 RMS\n"
    f"Photon noise error: {photon_noise_error:.3e} rad^2 RMS\n"
    f"Residual phase std: {residual_phase_std_analytical:.3e} rad RMS\n"
    f"Analytical Strehl ratio: {strehl_analytical:.3f}"
)

# %% SEED

seed = 1  # seed for atmosphere computation

# %% Close the loop - LSE

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
    modal_dm,
    bioedge,
    reconstructor_lse,
    loop_integrator_gain,
    n_iter,
    delay=loop_delay,
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
plt.plot(total_lse * 1e-9 * 2 * np.pi / wavelength, label="total_lse")
plt.plot(residual_lse * 1e-9 * 2 * np.pi / wavelength, label="residual_lse")
plt.axhline(
    y=residual_phase_std_analytical,
    color="k",
    linestyle="--",
    label=f"analytical residual phase std\n{residual_phase_std_analytical:.3e} rad RMS",
)
plt.xlabel("loop iteration")
plt.ylabel("residual phase RMS [rad]")
plt.title("Closed Loop residuals")
plt.legend()
plt.savefig(fig_dir / "residuals.png", bbox_inches="tight")

# strehls
plt.figure()
plt.plot(strehl_lse, label="strehl_lse")
plt.axhline(
    y=strehl_analytical,
    color="k",
    linestyle="--",
    label="analytical strehl ratio\n(fitting + temporal + readout noise + photon noise)",
)
plt.ylabel("Strehl ratio")
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
plt.title(f"long_exposure_psf_lse\nBi-O edge - {n_controlled_modes} controlled modes")

plt.show()

# %%
