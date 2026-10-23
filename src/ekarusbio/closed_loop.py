from dataclasses import dataclass

from tqdm import tqdm
import numpy as np

from OOPAO.Source import Source
from OOPAO.Atmosphere import Atmosphere
from OOPAO.Telescope import Telescope
from OOPAO.DeformableMirror import DeformableMirror
from OOPAO.Pyramid import Pyramid
from OOPAO.BioEdge import BioEdge
from OOPAO.GainSensingCamera import GainSensingCamera
from OOPAO.Detector import Detector


# tbd
@dataclass
class OpenLoopResult:
    short_exposure_psf: np.ndarray
    phase_screens: np.ndarray


@dataclass
class ClosedLoopResult:
    total: np.ndarray
    residual: np.ndarray
    strehl: np.ndarray = None
    optical_gains: np.ndarray = None
    dm_coefs: np.ndarray = None
    turbulence_phase_screens: np.ndarray = None
    residual_phase_screens: np.ndarray = None
    wfs_frames: np.ndarray = None
    wfs_signals: np.ndarray = None
    short_exposure_psf: np.ndarray = None
    gsc: GainSensingCamera = None


# tbd
def record_open_loop(ngs, atm, tel, projector=None):
    """
    Record the open loop phase screens for a given telescope, natural guide star, and atmosphere.

    Parameters
    ----------
    ngs : Source
        The natural guide star.
    atm : Atmosphere
        The atmosphere.
    tel : Telescope
        The telescope.

    Returns
    -------
    open_loop_phase_screens : np.ndarray
        The open loop phase screens.

    """

    n_iter = 100  # number of iterations

    open_loop_phase_screens = np.zeros([tel.OPD.shape[0], tel.OPD.shape[1]] + [n_iter])

    atm.initializeAtmosphere(tel)

    for k in tqdm(range(n_iter)):

        atm.update()

        ngs * tel

        open_loop_phase_screens[:, :, k] = tel.OPD

    return open_loop_phase_screens


def close_the_loop(
    ngs: Source,
    atm: Atmosphere,
    tel: Telescope,
    dm: DeformableMirror,
    wfs: Pyramid | BioEdge,
    reconstructor,
    loop_gain,
    n_iter=100,
    delay=1,
    photon_noise=False,
    read_out_noise=0.0,
    seed=0,
    polc=False,
    interaction_matrix=None,
    compute_strehl=True,
    compute_optical_gains=False,
    save_telemetry=False,
    save_psf=False,
    science_camera: Detector = None,
):
    """
    Close the loop for a given telescope, natural guide star, atmosphere, deformable mirror, and wavefront sensor.

    Parameters
    ----------
    tel : Telescope
        The telescope.
    ngs : Source
        The natural guide star.
    atm : Atmosphere
        The atmosphere.
    dm : DeformableMirror
        The deformable mirror. Should be a modal DM containing the calibration basis
    wfs : Pyramid | BioEdge
        The wavefront sensor.
    reconstructor : np.ndarray
        The reconstructor matrix.
    loop_gain : float
        The loop gain.
    n_iter : int, optional
        The number of iterations. Default is 100.
    delay : int, optional
        The delay in the loop. Default is 1.
    photon_noise : bool, optional
        Whether to include photon noise. Default is False.
    read_out_noise : float, optional
        The readout noise. Default is 0.0.
    seed : int, optional
        The random seed. Default is 0.
    polc : bool, optional
        Whether to use POLC. Default is False.
    interaction_matrix : np.ndarray, optional
        The interaction matrix. Default is None.
    compute_strehl : bool, optional
        Whether to compute the Strehl ratio. Default is True.
    compute_optical_gains : bool, optional
        Whether to compute the optical gains. Default is False.
    save_telemetry : bool, optional
        Whether to save the telemetry data. Default is False.
    save_psf : bool, optional
        Whether to save the PSF data. Default is False.

    Returns
    -------
    closed_loop_result : ClosedLoopResult
        The result of the closed-loop simulation.

    """

    wfs.cam.photonNoise = photon_noise
    wfs.cam.readoutNoise = read_out_noise

    ngs * tel

    # Initialization

    total = np.zeros(n_iter)  # turbulence phase std [nm]
    residual = np.zeros(n_iter)  # residual phase std [nm]
    buffer_wfs_measure = np.zeros([wfs.signal.shape[0]] + [delay])

    if compute_strehl:
        strehl = np.zeros(n_iter)  # Strehl Ratio

    if compute_optical_gains:
        wfs.focal_plane_camera.resolution = wfs.resolution
        gsc = GainSensingCamera(
            mask=wfs.mask,
            basis=dm.modes.reshape(
                (tel.pupil.shape[0], tel.pupil.shape[1], dm.modes.shape[-1])
            ),
        )
        ngs**tel * wfs  # reset OPD and propagate a flat wavefront towards the wavefront sensor
        wfs * wfs.focal_plane_camera  # propagate to the focal plane camera
        wfs.focal_plane_camera * gsc  # calibrate the gsc (first mul of wavefront sensor focal plane camera and GSC is for calibration)
        optical_gains = np.zeros([dm.modes.shape[-1], n_iter])

    if save_telemetry:
        dm_coefs = np.zeros([dm.nValidAct, n_iter])
        turbulence_phase_screens = np.zeros(
            [tel.OPD.shape[0], tel.OPD.shape[1]] + [n_iter]
        )
        residual_phase_screens = np.zeros(
            [tel.OPD.shape[0], tel.OPD.shape[1]] + [n_iter]
        )
        wfs_frames = np.zeros(
            [wfs.cam.frame.shape[0], wfs.cam.frame.shape[1]] + [n_iter]
        )
        wfs_signals = np.zeros([wfs.signal.shape[0]] + [n_iter])

    if save_psf:
        if science_camera is None:
            raise ValueError(
                "Science camera must be provided for PSF saving. Please provide it as an argument."
            )
        ngs**tel * science_camera  # reset OPD and propagate a flat wavefront towards the science camera
        short_exposure_psf = np.zeros(
            [science_camera.frame.shape[0], science_camera.frame.shape[1]] + [n_iter]
        )

    atm.initializeAtmosphere(tel)
    atm.generateNewPhaseScreen(seed=seed)
    tel + atm

    dm.coefs = 0

    ngs**atm * tel * dm * wfs

    # close the loop

    for k in tqdm(range(n_iter)):

        atm.update()  # update the atmosphere

        total[k] = np.std(tel.OPD[np.where(tel.pupil > 0)]) * 1e9  # [nm]

        if save_telemetry:
            turbulence_phase_screens[:, :, k] = tel.OPD

        ngs**atm * tel * dm * wfs  # propagation through the wfs branch

        residual[k] = np.std(tel.OPD[np.where(tel.pupil > 0)]) * 1e9  # [nm]

        buffer_wfs_measure = np.roll(buffer_wfs_measure, -1, axis=1)
        buffer_wfs_measure[:, -1] = wfs.signal

        if compute_optical_gains:
            wfs * wfs.focal_plane_camera
            wfs.focal_plane_camera * gsc
            optical_gains[:, k] = gsc.og

        if save_telemetry:

            residual_phase_screens[:, :, k] = tel.OPD
            dm_coefs[:, k] = dm.coefs
            wfs_frames[:, :, k] = wfs.cam.frame
            wfs_signals[:, k] = buffer_wfs_measure[:, 0]

        if compute_strehl:
            strehl[k] = (
                np.abs(
                    np.sum(
                        np.exp(
                            1j
                            * tel.OPD[np.where(tel.pupil > 0)]
                            * 2
                            * np.pi
                            / ngs.wavelength
                        )
                    )
                )
                ** 2
            ) / tel.pupil.sum() ** 2  # Ref. V. N. Mahajan (1982)

        if polc:
            if interaction_matrix is None:
                raise ValueError(
                    "Interaction matrix must be provided for POLC. Please provide it as an argument."
                )
            else:
                pseudo_open_loop_measures = (
                    buffer_wfs_measure[:, 0] - interaction_matrix @ dm.coefs
                )
                dm.coefs = (
                    1 - loop_gain
                ) * dm.coefs - loop_gain * reconstructor @ pseudo_open_loop_measures
        else:
            dm.coefs = dm.coefs - loop_gain * reconstructor @ buffer_wfs_measure[:, 0]

        if save_psf:

            ngs**atm * tel * dm * science_camera  # propagation through the science branch
            short_exposure_psf[:, :, k] = science_camera.frame

    # wrap results in a dataclass
    closed_loop_result = ClosedLoopResult(
        total=total,
        residual=residual,
    )

    if compute_strehl:
        closed_loop_result.strehl = strehl

    if compute_optical_gains:
        closed_loop_result.optical_gains = optical_gains
        closed_loop_result.gsc = gsc

    if save_telemetry:
        closed_loop_result.dm_coefs = dm_coefs
        closed_loop_result.turbulence_phase_screens = turbulence_phase_screens
        closed_loop_result.residual_phase_screens = residual_phase_screens
        closed_loop_result.wfs_frames = wfs_frames
        closed_loop_result.wfs_signals = wfs_signals

    if save_psf:
        closed_loop_result.short_exposure_psf = short_exposure_psf

    # return
    return closed_loop_result
