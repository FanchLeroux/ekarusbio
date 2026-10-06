import numpy as np


def compute_fitting(r0, d):
    """
    Compute the fitting error term of the error budget.

    Parameters
    ----------
    r0 : float
        Fried parameter [m]
    d : float
        Deformable mirror actuator pitch [m]

    Returns
    -------
    fitting_error : float
        The fitting error of the DM.
    """

    var_fitting_error = 0.275 * (r0 / d) ** (-5 / 3)  # [rad^2 RMS]

    return var_fitting_error


def compute_temporal(
    loop_frequency,
    loop_delay,
    integrator_gain,
    n_controlled_modes,
    telescope_diameter,
    wind_speed,
    r0,
):
    """
    Compute the temporal error term of the error budget.

    Parameters
    ----------
    loop_frequency : float
        The frequency of the AO loop [Hz].
    loop_delay : float
        The delay of the AO loop [s].
    integrator_gain : float
        The gain of the integrator.
    n_controlled_modes : int
        The number of controlled modes.
    telescope_diameter : float
        The diameter of the telescope [m].
    wind_speed : float
        The speed of the wind [m/s].
    r0 : float
        The Fried parameter [m].

    Returns
    -------
    var_temporal_error : float
        The temporal error of the AO system.

    """

    # temporal error
    bandwidth = (
        loop_frequency
        / (2 * np.pi)
        * (integrator_gain / (1 + 2 * loop_delay * loop_frequency)) ** 0.5
    )  # [Hz]

    Nr = (-1 + (1 + 8 * n_controlled_modes) ** 0.5) / 2  # number of radial modes

    var_temporal_error = (
        0.135
        * (wind_speed / (bandwidth * telescope_diameter)) ** 2
        * (telescope_diameter / r0) ** (5 / 3)
        * ((Nr + 1) ** (1 / 3) - 1.15)
    )  # [rad^2 RMS]

    return var_temporal_error


def compute_readout_noise(n_photons, reconstructor, readout_noise):
    """
    Compute the read-out noise term of the error budget.

    Parameters
    ----------
    n_photons : float
        The number of photons per frame.
    readout_noise : float
        The read-out noise of the wavefront sensor (std) [e-/frame/pixel].
    reconstructor : np.ndarray
        The reconstructor marrix [rad].

    Returns
    -------
    var_readout_noise : float
        The read-out noise error term of the AO system. [rad^2].

    """

    var_readout_noise = (readout_noise / n_photons) ** 2 * np.trace(
        reconstructor @ reconstructor.T
    )  # [rad^2]
    return var_readout_noise


def compute_photon_noise(n_photons, reconstructor, reference_intensities):
    """
    Compute the photon noise term of the error budget.

    Parameters
    ----------
    n_photons : float
        The number of photons per frame.
    reconstructor : np.ndarray
        The reconstructor marrix [rad].
    reference_intensities : np.ndarray
        The reference intensities I(phi=0)/N_ph.

    Returns
    -------
    var_photon_noise : float
        The photon noise error term of the AO system. [rad^2].

    """

    var_photon_noise = (1 / n_photons) * np.trace(
        reconstructor @ np.diag(reference_intensities) @ reconstructor.T
    )  # [rad^2]
    return var_photon_noise
