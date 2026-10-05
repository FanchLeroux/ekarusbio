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
