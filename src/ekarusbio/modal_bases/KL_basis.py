from functools import lru_cache

import numpy as np

from OOPAO.calibration.compute_KL_modal_basis import compute_M2C
from OOPAO.Telescope import Telescope
from OOPAO.Atmosphere import Atmosphere
from OOPAO.DeformableMirror import DeformableMirror


@lru_cache(maxsize=None)
def compute_KL_basis(
    tel: Telescope,
    atm: Atmosphere,
    dm: DeformableMirror,
    return_covariance: bool = False,
) -> np.ndarray:

    M2C_KL_full, HHt, PSD_atm, df = compute_M2C(
        telescope=tel,
        atmosphere=atm,
        deformableMirror=dm,
        param=None,
        nameFolder=None,
        remove_piston=False,
        HHtName="KL_covariance_matrix",
        baseName="KL_basis",
        mem_available=6.1e9,
        minimF=False,
        nmo=None,
        ortho_spm=True,
        SZ=np.int64(2 * tel.OPD.shape[0]),
        nZer=3,
        NDIVL=1,
        lim_inversion=1e-16,
        returnHHt_PSD_df=True,
        save_output=False,
    )

    m2c = M2C_KL_full[:, 1:]  # remove piston

    dm.coefs = np.zeros(dm.nValidAct)  # reset dm.OPD

    if return_covariance:

        c_phi = (
            (1.0 / tel.pupil.sum() ** 2.0)
            * m2c.T
            @ HHt
            @ m2c
            * (tel.src.wavelength / (2.0 * np.pi)) ** 2
        )

        return m2c, c_phi
    else:
        return m2c
