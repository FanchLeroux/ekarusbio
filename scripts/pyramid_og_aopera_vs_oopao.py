# %%

"""
Run a full SH-WFS or P-WFS simulation, both on OOAPO and aopera.
"""

import os
import builtins

# In some SLURM environments, CuPy import fails or wastes time if no GPU is allocated.
# If CUDA_VISIBLE_DEVICES is not set, we block CuPy import at runtime.
if not os.environ.get("CUDA_VISIBLE_DEVICES"):  # None or ""
    _real_import = builtins.__import__

    def _blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "cupy" or name.startswith("cupy."):
            raise ImportError("CuPy disabled (no GPU allocated)")
        return _real_import(name, globals, locals, fromlist, level)

    builtins.__import__ = _blocked_import

os.environ["AOPERA_DISABLE_GPU"] = "TRUE"

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from numpy.fft import fft2, fftshift

from OOPAO.Telescope import Telescope
from OOPAO.Source import Source
from OOPAO.Atmosphere import Atmosphere
from OOPAO.Detector import Detector
from OOPAO.DeformableMirror import DeformableMirror
from OOPAO.ShackHartmann import ShackHartmann
from OOPAO.Pyramid import Pyramid
from OOPAO.calibration.compute_KL_modal_basis import compute_KL_basis
from OOPAO.calibration.InteractionMatrix import InteractionMatrix
from OOPAO.tools.displayTools import cl_plot
from OOPAO.GainSensingCamera import GainSensingCamera

from aopera.turbulence import Atmosphere as Atmosphere_aopera
from aopera.turbulence import equivalent_wind_speed
from aopera.control import RTC
from aopera.otfpsf import Pupil
from aopera.shwfs import SHWFS
from aopera.ffwfs import PWFS
from aopera.photometry import SourceScience, SourceWFS
from aopera.utils import circavg, circsum, rad2arcsec, rfftconvolve
from aopera.simulation import simulation

# %% PARAMETERS

### SIMULATION
nLoop = 300  # OOPAO number of iterations
use_sh = False  # choose between SH and Pyramid WFS

### ATMOSPHERE
r0 = 0.07  # [m]
wind_equiv = 8  # equivalent wind speed [m/s]

### ADAPTIVE OPTICS
diameter = 1.5  # [m]
nAct = 20  # number of DM actuators
ao_freq = 1000  # [Hz]
ngs_mag = 9
obj_diameter_arcsec = 0  # object diameter [arcsec]. Zero for point source.
gainCL = 0.4
frame_delay = 1  # choose 1 or 2 total frame delay
is_photon_noise = True  # activate WFS photon noise
ron = 0.2
leaky = 0.999
mode_ratio = 0.99

### PYRAMID
modulation_radius = 3

### SHACK-HARTMANN SPOT ALGORITHM
# FIXME: difference between OOPAO and aopera !
npix_cog = 10
threshold_cog = 0.01

# %% TELESCOPE and NGS and SRC and ATMOSPHERE
n_subaperture = nAct

tel = Telescope(
    resolution=8 * n_subaperture,  # resolution of the telescope in [pix]
    diameter=diameter,  # diameter in [m]
    samplingTime=1 / ao_freq,  # Sampling time in [s] of the AO loop
    centralObstruction=0,  # force to zero to compute PSD within array
    display_optical_path=False,  # Flag to display optical path
    fov=10,
)  # field of view in [arcsec]. If set to 0 (default) this speeds up the computation of the phase screens but is uncompatible with off-axis targets

ngs = Source(optBand="I", magnitude=ngs_mag, coordinates=[0, 0])

ngs * tel  # combine the NGS to the telescope

src = Source(optBand="J", magnitude=0, coordinates=[0, 0])

fractional_cn2dh = np.array([0.6, 0.2, 0.2])
wind_ratio = np.array([10, 12, 15])
wind_speed = (
    wind_ratio * wind_equiv / equivalent_wind_speed(fractional_cn2dh, wind_ratio)
)

atm = Atmosphere(
    telescope=tel,
    r0=r0,
    L0=25,
    fractionalR0=fractional_cn2dh,  # Cn2 Profile
    windSpeed=wind_speed,  # Wind Speed in [m]
    windDirection=[0, 30, 60],  # Wind Direction in [degrees]
    altitude=[0, 500, 10000],
)  # Altitude Layers in [m]

atm.initializeAtmosphere(tel)  # initialize atmosphere with current Telescope
atm.update()  # The phase screen can be updated using atm.update method (Temporal sampling given by tel.samplingTime)

# %% DEFORMABLE MIRROR
dm = DeformableMirror(
    telescope=tel,
    nSubap=nAct
    - 1,  # number of subaperture of the system considered (by default the DM has n_subaperture + 1 actuators to be in a Fried Geometry)
    mechCoupling=0.35,
    coordinates=None,  # coordinates in [m]. Should be input as an array of size [n_actuators, 2]
    pitch=tel.D / nAct,
    floating_precision=32,
)  # inter actuator distance. Only used to compute the influence function coupling. The default is based on the n_subaperture value.

tel.isPaired = False  # make sure tel and atm are separated to initialize the PWFS
ngs**tel  # tel.resetOPD()

# %% WAVEFRONT SENSOR
if use_sh:
    wfs = ShackHartmann(
        nSubap=n_subaperture,
        telescope=tel,
        threshold_cog=threshold_cog,
        lightRatio=0.5,
        shannon_sampling=True,
    )
else:
    wfs = Pyramid(
        nSubap=n_subaperture,
        telescope=tel,
        lightRatio=0.5,
        modulation=modulation_radius,
        n_pix_separation=4,
        n_pix_edge=2,
        postProcessing="slopesMaps_incidence_flux",
    )

    xx_mod_oopao_calib = [
        wfs.modulation_path[i][0] for i in range(len(wfs.modulation_path))
    ]
    yy_mod_oopao_calib = [
        wfs.modulation_path[i][1] for i in range(len(wfs.modulation_path))
    ]

# %% CALIBRATION
M2C_KL = compute_KL_basis(tel, atm, dm, lim=1e-3)  # matrix to apply modes on the DM
nmode_correct = round(mode_ratio * M2C_KL.shape[0])
M2C_modal = M2C_KL[:, : nmode_correct + 1]  # modal Interaction Matrix
stroke = 1e-9  # amplitude of the modes in m

calib_modal = InteractionMatrix(
    ngs=ngs,
    atm=atm,
    tel=tel,
    dm=dm,
    wfs=wfs,
    M2C=M2C_modal,  # M2C matrix used
    stroke=stroke,  # stroke for the push/pull in M2C units
    nMeasurements=12,  # number of simultaneous measurements
    noise="off",  # disable wfs.cam noise
    display=True,  # display the time using tqdm
    single_pass=True,
)  # only push to compute the interaction matrix instead of push-pull

# %% GSC
if not use_sh:
    ngs**tel  # tel.resetOPD()
    dm.coefs = M2C_modal
    tel * dm

    basis_3D = tel.OPD.copy()
    # flatten the npix*npix dimension to compute pseudo inverse
    basis = basis_3D.reshape(basis_3D.shape[0] * basis_3D.shape[1], basis_3D.shape[2])

    gsc = GainSensingCamera(mask=wfs.mask, basis=basis_3D)

    # make sure that the GSC has the right resolution (cropped by default)
    wfs.focal_plane_camera.resolution = wfs.nRes

    # GSC calibration using flat OPD
    ngs**tel  # tel.resetOPD()
    tel * wfs
    wfs * wfs.focal_plane_camera
    wfs.focal_plane_camera * gsc

# %% OOPAO LOOP

# EXTENDED OBJECT
obj_diameter_wvl_D = obj_diameter_arcsec / rad2arcsec(ngs.wavelength / diameter)

if not use_sh:
    if obj_diameter_wvl_D > 0:
        print("INFO : unactivate on-sky modulation for extended objects")
        if (2 * modulation_radius) > obj_diameter_wvl_D:
            print(
                "WARNING : extended object is smaller than calibration modulation. This increases OG."
            )
        radius_obj = obj_diameter_wvl_D / 2
        xx, yy = np.mgrid[
            -radius_obj : radius_obj : 1 / 2, -radius_obj : radius_obj : 1 / 2
        ]
        rr = np.sqrt(xx**2 + yy**2)
        xx_mod_oopao = xx[rr <= radius_obj]
        yy_mod_oopao = yy[rr <= radius_obj]
        user_modulation_path = [
            [xx_mod_oopao[i], yy_mod_oopao[i]] for i in range(len(xx_mod_oopao))
        ]
        wfs.user_modulation_path = user_modulation_path
        wfs.modulation = 1
    else:
        xx_mod_oopao = xx_mod_oopao_calib
        yy_mod_oopao = yy_mod_oopao_calib
else:
    if obj_diameter_wvl_D > 0:
        raise ValueError("Cannot use SH on extended source")

# instrument path
src_cam = Detector(tel.resolution * 4)
src_cam.psf_sampling = 4
src_cam.integrationTime = tel.samplingTime

# WFS path
ngs_cam = Detector(tel.resolution * 2)
ngs_cam.psf_sampling = 4
ngs_cam.integrationTime = tel.samplingTime

# get diffraction limit PSF
tel - atm
src * tel * src_cam
psf_diff_src_oopao = src_cam.frame
ngs * tel * ngs_cam
psf_diff_ngs_oopao = ngs_cam.frame

# initialize Telescope DM commands
src**tel  # tel.resetOPD()
dm.coefs = 0
ngs * tel * dm * wfs
atm.generateNewPhaseScreen(seed=10)
tel + atm  # combine telescope with atmosphere
tel.computePSF(4)

# initialize DM commands
atm * ngs * tel * ngs_cam
atm * src * tel * src_cam

opd_atmo_std = np.zeros(nLoop)
opd_src_std = np.zeros(nLoop)
opd_ngs_std = np.zeros(nLoop)
opd_cube = np.zeros(tel.OPD.shape + (nLoop,))
if not use_sh:
    og_oopao = np.ones((basis_3D.shape[2], nLoop))

psf_src_le = np.zeros(src_cam.frame.shape)
psf_ngs_le = np.zeros(ngs_cam.frame.shape)

wfsSignal = np.arange(0, wfs.nSignal) * 0

close_loop_iter = 30  # nb iter before being sure that loop closed completely
wfs.cam.photonNoise = is_photon_noise
wfs.cam.readoutNoise = ron
display = False
reconstructor = M2C_modal @ calib_modal.M

plot_obj = cl_plot(
    list_fig=[
        atm.OPD,
        tel.mean_removed_OPD,
        wfs.cam.frame,
        [dm.coordinates[:, 0], np.flip(dm.coordinates[:, 1]), dm.coefs],
        [[0, 0], [0, 0], [0, 0]],
        np.log10(tel.PSF),
    ],
    type_fig=["imshow", "imshow", "imshow", "scatter", "plot", "imshow"],
    list_title=[
        "Turbulence [nm]",
        "SRC residual [m]",
        "WFS Detector",
        "DM Commands",
        None,
        None,
    ],
    list_legend=[
        None,
        None,
        None,
        None,
        [
            "SRC@" + str(src.coordinates[0]) + '"',
            "NGS@" + str(ngs.coordinates[0]) + '"',
        ],
        None,
    ],
    list_label=[
        None,
        None,
        None,
        None,
        ["Time", "WFE [nm]"],
        [
            "SRC PSF@"
            + str(src.coordinates[0])
            + '" -- FOV: '
            + str(np.round(src_cam.fov_arcsec, 2))
            + '"',
            "",
        ],
    ],
    n_subplot=[3, 2],
    list_display_axis=[None, None, None, None, True, None],
    list_ratio=[[0.95, 0.95, 0.1], [1, 1, 1]],
    s=20,
)

for i_lp in range(nLoop):

    atm.update()  # update phase screens => overwrite tel.OPD and consequently tel.src.phase
    opd_atmo_std[i_lp] = np.std(tel.OPD[np.where(tel.pupil > 0)]) * 1e9

    atm * ngs * tel * dm * wfs * ngs_cam  # propagate light from the NGS through the atmosphere, telescope, DM to the WFS and NGS camera with the CL commands applied

    ## GSC
    if not use_sh:
        wfs * wfs.focal_plane_camera
        wfs.focal_plane_camera * gsc
        og_oopao[:, i_lp] = gsc.og

    opd_ngs_std[i_lp] = (
        np.std(tel.OPD[np.where(tel.pupil > 0)]) * 1e9
    )  # save residuals corresponding to the NGS
    OPD_NGS = tel.mean_removed_OPD.copy()

    ngs_psf = ngs_cam.frame
    if i_lp > close_loop_iter:
        psf_ngs_le += ngs_psf
    NGS_PSF = np.log10(np.abs(ngs_psf))

    atm * src * tel * dm * src_cam  # propagate light from the SRC through the atmosphere, telescope, DM to the Instrument camera

    src_psf = src_cam.frame
    if i_lp > close_loop_iter:
        psf_src_le += src_psf
    SRC_PSF = np.log10(np.abs(src_psf))

    # save residuals corresponding to the NGS
    opd_src_std[i_lp] = np.std(tel.OPD[np.where(tel.pupil > 0)]) * 1e9
    OPD_SRC = tel.mean_removed_OPD.copy()

    opd_cube[:, :, i_lp] = tel.mean_removed_OPD * 1e9

    if frame_delay == 1:
        wfsSignal = wfs.signal

    # apply the commands on the DM
    dm.coefs = leaky * dm.coefs - gainCL * np.matmul(reconstructor, wfsSignal)

    # store the slopes after computing the commands => 2 frames delay
    if frame_delay == 2:
        wfsSignal = wfs.signal

    # update displays if required
    if display == True and i_lp > 1 and (i_lp % 10) == 0:
        plot_obj.list_lim = [
            None,
            None,
            None,
            None,
            [0, 300],
            [SRC_PSF.max() - 6, SRC_PSF.max()],
        ]
        plot_obj.list_title = [
            "Turbulence " + str(np.round(opd_atmo_std[i_lp])) + "[nm]",
            "SRC residual " + str(np.round(opd_src_std[i_lp])) + "[nm]",
            "WFS Detector",
            "DM Commands",
            None,
            None,
        ]

        cl_plot(
            list_fig=[
                1e9 * atm.OPD,
                1e9 * OPD_SRC,
                wfs.cam.frame,
                dm.coefs,
                [np.arange(i_lp + 1), opd_src_std[: i_lp + 1], opd_ngs_std[: i_lp + 1]],
                SRC_PSF,
            ],
            plt_obj=plot_obj,
        )
        plt.pause(0.01)
        if plot_obj.keep_going is False:
            break

    if (i_lp % 10) == 0:
        print("%4u/%4u     OPD=%3u nm" % (i_lp, nLoop, round(opd_ngs_std[i_lp])))

# %% OOPAO METRICS
print()
print("OOPAO NGS OPD : %u nm" % np.mean(opd_ngs_std[close_loop_iter:i_lp]))
print("OOPAO SRC OPD : %u nm" % np.mean(opd_src_std[close_loop_iter:i_lp]))

sr_oopao_ngs = (
    100
    * np.max(psf_ngs_le / np.sum(psf_ngs_le))
    / np.max(psf_diff_ngs_oopao / np.sum(psf_diff_ngs_oopao))
)
sr_oopao_src = (
    100
    * np.max(psf_src_le / np.sum(psf_src_le))
    / np.max(psf_diff_src_oopao / np.sum(psf_diff_src_oopao))
)

# %% COMPUTE OOPAO PSD
nx = tel.OPD.shape[0]
dx = tel.D / nx
nx_in = 2 * (int(np.floor(nx / np.sqrt(2))) // 2)
L_in = tel.D * nx_in / nx
oopao_df = 1 / (L_in)

oopao_opd_psd = np.zeros((nx_in, nx_in))
nbiter = 0
x0 = nx // 2 - nx_in // 2
x1 = x0 + nx_in

for i in range(close_loop_iter, i_lp):
    nbiter += 1
    frame = opd_cube[x0:x1, x0:x1, i]
    frame = frame - np.mean(frame)
    frame_ft = fftshift(fft2(fftshift(frame))) / nx_in**2
    psd_instant = np.abs(frame_ft) ** 2 / oopao_df**2
    oopao_opd_psd += psd_instant

oopao_opd_psd /= nbiter
oopao_opd_psd[nx_in // 2, nx_in // 2] = 0  # filter null frequency


# %% AOPERA SIMULATION
src_sci = SourceScience.from_oopao(src)
src_wfs = SourceWFS.from_oopao(ngs, src)
atmo = Atmosphere_aopera.from_oopao(atm)
pupil = Pupil.from_oopao(tel, dm, reconstructor)
rtc = RTC.from_oopao(ao_freq, frame_delay, gainCL, leak=leaky)

if use_sh:
    wfs_aopera = SHWFS.from_oopao(wfs, npix_cog)
else:
    wfs_aopera = PWFS.from_oopao(wfs)

src_wfs.size = obj_diameter_arcsec
if obj_diameter_arcsec > 0:
    wfs_aopera.modulation = modulation_radius  # has been updated to 1 in OOPAO WFS
    wfs_aopera.modulation_sky = False

nx_aopera = 700
samp = ngs_cam.psf_sampling

ignore = ["chromatic", "jitter"]
if not is_photon_noise:
    ignore += ["photon"]

# pupil.nact = pupil.nact + 1
# wfs_aopera.lenslet = wfs_aopera.lenslet + 1

result = simulation(
    nx_aopera,
    samp,
    pupil,
    atmo,
    rtc,
    wfs_aopera,
    src_sci,
    src_wfs,
    verbose=True,
    split_fit_temp=True,
    ignore=ignore,
)
psd_total = sum([result.psd[k] for k in result.psd.keys()])

opd_num = {}
rad_to_nm = ngs.wavelength * 1e9 / (2 * np.pi)
for k in result.var.keys():
    opd_num[k] = np.sqrt(result.var[k]) * rad_to_nm

# %% AOPERA PSF at NGS
psf_ao_ngs = pupil.psf_ao(psd_total, samp)
psf_diff_ngs = pupil.psf_diffraction(nx_aopera, samp)
strehl_wfs = 100 * np.max(psf_ao_ngs) / np.max(psf_diff_ngs)

wvl_ratio = src_sci.wvl / src_wfs.wvl
psf_ao_sci = pupil.psf_ao(psd_total, samp, wvl_scale=wvl_ratio)
psf_diff_sci = pupil.psf_diffraction(nx_aopera, samp * wvl_ratio)
strehl_sci = 100 * np.max(psf_ao_sci) / np.max(psf_diff_sci)

# %% AOPERA LINEAR
result_lin = simulation(
    nx_aopera,
    samp,
    pupil,
    atmo,
    rtc,
    wfs_aopera,
    src_sci,
    src_wfs,
    verbose=False,
    nb_iter_og=1,
    ignore=ignore,
)
psd_total_lin = sum([result_lin.psd[k] for k in result_lin.psd.keys()])
psf_ao_ngs_lin = pupil.psf_ao(psd_total_lin, samp)
psf_ao_sci_lin = pupil.psf_ao(psd_total_lin, samp, wvl_scale=wvl_ratio)

opd_num_lin = {}
rad_to_nm = ngs.wavelength * 1e9 / (2 * np.pi)
for k in result_lin.var.keys():
    opd_num_lin[k] = np.sqrt(result_lin.var[k]) * rad_to_nm

# %% PLOT
df = pupil.frequency_step(samp)
samp_sci = samp * wvl_ratio


def setplt():
    plt.legend(loc="lower left")
    plt.xlabel("Position [$\\lambda/D$]")


def setimg(psf, samp_ratio):
    nc = int(round(70 * samp_ratio))
    nx = psf.shape[0] // 2
    psf_norm = psf / np.sum(psf)
    plt.imshow(
        psf_norm,
        cmap="Spectral_r",
        norm=LogNorm(vmin=psf_norm[nx - nc, nx - nc], vmax=psf_norm[nx, nx]),
    )
    plt.xlim(nx - nc, nx + nc)
    plt.ylim(nx - nc, nx + nc)
    plt.axis("off")
    plt.colorbar()


plt.figure(1, figsize=(14, 9))
plt.clf()

plt.subplot(231)
plt.title("OPD")

y_oopao = circavg(oopao_opd_psd)
x_oopao = np.arange(len(y_oopao)) * oopao_df
std_oopao = np.sqrt(np.sum(oopao_opd_psd) * oopao_df**2)

y_aopera = circavg(psd_total) * rad_to_nm**2
x_aopera = np.arange(len(y_aopera)) * df
std_aopera = np.sqrt(np.sum(psd_total) * (rad_to_nm**2) * (df**2))
std_aopera_no_piston = np.sqrt(np.sum([opd_num[k] ** 2 for k in opd_num.keys()]))

y_aopera_lin = circavg(psd_total_lin) * rad_to_nm**2
std_aopera_lin = np.sqrt(np.sum(psd_total_lin) * (rad_to_nm**2) * (df**2))
std_aopera_lin_no_piston = np.sqrt(
    np.sum([opd_num_lin[k] ** 2 for k in opd_num_lin.keys()])
)

plt.loglog(x_oopao[1:], y_oopao[1:], label="OOPAO : %u nm" % std_oopao)
plt.loglog(x_aopera[1:], y_aopera[1:], label="AOPERA : %u nm" % std_aopera_no_piston)
for k in ["fitting", "aliasing", "temporal", "photon", "ron"]:
    plt.loglog(
        x_aopera[1:], circavg(result.psd[k])[1:] * rad_to_nm**2, ls="--", label=k
    )

plt.xlabel("Frequency [1/m]")
plt.ylabel("PSD [nm²m²]")
plt.xlim(right=oopao_df * nx_in / 2)
plt.ylim(bottom=0.1)
plt.legend()


plt.subplot(232)
plt.title("cumulative OPD from halo")

y_oopao = np.sqrt(circsum(oopao_opd_psd, backward=True) * oopao_df**2)
x_oopao = np.arange(len(y_oopao)) * oopao_df

y_aopera = np.sqrt(circsum(psd_total, backward=True) * df**2 * rad_to_nm**2)
x_aopera = np.arange(len(y_aopera)) * df

y_aopera_lin = np.sqrt(circsum(psd_total_lin, backward=True) * df**2 * rad_to_nm**2)

plt.semilogx(x_oopao[1:], y_oopao[1:], label="OOPAO : %u nm" % std_oopao)
plt.semilogx(x_aopera[1:], y_aopera[1:], label="AOPERA : %u nm" % std_aopera_no_piston)
# plt.semilogx(x_aopera[1:], y_aopera_lin[1:], label='AOPERA linear : %u nm'%std_aopera_lin_no_piston, zorder=0)

plt.xlabel("Frequency [1/m]")
plt.ylabel("std [nm]")
plt.xlim(right=oopao_df * nx_in / 2)
plt.ylim(bottom=0)
plt.legend()


plt.subplot(233)
plt.title("PSF OOPAO WFS")
setimg(psf_ngs_le, 1)


plt.subplot(234)
plt.title("WFS PSF @ %u nm" % src_wfs.wvl_nm)

# OOPAO PSF centered on 4 pixels
y_oopao = circavg(
    psf_ngs_le / np.sum(psf_ngs_le),
    center=psf_ngs_le.shape[0] // 2 - 0.5,
    rmax=psf_ngs_le.shape[0] // 2,
)
x_oopao = (np.arange(len(y_oopao)) + 0.5) / ngs_cam.psf_sampling

# aopera PSF centered on 1 pixel
y_aopera = circavg(psf_ao_ngs, rmax=nx_aopera // 2)
x_aopera = np.arange(len(y_aopera)) / samp
y_aopera_lin = circavg(psf_ao_ngs_lin, rmax=nx_aopera // 2)

y_diff = circavg(psf_diff_ngs, rmax=nx_aopera // 2)

plt.loglog(x_oopao, y_oopao, label="OOPAO : Sr=%.1f%%" % sr_oopao_ngs)
plt.loglog(x_aopera, y_aopera, label="AOPERA : Sr=%.1f%%" % strehl_wfs)
# plt.loglog(x_aopera, y_aopera_lin, label='AOPERA linear', zorder=0)
plt.loglog(x_aopera, y_diff, c="grey", zorder=0, label="diffraction", alpha=0.7)
setplt()
plt.ylim(bottom=1e-7)

plt.subplot(235)
plt.title("Science PSF @ %u nm" % src_sci.wvl_nm)

# OOPAO PSF centered on 4 pixels
y_oopao = (
    circavg(
        psf_src_le / np.sum(psf_src_le),
        center=psf_src_le.shape[0] // 2 - 0.5,
        rmax=psf_src_le.shape[0] // 2,
    )
    * src_cam.psf_sampling**2
)
x_oopao = (np.arange(len(y_oopao)) + 0.5) / src_cam.psf_sampling

# aopera PSF centered on 1 pixel
y_aopera = circavg(psf_ao_sci, rmax=nx_aopera // 2) * samp_sci**2
x_aopera = np.arange(len(y_aopera)) / samp_sci
y_aopera_lin = circavg(psf_ao_sci_lin, rmax=nx_aopera // 2) * samp_sci**2

y_diff = circavg(psf_diff_sci, rmax=nx_aopera // 2) * samp_sci**2

plt.loglog(x_oopao, y_oopao, label="OOPAO : Sr=%.1f%%" % sr_oopao_src)
plt.loglog(x_aopera, y_aopera, label="AOPERA : Sr=%.1f%%" % strehl_sci)
# plt.loglog(x_aopera, y_aopera_lin, label='AOPERA linear', zorder=0)
plt.loglog(x_aopera, y_diff, c="grey", zorder=0, label="diffraction", alpha=0.7)
setplt()
plt.ylim(bottom=1e-6)

plt.subplot(236)
plt.title("PSF AOPERA WFS")
setimg(psf_ao_ngs, 1)  # samp_sci/src_cam.psf_sampling)

plt.tight_layout()

# %% PLOT MODULATION and OPTICAL GAINS
if not use_sh:

    og_mean = np.mean(og_oopao[:, close_loop_iter:i_lp], axis=1)
    og_std = np.std(og_oopao[:, close_loop_iter:i_lp], axis=1)

    def mode2freq(modes):
        return np.sqrt(modes / np.pi) / diameter

    freq_oopao = mode2freq(np.arange(len(og_mean)))

    obj = src_wfs.image(nx_aopera, pupil.diameter, samp)
    msk = wfs_aopera.modulation_path(nx_aopera, samp, modu=wfs_aopera.modulation_sky)
    img_sky = rfftconvolve(obj, msk)
    ext = nx_aopera / samp / 2
    xlim = 1.2 * max(obj_diameter_wvl_D / 2, modulation_radius)

    if obj_diameter_arcsec:
        og_aopera = wfs_aopera.optical_gain(
            samp, psf_diff_ngs, rfftconvolve(psf_ao_ngs, obj)
        )
    else:
        og_aopera = wfs_aopera.optical_gain(samp, psf_diff_ngs, psf_ao_ngs)
    og_aopera = circavg(og_aopera)
    freq_aopera = np.arange(len(og_aopera)) * pupil.frequency_step(samp)

    plt.figure(2, figsize=(15, 5))
    plt.clf()

    plt.subplot(131)
    plt.title("Calib modulation")
    plt.imshow(
        wfs_aopera.modulation_path(nx_aopera, samp), extent=(-ext, ext, -ext, ext)
    )
    plt.scatter(xx_mod_oopao_calib, yy_mod_oopao_calib, alpha=0.8, c="r", label="OOPAO")
    plt.axhline(0, c="w", ls=":")
    plt.axvline(0, c="w", ls=":")
    plt.axis("square")
    plt.xlim(-xlim, xlim)
    plt.ylim(-xlim, xlim)
    plt.xlabel("Position [$\\lambda/D$]")
    plt.ylabel("Position [$\\lambda/D$]")
    plt.legend()

    plt.subplot(132)
    plt.title("Sky modulation")
    plt.imshow(img_sky, extent=(-ext, ext, -ext, ext))
    plt.scatter(xx_mod_oopao, yy_mod_oopao, alpha=0.8, c="r", label="OOPAO")
    plt.axhline(0, c="w", ls=":")
    plt.axvline(0, c="w", ls=":")
    plt.axis("square")
    plt.xlim(-xlim, xlim)
    plt.ylim(-xlim, xlim)
    plt.xlabel("Position [$\\lambda/D$]")
    plt.ylabel("Position [$\\lambda/D$]")
    plt.legend()

    plt.subplot(133)
    plt.ylabel("Optical gains")
    plt.plot(freq_oopao, og_mean, label="OOPAO")
    for k in range(3):
        plt.fill_between(
            freq_oopao,
            og_mean - (k + 1) * og_std,
            og_mean + (k + 1) * og_std,
            alpha=0.3,
            color="C0",
            ec="none",
        )
    plt.plot(freq_aopera, og_aopera, label="AOPERA")
    plt.ylim(0.5, 1.0)
    plt.xlabel("Frequency [1/m]")
    plt.legend()
    plt.xlim(0, np.max(freq_oopao))
    plt.tight_layout()

# %%
