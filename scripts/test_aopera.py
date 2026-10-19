# %%

"""
Perform a full AO simulation with AOPERA.

Outputs:
    - AO error budget
    - phase PSDs
    - PSF
    - Strehl ratio

date: 2025-02-12
author: rfetick
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from aopera.turbulence import Atmosphere
from aopera.control import RTC
from aopera.otfpsf import Pupil
from aopera.ffwfs import PWFS
from aopera.shwfs import SHWFS
from aopera.photometry import SourceScience, SourceWFS, Band
from aopera.utils import rad2arcsec, circavg, circsum, print_std
from aopera.simulation import simulation
from aopera.variance import var_temporal, var_fitting

# %% PARAMETERS
file_atmo = "ohp.ini"
file_instru = "papyrus.ini"
nx = 512  # number of pixels for simulation
samp = 3  # sampling at WFS wavelength
magV = 6
zenith_deg = 30

# %% READ CONFIG FILE
src_sci = SourceScience.from_file(file_instru)
src_wfs = SourceWFS.from_file(file_instru)
atm = Atmosphere.from_file(file_atmo)
tel = Pupil.from_file(file_instru)
rtc = RTC.from_file(file_instru)
try:
    wfs = PWFS.from_file(file_instru)
except:
    wfs = SHWFS.from_file(file_instru)

src_sci.zenith_deg = zenith_deg
src_wfs.flux = Band("V").mag2photon(magV)

print(src_sci)
print(src_wfs)
print(atm)
print(tel)
print(rtc)
print(wfs)

# %% AO SIMULATION
wfs.modulation_sky = src_wfs.size == 0
result = simulation(nx, samp, tel, atm, rtc, wfs, src_sci, src_wfs, verbose=True)

psf_ao = result.psf_ao(src_wfs.wvl)
psf_diff = result.psf_diffraction(src_wfs.wvl)
strehl = 100 * result.strehl(src_wfs.wvl)

psf_ao_sci = result.psf_ao(src_sci.wvl)
psf_diff_sci = result.psf_diffraction(src_sci.wvl)
strehl_sci = 100 * result.strehl(src_sci.wvl)

# %% ANALYTICAL COMPUTATION
r0 = atm.r0(src_wfs.wvl, zenith=src_sci.zenith_rad)
var_ana = {"temporal": 0}
for i in range(atm.nlayer):
    bw = rtc.bandwidth * np.sqrt(result.param["og_avg"])
    var_ana["temporal"] += (
        atm.cn2dh_ratio[i]
        * var_temporal(tel.diameter, r0, tel.radial_max, atm.wind_speed[i], 1e3)
        * 1e6
        / bw**2
    )

var_ana["fitting"] = var_fitting(r0, 1 / (2 * tel.pitch_dm))
var_ana["aliasing"] = var_ana["fitting"] / 10  # very rough computation
var_ana["ncpa"] = (tel.ncpa * 2 * np.pi / src_wfs.wvl_nm) ** 2

std_ana = {
    k: np.sqrt(var_ana[k]) * src_wfs.wvl_nm / (2 * np.pi) for k in var_ana.keys()
}

print()
print_std(std_ana, title="ANALYTICAL WAVEFRONT ERROR [nm RMS]")

# %% PRINT STREHL
print()
print("Strehl at %4u nm : %4.1f %%" % (src_wfs.wvl_nm, strehl))
print("Strehl at %4u nm : %4.1f %%" % (src_sci.wvl_nm, strehl_sci))
print()

# %% GET CIRCAVG for PLOT
print("Computing circular averages...")

df = tel.frequency_step(samp)
rad_to_nm = 1e9 * src_wfs.wvl / (2 * np.pi)
psdtotavg = circavg(result.psd_total) * rad_to_nm**2

pf = tel.piston_filter(nx, samp)
psdtotsum = circsum(pf * result.psd_total * df**2) * rad_to_nm**2

psdavg = {}
for k in result.psd.keys():
    psdavg[k] = circavg(result.psd[k]) * rad_to_nm**2

psdsum = {}
for k in result.psd.keys():
    if k == "temporal":
        pf = tel.piston_filter(nx, samp)
    else:
        pf = 1
    psdsum[k] = circsum(pf * result.psd[k] * df**2) * rad_to_nm**2

# %% PLOT

lw = 2

plt.figure(1, figsize=(7, 4))
plt.clf()

x = np.arange(len(psdtotavg)) * df

plt.subplot(121)
for k in psdavg.keys():
    plt.loglog(x, psdavg[k], lw=lw, label="%s" % k)
plt.loglog(x, psdtotavg, lw=lw, c="k", ls=":", label="AO total")
plt.xlim(x[1], 60)
plt.ylim(1e-2, 1e4)
plt.grid()
plt.title("PSD [nm²m²]")
plt.xlabel("Frequency [1/m]")
plt.legend(handlelength=1.2)

plt.subplot(122)
for k in psdavg.keys():
    plt.semilogx(x, np.sqrt(psdsum[k]), lw=lw, label="%s" % k)
plt.semilogx(x, np.sqrt(psdtotsum), lw=lw, c="k", ls=":", label="AO total")
plt.xlim(x[1], 60)
plt.ylim(bottom=0)
plt.grid()
plt.title("cumul. OPD [nm RMS]")
plt.xlabel("Frequency [1/m]")
plt.legend(handlelength=1.2)

plt.tight_layout()


def setplt(tab, nc=256, hfov=2, norm=LogNorm(vmin=1e-7, vmax=1e-1), **kwargs):
    pix_size_arcsec = rad2arcsec((src_wfs.wvl / tel.diameter) / samp)
    x = (np.arange(2 * nc) - nc) * pix_size_arcsec
    out = plt.pcolormesh(
        x,
        x,
        tab[nx // 2 - nc : nx // 2 + nc, nx // 2 - nc : nx // 2 + nc],
        norm=norm,
        **kwargs,
    )
    plt.axis("image")
    plt.xlim(-hfov, hfov)
    plt.ylim(-hfov, hfov)
    return out


plt.figure(2, figsize=(6, 5.5))
plt.clf()

plt.subplot(221)
setplt(psf_diff)
plt.ylabel("Position [arcsec]")
plt.title("WFS diffraction")

plt.subplot(222)
setplt(psf_ao)
plt.title("WFS AO : Sr=%.1f%%" % strehl)

plt.subplot(223)
setplt(psf_diff_sci, hfov=5)
plt.ylabel("Position [arcsec]")
plt.title("SCI diffraction")

plt.subplot(224)
out = setplt(psf_ao_sci, hfov=4)
plt.title("SCI AO : Sr=%.1f%%" % strehl_sci)

fig = plt.gcf()
fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.89, 0.15, 0.03, 0.7])
fig.colorbar(out, cax=cbar_ax)

plt.show()
