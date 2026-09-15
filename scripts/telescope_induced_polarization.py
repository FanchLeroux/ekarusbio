# %%

import numpy as np
import matplotlib.pyplot as plt
from ekarusbio.config import Config
from ekarusbio.miscellaneous import compact_square_layout

config = Config()

# %% inspect dark

data_dir = config.root_dir / "data" / "20260902_Polarization_tests" / "Measurement 2"

dark_filename = data_dir / "20260902_211436" / "20260902_211436_orca_data.npy"

dark_dict = np.load(dark_filename, allow_pickle=True).item()

print(dark_dict.keys())

for key, value in dark_dict.items():
    print(
        f"{key}: type={type(value)}, shape={getattr(value, 'shape', None)}, dtype={getattr(value, 'dtype', None)}"
    )

dark_cube = dark_dict["orcaCube"]
dark_bg = dark_dict["orcaBg"]

fig_dark, ax = plt.subplots(
    1,
    3,
    constrained_layout=True,
)

im0 = ax[0].imshow(dark_cube.mean(axis=0), cmap="viridis")
im1 = ax[1].imshow(dark_bg, cmap="viridis")
im2 = ax[2].imshow(dark_cube.mean(axis=0) - dark_bg, cmap="viridis")

for axis in ax:
    axis.set_xticks([])
    axis.set_yticks([])

# Colorbar for first two images
cbar01 = fig_dark.colorbar(
    im0,
    ax=ax[:2],
    location="right",
    shrink=0.35,  # shorter
    aspect=20,  # thinner
    pad=0.03,  # distance from images
)

# Colorbar for difference image
cbar2 = fig_dark.colorbar(
    im2,
    ax=ax[2],
    location="right",
    shrink=0.35,
    aspect=20,
    pad=0.03,
)

ax[0].set_title("orcaCube mean frame", fontsize=10)
ax[1].set_title("orcaBg", fontsize=10)
ax[2].set_title("orcaCube mean frame - orcaBg", fontsize=10)

fig_dark.savefig(
    config.root_dir / "outputs" / "darks.svg",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.01,
)

# %%

print(np.allclose(dark_cube.mean(axis=0), dark_bg))

# orientations

orientations = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360]  # [deg]

ids = [
    "20260902_212145",
    "20260902_212232",
    "20260902_212312",
    "20260902_212426",
    "20260902_212520",
    "20260902_212553",
    "20260902_212624",
    "20260902_212701",
    "20260902_212729",
    "20260902_212754",
    "20260902_212817",
    "20260902_212841",
    "20260902_212907",
]

data_images = []
irradiance_max = []
irradiance_mean = []

for id in ids:
    data_filename = data_dir / id / f"{id}_orca_data.npy"

    data_dict = np.load(data_filename, allow_pickle=True).item()
    data_cube = data_dict["orcaCube"]

    data = data_cube.mean(axis=0) - dark_bg

    data_images.append(data)
    irradiance_max.append(data.max())
    irradiance_mean.append(data.mean())

irradiance_max = np.array(irradiance_max)
irradiance_mean = np.array(irradiance_mean)

vmin = min(data.min() for data in data_images)
vmax = max(data.max() for data in data_images)

fig_data, axs_data = plt.subplots(
    *compact_square_layout(len(ids)),
    figsize=(config.width_single_column, 1.0 * config.width_single_column),
    constrained_layout=True,
)
axs_data = axs_data.ravel()  # flatten the 2D array of axes to 1D

for i, (orientation, data) in enumerate(zip(orientations, data_images)):
    im = axs_data[i].imshow(
        data,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )
    axs_data[i].set_title(f"{orientation}°", fontsize=10)
    axs_data[i].set_xticks([])
    axs_data[i].set_yticks([])

for j in range(i + 1, len(axs_data)):
    axs_data[j].axis("off")  # turn off unused axes

fig_data.colorbar(
    im,
    ax=axs_data[: len(data_images)],
    location="bottom",
    fraction=0.04,
    pad=0.04,
    aspect=40,
    label="Irradiance (a.u.)",
)

fig_data.savefig(
    config.root_dir / "outputs" / f"data_images.svg",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.01,
)

# %% plot irradiance dependance with polarizor orientation

irradiance_max_normalized = irradiance_max / irradiance_max.max()
irradiance_mean_normalized = irradiance_mean / irradiance_mean.max()

relative_error_irradiance_max = (
    irradiance_max_normalized.max() - irradiance_max_normalized.min()
) / irradiance_max_normalized.mean()
print(f"{relative_error_irradiance_max:.2%}")

relative_error_irradiance_mean = (
    irradiance_mean_normalized.max() - irradiance_mean_normalized.min()
) / irradiance_mean_normalized.mean()
print(f"{relative_error_irradiance_mean:.2%}")

fig_telescope_induced_polarization = plt.figure(
    figsize=(config.width_single_column, 1.0 * config.width_single_column),
    constrained_layout=True,
)
plt.plot(orientations, irradiance_max_normalized, marker="o", label="Max")
plt.plot(orientations, irradiance_mean_normalized, marker="s", label="Mean")
plt.axhline(
    y=irradiance_max_normalized.mean(), color="r", linestyle="--", label="Mean (Max)"
)
plt.axhline(
    y=irradiance_mean_normalized.mean(), color="b", linestyle="--", label="Mean (Mean)"
)
plt.legend(loc="lower left", fontsize=8)
plt.xlabel("Orientation [deg]")
plt.ylabel("Normalized irradiance")
plt.title(
    f"Telescope-induced polarization"
    f"\nRelative error (Max): {relative_error_irradiance_max:.2%}, "
    f"Relative error (Mean): {relative_error_irradiance_mean:.2%}\n"
)

fig_telescope_induced_polarization.savefig(
    config.root_dir / "outputs" / "telescope_induced_polarization.svg",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.01,
)
