import numpy as np
import matplotlib.pyplot as plt


def get_labels(dimension: int, remove_piston: bool = False):
    labels = []

    half = dimension // 2
    nyquist = dimension % 2 == 0

    freq_x = np.arange(-half + 1, half + 1)
    freq_y = np.arange(0, half + 1)

    for nu_x in freq_x:
        for nu_y in freq_y:

            # -------------------
            # Special cases
            # -------------------
            special = (
                (nu_x == 0 and nu_y == 0)
                or (nyquist and nu_x == 0 and nu_y == half)
                or (nyquist and nu_x == half and nu_y == 0)
                or (nyquist and nu_x == half and nu_y == half)
            )

            if special:
                if nu_x == 0 and nu_y == 0:
                    if not remove_piston:
                        labels.append((0, 0, "piston"))
                else:
                    labels.append((nu_x, nu_y, "cos"))

                continue

            # -------------------
            # General case
            # -------------------
            if (nu_y != 0 or nu_x >= 0) and (nu_y != half or nu_x >= 0):

                labels.append((nu_x, nu_y, "cos"))
                labels.append((nu_x, nu_y, "sin"))

    labels.sort(
        key=lambda label: (
            label[0] ** 2 + label[1] ** 2,
            label[0],
            label[1],
            label[2],
        )
    )

    return labels


def compute_fourier_mode(
    n_pixels: int,
    nu_x: int,
    nu_y: int,
    kind="cos",
    pupil_mask=None,
    shift_x: float = 0.0,
    shift_y: float = 0.0,
):
    coords = np.arange(-n_pixels // 2, n_pixels // 2)
    X, Y = np.meshgrid(coords, coords)
    X = X - shift_x
    Y = Y - shift_y

    phase = 2 * np.pi * (nu_x * X + nu_y * Y) / n_pixels

    if kind == "piston":
        mode = np.ones(phase.shape)
    elif kind == "sin":
        mode = np.sin(phase)
    elif kind == "cos":
        mode = np.cos(phase)
    else:
        raise ValueError("kind must be 'sin' or 'cos' or 'piston'")

    if pupil_mask is None:
        pupil_mask = np.ones(mode.shape, dtype=bool)

    mode *= pupil_mask

    if kind != "piston":
        mode[pupil_mask] -= mode[pupil_mask].mean()
        mode[pupil_mask] /= mode[pupil_mask].std()

    return mode


def compute_fourier_basis(
    n_pixels: int,
    labels: list | None = None,
    remove_piston: bool = False,
    pupil_mask: np.ndarray | None = None,
    return_labels: bool = False,
    shift_x: float = 0.0,
    shift_y: float = 0.0,
):
    basis = []

    if labels is None:
        labels = get_labels(n_pixels, remove_piston=remove_piston)

    for nu_x, nu_y, kind in labels:
        if kind == "piston" and remove_piston:
            continue
        mode = compute_fourier_mode(
            n_pixels, nu_x, nu_y, kind, pupil_mask, shift_x, shift_y
        )
        basis.append(mode)

    basis = np.array(basis)

    if return_labels:
        return basis, labels
    return basis


def draw_labels(labels, plot_hermitian=False, title=None):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")

    for nu_x, nu_y, kind in labels:
        if kind == "piston":
            ax.plot(
                nu_x,
                nu_y,
                marker="o",
                color="green",
                markersize=8,
            )
        if kind == "cos":
            ax.plot(
                nu_x,
                nu_y,
                marker="o",
                color="red",
                markersize=8,
            )
        if kind == "sin":
            ax.plot(
                nu_x,
                nu_y,
                marker="+",
                color="blue",
                markersize=8,
            )

        if plot_hermitian:
            ax.plot(
                -nu_x,
                -nu_y,
                marker="x",
                color="grey",
                markersize=8,
            )

    ax.set_xlabel(r"$\nu_x$")
    ax.set_ylabel(r"$\nu_y$")

    if title is not None:
        ax.set_title(title)

    return fig, ax


def draw_multiple_labels(
    labels_list,
    titles=None,
    plot_hermitian=False,
    ncols=None,
    markersize=4,
    figsize=None,
):

    n = len(labels_list)

    if ncols is None:
        ncols = n

    nrows = int(np.ceil(n / ncols))

    if figsize is None:
        figsize = (4 * ncols, 4 * nrows)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        squeeze=False,
    )

    axes = axes.ravel()

    for i, labels in enumerate(labels_list):
        ax = axes[i]
        ax.set_aspect("equal")

        for nu_x, nu_y, kind in labels:
            if kind == "piston":
                ax.plot(
                    nu_x,
                    nu_y,
                    marker="o",
                    color="green",
                    markersize=markersize,
                )

            elif kind == "cos":
                ax.plot(
                    nu_x,
                    nu_y,
                    marker="o",
                    color="red",
                    markersize=markersize,
                )

            elif kind == "sin":
                ax.plot(
                    nu_x,
                    nu_y,
                    marker="+",
                    color="blue",
                    markersize=markersize,
                )

            if plot_hermitian:
                ax.plot(
                    -nu_x,
                    -nu_y,
                    marker="x",
                    color="grey",
                    markersize=markersize,
                )

            ax.set_xlabel(r"$\nu_x$")
            ax.set_ylabel(r"$\nu_y$")

        if titles is not None:
            ax.set_title(titles[i])

    # Hide unused axes
    for ax in axes[n:]:
        ax.set_visible(False)

    fig.tight_layout()

    return fig, axes


def rotate_fourier_labels(
    labels,
    angle=np.pi / 4,
):
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    rotated_labels = []

    for nu_x, nu_y, kind in labels:
        fx = cos_a * nu_x - sin_a * nu_y
        fy = sin_a * nu_x + cos_a * nu_y

        rotated_labels.append((fx, fy, kind))

    return rotated_labels
