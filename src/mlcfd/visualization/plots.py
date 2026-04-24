"""Plotting helpers for modes and reconstruction diagnostics."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import image as mpl_image
from matplotlib.colors import LinearSegmentedColormap
from numpy.typing import NDArray

from mlcfd.io.storage import ensure_directory, write_matrix_csv
from mlcfd.logging_config import get_logger
from mlcfd.mesh.mesh import Mesh
from mlcfd.preprocessing.pipeline import reconstruct_field

LOGGER = get_logger("visualization")

PLOT_DPI: int = 200

MODE_COLORMAP_POINTS: list[tuple[float, str]] = [
    (0.0, "cyan"),
    (0.40, "blue"),
    (0.5, "black"),
    (0.60, "red"),
    (1.0, "yellow"),
]

MODE_COLORMAP = LinearSegmentedColormap.from_list("mlcfd_mode_cmap", MODE_COLORMAP_POINTS)


def save_modes(
    x_modes: NDArray[np.floating],
    output_dir: Path,
    modes: int,
    mesh: Mesh,
) -> None:
    """Persist spatial modes as CSV and PNG images.

    Args:
        x_modes: Mode matrix with shape ``(n_free, n_modes)`` matching masked rows.
        output_dir: Directory to write ``mode_*.csv`` and ``mode_*.png``.
        modes: Number of leading modes to export.
        mesh: Mesh used to reconstruct full-grid layouts.

    Raises:
        ValueError: If ``modes`` exceeds available columns.
    """
    if modes > x_modes.shape[1]:
        msg = f"Requested {modes} modes but matrix has {x_modes.shape[1]} columns"
        raise ValueError(msg)
    ensure_directory(output_dir)
    LOGGER.info("Saving %s modes under %s", modes, output_dir)
    for i in range(modes):
        column = x_modes[:, i].reshape(-1, 1)
        field = reconstruct_field(column, mesh)[:, 0]
        min_val = float(np.min(field))
        max_val = float(np.max(field))
        if max_val > min_val:
            field = (field - min_val) / (max_val - min_val)
        grid = field.reshape((mesh.ny, mesh.nx))
        csv_path = output_dir / f"mode_{i + 1}.csv"
        png_path = output_dir / f"mode_{i + 1}.png"
        write_matrix_csv(csv_path, grid)
        mpl_image.imsave(png_path, grid, cmap=MODE_COLORMAP)


def plot_reconstruction_error(
    err_rec: NDArray[np.floating],
    r_max: int,
    r_step: int,
    method_name: str,
    output_path: Path,
) -> None:
    """Plot relative reconstruction error versus retained mode count.

    Args:
        err_rec: Relative errors aligned with ``range(1, r_max + 1, r_step)``.
        r_max: Maximum mode count used in the sweep.
        r_step: Step between evaluated mode counts.
        method_name: Label used in the title and log messages.
        output_path: Destination image path (``.png`` recommended).
    """
    ensure_directory(output_path.parent)
    x_vals = list(range(1, r_max + 1, r_step))
    if len(x_vals) != len(err_rec):
        msg = f"x length {len(x_vals)} does not match err_rec length {len(err_rec)}"
        raise ValueError(msg)
    LOGGER.info("Writing reconstruction error plot to %s", output_path)
    figure, axis = plt.subplots()
    axis.grid(True, which="both")
    axis.plot(x_vals, err_rec, linestyle="--", marker="o")
    axis.set_xlabel("Modes used for reconstruction")
    axis.set_ylabel("Relative reconstruction error")
    axis.set_title(f"{method_name} reconstruction error")
    axis.semilogy()
    figure.tight_layout()
    figure.savefig(output_path, dpi=PLOT_DPI)
    plt.close(figure)
