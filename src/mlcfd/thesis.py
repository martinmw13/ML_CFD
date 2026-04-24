"""Notebook helpers that mirror the legacy ``dataprocess`` module API."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from mlcfd.config.schemas import MeshConfig
from mlcfd.io.storage import read_matrix_csv
from mlcfd.mesh.mesh import Mesh
from mlcfd.visualization.plots import plot_reconstruction_error, save_modes

PathLike = str | Path


def thesis_mesh(
    nx: int,
    ny: int,
    lx: float,
    ly: float,
    x0: float,
    y0: float,
    r: float,
) -> Mesh:
    """Build a :class:`Mesh` using thesis-style names (Lx, Ly, X0, Y0) as parameters."""
    return Mesh(
        MeshConfig(
            nx=nx,
            ny=ny,
            lx=lx,
            ly=ly,
            x0=x0,
            y0=y0,
            r=r,
        ),
    )


def read_snapshot_matrix(data_path: PathLike) -> NDArray[np.float64]:
    """Read a CSV snapshot matrix as ``float64`` (legacy ``read_X_csv``)."""
    return read_matrix_csv(Path(data_path))


read_X_csv = read_snapshot_matrix  # noqa: N816


def save_thesis_modes(
    x_modes: NDArray[np.floating],
    outfile_dir: PathLike,
    *args: int | Mesh,
) -> None:
    """``save_modes`` with thesis argument ordering (matches most notebooks)."""
    if x_modes.ndim < 2:
        msg = f"Mode matrix must be at least 2D for export, got shape {x_modes.shape}"
        raise ValueError(msg)
    if len(args) == 1 and isinstance(args[0], Mesh):
        mesh = args[0]
        n_modes = int(x_modes.shape[1])
    elif len(args) == 2 and isinstance(args[0], int) and isinstance(args[1], Mesh):
        n_modes, mesh = int(args[0]), args[1]
    else:
        msg = "Expected (x, dir, mesh) or (x, dir, n_modes, mesh) like legacy notebooks."
        raise TypeError(msg)
    n_modes = min(n_modes, int(x_modes.shape[1]))
    save_modes(x_modes, Path(outfile_dir), n_modes, mesh)


def plot_save_reconst(
    err_rec: NDArray[np.floating],
    r_max: int,
    r_step: int,
    dr_method: str = "SVD",
    output_path: PathLike | None = None,
) -> None:
    """Wrapper around :func:`mlcfd.visualization.plots.plot_reconstruction_error` for notebooks."""
    out = Path(f"{dr_method}_rec_error.png") if output_path is None else Path(output_path)
    plot_reconstruction_error(err_rec, r_max, r_step, str(dr_method), out)
