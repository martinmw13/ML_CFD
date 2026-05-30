"""End-to-end ``save_modes`` behaviour for a sweep run.

Drives :func:`run_from_config` with a synthetic snapshot CSV and a mesh whose
cylinder mask is empty (so every grid point is a free row), exercising the real
orchestration path without touching CFD data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mlcfd.config.schemas import (
    DataConfig,
    MeshConfig,
    OutputConfig,
    RunConfig,
    SweepModelConfig,
)
from mlcfd.io.storage import write_matrix_csv
from mlcfd.pipeline.runner import run_from_config

NX, NY = 6, 4
N_SNAPSHOTS = 8
R_MAX = 3


def _make_run(
    tmp_path: Path, *, save_modes: bool, model_name: str = "pca_svd"
) -> RunConfig:
    """Build a sweep run config backed by a synthetic snapshot matrix on disk."""
    input_dir = tmp_path / "data"
    input_dir.mkdir()
    matrix = np.random.default_rng(0).standard_normal((NX * NY, N_SNAPSHOTS))
    write_matrix_csv(input_dir / "modVcropRe50.csv", matrix)

    return RunConfig(
        # x0/y0 far outside the domain with a small radius -> empty cylinder mask,
        # so n_free == nx * ny and the SVD basis maps cleanly back onto the grid.
        mesh=MeshConfig(nx=NX, ny=NY, lx=1.0, ly=1.0, x0=-10.0, y0=-10.0, r=0.5),
        data=DataConfig(input_dir=input_dir, reynolds_number=50),
        output=OutputConfig(
            output_dir=tmp_path / "out", save_modes=save_modes, save_plots=False
        ),
        model_name=model_name,
        model_params=SweepModelConfig(r_max=R_MAX, r_step=1),
    )


def test_run_writes_mode_artifacts_when_enabled(tmp_path: Path) -> None:
    """``save_modes: true`` writes ``mode_*.csv`` and ``mode_*.png`` for the leading modes."""
    run = _make_run(tmp_path, save_modes=True)

    run_from_config(run)

    out_dir = run.output.output_dir
    assert (out_dir / "mode_1.csv").exists()
    assert (out_dir / "mode_1.png").exists()
    assert (out_dir / f"mode_{R_MAX}.csv").exists()
    assert (out_dir / f"mode_{R_MAX}.png").exists()


def test_run_writes_no_mode_artifacts_when_disabled(tmp_path: Path) -> None:
    """``save_modes: false`` produces no mode artifacts (the run still completes)."""
    run = _make_run(tmp_path, save_modes=False)

    run_from_config(run)

    out_dir = run.output.output_dir
    assert not list(out_dir.glob("mode_*.csv"))
    assert not list(out_dir.glob("mode_*.png"))


def test_run_skips_modes_for_model_without_basis(tmp_path: Path) -> None:
    """A non-modal sweep model with ``save_modes: true`` skips cleanly, without crashing."""
    run = _make_run(tmp_path, save_modes=True, model_name="pca_sklearn")

    run_from_config(run)

    out_dir = run.output.output_dir
    # The run still completes and writes its error curve, but no modes.
    assert (out_dir / "pca_sklearn_error_rec.csv").exists()
    assert not list(out_dir.glob("mode_*.csv"))
    assert not list(out_dir.glob("mode_*.png"))
