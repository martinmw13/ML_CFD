"""Direct ``write_run_results`` behaviour, driven with synthetic outcomes.

Exercises the artifact contract — filenames, formats, and the
``save_modes`` / ``save_plots`` flag decisions — without fitting any model.
The mesh uses an empty cylinder mask (cylinder placed far outside the domain)
so every grid point is a free row and synthetic modes reshape onto the grid.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mlcfd.config.schemas import MeshConfig, OutputConfig
from mlcfd.mesh.mesh import Mesh
from mlcfd.pipeline.results import SweepOutcome, TrainableOutcome, write_run_results

NX, NY = 6, 4
R_MAX, R_STEP = 3, 1


def _mesh() -> Mesh:
    """Mesh whose cylinder sits far outside the domain, leaving an empty mask."""
    return Mesh(MeshConfig(nx=NX, ny=NY, lx=1.0, ly=1.0, x0=-10.0, y0=-10.0, r=0.5))


def _sweep_outcome(*, with_modes: bool) -> SweepOutcome:
    """Synthetic sweep outcome; ``with_modes`` toggles a full-grid modal basis."""
    rng = np.random.default_rng(0)
    errors = rng.random(R_MAX)
    modes = rng.standard_normal((NX * NY, R_MAX)) if with_modes else None
    return SweepOutcome(
        model_name="pca_svd",
        errors=errors,
        r_max=R_MAX,
        r_step=R_STEP,
        modes=modes,
    )


def test_sweep_writes_all_artifacts_when_flags_enabled(tmp_path: Path) -> None:
    """Both flags on: error curve, error plot, and leading ``mode_*`` files appear."""
    outcome = _sweep_outcome(with_modes=True)
    output = OutputConfig(output_dir=tmp_path / "out", save_modes=True, save_plots=True)

    write_run_results(outcome, output, _mesh())

    out_dir = output.output_dir
    assert (out_dir / "pca_svd_error_rec.csv").exists()
    assert (out_dir / "pca_svd_rec_error.png").exists()
    assert (out_dir / "mode_1.csv").exists()
    assert (out_dir / "mode_1.png").exists()
    assert (out_dir / f"mode_{R_MAX}.csv").exists()
    assert (out_dir / f"mode_{R_MAX}.png").exists()


def test_sweep_error_csv_round_trips_without_header(tmp_path: Path) -> None:
    """The error curve persists as a bare 1D column (np.savetxt layout, no header)."""
    outcome = _sweep_outcome(with_modes=False)
    output = OutputConfig(output_dir=tmp_path / "out", save_modes=False, save_plots=False)

    write_run_results(outcome, output, _mesh())

    csv_path = output.output_dir / "pca_svd_error_rec.csv"
    loaded = np.loadtxt(csv_path, delimiter=",")
    np.testing.assert_allclose(loaded, outcome.errors)


def test_sweep_skips_plot_and_modes_when_flags_disabled(tmp_path: Path) -> None:
    """Both flags off: only the error curve is written, no plot or modes."""
    outcome = _sweep_outcome(with_modes=True)
    output = OutputConfig(output_dir=tmp_path / "out", save_modes=False, save_plots=False)

    write_run_results(outcome, output, _mesh())

    out_dir = output.output_dir
    assert (out_dir / "pca_svd_error_rec.csv").exists()
    assert not (out_dir / "pca_svd_rec_error.png").exists()
    assert not list(out_dir.glob("mode_*.csv"))
    assert not list(out_dir.glob("mode_*.png"))


def test_sweep_skips_modes_when_basis_absent(tmp_path: Path) -> None:
    """``save_modes`` on but a ``None`` basis skips mode export without crashing."""
    outcome = _sweep_outcome(with_modes=False)
    output = OutputConfig(output_dir=tmp_path / "out", save_modes=True, save_plots=False)

    write_run_results(outcome, output, _mesh())

    out_dir = output.output_dir
    assert (out_dir / "pca_svd_error_rec.csv").exists()
    assert not list(out_dir.glob("mode_*.csv"))


def test_trainable_writes_metrics_json(tmp_path: Path) -> None:
    """A trainable outcome writes only its metrics JSON, into a fresh output dir."""
    outcome = TrainableOutcome(model_name="ae_spatial", metrics={"mse": 0.25, "r2": 0.9})
    output = OutputConfig(output_dir=tmp_path / "out", save_modes=True, save_plots=True)

    write_run_results(outcome, output, _mesh())

    metrics_path = output.output_dir / "ae_spatial_metrics.json"
    assert metrics_path.exists()
    assert json.loads(metrics_path.read_text(encoding="utf-8")) == outcome.metrics
