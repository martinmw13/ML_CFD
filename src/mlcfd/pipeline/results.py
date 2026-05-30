"""Run-results writer: the single home for a run's on-disk artifact contract.

A run produces an *outcome* — a dimension sweep's error curve plus an optional
modal basis, or a trainable model's metrics. This module owns what files those
outcomes become: filenames, formats, and the ``save_modes`` / ``save_plots``
decisions. The orchestration layer (:mod:`mlcfd.pipeline.runner`) builds an
outcome and hands it here, so persistence lives in one place and stays testable
without fitting a model.

Raw writes go through :mod:`mlcfd.io.storage`; figures and modal exports go
through :mod:`mlcfd.visualization.plots`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from mlcfd.config.schemas import OutputConfig
from mlcfd.io.storage import write_json, write_vector_csv
from mlcfd.logging_config import get_logger
from mlcfd.mesh.mesh import Mesh
from mlcfd.visualization.plots import plot_reconstruction_error, save_modes

LOGGER = get_logger("pipeline")


@dataclass(frozen=True, eq=False)
class SweepOutcome:
    """Result of a dimension-sweep run, ready to persist.

    Attributes:
        model_name: Model identifier used to derive artifact filenames.
        errors: Relative reconstruction error per evaluated mode count (1D).
        r_max: Maximum mode count used in the sweep (drives the plot x-axis).
        r_step: Step between evaluated mode counts.
        modes: Spatial mode basis with shape ``(n_free, n_modes)``, or ``None``
            when the model exposes no modal basis (modes are then skipped).
    """

    model_name: str
    errors: NDArray[np.floating]
    r_max: int
    r_step: int
    modes: NDArray[np.floating] | None


@dataclass(frozen=True, eq=False)
class TrainableOutcome:
    """Result of a trainable-model run, ready to persist.

    Attributes:
        model_name: Model identifier used to derive artifact filenames.
        metrics: Scalar evaluation metrics on the held-out set.
    """

    model_name: str
    metrics: dict[str, float]


def write_run_results(
    outcome: SweepOutcome | TrainableOutcome,
    output: OutputConfig,
    mesh: Mesh,
) -> None:
    """Persist a run outcome under ``output.output_dir``, honouring the save flags.

    Dispatches on the outcome type. The output directory is created on demand by
    the underlying :mod:`mlcfd.io.storage` writers, so callers need not prepare it.
    ``mesh`` is only consulted for sweep modal exports and is ignored for trainable
    outcomes.

    Args:
        outcome: Sweep or trainable run outcome to persist.
        output: Destination directory and ``save_modes`` / ``save_plots`` flags.
        mesh: Mesh used to reconstruct full-grid layouts for mode exports.
    """
    if isinstance(outcome, SweepOutcome):
        _write_sweep(outcome, output, mesh)
    else:
        _write_trainable(outcome, output)


def _write_sweep(outcome: SweepOutcome, output: OutputConfig, mesh: Mesh) -> None:
    """Write the sweep error curve, optional error plot, and optional modes."""
    out_dir = output.output_dir
    csv_path = out_dir / f"{outcome.model_name}_error_rec.csv"
    write_vector_csv(csv_path, outcome.errors)
    LOGGER.info("Wrote sweep errors to %s", csv_path)

    if output.save_plots:
        plot_path = out_dir / f"{outcome.model_name}_rec_error.png"
        plot_reconstruction_error(
            outcome.errors,
            outcome.r_max,
            outcome.r_step,
            outcome.model_name.upper(),
            plot_path,
        )

    _export_modes(outcome, output, mesh, out_dir)


def _export_modes(outcome: SweepOutcome, output: OutputConfig, mesh: Mesh, out_dir: Path) -> None:
    """Export spatial modes when enabled and the outcome carries a modal basis.

    Writes the leading ``min(r_max, n_modes)`` columns as ``mode_*.csv`` and
    ``mode_*.png``. A ``None`` basis is skipped with a log message rather than
    crashing, matching models that expose no meaningful modes.
    """
    if not output.save_modes:
        return
    if outcome.modes is None:
        LOGGER.info("Model exposes no spatial modes; skipping mode export")
        return
    n_modes = min(outcome.r_max, outcome.modes.shape[1])
    save_modes(outcome.modes, out_dir, n_modes, mesh)
    LOGGER.info("Wrote %s spatial modes to %s", n_modes, out_dir)


def _write_trainable(outcome: TrainableOutcome, output: OutputConfig) -> None:
    """Write trainable-model evaluation metrics as JSON."""
    metrics_path = output.output_dir / f"{outcome.model_name}_metrics.json"
    write_json(metrics_path, outcome.metrics)
    LOGGER.info("Wrote metrics to %s", metrics_path)
