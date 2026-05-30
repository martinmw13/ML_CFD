"""High-level orchestration for CLI and notebook entry points."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from mlcfd.config.schemas import OutputConfig, RunConfig
from mlcfd.io.storage import ensure_directory, read_matrix_csv
from mlcfd.logging_config import get_logger
from mlcfd.mesh.mesh import Mesh
from mlcfd.models.base import SweepableModel
from mlcfd.models.factory import (
    build_sweep_model,
    build_trainable_model,
    is_sweep_model,
    is_trainable_model,
)
from mlcfd.preprocessing.pipeline import DataPipeline
from mlcfd.visualization.plots import plot_reconstruction_error, save_modes

LOGGER = get_logger("pipeline")


def export_sweep_modes(
    model: SweepableModel,
    output: OutputConfig,
    mesh: Mesh,
    out_dir: Path,
) -> None:
    """Export spatial modes for a fitted sweep model, honouring ``output.save_modes``.

    When the flag is set and the model exposes a modal basis (currently PCA-SVD), the
    leading ``min(params.r_max, n_modes)`` columns are written as ``mode_*.csv`` and
    ``mode_*.png`` through :func:`mlcfd.visualization.plots.save_modes`. Models that
    return ``None`` from :meth:`SweepableModel.spatial_modes` are skipped with a log
    message rather than crashing.
    """
    if not output.save_modes:
        return
    modes = model.spatial_modes()
    if modes is None:
        LOGGER.info("Model exposes no spatial modes; skipping mode export")
        return
    n_modes = min(model.params.r_max, modes.shape[1])
    save_modes(modes, out_dir, n_modes, mesh)
    LOGGER.info("Wrote %s spatial modes to %s", n_modes, out_dir)


def run_from_yaml(path: Path) -> None:
    """Load a YAML ``RunConfig``, validate it, and execute the experiment."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    run = RunConfig.model_validate(raw)
    run_from_config(run)


def run_from_config(run: RunConfig) -> None:
    """Execute preprocessing, model fitting, and artifact export."""
    LOGGER.info("Starting pipeline for model %s", run.model_name)
    mesh = Mesh(run.mesh)
    pipeline = DataPipeline(mesh, run.data)
    matrix = read_matrix_csv(run.data.snapshot_csv_path())
    x_train, x_test = pipeline.run(matrix)
    out_dir = run.output.output_dir
    ensure_directory(out_dir)

    if is_sweep_model(run.model_name):
        model = build_sweep_model(run)
        model.fit(x_train)
        _, errors = model.reconstruction_error(x_test)
        csv_path = out_dir / f"{run.model_name}_error_rec.csv"
        np.savetxt(csv_path, errors, delimiter=",")
        LOGGER.info("Wrote sweep errors to %s", csv_path)
        if run.output.save_plots:
            params = model.params
            plot_path = out_dir / f"{run.model_name}_rec_error.png"
            plot_reconstruction_error(
                errors,
                params.r_max,
                params.r_step,
                run.model_name.upper(),
                plot_path,
            )
        export_sweep_modes(model, run.output, mesh, out_dir)
    elif is_trainable_model(run.model_name):
        trainable = build_trainable_model(run)
        trainable.fit(x_train)
        metrics = trainable.evaluate(x_test)
        metrics_path = out_dir / f"{run.model_name}_metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        LOGGER.info("Wrote metrics to %s", metrics_path)
    else:
        msg = f"Unsupported model_name: {run.model_name}"
        raise ValueError(msg)
