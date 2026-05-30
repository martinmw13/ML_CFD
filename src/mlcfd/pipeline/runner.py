"""High-level orchestration for CLI and notebook entry points."""

from __future__ import annotations

from pathlib import Path

import yaml

from mlcfd.config.schemas import RunConfig
from mlcfd.io.storage import read_matrix_csv
from mlcfd.logging_config import get_logger
from mlcfd.mesh.mesh import Mesh
from mlcfd.models.factory import (
    build_sweep_model,
    build_trainable_model,
    is_sweep_model,
    is_trainable_model,
)
from mlcfd.pipeline.results import SweepOutcome, TrainableOutcome, write_run_results
from mlcfd.preprocessing.pipeline import DataPipeline

LOGGER = get_logger("pipeline")


def run_from_yaml(path: Path) -> None:
    """Load a YAML ``RunConfig``, validate it, and execute the experiment."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    run = RunConfig.model_validate(raw)
    run_from_config(run)


def run_from_config(run: RunConfig) -> None:
    """Execute preprocessing and model fitting, then delegate persistence.

    The orchestration layer builds a run outcome and hands it to
    :func:`mlcfd.pipeline.results.write_run_results`; the artifact filesystem
    contract (filenames, formats, ``save_modes`` / ``save_plots`` flags) lives in
    that writer, not here.
    """
    LOGGER.info("Starting pipeline for model %s", run.model_name)
    mesh = Mesh(run.mesh)
    pipeline = DataPipeline(mesh, run.data)
    matrix = read_matrix_csv(run.data.snapshot_csv_path())
    x_train, x_test = pipeline.run(matrix)

    if is_sweep_model(run.model_name):
        model = build_sweep_model(run)
        model.fit(x_train)
        _, errors = model.reconstruction_error(x_test)
        outcome: SweepOutcome | TrainableOutcome = SweepOutcome(
            model_name=run.model_name,
            errors=errors,
            r_max=model.params.r_max,
            r_step=model.params.r_step,
            modes=model.spatial_modes(),
        )
    elif is_trainable_model(run.model_name):
        trainable = build_trainable_model(run)
        trainable.fit(x_train)
        outcome = TrainableOutcome(
            model_name=run.model_name,
            metrics=trainable.evaluate(x_test),
        )
    else:
        msg = f"Unsupported model_name: {run.model_name}"
        raise ValueError(msg)

    write_run_results(outcome, run.output, mesh)
