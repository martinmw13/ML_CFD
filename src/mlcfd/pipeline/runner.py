"""High-level orchestration for CLI and notebook entry points."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from mlcfd.config.schemas import RunConfig
from mlcfd.io.storage import ensure_directory
from mlcfd.logging_config import get_logger
from mlcfd.mesh.mesh import Mesh
from mlcfd.models.factory import (
    build_sweep_model,
    build_trainable_model,
    is_sweep_model,
    is_trainable_model,
)
from mlcfd.preprocessing.pipeline import DataPipeline
from mlcfd.visualization.plots import plot_reconstruction_error

LOGGER = get_logger("pipeline")


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
    x_train, x_test = pipeline.run()
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
