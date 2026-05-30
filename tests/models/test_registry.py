"""The model registry is the single source of truth for the taxonomy.

These tests pin the invariants that let the registry replace the old scattered
name sets: every declared ``ModelName`` resolves through the registry to a
builder and run kind, each entry is wired to a model of the matching base type,
and config validation reads its name→param-type rule from the registry.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from mlcfd.config.schemas import (
    DataConfig,
    MeshConfig,
    OutputConfig,
    RunConfig,
    SweepModelConfig,
    model_name_literal_values,
)
from mlcfd.models.base import SweepableModel, TrainableModel
from mlcfd.models.registry import MODEL_REGISTRY


def test_registry_covers_exactly_the_declared_model_names() -> None:
    """Every ``ModelName`` literal has a registry entry, and the registry adds none."""
    assert set(MODEL_REGISTRY) == set(model_name_literal_values())


@pytest.mark.parametrize("name", sorted(MODEL_REGISTRY))
def test_every_model_resolves_to_a_builder_of_its_run_kind(name: str) -> None:
    """Each entry builds a concrete model whose base type matches its run kind.

    Builds from the spec's own ``param_type`` defaults, so a miswired entry —
    e.g. a ``"sweep"`` kind pointing at a trainable class — fails here.
    """
    spec = MODEL_REGISTRY[name]
    model = spec.builder(spec.param_type())
    expected_base = SweepableModel if spec.run_kind == "sweep" else TrainableModel
    assert isinstance(model, expected_base)


def _base_run_kwargs(tmp_path: Path) -> dict[str, object]:
    """Minimal mesh/data/output configs for building a ``RunConfig`` off disk."""
    return {
        "mesh": MeshConfig(nx=4, ny=4, lx=1.0, ly=1.0, x0=0.0, y0=0.0, r=0.1),
        "data": DataConfig(input_dir=tmp_path, reynolds_number=50),
        "output": OutputConfig(output_dir=tmp_path),
    }


def test_validator_accepts_registry_matched_params(tmp_path: Path) -> None:
    """A run whose params match the registry's ``param_type`` validates.

    Also confirms the validator's lazy registry import resolves (no import cycle).
    """
    run = RunConfig(
        **_base_run_kwargs(tmp_path),
        model_name="pca_svd",
        model_params=SweepModelConfig(),
    )
    assert run.model_name == "pca_svd"


def test_validator_rejects_registry_mismatched_params(tmp_path: Path) -> None:
    """Params that aren't the registry's ``param_type`` for the name are rejected."""
    with pytest.raises(ValidationError, match="KPCAModelConfig"):
        RunConfig(
            **_base_run_kwargs(tmp_path),
            model_name="kpca",
            model_params=SweepModelConfig(),
        )
