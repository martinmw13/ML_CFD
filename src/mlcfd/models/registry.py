"""Single source of truth for the model taxonomy.

Each model identifier maps to one :class:`ModelSpec` describing everything the
rest of the package needs to know about it: which parameter schema it accepts,
whether it runs as a dimension *sweep* or a *trainable* optimisation, and how to
build the concrete implementation. Config validation
(:meth:`mlcfd.config.schemas.RunConfig.model_name_matches_params`), construction,
and run dispatch (:mod:`mlcfd.pipeline.runner`) all read from this registry, so
the taxonomy lives in exactly one place.

Adding a model is a single edit here: register one ``MODEL_REGISTRY`` entry
naming its parameter schema, run kind, and builder. The discriminated-union
parameter typing (``profile`` discriminator on :data:`~mlcfd.config.schemas.ModelParams`)
already guarantees the params handed to a builder match its declared
``param_type``, so builders trust their argument instead of re-checking it.

For example, a hypothetical sweep model ``"pca_randomized"`` reusing
``SweepModelConfig`` would be wired up with a single line::

    "pca_randomized": ModelSpec(SweepModelConfig, "sweep", PCARandomizedModel),

with no edits to the validator, the runner, or any name set.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, TypeAlias

from pydantic import BaseModel

from mlcfd.config.schemas import (
    AutoencoderModelConfig,
    KPCAModelConfig,
    ManifoldModelConfig,
    ModelParams,
    SweepModelConfig,
)
from mlcfd.models.autoencoder import LinearAutoencoder
from mlcfd.models.base import SweepableModel, TrainableModel
from mlcfd.models.isomap import IsomapModel
from mlcfd.models.kpca import KPCAModel
from mlcfd.models.lle import LLEModel
from mlcfd.models.pca_sklearn import PCASklearnModel
from mlcfd.models.pca_svd import PCASVDModel
from mlcfd.orientation import SnapshotOrientation

RunKind: TypeAlias = Literal["sweep", "trainable"]
"""How a model is exercised: a dimension sweep, or an iterative training run."""

ModelBuilder: TypeAlias = Callable[[ModelParams], SweepableModel | TrainableModel]
"""Builds a concrete model from its validated parameter bundle.

The bundle is guaranteed (by the discriminated union plus the ``RunConfig``
validator) to be the concrete schema named in the spec's ``param_type``, so
builders may assume their argument's type without re-checking it.
"""


@dataclass(frozen=True)
class ModelSpec:
    """One model's place in the taxonomy.

    Attributes:
        param_type: Concrete parameter schema this model accepts; the config
            validator checks the run's ``model_params`` against it.
        run_kind: ``"sweep"`` for reconstruction-error sweeps, ``"trainable"``
            for iteratively optimised models. Drives run dispatch.
        builder: Callable turning the validated params into a concrete model.
    """

    param_type: type[BaseModel]
    run_kind: RunKind
    builder: ModelBuilder


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "pca_svd": ModelSpec(SweepModelConfig, "sweep", PCASVDModel),
    "pca_sklearn": ModelSpec(SweepModelConfig, "sweep", PCASklearnModel),
    "lle": ModelSpec(ManifoldModelConfig, "sweep", LLEModel),
    "isomap": ModelSpec(ManifoldModelConfig, "sweep", IsomapModel),
    "kpca": ModelSpec(KPCAModelConfig, "sweep", KPCAModel),
    "ae_spatial": ModelSpec(
        AutoencoderModelConfig,
        "trainable",
        lambda params: LinearAutoencoder(params, orientation=SnapshotOrientation.SNAPSHOTS_AS_ROWS),
    ),
    "ae_temporal": ModelSpec(
        AutoencoderModelConfig,
        "trainable",
        lambda params: LinearAutoencoder(
            params, orientation=SnapshotOrientation.SNAPSHOTS_AS_COLUMNS
        ),
    ),
}
"""Maps each model identifier to its :class:`ModelSpec` — the only taxonomy copy."""
