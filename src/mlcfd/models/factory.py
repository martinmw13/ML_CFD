"""Factory helpers mapping ``RunConfig`` to concrete model implementations."""

from __future__ import annotations

from mlcfd.config.schemas import (
    AutoencoderModelConfig,
    KPCAModelConfig,
    ManifoldModelConfig,
    RunConfig,
    SweepModelConfig,
)
from mlcfd.models.autoencoder import LinearAutoencoder
from mlcfd.models.base import SweepableModel, TrainableModel
from mlcfd.models.isomap import IsomapModel
from mlcfd.models.kpca import KPCAModel
from mlcfd.models.lle import LLEModel
from mlcfd.models.pca_sklearn import PCASklearnModel
from mlcfd.models.pca_svd import PCASVDModel


def build_sweep_model(run: RunConfig) -> SweepableModel:
    """Instantiate a sweepable model implementation from ``run``."""
    name = run.model_name
    params = run.model_params
    if name == "pca_svd":
        if not isinstance(params, SweepModelConfig):
            msg = "pca_svd requires SweepModelConfig parameters"
            raise TypeError(msg)
        return PCASVDModel(params)
    if name == "pca_sklearn":
        if not isinstance(params, SweepModelConfig):
            msg = "pca_sklearn requires SweepModelConfig parameters"
            raise TypeError(msg)
        return PCASklearnModel(params)
    if name == "lle":
        if not isinstance(params, ManifoldModelConfig):
            msg = "lle requires ManifoldModelConfig parameters"
            raise TypeError(msg)
        return LLEModel(params)
    if name == "isomap":
        if not isinstance(params, ManifoldModelConfig):
            msg = "isomap requires ManifoldModelConfig parameters"
            raise TypeError(msg)
        return IsomapModel(params)
    if name == "kpca":
        if not isinstance(params, KPCAModelConfig):
            msg = "kpca requires KPCAModelConfig parameters"
            raise TypeError(msg)
        return KPCAModel(params)
    msg = f"Model {name!r} is not a sweepable implementation"
    raise ValueError(msg)


def build_trainable_model(run: RunConfig) -> TrainableModel:
    """Instantiate a trainable model (autoencoder) from ``run``."""
    name = run.model_name
    params = run.model_params
    if not isinstance(params, AutoencoderModelConfig):
        msg = f"{name} requires AutoencoderModelConfig parameters"
        raise TypeError(msg)
    if name == "ae_spatial":
        return LinearAutoencoder(params, layout="spatial")
    if name == "ae_temporal":
        return LinearAutoencoder(params, layout="temporal")
    msg = f"Model {name!r} is not a trainable implementation"
    raise ValueError(msg)


def is_sweep_model(name: str) -> bool:
    """Return True if the model identifier uses sweepable reconstruction."""
    return name in {"pca_svd", "pca_sklearn", "lle", "isomap", "kpca"}


def is_trainable_model(name: str) -> bool:
    """Return True if the model identifier uses iterative training."""
    return name in {"ae_spatial", "ae_temporal"}
