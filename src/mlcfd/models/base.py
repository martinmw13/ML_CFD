"""Abstract bases for sweep-based and trainable reduction models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from mlcfd.config.schemas import (
    AutoencoderModelConfig,
    KPCAModelConfig,
    ManifoldModelConfig,
    SweepModelConfig,
)

SweepParams: TypeAlias = SweepModelConfig | ManifoldModelConfig | KPCAModelConfig


def sklearn_layout(matrix: NDArray[np.floating], transpose_flag: bool) -> NDArray[np.floating]:
    """Match notebook convention: transpose when ``transpose_flag`` is False."""
    if not transpose_flag:
        return matrix.T
    return matrix


class SweepableModel(ABC):
    """Models that evaluate reconstruction error while sweeping retained dimensions."""

    def __init__(self, params: SweepParams) -> None:
        """Store validated sweep parameters shared by PCA and manifold methods."""
        self._params = params

    @property
    def params(self) -> SweepParams:
        """Return the configuration bundle for this model."""
        return self._params

    @abstractmethod
    def fit(self, x_train: NDArray[np.floating]) -> None:
        """Fit internal estimators on the training matrix."""

    @abstractmethod
    def reconstruction_error(
        self,
        x_test: NDArray[np.floating],
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Return the last reconstruction and relative errors for each ``r`` sweep."""


class TrainableModel(ABC):
    """Models that require iterative optimization (for example, autoencoders)."""

    def __init__(self, params: AutoencoderModelConfig) -> None:
        """Store autoencoder training hyperparameters."""
        self._params = params

    @property
    def params(self) -> AutoencoderModelConfig:
        """Return autoencoder hyperparameters."""
        return self._params

    @abstractmethod
    def fit(self, x_train: NDArray[np.floating]) -> None:
        """Optimize model parameters."""

    @abstractmethod
    def evaluate(self, x_test: NDArray[np.floating]) -> dict[str, float]:
        """Return scalar metrics on the held-out set."""
