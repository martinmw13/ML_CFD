"""Locally linear embedding with notebook-compatible inverse maps."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.manifold import LocallyLinearEmbedding

from mlcfd.config.schemas import ManifoldModelConfig
from mlcfd.logging_config import get_logger
from mlcfd.models.base import SweepableModel, sklearn_layout
from mlcfd.models.manifold_inverse import barycenter_reconstruction_curve

LOGGER = get_logger("models")


class LLEModel(SweepableModel):
    """LLE embedding plus barycenter reconstruction sweeps."""

    def __init__(self, params: ManifoldModelConfig) -> None:
        """Attach manifold hyperparameters."""
        super().__init__(params)
        self._embedding: LocallyLinearEmbedding | None = None

    def fit(self, x_train: NDArray[np.floating]) -> None:
        """Fit sklearn LLE on the training matrix."""
        if not isinstance(self._params, ManifoldModelConfig):
            msg = "LLEModel requires ManifoldModelConfig"
            raise TypeError(msg)
        x_use = sklearn_layout(x_train, self._params.transpose_flag)
        self._embedding = LocallyLinearEmbedding(
            n_components=self._params.r_max,
            n_neighbors=self._params.k_neighbors,
            reg=self._params.reg,
            random_state=42,
        ).fit(x_use)
        LOGGER.info("Fitted LLE embedding with shape %s", self._embedding.embedding_.shape)

    def reconstruction_error(
        self,
        x_test: NDArray[np.floating],
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Run the barycenter inverse sweep on the test matrix."""
        if self._embedding is None:
            msg = "Call fit() before reconstruction_error()"
            raise RuntimeError(msg)
        params = self._params
        if not isinstance(params, ManifoldModelConfig):
            msg = "Internal configuration must remain ManifoldModelConfig"
            raise TypeError(msg)
        return barycenter_reconstruction_curve(
            self._embedding,
            x_test,
            r_max=params.r_max,
            r_step=params.r_step,
            k_neighbors=params.k_neighbors,
            reg=params.reg,
            transpose_flag=params.transpose_flag,
        )
