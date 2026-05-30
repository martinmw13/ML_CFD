"""Isomap embedding with notebook-compatible inverse maps."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.manifold import Isomap

from mlcfd.config.schemas import ManifoldModelConfig
from mlcfd.logging_config import get_logger
from mlcfd.models.base import SweepableModel
from mlcfd.models.manifold_inverse import barycenter_reconstruct
from mlcfd.orientation import orient

LOGGER = get_logger("models")


class IsomapModel(SweepableModel):
    """Isomap embedding plus barycenter reconstruction sweeps."""

    def __init__(self, params: ManifoldModelConfig) -> None:
        """Attach manifold hyperparameters."""
        super().__init__(params)
        self._embedding: Isomap | None = None

    def fit(self, x_train: NDArray[np.floating]) -> None:
        """Fit sklearn Isomap on the training matrix."""
        if not isinstance(self._params, ManifoldModelConfig):
            msg = "IsomapModel requires ManifoldModelConfig"
            raise TypeError(msg)
        x_use = orient(x_train, self._params.orientation)
        self._embedding = Isomap(
            n_components=self._params.r_max,
            n_neighbors=self._params.k_neighbors,
        ).fit(x_use)
        LOGGER.info("Fitted Isomap embedding with shape %s", self._embedding.embedding_.shape)

    def _prepare_test(self, x_test: NDArray[np.floating]) -> NDArray[np.floating]:
        """Put the test matrix in the orientation the embedding was fitted on."""
        params = self._params
        if not isinstance(params, ManifoldModelConfig):
            msg = "Internal configuration must remain ManifoldModelConfig"
            raise TypeError(msg)
        return orient(x_test, params.orientation)

    def reconstruct(self, x_test: NDArray[np.floating], r: int) -> NDArray[np.floating]:
        """Reconstruct ``x_test`` from its rank-``r`` Isomap embedding via barycenter weights."""
        if self._embedding is None:
            msg = "Call fit() before reconstruction_error()"
            raise RuntimeError(msg)
        params = self._params
        if not isinstance(params, ManifoldModelConfig):
            msg = "Internal configuration must remain ManifoldModelConfig"
            raise TypeError(msg)
        return barycenter_reconstruct(
            self._embedding,
            x_test,
            r,
            k_neighbors=params.k_neighbors,
            reg=params.reg,
        )
