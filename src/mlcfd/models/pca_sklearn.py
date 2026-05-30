"""PCA via scikit-learn with per-rank refits matching the thesis notebooks."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA

from mlcfd.config.schemas import SweepModelConfig
from mlcfd.logging_config import get_logger
from mlcfd.models.base import SweepableModel

LOGGER = get_logger("models")


class PCASklearnModel(SweepableModel):
    """Wrapper that refits ``sklearn.decomposition.PCA`` for every rank in the sweep."""

    def __init__(self, params: SweepModelConfig) -> None:
        """Attach sweep bounds."""
        super().__init__(params)
        self._x_train: NDArray[np.floating] | None = None

    def fit(self, x_train: NDArray[np.floating]) -> None:
        """Store the training matrix; PCA objects are refit inside the sweep."""
        self._x_train = np.asarray(x_train, dtype=np.float64)
        LOGGER.info(
            "Stored training matrix for PCA-sklearn sweeps with shape %s", self._x_train.shape
        )

    def _prepare_test(self, x_test: NDArray[np.floating]) -> NDArray[np.floating]:
        """Cast the test matrix to float64 for sklearn estimators."""
        return np.asarray(x_test, dtype=np.float64)

    def reconstruct(self, x_test: NDArray[np.floating], r: int) -> NDArray[np.floating]:
        """Refit PCA at rank ``r`` and round-trip ``x_test`` through it."""
        if self._x_train is None:
            msg = "Call fit() before reconstruction_error()"
            raise RuntimeError(msg)
        model = PCA(n_components=r)
        model.fit(self._x_train)
        return model.inverse_transform(model.transform(x_test))
