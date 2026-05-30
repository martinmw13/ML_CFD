"""PCA via truncated left singular vectors (numpy SVD)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from mlcfd.config.schemas import SweepModelConfig
from mlcfd.logging_config import get_logger
from mlcfd.models.base import SweepableModel

LOGGER = get_logger("models")


class PCASVDModel(SweepableModel):
    """Left singular vectors from ``numpy.linalg.svd`` with column-space projection."""

    def __init__(self, params: SweepModelConfig) -> None:
        """Attach sweep bounds for reconstruction evaluation."""
        super().__init__(params)
        self._u: NDArray[np.floating] | None = None

    def fit(self, x_train: NDArray[np.floating]) -> None:
        """Compute the economy SVD of the training matrix."""
        u, _s, _vh = np.linalg.svd(x_train, full_matrices=False)
        self._u = u
        LOGGER.info("Fitted PCA-SVD on training matrix with shape %s", x_train.shape)

    def spatial_modes(self) -> NDArray[np.floating]:
        """Return the left singular vectors as the spatial mode basis ``(n_free, n_modes)``."""
        if self._u is None:
            msg = "Call fit() before spatial_modes()"
            raise RuntimeError(msg)
        return self._u

    def reconstruct(self, x_test: NDArray[np.floating], r: int) -> NDArray[np.floating]:
        """Project ``x_test`` onto the first ``r`` left singular vectors."""
        if self._u is None:
            msg = "Call fit() before reconstruction_error()"
            raise RuntimeError(msg)
        basis = self._u[:, : min(r, self._u.shape[1])]
        return basis @ (basis.T @ x_test)
