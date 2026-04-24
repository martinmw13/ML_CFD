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

    def reconstruction_error(
        self,
        x_test: NDArray[np.floating],
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Project ``x_test`` onto the first ``r`` left singular vectors for each sweep."""
        if self._u is None:
            msg = "Call fit() before reconstruction_error()"
            raise RuntimeError(msg)
        errors: list[float] = []
        last_recon = x_test
        for rank in range(1, self._params.r_max + 1, self._params.r_step):
            r = min(rank, self._u.shape[1])
            basis = self._u[:, :r]
            recon = basis @ (basis.T @ x_test)
            last_recon = recon
            num = float(np.linalg.norm(x_test - recon, ord="fro"))
            den = float(np.linalg.norm(x_test, ord="fro"))
            errors.append(num / den if den > 0.0 else float("inf"))
        LOGGER.debug("PCA-SVD sweep produced %s error samples", len(errors))
        return last_recon, np.asarray(errors, dtype=np.float64)
