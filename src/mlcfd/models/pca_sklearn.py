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
        LOGGER.info("Stored training matrix for PCA-sklearn sweeps with shape %s", self._x_train.shape)

    def reconstruction_error(
        self,
        x_test: NDArray[np.floating],
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Refit PCA for each ``r`` and measure relative Frobenius reconstruction error."""
        if self._x_train is None:
            msg = "Call fit() before reconstruction_error()"
            raise RuntimeError(msg)
        x_test_arr = np.asarray(x_test, dtype=np.float64)
        errors: list[float] = []
        last_recon = x_test_arr
        for rank in range(1, self._params.r_max + 1, self._params.r_step):
            model = PCA(n_components=rank)
            model.fit(self._x_train)
            latent = model.transform(x_test_arr)
            recon = model.inverse_transform(latent)
            last_recon = recon
            num = float(np.linalg.norm(x_test_arr - recon, ord="fro"))
            den = float(np.linalg.norm(x_test_arr, ord="fro"))
            errors.append(num / den if den > 0.0 else float("inf"))
        LOGGER.debug("PCA-sklearn sweep produced %s error samples", len(errors))
        return last_recon, np.asarray(errors, dtype=np.float64)
