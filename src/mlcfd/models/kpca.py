"""Kernel PCA with per-rank refits mirroring the thesis notebooks."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import KernelPCA

from mlcfd.config.schemas import KPCAModelConfig
from mlcfd.logging_config import get_logger
from mlcfd.models.base import SweepableModel
from mlcfd.orientation import orient

LOGGER = get_logger("models")


class KPCAModel(SweepableModel):
    """Kernel PCA sweeps that refit estimators for every retained dimension."""

    def __init__(self, params: KPCAModelConfig) -> None:
        """Attach kernel PCA hyperparameters."""
        super().__init__(params)
        self._x_train: NDArray[np.floating] | None = None

    def fit(self, x_train: NDArray[np.floating]) -> None:
        """Store the training matrix in the configured orientation for subsequent refits."""
        if not isinstance(self._params, KPCAModelConfig):
            msg = "KPCAModel requires KPCAModelConfig"
            raise TypeError(msg)
        self._x_train = np.asarray(
            orient(x_train, self._params.orientation),
            dtype=np.float64,
        )
        LOGGER.info("Stored KPCA training matrix with shape %s", self._x_train.shape)

    def _prepare_test(self, x_test: NDArray[np.floating]) -> NDArray[np.floating]:
        """Cast the test matrix to a float64 view in the configured orientation."""
        params = self._params
        if not isinstance(params, KPCAModelConfig):
            msg = "Internal configuration must remain KPCAModelConfig"
            raise TypeError(msg)
        return np.asarray(orient(x_test, params.orientation), dtype=np.float64)

    def reconstruct(self, x_test: NDArray[np.floating], r: int) -> NDArray[np.floating]:
        """Refit KernelPCA at rank ``r`` and inverse-transform ``x_test``."""
        if self._x_train is None:
            msg = "Call fit() before reconstruction_error()"
            raise RuntimeError(msg)
        params = self._params
        if not isinstance(params, KPCAModelConfig):
            msg = "Internal configuration must remain KPCAModelConfig"
            raise TypeError(msg)
        model = KernelPCA(
            n_components=r,
            kernel=params.kernel,
            fit_inverse_transform=True,
            gamma=params.gamma,
            degree=params.degree,
            alpha=params.alpha,
        )
        model.fit(self._x_train)
        return model.inverse_transform(model.transform(x_test))
