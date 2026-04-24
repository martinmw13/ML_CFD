"""Kernel PCA with per-rank refits mirroring the thesis notebooks."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import KernelPCA

from mlcfd.config.schemas import KPCAModelConfig
from mlcfd.logging_config import get_logger
from mlcfd.models.base import SweepableModel, sklearn_layout

LOGGER = get_logger("models")


class KPCAModel(SweepableModel):
    """Kernel PCA sweeps that refit estimators for every retained dimension."""

    def __init__(self, params: KPCAModelConfig) -> None:
        """Attach kernel PCA hyperparameters."""
        super().__init__(params)
        self._x_train: NDArray[np.floating] | None = None

    def fit(self, x_train: NDArray[np.floating]) -> None:
        """Store the training matrix in sklearn layout for subsequent refits."""
        if not isinstance(self._params, KPCAModelConfig):
            msg = "KPCAModel requires KPCAModelConfig"
            raise TypeError(msg)
        self._x_train = np.asarray(
            sklearn_layout(x_train, self._params.transpose_flag),
            dtype=np.float64,
        )
        LOGGER.info("Stored KPCA training matrix with shape %s", self._x_train.shape)

    def reconstruction_error(
        self,
        x_test: NDArray[np.floating],
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Refit KernelPCA for each ``r`` and compute inverse-transform reconstructions."""
        if self._x_train is None:
            msg = "Call fit() before reconstruction_error()"
            raise RuntimeError(msg)
        params = self._params
        if not isinstance(params, KPCAModelConfig):
            msg = "Internal configuration must remain KPCAModelConfig"
            raise TypeError(msg)
        x_test_use = np.asarray(
            sklearn_layout(x_test, params.transpose_flag),
            dtype=np.float64,
        )
        errors: list[float] = []
        last_recon = x_test_use
        for rank in range(1, params.r_max + 1, params.r_step):
            model = KernelPCA(
                n_components=rank,
                kernel=params.kernel,
                fit_inverse_transform=True,
                gamma=params.gamma,
                degree=params.degree,
                alpha=params.alpha,
            )
            model.fit(self._x_train)
            latent = model.transform(x_test_use)
            recon = model.inverse_transform(latent)
            last_recon = recon
            num = float(np.linalg.norm(x_test_use - recon, ord="fro"))
            den = float(np.linalg.norm(x_test_use, ord="fro"))
            errors.append(num / den if den > 0.0 else float("inf"))
        LOGGER.debug("KPCA sweep produced %s error samples", len(errors))
        return last_recon, np.asarray(errors, dtype=np.float64)
