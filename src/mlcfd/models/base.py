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
from mlcfd.logging_config import get_logger

LOGGER = get_logger("models")

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

    def reconstruction_error(
        self,
        x_test: NDArray[np.floating],
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Return the last reconstruction and relative errors for each ``r`` sweep.

        Owns the rank iteration and the relative-Frobenius metric for every sweep model.
        Subclasses supply only the per-rank :meth:`reconstruct` (and, when the test matrix
        needs a layout or dtype adjustment, :meth:`_prepare_test`); the error is always
        measured against the prepared matrix that ``reconstruct`` operates on.
        """
        x_ref = self._prepare_test(x_test)
        errors: list[float] = []
        last_recon = x_ref
        for r in self._sweep_ranks():
            recon = self.reconstruct(x_ref, r)
            last_recon = recon
            errors.append(self._relative_frobenius_error(x_ref, recon))
        LOGGER.debug("%s sweep produced %s error samples", type(self).__name__, len(errors))
        return last_recon, np.asarray(errors, dtype=np.float64)

    @abstractmethod
    def reconstruct(self, x_test: NDArray[np.floating], r: int) -> NDArray[np.floating]:
        """Return the rank-``r`` reconstruction of the prepared test matrix ``x_test``."""

    def _prepare_test(self, x_test: NDArray[np.floating]) -> NDArray[np.floating]:
        """Return the layout/dtype-adjusted test matrix used by the sweep.

        The returned matrix is both the input handed to :meth:`reconstruct` and the
        reference the relative-Frobenius metric is measured against, so per-model layout
        differences (raw matrix vs :func:`sklearn_layout`, float64 views) live here. The
        default is the identity; models operating in sklearn layout override it.
        """
        return x_test

    def _sweep_ranks(self) -> range:
        """Yield the retained-dimension grid (``1..r_max`` step ``r_step``) shared by all sweeps."""
        return range(1, self._params.r_max + 1, self._params.r_step)

    @staticmethod
    def _relative_frobenius_error(
        reference: NDArray[np.floating],
        reconstruction: NDArray[np.floating],
    ) -> float:
        """Relative Frobenius error ``‖x - x̂‖_F / ‖x‖_F`` with a zero-norm guard."""
        num = float(np.linalg.norm(reference - reconstruction, ord="fro"))
        den = float(np.linalg.norm(reference, ord="fro"))
        return num / den if den > 0.0 else float("inf")

    def spatial_modes(self) -> NDArray[np.floating] | None:
        """Return the spatial mode basis for a fitted model, or ``None`` if unavailable.

        When present, the basis has shape ``(n_free, n_modes)`` — one column per spatial
        mode living in the cylinder-masked row space, ready to hand to
        :func:`mlcfd.visualization.plots.save_modes`. Callers typically export only the
        leading modes (``min(params.r_max, n_modes)`` columns).

        The default returns ``None`` so that models without a meaningful modal basis
        (manifold embeddings, per-rank refit PCA, kernel PCA) signal "no modes" and let
        callers skip mode export predictably. Subclasses that hold a basis override this.
        """
        return None


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
