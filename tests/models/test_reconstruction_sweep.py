"""The shared reconstruction-error sweep driver on :class:`SweepableModel`.

The rank iteration, the relative-Frobenius metric, and the ``den > 0`` guard live in
the base; subclasses supply only a per-rank ``reconstruct``. These tests pin the driver
in isolation against a closed-form synthetic model, anchor PCA-SVD numerically, and smoke
the three sklearn-backed models (KPCA, Isomap, LLE) whose orientation path applies
:func:`mlcfd.orientation.orient` -- a misplaced transpose there fails silently otherwise.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from mlcfd.config.schemas import KPCAModelConfig, ManifoldModelConfig, SweepModelConfig
from mlcfd.models.base import SweepableModel
from mlcfd.models.isomap import IsomapModel
from mlcfd.models.kpca import KPCAModel
from mlcfd.models.lle import LLEModel
from mlcfd.models.pca_svd import PCASVDModel


class _ScalingSweep(SweepableModel):
    """Rank-``r`` reconstruction is ``(r / r_max) * x``.

    With ``x̂ = c·x`` the relative-Frobenius error collapses to the closed form
    ``‖x − c·x‖_F / ‖x‖_F = |1 − c|``, letting the driver's output be predicted exactly.
    """

    def fit(self, x_train: NDArray[np.floating]) -> None:  # noqa: ARG002
        """No state to fit; the reconstruction is a pure function of ``r``."""
        return None

    def reconstruct(self, x_test: NDArray[np.floating], r: int) -> NDArray[np.floating]:
        """Scale the reference by ``r / r_max``."""
        return (r / self._params.r_max) * x_test


def test_driver_curve_matches_closed_form_relative_frobenius() -> None:
    """The driver produces one relative-Frobenius value per rank, matching the closed form."""
    x = np.random.default_rng(0).standard_normal((7, 4))
    model = _ScalingSweep(SweepModelConfig(r_max=6, r_step=2))
    model.fit(x)

    last_recon, errors = model.reconstruction_error(x)

    ranks = list(range(1, 6 + 1, 2))  # [1, 3, 5]
    assert errors.shape == (len(ranks),)
    expected = np.array([abs(1.0 - r / 6) for r in ranks])
    np.testing.assert_allclose(errors, expected)
    # The returned reconstruction is the final rank's output, not an intermediate one.
    np.testing.assert_allclose(last_recon, (ranks[-1] / 6) * x)


@pytest.mark.parametrize(("r_max", "r_step"), [(12, 1), (12, 3), (5, 2), (1, 1)])
def test_driver_curve_length_matches_rank_grid(r_max: int, r_step: int) -> None:
    """The error-curve length equals the number of ranks in ``1..r_max`` step ``r_step``."""
    x = np.random.default_rng(1).standard_normal((5, 3))
    model = _ScalingSweep(SweepModelConfig(r_max=r_max, r_step=r_step))
    model.fit(x)

    _, errors = model.reconstruction_error(x)

    assert errors.shape == (len(range(1, r_max + 1, r_step)),)


def test_driver_zero_reference_triggers_den_guard() -> None:
    """A zero reference matrix (``den == 0``) yields ``inf`` for every rank, not a divide error."""
    x = np.zeros((4, 3))
    model = _ScalingSweep(SweepModelConfig(r_max=3, r_step=1))
    model.fit(x)

    _, errors = model.reconstruction_error(x)

    assert errors.shape == (3,)
    assert np.isinf(errors).all()


def test_pca_svd_rides_driver_and_recovers_at_full_rank() -> None:
    """PCA-SVD inherits the driver: error is monotone non-increasing and ~0 at full rank."""
    n_free, n_train = 9, 5
    x = np.random.default_rng(0).standard_normal((n_free, n_train))
    model = PCASVDModel(SweepModelConfig(r_max=n_train, r_step=1))
    model.fit(x)

    last_recon, errors = model.reconstruction_error(x)

    assert errors.shape == (n_train,)
    assert last_recon.shape == x.shape
    # The column space is rank n_train, so the top rank reconstructs x exactly.
    assert errors[-1] < 1e-10
    np.testing.assert_array_less(np.diff(errors), 1e-12)


# --- sklearn-backed models: the layout path the refactor rewrote ------------------

_R_MAX, _R_STEP, _K = 3, 1, 5


def _manifold_matrices() -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Train/test matrices shaped ``(n_features, n_samples)``.

    With the default ``SNAPSHOTS_AS_ROWS`` orientation the models apply
    :func:`mlcfd.orientation.orient` (a transpose), so the *sample* axis becomes 15
    (train) / 8 (test) -- both above ``k_neighbors`` and the barycenter graph's
    ``k + 1`` requirement.
    """
    rng = np.random.default_rng(0)
    x_train = rng.standard_normal((6, 15))
    x_test = rng.standard_normal((6, 8))
    return x_train, x_test


def _assert_sweep_curve(last_recon: NDArray[np.floating], errors: NDArray[np.floating]) -> None:
    """Every sklearn-backed sweep returns a finite curve and a recon in sklearn layout."""
    assert errors.shape == (len(range(1, _R_MAX + 1, _R_STEP)),)
    assert np.isfinite(errors).all()
    # Error is measured in the SNAPSHOTS_AS_ROWS orientation, so the recon carries the
    # transposed test shape; this catches a double-applied or dropped orient().
    assert last_recon.shape == (8, 6)


def test_kpca_sweep_rides_base_driver() -> None:
    """KPCA's per-rank refit feeds the shared driver and yields a finite curve."""
    x_train, x_test = _manifold_matrices()
    model = KPCAModel(KPCAModelConfig(r_max=_R_MAX, r_step=_R_STEP))
    model.fit(x_train)

    _assert_sweep_curve(*model.reconstruction_error(x_test))


def test_isomap_sweep_rides_base_driver() -> None:
    """Isomap's barycenter reconstruction feeds the shared driver and yields a finite curve."""
    x_train, x_test = _manifold_matrices()
    model = IsomapModel(ManifoldModelConfig(r_max=_R_MAX, r_step=_R_STEP, k_neighbors=_K))
    model.fit(x_train)

    _assert_sweep_curve(*model.reconstruction_error(x_test))


def test_lle_sweep_rides_base_driver() -> None:
    """LLE's barycenter reconstruction feeds the shared driver and yields a finite curve."""
    x_train, x_test = _manifold_matrices()
    model = LLEModel(ManifoldModelConfig(r_max=_R_MAX, r_step=_R_STEP, k_neighbors=_K))
    model.fit(x_train)

    _assert_sweep_curve(*model.reconstruction_error(x_test))
