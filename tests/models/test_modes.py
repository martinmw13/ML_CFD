"""Spatial-modes accessor on the sweep model interface.

Driven from in-memory matrices only — no filesystem or CFD CSVs — anchoring the
``spatial_modes`` contract added to :class:`SweepableModel`.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlcfd.config.schemas import SweepModelConfig
from mlcfd.models.pca_sklearn import PCASklearnModel
from mlcfd.models.pca_svd import PCASVDModel


def test_pca_svd_spatial_modes_returns_basis_in_free_row_space() -> None:
    """A fitted PCA-SVD model returns left singular vectors shaped ``(n_free, n_modes)``."""
    n_free, n_train = 20, 8
    x_train = np.random.default_rng(0).standard_normal((n_free, n_train))

    model = PCASVDModel(SweepModelConfig(r_max=5, r_step=1))
    model.fit(x_train)
    modes = model.spatial_modes()

    assert modes is not None
    # The spatial dimension (rows) must match the free (cylinder-masked) row count.
    assert modes.shape[0] == n_free
    assert modes.shape == (n_free, min(n_free, n_train))


def test_pca_svd_spatial_modes_before_fit_raises() -> None:
    """Asking for modes before fitting raises rather than returning a stale basis."""
    model = PCASVDModel(SweepModelConfig())

    with pytest.raises(RuntimeError):
        model.spatial_modes()


def test_sweep_model_without_basis_returns_none() -> None:
    """A sweep model with no persistent modal basis signals 'no modes' as ``None``."""
    model = PCASklearnModel(SweepModelConfig())
    model.fit(np.random.default_rng(0).standard_normal((20, 8)))

    assert model.spatial_modes() is None
