"""Parity guard for the snapshot-orientation refactor (issue #7).

Three mechanisms once decided "is a snapshot a row or a column?" independently:
``spatial_reduction`` (scaler), ``transpose_flag``/``sklearn_layout`` (manifold +
KPCA), and the autoencoder ``layout``. They were replaced by one
:class:`~mlcfd.orientation.SnapshotOrientation` applied through
:func:`~mlcfd.orientation.orient`. These transpose decisions are load-bearing for
parity with the thesis notebooks, so this module pins the new primitive against the
*exact legacy formulas* and drives the real :class:`~mlcfd.preprocessing.pipeline.DataPipeline`
scaler to show outputs are byte-for-byte unchanged on a synthetic case.

The legacy formulas are inlined (not read from a captured golden) so the check is
self-contained and cannot drift or go circular: if ``orient`` reproduces every old
transpose decision, then every downstream consumer -- whose only behavioural change
was swapping the transpose primitive -- feeds identical arrays to deterministic
sklearn/torch code and necessarily produces identical outputs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from sklearn.preprocessing import StandardScaler

from mlcfd.config.schemas import DataConfig, MeshConfig
from mlcfd.mesh.mesh import Mesh
from mlcfd.orientation import SnapshotOrientation, orient
from mlcfd.preprocessing.pipeline import DataPipeline


def _legacy_sklearn_layout(
    matrix: NDArray[np.floating], *, transpose_flag: bool
) -> NDArray[np.floating]:
    """The deleted ``base.sklearn_layout``: transpose when ``transpose_flag`` is False."""
    return matrix if transpose_flag else matrix.T


def _legacy_ae_layout(
    matrix: NDArray[np.floating], *, layout: str
) -> NDArray[np.floating]:
    """The deleted autoencoder rule: transpose for ``"spatial"``, identity for ``"temporal"``."""
    return matrix.T if layout == "spatial" else matrix


@pytest.fixture
def matrix() -> NDArray[np.floating]:
    """A non-square synthetic snapshot matrix so a stray transpose changes the shape."""
    return np.random.default_rng(0).standard_normal((6, 4))


# --- orient() reproduces every legacy transpose decision -------------------------


def test_orient_rows_is_transpose_columns_is_identity(matrix: NDArray[np.floating]) -> None:
    """SNAPSHOTS_AS_ROWS transposes the canonical matrix; SNAPSHOTS_AS_COLUMNS is identity."""
    np.testing.assert_array_equal(orient(matrix, SnapshotOrientation.SNAPSHOTS_AS_ROWS), matrix.T)
    np.testing.assert_array_equal(
        orient(matrix, SnapshotOrientation.SNAPSHOTS_AS_COLUMNS), matrix
    )


@pytest.mark.parametrize(
    ("orientation", "transpose_flag"),
    [
        # Old defaults map to SNAPSHOTS_AS_ROWS: sklearn_layout transposed when the flag
        # was False, so ROWS (transpose) is its False branch and COLUMNS its True branch.
        (SnapshotOrientation.SNAPSHOTS_AS_ROWS, False),
        (SnapshotOrientation.SNAPSHOTS_AS_COLUMNS, True),
    ],
)
def test_orient_matches_legacy_sklearn_layout(
    matrix: NDArray[np.floating],
    orientation: SnapshotOrientation,
    transpose_flag: bool,
) -> None:
    """The manifold/KPCA edge: orient(.) equals the old ``sklearn_layout(., transpose_flag)``."""
    np.testing.assert_array_equal(
        orient(matrix, orientation),
        _legacy_sklearn_layout(matrix, transpose_flag=transpose_flag),
    )


@pytest.mark.parametrize(
    ("orientation", "layout"),
    [
        (SnapshotOrientation.SNAPSHOTS_AS_ROWS, "spatial"),
        (SnapshotOrientation.SNAPSHOTS_AS_COLUMNS, "temporal"),
    ],
)
def test_orient_matches_legacy_autoencoder_layout(
    matrix: NDArray[np.floating],
    orientation: SnapshotOrientation,
    layout: str,
) -> None:
    """The autoencoder edge: ae_spatial -> ROWS, ae_temporal -> COLUMNS, same transpose."""
    np.testing.assert_array_equal(
        orient(matrix, orientation),
        _legacy_ae_layout(matrix, layout=layout),
    )


@pytest.mark.parametrize(
    "orientation",
    [SnapshotOrientation.SNAPSHOTS_AS_ROWS, SnapshotOrientation.SNAPSHOTS_AS_COLUMNS],
)
def test_orient_is_its_own_inverse(
    matrix: NDArray[np.floating], orientation: SnapshotOrientation
) -> None:
    """orient(orient(m)) restores the original, so a round-tripped consumer is layout-safe."""
    np.testing.assert_array_equal(orient(orient(matrix, orientation), orientation), matrix)


# --- the real DataPipeline scaler matches the old spatial_reduction formula ------


def _pipeline(orientation: SnapshotOrientation) -> DataPipeline:
    """A pipeline whose scaler obeys ``orientation``; the mesh is unused by scaling."""
    config = DataConfig(
        input_dir=Path("unused"),
        reynolds_number=50,
        orientation=orientation,
        random_seed=0,
    )
    mesh = Mesh(MeshConfig(nx=5, ny=4, lx=4.0, ly=3.0, x0=2.0, y0=1.5, r=0.6))
    return DataPipeline(mesh, config)


@pytest.mark.parametrize(
    ("orientation", "spatial_reduction"),
    [
        # Old DataConfig.spatial_reduction default was True -> SNAPSHOTS_AS_ROWS.
        (SnapshotOrientation.SNAPSHOTS_AS_ROWS, True),
        (SnapshotOrientation.SNAPSHOTS_AS_COLUMNS, False),
    ],
)
def test_fit_scaler_matches_legacy_spatial_reduction(
    matrix: NDArray[np.floating],
    orientation: SnapshotOrientation,
    spatial_reduction: bool,
) -> None:
    """fit_scaler reproduces the old ``fit_transform(x.T).T`` (True) / ``fit_transform(x)`` (False)."""
    pipeline = _pipeline(orientation)

    scaled = pipeline.fit_scaler(matrix)

    legacy_scaler = StandardScaler()
    expected = (
        legacy_scaler.fit_transform(matrix.T).T
        if spatial_reduction
        else legacy_scaler.fit_transform(matrix)
    )
    np.testing.assert_allclose(scaled, expected)


@pytest.mark.parametrize(
    ("orientation", "spatial_reduction"),
    [
        (SnapshotOrientation.SNAPSHOTS_AS_ROWS, True),
        (SnapshotOrientation.SNAPSHOTS_AS_COLUMNS, False),
    ],
)
def test_transform_matches_legacy_spatial_reduction(
    matrix: NDArray[np.floating],
    orientation: SnapshotOrientation,
    spatial_reduction: bool,
) -> None:
    """transform() applies the fitted scaler under the same orientation as the old flag."""
    pipeline = _pipeline(orientation)
    other = np.random.default_rng(1).standard_normal(matrix.shape)

    pipeline.fit_scaler(matrix)
    transformed = pipeline.transform(other)

    legacy_scaler = StandardScaler()
    if spatial_reduction:
        legacy_scaler.fit(matrix.T)
        expected = legacy_scaler.transform(other.T).T
    else:
        legacy_scaler.fit(matrix)
        expected = legacy_scaler.transform(other)
    np.testing.assert_allclose(transformed, expected)
