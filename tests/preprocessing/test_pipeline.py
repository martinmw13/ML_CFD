"""Tests for the data-free preprocessing flow.

These exercise real package code without touching the filesystem or any CFD
CSVs. The ``DataPipeline.run`` cases drive the full split -> cylinder mask ->
scale flow on an in-memory matrix, the seam where the spatial-reduction
transpose wiring is most likely to break.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mlcfd.config.schemas import DataConfig, MeshConfig
from mlcfd.mesh.mesh import Mesh
from mlcfd.preprocessing.pipeline import DataPipeline, subtract_mean


def _build_mesh() -> Mesh:
    """Small 5x4 grid whose cylinder mask erases some -- but not all -- rows."""
    return Mesh(MeshConfig(nx=5, ny=4, lx=4.0, ly=3.0, x0=2.0, y0=1.5, r=0.6))


def _build_data_config(*, spatial_reduction: bool) -> DataConfig:
    """Data config with placeholder IO fields; the matrix is supplied in memory."""
    return DataConfig(
        input_dir=Path("unused"),
        reynolds_number=50,
        test_size=0.5,
        spatial_reduction=spatial_reduction,
        random_seed=0,
    )


def test_subtract_mean_centers_along_default_axis() -> None:
    """Centering along axis 1 yields a zero per-row mean and the original mean."""
    data = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])

    centered, mean = subtract_mean(data)

    # Row means of the centered array collapse to zero.
    np.testing.assert_allclose(centered.mean(axis=1), np.zeros(2), atol=1e-12)
    # The returned mean is keepdims-shaped and matches the input row means.
    assert mean.shape == (2, 1)
    np.testing.assert_allclose(mean, np.array([[2.0], [20.0]]))


def test_subtract_mean_does_not_mutate_input() -> None:
    """The transform returns a new array and leaves the caller's data untouched."""
    data = np.array([[1.0, 2.0, 3.0]])
    original = data.copy()

    centered, _ = subtract_mean(data)

    np.testing.assert_array_equal(data, original)
    assert centered is not data


@pytest.mark.parametrize("spatial_reduction", [True, False])
def test_run_prepares_split_mask_and_scaling_in_memory(spatial_reduction: bool) -> None:
    """run() splits, masks, and scales a caller-supplied matrix with no file on disk."""
    mesh = _build_mesh()
    pipeline = DataPipeline(mesh, _build_data_config(spatial_reduction=spatial_reduction))

    n_points = mesh.n_points
    n_snapshots = 6
    n_free = int(np.count_nonzero(~mesh.mask))
    # Guard the fixture: the mask must drop some rows but not all, or the
    # cylinder step is a silent no-op and goes untested.
    assert 0 < n_free < n_points

    rng = np.random.default_rng(0)
    matrix = rng.standard_normal((n_points, n_snapshots))

    x_train_s, x_test_s = pipeline.run(matrix)

    # Cylinder rows removed; columns split 50/50 across the snapshot axis.
    n_train = n_snapshots // 2
    assert x_train_s.shape == (n_free, n_train)
    assert x_test_s.shape == (n_free, n_snapshots - n_train)
    assert x_train_s.shape[1] + x_test_s.shape[1] == n_snapshots

    # Standardization pins the config -> orientation wiring: a self-consistent
    # transpose bug would round-trip fine, so assert the scaled train stats
    # along the axis the spatial_reduction flag selects (ddof=0, as sklearn).
    scaled_axis = 1 if spatial_reduction else 0
    np.testing.assert_allclose(x_train_s.mean(axis=scaled_axis), 0.0, atol=1e-9)
    np.testing.assert_allclose(x_train_s.std(axis=scaled_axis), 1.0, atol=1e-9)

    # Scaling round-trip through the public inverse recovers the masked inputs.
    x_train, x_test = pipeline.train_test_split_no_shuffle(matrix)
    x_train_f = pipeline.erase_cylinder(x_train)
    x_test_f = pipeline.erase_cylinder(x_test)
    np.testing.assert_allclose(pipeline.inverse_transform(x_train_s), x_train_f, atol=1e-9)
    np.testing.assert_allclose(pipeline.inverse_transform(x_test_s), x_test_f, atol=1e-9)
