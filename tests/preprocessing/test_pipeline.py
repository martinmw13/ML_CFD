"""Smoke tests for pure, data-free preprocessing transforms.

These exercise real package code without touching the filesystem or any CFD
CSVs, anchoring the test harness so the architecture work that follows is
verifiable on its own.
"""

from __future__ import annotations

import numpy as np

from mlcfd.preprocessing.pipeline import subtract_mean


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
