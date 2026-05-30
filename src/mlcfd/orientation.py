"""The one snapshot-orientation concept: is a snapshot a row or a column?

The pipeline's canonical snapshot matrix is ``(n_free, n_snapshots)`` --
*snapshots-as-columns*. Several consumers instead need each snapshot to be a
row/sample: the ``StandardScaler`` axis, the sklearn-backed manifold and Kernel
PCA models, and the spatial autoencoder all operate in a *snapshots-as-rows*
view. Before this module, that single question -- "is a snapshot a row or a
column?" -- was decided three different ways under three different names
(``spatial_reduction``, ``transpose_flag``, the autoencoder ``layout``), with the
``transpose_flag`` polarity inverted ("transpose when False").

This module states the convention once. A consumer declares the orientation it
needs as a :class:`SnapshotOrientation`, and :func:`orient` applies it at that
consumer's edge, so modules ask for an orientation rather than re-deriving
transposes.
"""

from __future__ import annotations

from enum import StrEnum

import numpy as np
from numpy.typing import NDArray


class SnapshotOrientation(StrEnum):
    """How a consumer lays out a snapshot matrix relative to the canonical one.

    The canonical pipeline matrix is ``(n_free, n_snapshots)``
    (:attr:`SNAPSHOTS_AS_COLUMNS`); :attr:`SNAPSHOTS_AS_ROWS` is its transpose,
    where each snapshot is a row/sample (the scikit-learn convention).
    """

    SNAPSHOTS_AS_ROWS = "snapshots_as_rows"
    SNAPSHOTS_AS_COLUMNS = "snapshots_as_columns"


def orient(
    matrix: NDArray[np.floating],
    orientation: SnapshotOrientation,
) -> NDArray[np.floating]:
    """Return the canonical (snapshots-as-columns) ``matrix`` in ``orientation``.

    :attr:`SnapshotOrientation.SNAPSHOTS_AS_COLUMNS` is the canonical layout and
    returns ``matrix`` unchanged; :attr:`SnapshotOrientation.SNAPSHOTS_AS_ROWS`
    returns its transpose so each snapshot becomes a row/sample. The mapping is its
    own inverse, so wrapping a round trip
    (``orient(op(orient(m, o)), o)``) restores the original orientation.
    """
    if orientation is SnapshotOrientation.SNAPSHOTS_AS_ROWS:
        return matrix.T
    return matrix
