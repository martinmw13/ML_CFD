"""Barycenter-based inverse maps used by LLE and Isomap notebooks."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.manifold._locally_linear import barycenter_kneighbors_graph

from mlcfd.models.base import sklearn_layout


def barycenter_reconstruction_curve(
    embedding,
    x_test: NDArray[np.floating],
    *,
    r_max: int,
    r_step: int,
    k_neighbors: int,
    reg: float,
    transpose_flag: bool,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Replicate notebook reconstruction sweeps using ``barycenter_kneighbors_graph``.

    Args:
        embedding: Fitted ``LocallyLinearEmbedding`` or ``Isomap`` estimator.
        x_test: Test matrix in the same layout as training before ``sklearn_layout``.
        r_max: Maximum latent dimension retained in the sweep.
        r_step: Increment between ranks.
        k_neighbors: Neighbor count for the barycenter graph.
        reg: Regularizer forwarded to ``barycenter_kneighbors_graph``.
        transpose_flag: Whether training data already matches sklearn layout.

    Returns:
        Tuple of the last reconstruction and the relative error vector.
    """
    x_test_use = sklearn_layout(x_test, transpose_flag)
    x_test_trans = embedding.transform(x_test_use)
    errors: list[float] = []
    last_recon = x_test_use
    for rank in range(1, r_max + 1, r_step):
        latent = x_test_trans[:, :rank]
        weights = barycenter_kneighbors_graph(
            latent,
            n_neighbors=k_neighbors,
            reg=reg,
        )
        recon = weights @ x_test_use
        last_recon = recon
        num = float(np.linalg.norm(x_test_use - recon, ord="fro"))
        den = float(np.linalg.norm(x_test_use, ord="fro"))
        errors.append(num / den if den > 0.0 else float("inf"))
    return last_recon, np.asarray(errors, dtype=np.float64)
