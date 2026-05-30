"""Barycenter-based inverse maps used by LLE and Isomap notebooks."""

from __future__ import annotations

from numpy.typing import NDArray
from sklearn.manifold._locally_linear import barycenter_kneighbors_graph


def barycenter_reconstruct(
    embedding,
    x_test: NDArray,
    r: int,
    *,
    k_neighbors: int,
    reg: float,
) -> NDArray:
    """Reconstruct ``x_test`` from its rank-``r`` manifold embedding via barycenter weights.

    The shared per-rank step behind the LLE and Isomap sweeps. The rank iteration and the
    relative-Frobenius metric live in :meth:`SweepableModel.reconstruction_error`; this
    helper only rebuilds one barycenter graph and applies it.

    Args:
        embedding: Fitted ``LocallyLinearEmbedding`` or ``Isomap`` estimator.
        x_test: Test matrix already in sklearn layout (the embedding's training layout).
        r: Latent dimensions retained before rebuilding the barycenter graph.
        k_neighbors: Neighbor count for the barycenter graph.
        reg: Regularizer forwarded to ``barycenter_kneighbors_graph``.

    Returns:
        The rank-``r`` reconstruction, in the same layout as ``x_test``.
    """
    latent = embedding.transform(x_test)[:, :r]
    weights = barycenter_kneighbors_graph(latent, n_neighbors=k_neighbors, reg=reg)
    return weights @ x_test
