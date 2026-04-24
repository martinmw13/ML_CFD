"""Structured mesh representation for CFD snapshot grids."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from mlcfd.config.schemas import MeshConfig


class Mesh:
    """Grid metadata and a boolean mask for points inside the cylinder."""

    def __init__(self, config: MeshConfig) -> None:
        """Build mesh geometry and precompute the cylinder mask.

        Args:
            config: Validated mesh parameters (grid size, domain extents, cylinder).
        """
        self._config = config
        self._mask: NDArray[np.bool_] = self._compute_cylinder_mask()

    @property
    def config(self) -> MeshConfig:
        """Return the immutable mesh configuration."""
        return self._config

    @property
    def nx(self) -> int:
        """Number of nodes in the x direction."""
        return self._config.nx

    @property
    def ny(self) -> int:
        """Number of nodes in the y direction."""
        return self._config.ny

    @property
    def n_points(self) -> int:
        """Total number of grid points (nx * ny)."""
        return self._config.nx * self._config.ny

    @property
    def mask(self) -> NDArray[np.bool_]:
        """Boolean mask that is True inside the cylinder radius."""
        return self._mask

    def mask_cyl(self) -> NDArray[np.bool_]:
        """Return the cylinder mask (alias kept for notebook compatibility)."""
        return self._mask

    def _compute_cylinder_mask(self) -> NDArray[np.bool_]:
        """Compute the boolean mask for points whose distance to the cylinder center is <= r."""
        cfg = self._config
        indx = np.arange(0, cfg.nx * cfg.ny, dtype=np.int64)
        x = (indx % cfg.nx) * (cfg.lx / (cfg.nx - 1))
        y = (indx // cfg.nx) * (cfg.ly / (cfg.ny - 1))
        distances = np.sqrt((x - cfg.x0) ** 2 + (y - cfg.y0) ** 2)
        return np.asarray(distances <= cfg.r, dtype=np.bool_)
