"""Training data preparation: load, split, mask cylinder, and scale."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from mlcfd.config.schemas import DataConfig
from mlcfd.io.storage import read_matrix_csv
from mlcfd.logging_config import get_logger
from mlcfd.mesh.mesh import Mesh

LOGGER = get_logger("preprocessing")


def reconstruct_field(
    u_filtered: NDArray[np.floating],
    mesh: Mesh,
) -> NDArray[np.floating]:
    """Insert cylinder-stripped rows back into a full grid layout.

    Args:
        u_filtered: Values for rows outside the cylinder, shape ``(n_free, n_cols)``.
        mesh: Mesh providing ``mask`` and ``n_points``.

    Returns:
        Full grid array of shape ``(mesh.n_points, n_cols)`` with cylinder rows zero-filled.

    Raises:
        ValueError: If the filtered row count does not match ``~mesh.mask``.
    """
    if u_filtered.ndim != 2:
        msg = f"Expected 2D filtered field, got shape {u_filtered.shape}"
        raise ValueError(msg)
    n_free = int(np.count_nonzero(~mesh.mask))
    if u_filtered.shape[0] != n_free:
        msg = f"Filtered rows {u_filtered.shape[0]} != free rows {n_free}"
        raise ValueError(msg)
    full = np.zeros((mesh.n_points, u_filtered.shape[1]), dtype=u_filtered.dtype)
    full[~mesh.mask, :] = u_filtered
    return full


def erase_cylinder_rows(
    matrix: NDArray[np.floating],
    mesh: Mesh,
) -> NDArray[np.floating]:
    """Remove matrix rows that lie inside the cylinder mask (thesis `erase_cyl`)."""
    return matrix[~mesh.mask, :]


def subtract_mean(
    data: NDArray[np.floating],
    axis: int = 1,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Subtract the mean along an axis without mutating the input.

    Args:
        data: Input array.
        axis: Axis used by ``numpy.mean``.

    Returns:
        Tuple of ``(centered, mean)`` where ``centered`` is a new array.
    """
    mean = np.mean(data, axis=axis, keepdims=True)
    return data - mean, mean


class DataPipeline:
    """Load CFD snapshots, split, remove cylinder nodes, and standardize."""

    def __init__(self, mesh: Mesh, data_config: DataConfig) -> None:
        """Attach mesh geometry and validated data settings.

        Args:
            mesh: Mesh used for cylinder masking.
            data_config: Data paths and scaling options.
        """
        self._mesh = mesh
        self._config = data_config
        self._scaler: StandardScaler | None = None

    def load_matrix(self) -> NDArray[np.float64]:
        """Read the snapshot matrix from disk."""
        path = self._config.snapshot_csv_path()
        LOGGER.info("Loading snapshot matrix from %s", path)
        return read_matrix_csv(path)

    def train_test_split_no_shuffle(
        self,
        matrix: NDArray[np.floating],
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Split columns into train and test without shuffling (time-ordered)."""
        x_train_t, x_test_t = train_test_split(
            matrix.T,
            test_size=self._config.test_size,
            shuffle=False,
            random_state=self._config.random_seed,
        )
        return x_train_t.T, x_test_t.T

    def erase_cylinder(self, matrix: NDArray[np.floating]) -> NDArray[np.floating]:
        """Remove rows lying inside the cylinder mask."""
        return erase_cylinder_rows(matrix, self._mesh)

    def fit_scaler(self, x_train: NDArray[np.floating]) -> NDArray[np.floating]:
        """Fit ``StandardScaler`` on the training matrix and return transformed data."""
        self._scaler = StandardScaler()
        if self._config.spatial_reduction:
            transformed = self._scaler.fit_transform(x_train.T).T
        else:
            transformed = self._scaler.fit_transform(x_train)
        LOGGER.debug("Fitted scaler on training matrix shape %s", x_train.shape)
        return transformed

    def transform(self, matrix: NDArray[np.floating]) -> NDArray[np.floating]:
        """Apply a previously fitted scaler."""
        if self._scaler is None:
            msg = "Scaler must be fitted before transform()"
            raise RuntimeError(msg)
        if self._config.spatial_reduction:
            return self._scaler.transform(matrix.T).T
        return self._scaler.transform(matrix)

    def run(self) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Execute load → split → cylinder mask → scaling on train and test."""
        matrix = self.load_matrix()
        x_train, x_test = self.train_test_split_no_shuffle(matrix)
        x_train_f = self.erase_cylinder(x_train)
        x_test_f = self.erase_cylinder(x_test)
        x_train_s = self.fit_scaler(x_train_f)
        x_test_s = self.transform(x_test_f)
        LOGGER.info(
            "Prepared data shapes train=%s test=%s",
            x_train_s.shape,
            x_test_s.shape,
        )
        return x_train_s, x_test_s
