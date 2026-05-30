"""Filesystem helpers for run artifacts (CSV matrices, error vectors, JSON)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from mlcfd.logging_config import get_logger

LOGGER = get_logger("io")


def ensure_directory(path: Path) -> None:
    """Create a directory if it does not already exist.

    Args:
        path: Directory path to ensure on disk.
    """
    if path.exists():
        return
    path.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Created directory %s", path)


def write_matrix_csv(path: Path, matrix: NDArray[np.floating]) -> None:
    """Write a 2D floating array to CSV without row indices.

    Args:
        path: Destination CSV path.
        matrix: Numeric matrix to persist.

    Raises:
        ValueError: If ``matrix`` is not two-dimensional.
    """
    if matrix.ndim != 2:
        msg = f"Expected a 2D matrix, got shape {matrix.shape}"
        raise ValueError(msg)
    ensure_directory(path.parent)
    LOGGER.debug("Writing matrix CSV to %s with shape %s", path, matrix.shape)
    pd.DataFrame(matrix).to_csv(path, index=False)


def write_vector_csv(path: Path, vector: NDArray[np.floating]) -> None:
    """Write a 1D floating array to CSV, one value per line, without a header.

    Mirrors the historical ``np.savetxt`` layout used for sweep error curves so
    existing artifacts stay byte-for-byte identical.

    Args:
        path: Destination CSV path.
        vector: One-dimensional numeric array to persist.

    Raises:
        ValueError: If ``vector`` is not one-dimensional.
    """
    if vector.ndim != 1:
        msg = f"Expected a 1D vector, got shape {vector.shape}"
        raise ValueError(msg)
    ensure_directory(path.parent)
    LOGGER.debug("Writing vector CSV to %s with length %s", path, vector.shape[0])
    np.savetxt(path, vector, delimiter=",")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON-serialisable mapping with two-space indentation as UTF-8.

    Args:
        path: Destination JSON path.
        payload: Mapping to serialise.
    """
    ensure_directory(path.parent)
    LOGGER.debug("Writing JSON to %s", path)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_matrix_csv(path: Path) -> NDArray[np.float64]:
    """Read a CSV matrix into a float64 ndarray.

    Args:
        path: CSV file path.

    Returns:
        Copy of the table values as ``float64``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    if not path.is_file():
        msg = f"CSV file not found: {path}"
        raise FileNotFoundError(msg)
    LOGGER.info("Reading matrix CSV from %s", path)
    frame = pd.read_csv(path)
    return frame.to_numpy(dtype=np.float64, copy=True)
