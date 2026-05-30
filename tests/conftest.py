"""Shared pytest setup.

Force a non-interactive matplotlib backend before any test (or the package's
visualization module) imports ``pyplot``, so figure/image exports work headless.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
