"""Core data structures shared across the application."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class PointCloudData:
    """A point cloud loaded from disk, with optional per-point attributes.

    Attributes
    ----------
    points : np.ndarray
        Array of shape ``(N, 3)`` holding the xyz coordinates of each point.
    colors : np.ndarray or None
        Array of shape ``(N, 3)`` with rgb values in ``[0, 1]``, or ``None``
        if the source file does not carry color information.
    normals : np.ndarray or None
        Array of shape ``(N, 3)`` with per-point normals, or ``None`` if the
        source file does not carry normals.
    source : Path or None
        Path of the file the cloud was loaded from.
    """

    points: np.ndarray
    colors: np.ndarray | None = None
    normals: np.ndarray | None = None
    source: Path | None = None

    @property
    def num_points(self) -> int:
        """Return the number of points in the cloud."""
        return len(self.points)
