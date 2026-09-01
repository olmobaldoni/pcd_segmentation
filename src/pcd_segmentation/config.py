"""Global configuration constants for the segmentation tool."""

from __future__ import annotations

UNASSIGNED: int = -1
"""Label value used for points that do not belong to any segment."""

MAX_UNDO: int = 100
"""Maximum number of undo records kept in memory."""

UNASSIGNED_COLOR: tuple[float, float, float] = (0.55, 0.55, 0.55)
"""RGB color in ``[0, 1]`` used to render points not assigned to any segment."""

SELECTION_COLOR: tuple[float, float, float] = (1.0, 0.85, 0.1)
"""RGB color in ``[0, 1]`` used to highlight the current selection."""

SEGMENT_PALETTE: tuple[tuple[float, float, float], ...] = (
    (0.90, 0.20, 0.20),
    (0.20, 0.60, 0.90),
    (0.35, 0.75, 0.35),
    (0.90, 0.65, 0.15),
    (0.75, 0.30, 0.75),
    (0.15, 0.75, 0.70),
    (0.95, 0.45, 0.55),
    (0.55, 0.55, 0.90),
    (0.70, 0.80, 0.25),
    (0.85, 0.55, 0.25),
    (0.45, 0.80, 0.90),
    (0.80, 0.30, 0.30),
    (0.30, 0.45, 0.85),
    (0.60, 0.80, 0.35),
    (0.85, 0.45, 0.85),
    (0.40, 0.70, 0.55),
    (0.90, 0.70, 0.40),
    (0.55, 0.40, 0.80),
    (0.75, 0.80, 0.85),
    (0.80, 0.75, 0.45),
)
"""Distinct color palette cycled through when creating new segments."""

CLOUD_NAME: str = "cloud"
"""Polyscope structure name of the main point cloud."""

SELECTION_NAME: str = "selection"
"""Polyscope structure name of the selection highlight cloud."""

DEFAULT_OUT_DIR: str = "segments"
"""Default directory for the exported per-segment PLY files."""
