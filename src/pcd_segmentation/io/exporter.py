"""Per-segment PLY export through pyvista."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pyvista as pv

from pcd_segmentation.core.model import PointCloudData
from pcd_segmentation.core.segments import SegmentManager


def sanitize_filename(name: str) -> str:
    """Turn a segment name into a safe file-name fragment.

    Parameters
    ----------
    name : str
        Segment name.

    Returns
    -------
    str
        Name with all non-alphanumeric characters replaced by underscores.
    """
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")
    return cleaned or "segment"


def export_segment(path: Path, points: np.ndarray, colors_u8: np.ndarray) -> None:
    """Write a subset of points to a binary PLY file.

    Parameters
    ----------
    path : Path
        Destination file.
    points : np.ndarray
        Array of shape ``(M, 3)`` with point coordinates.
    colors_u8 : np.ndarray
        Array of shape ``(M, 3)`` with rgb values in ``[0, 255]``.
    """
    pcd = pv.PolyData(np.asarray(points, dtype=np.float32))
    pcd.point_data["RGB"] = np.asarray(colors_u8, dtype=np.uint8)
    pcd.save(str(path), binary=True, texture="RGB")


def export_segments(
    out_dir: Path,
    cloud: PointCloudData,
    manager: SegmentManager,
) -> list[Path]:
    """Export every non-empty segment of the cloud as an individual PLY file.

    Points keep their original colors when the source cloud has any;
    otherwise the segment color is used. Files are named
    ``<cloud_stem>_seg_<segment_name>.ply``.

    Parameters
    ----------
    out_dir : Path
        Destination directory, created if missing.
    cloud : PointCloudData
        Source cloud.
    manager : SegmentManager
        Segment bookkeeping to export.

    Returns
    -------
    list of Path
        Paths of the written files.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = cloud.source.stem if cloud.source is not None else "cloud"
    written: list[Path] = []
    for segment in manager.segments:
        indices = manager.indices_of(segment.id)
        if len(indices) == 0:
            continue
        points = cloud.points[indices]
        if cloud.colors is not None:
            colors_u8 = np.round(cloud.colors[indices] * 255.0).astype(np.uint8)
        else:
            fallback = np.round(np.asarray(segment.color) * 255.0).astype(np.uint8)
            colors_u8 = np.tile(fallback, (len(indices), 1))
        target = out_dir / f"{stem}_seg_{sanitize_filename(segment.name)}.ply"
        export_segment(target, points, colors_u8)
        written.append(target)
    return written
