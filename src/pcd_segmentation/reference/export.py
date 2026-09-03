"""Write the re-referenced cloud to disk: PLY plus a JSON sidecar."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyvista as pv

from pcd_segmentation.core.model import PointCloudData

METADATA_VERSION: int = 1
"""Version of the JSON sidecar format."""


def write_cloud_ply(path: Path, cloud: PointCloudData) -> Path:
    """Write the whole cloud to a binary PLY file, preserving original colors.

    Parameters
    ----------
    path : Path
        Destination file (its parent directory is created if needed).
    cloud : PointCloudData
        Cloud whose current points and colors are written.

    Returns
    -------
    Path
        The resolved destination path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh = pv.PolyData(np.asarray(cloud.points, dtype=np.float32))
    if cloud.colors is not None:
        colors_u8 = np.round(np.asarray(cloud.colors) * 255.0).astype(np.uint8)
        mesh.point_data["RGB"] = colors_u8
        mesh.save(str(path), binary=True, texture="RGB")
    else:
        mesh.save(str(path), binary=True)
    return path


def write_sidecar_json(
    path: Path,
    cloud: PointCloudData,
    *,
    matrix: np.ndarray,
    translation: np.ndarray,
    source_convention: str,
    target_convention: str,
    reference_index: int | None = None,
    reference_original: np.ndarray | None = None,
) -> Path:
    """Write the JSON sidecar describing the applied re-referencing.

    Parameters
    ----------
    path : Path
        Destination JSON file (its parent directory is created if needed).
    cloud : PointCloudData
        Source cloud the metadata refers to.
    matrix : np.ndarray
        3x3 linear part mapping the original coordinates to the current ones.
    translation : np.ndarray
        Translation part (shape ``(3,)``) of the same mapping.
    source_convention : str
        Axis triple of the original data frame (``conventions.NONE_TRIPLE``
        when no reorientation was applied).
    target_convention : str
        Axis triple of the current data frame.
    reference_index : int or None
        Index of the point chosen as the new origin, if any.
    reference_original : np.ndarray or None
        Coordinates of the reference point in the original frame.

    Returns
    -------
    Path
        The resolved destination path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    source = str(cloud.source) if cloud.source is not None else None
    reference = {
        "index": reference_index,
        "original": _json_array(reference_original),
    }
    document = {
        "version": METADATA_VERSION,
        "application": "pcd_segmentation.reference",
        "source": source,
        "num_points": cloud.num_points,
        "reference": reference,
        "base_to_current": {
            "matrix": _json_array(matrix),
            "translation": _json_array(translation),
        },
        "axes_conversion": {
            "source": source_convention,
            "target": target_convention,
        },
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2)
    return path


def export_referenced(
    ply_path: Path,
    cloud: PointCloudData,
    *,
    matrix: np.ndarray,
    translation: np.ndarray,
    source_convention: str,
    target_convention: str,
    reference_index: int | None = None,
    reference_original: np.ndarray | None = None,
) -> tuple[Path, Path]:
    """Write the current cloud and its JSON sidecar.

    Parameters
    ----------
    ply_path : Path
        Destination PLY file.
    cloud : PointCloudData
        Cloud to export.
    matrix : np.ndarray
        3x3 linear part mapping the original coordinates to the current ones.
    translation : np.ndarray
        Translation part (shape ``(3,)``).
    source_convention : str
        Axis triple of the original data frame.
    target_convention : str
        Axis triple of the current data frame.
    reference_index : int or None
        Index of the point chosen as the new origin, if any.
    reference_original : np.ndarray or None
        Coordinates of the reference point in the original frame.

    Returns
    -------
    tuple of (Path, Path)
        The written PLY and JSON paths.
    """
    ply_path = write_cloud_ply(ply_path, cloud)
    json_path = write_sidecar_json(
        ply_path.with_suffix(".json"),
        cloud,
        matrix=matrix,
        translation=translation,
        source_convention=source_convention,
        target_convention=target_convention,
        reference_index=reference_index,
        reference_original=reference_original,
    )
    return ply_path, json_path


def _json_array(value: np.ndarray | None) -> list[float] | None:
    """Flatten a numpy array to nested native floats for JSON output."""
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    return array.tolist()
