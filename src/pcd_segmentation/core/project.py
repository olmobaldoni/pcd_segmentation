"""Save and load segmentation projects as JSON files."""

from __future__ import annotations

import base64
import json
import zlib
from pathlib import Path
from typing import Any

import numpy as np

from pcd_segmentation.core.segments import Segment

PROJECT_VERSION: int = 1
"""Version of the project file format."""


def save_project(
    path: Path,
    source: Path | None,
    segments: list[Segment],
    labels: np.ndarray,
    transform_matrix: np.ndarray | None = None,
) -> None:
    """Serialize the segmentation state of a cloud to a JSON project file.

    Parameters
    ----------
    path : Path
        Destination path of the project file.
    source : Path or None
        Path of the source point cloud, stored relative to the project file.
    segments : list of Segment
        Segments defined for the cloud.
    labels : np.ndarray
        Per-point segment ids (``config.UNASSIGNED`` for unassigned points).
    transform_matrix : np.ndarray or None
        Optional 4x4 transform applied to the cloud in the viewer.

    Notes
    -----
    The labels array is stored as a base64-encoded zlib-compressed buffer to
    keep the file compact even for large clouds.
    """
    labels_array = np.asarray(labels, dtype=np.int64)
    payload = zlib.compress(labels_array.tobytes())
    doc: dict[str, Any] = {
        "version": PROJECT_VERSION,
        "source": str(source) if source is not None else None,
        "num_points": int(len(labels_array)),
        "segments": [
            {
                "id": segment.id,
                "name": segment.name,
                "color": [float(component) for component in segment.color],
                "visible": bool(segment.visible),
            }
            for segment in segments
        ],
        "labels_b64": base64.b64encode(payload).decode("ascii"),
    }
    if transform_matrix is not None:
        doc["transform"] = np.asarray(transform_matrix, dtype=np.float64).flatten().tolist()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(doc, handle, indent=2)


def load_project(path: Path) -> dict[str, Any]:
    """Read a project file and return its segmentation state.

    Parameters
    ----------
    path : Path
        Path of the project file.

    Returns
    -------
    dict
        Dictionary with keys ``source`` (resolved :class:`Path` or ``None``),
        ``segments`` (list of :class:`Segment`), ``labels`` (np.ndarray) and
        ``transform`` (4x4 np.ndarray, identity when not stored).

    Raises
    ------
    ValueError
        If the file is not a valid project or its version is unsupported.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        doc = json.load(handle)
    if doc.get("version") != PROJECT_VERSION:
        raise ValueError(
            f"Unsupported project version {doc.get('version')!r}, expected {PROJECT_VERSION}"
        )
    labels = np.frombuffer(
        zlib.decompress(base64.b64decode(doc["labels_b64"])),
        dtype=np.int64,
    )
    if len(labels) != int(doc["num_points"]):
        raise ValueError("Project labels do not match the declared number of points")
    segments = [
        Segment(
            id=int(item["id"]),
            name=str(item["name"]),
            color=tuple(float(c) for c in item["color"]),
            visible=bool(item.get("visible", True)),
        )
        for item in doc.get("segments", [])
    ]
    source_path: Path | None = None
    if doc.get("source"):
        source_path = Path(doc["source"])
        if not source_path.is_absolute():
            source_path = path.parent / source_path
    transform_matrix = np.eye(4, dtype=np.float64)
    if "transform" in doc:
        transform_matrix = np.asarray(doc["transform"], dtype=np.float64).reshape(4, 4)
    return {
        "source": source_path,
        "segments": segments,
        "labels": labels,
        "transform": transform_matrix,
    }
