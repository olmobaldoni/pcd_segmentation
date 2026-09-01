"""Point cloud loading backed by pyvista and format-specific readers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv

from pcd_segmentation.core.model import PointCloudData
from pcd_segmentation.io import readers

SUPPORTED_EXTENSIONS = frozenset(
    {
        ".ply",
        ".obj",
        ".stl",
        ".vtk",
        ".vtp",
        ".vtu",
        ".pcd",
        ".las",
        ".laz",
    }
    | readers.ASCII_EXTENSIONS
)
"""File extensions the loader accepts."""


def is_supported(path: Path) -> bool:
    """Return whether the file extension can be loaded.

    Parameters
    ----------
    path : Path
        Path to check.

    Returns
    -------
    bool
        ``True`` if the extension is supported.
    """
    return Path(path).suffix.lower() in SUPPORTED_EXTENSIONS


def load_point_cloud(path: Path, **kwargs: Any) -> PointCloudData:
    """Load a point cloud from disk.

    PLY, OBJ, STL and the other formats understood by ``pyvista.read`` are
    loaded through pyvista; PCD, LAS/LAZ and XYZ-style text clouds use the
    dedicated readers. Meshes with faces are flattened to their vertices.

    Parameters
    ----------
    path : Path
        Path of the file to load.
    **kwargs : dict
        Extra keyword arguments, reserved for future use.

    Returns
    -------
    PointCloudData
        The loaded cloud, with colors in ``[0, 1]`` when available.

    Raises
    ------
    ValueError
        If the extension is unsupported or the file cannot be parsed.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in readers.ASCII_EXTENSIONS:
        return readers.read_ascii_cloud(path)
    if suffix == ".pcd":
        return readers.read_pcd(path)
    if suffix in {".las", ".laz"}:
        return _load_las(path)
    if not is_supported(path):
        raise ValueError(f"Unsupported point cloud format: {suffix}")
    try:
        mesh = pv.read(path)
    except (OSError, ValueError) as error:
        raise ValueError(f"Failed to read {path}: {error}") from error
    return _from_pyvista(mesh, path)


def _load_las(path: Path) -> PointCloudData:
    """Load a LAS/LAZ file through laspy.

    Parameters
    ----------
    path : Path
        Path of the LAS/LAZ file.

    Returns
    -------
    PointCloudData
        The loaded cloud.

    Raises
    ------
    ValueError
        If the file cannot be read.
    """
    try:
        import laspy
    except ImportError as error:  # pragma: no cover - dependency is required
        raise ValueError("laspy is required to read LAS/LAZ files") from error
    try:
        las = laspy.read(path)
    except Exception as error:  # noqa: BLE001 - laspy raises various error types
        raise ValueError(f"Failed to read {path}: {error}") from error
    header = las.header
    points = np.stack(
        (
            las.x * header.scales[0] + header.offsets[0],
            las.y * header.scales[1] + header.offsets[1],
            las.z * header.scales[2] + header.offsets[2],
        ),
        axis=1,
    ).astype(np.float64)
    colors = None
    if all(hasattr(las, name) for name in ("red", "green", "blue")):
        rgb = np.stack((las.red, las.green, las.blue), axis=1)
        colors = readers._normalize_colors(rgb)
    return PointCloudData(points=points, colors=colors, source=path)


def _from_pyvista(mesh: Any, path: Path) -> PointCloudData:
    """Convert a pyvista dataset into a :class:`PointCloudData`.

    Parameters
    ----------
    mesh : pyvista.DataSet
        Dataset returned by ``pyvista.read``.
    path : Path
        Source path, kept for reference.

    Returns
    -------
    PointCloudData
        The flattened point cloud.

    Raises
    ------
    ValueError
        If the dataset contains no points.
    """
    if mesh.n_points == 0:
        raise ValueError(f"{path} contains no points")
    points = np.asarray(mesh.points, dtype=np.float64)
    colors = None
    if "RGB" in mesh.point_data:
        colors = readers._normalize_colors(mesh.point_data["RGB"])
    elif "RGBA" in mesh.point_data:
        colors = readers._normalize_colors(mesh.point_data["RGBA"][:, :3])
    normals = None
    if "Normals" in mesh.point_data:
        normals = np.asarray(mesh.point_data["Normals"], dtype=np.float64)
    return PointCloudData(points=points, colors=colors, normals=normals, source=path)
