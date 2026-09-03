"""Readers for point cloud formats not covered by pyvista."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pcd_segmentation.core.model import PointCloudData

ASCII_EXTENSIONS = frozenset({".xyz", ".txt", ".pts", ".asc", ".csv"})
"""Text-based point cloud extensions handled by the internal parser."""

_TYPE_BASES = {"F": "f", "I": "i", "U": "u"}
"""PCD type letters mapped to numpy dtype characters."""


def read_ascii_cloud(path: Path) -> PointCloudData:
    """Read a whitespace- or comma-separated point cloud file.

    The first three columns are the xyz coordinates; if at least six columns
    are present, the following three are interpreted as rgb colors.

    Parameters
    ----------
    path : Path
        Path of the text file.

    Returns
    -------
    PointCloudData
        The parsed cloud.

    Raises
    ------
    ValueError
        If the file cannot be parsed as numeric columns.
    """
    delimiter = _detect_delimiter(path)
    try:
        data = np.loadtxt(path, delimiter=delimiter, comments="#", ndmin=2)
    except ValueError as error:
        raise ValueError(f"Cannot parse ASCII point cloud {path}: {error}") from error
    if data.shape[1] < 3:
        raise ValueError(f"ASCII point cloud {path} has fewer than 3 columns")
    points = data[:, :3].astype(np.float64)
    colors = _normalize_colors(data[:, 3:6]) if data.shape[1] >= 6 else None
    return PointCloudData(points=points, colors=colors, source=path)


def read_pcd(path: Path) -> PointCloudData:
    """Read a PCD v0.7 file in ascii or binary (uncompressed) format.

    Parameters
    ----------
    path : Path
        Path of the PCD file.

    Returns
    -------
    PointCloudData
        The parsed cloud.

    Raises
    ------
    ValueError
        If the header is malformed or the data layout is unsupported.
    """
    header, data_mode = _parse_pcd_header(path)
    count = int(header["POINTS"])
    fields = header["FIELDS"].split()
    _validate_pcd_layout(path, header, fields)
    if data_mode == "binary_compressed":
        raise ValueError(f"PCD file {path} uses binary_compressed storage, which is not supported")
    if data_mode == "ascii":
        points, colors = _read_pcd_ascii(path, header, count, fields)
    else:
        points, colors = _read_pcd_binary(path, header, count, fields)
    return PointCloudData(points=points, colors=colors, source=path)


def _read_pcd_ascii(
    path: Path,
    header: dict[str, int | str],
    count: int,
    fields: list[str],
) -> tuple[np.ndarray, np.ndarray | None]:
    """Read the point rows of an ascii PCD file.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray or None)
        The ``(N, 3)`` points and the ``(N, 3)`` colors (``None`` when the
        file has no ``rgb`` field).
    """
    with path.open("rb") as handle:
        rows = np.loadtxt(handle, skiprows=header["_data_line"], ndmin=2, max_rows=count)
    if rows.shape[0] != count:
        raise ValueError(f"PCD file {path} has fewer points than declared")
    points = rows[:, :3].astype(np.float64)
    colors = None
    if "rgb" in fields:
        colors = _unpack_packed_rgb(rows[:, fields.index("rgb")])
    return points, colors


def _read_pcd_binary(
    path: Path,
    header: dict[str, int | str],
    count: int,
    fields: list[str],
) -> tuple[np.ndarray, np.ndarray | None]:
    """Read the point rows of a binary PCD file.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray or None)
        The ``(N, 3)`` points and the ``(N, 3)`` colors (``None`` when the
        file has no ``rgb`` field).
    """
    types = header["TYPE"].split()
    sizes = [int(value) for value in header["SIZE"].split()]
    dtype = _pcd_dtype(fields, types, sizes)
    with path.open("rb") as handle:
        handle.seek(header["_data_offset"])
        payload = handle.read(count * dtype.itemsize)
    rows = np.frombuffer(payload, dtype=dtype, count=count)
    points = np.stack([rows[f].astype(np.float64) for f in ("x", "y", "z")], axis=1)
    colors = _unpack_packed_rgb(rows["rgb"], from_bits=True) if "rgb" in fields else None
    return points, colors


def _validate_pcd_layout(path: Path, header: dict[str, int | str], fields: list[str]) -> None:
    """Check that the PCD header describes a supported scalar layout.

    Raises
    ------
    ValueError
        If the header is inconsistent or uses multi-dimensional fields.
    """
    types = header["TYPE"].split()
    sizes = header["SIZE"].split()
    counts = header["COUNT"].split()
    if not (len(fields) == len(types) == len(sizes) == len(counts)):
        raise ValueError(f"Inconsistent PCD header in {path}")
    if any(int(value) != 1 for value in counts):
        raise ValueError(f"PCD file {path} uses multi-dimensional fields, not supported")
    if not {"x", "y", "z"}.issubset(fields):
        raise ValueError(f"PCD file {path} has no xyz coordinates")


def _detect_delimiter(path: Path) -> str | None:
    """Return the delimiter of a text cloud: comma or whitespace."""
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                return "," if "," in stripped else None
    return None


def _normalize_colors(rgb: np.ndarray) -> np.ndarray:
    """Normalize raw color columns to floats in ``[0, 1]``."""
    values = rgb.astype(np.float64)
    vmax = float(values.max()) if len(values) else 0.0
    if vmax > 1.0:
        values /= 65535.0 if vmax > 255.0 else 255.0
    return np.clip(values, 0.0, 1.0).astype(np.float32)


def _parse_pcd_header(path: Path) -> tuple[dict[str, int | str], str]:
    """Parse the text header of a PCD file.

    Parameters
    ----------
    path : Path
        Path of the PCD file.

    Returns
    -------
    tuple of (dict of int or str, str)
        The header key/value pairs (with private ``_data_line`` and
        ``_data_offset`` keys) and the data storage mode.

    Raises
    ------
    ValueError
        If the file has no DATA section.
    """
    header: dict[str, int | str] = {}
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.decode("ascii", errors="replace").strip()
            if line.startswith("#") or not line:
                continue
            if line.startswith("DATA "):
                header["_data_line"] = line_number
                header["_data_offset"] = handle.tell()
                return header, line.split()[1]
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                header[parts[0]] = parts[1]
    raise ValueError(f"PCD file {path} has no DATA section")


def _pcd_dtype(fields: list[str], types: list[str], sizes: list[int]) -> np.dtype:
    """Build the numpy dtype described by the PCD header fields.

    Only scalar fields (``COUNT 1``) are supported.

    Parameters
    ----------
    fields : list of str
        Field names.
    types : list of str
        PCD type letters (``F``, ``I`` or ``U``).
    sizes : list of int
        Field byte sizes (1, 2, 4 or 8).

    Returns
    -------
    np.dtype
        The structured dtype of one point record.
    """
    descriptors = [
        (field, f"{_TYPE_BASES[pcd_type]}{size}")
        for field, pcd_type, size in zip(fields, types, sizes, strict=True)
    ]
    return np.dtype(descriptors)


def _unpack_packed_rgb(packed: np.ndarray, from_bits: bool = False) -> np.ndarray:
    """Decode packed float32 rgb values into rgb floats in ``[0, 1]``.

    Parameters
    ----------
    packed : np.ndarray
        Array of packed color values.
    from_bits : bool
        When ``True`` the float bit pattern encodes the packed integer (as
        stored in binary PCD files); otherwise the float holds the numeric
        value of the packed integer (as parsed from ascii files).

    Returns
    -------
    np.ndarray
        Array of shape ``(N, 3)`` with rgb values in ``[0, 1]``.
    """
    values = np.asarray(packed, dtype=np.float32)
    packed_int = values.view(np.uint32) if from_bits else values.astype(np.uint32)
    red = (packed_int >> 16) & 0xFF
    green = (packed_int >> 8) & 0xFF
    blue = packed_int & 0xFF
    return np.stack((red, green, blue), axis=1).astype(np.float32) / 255.0
