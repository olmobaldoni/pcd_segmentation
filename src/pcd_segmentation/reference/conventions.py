"""Coordinate-system conventions expressed as axis triples.

A convention states where the positive X, Y and Z axes of a point cloud
point in the physical world. Each axis is described by a letter taken from
``R``/``L`` (right/left), ``U``/``D`` (up/down) and ``F``/``B``
(forward/back). Knowing the source and the target convention is enough to
derive the signed permutation that converts point coordinates between the
two, for example ``RDF -> RUB`` maps ``(x, y, z)`` to ``(x, -y, -z)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NONE_TRIPLE: str = "NONE"
"""Triple key used to request no reorientation (identity mapping)."""

NONE_DESCRIPTION: str = "None (keep axes as they are)"


@dataclass(frozen=True)
class Convention:
    """A named axis convention.

    Attributes
    ----------
    triple : str
        Concatenated axis letters, e.g. ``"RDF"``.
    directions : tuple of str
        The three axis letters in X, Y, Z order.
    description : str
        Human-readable meaning of the triple.
    """

    triple: str
    directions: tuple[str, str, str]
    description: str


CONVENTIONS: tuple[Convention, ...] = (
    Convention("RUB", ("R", "U", "B"), "X=right Y=up Z=back"),
    Convention("RDF", ("R", "D", "F"), "X=right Y=down Z=forward"),
    Convention("LUF", ("L", "U", "F"), "X=left Y=up Z=forward"),
    Convention("FLU", ("F", "L", "U"), "X=forward Y=left Z=up"),
    Convention("FRD", ("F", "R", "D"), "X=forward Y=right Z=down"),
)
"""The convention presets offered to the user, keyed by their triple."""

_CONVENTIONS_BY_TRIPLE: dict[str, Convention] = {
    convention.triple: convention for convention in CONVENTIONS
}

_AXIS_VECTORS: dict[str, tuple[float, float, float]] = {
    "R": (1.0, 0.0, 0.0),
    "L": (-1.0, 0.0, 0.0),
    "U": (0.0, 1.0, 0.0),
    "D": (0.0, -1.0, 0.0),
    "B": (0.0, 0.0, 1.0),
    "F": (0.0, 0.0, -1.0),
}
"""Numeric mapping of the physical direction letters onto the canonical axes."""

_DIRECTION_WORDS: dict[str, str] = {
    "R": "right",
    "L": "left",
    "U": "up",
    "D": "down",
    "F": "forward",
    "B": "back",
}
"""Lowercase world names of the physical direction letters."""


def preset_triples() -> list[str]:
    """Return the selectable triple keys, ``NONE`` first.

    Returns
    -------
    list of str
        Keys accepted by :func:`axis_matrix`, in UI order.
    """
    return [NONE_TRIPLE, *(convention.triple for convention in CONVENTIONS)]


def description(triple: str) -> str:
    """Return the human-readable description of a triple key.

    Parameters
    ----------
    triple : str
        A triple returned by :func:`preset_triples`.

    Returns
    -------
    str
        The description text.

    Raises
    ------
    ValueError
        If the triple is unknown.
    """
    if triple == NONE_TRIPLE:
        return NONE_DESCRIPTION
    if triple not in _CONVENTIONS_BY_TRIPLE:
        raise ValueError(f"Unknown axis convention: {triple!r}")
    return _CONVENTIONS_BY_TRIPLE[triple].description


def axis_letters(triple: str) -> str:
    """Describe each data axis with the world direction it points to.

    Parameters
    ----------
    triple : str
        A triple returned by :func:`preset_triples`.

    Returns
    -------
    str
        Text like ``"x=right, y=down, z=forward"``, or an empty string for
        the ``NONE`` triple.

    Raises
    ------
    ValueError
        If the triple is unknown.
    """
    if triple == NONE_TRIPLE:
        return ""
    if triple not in _CONVENTIONS_BY_TRIPLE:
        raise ValueError(f"Unknown axis convention: {triple!r}")
    axes = ("x", "y", "z")
    directions = _CONVENTIONS_BY_TRIPLE[triple].directions
    parts = [
        f"{axis}={_DIRECTION_WORDS[letter]}" for axis, letter in zip(axes, directions, strict=True)
    ]
    return ", ".join(parts)


def axis_matrix(triple: str) -> np.ndarray:
    """Build the 3x3 matrix whose rows are the directions of the three axes.

    The matrix maps coordinates expressed in the canonical frame onto the
    frame described by the convention: its ``i``-th row is the direction of
    the ``i``-th data axis.

    Parameters
    ----------
    triple : str
        A triple returned by :func:`preset_triples`.

    Returns
    -------
    np.ndarray
        Array of shape ``(3, 3)``.

    Raises
    ------
    ValueError
        If the triple is unknown or does not form an orthonormal basis.
    """
    if triple == NONE_TRIPLE:
        return np.eye(3, dtype=np.float64)
    if triple not in _CONVENTIONS_BY_TRIPLE:
        raise ValueError(f"Unknown axis convention: {triple!r}")
    rows = [_AXIS_VECTORS[letter] for letter in _CONVENTIONS_BY_TRIPLE[triple].directions]
    matrix = np.asarray(rows, dtype=np.float64)
    if not np.allclose(matrix @ matrix.T, np.eye(3)):
        raise ValueError(f"Convention {triple!r} is not an orthonormal axis frame")
    return matrix


def handedness(triple: str) -> str:
    """Return ``"right"`` or ``"left"`` for an axis convention.

    Parameters
    ----------
    triple : str
        A triple returned by :func:`preset_triples`.

    Returns
    -------
    str
        Handedness of the convention.

    Raises
    ------
    ValueError
        If the triple is unknown.
    """
    if triple == NONE_TRIPLE:
        return "right"
    determinant = np.linalg.det(axis_matrix(triple))
    return "right" if determinant > 0.0 else "left"


def axes_matrix(source: str, target: str) -> np.ndarray:
    """Return the matrix converting coordinates from one convention to another.

    Given a source frame ``S`` and a target frame ``T``, the returned
    ``M`` satisfies ``M @ p`` = the coordinates of a point ``p`` (expressed
    in ``S``) re-expressed in ``T``.

    Parameters
    ----------
    source : str
        Triple describing the current data frame.
    target : str
        Triple describing the desired data frame.

    Returns
    -------
    np.ndarray
        Array of shape ``(3, 3)`` mapping source coordinates to target
        coordinates.
    """
    source_matrix = axis_matrix(source)
    target_matrix = axis_matrix(target)
    return target_matrix @ source_matrix.T


def convert_points(points: np.ndarray, source: str, target: str) -> np.ndarray:
    """Re-express point coordinates between two axis conventions.

    Parameters
    ----------
    points : np.ndarray
        Array of shape ``(N, 3)`` in the source convention.
    source : str
        Triple describing the current data frame.
    target : str
        Triple describing the desired data frame.

    Returns
    -------
    np.ndarray
        Array of shape ``(N, 3)`` in the target convention.
    """
    matrix = axes_matrix(source, target)
    return (matrix @ np.asarray(points, dtype=np.float64).T).T
