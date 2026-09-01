"""Rigid/affine transform helpers for the transformation gizmo."""

from __future__ import annotations

import numpy as np


def apply_transform(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply a 4x4 homogeneous transform to a set of points.

    Parameters
    ----------
    points : np.ndarray
        Array of shape ``(N, 3)`` with point coordinates.
    matrix : np.ndarray
        Array of shape ``(4, 4)`` with the transform.

    Returns
    -------
    np.ndarray
        Array of shape ``(N, 3)`` with the transformed coordinates.
    """
    points = np.asarray(points, dtype=np.float64)
    matrix = np.asarray(matrix, dtype=np.float64)
    homogenous = np.hstack((points, np.ones((len(points), 1), dtype=np.float64)))
    transformed = (matrix @ homogenous.T).T
    return (transformed[:, :3] / transformed[:, 3:4]).astype(np.float64)


def rotate_vectors(vectors: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Rotate direction vectors with the linear part of a transform.

    The inverse-transpose of the upper-left 3x3 block is used, which maps
    normals correctly also under non-uniform scaling.

    Parameters
    ----------
    vectors : np.ndarray
        Array of shape ``(N, 3)`` with direction vectors.
    matrix : np.ndarray
        Array of shape ``(4, 4)`` with the transform.

    Returns
    -------
    np.ndarray
        Array of shape ``(N, 3)`` with the rotated vectors.
    """
    linear = np.asarray(matrix, dtype=np.float64)[:3, :3]
    normal_matrix = np.linalg.inv(linear).T
    return (np.asarray(vectors, dtype=np.float64) @ normal_matrix.T).astype(np.float64)


def translation_matrix(points: np.ndarray) -> np.ndarray:
    """Build the 4x4 matrix translating the origin to the point centroid.

    Parameters
    ----------
    points : np.ndarray
        Array of shape ``(N, 3)`` with point coordinates.

    Returns
    -------
    np.ndarray
        Array of shape ``(4, 4)`` with the translation.
    """
    matrix = np.eye(4, dtype=np.float64)
    if len(points):
        matrix[:3, 3] = np.asarray(points, dtype=np.float64).mean(axis=0)
    return matrix


def extract_translation(matrix: np.ndarray) -> np.ndarray:
    """Return the translation component of a 4x4 transform.

    Parameters
    ----------
    matrix : np.ndarray
        Array of shape ``(4, 4)`` with the transform.

    Returns
    -------
    np.ndarray
        Array of shape ``(3,)`` with the translation.
    """
    return np.asarray(matrix, dtype=np.float64)[:3, 3]


def relative_transform(combined: np.ndarray, base: np.ndarray) -> np.ndarray:
    """Extract the transform relative to a base transform.

    Given ``combined = base @ relative``, return ``relative``.

    Parameters
    ----------
    combined : np.ndarray
        Array of shape ``(4, 4)`` with the combined transform.
    base : np.ndarray
        Array of shape ``(4, 4)`` with the base transform.

    Returns
    -------
    np.ndarray
        Array of shape ``(4, 4)`` with the relative transform.
    """
    return np.linalg.inv(np.asarray(base, dtype=np.float64)) @ np.asarray(
        combined, dtype=np.float64
    )
