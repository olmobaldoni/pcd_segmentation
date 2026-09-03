"""Screen-space projection and containment tests for selection tools."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ViewState:
    """Snapshot of the camera and viewport needed for world-to-screen projection.

    Attributes
    ----------
    extrinsics : np.ndarray
        4x4 matrix mapping world coordinates to camera space (OpenGL
        convention, the camera looks down the negative Z axis).
    fov_deg : float
        Vertical field of view in degrees.
    aspect : float
        Width/height ratio of the viewport.
    width : int
        Viewport width in pixels.
    height : int
        Viewport height in pixels.
    orthographic : bool
        Whether the view uses orthographic projection.
    length_scale : float
        Scene length scale, used to size the orthographic frustum.
    """

    extrinsics: np.ndarray
    fov_deg: float
    aspect: float
    width: int
    height: int
    orthographic: bool
    length_scale: float


def project_to_screen(points: np.ndarray, view: ViewState) -> np.ndarray:
    """Project world-space points to pixel coordinates.

    Parameters
    ----------
    points : np.ndarray
        Array of shape ``(N, 3)`` with world coordinates.
    view : ViewState
        Camera and viewport description.

    Returns
    -------
    np.ndarray
        Array of shape ``(N, 2)`` with pixel coordinates, origin at the
        top-left corner of the viewport and y axis pointing downwards.
    """
    points = np.asarray(points, dtype=np.float64)
    homogenous = np.hstack((points, np.ones((len(points), 1), dtype=np.float64)))
    camera_space = (view.extrinsics @ homogenous.T).T
    half_tan = np.tan(np.deg2rad(view.fov_deg) * 0.5)
    if view.orthographic:
        scale = max(view.length_scale * half_tan * 2.0, 1e-12)
        x_ndc = camera_space[:, 0] / (scale * view.aspect)
        y_ndc = camera_space[:, 1] / scale
    else:
        depth = np.maximum(-camera_space[:, 2], 1e-12)
        x_ndc = camera_space[:, 0] / (depth * half_tan * view.aspect)
        y_ndc = camera_space[:, 1] / (depth * half_tan)
    screen_x = (x_ndc * 0.5 + 0.5) * view.width
    screen_y = (0.5 - y_ndc * 0.5) * view.height
    return np.stack((screen_x, screen_y), axis=1)


def points_in_polygon(screen_points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Test which 2D points lie inside a (possibly concave) polygon.

    Uses the vectorized even-odd ray casting rule.

    Parameters
    ----------
    screen_points : np.ndarray
        Array of shape ``(N, 2)`` with the query points.
    polygon : np.ndarray
        Array of shape ``(M, 2)`` with the polygon vertices in order.

    Returns
    -------
    np.ndarray
        Boolean array of shape ``(N,)``.
    """
    polygon = np.asarray(polygon, dtype=np.float64)
    if len(polygon) < 3:
        return np.zeros(len(screen_points), dtype=bool)
    x = screen_points[:, 0]
    y = screen_points[:, 1]
    inside = np.zeros(len(screen_points), dtype=bool)
    j = len(polygon) - 1
    with np.errstate(divide="ignore", invalid="ignore"):
        for i in range(len(polygon)):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            crosses = (yi > y) != (yj > y)
            x_edge = (xj - xi) * (y - yi) / (yj - yi) + xi
            inside ^= crosses & (x < x_edge)
            j = i
    return inside


def points_in_rect(
    screen_points: np.ndarray,
    corner_a: tuple[float, float],
    corner_b: tuple[float, float],
) -> np.ndarray:
    """Test which 2D points lie inside an axis-aligned rectangle.

    Parameters
    ----------
    screen_points : np.ndarray
        Array of shape ``(N, 2)`` with the query points.
    corner_a : tuple of float
        First rectangle corner in pixel coordinates.
    corner_b : tuple of float
        Opposite rectangle corner in pixel coordinates.

    Returns
    -------
    np.ndarray
        Boolean array of shape ``(N,)``.
    """
    x_min, x_max = sorted((corner_a[0], corner_b[0]))
    y_min, y_max = sorted((corner_a[1], corner_b[1]))
    x = screen_points[:, 0]
    y = screen_points[:, 1]
    return (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)


def is_simple_polygon(polygon: np.ndarray) -> bool:
    """Return whether a polygon is simple (its edges never self-intersect).

    A concave but non-self-intersecting polygon is still simple. Simple
    polygons can be filled reliably; self-intersecting ones produce undefined
    fills with most triangulators and are rendered as an outline only.

    Parameters
    ----------
    polygon : np.ndarray
        Array of shape ``(M, 2)`` with the polygon vertices in order.

    Returns
    -------
    bool
        ``True`` when the polygon is simple, ``False`` when it self-intersects.
    """
    polygon = np.asarray(polygon, dtype=np.float64)
    count = len(polygon)
    if count < 4:
        return True
    for i in range(count):
        a0, a1 = polygon[i], polygon[(i + 1) % count]
        for j in range(i + 1, count):
            b0, b1 = polygon[j], polygon[(j + 1) % count]
            if _segments_properly_intersect(a0, a1, b0, b1):
                return False
    return True


def _segments_properly_intersect(
    a0: np.ndarray,
    a1: np.ndarray,
    b0: np.ndarray,
    b1: np.ndarray,
) -> bool:
    """Return whether two segments cross at an interior point.

    Shared endpoints (adjacent polygon edges) are not counted as crossings.

    Parameters
    ----------
    a0, a1, b0, b1 : np.ndarray
        Endpoints of the two segments.

    Returns
    -------
    bool
        ``True`` when the segments properly intersect.
    """
    d1 = _cross2d(a1 - a0, b0 - a0)
    d2 = _cross2d(a1 - a0, b1 - a0)
    d3 = _cross2d(b1 - b0, a0 - b0)
    d4 = _cross2d(b1 - b0, a1 - b0)
    if ((d1 > 0 > d2) or (d2 > 0 > d1)) and ((d3 > 0 > d4) or (d4 > 0 > d3)):
        return True
    return False


def _cross2d(u: np.ndarray, v: np.ndarray) -> float:
    """Return the 2D cross product (z component) of two in-plane vectors."""
    return float(u[0] * v[1] - u[1] * v[0])
