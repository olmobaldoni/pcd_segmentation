"""Helpers to draw screen-space overlays with ImGui draw lists."""

from __future__ import annotations

from typing import Any

import numpy as np
import polyscope.imgui as psim


def col32(rgb: tuple[float, float, float], alpha: float = 1.0) -> int:
    """Convert an rgb tuple in ``[0, 1]`` to a packed ImGui color.

    Parameters
    ----------
    rgb : tuple of float
        RGB color in ``[0, 1]``.
    alpha : float
        Alpha component in ``[0, 1]``.

    Returns
    -------
    int
        Packed RGBA color for ImGui draw lists.
    """
    red, green, blue = (int(round(component * 255)) for component in rgb)
    alpha_byte = int(round(alpha * 255))
    return psim.IM_COL32(red, green, blue, alpha_byte)


def draw_closed_polyline(
    draw_list: Any,
    points: np.ndarray,
    color: int,
    thickness: float = 2.0,
) -> None:
    """Draw a closed polyline joining the given pixel coordinates.

    Parameters
    ----------
    draw_list : ImDrawList
        Target draw list.
    points : np.ndarray
        Array of shape ``(M, 2)`` with pixel coordinates.
    color : int
        Packed RGBA color.
    thickness : float
        Line thickness in pixels.
    """
    vertices = np.asarray(points, dtype=np.float32)
    if len(vertices) == 0:
        return
    if len(vertices) == 1:
        draw_list.AddCircleFilled(tuple(vertices[0]), thickness, color)
        return
    closed = np.vstack((vertices, vertices[:1]))
    draw_list.AddPolyline(closed, color, 0, thickness)


def draw_open_polyline(
    draw_list: Any,
    points: np.ndarray,
    color: int,
    thickness: float = 2.0,
) -> None:
    """Draw an open polyline joining the given pixel coordinates.

    Parameters
    ----------
    draw_list : ImDrawList
        Target draw list.
    points : np.ndarray
        Array of shape ``(M, 2)`` with pixel coordinates.
    color : int
        Packed RGBA color.
    thickness : float
        Line thickness in pixels.
    """
    vertices = np.asarray(points, dtype=np.float32)
    if len(vertices) < 2:
        return
    draw_list.AddPolyline(vertices, color, 0, thickness)


def draw_filled_polygon(draw_list: Any, points: np.ndarray, color: int, alpha: float) -> None:
    """Draw a filled, possibly concave polygon.

    Parameters
    ----------
    draw_list : ImDrawList
        Target draw list.
    points : np.ndarray
        Array of shape ``(M, 2)`` with the polygon vertices in order.
    color : int
        Packed RGBA color of the polygon edge.
    alpha : float
        Fill opacity in ``[0, 1]``.
    """
    vertices = np.asarray(points, dtype=np.float32)
    if len(vertices) < 3:
        return
    r, g, b, _ = psim.ColorConvertU32ToFloat4(color)
    fill = psim.GetColorU32((r, g, b, alpha))
    draw_list.AddConcavePolyFilled(vertices, fill)
    draw_list.AddPolyline(np.vstack((vertices, vertices[:1])), color, 0, 2.0)


def draw_rect_border(
    draw_list: Any,
    corner_a: tuple[float, float],
    corner_b: tuple[float, float],
    color: int,
    thickness: float = 2.0,
) -> None:
    """Draw the border of an axis-aligned rectangle.

    Parameters
    ----------
    draw_list : ImDrawList
        Target draw list.
    corner_a : tuple of float
        First corner in pixel coordinates.
    corner_b : tuple of float
        Opposite corner in pixel coordinates.
    color : int
        Packed RGBA color.
    thickness : float
        Border thickness in pixels.
    """
    draw_list.AddRect(
        _rect_min(corner_a, corner_b),
        _rect_max(corner_a, corner_b),
        color,
        0.0,
        0,
        thickness,
    )


def draw_rect_filled(
    draw_list: Any,
    corner_a: tuple[float, float],
    corner_b: tuple[float, float],
    color: int,
    alpha: float,
) -> None:
    """Draw a filled axis-aligned rectangle.

    Parameters
    ----------
    draw_list : ImDrawList
        Target draw list.
    corner_a : tuple of float
        First corner in pixel coordinates.
    corner_b : tuple of float
        Opposite corner in pixel coordinates.
    color : int
        Packed RGBA color of the border.
    alpha : float
        Fill opacity in ``[0, 1]``.
    """
    r, g, b, _ = psim.ColorConvertU32ToFloat4(color)
    fill = psim.GetColorU32((r, g, b, alpha))
    draw_list.AddRectFilled(
        _rect_min(corner_a, corner_b),
        _rect_max(corner_a, corner_b),
        fill,
    )


def _rect_min(corner_a: tuple[float, float], corner_b: tuple[float, float]) -> tuple[float, float]:
    """Return the lower-left corner of the rectangle."""
    return (min(corner_a[0], corner_b[0]), min(corner_a[1], corner_b[1]))


def _rect_max(corner_a: tuple[float, float], corner_b: tuple[float, float]) -> tuple[float, float]:
    """Return the upper-right corner of the rectangle."""
    return (max(corner_a[0], corner_b[0]), max(corner_a[1], corner_b[1]))
