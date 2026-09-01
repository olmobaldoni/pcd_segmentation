"""Tests for the screen-space projection and containment helpers."""

from __future__ import annotations

import numpy as np
from conftest import make_view

from pcd_segmentation.selection.geometry import (
    points_in_polygon,
    points_in_rect,
    project_to_screen,
)

# The test camera maps camera space to world space by adding (0, 0, 5).
WORLD_OFFSET = np.array([0.0, 0.0, 5.0])


def to_world(camera_points: np.ndarray) -> np.ndarray:
    """Convert camera-space points to world coordinates for the test camera."""
    return camera_points + WORLD_OFFSET


def test_perspective_projection(view):
    """Perspective projection maps camera-space offsets to screen quadrants."""
    assert np.allclose(
        project_to_screen(to_world(np.array([[0.0, 0.0, -5.0]])), view)[0],
        [400.0, 300.0],
    )
    camera_point = np.array([[view.aspect * 2.5, 2.5, -5.0]])
    assert np.allclose(project_to_screen(to_world(camera_point), view)[0], [600.0, 150.0])
    screen = project_to_screen(to_world(np.array([[1.0, 1.0, 10.0]])), view)
    assert np.all(np.isfinite(screen))


def test_orthographic_projection(view):
    """In orthographic mode the projection ignores depth."""
    ortho = make_view(orthographic=True)
    near_point = np.array([[view.aspect, 1.0, -100.0]])
    far_point = np.array([[view.aspect, 1.0, 100.0]])
    near_screen = project_to_screen(to_world(near_point), ortho)
    far_screen = project_to_screen(to_world(far_point), ortho)
    assert np.allclose(near_screen, far_screen)
    assert np.allclose(near_screen[0], [600.0, 150.0])


def test_points_in_polygon_square():
    """A square polygon contains exactly the points inside it."""
    square = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    points = np.array(
        [
            [5.0, 5.0],  # inside
            [20.0, 20.0],  # outside
            [-5.0, 5.0],  # outside
            [0.0, 0.0],  # on a corner
        ]
    )
    inside = points_in_polygon(points, square)
    assert inside.tolist() == [True, False, False, True]


def test_points_in_polygon_concave_and_degenerate():
    """Concave polygons exclude notch points; degenerate ones contain nothing."""
    polygon = np.array(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [10.0, 10.0],
            [5.0, 10.0],
            [5.0, 5.0],
            [0.0, 5.0],
        ]
    )
    points = np.array([[7.5, 7.5], [2.5, 7.5]])  # inside  # in the notch (outside)
    assert points_in_polygon(points, polygon).tolist() == [True, False]
    point = np.array([[0.0, 0.0]])
    assert not points_in_polygon(point, np.array([[0.0, 0.0], [1.0, 1.0]])).any()
    assert not points_in_polygon(point, np.empty((0, 2))).any()


def test_points_in_rect():
    """The rectangle test is inclusive on all borders."""
    points = np.array([[2.0, 2.0], [0.0, 0.0], [10.0, 10.0], [11.0, 5.0], [5.0, -1.0]])
    inside = points_in_rect(points, (0.0, 10.0), (10.0, 0.0))
    assert inside.tolist() == [True, True, True, False, False]
