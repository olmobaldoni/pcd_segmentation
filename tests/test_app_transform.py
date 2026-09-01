"""Tests for the transformation gizmo integration in the application."""

from __future__ import annotations

import numpy as np
import pytest

from pcd_segmentation.app import App
from pcd_segmentation.core import transform
from pcd_segmentation.core.model import PointCloudData
from pcd_segmentation.core.segments import SegmentManager


class FakeGizmo:
    """Stand-in for the polyscope TransformationGizmo."""

    def __init__(self) -> None:
        self.enabled = False
        self.matrix = np.eye(4, dtype=np.float64)

    def set_enabled(self, enabled: bool) -> None:
        """Record the widget enabled state."""
        self.enabled = enabled

    def get_enabled(self) -> bool:
        """Return the widget enabled state."""
        return self.enabled

    def set_transform(self, matrix: np.ndarray) -> None:
        """Record the widget matrix."""
        self.matrix = np.asarray(matrix, dtype=np.float64).copy()

    def get_transform(self) -> np.ndarray:
        """Return a copy of the widget matrix."""
        return self.matrix.copy()


def make_app(monkeypatch: pytest.MonkeyPatch) -> tuple[App, FakeGizmo]:
    """Build an App with a cloud, a manager, and a fake gizmo.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest monkeypatch fixture.

    Returns
    -------
    tuple of (App, FakeGizmo)
        The application and the fake gizmo attached to it.
    """
    app = App()
    rng = np.random.default_rng(0)
    points = rng.uniform(0.0, 10.0, size=(50, 3))
    app.cloud = PointCloudData(points=points)
    app.manager = SegmentManager(50)
    app._gizmo_base = transform.translation_matrix(points)
    gizmo = FakeGizmo()
    app._gizmo = gizmo
    monkeypatch.setattr("pcd_segmentation.app.ps.set_do_default_mouse_interaction", lambda _: None)
    return app, gizmo


def test_gizmo_enable_disable(monkeypatch):
    """Enabling shows the widget and exits selection mode; disabling hides it."""
    app, gizmo = make_app(monkeypatch)
    app.set_active_tool("lasso")
    app.set_gizmo_enabled(True)
    assert app.gizmo_enabled
    assert app.active_tool_name == "none"
    assert gizmo.get_enabled()
    assert np.allclose(gizmo.get_transform(), app._gizmo_base)
    app.set_gizmo_enabled(False)
    assert not app.gizmo_enabled
    assert not gizmo.get_enabled()


def test_poll_gizmo_updates_transform(monkeypatch):
    """Dragging the gizmo updates the transform; idle polling is a no-op."""
    app, gizmo = make_app(monkeypatch)
    app.set_gizmo_enabled(True)
    app._dirty = False
    app._poll_gizmo()
    assert not app._dirty
    object_matrix = np.eye(4)
    object_matrix[:3, 3] = [1.0, 2.0, 3.0]
    gizmo.set_transform(app._gizmo_base @ object_matrix)
    app._poll_gizmo()
    assert np.allclose(app.transform, object_matrix)
    assert app._dirty


def test_poll_gizmo_rotates_about_gizmo_center(monkeypatch):
    """Rotating the gizmo about its center pivots the cloud about that point."""
    app, gizmo = make_app(monkeypatch)
    pivot = np.array([3.0, 4.0, 5.0])
    app.set_gizmo_enabled(True)
    app._set_gizmo_center(pivot)
    app._sync_gizmo_transform()

    angle = np.pi / 2.0
    rot = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rot4 = np.eye(4)
    rot4[:3, :3] = rot
    base = np.eye(4)
    base[:3, 3] = pivot
    gizmo.set_transform(base @ rot4)  # ImGuizmo keeps the center fixed on rotation
    app._poll_gizmo()

    expected = base @ rot4 @ np.linalg.inv(base)
    assert np.allclose(app.transform, expected)
    object_pivot = np.linalg.inv(app.transform) @ np.array([*pivot, 1.0])
    moved = transform.apply_transform(object_pivot[:3][None, :], app.transform)[0]
    assert np.allclose(moved, pivot)


def test_poll_gizmo_disabled_is_noop(monkeypatch):
    """A disabled gizmo is never polled."""
    app, gizmo = make_app(monkeypatch)
    object_matrix = np.eye(4)
    object_matrix[:3, 3] = [9.0, 9.0, 9.0]
    gizmo.set_transform(app._gizmo_base @ object_matrix)
    app._poll_gizmo()
    assert np.allclose(app.transform, np.eye(4))


def test_apply_transform_bakes_points(monkeypatch):
    """Applying bakes the transform into the points and resets the gizmo."""
    app, gizmo = make_app(monkeypatch)
    original = app.cloud.points.copy()
    app.transform[:3, 3] = [1.0, 2.0, 3.0]
    app.apply_transform_to_cloud()
    assert np.allclose(app.cloud.points, original + [1.0, 2.0, 3.0])
    assert np.allclose(app.transform, np.eye(4))
    assert app._dirty
    assert np.allclose(gizmo.get_transform(), app._gizmo_base)


def test_apply_transform_identity_is_noop(monkeypatch):
    """Baking an identity transform leaves the cloud untouched."""
    app, gizmo = make_app(monkeypatch)
    original = app.cloud.points.copy()
    app.apply_transform_to_cloud()
    assert np.array_equal(app.cloud.points, original)
    assert "identity" in app.status_text


def test_reset_transform(monkeypatch):
    """Reset clears the accumulated transform."""
    app, gizmo = make_app(monkeypatch)
    app.transform[:3, 3] = [4.0, 5.0, 6.0]
    app.reset_transform()
    assert np.allclose(app.transform, np.eye(4))
    assert app._dirty


def test_selecting_tool_disables_gizmo(monkeypatch):
    """Entering a selection mode turns the gizmo off."""
    app, gizmo = make_app(monkeypatch)
    app.set_gizmo_enabled(True)
    app.set_active_tool("box")
    assert not app.gizmo_enabled
    assert not gizmo.get_enabled()
    assert app.active_tool_name == "box"


def test_visible_points_include_transform(monkeypatch):
    """The world coordinates returned for selection use the transform."""
    app, gizmo = make_app(monkeypatch)
    app.transform[:3, 3] = [10.0, 0.0, 0.0]
    assert np.allclose(app.visible_points(), app.cloud.points + [10.0, 0.0, 0.0])


def test_visible_colors(monkeypatch):
    """Unassigned points use their original colors by default."""
    from pcd_segmentation import config

    app, gizmo = make_app(monkeypatch)
    original = np.random.default_rng(0).uniform(0.0, 1.0, size=(50, 3)).astype(np.float32)
    app.cloud.colors = original
    assert np.allclose(app._visible_colors(np.arange(10)), original[:10])
    app.show_original_colors = False
    expected_gray = np.tile(config.UNASSIGNED_COLOR, (10, 1))
    assert np.allclose(app._visible_colors(np.arange(10)), expected_gray)
