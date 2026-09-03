"""Tests for the reference-point application logic."""

from __future__ import annotations

import json

import numpy as np

from pcd_segmentation.core.model import PointCloudData
from pcd_segmentation.reference import conventions as conv
from pcd_segmentation.reference.app import CANDIDATE_RADIUS_SCALE, CLOUD_NAME, ReferenceApp
from pcd_segmentation.selection.geometry import ViewState

_MOUSE_PATCH = "pcd_segmentation.reference.app.ps.set_do_default_mouse_interaction"


def make_view() -> ViewState:
    """Return a synthetic perspective camera at (0, 0, 5) over 800x600."""
    extrinsics = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, -5.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return ViewState(
        extrinsics=extrinsics,
        fov_deg=90.0,
        aspect=800.0 / 600.0,
        width=800,
        height=600,
        orthographic=False,
        length_scale=1.0,
    )


def build_app() -> tuple[ReferenceApp, np.ndarray]:
    """Return an app with a synthetic cloud and its base coordinates."""
    app = ReferenceApp()
    rng = np.random.default_rng(0)
    points = rng.uniform(0.0, 10.0, size=(40, 3))
    cloud = PointCloudData(points=points)
    app._attach(cloud)
    return app, points


def test_attach_resets_state():
    """Attaching a cloud keeps its points and resets the mapping."""
    app, points = build_app()
    assert np.array_equal(app.cloud.points, points)
    assert np.allclose(app._axes, np.eye(3))
    assert np.array_equal(app._translation, np.zeros(3))
    assert app.origin_index is None
    assert not app.can_undo()


def test_origin_is_translation_to_zero(monkeypatch):
    """Confirming a candidate translates the cloud so the point lands at 0."""
    app, points = build_app()
    monkeypatch.setattr(_MOUSE_PATCH, lambda _: None)
    app.set_candidate(3)
    app.set_origin_from_candidate()
    assert np.allclose(app.cloud.points[3], np.zeros(3))
    assert np.allclose(app.cloud.points, points - points[3])
    assert app.origin_index == 3
    assert "0, 0, 0" in app.origin_text()


def test_origin_without_candidate_reports_status():
    """Setting the origin without a picked point shows a message."""
    app, _ = build_app()
    app.set_origin_from_candidate()
    assert "Pick a point" in app.status_text
    assert app.origin_index is None


def test_undo_redo_origin_round_trip(monkeypatch):
    """Undo restores the base coordinates and redo re-applies the origin."""
    app, points = build_app()
    monkeypatch.setattr(_MOUSE_PATCH, lambda _: None)
    app.set_candidate(3)
    app.set_origin_from_candidate()
    app.undo()
    assert np.array_equal(app.cloud.points, points)
    assert app.origin_index is None
    app.redo()
    assert np.allclose(app.cloud.points[3], np.zeros(3))


def test_apply_axes_converts_points(monkeypatch):
    """Applying RDF -> RUB negates the Y and Z coordinates."""
    app, points = build_app()
    monkeypatch.setattr(_MOUSE_PATCH, lambda _: None)
    app.source_convention = "RDF"
    app.target_convention = "RUB"
    app.apply_axes()
    expected = conv.convert_points(points, "RDF", "RUB")
    assert np.allclose(app.cloud.points, expected)
    app.undo()
    assert np.array_equal(app.cloud.points, points)


def test_apply_axes_guards(monkeypatch):
    """Identity or undefined conversions are rejected with a message."""
    app, _ = build_app()
    monkeypatch.setattr(_MOUSE_PATCH, lambda _: None)
    app.source_convention = conv.NONE_TRIPLE
    app.target_convention = "RDF"
    app.apply_axes()
    assert "two real conventions" in app.status_text
    app.source_convention = "RDF"
    app.target_convention = "RDF"
    app.apply_axes()
    assert "identical" in app.status_text


def test_axes_and_origin_compose(monkeypatch):
    """Axes conversion keeps the origin at (0, 0, 0)."""
    app, points = build_app()
    monkeypatch.setattr(_MOUSE_PATCH, lambda _: None)
    app.set_candidate(3)
    app.set_origin_from_candidate()
    app.source_convention = "RDF"
    app.target_convention = "RUB"
    app.apply_axes()
    assert np.allclose(app.cloud.points[3], np.zeros(3))
    origin_before = points[3]
    expected = conv.convert_points(points - origin_before, "RDF", "RUB")
    assert np.allclose(app.cloud.points, expected)


def test_clear_candidate():
    """Clearing the candidate removes the marker request."""
    app, _ = build_app()
    app.set_candidate(0)
    assert app.candidate_index == 0
    app.clear_candidate()
    assert app.candidate_index is None


def test_set_candidate_out_of_range():
    """Out-of-range candidate indices are rejected."""
    app, _ = build_app()
    app.set_candidate(999)
    assert app.candidate_index is None
    assert "out of range" in app.status_text


def test_pick_candidate_from_screen():
    """The point closest to the cursor within tolerance is picked."""
    app, _ = build_app()
    center = np.array([0.0, 0.0, 0.0])
    cloud = PointCloudData(points=np.vstack((center, np.ones((5, 3)) * 20.0)))
    app._attach(cloud)
    app._view_state = lambda: make_view()
    index = app._candidate_from_position((400.0, 300.0))
    assert index == 0
    assert app._candidate_from_position((5.0, 5.0)) is None


def test_toggle_pick_requires_cloud(monkeypatch):
    """Pick mode cannot be enabled before loading a cloud."""
    app = ReferenceApp()
    monkeypatch.setattr(_MOUSE_PATCH, lambda _: None)
    app.set_pick_active(True)
    assert not app.pick_active
    assert "Load a cloud" in app.status_text


def test_set_pick_active_toggles_mouse_interaction(monkeypatch):
    """Enabling pick disables the default mouse interaction and vice versa."""
    app, _ = build_app()
    calls: list[bool] = []
    monkeypatch.setattr(
        "pcd_segmentation.reference.app.ps.set_do_default_mouse_interaction",
        lambda enabled: calls.append(enabled),
    )
    app.set_pick_active(True)
    assert app.pick_active
    assert calls == [False]
    app.set_pick_active(False)
    assert not app.pick_active
    assert calls == [False, True]


def test_default_export_path_naming():
    """The default output is named after the source cloud stem."""
    app, _ = build_app()
    app.cloud.source = None
    assert str(app.default_export_path()) == "cloud_referenced.ply"


def test_export_current_writes_files(tmp_path):
    """Exporting writes the PLY and the sidecar JSON."""
    app, points = build_app()
    app.cloud.colors = np.zeros((len(points), 3), dtype=np.float32)
    app.export_path_text = str(tmp_path / "out.ply")
    app.set_candidate(1)
    app.set_origin_from_candidate()
    app.export_current()
    assert (tmp_path / "out.ply").is_file()
    sidecar = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    assert sidecar["reference"]["index"] == 1
    assert np.allclose(app.cloud.points, points - points[1])


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


def test_enable_gizmo_requires_cloud():
    """The gizmo cannot be enabled before loading a cloud."""
    app = ReferenceApp()
    app.set_gizmo_enabled(True)
    assert not app.gizmo_enabled
    assert "Load a cloud" in app.status_text


def test_enabling_gizmo_switches_pick_off(monkeypatch):
    """Enabling the gizmo turns the pick mode off and locks it."""
    app, _ = build_app()
    monkeypatch.setattr(_MOUSE_PATCH, lambda _: None)
    app.set_pick_active(True)
    app.set_gizmo_enabled(True)
    assert app.gizmo_enabled
    assert not app.pick_active
    app.set_pick_active(True)
    assert not app.pick_active
    assert "gizmo" in app.status_text


def test_gizmo_pivot_is_origin_or_centroid():
    """The gizmo pivots on the origin point when set, else the centroid."""
    app, points = build_app()
    assert np.allclose(app._gizmo_pivot(), points.mean(axis=0))
    app.set_candidate(5)
    app.set_origin_from_candidate()
    assert np.allclose(app._gizmo_pivot(), np.zeros(3))


def test_apply_gizmo_translation_bakes(monkeypatch):
    """Baking a gizmo translation moves the points and is undoable."""
    app, points = build_app()
    monkeypatch.setattr(_MOUSE_PATCH, lambda _: None)
    app.transform[:3, 3] = [1.0, 2.0, 3.0]
    app.apply_gizmo_to_cloud()
    assert np.allclose(app.cloud.points, points + [1.0, 2.0, 3.0])
    assert np.allclose(app.transform, np.eye(4))
    assert np.allclose(app._translation, [1.0, 2.0, 3.0])
    app.undo()
    assert np.array_equal(app.cloud.points, points)


def test_apply_identity_gizmo_is_noop(monkeypatch):
    """Baking an identity gizmo transform leaves the cloud untouched."""
    app, points = build_app()
    monkeypatch.setattr(_MOUSE_PATCH, lambda _: None)
    app.apply_gizmo_to_cloud()
    assert np.array_equal(app.cloud.points, points)
    assert "identity" in app.status_text


def test_poll_gizmo_updates_transform(monkeypatch):
    """Dragging the gizmo updates the transform; idle polling is a no-op."""
    app, _ = build_app()
    monkeypatch.setattr(_MOUSE_PATCH, lambda _: None)
    gizmo = FakeGizmo()
    app._gizmo = gizmo
    app.set_gizmo_enabled(True)
    app._transform_dirty = False
    app._poll_gizmo()
    assert not app._transform_dirty
    object_matrix = np.eye(4)
    object_matrix[:3, 3] = [1.0, 2.0, 3.0]
    gizmo.set_transform(app._gizmo_base @ object_matrix)
    app._poll_gizmo()
    assert np.allclose(app.transform, object_matrix)
    assert app._transform_dirty


def test_gizmo_blocks_axes_and_origin(monkeypatch):
    """Origin, axes and export edits are refused while the gizmo is active."""
    app, _ = build_app()
    monkeypatch.setattr(_MOUSE_PATCH, lambda _: None)
    app.set_gizmo_enabled(True)
    app.source_convention = "RDF"
    app.target_convention = "RUB"
    app.apply_axes()
    assert "Apply or reset" in app.status_text
    app.set_candidate(0)
    app.set_origin_from_candidate()
    assert "Apply or reset" in app.status_text


def test_pending_transform_blocks_editing(monkeypatch):
    """A gizmo motion not yet applied blocks edits even when hidden."""
    app, _ = build_app()
    monkeypatch.setattr(_MOUSE_PATCH, lambda _: None)
    app.transform[:3, 3] = [1.0, 0.0, 0.0]
    assert app.transform_pending
    app.set_pick_active(True)
    assert not app.pick_active
    assert "Apply or reset" in app.status_text
    app.apply_axes()
    assert "Apply or reset" in app.status_text
    app.reset_gizmo()
    assert not app.transform_pending


class FakeCloud:
    """Stand-in for a polyscope point cloud exposing a radius."""

    def __init__(self, radius: float) -> None:
        self.radius = radius

    def get_radius(self) -> float:
        """Return the stored radius."""
        return self.radius

    def set_radius(self, radius: float, relative: bool = True) -> None:
        """Store the given radius."""
        self.radius = float(radius)


def test_candidate_radius_scale_constant():
    """The candidate marker is 1.5x the cloud-point radius."""
    assert CANDIDATE_RADIUS_SCALE == 1.5


def test_candidate_radius_follows_cloud(monkeypatch):
    """The candidate radius is the scale times the current cloud radius."""
    app, _ = build_app()
    cloud = FakeCloud(0.0001)
    monkeypatch.setattr("pcd_segmentation.reference.app.ps.has_point_cloud", lambda name: True)
    monkeypatch.setattr("pcd_segmentation.reference.app.ps.get_point_cloud", lambda name: cloud)
    assert np.isclose(app._candidate_radius(), 0.00015)
    cloud.radius = 0.0002
    assert np.isclose(app._candidate_radius(), 0.0003)


def test_candidate_radius_zero_without_cloud(monkeypatch):
    """Without a registered cloud the candidate radius is zero."""
    app, _ = build_app()
    monkeypatch.setattr("pcd_segmentation.reference.app.ps.has_point_cloud", lambda name: False)
    assert app._candidate_radius() == 0.0


def test_sync_candidate_radius_tracks_cloud_changes(monkeypatch):
    """Changing the cloud radius updates the candidate marker radius."""
    app, _ = build_app()
    cloud = FakeCloud(0.0001)
    candidate = FakeCloud(0.00015)
    monkeypatch.setattr("pcd_segmentation.reference.app.ps.has_point_cloud", lambda name: True)
    monkeypatch.setattr(
        "pcd_segmentation.reference.app.ps.get_point_cloud",
        lambda name: cloud if name == CLOUD_NAME else candidate,
    )
    app.cloud = PointCloudData(points=np.zeros((5, 3)))
    app._sync_candidate_radius()
    assert np.isclose(candidate.radius, 0.00015)
    cloud.radius = 0.0002
    app._sync_candidate_radius()
    assert np.isclose(candidate.radius, 0.0003)


def test_sync_candidate_radius_without_candidate_is_noop(monkeypatch):
    """No candidate marker means the sync does nothing."""
    app, _ = build_app()
    monkeypatch.setattr("pcd_segmentation.reference.app.ps.has_point_cloud", lambda name: False)
    app._sync_candidate_radius()
