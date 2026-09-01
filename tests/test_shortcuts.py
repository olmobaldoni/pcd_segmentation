"""Tests for keyboard shortcuts and tool input handling."""

from __future__ import annotations

import numpy as np
import polyscope.imgui as psim
import pytest

from pcd_segmentation.app import App
from pcd_segmentation.core.segments import SegmentManager


class FakeIO:
    """Minimal stand-in for the ImGuiIO object."""

    def __init__(
        self,
        mouse_pos: tuple[float, float] = (100.0, 100.0),
        clicked: bool = False,
        want_capture_mouse: bool = False,
        want_text: bool = False,
        ctrl: bool = False,
    ) -> None:
        self.MousePos = mouse_pos
        self.MouseClicked = [clicked, False, False]
        self.MouseDoubleClicked = [False, False, False]
        self.MouseReleased = [False, False, False]
        self.WantCaptureMouse = want_capture_mouse
        self.WantTextInput = want_text
        self.KeyCtrl = ctrl
        self.KeyShift = False


def make_app(
    monkeypatch: pytest.MonkeyPatch,
    pressed_keys: tuple[int, ...] = (),
    io: FakeIO | None = None,
) -> tuple[App, FakeIO]:
    """Build an App with fake input state and a manager for 20 points.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest monkeypatch fixture.
    pressed_keys : tuple of int
        ImGuiKey values reported as pressed this frame.
    io : FakeIO or None
        Input state to feed the callback; a default one is created.

    Returns
    -------
    tuple of (App, FakeIO)
        The prepared application and the fake input state.
    """
    app = App()
    app.manager = SegmentManager(20)
    fake_io = io if io is not None else FakeIO()
    monkeypatch.setattr("pcd_segmentation.app.psim.GetIO", lambda: fake_io)
    pressed = set(pressed_keys)
    monkeypatch.setattr("pcd_segmentation.app.psim.IsKeyPressed", lambda key: key in pressed)
    monkeypatch.setattr("pcd_segmentation.app.ps.set_do_default_mouse_interaction", lambda _: None)
    monkeypatch.setattr("pcd_segmentation.app.ps.unshow", lambda: None)
    return app, fake_io


def test_undo_redo_shortcuts(monkeypatch):
    """Ctrl+Z/``u`` undo and Ctrl+R/Ctrl+Y redo the last operation."""
    for key, ctrl, expected in [
        (psim.ImGuiKey_Z, True, 0),
        (psim.ImGuiKey_U, False, 0),
    ]:
        app, _ = make_app(monkeypatch, (key,), FakeIO(ctrl=ctrl))
        seg_id = app.manager.create_segment("a", (1.0, 0.0, 0.0))
        app.manager.assign(seg_id, np.arange(5))
        app._handle_shortcuts()
        assert app.manager.count(seg_id) == expected
    for key in (psim.ImGuiKey_R, psim.ImGuiKey_Y):
        app, _ = make_app(monkeypatch, (key,), FakeIO(ctrl=True))
        seg_id = app.manager.create_segment("a", (1.0, 0.0, 0.0))
        app.manager.assign(seg_id, np.arange(5))
        app.manager.undo()
        app._handle_shortcuts()
        assert app.manager.count(seg_id) == 5


def test_tool_keys_switch_tools(monkeypatch):
    """The single-letter tool keys activate and deactivate tools."""
    for key, tool in [
        (psim.ImGuiKey_V, "lasso"),
        (psim.ImGuiKey_B, "box"),
        (psim.ImGuiKey_Escape, "none"),
    ]:
        app, _ = make_app(monkeypatch, (key,))
        app._handle_shortcuts()
        assert app.active_tool_name == tool


def test_selection_action_keys(monkeypatch):
    """``c``/``s`` assign the selection; ``x`` clears and ``i`` inverts it."""
    for key in (psim.ImGuiKey_C, psim.ImGuiKey_S):
        app, _ = make_app(monkeypatch, (key,))
        app.selection = np.arange(5)
        app._handle_shortcuts()
        assert len(app.manager.segments) == 1
        assert app.manager.count(app.manager.segments[0].id) == 5
    app, _ = make_app(monkeypatch, (psim.ImGuiKey_X,))
    app.selection = np.arange(5)
    app._handle_shortcuts()
    assert len(app.selection) == 0
    app, _ = make_app(monkeypatch, (psim.ImGuiKey_I,))
    app.selection = np.arange(5)
    app._handle_shortcuts()
    assert len(app.selection) == 15


def test_quit_key_closes_window(monkeypatch):
    """``q`` requests the viewer to close."""
    app, _ = make_app(monkeypatch, (psim.ImGuiKey_Q,))
    quit_calls: list[int] = []
    monkeypatch.setattr("pcd_segmentation.app.ps.unshow", lambda: quit_calls.append(1))
    app._handle_shortcuts()
    assert quit_calls == [1]


def test_shortcuts_ignored_while_typing(monkeypatch):
    """No shortcut fires while a text field wants keyboard input."""
    io = FakeIO(want_text=True, ctrl=True)
    app, _ = make_app(monkeypatch, (psim.ImGuiKey_Z,), io)
    seg_id = app.manager.create_segment("a", (1.0, 0.0, 0.0))
    app.manager.assign(seg_id, np.arange(5))
    app._handle_shortcuts()
    assert app.manager.count(seg_id) == 5
    assert app.active_tool_name == "none"


def test_lasso_click_guards(monkeypatch):
    """Lasso clicks add vertices only on empty, unclaimed screen space."""
    for want_capture_mouse, hover_ui, expected in [
        (False, False, [(120.0, 240.0)]),
        (False, True, []),
        (True, False, []),
    ]:
        app, _ = make_app(monkeypatch)
        app.set_active_tool("lasso")
        monkeypatch.setattr("pcd_segmentation.app.psim.IsMousePosValid", lambda: True)
        monkeypatch.setattr(
            "pcd_segmentation.app.psim.IsAnyItemHovered",
            lambda hover_ui=hover_ui: hover_ui,
        )
        io = FakeIO(
            mouse_pos=(120.0, 240.0),
            clicked=True,
            want_capture_mouse=want_capture_mouse,
        )
        app._tools["lasso"].handle_input(io)
        assert app._tools["lasso"].vertices == expected


def test_box_drag_start_on_ui_is_ignored(monkeypatch):
    """Box drags cannot start on ImGui items."""
    app, _ = make_app(monkeypatch)
    app.set_active_tool("box")
    monkeypatch.setattr("pcd_segmentation.app.psim.IsMousePosValid", lambda: True)
    monkeypatch.setattr("pcd_segmentation.app.psim.IsAnyItemHovered", lambda: True)
    tool = app._tools["box"]
    tool.handle_input(FakeIO(mouse_pos=(120.0, 240.0), clicked=True))
    assert not tool.dragging


class FakePick:
    """Stand-in for the polyscope PickResult."""

    def __init__(self, is_hit: bool = True) -> None:
        self.is_hit = is_hit
        self.position = np.array([1.0, 2.0, 3.0])


def test_double_right_click_moves_gizmo_to_point(monkeypatch):
    """A double right-click places the gizmo center on the picked point."""
    from pcd_segmentation.core.model import PointCloudData

    io = FakeIO(mouse_pos=(120.0, 240.0))
    io.MouseDoubleClicked = [False, True, False]
    app, _ = make_app(monkeypatch, io=io)
    app.cloud = PointCloudData(points=np.zeros((5, 3)))
    view_calls: list[object] = []
    monkeypatch.setattr("pcd_segmentation.app.psim.IsMousePosValid", lambda: True)
    monkeypatch.setattr("pcd_segmentation.app.psim.IsAnyItemHovered", lambda: False)
    monkeypatch.setattr("pcd_segmentation.app.ps.pick",
                        lambda screen_coords=None, buffer_inds=None: FakePick())
    monkeypatch.setattr("pcd_segmentation.app.ps.set_view_center_and_look_at",
                        lambda pos, fly_to=False: view_calls.append(pos))
    app._handle_recenter_gesture()
    assert app.gizmo_enabled
    assert np.allclose(app._gizmo_base[:3, 3], [1.0, 2.0, 3.0])
    assert view_calls == []  # the view must not be recentered


def test_double_right_click_noop_cases(monkeypatch):
    """The gesture is ignored without a hit, over UI, or with no cloud."""
    no_hit = FakeIO(mouse_pos=(120.0, 240.0))
    no_hit.MouseDoubleClicked = [False, True, False]
    app, _ = make_app(monkeypatch, io=no_hit)
    monkeypatch.setattr("pcd_segmentation.app.psim.IsMousePosValid", lambda: True)
    monkeypatch.setattr("pcd_segmentation.app.psim.IsAnyItemHovered", lambda: False)
    monkeypatch.setattr("pcd_segmentation.app.ps.pick",
                        lambda screen_coords=None, buffer_inds=None: FakePick(is_hit=False))
    app._handle_recenter_gesture()
    assert np.allclose(app._gizmo_base, np.eye(4))

    hover = FakeIO(mouse_pos=(120.0, 240.0))
    hover.MouseDoubleClicked = [False, True, False]
    app, _ = make_app(monkeypatch, io=hover)
    monkeypatch.setattr("pcd_segmentation.app.psim.IsAnyItemHovered", lambda: True)
    monkeypatch.setattr("pcd_segmentation.app.ps.pick",
                        lambda screen_coords=None, buffer_inds=None: FakePick())
    app._handle_recenter_gesture()
    assert np.allclose(app._gizmo_base, np.eye(4))

    no_cloud = FakeIO(mouse_pos=(120.0, 240.0))
    no_cloud.MouseDoubleClicked = [False, True, False]
    app, _ = make_app(monkeypatch, io=no_cloud)
    monkeypatch.setattr("pcd_segmentation.app.psim.IsAnyItemHovered", lambda: False)
    monkeypatch.setattr("pcd_segmentation.app.ps.pick",
                        lambda screen_coords=None, buffer_inds=None: FakePick())
    app._handle_recenter_gesture()
    assert np.allclose(app._gizmo_base, np.eye(4))
