"""Lasso (screen-space polygon) selection tool."""

from __future__ import annotations

from typing import Any

import numpy as np
import polyscope.imgui as psim

from pcd_segmentation.selection.base import SelectorTool, mode_from_modifiers
from pcd_segmentation.selection.geometry import points_in_polygon, project_to_screen
from pcd_segmentation.ui.drawing import col32, draw_closed_polyline, draw_filled_polygon

CLOSE_DISTANCE_PX: float = 10.0
"""Distance in pixels below which a click closes the polygon."""

MIN_VERTEX_SPACING_PX: float = 3.0
"""Minimum distance between consecutive polygon vertices."""


class LassoTool(SelectorTool):
    """Select points by drawing a polygon on the screen.

    Click to add vertices; the polygon is applied when it is closed by
    pressing Enter, double-clicking, or clicking near the first vertex.
    Escape cancels the polygon in progress.

    Attributes
    ----------
    vertices : list of tuple
        Screen-space vertices of the polygon being drawn, in pixels.
    """

    name = "lasso"

    def __init__(self, app: Any) -> None:
        super().__init__(app)
        self.vertices: list[tuple[float, float]] = []

    def handle_input(self, io: Any) -> None:
        """Consume one frame of ImGui input state.

        Clicks are ignored when they land on any ImGui item (e.g. the side
        panels): the user callback runs before the Polyscope UI is built, so
        ``io.WantCaptureMouse`` lags one frame and is not sufficient to
        filter UI clicks.
        """
        over_ui = psim.IsAnyItemHovered()
        mouse_valid = psim.IsMousePosValid()
        if mouse_valid and io.MouseClicked[0] and not io.WantCaptureMouse and not over_ui:
            position = tuple(io.MousePos)
            self._add_vertex(position)
            if self._closes_polygon(position):
                self._finish(io)
        if mouse_valid and io.MouseDoubleClicked[0] and len(self.vertices) >= 3 and not over_ui:
            self._finish(io)
        if psim.IsKeyPressed(psim.ImGuiKey_Enter) and len(self.vertices) >= 3:
            self._finish(io)
        if psim.IsKeyPressed(psim.ImGuiKey_Escape):
            self.cancel()

    def draw(self) -> None:
        """Draw the polygon in progress on top of the scene."""
        if not self.vertices:
            return
        draw_list = psim.GetForegroundDrawList()
        color = col32((1.0, 0.7, 0.1))
        outline = np.asarray(self.vertices, dtype=np.float32)
        if psim.IsMousePosValid():
            outline = np.vstack((outline, np.asarray(psim.GetMousePos(), dtype=np.float32)))
        draw_closed_polyline(draw_list, outline, color)
        if len(self.vertices) >= 3:
            draw_filled_polygon(draw_list, np.asarray(self.vertices, dtype=np.float32), color, 0.25)

    def cancel(self) -> None:
        """Discard the polygon in progress."""
        self.vertices.clear()

    def _add_vertex(self, position: tuple[float, float]) -> None:
        """Add a screen-space vertex, ignoring clicks too close to the last."""
        if self.vertices:
            last = self.vertices[-1]
            distance = np.hypot(position[0] - last[0], position[1] - last[1])
            if distance < MIN_VERTEX_SPACING_PX:
                return
        self.vertices.append(position)

    def _closes_polygon(self, position: tuple[float, float]) -> bool:
        """Return whether a click at this position closes the polygon."""
        if len(self.vertices) < 3:
            return False
        first = self.vertices[0]
        distance = np.hypot(position[0] - first[0], position[1] - first[1])
        return distance <= CLOSE_DISTANCE_PX

    def _finish(self, io: Any) -> None:
        """Apply the polygon selection and start a fresh one."""
        polygon = np.asarray(self.vertices, dtype=np.float64)
        self.cancel()
        view = self.app.view_state()
        if view is None or len(polygon) < 3:
            return
        visible = self.app.visible_points()
        if len(visible) == 0:
            return
        screen = project_to_screen(visible, view)
        mask = points_in_polygon(screen, polygon)
        mode = mode_from_modifiers(bool(io.KeyCtrl), bool(io.KeyShift))
        self.apply(mask, mode)
