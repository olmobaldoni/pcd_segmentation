"""Drag-rectangle selection tool."""

from __future__ import annotations

from typing import Any

import polyscope.imgui as psim

from pcd_segmentation.selection.base import SelectorTool, mode_from_modifiers
from pcd_segmentation.selection.geometry import points_in_rect, project_to_screen
from pcd_segmentation.ui.drawing import col32, draw_rect_border, draw_rect_filled

MIN_DRAG_PX: float = 4.0
"""Drags smaller than this threshold (in pixels) are treated as clicks."""


class BoxTool(SelectorTool):
    """Select points by dragging a rectangle on the screen.

    All visible points whose projection falls inside the rectangle are
    selected on mouse release. Escape cancels the drag in progress.

    Attributes
    ----------
    drag_start : tuple of float or None
        Screen position where the drag started, in pixels.
    drag_end : tuple of float or None
        Current (or final) screen position of the drag.
    """

    name = "box"

    def __init__(self, app: Any) -> None:
        super().__init__(app)
        self.drag_start: tuple[float, float] | None = None
        self.drag_end: tuple[float, float] | None = None

    @property
    def dragging(self) -> bool:
        """Return whether a drag is currently in progress."""
        return self.drag_start is not None

    def handle_input(self, io: Any) -> None:
        """Consume one frame of ImGui input state."""
        if not psim.IsMousePosValid():
            return
        position = tuple(io.MousePos)
        if not self.dragging:
            if io.MouseClicked[0] and not io.WantCaptureMouse and not psim.IsAnyItemHovered():
                self.drag_start = position
                self.drag_end = position
        else:
            self.drag_end = position
            if io.MouseReleased[0]:
                self._finish(io)
        if psim.IsKeyPressed(psim.ImGuiKey_Escape):
            self.cancel()

    def draw(self) -> None:
        """Draw the drag rectangle on top of the scene."""
        if self.drag_start is None:
            return
        draw_list = psim.GetForegroundDrawList()
        color = col32((0.2, 0.8, 1.0))
        end = self.drag_end or self.drag_start
        draw_rect_filled(draw_list, self.drag_start, end, color, 0.15)
        draw_rect_border(draw_list, self.drag_start, end, color)

    def cancel(self) -> None:
        """Abort the drag in progress."""
        self.drag_start = None
        self.drag_end = None

    def _finish(self, io: Any) -> None:
        """Apply the rectangle selection on mouse release."""
        start = self.drag_start
        end = self.drag_end
        self.cancel()
        if start is None or end is None:
            return
        width = abs(end[0] - start[0])
        height = abs(end[1] - start[1])
        if width < MIN_DRAG_PX and height < MIN_DRAG_PX:
            return
        view = self.app.view_state()
        if view is None:
            return
        visible = self.app.visible_points()
        if len(visible) == 0:
            return
        screen = project_to_screen(visible, view)
        mask = points_in_rect(screen, start, end)
        mode = mode_from_modifiers(bool(io.KeyCtrl), bool(io.KeyShift))
        self.apply(mask, mode)
