"""Lasso (screen-space polygon) selection tool."""

from __future__ import annotations

from typing import Any

import numpy as np
import polyscope.imgui as psim

from pcd_segmentation.selection.base import SelectorTool, mode_from_modifiers
from pcd_segmentation.selection.geometry import (
    is_simple_polygon,
    points_in_polygon,
    project_to_screen,
)
from pcd_segmentation.ui.drawing import (
    col32,
    draw_closed_polyline,
    draw_filled_polygon,
    draw_open_polyline,
)

CLOSE_DISTANCE_PX: float = 10.0
"""Distance in pixels below which a click closes the polygon."""

MIN_VERTEX_SPACING_PX: float = 3.0
"""Minimum distance between consecutive polygon vertices."""

NODE_SIZE_PX: float = 4.0
"""Half-size of the square node drawn at each polygon vertex."""

NODE_COLOR: tuple[float, float, float] = (1.0, 0.95, 0.6)
"""Color of a regular polygon node."""

CLOSE_NODE_COLOR: tuple[float, float, float] = (0.3, 1.0, 0.4)
"""Color of the first node, the target that closes the polygon."""


class LassoTool(SelectorTool):
    """Select points by drawing a polygon on the screen.

    Click to add vertices; every vertex is drawn as a small node. The polygon
    is applied when it is closed by clicking on (or near) the first node,
    pressing Enter, or double-clicking. Escape cancels the polygon.

    Attributes
    ----------
    vertices : list of tuple
        Screen-space vertices of the polygon being drawn, in pixels.
    """

    name = "lasso"

    def __init__(self, app: Any) -> None:
        super().__init__(app)
        self.vertices: list[tuple[float, float]] = []
        self._pending_click: tuple[float, float] | None = None

    def handle_input(self, io: Any) -> None:
        """Consume one frame of ImGui input state.

        Clicks are accepted one frame late: the user callback runs before the
        Polyscope UI is built, so ``io.WantCaptureMouse`` lags a frame. A
        click that lands on a side panel is therefore only discarded once the
        up-to-date capture flag is available, which prevents stray vertices
        from appearing at the panel (e.g. the top-left corner of the scene).
        """
        over_ui = psim.IsAnyItemHovered()
        mouse_valid = psim.IsMousePosValid()
        self._accept_pending_click(io, over_ui)
        if mouse_valid and io.MouseClicked[0] and not io.WantCaptureMouse and not over_ui:
            position = self._click_position(io)
            if position is not None:
                self._pending_click = position
        if mouse_valid and io.MouseDoubleClicked[0] and len(self.vertices) >= 3 and not over_ui:
            self._finish(io)
        if psim.IsKeyPressed(psim.ImGuiKey_Enter) and len(self.vertices) >= 3:
            self._finish(io)
        if psim.IsKeyPressed(psim.ImGuiKey_Escape):
            self.cancel()

    def _accept_pending_click(self, io: Any, over_ui: bool) -> None:
        """Accept a click queued on the previous frame if it is still valid.

        Parameters
        ----------
        io : Any
            Current frame ImGui input state.
        over_ui : bool
            Whether the cursor is hovering an ImGui item this frame.
        """
        if self._pending_click is None:
            return
        position = self._pending_click
        self._pending_click = None
        if io.WantCaptureMouse or over_ui:
            return
        if self._closes_polygon(position):
            self._finish(io)
        else:
            self._add_vertex(position)

    def _click_position(self, io: Any) -> tuple[float, float] | None:
        """Return the validated screen position of the current click.

        The current cursor position is used (it is kept up to date by the
        backend on every frame and equals the click location on the click
        frame). Positions that are not finite or fall outside the window are
        rejected.

        Parameters
        ----------
        io : Any
            Current frame ImGui input state.

        Returns
        -------
        tuple of float or None
            The click position, or ``None`` when it is not usable.
        """
        position = tuple(io.MousePos)
        if not np.all(np.isfinite(position)):
            return None
        width, height = io.DisplaySize
        if not (0.0 <= position[0] <= width and 0.0 <= position[1] <= height):
            return None
        return position

    def draw(self) -> None:
        """Draw the polygon in progress, its nodes, and the close target.

        The polyline is drawn as an open path following the cursor, so it
        never snaps to a far-away or side-panel position by closing back to
        the first vertex. The translucent fill is drawn only when the polygon
        is simple (non self-intersecting): self-intersecting lassos render as
        an outline only, since their fill is undefined.
        """
        if not self.vertices:
            return
        draw_list = psim.GetForegroundDrawList()
        color = col32((1.0, 0.7, 0.1))
        vertices = np.asarray(self.vertices, dtype=np.float32)
        io = psim.GetIO()
        if len(vertices) >= 3 and is_simple_polygon(self.vertices):
            draw_filled_polygon(draw_list, vertices, color, 0.25)
            draw_closed_polyline(draw_list, vertices, color)
        if psim.IsMousePosValid() and not io.WantCaptureMouse and not psim.IsAnyItemHovered():
            path = np.vstack((vertices, np.asarray(psim.GetMousePos(), dtype=np.float32)))
            draw_open_polyline(draw_list, path, color)
        elif len(vertices) >= 2:
            draw_open_polyline(draw_list, vertices, color)
        self._draw_nodes(draw_list)

    def _draw_nodes(self, draw_list: Any) -> None:
        """Draw a square node at every vertex, highlighting the close target.

        Parameters
        ----------
        draw_list : ImDrawList
            Target draw list.
        """
        for index, vertex in enumerate(self.vertices):
            color = col32(NODE_COLOR)
            size = NODE_SIZE_PX
            if index == 0:
                color = col32(CLOSE_NODE_COLOR)
                if self._cursor_near(vertex):
                    size = NODE_SIZE_PX + 2.0
            draw_list.AddRectFilled(
                (vertex[0] - size, vertex[1] - size),
                (vertex[0] + size, vertex[1] + size),
                color,
            )

    def _cursor_near(self, vertex: tuple[float, float]) -> bool:
        """Return whether the cursor is within the close distance of a vertex.

        Parameters
        ----------
        vertex : tuple of float
            Screen-space vertex position.

        Returns
        -------
        bool
            ``True`` when the cursor is close enough to close the polygon.
        """
        if not psim.IsMousePosValid():
            return False
        io = psim.GetIO()
        distance = np.hypot(io.MousePos[0] - vertex[0], io.MousePos[1] - vertex[1])
        return distance <= CLOSE_DISTANCE_PX

    def cancel(self) -> None:
        """Discard the polygon in progress."""
        self.vertices.clear()
        self._pending_click = None

    def _add_vertex(self, position: tuple[float, float]) -> None:
        """Add a screen-space vertex, ignoring clicks too close to the last."""
        if self.vertices:
            last = self.vertices[-1]
            distance = np.hypot(position[0] - last[0], position[1] - last[1])
            if distance < MIN_VERTEX_SPACING_PX:
                return
        self.vertices.append(position)

    def _closes_polygon(self, position: tuple[float, float]) -> bool:
        """Return whether a click at this position closes the polygon.

        The polygon closes only when clicking on (or near) its first node.

        Parameters
        ----------
        position : tuple of float
            Click position in screen space.

        Returns
        -------
        bool
            ``True`` when the click is within the close distance of the first
            vertex and the polygon has at least three vertices.
        """
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
