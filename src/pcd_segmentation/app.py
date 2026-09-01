"""Main application: state, polyscope wiring, and the UI callback."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polyscope as ps
import polyscope.imgui as psim

from pcd_segmentation import config
from pcd_segmentation.core import project, transform
from pcd_segmentation.core.model import PointCloudData
from pcd_segmentation.core.segments import SegmentManager
from pcd_segmentation.io import exporter, loader
from pcd_segmentation.selection.base import SelectionMode
from pcd_segmentation.selection.box import BoxTool
from pcd_segmentation.selection.geometry import ViewState
from pcd_segmentation.selection.lasso import LassoTool
from pcd_segmentation.ui import panels

TOOL_KEYS: dict[str, str] = {"v": "lasso", "b": "box"}
"""Single-key (nvim-style) shortcuts that activate a selection tool."""

ACTION_KEYS: dict[str, str] = {
    "u": "undo",
    "x": "clear_selection",
    "c": "confirm_selection",
    "s": "create_segment_from_selection",
    "i": "invert_selection",
}
"""Single-key (nvim-style) shortcuts mapped to application actions."""

QUIT_KEY: str = "q"
"""Single-key shortcut that closes the viewer window."""


class App:
    """Top-level application state and event handling.

    Attributes
    ----------
    cloud : PointCloudData or None
        Currently loaded point cloud.
    manager : SegmentManager or None
        Segment bookkeeping for the loaded cloud.
    selection : np.ndarray
        Global indices of the currently selected points.
    active_tool_name : str
        Key of the active selection tool (``"none"``, ``"lasso"`` or ``"box"``).
    status_text : str
        Last message shown in the UI.
    transform : np.ndarray
        Accumulated 4x4 transform applied to the cloud (identity when the
        cloud is in its loaded position).
    gizmo_enabled : bool
        Whether the ImGuizmo transformation gizmo is active.
    """

    def __init__(self) -> None:
        self.cloud: PointCloudData | None = None
        self.manager: SegmentManager | None = None
        self.selection: np.ndarray = np.empty(0, dtype=np.int64)
        self._tools: dict[str, Any] = {"lasso": LassoTool(self), "box": BoxTool(self)}
        self.active_tool_name = "none"
        self._dirty = True
        self.status_text = ""
        self.show_unassigned = True
        self.show_original_colors = True
        self.target_segment_id = config.UNASSIGNED
        self.merge_keep_id = config.UNASSIGNED
        self.merge_drop_id = config.UNASSIGNED
        self.open_path_text = ""
        self.project_path_text = ""
        self.export_dir_text = config.DEFAULT_OUT_DIR
        self.transform: np.ndarray = np.eye(4, dtype=np.float64)
        self.gizmo_enabled = False
        self._gizmo: Any = None
        self._gizmo_base: np.ndarray = np.eye(4, dtype=np.float64)
        self._gizmo_last: np.ndarray = np.eye(4, dtype=np.float64)

    def run(self) -> None:
        """Open the interactive Polyscope window and run the event loop."""
        ps.set_program_name("pcd-segmentation")
        ps.init()
        ps.set_up_dir("z_up")
        self._gizmo = ps.add_transformation_gizmo("move cloud")
        self._gizmo.set_enabled(False)
        self._sync_gizmo_transform()
        if self.cloud is not None:
            self._rebuild_scene()
            ps.reset_camera_to_home_view()
        ps.set_user_callback(self._callback)
        ps.set_files_dropped_callback(self._on_files_dropped)
        ps.show()

    def load_cloud(self, path: str | Path) -> bool:
        """Load a point cloud file, replacing any previously loaded cloud.

        Parameters
        ----------
        path : str or Path
            Path of the file to load.

        Returns
        -------
        bool
            ``True`` on success.
        """
        path = Path(path)
        if not path.is_file():
            self.status_text = f"File not found: {path}"
            return False
        try:
            cloud = loader.load_point_cloud(path)
        except (OSError, ValueError) as error:
            self.status_text = f"Failed to load {path}: {error}"
            return False
        self.cloud = cloud
        self.manager = SegmentManager(cloud.num_points)
        self.selection = np.empty(0, dtype=np.int64)
        self.transform = np.eye(4, dtype=np.float64)
        self.gizmo_enabled = False
        self._gizmo_base = transform.translation_matrix(cloud.points)
        self._reset_combo_ids()
        self._dirty = True
        self.status_text = f"Loaded {path} ({cloud.num_points} points)"
        return True

    def mark_dirty(self) -> None:
        """Request a scene rebuild on the next frame."""
        self._dirty = True

    def visible_indices(self) -> np.ndarray:
        """Return the global indices of the points currently rendered.

        Points of hidden segments are excluded; unassigned points are
        included only when ``show_unassigned`` is enabled.

        Returns
        -------
        np.ndarray
            Array of global point indices.
        """
        if self.manager is None:
            return np.empty(0, dtype=np.int64)
        visible_ids = [segment.id for segment in self.manager.segments if segment.visible]
        labels = self.manager.labels
        mask = np.isin(labels, visible_ids)
        if self.show_unassigned:
            mask |= labels == config.UNASSIGNED
        return np.flatnonzero(mask)

    def visible_points(self) -> np.ndarray:
        """Return the world coordinates of the visible points.

        The accumulated gizmo transform is applied, matching what is shown
        in the scene.

        Returns
        -------
        np.ndarray
            Array of shape ``(M, 3)`` with the visible point coordinates.
        """
        if self.cloud is None:
            return np.empty((0, 3), dtype=np.float64)
        points = self.cloud.points[self.visible_indices()]
        return transform.apply_transform(points, self.transform)

    def view_state(self) -> ViewState | None:
        """Snapshot the current camera and viewport for screen projection.

        Returns
        -------
        ViewState or None
            ``None`` when no cloud is loaded.
        """
        if self.cloud is None:
            return None
        params = ps.get_view_camera_parameters()
        width, height = ps.get_buffer_size()
        return ViewState(
            extrinsics=np.asarray(params.get_E(), dtype=np.float64),
            fov_deg=float(params.get_fov_vertical_deg()),
            aspect=float(params.get_aspect()),
            width=int(width),
            height=int(height),
            orthographic=(ps.get_view_projection_mode() == "orthographic"),
            length_scale=float(ps.get_length_scale()),
        )

    def add_selection_from_visible(self, mask: np.ndarray, mode: SelectionMode) -> None:
        """Update the selection from a mask over the visible points.

        Parameters
        ----------
        mask : np.ndarray
            Boolean array over the visible points.
        mode : SelectionMode
            How to combine the result with the current selection.
        """
        indices = self.visible_indices()[np.asarray(mask, dtype=bool)]
        self.add_selection(indices, mode)

    def add_selection(self, indices: np.ndarray, mode: SelectionMode) -> None:
        """Combine the given point indices with the current selection.

        Parameters
        ----------
        indices : np.ndarray
            Global point indices to add, remove, or set.
        mode : SelectionMode
            Combination mode.
        """
        indices = np.asarray(indices, dtype=np.int64)
        if mode is SelectionMode.SET:
            self.selection = np.unique(indices)
        elif mode is SelectionMode.ADD:
            self.selection = np.union1d(self.selection, indices)
        else:
            self.selection = np.setdiff1d(self.selection, indices)
        self._dirty = True

    def clear_selection(self) -> None:
        """Discard the current selection."""
        self.selection = np.empty(0, dtype=np.int64)
        self._dirty = True

    def invert_selection(self) -> None:
        """Invert the selection over the visible points."""
        visible = self.visible_indices()
        self.selection = np.setdiff1d(visible, self.selection)
        self._dirty = True

    def confirm_selection(self) -> None:
        """Assign the selected points to the target segment."""
        if self.manager is None:
            return
        if len(self.selection) == 0:
            self.status_text = "No points selected."
            return
        target = self.target_segment_id
        if target == config.UNASSIGNED or not self.manager.has(target):
            target = self.manager.create_segment(
                self._default_segment_name(),
                self.manager.next_color(),
                self.selection,
            )
        else:
            self.manager.assign(target, self.selection)
        self.target_segment_id = target
        self._dirty = True
        self.status_text = (
            f"Assigned {len(self.selection)} points to '{self.manager.get(target).name}'"
        )

    def create_segment_from_selection(self) -> int | None:
        """Create a new segment containing the current selection.

        Returns
        -------
        int or None
            Id of the new segment, or ``None`` when there is no selection.
        """
        if self.manager is None:
            return None
        if len(self.selection) == 0:
            self.status_text = "Select some points first."
            return None
        seg_id = self.manager.create_segment(
            self._default_segment_name(),
            self.manager.next_color(),
            self.selection,
        )
        self.target_segment_id = seg_id
        self._dirty = True
        self.status_text = (
            f"Created '{self.manager.get(seg_id).name}' ({len(self.selection)} points)"
        )
        return seg_id

    def unassign_selected(self) -> None:
        """Remove the selected points from their current segment."""
        if self.manager is None or len(self.selection) == 0:
            return
        self.manager.unassign(self.selection)
        self._dirty = True
        self.status_text = f"Unassigned {len(self.selection)} points"

    def rename_segment(self, seg_id: int, name: str) -> None:
        """Rename a segment and refresh the scene.

        Parameters
        ----------
        seg_id : int
            Segment id.
        name : str
            New name.
        """
        if self.manager is None:
            return
        self.manager.rename(seg_id, name)
        self._dirty = True

    def set_segment_color(self, seg_id: int, color: tuple[float, float, float]) -> None:
        """Update the display color of a segment.

        Parameters
        ----------
        seg_id : int
            Segment id.
        color : tuple of float
            New RGB color in ``[0, 1]``.
        """
        if self.manager is None:
            return
        self.manager.set_color(seg_id, color)
        self._dirty = True

    def set_segment_visible(self, seg_id: int, visible: bool) -> None:
        """Show or hide a segment.

        Parameters
        ----------
        seg_id : int
            Segment id.
        visible : bool
            New visibility state.
        """
        if self.manager is None:
            return
        self.manager.set_visible(seg_id, visible)
        self._dirty = True

    def delete_segment(self, seg_id: int) -> None:
        """Delete a segment, unassigning its points.

        Parameters
        ----------
        seg_id : int
            Segment id to delete.
        """
        if self.manager is None:
            return
        self.manager.delete_segment(seg_id)
        self._reset_combo_ids()
        self._dirty = True

    def merge_selected_segments(self) -> None:
        """Merge the segment selected as 'drop' into the one selected as 'keep'."""
        if self.manager is None:
            return
        if self.merge_keep_id == self.merge_drop_id:
            self.status_text = "Pick two different segments to merge."
            return
        if not self.manager.has(self.merge_keep_id) or not self.manager.has(self.merge_drop_id):
            self.status_text = "Pick two valid segments to merge."
            return
        self.manager.merge(self.merge_keep_id, self.merge_drop_id)
        self._reset_combo_ids()
        self._dirty = True
        self.status_text = "Merged segments"

    def undo(self) -> None:
        """Revert the most recent segmentation operation."""
        if self.manager is not None and self.manager.undo():
            self._reset_combo_ids()
            self._dirty = True

    def redo(self) -> None:
        """Re-apply the most recently reverted operation."""
        if self.manager is not None and self.manager.redo():
            self._reset_combo_ids()
            self._dirty = True

    def export_segments_text(self) -> None:
        """Export all non-empty segments to the directory in the UI."""
        if self.manager is None:
            self.status_text = "No cloud loaded."
            return
        try:
            written = exporter.export_segments(self.export_dir_text, self.cloud, self.manager)
        except OSError as error:
            self.status_text = f"Export failed: {error}"
            return
        self.status_text = f"Exported {len(written)} segment(s) to {self.export_dir_text}"

    def save_project_text(self) -> None:
        """Save the current segmentation state to the project path in the UI."""
        if self.manager is None:
            self.status_text = "No cloud loaded."
            return
        path = Path(self.project_path_text or "segmentation.json")
        if not path.suffix:
            path = path.with_suffix(".json")
        try:
            project.save_project(
                path,
                self.cloud.source,
                self.manager.segments,
                self.manager.labels,
                self.transform,
            )
        except OSError as error:
            self.status_text = f"Failed to save project: {error}"
            return
        self.project_path_text = str(path)
        self.status_text = f"Saved project to {path}"

    def load_project_text(self) -> None:
        """Resume the segmentation state from the project path in the UI."""
        path = Path(self.project_path_text)
        if not path.is_file():
            self.status_text = f"Project not found: {path}"
            return
        try:
            state = project.load_project(path)
        except (OSError, ValueError, KeyError) as error:
            self.status_text = f"Failed to load project: {error}"
            return
        source = state["source"]
        if source is None or not source.is_file():
            self.status_text = f"Source cloud not found: {source}"
            return
        if not self.load_cloud(source):
            return
        self.manager.apply_state(state["segments"], state["labels"])
        self.transform = state["transform"]
        self._gizmo_base = transform.translation_matrix(self.cloud.points)
        self._sync_gizmo_transform()
        self._reset_combo_ids()
        self._dirty = True
        self.status_text = f"Loaded project {path}"

    def load_path_text(self) -> None:
        """Load the cloud whose path is in the Open input field."""
        if self.open_path_text:
            self.load_cloud(self.open_path_text)

    def target_segment_name(self) -> str:
        """Return the display name of the target segment."""
        if self.target_segment_id == config.UNASSIGNED:
            return "(new segment)"
        if self.manager is not None and self.manager.has(self.target_segment_id):
            return self.manager.get(self.target_segment_id).name
        return "(invalid)"

    def set_active_tool(self, name: str) -> None:
        """Switch the active selection tool.

        The default camera interaction is disabled while a tool is active so
        that clicks and drags are not interpreted as camera commands. The
        transformation gizmo is deactivated when entering a selection mode.

        Parameters
        ----------
        name : str
            Tool key (``"none"``, ``"lasso"`` or ``"box"``).
        """
        if name == self.active_tool_name:
            return
        if self.active_tool_name in self._tools:
            self._tools[self.active_tool_name].cancel()
        self.active_tool_name = name
        if name != "none" and self.gizmo_enabled:
            self.set_gizmo_enabled(False)
        ps.set_do_default_mouse_interaction(name == "none")

    def set_gizmo_enabled(self, enabled: bool) -> None:
        """Show or hide the ImGuizmo transformation gizmo.

        When enabled, the gizmo is positioned at the cloud centroid and the
        active selection tool is switched off. The accumulated transform is
        kept, so the gizmo re-appears where the transformed cloud is.

        Parameters
        ----------
        enabled : bool
            New gizmo state.
        """
        self.gizmo_enabled = enabled
        if enabled:
            self.set_active_tool("none")
            if self.cloud is not None:
                self._gizmo_base = transform.translation_matrix(self.cloud.points)
            self._sync_gizmo_transform()
            if self._gizmo is not None:
                self._gizmo.set_enabled(True)
        elif self._gizmo is not None:
            self._gizmo.set_enabled(False)

    def apply_transform_to_cloud(self) -> None:
        """Bake the accumulated gizmo transform into the point coordinates.

        After baking, the transform is reset to identity, the gizmo moves to
        the centroid of the transformed cloud, and exports reflect the new
        coordinates.
        """
        if self.cloud is None:
            return
        if np.allclose(self.transform, np.eye(4)):
            self.status_text = "Nothing to apply: transform is identity."
            return
        self.cloud.points = transform.apply_transform(self.cloud.points, self.transform)
        if self.cloud.normals is not None:
            self.cloud.normals = transform.rotate_vectors(self.cloud.normals, self.transform)
        self.transform = np.eye(4, dtype=np.float64)
        self._gizmo_base = transform.translation_matrix(self.cloud.points)
        self._sync_gizmo_transform()
        self._dirty = True
        self.status_text = "Transform applied to cloud"

    def reset_transform(self) -> None:
        """Reset the accumulated gizmo transform to identity."""
        self.transform = np.eye(4, dtype=np.float64)
        self._gizmo_base = transform.translation_matrix(
            self.cloud.points if self.cloud is not None else np.empty((0, 3))
        )
        self._sync_gizmo_transform()
        self._dirty = True
        self.status_text = "Transform reset"

    def transform_translation_text(self) -> str:
        """Return the translation component of the transform as text."""
        translation = transform.extract_translation(self.transform)
        return f"{translation[0]:.4g}, {translation[1]:.4g}, {translation[2]:.4g}"

    def _callback(self) -> None:
        """Per-frame UI callback invoked by Polyscope."""
        self._poll_gizmo()
        panels.draw_file_panel(self)
        panels.draw_toolbar(self)
        panels.draw_segments_panel(self)
        panels.draw_transform_panel(self)
        self._handle_tool_input()
        self._handle_shortcuts()
        self._rebuild_scene_if_dirty()

    def _handle_tool_input(self) -> None:
        """Feed the current frame input to the active selection tool."""
        if self.active_tool_name == "none":
            return
        tool = self._tools[self.active_tool_name]
        io = psim.GetIO()
        tool.handle_input(io)
        tool.draw()

    def _poll_gizmo(self) -> None:
        """Read the gizmo matrix and update the accumulated transform."""
        if self._gizmo is None or not self._gizmo.get_enabled():
            return
        combined = np.asarray(self._gizmo.get_transform(), dtype=np.float64)
        if np.allclose(combined, self._gizmo_last):
            return
        self._gizmo_last = combined
        self.transform = transform.relative_transform(combined, self._gizmo_base)
        self._dirty = True

    def _sync_gizmo_transform(self) -> None:
        """Push the accumulated transform to the gizmo widget."""
        if self._gizmo is None:
            return
        self._gizmo_last = self._gizmo_base @ self.transform
        self._gizmo.set_transform(self._gizmo_last)

    def _handle_shortcuts(self) -> None:
        """Process the global keyboard shortcuts.

        Shortcuts are ignored while a text field wants keyboard input. The
        single-key (nvim-style) bindings are: ``v`` lasso tool, ``b`` box
        tool, ``Esc`` back to navigation, ``c`` confirm selection, ``s``
        create segment from selection, ``x`` clear selection, ``i`` invert
        selection, ``u`` undo, ``Ctrl+Z``/``Ctrl+R``/``Ctrl+Y`` undo/redo,
        ``q`` quit.
        """
        io = psim.GetIO()
        if io.WantTextInput:
            return
        self._handle_ctrl_shortcuts(io)
        self._handle_single_key_shortcuts()

    def _handle_ctrl_shortcuts(self, io: Any) -> None:
        """Process the Ctrl-modified undo/redo shortcuts.

        Parameters
        ----------
        io : ImGuiIO
            Current frame input state.
        """
        if not io.KeyCtrl:
            return
        if psim.IsKeyPressed(psim.ImGuiKey_Z):
            self.undo()
        if psim.IsKeyPressed(psim.ImGuiKey_R) or psim.IsKeyPressed(psim.ImGuiKey_Y):
            self.redo()

    def _handle_single_key_shortcuts(self) -> None:
        """Process the nvim-style single-key shortcuts."""
        if psim.IsKeyPressed(psim.ImGuiKey_Escape):
            self.set_active_tool("none")
        for key, tool_name in TOOL_KEYS.items():
            if psim.IsKeyPressed(getattr(psim, f"ImGuiKey_{key.upper()}")):
                self.set_active_tool(tool_name)
        for key, action in ACTION_KEYS.items():
            if psim.IsKeyPressed(getattr(psim, f"ImGuiKey_{key.upper()}")):
                getattr(self, action)()
        if psim.IsKeyPressed(psim.ImGuiKey_Q):
            ps.unshow()

    def _on_files_dropped(self, paths: list[str]) -> None:
        """Handle files dropped onto the Polyscope window.

        Parameters
        ----------
        paths : list of str
            Paths of the dropped files.
        """
        for path in paths:
            suffix = Path(path).suffix.lower()
            if suffix == ".json":
                self.project_path_text = path
                self.load_project_text()
            elif loader.is_supported(Path(path)):
                self.load_cloud(path)
            else:
                self.status_text = f"Unsupported file: {path}"

    def _rebuild_scene_if_dirty(self) -> None:
        """Rebuild the Polyscope structures when the scene is stale."""
        if self._dirty:
            self._dirty = False
            self._rebuild_scene()

    def _rebuild_scene(self) -> None:
        """Register the visible cloud and the selection highlight."""
        if self.cloud is None:
            ps.remove_point_cloud(config.CLOUD_NAME, error_if_absent=False)
            ps.remove_point_cloud(config.SELECTION_NAME, error_if_absent=False)
            return
        indices = self.visible_indices()
        if len(indices) == 0:
            ps.remove_point_cloud(config.CLOUD_NAME, error_if_absent=False)
        else:
            ps_cloud = ps.register_point_cloud(
                config.CLOUD_NAME,
                self.cloud.points[indices],
                point_render_mode="quad",
                enabled=True,
            )
            ps_cloud.add_color_quantity("segments", self._visible_colors(indices), enabled=True)
            ps_cloud.set_transform(self.transform)
        self._rebuild_selection_cloud()

    def _rebuild_selection_cloud(self) -> None:
        """Register the highlight cloud for the current selection."""
        if self.cloud is None or len(self.selection) == 0:
            ps.remove_point_cloud(config.SELECTION_NAME, error_if_absent=False)
            return
        base_radius = 0.005
        if ps.has_point_cloud(config.CLOUD_NAME):
            base_radius = ps.get_point_cloud(config.CLOUD_NAME).get_radius()
        ps_cloud = ps.register_point_cloud(
            config.SELECTION_NAME,
            self.cloud.points[self.selection],
            color=config.SELECTION_COLOR,
            point_render_mode="quad",
        )
        ps_cloud.set_radius(base_radius * 2.0)
        ps_cloud.set_transform(self.transform)

    def _visible_colors(self, indices: np.ndarray) -> np.ndarray:
        """Build the per-point RGB array for the visible points.

        Assigned points use their segment color; unassigned points keep the
        original cloud colors (or a neutral gray) depending on the display
        options.

        Parameters
        ----------
        indices : np.ndarray
            Visible point indices.

        Returns
        -------
        np.ndarray
            Array of shape ``(M, 3)`` with rgb values in ``[0, 1]``.
        """
        labels = self.manager.labels[indices]
        if self.show_original_colors and self.cloud.colors is not None:
            colors = self.cloud.colors[indices].astype(np.float32).copy()
        else:
            base = np.asarray(config.UNASSIGNED_COLOR, dtype=np.float32)
            colors = np.tile(base, (len(indices), 1))
        for segment in self.manager.segments:
            colors[labels == segment.id] = np.asarray(segment.color, dtype=np.float32)
        return colors

    def _default_segment_name(self) -> str:
        """Return the default name for the next created segment."""
        return f"segment_{len(self.manager.segments) + 1}"

    def _segment_combo_name(self, seg_id: int) -> str:
        """Return the display name of a segment id for a combo box."""
        if self.manager is not None and self.manager.has(seg_id):
            return self.manager.get(seg_id).name
        return "(none)"

    def _reset_combo_ids(self) -> None:
        """Clamp the UI combo selections to the current segment list."""
        self.target_segment_id = config.UNASSIGNED
        self.merge_keep_id = config.UNASSIGNED
        self.merge_drop_id = config.UNASSIGNED
        if self.manager is None or not self.manager.segments:
            return
        self.merge_keep_id = self.manager.segments[0].id
        if len(self.manager.segments) > 1:
            self.merge_drop_id = self.manager.segments[1].id
        else:
            self.merge_drop_id = self.manager.segments[0].id
