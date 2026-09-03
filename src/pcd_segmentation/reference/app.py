"""Standalone app: pick a reference point and convert the cloud's axes.

Unlike the segmentation tool, this application is not meant to label points:
its single purpose is to let the user choose one point of the cloud that
becomes the new origin ``(0, 0, 0)`` and, optionally, to re-express the
coordinates in a different axis convention (for example ``RDF`` to ``RUB``).
The current state is applied to the cloud in place and can be exported as a
PLY file together with a JSON sidecar describing the applied mapping.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polyscope as ps
import polyscope.imgui as psim

from pcd_segmentation.core import transform as tf
from pcd_segmentation.core.model import PointCloudData
from pcd_segmentation.io import loader
from pcd_segmentation.reference import conventions, export, ui
from pcd_segmentation.selection.geometry import ViewState, project_to_screen

PROGRAM_NAME: str = "pcd-reference"
"""Name shown in the Polyscope window title."""

CLOUD_NAME: str = "reference_cloud"
"""Polyscope structure name of the edited point cloud."""

CANDIDATE_NAME: str = "reference_candidate"
"""Polyscope structure name of the picked point marker."""

AXES_PREFIX: str = "reference_axis"
"""Prefix of the Polyscope structures drawing the origin axes."""

CANDIDATE_COLOR: tuple[float, float, float] = (1.0, 0.85, 0.1)
"""RGB color of the candidate point marker."""

CANDIDATE_RADIUS_SCALE: float = 1.5
"""The candidate marker is this many times larger than the cloud points."""

UNCOLORED_COLOR: tuple[float, float, float] = (0.75, 0.75, 0.75)
"""RGB color used for clouds without original colors."""

AXIS_COLORS: dict[str, tuple[float, float, float]] = {
    "x": (1.0, 0.25, 0.2),
    "y": (0.2, 0.85, 0.3),
    "z": (0.25, 0.5, 1.0),
}
"""Colors of the three origin axes."""

PICK_TOLERANCE_PX: float = 12.0
"""Maximum pixel distance between a click and a point to pick it."""

AXIS_LENGTH_SCALE: float = 0.12
"""Origin-axis length as a fraction of the Polyscope length scale."""

MAX_UNDO: int = 30
"""Maximum number of undo records kept in memory."""

ACTION_KEYS: dict[str, str] = {
    "p": "toggle_pick",
    "c": "set_origin_from_candidate",
    "x": "clear_candidate",
    "a": "apply_axes",
    "u": "undo",
}
"""Single-key (nvim-style) shortcuts mapped to application actions."""

QUIT_KEY: str = "q"
"""Single-key shortcut that closes the viewer window."""

UndoRecord = tuple[str, np.ndarray, np.ndarray, int | None, np.ndarray | None]
"""Undo record: ``(description, axes, translation, origin index, origin)``."""


class ReferenceApp:
    """State and event handling of the reference-point application.

    Attributes
    ----------
    cloud : PointCloudData or None
        Currently loaded point cloud; its ``points`` always hold the current
        (transformed) coordinates.
    candidate_index : int or None
        Index of the point picked as a candidate for the new origin.
    origin_index : int or None
        Index of the point that was turned into the origin, if any.
    pick_active : bool
        Whether a click chooses the candidate reference point.
    source_convention : str
        Triple describing the frame the loaded cloud is expressed in.
    target_convention : str
        Triple describing the frame the cloud should be converted to.
    show_axes : bool
        Whether the origin axes are drawn in the scene.
    transform : np.ndarray
        Accumulated 4x4 transform of the transformation gizmo (identity when
        no gizmo motion has been applied).
    gizmo_enabled : bool
        Whether the ImGuizmo transformation gizmo is active.
    status_text : str
        Last message shown in the UI.
    """

    def __init__(self) -> None:
        self.cloud: PointCloudData | None = None
        self._base_points: np.ndarray | None = None
        self._base_normals: np.ndarray | None = None
        self._axes: np.ndarray = np.eye(3, dtype=np.float64)
        self._translation: np.ndarray = np.zeros(3, dtype=np.float64)
        self.candidate_index: int | None = None
        self.origin_index: int | None = None
        self.origin_original: np.ndarray | None = None
        self.pick_active = False
        self.source_convention = conventions.NONE_TRIPLE
        self.target_convention = conventions.NONE_TRIPLE
        self.show_axes = True
        self.last_axes_apply: tuple[str, str] | None = None
        self.open_path_text = ""
        self.export_path_text = ""
        self.status_text = ""
        self.transform: np.ndarray = np.eye(4, dtype=np.float64)
        self.gizmo_enabled = False
        self._gizmo: Any = None
        self._gizmo_base: np.ndarray = np.eye(4, dtype=np.float64)
        self._gizmo_last: np.ndarray = np.eye(4, dtype=np.float64)
        self._pending_click: tuple[float, float] | None = None
        self._dirty = True
        self._candidate_dirty = False
        self._axes_dirty = True
        self._transform_dirty = False
        self._home_view_pending = False
        self._undo_stack: list[UndoRecord] = []
        self._redo_stack: list[UndoRecord] = []

    def run(self) -> None:
        """Open the interactive Polyscope window and run the event loop."""
        ps.set_program_name(PROGRAM_NAME)
        ps.init()
        ps.set_up_dir("z_up")
        self._home_view_pending = self.cloud is not None
        self._gizmo = ps.add_transformation_gizmo("transform cloud")
        self._gizmo.set_enabled(False)
        self._sync_gizmo_transform()
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
        self._attach(cloud)
        self.status_text = f"Loaded {path} ({cloud.num_points} points)"
        return True

    def load_path_text(self) -> None:
        """Load the cloud whose path is in the Open input field."""
        if self.open_path_text:
            self.load_cloud(self.open_path_text)

    def _attach(self, cloud: PointCloudData) -> None:
        """Adopt a loaded cloud and reset all editing state.

        Parameters
        ----------
        cloud : PointCloudData
            Cloud to edit.
        """
        self.cloud = cloud
        self._base_points = np.asarray(cloud.points, dtype=np.float64).copy()
        self.cloud.points = self._base_points.copy()
        if cloud.normals is not None:
            self._base_normals = np.asarray(cloud.normals, dtype=np.float64).copy()
            self.cloud.normals = self._base_normals.copy()
        self._axes = np.eye(3, dtype=np.float64)
        self._translation = np.zeros(3, dtype=np.float64)
        self.candidate_index = None
        self.origin_index = None
        self.origin_original = None
        self.pick_active = False
        self.last_axes_apply = None
        self.transform = np.eye(4, dtype=np.float64)
        self.gizmo_enabled = False
        self._gizmo_base = np.eye(4, dtype=np.float64)
        self._gizmo_last = np.eye(4, dtype=np.float64)
        self._pending_click = None
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._dirty = True
        self._candidate_dirty = False
        self._axes_dirty = True
        self._transform_dirty = False
        self._home_view_pending = True
        self.export_path_text = str(self.default_export_path())

    # ------------------------------------------------------------------ view

    def _view_state(self) -> ViewState | None:
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

    # ------------------------------------------------------------ transform

    def set_gizmo_enabled(self, enabled: bool) -> None:
        """Show or hide the ImGuizmo transformation gizmo.

        Enabling the gizmo switches the pick mode off and positions the
        widget on the current origin point (or on the cloud centroid when no
        origin has been set), so rotations and scales pivot about that point.

        Parameters
        ----------
        enabled : bool
            New gizmo state.
        """
        if enabled and self.cloud is None:
            self.status_text = "Load a cloud before enabling the gizmo."
            return
        self.gizmo_enabled = enabled
        if enabled:
            self.set_pick_active(False)
            self._set_gizmo_center(self._gizmo_pivot())
            self._sync_gizmo_transform()
            if self._gizmo is not None:
                self._gizmo.set_enabled(True)
        elif self._gizmo is not None:
            self._gizmo.set_enabled(False)
        self._transform_dirty = True

    def apply_gizmo_to_cloud(self) -> None:
        """Bake the accumulated gizmo transform into the point coordinates.

        The current transform is folded into the base-to-current mapping, so
        it can still be reverted with undo and is reflected in the exported
        sidecar. The gizmo itself is reset afterwards.
        """
        if self.cloud is None:
            self.status_text = "Load a cloud first."
            return
        if np.allclose(self.transform, np.eye(4)):
            self.status_text = "Nothing to apply: gizmo transform is identity."
            return
        linear = np.asarray(self.transform[:3, :3], dtype=np.float64)
        offset = np.asarray(self.transform[:3, 3], dtype=np.float64)
        self._push("Apply gizmo transform to cloud")
        self._set_state(linear @ self._axes, linear @ self._translation + offset)
        self.transform = np.eye(4, dtype=np.float64)
        self._set_gizmo_center(self._gizmo_pivot())
        self._sync_gizmo_transform()
        self._transform_dirty = True
        self.status_text = "Gizmo transform applied to cloud."

    def reset_gizmo(self) -> None:
        """Discard the accumulated gizmo transform."""
        self.transform = np.eye(4, dtype=np.float64)
        self._set_gizmo_center(self._gizmo_pivot())
        self._sync_gizmo_transform()
        self._transform_dirty = True
        self.status_text = "Gizmo transform reset."

    def transform_text(self) -> str:
        """Return the translation component of the gizmo transform as text."""
        translation = tf.extract_translation(self.transform)
        return f"{translation[0]:.4g}, {translation[1]:.4g}, {translation[2]:.4g}"

    @property
    def transform_pending(self) -> bool:
        """Return whether an un-applied gizmo transform is on display.

        A transform is pending either while the gizmo widget is enabled or
        when the gizmo was moved but its motion was not applied nor reset.
        """
        return self.gizmo_enabled or not np.allclose(self.transform, np.eye(4))

    def _gizmo_pivot(self) -> np.ndarray:
        """Return the world-space point the gizmo should pivot about.

        Returns
        -------
        np.ndarray
            Pivot position, shape ``(3,)``.
        """
        if self.cloud is None:
            return np.zeros(3, dtype=np.float64)
        if self.origin_index is not None:
            return np.asarray(self.cloud.points[self.origin_index], dtype=np.float64)
        return np.asarray(self.cloud.points.mean(axis=0), dtype=np.float64)

    def _poll_gizmo(self) -> None:
        """Read the gizmo matrix and update the accumulated transform.

        The gizmo's incremental motion is applied about the gizmo center, so
        rotations and scales pivot around wherever the widget is placed (the
        origin point or the cloud centroid).
        """
        if self._gizmo is None or not self._gizmo.get_enabled():
            return
        combined = np.asarray(self._gizmo.get_transform(), dtype=np.float64)
        if np.allclose(combined, np.asarray(self._gizmo_last, dtype=np.float32)):
            return
        delta = combined @ np.linalg.inv(self._gizmo_last)
        self.transform = delta @ self.transform
        self._gizmo_last = combined
        self._transform_dirty = True

    def _sync_gizmo_transform(self) -> None:
        """Push the accumulated transform to the gizmo widget."""
        if self._gizmo is None:
            return
        self._gizmo_last = self._gizmo_base @ self.transform
        self._gizmo.set_transform(self._gizmo_last)

    def _set_gizmo_center(self, world_point: np.ndarray) -> None:
        """Point the gizmo base so its widget center sits at a world point.

        Parameters
        ----------
        world_point : np.ndarray
            Desired world-space gizmo center, shape ``(3,)``.
        """
        center = np.asarray(world_point, dtype=np.float64)
        translation = tf.extract_translation(self.transform)
        self._gizmo_base = tf.translation_matrix((center - translation)[None, :])

    def _apply_display_transform(self) -> None:
        """Push the gizmo transform to every registered structure."""
        if self.cloud is None:
            return
        if ps.has_point_cloud(CLOUD_NAME):
            ps.get_point_cloud(CLOUD_NAME).set_transform(self.transform)
        if ps.has_point_cloud(CANDIDATE_NAME):
            ps.get_point_cloud(CANDIDATE_NAME).set_transform(self.transform)
        for axis in AXIS_COLORS:
            name = f"{AXES_PREFIX}_{axis}"
            if ps.has_curve_network(name):
                ps.get_curve_network(name).set_transform(self.transform)

    # ----------------------------------------------------------- reference

    def toggle_pick(self) -> None:
        """Toggle the click-to-pick mode."""
        self.set_pick_active(not self.pick_active)

    def set_pick_active(self, active: bool) -> None:
        """Enable or disable the click-to-pick mode.

        While the pick mode is active the default camera interaction is
        disabled so that clicks are not interpreted as camera commands.

        Parameters
        ----------
        active : bool
            Desired pick state.
        """
        if active and self.cloud is None:
            self.status_text = "Load a cloud before picking a point."
            return
        if active and self.transform_pending:
            self.status_text = "Apply or reset the transformation gizmo first."
            return
        if active == self.pick_active:
            return
        self.pick_active = active
        self._pending_click = None
        ps.set_do_default_mouse_interaction(not active)

    def _handle_pick_input(self) -> None:
        """Consume the current frame input while the pick mode is active."""
        if not self.pick_active or self.cloud is None:
            return
        io = psim.GetIO()
        self._accept_pending_click(io)
        if not psim.IsMousePosValid():
            return
        if (
            io.MouseClicked[0]
            and not io.MouseDoubleClicked[0]
            and not io.WantCaptureMouse
            and not psim.IsAnyItemHovered()
        ):
            position = self._validated_position(io)
            if position is not None:
                self._pending_click = position

    def _accept_pending_click(self, io: Any) -> None:
        """Accept a click queued on a previous frame if it is still valid.

        Clicks are accepted one frame late: the user callback runs before the
        Polyscope UI is built, so ``io.WantCaptureMouse`` lags a frame. A
        click that lands on a side panel is therefore only discarded once the
        up-to-date capture flag is available.

        Parameters
        ----------
        io : Any
            Current frame ImGui input state.
        """
        if self._pending_click is None:
            return
        position = self._pending_click
        self._pending_click = None
        if io.WantCaptureMouse or psim.IsAnyItemHovered():
            return
        index = self._candidate_from_position(position)
        if index is None:
            self.status_text = "No cloud point under the cursor."
        else:
            self.set_candidate(index)

    def _validated_position(self, io: Any) -> tuple[float, float] | None:
        """Return the click position, rejecting invalid or out-of-window ones.

        Parameters
        ----------
        io : Any
            Current frame ImGui input state.

        Returns
        -------
        tuple of float or None
            The click position in pixels, or ``None`` when unusable.
        """
        position = tuple(io.MousePos)
        if not np.all(np.isfinite(position)):
            return None
        width, height = io.DisplaySize
        if not (0.0 <= position[0] <= width and 0.0 <= position[1] <= height):
            return None
        return position

    def _candidate_from_position(self, position: tuple[float, float]) -> int | None:
        """Return the index of the point closest to a screen position.

        Parameters
        ----------
        position : tuple of float
            Click position in pixels.

        Returns
        -------
        int or None
            Index of the closest point, or ``None`` when the closest point is
            farther than :data:`PICK_TOLERANCE_PX` pixels.
        """
        if self.cloud is None:
            return None
        view = self._view_state()
        points = self.cloud.points
        if view is None or len(points) == 0:
            return None
        screen = project_to_screen(points, view)
        delta = screen - np.asarray(position, dtype=np.float64)
        squared = np.einsum("ij,ij->i", delta, delta)
        index = int(np.argmin(squared))
        if squared[index] > PICK_TOLERANCE_PX**2:
            return None
        return index

    def set_candidate(self, index: int) -> None:
        """Mark a cloud point as the candidate for the new origin.

        Parameters
        ----------
        index : int
            Global point index of the candidate.
        """
        if self.cloud is None or not 0 <= index < self.cloud.num_points:
            self.status_text = "Candidate index out of range."
            return
        self.candidate_index = index
        self._candidate_dirty = True
        self.status_text = f"Picked point #{index} {self.candidate_text()}"

    def clear_candidate(self) -> None:
        """Discard the candidate reference point."""
        self.candidate_index = None
        self._candidate_dirty = True
        self.status_text = "Chosen point cleared."

    def set_origin_from_candidate(self) -> None:
        """Translate the cloud so the candidate point becomes the origin."""
        if self.cloud is None:
            self.status_text = "Load a cloud first."
            return
        if self._gizmo_guard():
            return
        index = self.candidate_index
        if index is None:
            self.status_text = "Pick a point first (p, then click, then confirm)."
            return
        origin = self._base_points[index]
        translation = -(self._axes @ origin)
        self._push(f"Origin at point #{index}")
        self._set_state(self._axes, translation)
        self.origin_index = index
        self.origin_original = origin.copy()
        self.status_text = f"Origin set to point #{index} -> (0, 0, 0)."

    def candidate_text(self) -> str:
        """Return the current coordinates of the candidate point as text."""
        if self.cloud is None or self.candidate_index is None:
            return ""
        point = self.cloud.points[self.candidate_index]
        return f"({point[0]:.4g}, {point[1]:.4g}, {point[2]:.4g})"

    def origin_text(self) -> str:
        """Return a short description of the current origin."""
        if self.origin_index is None:
            return "none"
        return f"point #{self.origin_index} -> (0, 0, 0)"

    # ---------------------------------------------------------------- axes

    def apply_axes(self) -> None:
        """Convert the current coordinates between the selected conventions."""
        if self.cloud is None:
            self.status_text = "Load a cloud first."
            return
        if self._gizmo_guard():
            return
        source = self.source_convention
        target = self.target_convention
        if source == conventions.NONE_TRIPLE or target == conventions.NONE_TRIPLE:
            self.status_text = "Pick two real conventions to convert the axes."
            return
        if source == target:
            self.status_text = "Source and target conventions are identical."
            return
        matrix = conventions.axes_matrix(source, target)
        self._push(f"Convert axes {source} -> {target}")
        self._set_state(matrix @ self._axes, matrix @ self._translation)
        self.last_axes_apply = (source, target)
        self.status_text = f"Converted axes {source} -> {target}."

    def axes_matrix_text(self) -> str:
        """Return a text preview of the selected axes conversion."""
        source = self.source_convention
        target = self.target_convention
        if source == conventions.NONE_TRIPLE or target == conventions.NONE_TRIPLE:
            return "No axes conversion selected."
        matrix = conventions.axes_matrix(source, target)
        rows = "\n".join(
            "  [" + ", ".join(f"{value:+.0f}" for value in row) + "]" for row in matrix
        )
        return f"Preview {source} -> {target}:\n{rows}"

    def set_show_axes(self, visible: bool) -> None:
        """Show or hide the origin axes overlay.

        Parameters
        ----------
        visible : bool
            New visibility state.
        """
        self.show_axes = visible
        self._axes_dirty = True

    # ------------------------------------------------------------ undo/redo

    def can_undo(self) -> bool:
        """Return whether an undo operation is available."""
        return bool(self._undo_stack)

    def can_redo(self) -> bool:
        """Return whether a redo operation is available."""
        return bool(self._redo_stack)

    def _gizmo_guard(self) -> bool:
        """Report whether an operation is blocked by a pending gizmo motion.

        Returns
        -------
        bool
            ``True`` when a gizmo transform is pending and the operation was
            refused.
        """
        if self.transform_pending:
            self.status_text = "Apply or reset the transformation gizmo first."
            return True
        return False

    def undo(self) -> None:
        """Revert the most recent editing operation."""
        if self._gizmo_guard():
            return
        if not self._undo_stack:
            self.status_text = "Nothing to undo."
            return
        description, axes, translation, origin_index, origin_original = self._undo_stack.pop()
        self._redo_stack.append(self._current_record(description))
        self._restore(axes, translation, origin_index, origin_original)
        self.status_text = f"Undo: {description}"

    def redo(self) -> None:
        """Re-apply the most recently reverted operation."""
        if self._gizmo_guard():
            return
        if not self._redo_stack:
            self.status_text = "Nothing to redo."
            return
        description, axes, translation, origin_index, origin_original = self._redo_stack.pop()
        self._undo_stack.append(self._current_record(description))
        self._restore(axes, translation, origin_index, origin_original)
        self.status_text = f"Redo: {description}"

    def _current_record(self, description: str) -> UndoRecord:
        """Capture the current editing state as an undo record.

        Parameters
        ----------
        description : str
            Description of the operation the record refers to.

        Returns
        -------
        tuple
            Record holding the description, the base-to-current mapping and
            the origin point bookkeeping.
        """
        original = self.origin_original.copy() if self.origin_original is not None else None
        return (
            description,
            self._axes.copy(),
            self._translation.copy(),
            self.origin_index,
            original,
        )

    def _push(self, description: str) -> None:
        """Record the current state for undo before the next mutation.

        Parameters
        ----------
        description : str
            Description of the pending operation.
        """
        self._undo_stack.append(self._current_record(description))
        del self._undo_stack[:-MAX_UNDO]
        self._redo_stack.clear()

    def _restore(
        self,
        axes: np.ndarray,
        translation: np.ndarray,
        origin_index: int | None,
        origin_original: np.ndarray | None,
    ) -> None:
        """Restore a recorded state, refreshing the cloud coordinates.

        Parameters
        ----------
        axes : np.ndarray
            Linear part of the base-to-current mapping.
        translation : np.ndarray
            Translation part of the mapping.
        origin_index : int or None
            Index of the origin point of the restored state.
        origin_original : np.ndarray or None
            Original coordinates of the restored origin point.
        """
        self._set_state(axes, translation)
        self.origin_index = origin_index
        self.origin_original = (
            np.asarray(origin_original, dtype=np.float64).copy()
            if origin_original is not None
            else None
        )

    def _set_state(self, axes: np.ndarray, translation: np.ndarray) -> None:
        """Apply a linear mapping (base -> current) and refresh the cloud.

        Parameters
        ----------
        axes : np.ndarray
            New 3x3 linear part of the base-to-current mapping.
        translation : np.ndarray
            New translation part, shape ``(3,)``.
        """
        self._axes = np.asarray(axes, dtype=np.float64).copy()
        self._translation = np.asarray(translation, dtype=np.float64).copy()
        if self.cloud is None or self._base_points is None:
            return
        points = (self._axes @ self._base_points.T).T + self._translation
        self.cloud.points = points.astype(np.float64)
        if self.cloud.normals is not None and self._base_normals is not None:
            self.cloud.normals = (self._base_normals @ self._axes.T).astype(np.float64)
        self._dirty = True
        self._candidate_dirty = self.candidate_index is not None

    # ---------------------------------------------------------------- export

    def default_export_path(self) -> Path:
        """Return the default destination path of the exported PLY.

        Returns
        -------
        Path
            ``<source_stem>_referenced.ply`` next to the source cloud.
        """
        if self.cloud is not None and self.cloud.source is not None:
            return self.cloud.source.with_name(f"{self.cloud.source.stem}_referenced.ply")
        return Path("cloud_referenced.ply")

    def export_current(self) -> None:
        """Write the current cloud and its sidecar to the path in the UI."""
        if self.cloud is None:
            self.status_text = "Load a cloud first."
            return
        if self._gizmo_guard():
            return
        path = Path(self.export_path_text or str(self.default_export_path()))
        if not path.suffix:
            path = path.with_suffix(".ply")
        try:
            ply_path, json_path = export.export_referenced(
                path,
                self.cloud,
                matrix=self._axes,
                translation=self._translation,
                source_convention=self.source_convention,
                target_convention=self.target_convention,
                reference_index=self.origin_index,
                reference_original=self.origin_original,
            )
        except OSError as error:
            self.status_text = f"Export failed: {error}"
            return
        self.export_path_text = str(ply_path)
        self.status_text = f"Exported {ply_path} (+ {json_path.name})"

    # ----------------------------------------------------------- main loop

    def _callback(self) -> None:
        """Per-frame UI callback invoked by Polyscope."""
        self._poll_gizmo()
        ui.draw_file_panel(self)
        ui.draw_reference_panel(self)
        ui.draw_axes_panel(self)
        ui.draw_transform_panel(self)
        ui.draw_export_panel(self)
        ui.draw_shortcuts_panel(self)
        self._handle_pick_input()
        self._handle_shortcuts()
        self._rebuild_if_dirty()

    def _handle_shortcuts(self) -> None:
        """Process the global keyboard shortcuts.

        Shortcuts are ignored while a text field wants keyboard input.
        """
        io = psim.GetIO()
        if io.WantTextInput:
            return
        if psim.IsKeyPressed(psim.ImGuiKey_Escape) and self.pick_active:
            self.set_pick_active(False)
            self.status_text = "Navigation mode."
        for key, action in ACTION_KEYS.items():
            if psim.IsKeyPressed(getattr(psim, f"ImGuiKey_{key.upper()}")):
                getattr(self, action)()
        if psim.IsKeyPressed(psim.ImGuiKey_Enter):
            self.set_origin_from_candidate()
        self._handle_ctrl_shortcuts(io)
        if psim.IsKeyPressed(psim.ImGuiKey_Q):
            ps.unshow()

    def _handle_ctrl_shortcuts(self, io: Any) -> None:
        """Process the Ctrl-modified undo/redo shortcuts.

        Parameters
        ----------
        io : Any
            Current frame ImGui input state.
        """
        if not io.KeyCtrl:
            return
        if psim.IsKeyPressed(psim.ImGuiKey_Z):
            self.undo()
        if psim.IsKeyPressed(psim.ImGuiKey_R) or psim.IsKeyPressed(psim.ImGuiKey_Y):
            self.redo()

    def _on_files_dropped(self, paths: list[str]) -> None:
        """Load the first cloud file dropped onto the window.

        Parameters
        ----------
        paths : list of str
            Paths of the dropped files.
        """
        for path in paths:
            if loader.is_supported(Path(path)):
                self.load_cloud(path)
                return
            self.status_text = f"Unsupported file: {path}"

    def _rebuild_if_dirty(self) -> None:
        """Rebuild the Polyscope structures when any state is stale."""
        touched = self._dirty or self._candidate_dirty or self._axes_dirty
        if self._dirty:
            self._dirty = False
            self._register_cloud()
            if self._home_view_pending:
                self._home_view_pending = False
                ps.reset_camera_to_home_view()
        if self._candidate_dirty:
            self._candidate_dirty = False
            self._rebuild_candidate()
        if self._axes_dirty:
            self._axes_dirty = False
            self._rebuild_axes()
        if touched or self._transform_dirty:
            self._transform_dirty = False
            self._apply_display_transform()
        self._sync_candidate_radius()

    def _register_cloud(self) -> None:
        """Register (or remove) the point cloud and its colors."""
        if self.cloud is None:
            ps.remove_point_cloud(CLOUD_NAME, error_if_absent=False)
            return
        ps_cloud = ps.register_point_cloud(
            CLOUD_NAME,
            self.cloud.points,
            point_render_mode="quad",
        )
        if self.cloud.colors is not None:
            colors = np.asarray(self.cloud.colors, dtype=np.float32)
            ps_cloud.add_color_quantity("RGB", colors, enabled=True)
        else:
            ps_cloud.set_color(UNCOLORED_COLOR)

    def _candidate_radius(self) -> float:
        """Return the absolute radius of the candidate marker.

        The marker is ``CANDIDATE_RADIUS_SCALE`` times larger than the
        current cloud-point radius, so it stays in step with the cloud when
        its radius changes.

        Returns
        -------
        float
            Absolute radius in world units.
        """
        base = 0.0
        if ps.has_point_cloud(CLOUD_NAME):
            base = float(ps.get_point_cloud(CLOUD_NAME).get_radius())
        return base * CANDIDATE_RADIUS_SCALE

    def _sync_candidate_radius(self) -> None:
        """Keep the candidate marker radius in step with the cloud radius.

        The cloud-point radius can be changed from the Polyscope UI; whenever
        it does, the marker is resized so that it stays
        ``CANDIDATE_RADIUS_SCALE`` times larger than the cloud.
        """
        if self.cloud is None or not ps.has_point_cloud(CANDIDATE_NAME):
            return
        target = self._candidate_radius()
        candidate = ps.get_point_cloud(CANDIDATE_NAME)
        if not np.isclose(candidate.get_radius(), target, rtol=1e-3, atol=1e-12):
            candidate.set_radius(target, relative=False)

    def _rebuild_candidate(self) -> None:
        """Register the candidate marker, or remove it when none is picked."""
        if self.cloud is None or self.candidate_index is None:
            ps.remove_point_cloud(CANDIDATE_NAME, error_if_absent=False)
            return
        point = self.cloud.points[self.candidate_index].reshape(1, 3)
        ps_cloud = ps.register_point_cloud(
            CANDIDATE_NAME,
            point,
            color=CANDIDATE_COLOR,
            point_render_mode="quad",
        )
        ps_cloud.set_radius(self._candidate_radius(), relative=False)

    def _rebuild_axes(self) -> None:
        """Draw or remove the three origin axes."""
        for axis, color in AXIS_COLORS.items():
            name = f"{AXES_PREFIX}_{axis}"
            if not self.show_axes or self.cloud is None:
                ps.remove_curve_network(name, error_if_absent=False)
                continue
            length = float(ps.get_length_scale()) * AXIS_LENGTH_SCALE
            tip = np.zeros(3, dtype=np.float64)
            tip[{"x": 0, "y": 1, "z": 2}[axis]] = length
            nodes = np.vstack((np.zeros(3, dtype=np.float64), tip))
            edges = np.array([[0, 1]], dtype=np.int32)
            curve = ps.register_curve_network(name, nodes, edges)
            curve.set_color(color)
