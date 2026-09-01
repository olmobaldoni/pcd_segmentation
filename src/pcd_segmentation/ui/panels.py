"""ImGui panels composing the application UI."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polyscope.imgui as psim

from pcd_segmentation import config

if TYPE_CHECKING:
    from pcd_segmentation.app import App
    from pcd_segmentation.core.segments import Segment

TOOL_NAMES: tuple[tuple[str, str], ...] = (
    ("none", "None (navigate)"),
    ("lasso", "Lasso polygon"),
    ("box", "Box"),
)
"""Tool keys and their display labels."""


def draw_file_panel(app: App) -> None:
    """Draw the file, project, and export controls.

    Parameters
    ----------
    app : App
        Owning application.
    """
    psim.SeparatorText("Files")
    changed, text = psim.InputText("Open", app.open_path_text)
    app.open_path_text = text
    psim.SameLine()
    if psim.Button("Load"):
        app.load_path_text()
    psim.TextUnformatted("...or drag & drop a cloud/project file onto the window.")
    changed, text = psim.InputText("Project", app.project_path_text)
    app.project_path_text = text
    if psim.Button("Save project"):
        app.save_project_text()
    psim.SameLine()
    if psim.Button("Load project"):
        app.load_project_text()
    changed, text = psim.InputText("Export dir", app.export_dir_text)
    app.export_dir_text = text
    if psim.Button("Export segments as PLY"):
        app.export_segments_text()
    if app.status_text:
        psim.Separator()
        psim.TextWrapped(app.status_text)


def draw_toolbar(app: App) -> None:
    """Draw the tool selector and the selection-action controls.

    Parameters
    ----------
    app : App
        Owning application.
    """
    _draw_tool_selector(app)
    psim.SeparatorText("Selection")
    psim.TextUnformatted(f"Selected: {len(app.selection)} points")
    if app.active_tool_name != "none":
        psim.TextDisabled("Ctrl: remove  |  Shift: add")
    psim.TextDisabled("v: lasso | b: box | Esc: navigate | q: quit")
    psim.TextDisabled("c: confirm | s: segment | x: clear | i: invert")
    psim.TextDisabled("u / Ctrl+Z: undo | Ctrl+R / Ctrl+Y: redo")
    if psim.Button("Confirm"):
        app.confirm_selection()
    psim.SameLine()
    if psim.Button("Unassign"):
        app.unassign_selected()
    if psim.Button("Clear"):
        app.clear_selection()
    psim.SameLine()
    if psim.Button("Invert"):
        app.invert_selection()
    _draw_target_combo(app)
    _draw_display_options(app)
    if psim.Button("Undo (Ctrl+Z)"):
        app.undo()
    psim.SameLine()
    if psim.Button("Redo (Ctrl+Y)"):
        app.redo()


def _draw_tool_selector(app: App) -> None:
    """Draw the tool radio buttons, disabled while the gizmo is active."""
    psim.SeparatorText("Tools")
    psim.BeginDisabled(app.gizmo_enabled)
    for key, label in TOOL_NAMES:
        if psim.RadioButton(label, app.active_tool_name == key):
            app.set_active_tool(key)
    psim.EndDisabled()
    if app.gizmo_enabled:
        psim.TextDisabled("Selection tools disabled in gizmo mode.")


def draw_segments_panel(app: App) -> None:
    """Draw the segment list and the merge controls.

    Parameters
    ----------
    app : App
        Owning application.
    """
    psim.SeparatorText("Segments")
    if app.manager is None:
        psim.TextDisabled("No cloud loaded.")
        return
    if psim.Button("Create from selection"):
        app.create_segment_from_selection()
    if not app.manager.segments:
        psim.TextDisabled("No segments yet.")
        return
    _draw_merge_controls(app)
    psim.Separator()
    for segment in app.manager.segments:
        _draw_segment_item(app, segment)


def draw_transform_panel(app: App) -> None:
    """Draw the transformation gizmo controls.

    Parameters
    ----------
    app : App
        Owning application.
    """
    psim.SeparatorText("Transform")
    if app.cloud is None:
        psim.TextDisabled("No cloud loaded.")
        return
    changed, enabled = psim.Checkbox("Transformation gizmo", app.gizmo_enabled)
    if changed:
        app.set_gizmo_enabled(enabled)
    if not app.gizmo_enabled:
        psim.TextDisabled("Move/rotate/scale the cloud with the gizmo.")
        return
    if psim.Button("Apply transform to cloud"):
        app.apply_transform_to_cloud()
    psim.SameLine()
    if psim.Button("Reset"):
        app.reset_transform()
    psim.TextUnformatted(f"Translation: {app.transform_translation_text()}")


def _draw_target_combo(app: App) -> None:
    """Draw the dropdown selecting the target segment for new assignments."""
    target = app.target_segment_name()
    if not psim.BeginCombo("Target segment", target):
        return
    if psim.Selectable("(new segment)", app.target_segment_id == config.UNASSIGNED):
        app.target_segment_id = config.UNASSIGNED
    picked = _pick_segment(app, app.target_segment_id)
    if picked is not None:
        app.target_segment_id = picked
    psim.EndCombo()


def _draw_display_options(app: App) -> None:
    """Draw the scene display toggles."""
    psim.SeparatorText("Display")
    changed, value = psim.Checkbox("Show unassigned points", app.show_unassigned)
    if changed:
        app.show_unassigned = value
        app.mark_dirty()
    changed, value = psim.Checkbox("Original colors (unassigned)", app.show_original_colors)
    if changed:
        app.show_original_colors = value
        app.mark_dirty()


def _draw_merge_controls(app: App) -> None:
    """Draw the two dropdowns and the button used to merge segments."""
    if app.manager is None:
        return
    picked = _combo_pick(app, "Merge keep", app.merge_keep_id)
    if picked is not None:
        app.merge_keep_id = picked
    picked = _combo_pick(app, "Merge drop", app.merge_drop_id)
    if picked is not None:
        app.merge_drop_id = picked
    psim.SameLine()
    if psim.Button("Merge"):
        app.merge_selected_segments()


def _combo_pick(app: App, label: str, current_id: int) -> int | None:
    """Draw a segment picker combo and return the picked segment id.

    Parameters
    ----------
    app : App
        Owning application.
    label : str
        Combo label.
    current_id : int
        Currently selected segment id.

    Returns
    -------
    int or None
        The newly picked segment id, or ``None`` when nothing changed.
    """
    if not psim.BeginCombo(label, app._segment_combo_name(current_id)):
        return None
    picked = _pick_segment(app, current_id)
    psim.EndCombo()
    return picked


def _pick_segment(app: App, current_id: int) -> int | None:
    """Draw the segment entries of an open combo.

    Parameters
    ----------
    app : App
        Owning application.
    current_id : int
        Currently selected segment id.

    Returns
    -------
    int or None
        The segment id clicked by the user, or ``None`` when nothing changed.
    """
    if app.manager is None:
        return None
    for segment in app.manager.segments:
        if psim.Selectable(segment.name, current_id == segment.id):
            return segment.id
    return None


def _draw_segment_item(app: App, segment: Segment) -> None:
    """Draw the editable controls of a single segment.

    Parameters
    ----------
    app : App
        Owning application.
    segment : Segment
        Segment to edit.
    """
    header = f"{segment.name} ({app.manager.count(segment.id)})"
    if not psim.CollapsingHeader(header):
        return
    changed, name = psim.InputText("Name", segment.name)
    if changed and name:
        app.rename_segment(segment.id, name)
    changed, color = psim.ColorEdit3("Color", segment.color)
    if changed:
        app.set_segment_color(segment.id, color)
    changed, visible = psim.Checkbox("Visible", segment.visible)
    if changed:
        app.set_segment_visible(segment.id, visible)
    if psim.Button("Delete segment"):
        app.delete_segment(segment.id)
