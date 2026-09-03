"""ImGui panels of the standalone reference-point utility."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polyscope.imgui as psim

from pcd_segmentation.reference import conventions

if TYPE_CHECKING:
    from pcd_segmentation.reference.app import ReferenceApp

KEYBINDINGS: tuple[tuple[str, str], ...] = (
    ("p", "pick a point (select mode)"),
    ("click", "choose the point under the cursor"),
    ("Enter / c", "set the chosen point as the origin (0,0,0)"),
    ("a", "apply the selected axes conversion"),
    ("x", "clear the chosen point"),
    ("Esc", "navigation mode"),
    ("u / Ctrl+Z", "undo"),
    ("Ctrl+R / Ctrl+Y", "redo"),
    ("q", "quit"),
    ("double left-click / Ctrl+Shift+click", "native view recenter"),
)
"""Keyboard shortcuts shown in the help panel."""


def draw_file_panel(app: ReferenceApp) -> None:
    """Draw the cloud loading controls and the status message.

    Parameters
    ----------
    app : ReferenceApp
        Owning application.
    """
    psim.SeparatorText("File")
    changed, text = psim.InputText("Open", app.open_path_text)
    app.open_path_text = text
    psim.SameLine()
    if psim.Button("Load"):
        app.load_path_text()
    psim.TextUnformatted("...or drag & drop a cloud file onto the window.")
    if app.status_text:
        psim.Separator()
        psim.TextWrapped(app.status_text)


def draw_reference_panel(app: ReferenceApp) -> None:
    """Draw the reference-point picking and editing controls.

    Parameters
    ----------
    app : ReferenceApp
        Owning application.
    """
    psim.SeparatorText("Reference point")
    if app.cloud is None:
        psim.TextDisabled("No cloud loaded.")
        return
    _draw_reference_actions(app)
    _draw_reference_info(app)
    psim.Separator()
    _draw_undo_redo(app)


def _draw_reference_actions(app: ReferenceApp) -> None:
    """Draw the pick toggle and the origin/clear buttons."""
    psim.BeginDisabled(app.transform_pending)
    changed, pick_active = psim.Checkbox("Pick a point (p)", app.pick_active)
    if changed:
        app.set_pick_active(pick_active)
    if psim.Button("Set as origin"):
        app.set_origin_from_candidate()
    psim.SameLine()
    if psim.Button("Clear"):
        app.clear_candidate()
    psim.EndDisabled()
    if app.pick_active:
        psim.TextDisabled("Click on a cloud point, then confirm with Enter/c.")


def _draw_reference_info(app: ReferenceApp) -> None:
    """Draw the candidate, origin and gizmo-status readouts."""
    if app.candidate_index is None:
        psim.TextUnformatted("Chosen point: none")
    else:
        psim.TextUnformatted(f"Chosen point: #{app.candidate_index} {app.candidate_text()}")
    psim.TextUnformatted(f"Origin: {app.origin_text()}")
    if app.transform_pending:
        psim.TextDisabled("Gizmo transform pending: editing disabled.")


def _draw_undo_redo(app: ReferenceApp) -> None:
    """Draw the undo and redo buttons."""
    psim.BeginDisabled(not app.can_undo() or app.transform_pending)
    if psim.Button("Undo (u)"):
        app.undo()
    psim.EndDisabled()
    psim.SameLine()
    psim.BeginDisabled(not app.can_redo() or app.transform_pending)
    if psim.Button("Redo (Ctrl+Y)"):
        app.redo()
    psim.EndDisabled()


def draw_axes_panel(app: ReferenceApp) -> None:
    """Draw the axis-convention pickers and the axes overlay toggle.

    Parameters
    ----------
    app : ReferenceApp
        Owning application.
    """
    psim.SeparatorText("Axes convention")
    if app.cloud is None:
        psim.TextDisabled("No cloud loaded.")
        return
    psim.BeginDisabled(app.transform_pending)
    picked = _convention_combo("Source", app.source_convention)
    if picked is not None:
        app.source_convention = picked
    picked = _convention_combo("Target", app.target_convention)
    if picked is not None:
        app.target_convention = picked
    if psim.Button("Apply axes conversion (a)"):
        app.apply_axes()
    changed, show_axes = psim.Checkbox("Show origin axes", app.show_axes)
    if changed:
        app.set_show_axes(show_axes)
    psim.EndDisabled()
    if app.transform_pending:
        psim.TextDisabled("Gizmo transform pending: axes conversion disabled.")
    psim.TextWrapped(conventions.description(app.target_convention))
    if conventions.handedness(app.target_convention) == "left":
        psim.TextDisabled("Left-handed target: the cloud will be mirrored.")
    psim.TextUnformatted(app.axes_matrix_text())


def draw_transform_panel(app: ReferenceApp) -> None:
    """Draw the ImGuizmo transformation gizmo controls.

    Parameters
    ----------
    app : ReferenceApp
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
        psim.TextDisabled("Move/rotate/scale the cloud with the gizmo; the center")
        psim.TextDisabled("pivots on the origin point (or the cloud centroid).")
        return
    if psim.Button("Apply to cloud"):
        app.apply_gizmo_to_cloud()
    psim.SameLine()
    if psim.Button("Reset"):
        app.reset_gizmo()
    psim.TextUnformatted(f"Translation: {app.transform_text()}")


def draw_export_panel(app: ReferenceApp) -> None:
    """Draw the export controls for the current cloud.

    Parameters
    ----------
    app : ReferenceApp
        Owning application.
    """
    psim.SeparatorText("Export")
    if app.cloud is None:
        psim.TextDisabled("No cloud loaded.")
        return
    changed, text = psim.InputText("Output PLY", app.export_path_text)
    app.export_path_text = text
    psim.BeginDisabled(app.transform_pending)
    if psim.Button("Export PLY"):
        app.export_current()
    psim.EndDisabled()
    if app.transform_pending:
        psim.TextDisabled("Apply or reset the gizmo before exporting.")


def draw_shortcuts_panel(app: ReferenceApp) -> None:
    """Draw the collapsible reference panel listing the keybindings.

    Parameters
    ----------
    app : ReferenceApp
        Owning application.
    """
    if not psim.CollapsingHeader("Shortcuts"):
        return
    if not psim.BeginTable("reference_shortcuts", 2, psim.ImGuiTableFlags_RowBg):
        return
    psim.TableSetupColumn("Key")
    psim.TableSetupColumn("Action")
    psim.TableHeadersRow()
    for key, action in KEYBINDINGS:
        psim.TableNextRow()
        psim.TableNextColumn()
        psim.TextUnformatted(key)
        psim.TableNextColumn()
        psim.TextUnformatted(action)
    psim.EndTable()


def _convention_combo(label: str, current: str) -> str | None:
    """Draw a convention picker combo, returning the newly picked triple.

    Parameters
    ----------
    label : str
        Combo label.
    current : str
        Currently selected triple key.

    Returns
    -------
    str or None
        The newly picked triple key, or ``None`` when unchanged.
    """
    if not psim.BeginCombo(label, _convention_label(current)):
        return None
    picked = None
    for triple in conventions.preset_triples():
        if psim.Selectable(_convention_label(triple), current == triple):
            picked = triple
    psim.EndCombo()
    return picked


def _convention_label(triple: str) -> str:
    """Return the text shown for a convention in the source/target combos.

    Real conventions are labelled with the world direction of each data
    axis, e.g. ``RDF (x=right, y=down, z=forward)``, while the ``NONE``
    option keeps its plain key.

    Parameters
    ----------
    triple : str
        A triple returned by :func:`conventions.preset_triples`.

    Returns
    -------
    str
        The label text.
    """
    letters = conventions.axis_letters(triple)
    if not letters:
        return triple
    return f"{triple} ({letters})"
