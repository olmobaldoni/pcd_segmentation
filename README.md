# pcd_segmentation

Interactive manual segmentation of point clouds, built on [Polyscope](https://polyscope.run/py/).

Inspired by the CloudCompare / MeshLab segmentation workflows: draw screen-space
selections, organize them into segments, move the cloud around with an
[ImGuizmo](https://github.com/CedricGuillemet/ImGuizmo) transformation gizmo, and
export each segment as a binary PLY file.

## Installation

The project is managed with [Pixi](https://pixi.sh/):

```bash
pixi install
```

## Usage

```bash
pixi run segment path/to/cloud.ply            # open one or more clouds
pixi run segment -p segmentation.json         # resume a saved project
python -m pcd_segmentation cloud.pcd cloud.xyz
```

Files can also be dragged & dropped onto the viewer window.

### Supported input formats

| Format | Backend |
| ------ | ------- |
| PLY, OBJ, STL, VTK, ... | [pyvista](https://docs.pyvista.org/api/utilities/_autosummary/pyvista.read.html) |
| PCD (ascii / binary) | internal numpy reader |
| LAS / LAZ | [laspy](https://laspy.readthedocs.io/) + lazrs |
| XYZ / TXT / PTS / ASC / CSV | internal numpy reader |

## Workflow

1. **Load a cloud** (menu, CLI argument, or drag & drop).
2. **Select points** with a tool:
   - **Lasso polygon** — click to add vertices, close with Enter/double-click
     or by clicking near the first vertex. `Esc` cancels.
   - **Box** — drag a rectangle on screen.
   - `Ctrl` removes the enclosed points from the selection, `Shift` adds them.
3. **Confirm** the selection (assigns it to the target segment) or press `s`
   to create a new segment from it.
4. **Edit segments**: rename, recolor, hide, delete, or merge two segments.
   Undo/redo any operation.
5. **Move the cloud** with the transformation gizmo (`Transform` panel):
   translate/rotate/scale the gizmo handles, then *Apply transform to cloud*
   to bake the matrix into the point coordinates (or *Reset* to discard).
6. **Export** every segment as `<dir>/<cloud>_seg_<name>.ply` (binary,
   original colors preserved when available).
7. **Save the project** (JSON) to resume later: source cloud, segments,
   labels, and the current transform are persisted.

### Keyboard shortcuts (nvim-style)

| Key | Action |
| --- | ------ |
| `v` | lasso tool |
| `b` | box tool |
| `Esc` | navigation mode |
| `c` | confirm selection |
| `s` | create segment from selection |
| `x` | clear selection |
| `i` | invert selection (over visible points) |
| `u` / `Ctrl+Z` | undo |
| `Ctrl+R` / `Ctrl+Y` | redo |
| `q` | quit |

Shortcuts are disabled while a text field has focus. A reference panel is
available in the sidebar: expand **Shortcuts**.

### Mouse gestures

| Gesture | Action |
| ------- | ------ |
| double right-click on a point | place the transformation gizmo center at that point (the gizmo is shown if hidden, and the active selection tool is switched off) |
| double left-click / Ctrl+Shift+click | native Polyscope view recenter |

## Development

```bash
pixi run test        # pytest suite
pixi run lint        # ruff (E/F/W/I/UP/B/D rules, numpy docstrings)
pixi run complexity  # complexipy, max cognitive complexity 10 per function
pixi run format      # ruff format
```

## Project layout

```
src/pcd_segmentation/
├── app.py             # application state, UI callback, shortcuts, gizmo
├── cli.py             # command line entry point
├── config.py          # constants, palette, key bindings
├── core/              # point cloud model, segments + undo, project, transforms
├── io/                # pyvista-based loading and per-segment PLY export
├── selection/         # projection math and the lasso/box tools
└── ui/                # ImGui panels and screen-space overlay drawing
```
