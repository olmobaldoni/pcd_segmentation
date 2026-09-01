"""Tests for the JSON project persistence."""

from __future__ import annotations

import json

import numpy as np
import pytest

from pcd_segmentation.core.project import PROJECT_VERSION, load_project, save_project
from pcd_segmentation.core.segments import SegmentManager


def test_project_round_trip(tmp_path):
    """A saved project restores segments and labels exactly."""
    manager = SegmentManager(50)
    manager.create_segment("a", (1.0, 0.0, 0.0), np.arange(10))
    manager.create_segment("b", (0.0, 1.0, 0.0), np.arange(20, 35))
    manager.set_visible(manager.segments[0].id, False)
    project_path = tmp_path / "sub" / "seg.json"
    save_project(project_path, tmp_path / "cloud.ply", manager.segments, manager.labels)
    state = load_project(project_path)
    assert state["source"] == tmp_path / "cloud.ply"
    assert np.array_equal(state["labels"], manager.labels)
    assert len(state["segments"]) == 2
    restored_a, restored_b = state["segments"]
    assert restored_a.name == "a"
    assert restored_a.color == (1.0, 0.0, 0.0)
    assert not restored_a.visible
    assert restored_b.visible


def test_project_transform_round_trip(tmp_path):
    """The viewer transform is persisted and restored with the project."""
    manager = SegmentManager(10)
    transform_matrix = np.eye(4)
    transform_matrix[:3, 3] = [1.0, 2.0, 3.0]
    project_path = tmp_path / "seg.json"
    save_project(
        project_path,
        tmp_path / "cloud.ply",
        manager.segments,
        manager.labels,
        transform_matrix,
    )
    state = load_project(project_path)
    assert np.allclose(state["transform"], transform_matrix)


def test_project_without_transform_defaults_to_identity(tmp_path):
    """Projects saved without a transform load with an identity matrix."""
    manager = SegmentManager(10)
    project_path = tmp_path / "seg.json"
    save_project(project_path, tmp_path / "cloud.ply", manager.segments, manager.labels)
    state = load_project(project_path)
    assert np.allclose(state["transform"], np.eye(4))


def test_apply_loaded_state(tmp_path):
    """A manager can adopt the state of a loaded project."""
    manager = SegmentManager(50)
    manager.create_segment("a", (1.0, 0.0, 0.0), np.arange(10))
    project_path = tmp_path / "seg.json"
    save_project(project_path, tmp_path / "cloud.ply", manager.segments, manager.labels)
    state = load_project(project_path)
    fresh = SegmentManager(50)
    fresh.apply_state(state["segments"], state["labels"])
    assert np.array_equal(fresh.labels, manager.labels)
    assert fresh.segments[0].name == "a"
    assert not fresh.can_undo()


def test_invalid_projects_raise(tmp_path):
    """Projects with an unknown version or corrupted labels are rejected."""
    bad_version = tmp_path / "bad.json"
    bad_version.write_text(json.dumps({"version": 999}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_project(bad_version)
    manager = SegmentManager(10)
    corrupted = tmp_path / "seg.json"
    save_project(corrupted, tmp_path / "c.ply", [], manager.labels)
    doc = json.loads(corrupted.read_text(encoding="utf-8"))
    doc["num_points"] = 999
    corrupted.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ValueError):
        load_project(corrupted)


def test_version_constant():
    """The project format version is pinned for compatibility checks."""
    assert PROJECT_VERSION == 1
