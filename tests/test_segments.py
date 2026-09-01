"""Tests for the segment manager and its undo/redo support."""

from __future__ import annotations

import numpy as np
import pytest

from pcd_segmentation import config
from pcd_segmentation.core.segments import Segment, SegmentManager


@pytest.fixture
def manager() -> SegmentManager:
    """Return a fresh manager for 100 points."""
    return SegmentManager(100)


def test_initial_state_all_unassigned(manager):
    """A fresh manager has no segments and all points unassigned."""
    assert manager.segments == []
    assert (manager.labels == config.UNASSIGNED).all()


def test_create_segment_assigns_points(manager):
    """Creating a segment with indices assigns exactly those points."""
    seg_id = manager.create_segment("a", (1.0, 0.0, 0.0), np.arange(10))
    assert manager.count(seg_id) == 10
    assert set(manager.indices_of(seg_id)) == set(range(10))
    assert manager.get(seg_id).name == "a"


def test_assign_moves_points_between_segments(manager):
    """Assigning points that belong elsewhere moves them."""
    first = manager.create_segment("first", (1.0, 0.0, 0.0), np.arange(20))
    second = manager.create_segment("second", (0.0, 0.0, 1.0))
    manager.assign(second, np.arange(10))
    assert manager.count(first) == 10
    assert manager.count(second) == 10
    assert set(manager.indices_of(first)) == set(range(10, 20))


def test_unassign(manager):
    """Unassigning points returns them to the unassigned pool."""
    seg_id = manager.create_segment("a", (1.0, 0.0, 0.0), np.arange(10))
    manager.unassign(np.arange(5))
    assert manager.count(seg_id) == 5
    assert (manager.labels[:5] == config.UNASSIGNED).all()


def test_delete_segment_unassigns_points(manager):
    """Deleting a segment removes it and frees its points."""
    seg_id = manager.create_segment("a", (1.0, 0.0, 0.0), np.arange(10))
    manager.delete_segment(seg_id)
    assert not manager.has(seg_id)
    assert (manager.labels == config.UNASSIGNED).all()


def test_merge_moves_points_and_drops_segment(manager):
    """Merging moves all points of the dropped segment to the kept one."""
    keep = manager.create_segment("keep", (1.0, 0.0, 0.0), np.arange(5))
    drop = manager.create_segment("drop", (0.0, 0.0, 1.0), np.arange(5, 15))
    manager.merge(keep, drop)
    assert manager.has(keep)
    assert not manager.has(drop)
    assert manager.count(keep) == 15


def test_merge_same_segment_raises(manager):
    """Merging a segment with itself is rejected."""
    seg_id = manager.create_segment("a", (1.0, 0.0, 0.0))
    with pytest.raises(ValueError):
        manager.merge(seg_id, seg_id)


def test_metadata_operations(manager):
    """Renaming, recoloring, and visibility toggles update the segment."""
    seg_id = manager.create_segment("a", (1.0, 0.0, 0.0))
    manager.rename(seg_id, "b")
    manager.set_color(seg_id, (0.0, 1.0, 0.0))
    manager.set_visible(seg_id, False)
    assert manager.get(seg_id).name == "b"
    assert manager.get(seg_id).color == (0.0, 1.0, 0.0)
    assert not manager.get(seg_id).visible
    manager.set_visible(seg_id, True)
    assert manager.get(seg_id).visible


def test_undo_redo_cycles():
    """Undo/redo restore state for assignment, creation, deletion, and merge."""
    manager = SegmentManager(100)
    seg_id = manager.create_segment("a", (1.0, 0.0, 0.0))
    manager.assign(seg_id, np.arange(5))
    assert manager.undo()
    assert manager.count(seg_id) == 0
    assert manager.redo()
    assert manager.count(seg_id) == 5

    manager = SegmentManager(100)
    seg_id = manager.create_segment("b", (0.0, 1.0, 0.0), np.arange(10))
    assert manager.undo()
    assert not manager.has(seg_id)
    assert manager.redo()
    assert manager.has(seg_id)
    assert manager.count(seg_id) == 10

    manager.delete_segment(seg_id)
    assert manager.undo()
    assert manager.count(seg_id) == 10

    drop = manager.create_segment("drop", (0.0, 0.0, 1.0), np.arange(10, 20))
    manager.merge(seg_id, drop)
    assert manager.undo()
    assert manager.has(drop)
    assert manager.count(drop) == 10
    assert manager.count(seg_id) == 10


def test_redo_cleared_by_new_mutation(manager):
    """A new mutation after undo clears the redo stack."""
    seg_id = manager.create_segment("a", (1.0, 0.0, 0.0))
    manager.assign(seg_id, np.arange(5))
    manager.undo()
    assert manager.can_redo()
    manager.assign(seg_id, np.arange(6, 10))
    assert not manager.can_redo()


def test_undo_stack_is_bounded():
    """The undo stack never exceeds the configured maximum size."""
    manager = SegmentManager(config.MAX_UNDO + 40)
    seg_id = manager.create_segment("a", (1.0, 0.0, 0.0))
    for start in range(config.MAX_UNDO + 20):
        manager.assign(seg_id, [start])
    undo_count = 0
    while manager.undo():
        undo_count += 1
    assert undo_count == config.MAX_UNDO  # the stack is trimmed to MAX_UNDO records


def test_apply_state_replaces_everything(manager):
    """Applying an external state resets segments, labels, and history."""
    manager.create_segment("a", (1.0, 0.0, 0.0), np.arange(5))
    labels = np.full(100, 7, dtype=np.int64)
    segments = [Segment(7, "restored", (0.0, 1.0, 0.0))]
    manager.apply_state(segments, labels)
    assert manager.has(7)
    assert (manager.labels == 7).all()
    assert not manager.can_undo()
    assert not manager.can_redo()


def test_assign_unknown_segment_raises(manager):
    """Assigning to a missing segment id raises."""
    with pytest.raises(KeyError):
        manager.assign(42, np.arange(3))


def test_boolean_mask_indices_accepted(manager):
    """Segment creation accepts a boolean mask instead of indices."""
    mask = np.zeros(100, dtype=bool)
    mask[:7] = True
    seg_id = manager.create_segment("a", (1.0, 0.0, 0.0), mask)
    assert manager.count(seg_id) == 7
