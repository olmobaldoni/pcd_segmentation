"""Segment bookkeeping with undo/redo support."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from pcd_segmentation import config

UndoAction = tuple[str, Callable[[], None], Callable[[], None]]
"""Undo record: ``(description, undo callable, redo callable)``."""


@dataclass
class Segment:
    """A labeled subset of points of the cloud.

    Attributes
    ----------
    id : int
        Unique identifier of the segment.
    name : str
        User-facing name of the segment.
    color : tuple of float
        RGB color in ``[0, 1]`` used to render the segment.
    visible : bool
        Whether the points of the segment are currently rendered.
    """

    id: int
    name: str
    color: tuple[float, float, float]
    visible: bool = True


class SegmentManager:
    """Own per-point labels, the segment list, and the undo history.

    Each point of the cloud is mapped to either :data:`config.UNASSIGNED`
    or the id of the segment it belongs to. Every mutation is recorded on
    an undo stack so that it can be reverted or re-applied.

    Attributes
    ----------
    labels : np.ndarray
        Array of shape ``(N,)`` mapping each point index to a segment id.
    segments : list of Segment
        Currently defined segments, in creation order.
    """

    def __init__(self, num_points: int) -> None:
        self.labels = np.full(num_points, config.UNASSIGNED, dtype=np.int64)
        self.segments: list[Segment] = []
        self._next_id = 0
        self._undo_stack: list[UndoAction] = []
        self._redo_stack: list[UndoAction] = []

    def get(self, seg_id: int) -> Segment:
        """Return the segment with the given id.

        Parameters
        ----------
        seg_id : int
            Segment id to look up.

        Returns
        -------
        Segment
            The matching segment.

        Raises
        ------
        KeyError
            If no segment with this id exists.
        """
        for segment in self.segments:
            if segment.id == seg_id:
                return segment
        raise KeyError(f"Unknown segment id {seg_id}")

    def has(self, seg_id: int) -> bool:
        """Return whether a segment with the given id exists."""
        return any(segment.id == seg_id for segment in self.segments)

    def indices_of(self, seg_id: int) -> np.ndarray:
        """Return the global point indices assigned to the segment.

        Parameters
        ----------
        seg_id : int
            Segment id.

        Returns
        -------
        np.ndarray
            Array of point indices assigned to the segment.
        """
        return np.flatnonzero(self.labels == seg_id)

    def count(self, seg_id: int) -> int:
        """Return the number of points assigned to the segment."""
        return int(np.count_nonzero(self.labels == seg_id))

    def counts(self) -> dict[int, int]:
        """Return a mapping from segment id to point count."""
        return {segment.id: self.count(segment.id) for segment in self.segments}

    def next_color(self) -> tuple[float, float, float]:
        """Return the next color of the palette for a new segment."""
        return config.SEGMENT_PALETTE[len(self.segments) % len(config.SEGMENT_PALETTE)]

    def create_segment(
        self,
        name: str,
        color: tuple[float, float, float],
        indices: np.ndarray | None = None,
    ) -> int:
        """Create a new segment, optionally assigning points to it.

        Parameters
        ----------
        name : str
            Name of the new segment.
        color : tuple of float
            RGB color in ``[0, 1]``.
        indices : np.ndarray or None
            Point indices to assign to the new segment.

        Returns
        -------
        int
            The id of the newly created segment.
        """
        seg_id = self._next_id
        self._next_id += 1
        segment = Segment(seg_id, name, tuple(color))
        assigned = self._as_indices(indices) if indices is not None else np.empty(0, dtype=np.int64)
        old_labels = self.labels[assigned].copy()
        self.labels[assigned] = seg_id
        self.segments.append(segment)

        def undo() -> None:
            self.labels[assigned] = old_labels
            self.segments.remove(segment)

        def redo() -> None:
            self.labels[assigned] = seg_id
            self.segments.append(segment)

        self._push(f"Create segment '{name}'", undo, redo)
        return seg_id

    def assign(self, seg_id: int, indices: np.ndarray) -> None:
        """Move the given points into the segment.

        Parameters
        ----------
        seg_id : int
            Target segment id.
        indices : np.ndarray
            Point indices to move.

        Raises
        ------
        KeyError
            If no segment with this id exists.
        """
        if not self.has(seg_id):
            raise KeyError(f"Unknown segment id {seg_id}")
        self._set_labels(indices, seg_id, "Assign points to segment")

    def unassign(self, indices: np.ndarray) -> None:
        """Remove the given points from their current segment.

        Parameters
        ----------
        indices : np.ndarray
            Point indices to unassign.
        """
        self._set_labels(indices, config.UNASSIGNED, "Unassign points")

    def delete_segment(self, seg_id: int) -> None:
        """Delete a segment and unassign all of its points.

        Parameters
        ----------
        seg_id : int
            Segment id to delete.

        Raises
        ------
        KeyError
            If no segment with this id exists.
        """
        segment = self.get(seg_id)
        position = self.segments.index(segment)
        indices = np.flatnonzero(self.labels == seg_id)
        self.labels[indices] = config.UNASSIGNED
        self.segments.remove(segment)

        def undo() -> None:
            self.labels[indices] = seg_id
            self.segments.insert(position, segment)

        def redo() -> None:
            self.labels[indices] = config.UNASSIGNED
            self.segments.remove(segment)

        self._push(f"Delete segment '{segment.name}'", undo, redo)

    def merge(self, keep_id: int, drop_id: int) -> None:
        """Merge a segment into another one, discarding the first.

        Parameters
        ----------
        keep_id : int
            Id of the segment that survives the merge.
        drop_id : int
            Id of the segment whose points are moved into ``keep_id``.

        Raises
        ------
        KeyError
            If either segment does not exist.
        """
        if keep_id == drop_id:
            raise ValueError("Cannot merge a segment with itself")
        keep = self.get(keep_id)
        drop = self.get(drop_id)
        position = self.segments.index(drop)
        indices = np.flatnonzero(self.labels == drop_id)
        self.labels[indices] = keep_id
        self.segments.remove(drop)

        def undo() -> None:
            self.labels[indices] = drop_id
            self.segments.insert(position, drop)

        def redo() -> None:
            self.labels[indices] = keep_id
            self.segments.remove(drop)

        self._push(f"Merge '{drop.name}' into '{keep.name}'", undo, redo)

    def rename(self, seg_id: int, name: str) -> None:
        """Rename a segment.

        Parameters
        ----------
        seg_id : int
            Segment id.
        name : str
            New name.

        Raises
        ------
        KeyError
            If no segment with this id exists.
        """
        segment = self.get(seg_id)
        old_name = segment.name
        segment.name = name

        def undo() -> None:
            segment.name = old_name

        def redo() -> None:
            segment.name = name

        self._push(f"Rename segment to '{name}'", undo, redo)

    def set_color(self, seg_id: int, color: tuple[float, float, float]) -> None:
        """Update the display color of a segment.

        Parameters
        ----------
        seg_id : int
            Segment id.
        color : tuple of float
            New RGB color in ``[0, 1]``.

        Raises
        ------
        KeyError
            If no segment with this id exists.
        """
        segment = self.get(seg_id)
        old_color = segment.color
        segment.color = tuple(color)

        def undo() -> None:
            segment.color = old_color

        def redo() -> None:
            segment.color = tuple(color)

        self._push(f"Recolor segment '{segment.name}'", undo, redo)

    def set_visible(self, seg_id: int, visible: bool) -> None:
        """Show or hide a segment in the scene.

        Parameters
        ----------
        seg_id : int
            Segment id.
        visible : bool
            New visibility state.

        Raises
        ------
        KeyError
            If no segment with this id exists.
        """
        segment = self.get(seg_id)
        old_visible = segment.visible
        segment.visible = visible

        def undo() -> None:
            segment.visible = old_visible

        def redo() -> None:
            segment.visible = visible

        self._push("Toggle segment visibility", undo, redo)

    def can_undo(self) -> bool:
        """Return whether an undo operation is available."""
        return bool(self._undo_stack)

    def can_redo(self) -> bool:
        """Return whether a redo operation is available."""
        return bool(self._redo_stack)

    def undo(self) -> bool:
        """Revert the most recent mutation.

        Returns
        -------
        bool
            ``True`` if an operation was reverted, ``False`` if the undo
            stack is empty.
        """
        if not self._undo_stack:
            return False
        description, undo_fn, redo_fn = self._undo_stack.pop()
        undo_fn()
        self._redo_stack.append((description, undo_fn, redo_fn))
        return True

    def redo(self) -> bool:
        """Re-apply the most recently reverted mutation.

        Returns
        -------
        bool
            ``True`` if an operation was re-applied, ``False`` if the redo
            stack is empty.
        """
        if not self._redo_stack:
            return False
        description, undo_fn, redo_fn = self._redo_stack.pop()
        redo_fn()
        self._undo_stack.append((description, undo_fn, redo_fn))
        return True

    def apply_state(self, segments: list[Segment], labels: np.ndarray) -> None:
        """Replace the whole state, discarding the undo history.

        Parameters
        ----------
        segments : list of Segment
            New segment list.
        labels : np.ndarray
            New per-point labels, shape ``(N,)``.
        """
        self.labels = np.asarray(labels, dtype=np.int64).copy()
        self.segments = list(segments)
        self._next_id = max((segment.id for segment in segments), default=-1) + 1
        self._undo_stack.clear()
        self._redo_stack.clear()

    def _set_labels(self, indices: np.ndarray, value: int, description: str) -> None:
        """Overwrite the labels of the given points, recording the change."""
        indices = self._as_indices(indices)
        if len(indices) == 0:
            return
        old_labels = self.labels[indices].copy()
        self.labels[indices] = value

        def undo() -> None:
            self.labels[indices] = old_labels

        def redo() -> None:
            self.labels[indices] = value

        self._push(description, undo, redo)

    def _push(
        self,
        description: str,
        undo_fn: Callable[[], None],
        redo_fn: Callable[[], None],
    ) -> None:
        """Push a new undo record, trimming the stack to its maximum size."""
        self._undo_stack.append((description, undo_fn, redo_fn))
        del self._undo_stack[: -config.MAX_UNDO]
        self._redo_stack.clear()

    @staticmethod
    def _as_indices(indices: np.ndarray) -> np.ndarray:
        """Normalize a boolean mask or an index list into an int64 array."""
        array = np.asarray(indices)
        if array.dtype == bool:
            array = np.flatnonzero(array)
        return array.astype(np.int64, copy=True)
