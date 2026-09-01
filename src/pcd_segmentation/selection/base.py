"""Base class and helpers for interactive selection tools."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from pcd_segmentation.app import App


class SelectionMode(Enum):
    """How a selection computed by a tool combines with the current selection."""

    SET = "set"
    ADD = "add"
    REMOVE = "remove"


def mode_from_modifiers(ctrl: bool, shift: bool) -> SelectionMode:
    """Translate keyboard modifiers into a selection combination mode.

    Parameters
    ----------
    ctrl : bool
        Whether Ctrl is held.
    shift : bool
        Whether Shift is held.

    Returns
    -------
    SelectionMode
        ``REMOVE`` when Ctrl is held, ``ADD`` when Shift is held, ``SET``
        otherwise.
    """
    if ctrl:
        return SelectionMode.REMOVE
    if shift:
        return SelectionMode.ADD
    return SelectionMode.SET


class SelectorTool(ABC):
    """Base class for mouse-driven point selection tools.

    A tool receives the raw ImGui input state on every frame while it is
    active and applies the resulting selections to the owning application.

    Attributes
    ----------
    app : App
        Owning application.
    """

    name: str = "tool"

    def __init__(self, app: App) -> None:
        self.app = app

    @abstractmethod
    def handle_input(self, io: Any) -> None:
        """Consume one frame of ImGui input state."""

    @abstractmethod
    def draw(self) -> None:
        """Draw the tool overlay (polygon, rectangle, ...)."""

    @abstractmethod
    def cancel(self) -> None:
        """Abort the interaction in progress, if any."""

    def apply(self, mask: np.ndarray, mode: SelectionMode) -> None:
        """Translate a visibility mask into a selection update.

        The mask refers to the currently visible points; the application
        maps it back to global point indices.

        Parameters
        ----------
        mask : np.ndarray
            Boolean array over the visible points.
        mode : SelectionMode
            How to combine the result with the current selection.
        """
        self.app.add_selection_from_visible(mask, mode)
