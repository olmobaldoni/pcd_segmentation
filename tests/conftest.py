"""Shared fixtures for the test suite."""

from __future__ import annotations

import numpy as np
import pytest

from pcd_segmentation.core.model import PointCloudData
from pcd_segmentation.selection.geometry import ViewState


def make_view(orthographic: bool = False) -> ViewState:
    """Build a synthetic camera at (0, 0, 5) looking at the origin.

    Parameters
    ----------
    orthographic : bool
        Whether the view uses orthographic projection.

    Returns
    -------
    ViewState
        Camera with 90 degrees vertical fov, 800x600 viewport, length scale 1.
    """
    extrinsics = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, -5.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return ViewState(
        extrinsics=extrinsics,
        fov_deg=90.0,
        aspect=800.0 / 600.0,
        width=800,
        height=600,
        orthographic=orthographic,
        length_scale=1.0,
    )


@pytest.fixture
def view() -> ViewState:
    """Return the synthetic perspective camera."""
    return make_view()


@pytest.fixture
def random_cloud():
    """Return a factory building synthetic colored point clouds.

    Returns
    -------
    callable
        Function ``build(n, colored=True) -> PointCloudData``.
    """

    def build(n: int = 100, colored: bool = True) -> PointCloudData:
        rng = np.random.default_rng(0)
        points = rng.uniform(-1.0, 1.0, size=(n, 3))
        colors = rng.uniform(0.0, 1.0, size=(n, 3)).astype(np.float32) if colored else None
        return PointCloudData(points=points, colors=colors)

    return build
