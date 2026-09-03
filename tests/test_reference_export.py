"""Tests for the PLY export and the JSON sidecar."""

from __future__ import annotations

import json

import numpy as np
import pytest
import pyvista as pv

from pcd_segmentation.core.model import PointCloudData
from pcd_segmentation.reference import export


@pytest.fixture
def cloud() -> PointCloudData:
    """Return a small synthetic colored cloud."""
    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]],
        dtype=np.float64,
    )
    colors = np.array(
        [[255.0, 0.0, 0.0], [0.0, 255.0, 0.0], [0.0, 0.0, 255.0], [128.0, 64.0, 32.0]]
    )
    return PointCloudData(points=points, colors=colors / 255.0)


def test_write_cloud_ply_preserves_points_and_colors(tmp_path, cloud):
    """The exported PLY contains the same points and colors."""
    path = export.write_cloud_ply(tmp_path / "out.ply", cloud)
    assert path.is_file()
    mesh = pv.read(path)
    assert np.allclose(mesh.points, cloud.points, atol=1e-5)
    written = np.asarray(mesh.point_data["RGB"], dtype=np.float64) / 255.0
    assert np.allclose(written, cloud.colors, atol=1e-3)


def test_write_cloud_ply_without_colors(tmp_path):
    """Clouds without colors are exported without an RGB attribute."""
    cloud = PointCloudData(points=np.zeros((3, 3)))
    path = export.write_cloud_ply(tmp_path / "plain.ply", cloud)
    mesh = pv.read(path)
    assert "RGB" not in mesh.point_data


def test_write_sidecar_json(tmp_path, cloud):
    """The sidecar records the mapping and the reference point."""
    matrix = np.diag([1.0, -1.0, -1.0])
    translation = np.array([0.0, -1.0, -2.0])
    path = export.write_sidecar_json(
        tmp_path / "out.json",
        cloud,
        matrix=matrix,
        translation=translation,
        source_convention="RDF",
        target_convention="RUB",
        reference_index=1,
        reference_original=np.array([1.0, 0.0, 0.0]),
    )
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["num_points"] == cloud.num_points
    assert document["reference"]["index"] == 1
    assert document["reference"]["original"] == [1.0, 0.0, 0.0]
    assert document["base_to_current"]["matrix"] == [
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
    ]
    assert document["base_to_current"]["translation"] == [0.0, -1.0, -2.0]
    assert document["axes_conversion"] == {"source": "RDF", "target": "RUB"}


def test_export_referenced_writes_both_files(tmp_path, cloud):
    """export_referenced writes the PLY and its JSON sidecar."""
    matrix = np.eye(3)
    translation = np.array([-0.0, -0.0, -0.0])
    ply_path, json_path = export.export_referenced(
        tmp_path / "result.ply",
        cloud,
        matrix=matrix,
        translation=translation,
        source_convention="NONE",
        target_convention="NONE",
    )
    assert ply_path.is_file()
    assert json_path.is_file()
    assert json_path.name == "result.json"
