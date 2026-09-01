"""Tests for the point cloud input/output routines."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import pyvista as pv

from pcd_segmentation.core.segments import SegmentManager
from pcd_segmentation.io import exporter, loader

PCD_ASCII = """\
# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z rgb
SIZE 4 4 4 4
TYPE F F F F
COUNT 1 1 1 1
WIDTH 3
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS 3
DATA ascii
0 0 0 1.6711680e+07
1 0 0 6.5254400e+06
0 1 0 2.5652000e+06
"""


def write_ply(path: Path, points: np.ndarray, colors_u8: np.ndarray | None) -> None:
    """Write a test PLY file, with colors when provided."""
    pcd = pv.PolyData(points.astype(np.float32))
    if colors_u8 is not None:
        pcd.point_data["RGB"] = colors_u8.astype(np.uint8)
        pcd.save(str(path), binary=True, texture="RGB")
    else:
        pcd.save(str(path), binary=True)


def test_ply_round_trip(tmp_path, random_cloud):
    """PLY files keep coordinates and colors through load/export cycles."""
    cloud = random_cloud(50)
    colors_u8 = np.round(cloud.colors * 255.0).astype(np.uint8)
    source = tmp_path / "in.ply"
    write_ply(source, cloud.points, colors_u8)
    loaded = loader.load_point_cloud(source)
    assert loaded.num_points == 50
    assert np.allclose(loaded.points, cloud.points, atol=1e-6)
    quantized = np.round(cloud.colors * 255.0) / 255.0
    assert np.allclose(loaded.colors, quantized, atol=1.5 / 255.0)
    target = tmp_path / "out.ply"
    exporter.export_segment(target, loaded.points, colors_u8)
    reloaded = loader.load_point_cloud(target)
    assert np.array_equal(reloaded.points.astype(np.float32), loaded.points.astype(np.float32))
    assert np.array_equal(np.round(reloaded.colors * 255.0), colors_u8)
    plain = tmp_path / "plain.ply"
    plain_cloud = random_cloud(30, colored=False)
    write_ply(plain, plain_cloud.points, None)
    assert loader.load_point_cloud(plain).colors is None


def test_ascii_text_loading(tmp_path):
    """XYZ/CSV text files are parsed, with optional rgb columns."""
    with_colors = tmp_path / "colored.xyz"
    with_colors.write_text(
        "0 0 0 255 0 0\n1 0 0 0 255 0\n0 1 0 0 0 255\n",
        encoding="utf-8",
    )
    loaded = loader.load_point_cloud(with_colors)
    assert loaded.num_points == 3
    assert np.allclose(loaded.colors[0], [1.0, 0.0, 0.0])
    plain = tmp_path / "plain.xyz"
    plain.write_text("0 0 0\n1 1 1\n2 2 2\n", encoding="utf-8")
    loaded = loader.load_point_cloud(plain)
    assert loaded.colors is None
    assert np.allclose(loaded.points[2], [2.0, 2.0, 2.0])
    csv_path = tmp_path / "cloud.csv"
    csv_path.write_text("0.1,0.2,0.3\n0.4,0.5,0.6\n", encoding="utf-8")
    loaded = loader.load_point_cloud(csv_path)
    assert np.allclose(loaded.points[1], [0.4, 0.5, 0.6])


def test_pcd_ascii_loading(tmp_path):
    """An ascii PCD file loads points and unpacked rgb colors."""
    pcd_path = tmp_path / "cloud.pcd"
    pcd_path.write_text(PCD_ASCII, encoding="utf-8")
    loaded = loader.load_point_cloud(pcd_path)
    assert loaded.num_points == 3
    assert np.allclose(loaded.points[1], [1.0, 0.0, 0.0])
    assert loaded.colors is not None
    assert np.allclose(loaded.colors[0], [1.0, 0.0, 0.0])


def test_pcd_binary_loading(tmp_path):
    """A binary PCD file loads points and rgb colors."""
    pcd_path = tmp_path / "cloud_bin.pcd"
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z rgb\n"
        "SIZE 4 4 4 4\n"
        "TYPE F F F F\n"
        "COUNT 1 1 1 1\n"
        "WIDTH 3\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        "POINTS 3\n"
        "DATA binary\n"
    )
    packed = np.array([0xFF0000, 0x00FF00, 0x0000FF], dtype=np.uint32).view(np.float32)
    rows = np.zeros(3, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<f4")])
    rows["x"] = [0.0, 1.0, 0.0]
    rows["y"] = [0.0, 0.0, 1.0]
    rows["rgb"] = packed
    with pcd_path.open("wb") as handle:
        handle.write(header.encode("ascii"))
        handle.write(rows.tobytes())
    loaded = loader.load_point_cloud(pcd_path)
    assert loaded.num_points == 3
    assert np.allclose(loaded.colors[1], [0.0, 1.0, 0.0])


def test_las_round_trip(tmp_path):
    """A LAS file written by laspy loads back with normalized colors."""
    laspy = pytest.importorskip("laspy")
    header = laspy.LasHeader(point_format=2, version="1.2")
    header.scales = [0.01, 0.01, 0.01]
    las = laspy.LasData(header)
    las.x = np.arange(10.0)
    las.y = np.zeros(10)
    las.z = np.zeros(10)
    las.red = np.full(10, 65535, dtype=np.uint16)
    las.green = np.zeros(10, dtype=np.uint16)
    las.blue = np.zeros(10, dtype=np.uint16)
    las_path = tmp_path / "cloud.las"
    las.write(las_path)
    loaded = loader.load_point_cloud(las_path)
    assert loaded.num_points == 10
    assert np.allclose(loaded.points[:, 0], np.arange(10.0) * 0.01)
    assert loaded.colors is not None
    assert np.allclose(loaded.colors[0], [1.0, 0.0, 0.0])


def test_unsupported_extension_raises(tmp_path):
    """Loading an unsupported extension raises ValueError."""
    bogus = tmp_path / "cloud.unknownext"
    bogus.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError):
        loader.load_point_cloud(bogus)


def test_missing_file_raises(tmp_path):
    """Loading a missing file raises ValueError."""
    with pytest.raises(ValueError):
        loader.load_point_cloud(tmp_path / "missing.ply")


def test_export_segments_names_and_colors(tmp_path, random_cloud):
    """Segment export writes one sanitized file per segment with colors."""
    cloud = random_cloud(40)
    manager = SegmentManager(cloud.num_points)
    manager.create_segment("ground plane", (1.0, 0.0, 0.0), np.arange(10))
    manager.create_segment("roof", (0.0, 0.0, 1.0), np.arange(10, 25))
    cloud.source = tmp_path / "city.ply"
    written = exporter.export_segments(tmp_path / "out", cloud, manager)
    names = {path.name for path in written}
    assert names == {"city_seg_ground_plane.ply", "city_seg_roof.ply"}
    first = loader.load_point_cloud(written[0])
    assert first.num_points == 10


def test_export_segments_fallback_color(tmp_path, random_cloud):
    """Clouds without original colors are exported with the segment color."""
    cloud = random_cloud(20, colored=False)
    manager = SegmentManager(cloud.num_points)
    manager.create_segment("a", (0.25, 0.5, 1.0), np.arange(5))
    written = exporter.export_segments(tmp_path, cloud, manager)
    loaded = loader.load_point_cloud(written[0])
    assert loaded.colors is not None
    expected = np.round(np.array([0.25, 0.5, 1.0]) * 255.0)
    assert np.array_equal(np.round(loaded.colors * 255.0)[0], expected)


def test_sanitize_filename():
    """Segment names become safe file-name fragments."""
    assert exporter.sanitize_filename("ground plane!") == "ground_plane"
    assert exporter.sanitize_filename("a/b\\c") == "a_b_c"
    assert exporter.sanitize_filename("___") == "segment"


def test_point_cloud_data_property(random_cloud):
    """The num_points property reports the cloud size."""
    cloud = random_cloud(12)
    assert cloud.num_points == 12
