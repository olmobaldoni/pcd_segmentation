"""Core segmentation state and project persistence."""

from pcd_segmentation.core.model import PointCloudData
from pcd_segmentation.core.project import load_project, save_project
from pcd_segmentation.core.segments import Segment, SegmentManager

__all__ = [
    "PointCloudData",
    "Segment",
    "SegmentManager",
    "load_project",
    "save_project",
]
