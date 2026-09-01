"""Point cloud input/output routines.

The main I/O backend is ``pyvista`` (``pyvista.read`` / ``PolyData.save``),
which covers PLY and many other mesh formats. Formats not supported by the
bundled VTK build (PCD, LAS/LAZ, XYZ-style text clouds) are handled by small
dedicated readers in :mod:`pcd_segmentation.io.readers`.
"""

from pcd_segmentation.io.exporter import export_segment, export_segments, sanitize_filename
from pcd_segmentation.io.loader import SUPPORTED_EXTENSIONS, is_supported, load_point_cloud

__all__ = [
    "SUPPORTED_EXTENSIONS",
    "is_supported",
    "load_point_cloud",
    "export_segment",
    "export_segments",
    "sanitize_filename",
]
