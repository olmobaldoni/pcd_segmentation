"""Standalone utility to re-reference a point cloud on a chosen point.

The user picks a point of the cloud that becomes the new origin
``(0, 0, 0)`` and may optionally re-express the coordinates in a different
axis convention (triples such as ``RDF``, ``RUB`` or ``LUF``). The result is
applied in place and can be exported to PLY together with a JSON sidecar
describing the applied mapping.
"""

from pcd_segmentation.reference.cli import main

__all__ = ["main"]
