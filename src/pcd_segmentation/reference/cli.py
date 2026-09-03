"""Command-line entry point of the reference-point utility."""

from __future__ import annotations

import argparse

from pcd_segmentation import __version__
from pcd_segmentation.reference.app import ReferenceApp


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the command line interface.

    Returns
    -------
    argparse.ArgumentParser
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        prog="pcd-reference",
        description="Set a point of a cloud as the origin and convert its axes.",
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Point cloud files to open (PLY, PCD, LAS/LAZ, XYZ, OBJ, ...)",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the application.

    Parameters
    ----------
    argv : list of str or None
        Command line arguments; defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit code.
    """
    args = build_parser().parse_args(argv)
    app = ReferenceApp()
    for path in args.files:
        app.load_cloud(path)
    app.run()
    return 0
