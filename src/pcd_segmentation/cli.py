"""Command-line entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

from pcd_segmentation import __version__
from pcd_segmentation.app import App


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the command line interface.

    Returns
    -------
    argparse.ArgumentParser
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        prog="pcd-segmentation",
        description="Interactive manual segmentation of point clouds.",
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Point cloud files to open (PLY, PCD, LAS/LAZ, XYZ, OBJ, ...)",
    )
    parser.add_argument(
        "-p",
        "--project",
        help="Segmentation project (JSON) to resume",
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
    app = App()
    if args.project is not None:
        app.project_path_text = str(Path(args.project))
        app.load_project_text()
    for path in args.files:
        app.load_cloud(path)
    app.run()
    return 0
