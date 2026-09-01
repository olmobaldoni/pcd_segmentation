"""Complexity checks for the source tree, powered by complexipy."""

from __future__ import annotations

from pathlib import Path

from complexipy import file_complexity

MAX_FUNCTION_COMPLEXITY = 10
"""Maximum cognitive complexity allowed for a single function."""

MAX_FILE_COMPLEXITY = 120
"""Maximum total cognitive complexity allowed for a single file."""

SOURCE_DIR = Path(__file__).resolve().parents[1] / "src" / "pcd_segmentation"


def test_source_complexity_within_bounds():
    """Every function of the package stays below the complexity threshold."""
    files = sorted(SOURCE_DIR.rglob("*.py"))
    assert files, f"No source files found under {SOURCE_DIR}"
    violations = []
    for path in files:
        result = file_complexity(str(path))
        if result.complexity > MAX_FILE_COMPLEXITY:
            violations.append(f"{path.name}: file complexity {result.complexity}")
        for function in result.functions:
            if function.complexity > MAX_FUNCTION_COMPLEXITY:
                violations.append(
                    f"{path.name}:{function.name} has complexity {function.complexity}"
                )
    assert not violations, "Complexity violations:\n" + "\n".join(violations)
