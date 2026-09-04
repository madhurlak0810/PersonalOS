#!/usr/bin/env python
"""Standalone architecture boundary check.

Same rules as ``tests/architecture/test_boundaries.py``, runnable with nothing
installed: it parses source with ``ast`` rather than importing the package, so
CI can fail a bad dependency before spending time on a dependency install.

    python scripts/check_boundaries.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tests.architecture.boundaries import (  # noqa: E402
    LAYERS,
    iter_source_files,
    layer_for_module,
    module_name_for_path,
    scan_imports,
    violations,
)


def main() -> int:
    """Report violations, returning a process exit code."""
    scan = scan_imports()
    problems = violations(scan)

    orphans = [
        module_name_for_path(path)
        for path in iter_source_files()
        if layer_for_module(module_name_for_path(path)) is None
    ]
    problems.extend(
        f"module '{module}' belongs to no declared layer; add it to LAYERS in "
        f"tests/architecture/boundaries.py"
        for module in orphans
    )

    print(
        f"checked {len(scan.files)} files, {len(scan.edges)} internal imports, "
        f"{len(LAYERS)} layers"
    )

    if problems:
        print(f"\n{len(problems)} architecture boundary violation(s):\n")
        for problem in problems:
            print(f"  - {problem}")
        print("\nSee docs/ARCHITECTURE_BOUNDARIES.md for the rules and how to change them.")
        return 1

    print("architecture boundaries OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
