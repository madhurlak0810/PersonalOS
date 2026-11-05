"""Check that the project tree in IMPLEMENTATION_GUIDE.md matches the repository.

    python scripts/check_guide_tree.py           # check; non-zero on drift
    python scripts/check_guide_tree.py --print   # print the directories it expects

The tree is the fenced block between `<!-- tree:begin -->` and
`<!-- tree:end -->`. Three rules, each against the files git tracks (plus
untracked ones that are not ignored, so a new package fails before it is
committed rather than after):

1. **Every path the tree names exists**, unless its comment says `(planned`.
2. **Every directory down to two levels is in the tree** (`personalos/policy`,
   `tests/unit`, ...), hidden ones and `__pycache__` aside. Deeper entries
   and individual files are optional.
3. **"Planned" means not built.** A directory holding nothing but
   placeholders (`.gitkeep`, `__init__.py`) must be marked `(planned`, and one
   marked so must not hold anything else.

Like `check_boundaries.py`, this needs only the standard library and runs in
the boundaries CI job, before anything is installed.
"""

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
GUIDE = REPO / "IMPLEMENTATION_GUIDE.md"
BEGIN, END = "<!-- tree:begin -->", "<!-- tree:end -->"
PLANNED = "(planned"
#: Files that hold a directory's place without building anything in it.
PLACEHOLDERS = frozenset({".gitkeep", "__init__.py"})
#: How deep the tree must list every directory: `a` and `a/b`.
REQUIRED_DEPTH = 2
#: The width of one tree level, as `tree` draws it ("│   ", "├── ").
INDENT = 4
TREE_CHARS = "│├└─  "


def tracked_files() -> set[str]:
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return {line for line in out.splitlines() if (REPO / line).exists()}


def directories(files: set[str]) -> dict[str, set[str]]:
    """Every directory holding a tracked file, mapped to the file names under it."""
    found: dict[str, set[str]] = {}
    for path in files:
        parts = path.split("/")
        for depth in range(1, len(parts)):
            found.setdefault("/".join(parts[:depth]), set()).add(parts[-1])
    return found


def guide_tree(text: str) -> dict[str, str]:
    """The tree block's entries, as repo-relative path -> the comment after it."""
    try:
        block = text.split(BEGIN, 1)[1].split(END, 1)[0]
    except IndexError:
        sys.exit(f"{GUIDE.name}: no {BEGIN} ... {END} block")
    lines = [line for line in block.splitlines() if line.strip() and not line.startswith("```")]

    entries: dict[str, str] = {}
    stack: list[str] = []
    for line in lines[1:]:  # the first line is the repository root
        name_at = len(line) - len(line.lstrip(TREE_CHARS))
        body = line[name_at:]
        if not body or body.startswith("#"):
            continue
        name, _, comment = body.partition("#")
        name = name.strip()
        depth = name_at // INDENT - 1
        stack = stack[:depth]
        path = "/".join([*stack, name.rstrip("/")])
        entries[path] = comment.strip()
        if name.endswith("/"):
            stack.append(name.rstrip("/"))
    return entries


def expected(dirs: dict[str, set[str]]) -> list[str]:
    return sorted(
        path
        for path in dirs
        if path.count("/") < REQUIRED_DEPTH
        and not any(part.startswith(".") or part == "__pycache__" for part in path.split("/"))
    )


def check() -> list[str]:
    files = tracked_files()
    dirs = directories(files)
    entries = guide_tree(GUIDE.read_text())
    problems = []

    for path, comment in entries.items():
        planned = PLANNED in comment
        exists = path in files or path in dirs
        if not exists and not planned:
            problems.append(f"{path}: in the guide's tree but not in the repository")
        if path in dirs:
            built = dirs[path] - PLACEHOLDERS
            if planned and built:
                problems.append(f"{path}: marked planned but holds {', '.join(sorted(built))}")
            if not planned and not built:
                problems.append(f"{path}: holds only placeholders; mark it '{PLANNED} ...)'")

    for path in expected(dirs):
        if path not in entries:
            problems.append(f"{path}/: in the repository but missing from the guide's tree")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--print", action="store_true", help="print the directories the tree must list"
    )
    args = parser.parse_args()
    if args.print:
        print("\n".join(f"{path}/" for path in expected(directories(tracked_files()))))
        return 0

    problems = check()
    if problems:
        print(f"{GUIDE.name}: the project tree has drifted from the repository:")
        print("\n".join(f"  - {problem}" for problem in problems))
        return 1
    print(f"{GUIDE.name}: project tree matches the repository")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
