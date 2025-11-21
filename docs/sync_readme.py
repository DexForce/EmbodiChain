#!/usr/bin/env python3
"""Sync project README.md into docs/source/introduction.md.

Idempotent copy. Exit code 0 on success.
"""
import shutil
from pathlib import Path
import sys


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    readme = repo_root / "README.md"
    dest = repo_root / "docs" / "source" / "introduction.md"

    if not readme.exists():
        print(f"ERROR: README not found at {readme}")
        return 2

    # Ensure destination directory exists
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Copy file
    shutil.copyfile(readme, dest)
    print(f"Copied {readme} -> {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
