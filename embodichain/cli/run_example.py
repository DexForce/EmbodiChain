# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

"""Run Python scripts directly from the ordinary examples directory."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from importlib import metadata
import os
from pathlib import Path
import sys

__all__: list[str] = []


def _examples_root() -> Path:
    repository = Path(__file__).resolve().parents[2]
    source = repository / "examples"
    if (repository / "pyproject.toml").is_file() and source.is_dir():
        return source
    # pip --target relocates data beside site-packages without rewriting its
    # RECORD's prefix-relative paths. Prefer that co-located tree when present.
    relocated = repository / "share/embodichain/examples"
    if (relocated / "__init__.py").is_file():
        return relocated
    try:
        distribution = metadata.distribution("embodichain")
    except metadata.PackageNotFoundError:
        distribution = None
    if distribution is not None:
        marker = ("share", "embodichain", "examples", "__init__.py")
        for entry in distribution.files or ():
            if entry.parts[-4:] == marker:
                root = Path(distribution.locate_file(entry)).resolve().parent
                if root.is_dir():
                    return root
    raise FileNotFoundError(
        "The examples directory is missing. Reinstall EmbodiChain with its example data."
    )


def _example_path(root: Path, name: str) -> Path:
    relative = Path(name)
    if relative.parts[:1] == ("examples",):
        relative = Path(*relative.parts[1:])
    candidate = root / relative
    if candidate.is_dir():
        candidate /= "__main__.py"
    elif candidate.suffix != ".py":
        candidate = Path(f"{candidate}.py")
    candidate = candidate.resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise ValueError("Example paths must stay inside the examples directory.")
    if not candidate.is_file() or candidate.suffix != ".py":
        raise ValueError(
            f"No Python example at {name!r}. Use --list to see script paths."
        )
    return candidate


def _example_names(root: Path) -> list[str]:
    names = []
    for path in root.rglob("*.py"):
        relative = path.relative_to(root)
        if any(
            part.startswith(".") or part == "__pycache__" for part in relative.parts
        ):
            continue
        if not path.resolve().is_relative_to(root.resolve()):
            continue
        if path.name == "__main__.py":
            names.append(relative.parent.as_posix())
        elif not path.name.startswith("_"):
            names.append(relative.with_suffix("").as_posix())
    return sorted(names)


def main(argv: Sequence[str] | None = None) -> None:
    """List example paths or replace this process with the selected script.

    Args:
        argv: Arguments following the command name, or process arguments.
    """
    parser = argparse.ArgumentParser(
        prog="embodichain run-example", description=__doc__
    )
    parser.add_argument(
        "--list", action="store_true", help="List example script paths."
    )
    parser.add_argument(
        "example", nargs="?", help="Path below examples/; .py may be omitted."
    )
    parser.add_argument(
        "arguments", nargs=argparse.REMAINDER, help="Example arguments."
    )
    args = parser.parse_args(argv)
    if args.example is None and not args.list:
        parser.print_help()
        return
    try:
        root = _examples_root()
    except FileNotFoundError as exc:
        parser.error(str(exc))
    if args.list:
        for name in _example_names(root):
            print(name)
        return
    try:
        script = _example_path(root, args.example)
    except ValueError as exc:
        parser.error(str(exc))
    arguments = args.arguments
    if arguments[:1] == ["--"]:
        arguments = arguments[1:]
    runner = Path(__file__).with_name("_example_runner.py").resolve()
    os.execv(
        sys.executable,
        [sys.executable, str(runner), str(root), str(script), *arguments],
    )


def atomic_action_cli(argv: Sequence[str] | None = None) -> None:
    """Forward the existing Atomic Skill command to its ordinary example path.

    Args:
        argv: Arguments following the legacy command name.
    """
    arguments = list(sys.argv[1:] if argv is None else argv)
    main(["sim/atomic_action", *arguments])
