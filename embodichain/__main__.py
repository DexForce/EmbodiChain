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

"""Unified command-line interface for EmbodiChain.

The ``embodichain`` console script and ``python -m embodichain`` both dispatch
through this module.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from embodichain import __version__


@dataclass(frozen=True)
class Command:
    """Description of one lazily imported CLI command."""

    name: str
    target: str
    help: str


COMMANDS = (
    Command(
        name="data",
        target="embodichain.data.download:main",
        help="List and download EmbodiChain data assets.",
    ),
    Command(
        name="simready",
        target="embodichain.gen_sim.simready_pipeline.cli.start:main",
        help="Convert a raw asset directory into a SimReady asset.",
    ),
    Command(
        name="scene-engine",
        target="embodichain.gen_sim.scene_engine.cli.start:main",
        help="Generate a scene export from an input image using gen_sim/.env.",
    ),
    Command(
        name="preview-scene",
        target="embodichain.gen_sim.scene_engine.cli.preview:main",
        help="Preview a generated Scene Engine scene export.",
    ),
    Command(
        name="preview-asset",
        target="embodichain.lab.scripts.preview_asset:cli",
        help="Preview a USD, URDF, or mesh asset in simulation.",
    ),
    Command(
        name="run-env",
        target="embodichain.lab.scripts.run_env:cli",
        help="Run an environment for data generation or preview.",
    ),
    Command(
        name="train-rl",
        target="embodichain.learning.rl.train:cli",
        help="Train an RL agent from a JSON or YAML config.",
    ),
    Command(
        name="annotate-grasp",
        target="embodichain.toolkits.graspkit.scripts.annotate_grasp:cli",
        help="Interactively annotate a grasp region on a mesh.",
    ),
    Command(
        name="decompose-urdf",
        target="embodichain.toolkits.acd.cli:main",
        help="Generate convex collision meshes for a URDF.",
    ),
    Command(
        name="benchmark",
        target="scripts.benchmark.__main__:main",
        help="Run EmbodiChain performance benchmarks.",
    ),
    Command(
        name="workspace-cache",
        target="embodichain.workspace_cache_cli:main",
        help="Inspect and clean workspace analyzer caches.",
    ),
    Command(
        name="analyze-workspace",
        target="embodichain.lab.scripts.analyze_workspace:cli",
        help="Analyze a robot's reachable workspace from a URDF/USD asset.",
    ),
)


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser.

    Returns:
        The parser used for top-level help and command validation.
    """
    parser = argparse.ArgumentParser(
        prog="embodichain",
        description="EmbodiChain command-line interface.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    subparsers = parser.add_subparsers(
        dest="command",
        metavar="COMMAND",
        title="commands",
    )
    for command in COMMANDS:
        # The command module owns its parser. Disabling help here lets
        # ``embodichain <command> --help`` reach that complete parser.
        subparsers.add_parser(
            command.name,
            add_help=False,
            help=command.help,
            description=command.help,
        )
    return parser


def _load_handler(target: str) -> Callable[[Sequence[str] | None], None]:
    """Load a command handler from a ``module:attribute`` target."""
    module_name, attribute = target.split(":", maxsplit=1)
    module = importlib.import_module(module_name)
    return getattr(module, attribute)


def main(argv: Sequence[str] | None = None) -> None:
    """Dispatch a command through the unified CLI.

    Args:
        argv: Arguments excluding the executable name. Uses ``sys.argv`` when
            omitted.
    """
    parser = build_parser()
    arguments = list(sys.argv[1:] if argv is None else argv)

    if not arguments:
        parser.print_help()
        return

    if arguments[0] in {"-h", "--help"}:
        parser.parse_args(arguments)
        return

    if arguments[0] == "--version":
        parser.parse_args(arguments)
        return

    command_by_name = {command.name: command for command in COMMANDS}
    command = command_by_name.get(arguments[0])
    if command is None:
        parser.error(
            f"argument COMMAND: invalid choice: {arguments[0]!r} "
            f"(choose from {', '.join(command_by_name)})"
        )

    handler = _load_handler(command.target)
    handler(arguments[1:])


if __name__ == "__main__":
    main()


__all__ = ["COMMANDS", "Command", "build_parser", "main"]
