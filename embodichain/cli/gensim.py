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

"""Lightweight routing for the GenSim domain and workflow commands."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import os
from pathlib import Path
import sys

from embodichain.cli.main import _load_handler

__all__: list[str] = []

_DOMAINS = {
    "asset": {
        "ingest": (
            "embodichain.gen_sim.asset_engine.cli.start:main",
            "Convert an existing asset directory into a SimReady asset.",
        ),
    },
    "scene": {
        "generate": (
            "embodichain.gen_sim.scene_engine.cli.start:main",
            "Generate a scene export from an image.",
        ),
        "edit": (
            "embodichain.gen_sim.scene_engine.cli.start:main",
            "Edit a selected scene workspace in place.",
        ),
        "expand": (
            "embodichain.gen_sim.scene_engine.cli.randomization:main",
            "Generate and save layout variants of an existing scene.",
        ),
        "preview": (
            "embodichain.lab.scripts.preview_scene:main",
            "Preview an exported standard USD scene through Lab.",
        ),
    },
    "task": {
        "plan": (
            "embodichain.gen_sim.task_engine.cli:main",
            "Plan image-grounded Semantic Calls without simulation.",
        ),
    },
}


def main(argv: Sequence[str] | None = None) -> None:
    """Dispatch one domain operation or a complete GenSim workflow.

    Args:
        argv: Arguments after ``gensim``; defaults to the process arguments.
    """
    parser = argparse.ArgumentParser(prog="embodichain gensim", description=__doc__)
    commands = parser.add_subparsers(dest="domain")
    for domain, operations in _DOMAINS.items():
        group = commands.add_parser(domain, help=f"GenSim {domain} operations.")
        group.set_defaults(_parser=group)
        actions = group.add_subparsers(dest="operation")
        for operation, (target, description) in operations.items():
            leaf = actions.add_parser(operation, add_help=False, help=description)
            leaf.set_defaults(_target=target, _prog=leaf.prog)
    run = commands.add_parser(
        "run", add_help=False, help="Generate, bind and execute a Task Program."
    )
    run.set_defaults(_target="embodichain.gen_sim._workflow:main", _prog=run.prog)
    ui = commands.add_parser("ui", help="Launch the Gradio workspace.")
    ui.add_argument("--host", help="Override GRADIO_SERVER_NAME.")
    ui.add_argument("--port", type=int, help="Override GRADIO_SERVER_PORT.")
    args, remaining = parser.parse_known_args(argv)
    if args.domain == "ui":
        if remaining:
            ui.error(f"unrecognized arguments: {' '.join(remaining)}")
        if args.port is not None and not 1 <= args.port <= 65535:
            ui.error("--port must be between 1 and 65535")
        if args.host is not None:
            os.environ["GRADIO_SERVER_NAME"] = args.host
        if args.port is not None:
            os.environ["GRADIO_SERVER_PORT"] = str(args.port)
        # The legacy UI owns its process, signal handlers and local imports.
        # Replacing this process preserves its cleanup and exit status.
        script = Path(__file__).resolve().parents[1] / "gen_sim/gradio_ui/gradio_app.py"
        package_root = str(Path(__file__).resolve().parents[2])
        inherited = os.environ.get("PYTHONPATH")
        os.environ["PYTHONPATH"] = package_root + (
            os.pathsep + inherited if inherited else ""
        )
        os.execv(sys.executable, [sys.executable, str(script)])
        return
    if not hasattr(args, "_target"):
        selected = getattr(args, "_parser", parser)
        if remaining:
            selected.error(f"unrecognized arguments: {' '.join(remaining)}")
        selected.print_help()
        return
    kwargs = {"prog": args._prog}
    if args.domain == "scene" and args.operation in {"generate", "edit"}:
        kwargs["operation"] = args.operation
    _load_handler(args._target)(remaining, **kwargs)
