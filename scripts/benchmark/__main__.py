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

"""Unified CLI entry point for ``python -m scripts.benchmark``.

Usage examples::

    python -m scripts.benchmark rl --tasks push_cube --algorithms ppo --suite default
    python -m scripts.benchmark rl --rebuild-report-only
    python -m scripts.benchmark robotics-kinematic-solver
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    """Dispatch to the appropriate benchmark sub-command CLI."""
    parser = argparse.ArgumentParser(
        prog="scripts.benchmark",
        description="EmbodiChain benchmark command-line interface.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # -- rl ------------------------------------------------------------------
    rl_parser = subparsers.add_parser(
        "rl",
        help="Run RL benchmark: train, evaluate, aggregate, and report results.",
    )
    from scripts.benchmark.rl.run_benchmark import main as rl_main

    rl_parser.set_defaults(func=rl_main)

    # -- robotics-kinematic-solver -------------------------------------------
    robotics_ks_parser = subparsers.add_parser(
        "robotics-kinematic-solver",
        help="Benchmark the OPW kinematic solver (FK/IK accuracy and speed).",
    )
    from scripts.benchmark.robotics.kinematic_solver.opw_solver import (
        benchmark_opw_solver,
    )

    robotics_ks_parser.set_defaults(func=benchmark_opw_solver)

    # -- Parse ---------------------------------------------------------------
    # If no sub-command is given, print help and exit.
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        parser.print_help()
        sys.exit(0)

    # Determine which sub-command was selected, then reconstruct argv so
    # that each sub-command's entry point can call ``parse_args()`` normally.
    known, _ = parser.parse_known_args()

    if hasattr(known, "func"):
        # Rewrite sys.argv so the sub-command's argparse sees only its own args.
        subcommand_argv = [f"scripts.benchmark {sys.argv[1]}"] + sys.argv[2:]
        original_argv = sys.argv
        sys.argv = subcommand_argv
        try:
            known.func()
        finally:
            sys.argv = original_argv
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
