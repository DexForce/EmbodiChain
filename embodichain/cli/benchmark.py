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

"""Unified benchmark CLI.

Usage examples::

    embodichain benchmark rl --tasks push_cube --algorithms ppo --suite default
    embodichain benchmark rl --rebuild-report-only
    embodichain benchmark robotics-kinematic-solver -s pytorch
    embodichain benchmark motion-generation --suite motion_panda_free_space_smoke
    embodichain benchmark motion-generation --suite atomic_franka_pgi_smoke
    embodichain benchmark atomic-skill-profile --smoke
    embodichain benchmark grasp-pose-generator --device cuda
    embodichain benchmark workspace-analyzer
    embodichain benchmark rendering --camera-count 2 --resolution 1280x720
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence


def _run_robotics_kinematic_solver_cli(args: argparse.Namespace) -> None:
    """Run robotics kinematic solver benchmark with forwarded CLI args."""
    from scripts.benchmark.robotics.kinematic_solver.run_benchmark import (
        run_all_benchmarks,
    )

    run_all_benchmarks(selected_solvers=args.solvers)


def _run_rl_cli(_: argparse.Namespace) -> None:
    """Run RL benchmark CLI entrypoint."""
    from scripts.benchmark.rl.run_benchmark import main as rl_main

    rl_main()


def _run_motion_generation_cli(args: argparse.Namespace) -> None:
    """Run the shared Motion Planning and Atomic Skills benchmark."""
    from embodichain_benchmarks.motion_generation.cli import run_from_args

    run_from_args(args)


def _run_atomic_skill_profile_cli(_: argparse.Namespace) -> None:
    """Run individual atomic skill profiling over preset scenes."""
    from scripts.benchmark.atomic_skill.run_benchmark import main as atomic_main

    atomic_main()


def _run_grasp_pose_generator_cli(_: argparse.Namespace) -> None:
    """Run grasp pose generator benchmark CLI entrypoint."""
    from scripts.benchmark.grasp_pose_generator.run_benchmark import main

    main()


def _run_workspace_analyzer_cli(_: argparse.Namespace) -> None:
    """Run workspace analyzer benchmarks."""
    from scripts.benchmark.workspace_analyzer.benchmark_workspace_analyzer import (
        run_all_benchmarks,
    )

    run_all_benchmarks()


def _run_rendering_cli(_: argparse.Namespace) -> None:
    """Run the configurable rendering benchmark entrypoint."""
    from scripts.benchmark.rendering.offscreen_denoising import main as rendering_main

    rendering_main()


def main(argv: Sequence[str] | None = None) -> None:
    """Dispatch to the appropriate benchmark sub-command CLI."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="embodichain benchmark",
        description="EmbodiChain benchmark command-line interface.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # -- rl ------------------------------------------------------------------
    rl_parser = subparsers.add_parser(
        "rl",
        add_help=False,
        help="Run RL benchmark: train, evaluate, aggregate, and report results.",
    )
    rl_parser.set_defaults(func=_run_rl_cli)

    # -- robotics-kinematic-solver -------------------------------------------
    robotics_ks_parser = subparsers.add_parser(
        "robotics-kinematic-solver",
        help="Benchmark the OPW kinematic solver (FK/IK accuracy and speed).",
    )
    robotics_ks_parser.add_argument(
        "--solvers",
        "-s",
        nargs="+",
        choices=("opw", "pytorch", "all"),
        default=["all"],
        help="Solvers to benchmark. Use one or more of: opw, pytorch, all.",
    )
    robotics_ks_parser.set_defaults(func=_run_robotics_kinematic_solver_cli)

    # -- motion-generation ---------------------------------------------------
    motion_generation_parser = subparsers.add_parser(
        "motion-generation",
        help="Compare planners with Motion Planning or Atomic Skills suites.",
        description=(
            "Compare planners with Motion Planning or Atomic Skills suites. "
            "Use --suite to select the cases and robot."
        ),
    )
    if arguments and arguments[0] == "motion-generation":
        from embodichain_benchmarks.motion_generation.cli import add_parser_arguments

        add_parser_arguments(motion_generation_parser)
    motion_generation_parser.set_defaults(func=_run_motion_generation_cli)

    # -- atomic-skill-profile -----------------------------------------------
    atomic_skill_profile_parser = subparsers.add_parser(
        "atomic-skill-profile",
        aliases=["atomic-action-profile"],
        add_help=False,
        help="Profile individual atomic skills over object presets and positions.",
    )
    atomic_skill_profile_parser.set_defaults(func=_run_atomic_skill_profile_cli)

    # -- grasp-pose-generator -----------------------------------------------
    grasp_pose_parser = subparsers.add_parser(
        "grasp-pose-generator",
        add_help=False,
        help="Benchmark grasp sampling and pose selection.",
    )
    grasp_pose_parser.set_defaults(func=_run_grasp_pose_generator_cli)

    # -- workspace-analyzer -------------------------------------------------
    workspace_parser = subparsers.add_parser(
        "workspace-analyzer",
        help="Benchmark workspace analysis operations.",
    )
    workspace_parser.set_defaults(func=_run_workspace_analyzer_cli)

    # -- rendering -----------------------------------------------------------
    rendering_parser = subparsers.add_parser(
        "rendering",
        add_help=False,
        help="Benchmark configurable offscreen rendering and denoising.",
    )
    rendering_parser.set_defaults(func=_run_rendering_cli)

    # -- Parse ---------------------------------------------------------------
    # If no sub-command is given, print help and exit.
    if not arguments or arguments[0] in ("-h", "--help"):
        parser.print_help()
        return

    # Determine which sub-command was selected, then reconstruct argv so
    # that each sub-command's entry point can call ``parse_args()`` normally.
    delegated_commands = {
        "rl",
        "atomic-skill-profile",
        "atomic-action-profile",
        "grasp-pose-generator",
        "rendering",
    }
    if arguments[0] in delegated_commands:
        known, _ = parser.parse_known_args(arguments)
    else:
        known = parser.parse_args(arguments)

    if hasattr(known, "func"):
        # Rewrite sys.argv so the sub-command's argparse sees only its own args.
        subcommand_argv = [
            f"embodichain benchmark {arguments[0]}",
            *arguments[1:],
        ]
        original_argv = sys.argv
        sys.argv = subcommand_argv
        try:
            known.func(known)
        finally:
            sys.argv = original_argv
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()


__all__ = ["main"]
