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
    embodichain benchmark planners-neural-planner --num-waypoints 1 3 5
    embodichain benchmark atomic-action --smoke
    embodichain benchmark grasp-pose-generator --device cuda
    embodichain benchmark workspace-analyzer
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


def _run_neural_planner_cli(args: argparse.Namespace) -> None:
    """Run NeuralPlanner benchmark with forwarded CLI args."""
    from scripts.benchmark.planners.neural_planner.run_benchmark import (
        run_all_benchmarks,
    )

    run_all_benchmarks(
        num_waypoints_list=args.num_waypoints,
        sim_device=args.device,
        headless=args.headless,
        checkpoint_path=args.checkpoint_path,
        num_trials=args.num_trials,
        warmup_trials=args.warmup_trials,
        sample_interval=args.sample_interval,
        compare_ik=args.compare_ik,
        compare_toppra=args.compare_toppra,
        include_trial_details=args.save_trial_details,
    )


def _run_atomic_action_cli(_: argparse.Namespace) -> None:
    """Run atomic action benchmark CLI entrypoint."""
    from scripts.benchmark.atomic_action.run_benchmark import main as atomic_main

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


def main(argv: Sequence[str] | None = None) -> None:
    """Dispatch to the appropriate benchmark sub-command CLI."""
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

    # -- planners-neural-planner --------------------------------------------
    neural_planner_parser = subparsers.add_parser(
        "planners-neural-planner",
        help="Benchmark NeuralPlanner planning latency and quality on Franka.",
    )
    neural_planner_parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Simulation and planner device. Auto uses CUDA when available.",
    )
    neural_planner_parser.add_argument(
        "--num-waypoints",
        nargs="+",
        type=int,
        default=[1, 3, 5],
        help="Number of EEF waypoints to sweep.",
    )
    neural_planner_parser.add_argument(
        "--num-trials",
        type=int,
        default=8,
        help="Measured trials per (impl, num_waypoints) configuration.",
    )
    neural_planner_parser.add_argument(
        "--warmup-trials",
        type=int,
        default=1,
        help="Warmup trials per configuration; excluded from summary aggregation.",
    )
    neural_planner_parser.add_argument(
        "--sample-interval",
        type=int,
        default=20,
        help="Resampled trajectory length for ik_interpolate and ik_toppra.",
    )
    neural_planner_parser.add_argument(
        "--compare-ik",
        action="store_true",
        help="Also benchmark sequential IK plus joint interpolation.",
    )
    neural_planner_parser.add_argument(
        "--compare-toppra",
        action="store_true",
        help="Also benchmark EEF IK interpolation followed by TOPPRA.",
    )
    neural_planner_parser.add_argument(
        "--save-trial-details",
        action="store_true",
        help="Include per-trial rows in the markdown report.",
    )
    neural_planner_parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Local neural planner checkpoint path. Skips HuggingFace download.",
    )
    neural_planner_parser.add_argument(
        "--headless",
        action="store_true",
        default=True,
        help="Run simulation headlessly (default: True).",
    )
    neural_planner_parser.add_argument(
        "--no-headless",
        action="store_false",
        dest="headless",
        help="Open the simulation viewer window.",
    )
    neural_planner_parser.set_defaults(func=_run_neural_planner_cli)

    # -- atomic-action -------------------------------------------------------
    atomic_action_parser = subparsers.add_parser(
        "atomic-action",
        add_help=False,
        help="Benchmark atomic actions over object presets and positions.",
    )
    atomic_action_parser.set_defaults(func=_run_atomic_action_cli)

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

    # -- Parse ---------------------------------------------------------------
    # If no sub-command is given, print help and exit.
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in ("-h", "--help"):
        parser.print_help()
        return

    # Determine which sub-command was selected, then reconstruct argv so
    # that each sub-command's entry point can call ``parse_args()`` normally.
    delegated_commands = {"rl", "atomic-action", "grasp-pose-generator"}
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
