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
    python -m scripts.benchmark robotics-kinematic-solver -s pytorch
    python -m scripts.benchmark planners-neural-planner --num-waypoints 1 3 5
    python -m scripts.benchmark atomic-action --smoke
"""

from __future__ import annotations

import argparse
import sys


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
        device=args.device,
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
        help="Benchmark atomic actions over object presets and positions.",
    )
    from scripts.benchmark.atomic_action.run_benchmark import add_benchmark_args

    add_benchmark_args(atomic_action_parser)
    atomic_action_parser.set_defaults(func=_run_atomic_action_cli)

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
            known.func(known)
        finally:
            sys.argv = original_argv
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()


__all__ = ["main"]
