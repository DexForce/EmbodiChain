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
    embodichain benchmark atomic-action-profile --smoke
    embodichain benchmark grasp-pose-generator --device cuda
    embodichain benchmark workspace-analyzer
    embodichain benchmark rendering --camera-count 2 --resolution 1280x720
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path


def _run_robotics_kinematic_solver_cli(args: argparse.Namespace) -> None:
    """Run robotics kinematic solver benchmark with forwarded CLI args."""
    from scripts.benchmark.robotics.kinematic_solver.run_benchmark import (
        run_all_benchmarks,
    )
    from scripts.benchmark.robotics.kinematic_solver.ssik_benchmark import (
        SSIKBenchmarkOptions,
    )

    options = {
        name.removeprefix("ssik_"): tuple(value) if isinstance(value, list) else value
        for name, value in vars(args).items()
        if name.startswith("ssik_") and value is not None
    }
    run_all_benchmarks(
        selected_solvers=args.solvers, ssik_options=SSIKBenchmarkOptions(**options)
    )


def _add_kinematic_solver_arguments(parser: argparse.ArgumentParser) -> None:
    """Expose the unified runner's arguments without importing numeric modules."""
    solvers = ("opw", "pytorch", "ur", "fep", "differentiable", "franka", "ssik", "all")
    parser.add_argument(
        "--solvers",
        "-s",
        nargs="+",
        choices=solvers,
        default=["all"],
        help="Solvers to benchmark. Franka and SSIK comparisons require explicit selection.",
    )
    for name, description in (
        ("smoke", "Use small SSIK workloads; does not certify acceptance gates."),
        ("skip-warp", "Measure baselines/reference without the Warp backend."),
        ("skip-baseline", "Run diagnostics without comparison baselines."),
        ("quality-only", "Validate SSIK fixtures without latency measurements."),
        ("timing-only", "Measure SSIK latency without rerunning fixture correctness."),
    ):
        parser.add_argument(f"--ssik-{name}", action="store_true", help=description)
    reference = parser.add_mutually_exclusive_group()
    reference.add_argument(
        "--ssik-reference",
        dest="ssik_skip_reference",
        action="store_false",
        help="Align correctness with the optional pinned SSIK CPU reference; exclude reference timing.",
    )
    reference.add_argument(
        "--ssik-skip-reference",
        dest="ssik_skip_reference",
        action="store_true",
        help="Retain the default of no optional SSIK reference import.",
    )
    parser.set_defaults(ssik_skip_reference=True)
    parser.add_argument(
        "--ssik-models",
        nargs="+",
        default=None,
        choices=(
            "aloha_left",
            "aloha_right",
            "marvin_bare_left",
            "marvin_bare_right",
            "marvin_gripper_left",
            "marvin_gripper_right",
            "arx5",
            "flexiv_rizon4s",
        ),
    )
    parser.add_argument(
        "--ssik-reference-artifacts",
        nargs="+",
        default=None,
        metavar="MODEL=PATH",
        help="Associate an SSIK reference artifact with each explicitly selected model.",
    )
    parser.add_argument(
        "--ssik-devices", nargs="+", choices=("cpu", "cuda"), default=["cpu", "cuda"]
    )
    parser.add_argument(
        "--ssik-batch-sizes",
        nargs="+",
        type=int,
        choices=(1, 8, 100, 1000, 10000),
        default=None,
    )
    parser.add_argument(
        "--ssik-baseline-variants",
        nargs="+",
        choices=("original", "matched"),
        default=["original", "matched"],
    )
    scenarios = ("wide", "boundary", "singular", "unreachable", "trajectory")
    parser.add_argument(
        "--ssik-scenarios", nargs="+", choices=scenarios, default=list(scenarios)
    )
    timing_scopes = ("resident", "transfer", "sustained")
    parser.add_argument(
        "--ssik-timing-scopes",
        nargs="+",
        choices=timing_scopes,
        default=list(timing_scopes),
    )
    parser.add_argument("--ssik-cpu-threads", type=int, default=1)
    parser.add_argument("--ssik-artifact-dir", type=Path, default=None)
    parser.add_argument(
        "--ssik-geometry-mode",
        choices=("strict", "near_parallel", "rr"),
        default=None,
        help="Override the selected presets' geometry mode; otherwise inherit each preset.",
    )
    parser.add_argument("--ssik-axis-alignment-tolerance", type=float, default=1e-5)
    parser.add_argument(
        "--ssik-srs-search",
        choices=("uniform", "seed_local"),
        default=None,
        help="Override SRS search; otherwise inherit each preset.",
    )
    parser.add_argument(
        "--ssik-cuda-rr",
        action="store_true",
        help="Add the prepared Rizon4S CUDA RR backend to the selected CUDA workload.",
    )
    parser.add_argument(
        "--ssik-rr-search",
        choices=("fixed", "seed_local"),
        default=None,
        help="Override RR search; presets are inherited, while the legacy CUDA RR comparison defaults to fixed.",
    )
    parser.add_argument(
        "--ssik-robot-e2e",
        action="store_true",
        help="Also measure real Robot.compute_ik in isolated engine-backed workers.",
    )


def _run_rl_cli(_: argparse.Namespace) -> None:
    """Run RL benchmark CLI entrypoint."""
    from scripts.benchmark.rl.run_benchmark import main as rl_main

    rl_main()


def _run_motion_generation_cli(args: argparse.Namespace) -> None:
    """Run the shared Motion Planning and Atomic Skills benchmark."""
    from embodichain_benchmarks.motion_generation.cli import run_from_args

    run_from_args(args)


def _run_atomic_action_profile_cli(_: argparse.Namespace) -> None:
    """Run individual atomic action profiling over preset scenes."""
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
        help="Benchmark analytical and numerical kinematic solvers, including SSIK.",
    )
    _add_kinematic_solver_arguments(robotics_ks_parser)
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

    # -- atomic-action-profile -----------------------------------------------
    atomic_action_profile_parser = subparsers.add_parser(
        "atomic-action-profile",
        add_help=False,
        help="Profile individual atomic actions over object presets and positions.",
    )
    atomic_action_profile_parser.set_defaults(func=_run_atomic_action_profile_cli)

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
