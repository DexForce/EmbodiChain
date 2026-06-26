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

"""Dispatch benchmarks for all atomic actions.

Run a single action benchmark or all action benchmarks in sequence.
Run: python -m scripts.benchmark.atomic_action.run_benchmark --action press
"""

from __future__ import annotations

import argparse
from pathlib import Path


ACTION_MODULES = {
    "move_end_effector": "scripts.benchmark.atomic_action.move_end_effector_benchmark",
    "move_joints": "scripts.benchmark.atomic_action.move_joints_benchmark",
    "pick_up": "scripts.benchmark.atomic_action.pickup_benchmark",
    "move_held_object": "scripts.benchmark.atomic_action.move_held_object_benchmark",
    "place": "scripts.benchmark.atomic_action.place_benchmark",
    "press": "scripts.benchmark.atomic_action.press_benchmark",
}
DEFAULT_ACTIONS = tuple(ACTION_MODULES.keys())


def add_benchmark_args(parser: argparse.ArgumentParser) -> None:
    """Add atomic-action aggregate benchmark CLI arguments."""
    parser.add_argument(
        "--action",
        nargs="+",
        choices=(*ACTION_MODULES.keys(), "all"),
        default=["press"],
        help="Atomic action benchmark(s) to run. Use 'all' for every action.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Forward --smoke to each selected benchmark.",
    )


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run one or more atomic-action benchmarks."
    )
    add_benchmark_args(parser)
    return parser.parse_args()


def _selected_actions(actions: list[str]) -> list[str]:
    """Resolve selected action names."""
    if "all" in actions:
        return list(DEFAULT_ACTIONS)
    return actions


def _make_child_args(smoke: bool) -> argparse.Namespace:
    """Build minimal child benchmark arguments for aggregate dispatch."""
    return argparse.Namespace(
        smoke=smoke,
        repeat=1,
        device="cpu",
        renderer="auto",
        object_types=["bottle", "mug"],
        position_cases=["all"],
        press_tolerance=0.01,
        pose_cases=["all"],
        sequence_cases=["all"],
        approach_cases=["top"],
        place_cases=["all"],
        held_object_cases=["all"],
        n_sample=1000 if smoke else 10000,
        force_reannotate=False,
    )


def run_all_benchmarks(args: argparse.Namespace | None = None) -> list[Path]:
    """Run selected atomic-action benchmarks."""
    args = _parse_args() if args is None else args
    selected_actions = _selected_actions(args.action)
    reports: list[Path] = []

    for action_name in selected_actions:
        module_name = ACTION_MODULES[action_name]
        module = __import__(module_name, fromlist=["run_all_benchmarks"])
        child_args = _make_child_args(smoke=args.smoke)
        report_path = module.run_all_benchmarks(child_args)
        reports.append(report_path)

    return reports


def main() -> None:
    """Run the CLI entry point."""
    try:
        reports = run_all_benchmarks()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    if reports:
        print("Generated reports:")
        for report in reports:
            print(f"  {report}")


if __name__ == "__main__":
    main()


__all__ = ["add_benchmark_args", "run_all_benchmarks"]
