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
import shlex
import subprocess
import sys
from pathlib import Path

from scripts.benchmark.atomic_action.common import (
    MESH_OBJECT_PRESETS,
    PICKUP_APPROACH_CASES,
    POSITION_CASES,
    add_profile_benchmark_args,
    add_video_benchmark_args,
    resolve_profile,
)


ACTION_MODULES = {
    "move_end_effector": "scripts.benchmark.atomic_action.move_end_effector_benchmark",
    "move_joints": "scripts.benchmark.atomic_action.move_joints_benchmark",
    "pick_up": "scripts.benchmark.atomic_action.pickup_benchmark",
    "move_held_object": "scripts.benchmark.atomic_action.move_held_object_benchmark",
    "place": "scripts.benchmark.atomic_action.place_benchmark",
    "press": "scripts.benchmark.atomic_action.press_benchmark",
}
DEFAULT_ACTIONS = tuple(ACTION_MODULES.keys())
MESH_OBJECT_ACTIONS = {"pick_up", "move_held_object", "place"}
PRESS_OBJECT_TYPES = {"bottle", "mug", "wooden_block", "all"}
MESH_OBJECT_TYPES = {*MESH_OBJECT_PRESETS.keys(), "all"}


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
        help="Alias for --profile smoke.",
    )
    add_profile_benchmark_args(parser)
    parser.add_argument(
        "--object_types",
        nargs="+",
        default=None,
        help=(
            "Optional object presets forwarded to selected object-conditioned "
            "benchmarks. Values must be valid for each selected action."
        ),
    )
    parser.add_argument(
        "--position_cases",
        nargs="+",
        choices=(*POSITION_CASES.keys(), "all"),
        default=None,
        help=(
            "Optional initial position cases forwarded to selected "
            "object-conditioned benchmarks."
        ),
    )
    parser.add_argument(
        "--approach_cases",
        nargs="+",
        choices=(*PICKUP_APPROACH_CASES, "all"),
        default=None,
        help=(
            "Optional PickUp approach cases forwarded to pick/place/held-object "
            "benchmarks."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Simulation device forwarded to each selected benchmark.",
    )
    parser.add_argument(
        "--renderer",
        type=str,
        choices=("auto", "hybrid", "fast-rt", "rt"),
        default="auto",
        help="Renderer backend forwarded to each selected benchmark.",
    )
    add_video_benchmark_args(parser)
    parser.add_argument(
        "--in_process",
        action="store_true",
        help=(
            "Run selected benchmarks in the current Python process. This is useful "
            "for debugging but can leave DexSim resources alive across actions."
        ),
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


def _validate_object_types_for_actions(
    selected_actions: list[str],
    object_types: list[str] | None,
) -> None:
    """Validate optional object filters against selected action namespaces."""
    if not object_types:
        return

    requested = set(object_types)
    validators: dict[str, set[str]] = {}
    for action_name in selected_actions:
        if action_name in MESH_OBJECT_ACTIONS:
            validators[action_name] = MESH_OBJECT_TYPES
        elif action_name == "press":
            validators[action_name] = PRESS_OBJECT_TYPES

    invalid_parts = []
    for action_name, valid_types in validators.items():
        invalid = sorted(requested - valid_types)
        if invalid:
            valid = ", ".join(sorted(valid_types))
            invalid_parts.append(
                f"{action_name}: invalid {invalid}; valid values are {valid}"
            )
    if invalid_parts:
        raise RuntimeError(
            "--object_types must be valid for every selected object-conditioned "
            "benchmark. " + " | ".join(invalid_parts)
        )


def _make_child_args(args: argparse.Namespace) -> argparse.Namespace:
    """Build minimal child benchmark arguments for aggregate dispatch."""
    profile = resolve_profile(args)
    return argparse.Namespace(
        smoke=profile == "smoke",
        profile=profile,
        repeat=1,
        device=args.device,
        renderer=args.renderer,
        object_types=args.object_types,
        position_cases=args.position_cases,
        press_tolerance=0.01,
        pose_cases=["all"],
        sequence_cases=["all"],
        approach_cases=args.approach_cases,
        place_cases=None,
        held_object_cases=None,
        n_sample=1000 if profile == "smoke" else 10000,
        force_reannotate=False,
        record_video=args.record_video,
        record_failed_video=args.record_failed_video,
        video_case_limit=args.video_case_limit,
        video_dir=args.video_dir,
        video_fps=args.video_fps,
        video_max_memory=args.video_max_memory,
        video_width=args.video_width,
        video_height=args.video_height,
        video_hold_steps=args.video_hold_steps,
    )


def _make_child_cli_args(args: argparse.Namespace, action_name: str) -> list[str]:
    """Build CLI arguments forwarded to one isolated benchmark subprocess."""
    profile = resolve_profile(args)
    child_args = [
        "--profile",
        profile,
        "--repeat",
        "1",
        "--device",
        args.device,
        "--renderer",
        args.renderer,
        "--video_case_limit",
        str(args.video_case_limit),
        "--video_dir",
        str(args.video_dir),
        "--video_fps",
        str(args.video_fps),
        "--video_max_memory",
        str(args.video_max_memory),
        "--video_width",
        str(args.video_width),
        "--video_height",
        str(args.video_height),
        "--video_hold_steps",
        str(args.video_hold_steps),
    ]
    if action_name in {"pick_up", "move_held_object", "place", "press"}:
        if args.object_types:
            child_args.append("--object_types")
            child_args.extend(args.object_types)
        if args.position_cases:
            child_args.append("--position_cases")
            child_args.extend(args.position_cases)
    if action_name in {"pick_up", "move_held_object", "place"}:
        if args.approach_cases:
            child_args.append("--approach_cases")
            child_args.extend(args.approach_cases)
    if args.record_video:
        child_args.append("--record_video")
        if args.record_failed_video:
            child_args.append("--record_failed_video")
    return child_args


def _run_action_subprocess(action_name: str, args: argparse.Namespace) -> Path:
    """Run one action benchmark in an isolated Python subprocess."""
    module_name = ACTION_MODULES[action_name]
    command = [
        sys.executable,
        "-m",
        module_name,
        *_make_child_cli_args(args, action_name),
    ]
    print("\n" + "=" * 60, flush=True)
    print(
        f"Running {action_name} benchmark in an isolated subprocess",
        flush=True,
    )
    print(
        "Command: " + " ".join(shlex.quote(part) for part in command),
        flush=True,
    )
    print("=" * 60, flush=True)

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    report_path: Path | None = None
    if process.stdout is not None:
        for line in process.stdout:
            print(line, end="", flush=True)
            if line.startswith("Markdown report saved:"):
                report_path = Path(line.split(":", maxsplit=1)[1].strip())

    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(
            f"{action_name} benchmark subprocess failed with exit code "
            f"{return_code}."
        )
    if report_path is None:
        raise RuntimeError(
            f"{action_name} benchmark subprocess finished without reporting "
            "a markdown report path."
        )
    return report_path


def _run_in_process_benchmarks(
    args: argparse.Namespace,
    selected_actions: list[str],
) -> list[Path]:
    """Run selected benchmarks in the current Python process."""
    reports: list[Path] = []
    for action_name in selected_actions:
        module_name = ACTION_MODULES[action_name]
        module = __import__(module_name, fromlist=["run_all_benchmarks"])
        child_args = _make_child_args(args)
        report_path = module.run_all_benchmarks(child_args)
        reports.append(report_path)
    return reports


def run_all_benchmarks(args: argparse.Namespace | None = None) -> list[Path]:
    """Run selected atomic-action benchmarks."""
    args = _parse_args() if args is None else args
    selected_actions = _selected_actions(args.action)
    _validate_object_types_for_actions(selected_actions, args.object_types)

    if args.in_process:
        return _run_in_process_benchmarks(args, selected_actions)

    return [
        _run_action_subprocess(action_name, args) for action_name in selected_actions
    ]


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
