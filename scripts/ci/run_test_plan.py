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

"""Execute the resource-aware pytest lanes described by ``test-plan.json``."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

__all__ = ["build_commands", "main", "run_plan"]

_LANE_ORDER = ("docs", "fast", "sim", "distributed", "gpu")
_SLOW_LANE_PREFIX = "slow-"
_DISTRIBUTED_TEST = "tests/learning/test_rl_distributed.py"
_PLAN_MODES = {"partial", "docs-only", "full-pr", "full"}


def _load_plan(path: Path) -> dict[str, Any]:
    try:
        plan = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read test plan {path}: {error}") from error
    if not isinstance(plan, dict) or plan.get("version") != 1:
        raise RuntimeError("unsupported or malformed test plan")
    if not isinstance(plan.get("mode"), str) or plan.get("mode") not in _PLAN_MODES:
        raise RuntimeError(f"unsupported test plan mode: {plan.get('mode')!r}")
    if not isinstance(plan.get("lanes"), dict):
        raise RuntimeError("test plan is missing lane selectors")
    return plan


def _safe_selector(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"invalid pytest selector: {value!r}")
    normalized = value.replace("\\", "/")
    if normalized.startswith("/") or ".." in Path(normalized).parts:
        raise RuntimeError(f"unsafe pytest selector: {value!r}")
    if not (normalized == "tests" or normalized.startswith("tests/")):
        raise RuntimeError(f"pytest selector outside tests/: {value!r}")
    if any(char in normalized for char in "*?["):
        raise RuntimeError(f"unexpanded pytest selector glob: {value!r}")
    return normalized


def _lane_expression(lane: str, include_slow: bool) -> str:
    is_slow_lane = lane.startswith(_SLOW_LANE_PREFIX)
    base_lane = lane.removeprefix(_SLOW_LANE_PREFIX) if is_slow_lane else lane
    if is_slow_lane:
        slow = "slow and "
    else:
        slow = "" if include_slow else "not slow and "
    if base_lane == "fast":
        return f"{slow}not requires_sim and not gpu"
    if base_lane == "sim":
        return f"{slow}requires_sim and not gpu"
    if base_lane in {"distributed", "gpu"}:
        return f"{slow}gpu"
    raise ValueError(f"unknown pytest lane: {lane}")


def build_commands(
    plan: dict[str, Any],
    *,
    python_executable: str | None = None,
    lanes: Sequence[str] | None = None,
) -> list[tuple[str, list[str]]]:
    """Build shell-free pytest commands for a serialized test plan.

    Args:
        plan: Parsed output from :mod:`select_tests`.
        python_executable: Python executable used to invoke pytest.
        lanes: Optional lane subset for local debugging.

    Returns:
        Ordered ``(lane, argv)`` pairs. Empty lanes are omitted.
    """
    selected_lanes = tuple(lanes) if lanes else _LANE_ORDER
    mode = str(plan.get("mode", "partial"))
    if mode not in _PLAN_MODES:
        raise RuntimeError(f"unsupported test plan mode: {mode!r}")
    include_slow = mode == "full"
    executable = python_executable or sys.executable
    raw_lane_map = plan.get("lanes", {})
    if not isinstance(raw_lane_map, dict):
        raise RuntimeError("test plan is missing lane selectors")
    raw_slow_lane_map = plan.get("slow_lanes", {})
    if not isinstance(raw_slow_lane_map, dict):
        raise RuntimeError("test plan slow lane selectors must be a mapping")
    unknown_lanes = set(raw_lane_map) - set(_LANE_ORDER)
    if unknown_lanes:
        raise RuntimeError(
            f"unsupported test plan lanes: {sorted(str(lane) for lane in unknown_lanes)!r}"
        )
    unknown_slow_lanes = set(raw_slow_lane_map) - {
        f"{_SLOW_LANE_PREFIX}{lane}" for lane in _LANE_ORDER
    }
    if unknown_slow_lanes:
        raise RuntimeError(
            "unsupported slow test plan lanes: "
            f"{sorted(str(lane) for lane in unknown_slow_lanes)!r}"
        )
    commands: list[tuple[str, list[str]]] = []
    for lane in selected_lanes:
        if lane not in _LANE_ORDER:
            raise ValueError(f"unknown pytest lane: {lane}")
        raw_selectors = raw_lane_map.get(lane, [])
        if not isinstance(raw_selectors, list):
            raise RuntimeError(f"lane {lane!r} selectors must be a list")
        selectors = [_safe_selector(value) for value in raw_selectors]
        if selectors:
            base = [executable, "-m", "pytest", "--durations=20"]
            if lane == "docs":
                command = [
                    *base,
                    *selectors,
                    "-q",
                    "--confcutdir=tests/docs",
                    "-m",
                    "slow or not slow" if include_slow else "not slow",
                ]
                commands.append((lane, command))
            else:
                marker = _lane_expression(lane, include_slow)
                command = [*base, *selectors, "-m", marker]
                if lane == "fast":
                    command.extend(
                        ("--ignore=tests/docs", "-n", "4", "--dist", "loadgroup")
                    )
                elif lane == "sim":
                    command.append("--ignore=tests/docs")
                elif lane == "distributed":
                    command.extend(("--run-gpu", "--ignore=tests/docs"))
                elif lane == "gpu":
                    command.extend(
                        (
                            "--run-gpu",
                            "--ignore=tests/docs",
                            f"--ignore={_DISTRIBUTED_TEST}",
                        )
                    )
                commands.append((lane, command))

        if include_slow:
            continue
        slow_lane = f"{_SLOW_LANE_PREFIX}{lane}"
        raw_slow_selectors = raw_slow_lane_map.get(slow_lane, [])
        if not isinstance(raw_slow_selectors, list):
            raise RuntimeError(f"lane {slow_lane!r} selectors must be a list")
        slow_selectors = [_safe_selector(value) for value in raw_slow_selectors]
        if not slow_selectors:
            continue
        slow_base = [executable, "-m", "pytest", "--durations=20", *slow_selectors]
        if lane == "docs":
            slow_command = [*slow_base, "-q", "--confcutdir=tests/docs", "-m", "slow"]
        else:
            slow_command = [
                *slow_base,
                "-m",
                _lane_expression(slow_lane, include_slow),
            ]
            if lane == "fast":
                slow_command.extend(("--ignore=tests/docs",))
            elif lane == "sim":
                slow_command.append("--ignore=tests/docs")
            elif lane == "distributed":
                slow_command.extend(("--run-gpu", "--ignore=tests/docs"))
            elif lane == "gpu":
                slow_command.extend(
                    (
                        "--run-gpu",
                        "--ignore=tests/docs",
                        f"--ignore={_DISTRIBUTED_TEST}",
                    )
                )
        commands.append((slow_lane, slow_command))
    return commands


def run_plan(
    plan: dict[str, Any],
    *,
    root: str | Path,
    python_executable: str | None = None,
    lanes: Sequence[str] | None = None,
    dry_run: bool = False,
) -> int:
    """Execute all selected lanes and return the first failing exit code.

    Pytest exit code 5 means that a marker-filtered lane had no matching tests;
    an empty optional lane is reported and treated as successful. Collection
    errors and test failures retain their non-zero exit code.
    """
    repository_root = Path(root).resolve()
    commands = build_commands(
        plan,
        python_executable=python_executable,
        lanes=lanes,
    )
    if not commands:
        if plan.get("selectors"):
            raise RuntimeError("test plan contains selectors but no runnable lanes")
        print("No pytest lanes selected by test plan.")
        return 0
    for lane, command in commands:
        print(f"\n[CI] {lane}: {shlex.join(command)}", flush=True)
        if dry_run:
            continue
        started = time.monotonic()
        result = subprocess.run(command, cwd=repository_root, check=False)
        elapsed = time.monotonic() - started
        print(
            f"[CI] {lane}: completed in {elapsed:.1f}s (exit {result.returncode}).",
            flush=True,
        )
        if result.returncode == 5:
            if lane in {"docs", "slow-docs"}:
                print("[CI] docs: no documentation tests were collected.")
                return result.returncode
            print(f"[CI] {lane}: no tests matched its marker expression; continuing.")
            continue
        if result.returncode:
            return result.returncode
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--python", dest="python_executable")
    parser.add_argument("--lane", action="append", dest="lanes")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for CI and local plan inspection."""
    args = _parser().parse_args(argv)
    try:
        plan = _load_plan(args.plan)
        return run_plan(
            plan,
            root=args.repo_root,
            python_executable=args.python_executable,
            lanes=args.lanes,
            dry_run=args.dry_run,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"test plan execution failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
