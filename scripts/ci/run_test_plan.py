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
import os
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence

__all__ = ["build_commands", "main", "run_plan"]

# Run GPU-backed lanes before the long simulation lane.  DexSim owns native
# Vulkan/CUDA resources that can outlive a pytest process briefly; keeping the
# GPU process ahead of that teardown avoids starting it against a fragmented
# device heap.  The workflow passes an explicit order as well, while this
# default keeps local plan execution safe.
_LANE_ORDER = ("docs", "fast", "distributed", "gpu", "sim")
_SLOW_LANE_PREFIX = "slow-"
_DISTRIBUTED_TEST = "tests/learning/test_rl_distributed.py"
_PLAN_MODES = {"partial", "docs-only", "full-pr", "full"}
_CPU_THREAD_ENVIRONMENTS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


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
    slow_commands: list[tuple[str, list[str]]] = []
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
                # Slow CPU selectors are independent by contract.  ``load``
                # distributes parametrized nodes from one module as well;
                # ``loadgroup`` would put the whole module on one worker and
                # leave a large reference matrix effectively serial.
                slow_command.extend(
                    ("--ignore=tests/docs", "-n", "4", "--dist", "load")
                )
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
        slow_commands.append((slow_lane, slow_command))

    # Keep all regular resource lanes together.  An impacted CPU slow test may
    # import a CUDA-backed package even when its marker is CPU-only; running it
    # between the GPU and simulation lanes can retain a native context long
    # enough to make the next lane fail during allocation.  Slow lanes are
    # independent coverage and can safely run after the regular lanes, when no
    # later resource lane needs a clean device.
    return [*commands, *slow_commands]


def _resource_collection_nodes(manifest: Path) -> list[str]:
    """Validate the actual ordered collection before selecting its exact nodes."""
    try:
        selected = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"cannot read resource collection {manifest}: {error}"
        ) from error
    if not isinstance(selected, dict) or selected.get("version") != 1:
        raise RuntimeError("unsupported resource collection manifest")
    files = selected.get("files")
    nodes = selected.get("nodeids")
    markers = selected.get("markers")
    if (
        not isinstance(files, list)
        or not all(isinstance(path, str) for path in files)
        or not isinstance(nodes, list)
        or not all(isinstance(node, str) and "::" in node for node in nodes)
        or not isinstance(markers, dict)
        or set(markers) != set(nodes)
        or not all(
            isinstance(values, list)
            and all(isinstance(value, str) for value in values)
            and values == sorted(set(values))
            for values in markers.values()
        )
    ):
        raise RuntimeError("malformed resource collection manifest")
    for path in files:
        if _safe_selector(path) != path or "::" in path:
            raise RuntimeError(f"invalid resource collection file: {path!r}")
    if files != list(dict.fromkeys(node.split("::", 1)[0] for node in nodes)):
        raise RuntimeError("resource collection files do not match ordered nodes")
    return nodes


def _resource_command_for_nodes(command: list[str], nodes: list[str]) -> list[str]:
    """Keep the original pytest options and select only discovered ordered nodes."""
    marker_index = command.index("-m", 3)
    return [*command[:4], *nodes, *command[marker_index:]]


def _run_resource_lane(lane: str, command: list[str], repository_root: Path) -> int:
    """Discover selected nodes, then run the same ordered collection freshly."""
    with tempfile.TemporaryDirectory(prefix="embodichain-test-selection-") as directory:
        manifest = Path(directory) / "selection.json"
        discovery = [
            *command,
            "--collect-only",
            "-q",
            f"--ci-collect-manifest={manifest}",
        ]
        print(
            f"[CI] {lane}: discover selected nodes: {shlex.join(discovery)}", flush=True
        )
        result = subprocess.run(discovery, cwd=repository_root, check=False, env=None)
        if result.returncode not in (0, 5):
            return result.returncode
        nodes = _resource_collection_nodes(manifest)
        if not nodes:
            return 5
        if result.returncode == 5:
            raise RuntimeError(
                "nonempty resource collection returned pytest exit code 5"
            )
        narrowed = _resource_command_for_nodes(command, nodes)
        narrowed.append(f"--ci-expected-collection={manifest}")
        print(f"[CI] {lane}: selected nodes: {shlex.join(narrowed)}", flush=True)
        return subprocess.run(
            narrowed, cwd=repository_root, check=False, env=None
        ).returncode


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

    Parallel CPU lanes default native thread pools to one thread per worker,
    preserving caller overrides and the parent process environment.

    Serial GPU and simulation lanes discover their selected nodes in a fresh
    collection-only process before execution. Discovery and execution are both
    included in the reported lane time; ordered nodes and markers must match.
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
        environment = None
        if lane in {"fast", "slow-fast"}:
            environment = os.environ.copy()
            for variable in _CPU_THREAD_ENVIRONMENTS:
                environment.setdefault(variable, "1")
        started = time.monotonic()
        if lane.removeprefix(_SLOW_LANE_PREFIX) in {"gpu", "sim"}:
            returncode = _run_resource_lane(lane, command, repository_root)
        else:
            returncode = subprocess.run(
                command, cwd=repository_root, check=False, env=environment
            ).returncode
        elapsed = time.monotonic() - started
        print(
            f"[CI] {lane}: completed in {elapsed:.1f}s (exit {returncode}).",
            flush=True,
        )
        if returncode == 5:
            if lane in {"docs", "slow-docs"}:
                print("[CI] docs: no documentation tests were collected.")
                return returncode
            print(f"[CI] {lane}: no tests matched its marker expression; continuing.")
            continue
        if returncode:
            return returncode
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
