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

"""Benchmark independent Atomic skills using measured physical success.

Run: embodichain benchmark atomic-action --robot ur5 --skill all --planner toppra
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Sequence
from copy import deepcopy
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from embodichain.lab.scripts._atomic_action.defaults import (
    MULTI_ROBOT_SCENES,
    PLANNERS,
    ROBOTS,
    SCENES,
    _recipes,
    _sample_defaults,
)
from scripts.benchmark.atomic_action.common import format_markdown_table

__all__ = ["main", "run_all_benchmarks"]

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_WORKER_MODULE = "scripts.benchmark.atomic_action.reference_benchmark"


def _positive(value: str) -> int:
    """Decode a positive CLI integer."""
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def _parser() -> argparse.ArgumentParser:
    """Build selectors without importing a simulator or task runtime."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", nargs="+", choices=(*ROBOTS, "all"), default=["ur5"])
    parser.add_argument("--skill", nargs="+", choices=(*SCENES, "all"), default=["all"])
    parser.add_argument(
        "--planner", nargs="+", choices=(*PLANNERS, "all"), default=["trapezoidal"]
    )
    parser.add_argument("--physics", choices=("default", "newton"), default="default")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--repeat", type=_positive, default=1)
    parser.add_argument("--num-envs", "--num_envs", type=_positive, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--renderer",
        choices=("auto", "hybrid", "fast-rt", "rt", "no-render"),
        default="auto",
    )
    parser.add_argument(
        "--headless", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--profile",
        choices=("smoke", "coverage", "full"),
        default="coverage",
        help="coverage/full evaluate all configured expansion recipes.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Select up to four recipes covering both expansion factors.",
    )
    parser.add_argument(
        "--no-expansion", action="store_true", help="Evaluate only the nominal skill."
    )
    parser.add_argument("--recipe-indices", "--recipe_indices", type=int, nargs="+")
    parser.add_argument("--affordance-branches", type=_positive)
    parser.add_argument("--trajectory-variants", type=_positive)
    parser.add_argument(
        "--steps-per-waypoint",
        type=_positive,
        default=16,
        help="Physics drive steps per waypoint (fixed across skills/planners).",
    )
    parser.add_argument(
        "--settle-steps",
        type=_positive,
        default=80,
        help="Steps in each of two terminal settling holds.",
    )
    parser.add_argument(
        "--save-trajectories",
        action="store_true",
        help="Save reference/augmented commands and measured replay traces as .pt files.",
    )
    parser.add_argument("--output-dir", "--output_dir", type=Path)
    parser.add_argument(
        "--timeout",
        type=_positive,
        default=600,
        help="Maximum seconds per robot/skill/planner subprocess.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate independent scene selections and recipes without a simulator.",
    )
    parser.add_argument("--_worker", type=Path, help=argparse.SUPPRESS)
    return parser


def _selection(values: Sequence[str], available: Sequence[str]) -> tuple[str, ...]:
    """Expand all and remove duplicates, preserving order."""
    return tuple(available) if "all" in values else tuple(dict.fromkeys(values))


def _cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Enumerate the direct sample capability matrix, including unsupported pairs."""
    return [
        {
            "case_id": f"{robot}/{skill}/{planner}",
            "robot": robot,
            "skill": skill,
            "planner": planner,
            "supported": robot == "ur5" or skill in MULTI_ROBOT_SCENES,
        }
        for robot in _selection(args.robot, ROBOTS)
        for skill in _selection(args.skill, SCENES)
        for planner in _selection(args.planner, PLANNERS)
    ]


def _configuration(
    case: dict[str, Any], options: dict[str, Any]
) -> tuple[dict[str, Any], tuple[int, ...]]:
    """Resolve sample-owned assets/options and benchmark expansion selection."""
    config = _sample_defaults(
        case["skill"],
        physics=options["physics"],
        robot=case["robot"],
        planner=case["planner"],
    )
    config["runtime"].update(
        {key: options[key] for key in ("num_envs", "device", "renderer", "headless")}
    )
    config["seed"] = options["seeds"][0]
    affordance = config["expansion"]["affordance"]
    trajectory = config["expansion"]["trajectory"]
    if options["no_expansion"]:
        affordance.update(enabled=False, branches_per_family=1)
        trajectory["variants_per_family"] = 1
        trajectory["spatial"]["enabled"] = False
    else:
        if options["affordance_branches"] is not None and affordance["enabled"]:
            affordance["branches_per_family"] = options["affordance_branches"]
        if options["trajectory_variants"] is not None:
            trajectory["variants_per_family"] = options["trajectory_variants"]
    count = len(_recipes(config))
    if options["recipe_indices"] is not None:
        selected = tuple(options["recipe_indices"])
    elif options["smoke"] or options["profile"] == "smoke":
        width = trajectory["variants_per_family"]
        selected = tuple(
            index for index in dict.fromkeys((0, 1, width, width + 1)) if index < count
        )
    else:
        selected = tuple(range(count))
    if (
        not selected
        or len(set(selected)) != len(selected)
        or any(index < 0 or index >= count for index in selected)
    ):
        raise ValueError(f"Recipe indices must be unique and in [0, {count})")
    return config, selected


def _write_result(path: Path, result: dict[str, Any]) -> None:
    """Persist completed batches atomically so later native failures retain them."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _failure_trial(
    seed: int, repeat: int, indices: Sequence[int], reason: str
) -> dict[str, Any]:
    """Count requested rows that could not complete, rather than silently omitting them."""
    return {
        "seed": seed,
        "repeat": repeat,
        "recipe_indices": list(indices),
        "rows": [
            {
                "env_index": row,
                "recipe_index": index,
                "planning_success": False,
                "success": False,
                "failure_stage": "execution",
                "failure_reason": reason,
                "stages": {},
                "metrics": {},
            }
            for row, index in enumerate(indices)
        ],
    }


def _fill_missing(result: dict[str, Any], options: dict[str, Any], reason: str) -> None:
    """Include unmeasured requests in the denominator after timeout or worker failure."""
    finished = {
        (trial["seed"], trial["repeat"], index)
        for trial in result["trials"]
        for index in trial["recipe_indices"]
    }
    for repeat in range(options["repeat"]):
        for seed in options["seeds"]:
            missing = [
                index
                for index in result["recipe_indices"]
                if (seed, repeat, index) not in finished
            ]
            if missing:
                result["trials"].append(_failure_trial(seed, repeat, missing, reason))


def _worker(spec_path: Path) -> None:
    """Build fresh scenes, compile Atomic invocations and measure their physical replay."""
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    directory = spec_path.parent
    case, options = spec["case"], spec["options"]
    result = dict(case, status="failed", trials=[])
    try:
        config, selected = _configuration(case, options)
        result.update(recipe_indices=list(selected), configuration=config)
        _write_result(directory / "configuration.json", config)
        if options["dry_run"]:
            result["status"] = "validated"
        else:
            import torch

            if not torch.cuda.is_available():
                result.update(
                    status="skipped",
                    reason="Simulation requires an available CUDA device",
                )
            elif case["planner"] == "curobo" and not options["device"].startswith(
                "cuda"
            ):
                result.update(status="skipped", reason="cuRobo requires --device cuda")
            elif (
                case["planner"] == "curobo"
                and importlib.util.find_spec("curobo") is None
            ):
                result.update(
                    status="skipped",
                    reason="The optional cuRobo runtime is not installed",
                )
            else:
                from scripts.benchmark.atomic_action.execution import run_batch

                result["status"] = "running"
                _write_result(directory / "result.json", result)
                batch_id = 0
                for repeat in range(options["repeat"]):
                    for seed in options["seeds"]:
                        seeded = deepcopy(config)
                        seeded["seed"] = seed
                        for start in range(0, len(selected), options["num_envs"]):
                            indices = selected[start : start + options["num_envs"]]
                            trial = run_batch(
                                seeded, indices, options, directory, batch_id
                            )
                            trial["repeat"] = repeat
                            result["trials"].append(trial)
                            _write_result(directory / "result.json", result)
                            batch_id += 1
                result["status"] = "completed"
    except Exception as error:
        import traceback

        traceback.print_exc()
        result.update(status="failed", reason=f"{type(error).__name__}: {error}")
        if "recipe_indices" in result:
            _fill_missing(result, options, "planner_exception")
    _write_result(directory / "result.json", result)


def _report(results: list[dict[str, Any]], directory: Path) -> Path:
    """Report physical successes and expansion evidence in exactly three tables."""
    timing, metrics = [], []
    by_planner: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        trials = result.get("trials", [])
        measured = [trial for trial in trials if "cost_time_ms" in trial]
        rows = [row for trial in trials for row in trial["rows"]]
        timing.append(
            {
                "case": result["case_id"],
                "status": result["status"],
                "cost_time_ms": (
                    round(sum(trial["cost_time_ms"] for trial in measured), 3)
                    if measured
                    else "n/a"
                ),
                **{
                    key: (
                        round(sum(trial[key] for trial in measured) / len(measured), 3)
                        if measured
                        else "n/a"
                    )
                    for key in ("cpu_delta_mb", "gpu_delta_mb")
                },
                "peak_gpu_mb": (
                    round(max(trial["peak_gpu_mb"] for trial in measured), 3)
                    if measured
                    else "n/a"
                ),
            }
        )
        stages: dict[str, list[bool]] = defaultdict(list)
        for row in rows:
            for stage, passed in row["stages"].items():
                stages[stage].append(passed)
        metric = {
            "case": result["case_id"],
            "attempted": len(rows),
            "succeeded": sum(row["success"] for row in rows),
            "success_rate": (
                sum(row["success"] for row in rows) / len(rows) if rows else "n/a"
            ),
            "planning_success_rate": (
                sum(row["planning_success"] for row in rows) / len(rows)
                if rows
                else "n/a"
            ),
            "stage_success": {
                stage: f"{sum(values)}/{len(values)}"
                for stage, values in stages.items()
            },
            "spatial_variants_effective": sum(
                row.get("expansion", {}).get("effective_spatial_expansion", False)
                for row in rows
            ),
            "protected_max_delta_rad": max(
                (
                    row.get("expansion", {}).get("protected_max_joint_delta_rad", 0.0)
                    for row in rows
                ),
                default=0.0,
            ),
            "failure_reasons": dict(
                Counter(row["failure_reason"] for row in rows if not row["success"])
            )
            or result.get("reason", ""),
        }
        metrics.append(metric)
        by_planner[result["planner"]].append(metric)
    leaderboard = []
    for planner, items in by_planner.items():
        attempted = sum(item["attempted"] for item in items)
        successes = sum(item["succeeded"] for item in items)
        leaderboard.append(
            {
                "rank": 0,
                "algorithm": planner,
                "attempted": attempted,
                "overall_success_rate": successes / attempted if attempted else "n/a",
            }
        )
    leaderboard.sort(
        key=lambda item: (
            -float(item["overall_success_rate"]) if item["attempted"] else 1.0,
            item["algorithm"],
        )
    )
    for rank, row in enumerate(leaderboard, 1):
        row["rank"] = rank
    lines = [
        "# Independent Atomic Skill Benchmark",
        "",
        "Scenes use shared Atomic sample assets and presets. Each trial directly compiles AtomicActionEngine invocations, drives the robot, and scores measured object/joint state. No Gym or Task Program host is involved.",
        "",
        "`success_rate` is physical skill success. Place and Pour include an explicitly scored prerequisite PickUp; MoveJoint visits three ordered targets; Slide performs one pull. Each batch starts in a fresh scene. No failed row is retried.",
        "",
        "Timing covers planning, augmentation and physical execution/observation; scene construction and persistence are excluded. CPU memory is process RSS; GPU memory is PyTorch allocation, excluding native simulator allocations. Cartesian contact paths use ik_interp and bypass the selected planner. Replay cadence and settling settings are recorded in results.json.",
        "",
    ]
    for title, table in (
        ("Time & Memory", timing),
        ("Success & Other Metrics", metrics),
        ("Leaderboard", leaderboard),
    ):
        escaped = [
            {
                key: str(value).replace("|", "\\|").replace("\n", "<br>")
                for key, value in row.items()
            }
            for row in table
        ]
        lines.extend([f"## {title}", "", *format_markdown_table(escaped), ""])
    lines.extend(
        [
            "Per-row stages, measurements, seeds and expansion provenance: `results.json`. Independent scene selections: per-case `configuration.json`. Optional commands and actual replay traces: `batch_*.pt`.",
            "",
        ]
    )
    path = directory / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_all_benchmarks(args: argparse.Namespace) -> Path:
    """Run independent skills in isolated processes and write an aggregate report.

    Args:
        args: Parsed skill, robot, planner and benchmark options.

    Returns:
        Path to the combined three-table Markdown report.
    """
    directory = (
        (
            args.output_dir
            or Path("outputs/benchmarks")
            / datetime.now().strftime("atomic_skills_%Y%m%d_%H%M%S_%f")
        )
        .expanduser()
        .resolve()
    )
    directory.mkdir(parents=True, exist_ok=True)
    options = {
        key: value
        for key, value in vars(args).items()
        if key not in {"output_dir", "_worker"}
    }
    results = []
    for index, case in enumerate(_cases(args)):
        if not case["supported"]:
            results.append(
                dict(
                    case,
                    status="unsupported",
                    reason="This shared sample scene supports UR5 only",
                    trials=[],
                )
            )
            continue
        # Validate before launching native workers, including a dry run.
        config, selected = _configuration(case, options)
        case_dir = (
            directory / f"{index:03d}_{case['robot']}_{case['skill']}_{case['planner']}"
        )
        case_dir.mkdir(parents=True, exist_ok=True)
        result_path = case_dir / "result.json"
        result_path.unlink(missing_ok=True)
        spec_path = case_dir / "request.json"
        _write_result(spec_path, {"case": case, "options": options})
        print(f"Running {case['case_id']} (log: {case_dir / 'run.log'})", flush=True)
        error = None
        failure_reason = "planner_exception"
        with (case_dir / "run.log").open("w", encoding="utf-8") as log:
            try:
                completed = subprocess.run(
                    [sys.executable, "-m", _WORKER_MODULE, "--_worker", str(spec_path)],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=args.timeout,
                    check=False,
                    cwd=_REPOSITORY_ROOT,
                )
                if completed.returncode:
                    error = f"Worker exited with code {completed.returncode}"
            except subprocess.TimeoutExpired:
                error = f"Worker exceeded {args.timeout}s timeout"
                failure_reason = "timeout"
        result = (
            json.loads(result_path.read_text(encoding="utf-8"))
            if result_path.is_file()
            else dict(
                case, trials=[], recipe_indices=list(selected), configuration=config
            )
        )
        if error is not None or not result_path.is_file():
            result.update(status="failed", reason=error or "Worker produced no result")
            _fill_missing(result, options, failure_reason)
        result["log"] = str(case_dir / "run.log")
        results.append(result)
        _write_result(
            directory / "results.json",
            {"schema_version": 2, "options": options, "cases": results},
        )
        rows = [row for trial in result["trials"] for row in trial["rows"]]
        print(
            f"  {result['status']}: {sum(row['success'] for row in rows)}/{len(rows)} physical successes {result.get('reason', '')}",
            flush=True,
        )
    _write_result(
        directory / "results.json",
        {"schema_version": 2, "options": options, "cases": results},
    )
    report = _report(results, directory)
    print(f"Markdown report saved: {report}", flush=True)
    if any(result["status"] == "failed" for result in results):
        raise RuntimeError(f"Some benchmark cases failed; see {report}")
    return report


def main(argv: Sequence[str] | None = None) -> None:
    """Run independent Atomic skill benchmarks.

    Args:
        argv: Arguments excluding executable and subcommand names.
    """
    parser = _parser()
    args = parser.parse_args(argv)
    if args._worker is not None:
        _worker(args._worker)
        return
    if len(set(args.seeds)) != len(args.seeds) or any(
        not 0 <= seed < 2**32 for seed in args.seeds
    ):
        parser.error("--seeds must contain unique integers in [0, 2**32)")
    if args.no_expansion and any(
        value is not None
        for value in (
            args.recipe_indices,
            args.affordance_branches,
            args.trajectory_variants,
        )
    ):
        parser.error("--no-expansion cannot be combined with expansion selectors")
    if not args.headless and args.renderer == "no-render":
        parser.error("--no-headless requires a window-capable renderer")
    try:
        run_all_benchmarks(args)
    except (RuntimeError, ValueError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
