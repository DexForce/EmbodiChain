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

"""Run the extensible free-space motion-generation benchmark.

cuRobo is the default primary baseline. IK interpolation and TOPPRA are
optional diagnostic baselines. NMG remains an explicitly configurable,
unsupported adapter stub until its production checkpoint contract is ready.

Run: ``embodichain benchmark motion-generation --suite smoke``
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING

from .compat import (
    IMPL_IK,
    IMPL_NEURAL,
    IMPL_TOPPRA,
    QUALITY_SUMMARY_COLUMNS,
    aggregate_legacy_rows,
    format_waypoint_grouped_tables,
)
from .config import PlannerSpecCfg, SuiteCfg, load_suite
from .metrics.trajectory import compute_waypoint_errors, get_pose_err

if TYPE_CHECKING:
    from .runner import BenchmarkRunResult

__all__ = [
    "IMPL_IK",
    "IMPL_NEURAL",
    "IMPL_TOPPRA",
    "QUALITY_SUMMARY_COLUMNS",
    "_aggregate_rows",
    "_format_waypoint_grouped_tables",
    "add_parser_arguments",
    "compute_waypoint_errors",
    "get_pose_err",
    "run_all_benchmarks",
    "run_from_args",
]

_aggregate_rows = aggregate_legacy_rows
_format_waypoint_grouped_tables = format_waypoint_grouped_tables


def add_parser_arguments(parser: argparse.ArgumentParser) -> None:
    """Add free-space benchmark options to an existing argument parser."""
    parser.add_argument(
        "--suite",
        default="smoke",
        help="Suite short name (smoke/coverage) or an explicit YAML path.",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=None,
        help="Override enabled suite algorithms by id.",
    )
    parser.add_argument(
        "--extra-baselines",
        nargs="+",
        choices=("ik_interpolate", "toppra"),
        default=[],
        help="Enable optional diagnostic baselines.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Simulation device; cuRobo itself always requires CUDA.",
    )
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=None)
    parser.add_argument("--num-waypoints", nargs="+", type=int, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--num-trials", type=int, default=None)
    parser.add_argument("--warmup-trials", type=int, default=None)
    parser.add_argument("--sample-interval", type=int, default=None)
    parser.add_argument("--validation-samples", type=int, default=None)
    parser.add_argument("--position-threshold-m", type=float, default=None)
    parser.add_argument("--rotation-threshold-rad", type=float, default=None)
    parser.add_argument(
        "--nmg-pos-eps",
        type=float,
        default=None,
        help="NMG internal waypoint position threshold in metres.",
    )
    parser.add_argument(
        "--nmg-rot-eps",
        type=float,
        default=None,
        help="NMG internal waypoint rotation threshold in radians.",
    )
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        help="Reserved NMG checkpoint path; the current NMG adapter remains a stub.",
    )
    parser.add_argument(
        "--compare-ik",
        action="store_true",
        help="Compatibility alias for --extra-baselines ik_interpolate.",
    )
    parser.add_argument(
        "--compare-toppra",
        action="store_true",
        help="Compatibility alias for --extra-baselines toppra.",
    )
    parser.add_argument(
        "--output-root", default="outputs/benchmarks", help="Artifact root directory."
    )
    parser.add_argument(
        "--headless", action="store_true", default=True, help="Run headlessly."
    )
    parser.add_argument(
        "--no-headless", action="store_false", dest="headless", help="Open a viewer."
    )
    parser.add_argument(
        "--save-trial-details",
        action="store_true",
        help="Deprecated: numeric trial details are always saved to trials.jsonl.",
    )


def _planner_by_id(suite: SuiteCfg, planner_id: str) -> PlannerSpecCfg:
    """Resolve one planner id from the suite with an actionable error."""
    for spec in suite.planners:
        if spec.id == planner_id:
            return spec
    raise ValueError(
        f"Suite {suite.name!r} does not declare planner {planner_id!r}; "
        f"available ids: {[spec.id for spec in suite.planners]}."
    )


def _resolve_planners(
    suite: SuiteCfg,
    algorithms: list[str] | None,
    extra_baselines: list[str],
) -> list[PlannerSpecCfg]:
    """Resolve enabled algorithms while retaining suite ordering and roles."""
    selected_ids = (
        list(algorithms)
        if algorithms is not None
        else [spec.id for spec in suite.planners if spec.enabled]
    )
    selected_ids = list(dict.fromkeys(selected_ids))
    for planner_id in extra_baselines:
        if planner_id not in selected_ids:
            selected_ids.append(planner_id)
    if not selected_ids:
        raise ValueError("No algorithms were selected for the benchmark.")
    return [deepcopy(_planner_by_id(suite, planner_id)) for planner_id in selected_ids]


def _apply_overrides(
    suite: SuiteCfg,
    *,
    batch_sizes: list[int] | None = None,
    num_waypoints: list[int] | None = None,
    seeds: list[int] | None = None,
    num_trials: int | None = None,
    warmup_trials: int | None = None,
    sample_interval: int | None = None,
    validation_samples: int | None = None,
    position_threshold_m: float | None = None,
    rotation_threshold_rad: float | None = None,
    nmg_pos_eps: float | None = None,
    nmg_rot_eps: float | None = None,
    checkpoint_path: str | None = None,
) -> None:
    """Apply explicit CLI/programmatic overrides to a loaded suite."""
    if batch_sizes is not None:
        suite.free_space.batch_sizes = batch_sizes
    if num_waypoints is not None:
        suite.free_space.waypoint_counts = num_waypoints
    if seeds is not None:
        suite.free_space.seeds = seeds
    if num_trials is not None:
        suite.protocol.measured_trials = num_trials
    if warmup_trials is not None:
        suite.protocol.warmup_trials = warmup_trials
    if sample_interval is not None:
        suite.protocol.sample_interval = sample_interval
    if validation_samples is not None:
        suite.protocol.validation_samples = validation_samples
    if position_threshold_m is not None:
        suite.protocol.position_threshold_m = position_threshold_m
    if rotation_threshold_rad is not None:
        suite.protocol.rotation_threshold_rad = rotation_threshold_rad

    nmg = next((spec for spec in suite.planners if spec.id == "nmg"), None)
    if nmg is not None:
        if nmg_pos_eps is not None:
            nmg.config["pos_eps"] = nmg_pos_eps
        if nmg_rot_eps is not None:
            nmg.config["rot_eps"] = nmg_rot_eps
        if checkpoint_path is not None:
            nmg.config["checkpoint_path"] = str(Path(checkpoint_path))
    suite.validate_benchmark()


def run_all_benchmarks(
    num_waypoints_list: list[int] | None = None,
    sim_device: str = "auto",
    headless: bool = True,
    checkpoint_path: str | None = None,
    *,
    suite_name: str = "smoke",
    algorithms: list[str] | None = None,
    extra_baselines: list[str] | None = None,
    batch_sizes: list[int] | None = None,
    seeds: list[int] | None = None,
    num_trials: int | None = None,
    warmup_trials: int | None = None,
    sample_interval: int | None = None,
    validation_samples: int | None = None,
    position_threshold_m: float | None = None,
    rotation_threshold_rad: float | None = None,
    nmg_pos_eps: float | None = None,
    nmg_rot_eps: float | None = None,
    compare_ik: bool = False,
    compare_toppra: bool = False,
    include_trial_details: bool = False,  # noqa: ARG001 - compatibility parameter
    output_root: str | Path = "outputs/benchmarks",
) -> BenchmarkRunResult:
    """Resolve configuration and run all selected free-space benchmarks."""
    from .runner import BenchmarkRunner

    suite = load_suite(suite_name)
    _apply_overrides(
        suite,
        batch_sizes=batch_sizes,
        num_waypoints=num_waypoints_list,
        seeds=seeds,
        num_trials=num_trials,
        warmup_trials=warmup_trials,
        sample_interval=sample_interval,
        validation_samples=validation_samples,
        position_threshold_m=position_threshold_m,
        rotation_threshold_rad=rotation_threshold_rad,
        nmg_pos_eps=nmg_pos_eps,
        nmg_rot_eps=nmg_rot_eps,
        checkpoint_path=checkpoint_path,
    )
    extras = list(extra_baselines or [])
    if compare_ik and "ik_interpolate" not in extras:
        extras.append("ik_interpolate")
    if compare_toppra and "toppra" not in extras:
        extras.append("toppra")
    specs = _resolve_planners(suite, algorithms, extras)
    return BenchmarkRunner(
        suite,
        specs,
        device=sim_device,
        headless=headless,
        output_root=output_root,
    ).run()


def run_from_args(args: argparse.Namespace) -> BenchmarkRunResult:
    """Run the benchmark from parsed unified-CLI arguments."""
    return run_all_benchmarks(
        num_waypoints_list=args.num_waypoints,
        sim_device=args.device,
        headless=args.headless,
        checkpoint_path=args.checkpoint_path,
        suite_name=args.suite,
        algorithms=args.algorithms,
        extra_baselines=args.extra_baselines,
        batch_sizes=args.batch_sizes,
        seeds=args.seeds,
        num_trials=args.num_trials,
        warmup_trials=args.warmup_trials,
        sample_interval=args.sample_interval,
        validation_samples=args.validation_samples,
        position_threshold_m=args.position_threshold_m,
        rotation_threshold_rad=args.rotation_threshold_rad,
        nmg_pos_eps=args.nmg_pos_eps,
        nmg_rot_eps=args.nmg_rot_eps,
        compare_ik=args.compare_ik,
        compare_toppra=args.compare_toppra,
        include_trial_details=args.save_trial_details,
        output_root=args.output_root,
    )


def _parse_args() -> argparse.Namespace:
    """Parse standalone module arguments using the unified option schema."""
    parser = argparse.ArgumentParser(
        description="Benchmark motion generation on fixed free-space cases."
    )
    add_parser_arguments(parser)
    return parser.parse_args()


if __name__ == "__main__":
    run_from_args(_parse_args())
