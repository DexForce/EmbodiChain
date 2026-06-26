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

"""Benchmark PickUp atomic action across grasp approach directions.

Measures planning latency, memory usage, grasp planning success, held-object
state creation, and trajectory length for the PickUp action.
Run: python -m scripts.benchmark.atomic_action.pickup_benchmark
"""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.benchmark.atomic_action.common import (
    CPU_MEMORY_BACKEND,
    add_common_benchmark_args,
    add_grasp_benchmark_args,
    build_single_action_leaderboard,
    ensure_repo_root,
    ensure_torch,
    format_float,
    reset_robot,
    timed_call,
    write_markdown_report,
)

APPROACH_CASES = ("top", "side", "side_y")
PICK_SAMPLE_INTERVAL = 120
HAND_INTERP_STEPS = 12


def add_benchmark_args(parser: argparse.ArgumentParser) -> None:
    """Add PickUp benchmark CLI arguments."""
    parser.add_argument(
        "--approach_cases",
        nargs="+",
        choices=(*APPROACH_CASES, "all"),
        default=["top"],
        help="Approach directions to benchmark. Use 'all' for every case.",
    )
    add_grasp_benchmark_args(parser)
    add_common_benchmark_args(parser)


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark PickUp over grasp approach directions."
    )
    add_benchmark_args(parser)
    return parser.parse_args()


def _selected_approaches(args: argparse.Namespace) -> list[str]:
    """Resolve selected approach cases."""
    if "all" in args.approach_cases:
        return list(APPROACH_CASES)
    return list(args.approach_cases)


def _make_pickup_args(args: argparse.Namespace, approach: str) -> argparse.Namespace:
    """Build a tutorial-compatible argparse namespace."""
    return argparse.Namespace(
        object="sugar_box",
        n_sample=1000 if args.smoke else args.n_sample,
        force_reannotate=args.force_reannotate,
        approach=approach,
        custom_approach_direction=None,
        device=args.device,
        renderer=args.renderer,
        headless=True,
    )


def _run_case(
    sim,
    robot,
    obj,
    motion_gen,
    initial_qpos,
    args,
    approach: str,
    repeat: int,
):
    """Run one PickUp benchmark case."""
    torch = ensure_torch()
    from embodichain.lab.sim.atomic_actions import (
        AtomicActionEngine,
        GraspTarget,
        PickUp,
        PickUpCfg,
    )
    from scripts.tutorials.atomic_action.pickup import (
        create_object_semantics,
        get_hand_open_close_qpos,
        initialize_pre_pick_robot_pose,
        resolve_approach_direction,
    )

    reset_robot(robot, initial_qpos)
    hand_open, hand_close = get_hand_open_close_qpos(robot, sim.device)
    initialize_pre_pick_robot_pose(robot, obj, hand_open)
    case_args = _make_pickup_args(args, approach)
    approach_direction = resolve_approach_direction(case_args, sim.device)
    atomic_engine = AtomicActionEngine(motion_generator=motion_gen)
    atomic_engine.register(
        PickUp(
            motion_gen,
            cfg=PickUpCfg(
                control_part="arm",
                hand_control_part="hand",
                hand_open_qpos=hand_open,
                hand_close_qpos=hand_close,
                approach_direction=approach_direction,
                pre_grasp_distance=0.15,
                lift_height=0.16,
                sample_interval=PICK_SAMPLE_INTERVAL,
                hand_interp_steps=HAND_INTERP_STEPS,
            ),
        )
    )
    semantics = create_object_semantics(obj, case_args)
    elapsed, mem_delta, peak_gpu, result = timed_call(
        lambda: atomic_engine.run(steps=[("pick_up", GraspTarget(semantics=semantics))])
    )
    is_success, traj, final_state = result
    held_created = bool(is_success and final_state.held_object is not None)
    lift_height_m = None
    if is_success and traj.shape[1] > 0:
        arm_joint_ids = robot.get_joint_ids(name="arm")
        start_pose = robot.compute_fk(
            qpos=traj[:, 0, arm_joint_ids], name="arm", to_matrix=True
        )[0]
        final_pose = robot.compute_fk(
            qpos=traj[:, -1, arm_joint_ids], name="arm", to_matrix=True
        )[0]
        lift_height_m = float(final_pose[2, 3] - start_pose[2, 3])
    success = bool(held_created)
    return {
        "case_id": f"{approach}:r{repeat}",
        "approach": approach,
        "repeat": repeat,
        "planning_success": bool(is_success),
        "held_created": held_created,
        "success": success,
        "cost_time_ms": elapsed * 1000.0,
        "cpu_delta_mb": mem_delta["cpu_mb"],
        "gpu_delta_mb": mem_delta["gpu_mb"],
        "peak_gpu_mb": peak_gpu,
        "lift_height_m": lift_height_m,
        "trajectory_waypoints": int(traj.shape[1]) if traj.ndim >= 2 else 0,
        "failure_reason": "" if success else "held_object_missing",
    }


def _build_rows(results: list[dict[str, object]]):
    """Build report rows for PickUp."""
    perf_rows = []
    metric_rows = []
    for result in results:
        perf_rows.append(
            {
                "sample_size": 1,
                "impl": "pick_up",
                "case_id": result["case_id"],
                "approach": result["approach"],
                "repeat": result["repeat"],
                "cost_time_ms": format_float(result["cost_time_ms"]),
                "cpu_delta_mb": format_float(result["cpu_delta_mb"]),
                "gpu_delta_mb": format_float(result["gpu_delta_mb"]),
                "peak_gpu_mb": format_float(result["peak_gpu_mb"]),
            }
        )
        metric_rows.append(
            {
                "sample_size": 1,
                "impl": "pick_up",
                "case_id": result["case_id"],
                "approach": result["approach"],
                "success_rate": f"{float(result['success']):.6f}",
                "planning_success_rate": f"{float(result['planning_success']):.6f}",
                "held_object_rate": f"{float(result['held_created']):.6f}",
                "lift_height_m": format_float(result["lift_height_m"]),
                "trajectory_waypoints": result["trajectory_waypoints"],
                "failure_reason": result["failure_reason"] or "N/A",
            }
        )
    return perf_rows, metric_rows


def run_all_benchmarks(args: argparse.Namespace | None = None) -> Path:
    """Run PickUp benchmark and write a markdown report."""
    args = _parse_args() if args is None else args
    if args.repeat < 1:
        raise ValueError("--repeat must be at least 1.")

    ensure_repo_root()
    ensure_torch()
    from embodichain.lab.sim.planners import MotionGenerator, MotionGenCfg
    from embodichain.lab.sim.planners import ToppraPlannerCfg
    from scripts.tutorials.atomic_action.pickup import (
        create_pick_object,
        create_robot,
        initialize_simulation,
    )

    approaches = _selected_approaches(args)
    repeat = args.repeat
    if args.smoke:
        approaches = ["top"]
        repeat = 1

    print("=" * 60)
    print("PickUp Atomic Action Benchmark")
    print("=" * 60)

    sim = initialize_simulation(args)
    robot = create_robot(sim)
    obj = create_pick_object(sim, "sugar_box")
    initial_qpos = robot.get_qpos().clone()
    motion_gen = MotionGenerator(
        cfg=MotionGenCfg(planner_cfg=ToppraPlannerCfg(robot_uid=robot.uid))
    )

    results: list[dict[str, object]] = []
    print("\n=== PickUp Approach Sweep ===")
    for approach in approaches:
        for repeat_index in range(repeat):
            result = _run_case(
                sim, robot, obj, motion_gen, initial_qpos, args, approach, repeat_index
            )
            results.append(result)
            print(
                f"  {result['case_id']:<16} "
                f"time={result['cost_time_ms']:>10.2f} ms | "
                f"success={result['success']} "
                f"lift={format_float(result['lift_height_m'], precision=4)}"
            )

    perf_rows, metric_rows = _build_rows(results)
    leaderboard_rows = build_single_action_leaderboard("pick_up", metric_rows)
    report_path = write_markdown_report(
        benchmark_name="atomic_action_pick_up",
        perf_rows=perf_rows,
        metric_rows=metric_rows,
        leaderboard_rows=leaderboard_rows,
        notes=[
            "Object preset: sugar_box from the PickUp tutorial.",
            f"CPU memory backend: {CPU_MEMORY_BACKEND}",
            f"n_sample: {1000 if args.smoke else args.n_sample}",
        ],
    )
    print(f"Markdown report saved: {report_path}")
    return report_path


def main() -> None:
    """Run the CLI entry point."""
    try:
        run_all_benchmarks()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()


__all__ = ["add_benchmark_args", "run_all_benchmarks"]
