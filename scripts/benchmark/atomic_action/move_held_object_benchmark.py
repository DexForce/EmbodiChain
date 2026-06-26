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

"""Benchmark MoveHeldObject atomic action after a PickUp precondition.

Measures MoveHeldObject-only planning latency and memory usage once a held
object state has been produced by PickUp.
Run: python -m scripts.benchmark.atomic_action.move_held_object_benchmark
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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


@dataclass(frozen=True)
class HeldObjectCase:
    """Held-object target pose benchmark case."""

    name: str
    xyz: tuple[float, float, float]


HELD_OBJECT_CASES = {
    "front_left": HeldObjectCase("front_left", (-0.30, -0.30, 0.50)),
    "front_right": HeldObjectCase("front_right", (-0.30, 0.30, 0.50)),
    "center_high": HeldObjectCase("center_high", (-0.42, 0.00, 0.58)),
}
PICK_SAMPLE_INTERVAL = 120
MOVE_SAMPLE_INTERVAL = 60
MOVE_HELD_OBJECT_SAMPLE_INTERVAL = 120
HAND_INTERP_STEPS = 12
OBJECT_APPROACH_DIRECTION = (0.0, 0.0, -1.0)


def add_benchmark_args(parser: argparse.ArgumentParser) -> None:
    """Add MoveHeldObject benchmark CLI arguments."""
    parser.add_argument(
        "--held_object_cases",
        nargs="+",
        choices=(*HELD_OBJECT_CASES.keys(), "all"),
        default=list(HELD_OBJECT_CASES.keys()),
        help="Held-object target cases to benchmark. Use 'all' for every case.",
    )
    add_grasp_benchmark_args(parser)
    add_common_benchmark_args(parser)


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark MoveHeldObject after a PickUp precondition."
    )
    add_benchmark_args(parser)
    return parser.parse_args()


def _selected_cases(case_names: list[str]) -> list[HeldObjectCase]:
    """Resolve selected held-object case names."""
    if "all" in case_names:
        return list(HELD_OBJECT_CASES.values())
    return [HELD_OBJECT_CASES[name] for name in case_names]


def _make_object_target_pose(device, xyz: tuple[float, float, float]):
    """Create a held-object target pose."""
    torch = ensure_torch()
    pose = torch.eye(4, dtype=torch.float32, device=device)
    pose[:3, :3] = torch.tensor(
        [
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
        device=device,
    )
    pose[:3, 3] = torch.tensor(xyz, dtype=torch.float32, device=device)
    return pose


def _make_pickup_args(args: argparse.Namespace) -> argparse.Namespace:
    """Build a tutorial-compatible argparse namespace."""
    return argparse.Namespace(
        n_sample=1000 if args.smoke else args.n_sample,
        force_reannotate=args.force_reannotate,
        device=args.device,
        renderer=args.renderer,
        headless=True,
    )


def _prepare_held_state(sim, robot, obj, motion_gen, args):
    """Run PickUp precondition outside the timed MoveHeldObject block."""
    torch = ensure_torch()
    from embodichain.lab.sim.atomic_actions import (
        AtomicActionEngine,
        EndEffectorPoseTarget,
        GraspTarget,
        MoveEndEffector,
        MoveEndEffectorCfg,
        PickUp,
        PickUpCfg,
    )
    from scripts.tutorials.atomic_action.move_held_object import (
        create_object_semantics,
        get_hand_open_close_qpos,
        make_pre_pick_eef_pose,
    )

    hand_open, hand_close = get_hand_open_close_qpos(robot, sim.device)
    atomic_engine = AtomicActionEngine(motion_generator=motion_gen)
    atomic_engine.register(
        MoveEndEffector(
            motion_gen,
            cfg=MoveEndEffectorCfg(
                control_part="arm",
                sample_interval=MOVE_SAMPLE_INTERVAL,
            ),
        )
    )
    atomic_engine.register(
        PickUp(
            motion_gen,
            cfg=PickUpCfg(
                control_part="arm",
                hand_control_part="hand",
                hand_open_qpos=hand_open,
                hand_close_qpos=hand_close,
                approach_direction=torch.tensor(
                    OBJECT_APPROACH_DIRECTION, dtype=torch.float32, device=sim.device
                ),
                pre_grasp_distance=0.15,
                lift_height=0.16,
                sample_interval=PICK_SAMPLE_INTERVAL,
                hand_interp_steps=HAND_INTERP_STEPS,
            ),
        )
    )
    semantics = create_object_semantics(obj, _make_pickup_args(args))
    obj_pose = obj.get_local_pose(to_matrix=True)
    move_position = obj_pose[0, :3, 3].clone()
    move_position[2] = 0.36
    move_target = make_pre_pick_eef_pose(robot, move_position)
    is_success, traj, state = atomic_engine.run(
        steps=[
            ("move_end_effector", EndEffectorPoseTarget(xpos=move_target)),
            ("pick_up", GraspTarget(semantics=semantics)),
        ]
    )
    if not is_success or state.held_object is None:
        raise RuntimeError(
            "Failed to prepare held-object state for MoveHeldObject benchmark."
        )
    robot.set_qpos(state.last_qpos)
    return state, hand_close, int(traj.shape[1])


def _run_case(
    sim,
    robot,
    obj,
    motion_gen,
    initial_qpos,
    args,
    case: HeldObjectCase,
    repeat: int,
):
    """Run one MoveHeldObject benchmark case."""
    from embodichain.lab.sim.atomic_actions import (
        AtomicActionEngine,
        HeldObjectPoseTarget,
        MoveHeldObject,
        MoveHeldObjectCfg,
    )

    reset_robot(robot, initial_qpos)
    state, hand_close, precondition_waypoints = _prepare_held_state(
        sim, robot, obj, motion_gen, args
    )
    atomic_engine = AtomicActionEngine(motion_generator=motion_gen)
    atomic_engine.register(
        MoveHeldObject(
            motion_gen,
            cfg=MoveHeldObjectCfg(
                control_part="arm",
                hand_control_part="hand",
                hand_close_qpos=hand_close,
                sample_interval=MOVE_HELD_OBJECT_SAMPLE_INTERVAL,
            ),
        )
    )
    target_pose = _make_object_target_pose(sim.device, case.xyz)
    elapsed, mem_delta, peak_gpu, result = timed_call(
        lambda: atomic_engine.run(
            steps=[
                (
                    "move_held_object",
                    HeldObjectPoseTarget(object_target_pose=target_pose),
                )
            ],
            state=state,
        )
    )
    is_success, traj, final_state = result
    still_held = bool(is_success and final_state.held_object is not None)
    return {
        "case_id": f"{case.name}:r{repeat}",
        "held_object_case": case.name,
        "repeat": repeat,
        "planning_success": bool(is_success),
        "still_held": still_held,
        "success": still_held,
        "cost_time_ms": elapsed * 1000.0,
        "cpu_delta_mb": mem_delta["cpu_mb"],
        "gpu_delta_mb": mem_delta["gpu_mb"],
        "peak_gpu_mb": peak_gpu,
        "precondition_waypoints": precondition_waypoints,
        "trajectory_waypoints": int(traj.shape[1]) if traj.ndim >= 2 else 0,
        "failure_reason": "" if still_held else "held_object_lost",
    }


def _build_rows(results: list[dict[str, object]]):
    """Build report rows for MoveHeldObject."""
    perf_rows = []
    metric_rows = []
    for result in results:
        perf_rows.append(
            {
                "sample_size": 1,
                "impl": "move_held_object",
                "case_id": result["case_id"],
                "held_object_case": result["held_object_case"],
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
                "impl": "move_held_object",
                "case_id": result["case_id"],
                "held_object_case": result["held_object_case"],
                "success_rate": f"{float(result['success']):.6f}",
                "planning_success_rate": f"{float(result['planning_success']):.6f}",
                "held_object_rate": f"{float(result['still_held']):.6f}",
                "precondition_waypoints": result["precondition_waypoints"],
                "trajectory_waypoints": result["trajectory_waypoints"],
                "failure_reason": result["failure_reason"] or "N/A",
            }
        )
    return perf_rows, metric_rows


def run_all_benchmarks(args: argparse.Namespace | None = None) -> Path:
    """Run MoveHeldObject benchmark and write a markdown report."""
    args = _parse_args() if args is None else args
    if args.repeat < 1:
        raise ValueError("--repeat must be at least 1.")

    ensure_repo_root()
    ensure_torch()
    from embodichain.lab.sim.planners import MotionGenerator, MotionGenCfg
    from embodichain.lab.sim.planners import ToppraPlannerCfg
    from scripts.tutorials.atomic_action.move_held_object import (
        create_pick_object,
        create_robot,
        initialize_simulation,
    )

    cases = _selected_cases(args.held_object_cases)
    repeat = args.repeat
    if args.smoke:
        cases = [HELD_OBJECT_CASES["front_left"]]
        repeat = 1

    print("=" * 60)
    print("MoveHeldObject Atomic Action Benchmark")
    print("=" * 60)

    sim = initialize_simulation(args)
    robot = create_robot(sim)
    obj = create_pick_object(sim)
    initial_qpos = robot.get_qpos().clone()
    motion_gen = MotionGenerator(
        cfg=MotionGenCfg(planner_cfg=ToppraPlannerCfg(robot_uid=robot.uid))
    )

    results: list[dict[str, object]] = []
    print("\n=== MoveHeldObject Target Sweep ===")
    for case in cases:
        for repeat_index in range(repeat):
            result = _run_case(
                sim, robot, obj, motion_gen, initial_qpos, args, case, repeat_index
            )
            results.append(result)
            print(
                f"  {result['case_id']:<20} "
                f"time={result['cost_time_ms']:>10.2f} ms | "
                f"success={result['success']}"
            )

    perf_rows, metric_rows = _build_rows(results)
    leaderboard_rows = build_single_action_leaderboard("move_held_object", metric_rows)
    report_path = write_markdown_report(
        benchmark_name="atomic_action_move_held_object",
        perf_rows=perf_rows,
        metric_rows=metric_rows,
        leaderboard_rows=leaderboard_rows,
        notes=[
            "Timed block includes MoveHeldObject only; PickUp precondition is "
            "prepared outside timing.",
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
