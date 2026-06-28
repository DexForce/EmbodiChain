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
    add_object_position_benchmark_args,
    add_pickup_approach_benchmark_args,
    build_single_action_leaderboard,
    build_video_output_path,
    create_antipodal_object_semantics,
    create_benchmark_object,
    describe_object_preset,
    ensure_repo_root,
    ensure_torch,
    format_float,
    format_vector3,
    MeshObjectPreset,
    park_rigid_object,
    pickup_approach_direction_tuple,
    PositionCase,
    record_static_scene_video,
    replay_trajectory_with_recording,
    reset_rigid_object,
    reset_rigid_object_xy,
    reset_robot,
    resolve_pickup_approach_direction,
    resolve_profile,
    select_mesh_object_presets,
    select_pickup_approaches,
    select_position_cases,
    should_record_case,
    SIDE_GRASP_MAX_OPEN_AXIS_ABS_Z,
    SIDE_GRASP_OPEN_AXIS_Z_COST_WEIGHT,
    summarize_video_recording,
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


def add_benchmark_args(parser: argparse.ArgumentParser) -> None:
    """Add MoveHeldObject benchmark CLI arguments."""
    add_pickup_approach_benchmark_args(parser)
    parser.add_argument(
        "--held_object_cases",
        nargs="+",
        choices=(*HELD_OBJECT_CASES.keys(), "all"),
        default=None,
        help=(
            "Held-object target cases to benchmark. Defaults are selected by "
            "--profile; use 'all' for every case."
        ),
    )
    add_object_position_benchmark_args(parser)
    add_grasp_benchmark_args(parser)
    add_common_benchmark_args(parser)


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark MoveHeldObject after a PickUp precondition."
    )
    add_benchmark_args(parser)
    return parser.parse_args()


def _selected_cases(
    case_names: list[str] | None,
    profile: str,
) -> list[HeldObjectCase]:
    """Resolve selected held-object case names."""
    if not case_names:
        return [HELD_OBJECT_CASES["front_left"]]
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


def _make_pickup_args(
    args: argparse.Namespace,
    object_preset: MeshObjectPreset,
    profile: str,
) -> argparse.Namespace:
    """Build a tutorial-compatible argparse namespace."""
    return argparse.Namespace(
        object=object_preset.label,
        n_sample=1000 if profile == "smoke" else args.n_sample,
        force_reannotate=args.force_reannotate,
        device=args.device,
        renderer=args.renderer,
        headless=True,
    )


def _prepare_held_state(
    sim,
    robot,
    obj,
    motion_gen,
    args,
    object_preset: MeshObjectPreset,
    position_case: PositionCase,
    pickup_approach: str,
    profile: str,
):
    """Run PickUp precondition outside the timed MoveHeldObject block."""
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
        build_grasp_generator_cfg,
        build_gripper_collision_cfg,
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
                approach_direction=resolve_pickup_approach_direction(
                    pickup_approach, position_case, sim.device
                ),
                pre_grasp_distance=0.15,
                lift_height=0.16,
                sample_interval=PICK_SAMPLE_INTERVAL,
                hand_interp_steps=HAND_INTERP_STEPS,
            ),
        )
    )
    semantics = create_antipodal_object_semantics(
        obj=obj,
        preset=object_preset,
        args=_make_pickup_args(args, object_preset, profile),
        build_gripper_collision_cfg=build_gripper_collision_cfg,
        build_grasp_generator_cfg=build_grasp_generator_cfg,
    )
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
    return state, hand_close, traj


def _run_case(
    sim,
    robot,
    motion_gen,
    initial_qpos,
    args,
    obj,
    base_obj_pose,
    object_preset: MeshObjectPreset,
    position_case: PositionCase,
    pickup_approach: str,
    case: HeldObjectCase,
    repeat: int,
    recorded_count: int,
    profile: str,
):
    """Run one MoveHeldObject benchmark case."""
    from embodichain.lab.sim.atomic_actions import (
        AtomicActionEngine,
        HeldObjectPoseTarget,
        MoveHeldObject,
        MoveHeldObjectCfg,
    )
    from scripts.tutorials.atomic_action.move_held_object import (
        compute_pick_close_end_step,
    )

    case_id = (
        f"{object_preset.object_type}:{position_case.name}:"
        f"{pickup_approach}:{case.name}:r{repeat}"
    )
    try:
        reset_robot(robot, initial_qpos)
        initial_obj_pose = reset_rigid_object_xy(
            obj=obj,
            base_pose=base_obj_pose,
            xy=position_case.xy,
            sim=sim,
            settle_steps=2,
        )
        reset_rigid_object(obj, initial_obj_pose)
        state, hand_close, precondition_traj = _prepare_held_state(
            sim,
            robot,
            obj,
            motion_gen,
            args,
            object_preset,
            position_case,
            pickup_approach,
            profile,
        )
        approach_direction_text = format_vector3(
            pickup_approach_direction_tuple(pickup_approach, position_case)
        )
        precondition_waypoints = int(precondition_traj.shape[1])
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
        video_path = None
        if should_record_case(args, recorded_count, bool(is_success)):
            torch = ensure_torch()
            reset_robot(robot, initial_qpos)
            reset_rigid_object(obj, initial_obj_pose)
            video_case_suffix = (
                f"{object_preset.object_type}_{position_case.name}_"
                f"{pickup_approach}_{case.name}_r{repeat}"
            )
            if getattr(traj, "ndim", 0) >= 3 and traj.shape[1] > 0:
                replay_traj = torch.cat((precondition_traj, traj), dim=1)
            else:
                replay_traj = precondition_traj

            if getattr(replay_traj, "ndim", 0) >= 3 and replay_traj.shape[1] > 0:
                post_grasp_clear_step = compute_pick_close_end_step()
                should_clear_object_dynamics = True

                def _on_step(waypoint_index: int) -> None:
                    nonlocal should_clear_object_dynamics
                    if (
                        should_clear_object_dynamics
                        and waypoint_index + 1 >= post_grasp_clear_step
                    ):
                        obj.clear_dynamics()
                        should_clear_object_dynamics = False

                video_path = replay_trajectory_with_recording(
                    sim=sim,
                    robot=robot,
                    traj=replay_traj,
                    args=args,
                    video_path=build_video_output_path(
                        args,
                        "atomic_action_move_held_object",
                        video_case_suffix,
                    ),
                    on_step=_on_step,
                )
            else:
                video_path = record_static_scene_video(
                    sim=sim,
                    args=args,
                    video_path=build_video_output_path(
                        args,
                        "atomic_action_move_held_object",
                        f"{video_case_suffix}_failed_static",
                    ),
                )
            reset_robot(robot, initial_qpos)
            reset_rigid_object(obj, initial_obj_pose)

        still_held = bool(is_success and final_state.held_object is not None)
        return {
            "case_id": case_id,
            "object_type": object_preset.object_type,
            "material": object_preset.material_name,
            "quadrant": position_case.quadrant,
            "position_case": position_case.name,
            "init_xy": position_case.xy,
            "pickup_approach": pickup_approach,
            "approach_direction": approach_direction_text,
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
            "video_path": str(video_path) if video_path is not None else "",
        }
    except Exception as exc:
        video_path = None
        if should_record_case(args, recorded_count, False):
            try:
                reset_robot(robot, initial_qpos)
                initial_obj_pose = reset_rigid_object_xy(
                    obj=obj,
                    base_pose=base_obj_pose,
                    xy=position_case.xy,
                    sim=sim,
                    settle_steps=2,
                )
                video_path = record_static_scene_video(
                    sim=sim,
                    args=args,
                    video_path=build_video_output_path(
                        args,
                        "atomic_action_move_held_object",
                        (
                            f"{object_preset.object_type}_{position_case.name}_"
                            f"{pickup_approach}_{case.name}_r{repeat}"
                            "_exception_static"
                        ),
                    ),
                )
                reset_rigid_object(obj, initial_obj_pose)
            except Exception:
                video_path = None
        return {
            "case_id": case_id,
            "object_type": object_preset.object_type,
            "material": object_preset.material_name,
            "quadrant": position_case.quadrant,
            "position_case": position_case.name,
            "init_xy": position_case.xy,
            "pickup_approach": pickup_approach,
            "approach_direction": "N/A",
            "held_object_case": case.name,
            "repeat": repeat,
            "planning_success": False,
            "still_held": False,
            "success": False,
            "cost_time_ms": 0.0,
            "cpu_delta_mb": 0.0,
            "gpu_delta_mb": 0.0,
            "peak_gpu_mb": 0.0,
            "precondition_waypoints": 0,
            "trajectory_waypoints": 0,
            "failure_reason": f"exception:{type(exc).__name__}:{exc}",
            "video_path": str(video_path) if video_path is not None else "",
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
                "object_type": result["object_type"],
                "material": result["material"],
                "quadrant": result["quadrant"],
                "position_case": result["position_case"],
                "init_xy": f"({result['init_xy'][0]:.3f},{result['init_xy'][1]:.3f})",
                "pickup_approach": result["pickup_approach"],
                "approach_direction": result["approach_direction"],
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
                "object_type": result["object_type"],
                "material": result["material"],
                "quadrant": result["quadrant"],
                "position_case": result["position_case"],
                "pickup_approach": result["pickup_approach"],
                "approach_direction": result["approach_direction"],
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
    profile = resolve_profile(args)

    ensure_repo_root()
    ensure_torch()
    from embodichain.lab.sim.planners import MotionGenerator, MotionGenCfg
    from embodichain.lab.sim.planners import ToppraPlannerCfg
    from scripts.tutorials.atomic_action.move_held_object import (
        create_robot,
        initialize_simulation,
    )

    object_presets = select_mesh_object_presets(args.object_types, profile)
    position_cases = select_position_cases(args.position_cases, profile)
    approaches = select_pickup_approaches(args.approach_cases, profile)
    cases = _selected_cases(args.held_object_cases, profile)
    repeat = 1 if profile == "smoke" else args.repeat

    print("=" * 60)
    print("MoveHeldObject Atomic Action Benchmark")
    print("=" * 60)
    print(
        "Coverage: "
        f"profile={profile}, {len(object_presets)} object(s) x "
        f"{len(position_cases)} position(s) x {len(approaches)} pickup approach(es) "
        f"x {len(cases)} held target(s) "
        f"x {repeat} repeat(s)"
    )

    sim = initialize_simulation(args)
    robot = create_robot(sim)
    initial_qpos = robot.get_qpos().clone()
    motion_gen = MotionGenerator(
        cfg=MotionGenCfg(planner_cfg=ToppraPlannerCfg(robot_uid=robot.uid))
    )
    object_pool = {}
    for object_index, object_preset in enumerate(object_presets):
        obj = create_benchmark_object(
            sim=sim,
            preset=object_preset,
            position_case=position_cases[0],
            uid_suffix="pool",
        )
        base_pose = obj.get_local_pose(to_matrix=True).clone()
        park_rigid_object(obj, base_pose, index=object_index, sim=sim)
        object_pool[object_preset.object_type] = (obj, base_pose)

    results: list[dict[str, object]] = []
    video_paths: list[str] = []
    print("\n=== MoveHeldObject Object/Position/Target Sweep ===")
    for object_preset in object_presets:
        obj, base_pose = object_pool[object_preset.object_type]
        for parked_index, parked_preset in enumerate(object_presets):
            if parked_preset.object_type == object_preset.object_type:
                continue
            parked_obj, parked_base_pose = object_pool[parked_preset.object_type]
            park_rigid_object(parked_obj, parked_base_pose, index=parked_index, sim=sim)
        for position_case in position_cases:
            for pickup_approach in approaches:
                for case in cases:
                    for repeat_index in range(repeat):
                        result = _run_case(
                            sim,
                            robot,
                            motion_gen,
                            initial_qpos,
                            args,
                            obj,
                            base_pose,
                            object_preset,
                            position_case,
                            pickup_approach,
                            case,
                            repeat_index,
                            len(video_paths),
                            profile,
                        )
                        results.append(result)
                        if result["video_path"]:
                            video_paths.append(str(result["video_path"]))
                        print(
                            f"  {result['case_id']:<54} "
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
            f"Profile: {profile}",
            "Object presets: "
            + ", ".join(describe_object_preset(preset) for preset in object_presets),
            "Position cases: "
            + ", ".join(
                f"{case.name}/{case.quadrant}/xy={case.xy}"
                for case in position_cases
            ),
            "PickUp approach cases: " + ", ".join(approaches),
            "Held-object target cases: " + ", ".join(case.name for case in cases),
            f"CPU memory backend: {CPU_MEMORY_BACKEND}",
            f"n_sample: {1000 if profile == 'smoke' else args.n_sample}",
            (
                "Side grasp opening-axis rule: prefer candidates with "
                f"abs(opening_axis_z) <= {SIDE_GRASP_MAX_OPEN_AXIS_ABS_Z:.2f}; "
                "fallback cost += "
                f"abs(opening_axis_z) * {SIDE_GRASP_OPEN_AXIS_Z_COST_WEIGHT:.2f}."
            ),
            *summarize_video_recording(args, results, video_paths),
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
