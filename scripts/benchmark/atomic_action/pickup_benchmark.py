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
    timed_call,
    write_markdown_report,
)

PICK_SAMPLE_INTERVAL = 120
HAND_INTERP_STEPS = 12


def add_benchmark_args(parser: argparse.ArgumentParser) -> None:
    """Add PickUp benchmark CLI arguments."""
    add_pickup_approach_benchmark_args(parser)
    add_object_position_benchmark_args(parser)
    add_grasp_benchmark_args(parser)
    add_common_benchmark_args(parser)


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark PickUp over grasp approach directions."
    )
    add_benchmark_args(parser)
    return parser.parse_args()


def _make_pickup_args(
    args: argparse.Namespace,
    approach: str,
    object_preset: MeshObjectPreset,
    profile: str,
) -> argparse.Namespace:
    """Build a tutorial-compatible argparse namespace."""
    return argparse.Namespace(
        object=object_preset.label,
        n_sample=1000 if profile == "smoke" else args.n_sample,
        force_reannotate=args.force_reannotate,
        approach="custom" if approach == "side" else approach,
        custom_approach_direction=None,
        device=args.device,
        renderer=args.renderer,
        headless=True,
    )


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
    approach: str,
    repeat: int,
    recorded_count: int,
    profile: str,
):
    """Run one PickUp benchmark case."""
    from embodichain.lab.sim.atomic_actions import (
        AtomicActionEngine,
        GraspTarget,
        PickUp,
        PickUpCfg,
    )
    from scripts.tutorials.atomic_action.pickup import (
        build_grasp_generator_cfg,
        build_gripper_collision_cfg,
        get_hand_open_close_qpos,
        initialize_pre_pick_robot_pose,
        compute_pick_close_end_step,
    )

    case_id = (
        f"{object_preset.object_type}:{position_case.name}:{approach}:r{repeat}"
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
        hand_open, hand_close = get_hand_open_close_qpos(robot, sim.device)
        initialize_pre_pick_robot_pose(robot, obj, hand_open)
        case_args = _make_pickup_args(args, approach, object_preset, profile)
        approach_direction = resolve_pickup_approach_direction(
            approach, position_case, sim.device
        )
        approach_direction_text = format_vector3(
            pickup_approach_direction_tuple(approach, position_case)
        )
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
        semantics = create_antipodal_object_semantics(
            obj=obj,
            preset=object_preset,
            args=case_args,
            build_gripper_collision_cfg=build_gripper_collision_cfg,
            build_grasp_generator_cfg=build_grasp_generator_cfg,
        )
        elapsed, mem_delta, peak_gpu, result = timed_call(
            lambda: atomic_engine.run(
                steps=[("pick_up", GraspTarget(semantics=semantics))]
            )
        )
        is_success, traj, final_state = result
        video_path = None
        if should_record_case(args, recorded_count, bool(is_success)):
            reset_robot(robot, initial_qpos)
            reset_rigid_object(obj, initial_obj_pose)
            initialize_pre_pick_robot_pose(robot, obj, hand_open)
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
                traj=traj,
                args=args,
                video_path=build_video_output_path(
                    args,
                    "atomic_action_pick_up",
                    (
                        f"{object_preset.object_type}_{position_case.name}_"
                        f"{approach}_r{repeat}"
                    ),
                ),
                on_step=_on_step,
            )
            reset_robot(robot, initial_qpos)
            reset_rigid_object(obj, initial_obj_pose)

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
            "case_id": case_id,
            "object_type": object_preset.object_type,
            "material": object_preset.material_name,
            "quadrant": position_case.quadrant,
            "position_case": position_case.name,
            "init_xy": position_case.xy,
            "approach": approach,
            "approach_direction": approach_direction_text,
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
            "video_path": str(video_path) if video_path is not None else "",
        }
    except Exception as exc:
        return {
            "case_id": case_id,
            "object_type": object_preset.object_type,
            "material": object_preset.material_name,
            "quadrant": position_case.quadrant,
            "position_case": position_case.name,
            "init_xy": position_case.xy,
            "approach": approach,
            "approach_direction": "N/A",
            "repeat": repeat,
            "planning_success": False,
            "held_created": False,
            "success": False,
            "cost_time_ms": 0.0,
            "cpu_delta_mb": 0.0,
            "gpu_delta_mb": 0.0,
            "peak_gpu_mb": 0.0,
            "lift_height_m": None,
            "trajectory_waypoints": 0,
            "failure_reason": f"exception:{type(exc).__name__}:{exc}",
            "video_path": "",
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
                "object_type": result["object_type"],
                "material": result["material"],
                "quadrant": result["quadrant"],
                "position_case": result["position_case"],
                "init_xy": f"({result['init_xy'][0]:.3f},{result['init_xy'][1]:.3f})",
                "approach": result["approach"],
                "approach_direction": result["approach_direction"],
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
                "object_type": result["object_type"],
                "material": result["material"],
                "quadrant": result["quadrant"],
                "position_case": result["position_case"],
                "approach": result["approach"],
                "approach_direction": result["approach_direction"],
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
    profile = resolve_profile(args)

    ensure_repo_root()
    ensure_torch()
    from embodichain.lab.sim.planners import MotionGenerator, MotionGenCfg
    from embodichain.lab.sim.planners import ToppraPlannerCfg
    from scripts.tutorials.atomic_action.pickup import (
        create_robot,
        initialize_simulation,
    )

    object_presets = select_mesh_object_presets(args.object_types, profile)
    position_cases = select_position_cases(args.position_cases, profile)
    approaches = select_pickup_approaches(args.approach_cases, profile)
    repeat = 1 if profile == "smoke" else args.repeat

    print("=" * 60)
    print("PickUp Atomic Action Benchmark")
    print("=" * 60)
    print(
        "Coverage: "
        f"profile={profile}, {len(object_presets)} object(s) x "
        f"{len(position_cases)} position(s) x {len(approaches)} approach(es) "
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
    print("\n=== PickUp Object/Position/Approach Sweep ===")
    for object_preset in object_presets:
        obj, base_pose = object_pool[object_preset.object_type]
        for parked_index, parked_preset in enumerate(object_presets):
            if parked_preset.object_type == object_preset.object_type:
                continue
            parked_obj, parked_base_pose = object_pool[parked_preset.object_type]
            park_rigid_object(parked_obj, parked_base_pose, index=parked_index, sim=sim)
        for position_case in position_cases:
            for approach in approaches:
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
                        approach,
                        repeat_index,
                        len(video_paths),
                        profile,
                    )
                    results.append(result)
                    if result["video_path"]:
                        video_paths.append(str(result["video_path"]))
                    print(
                        f"  {result['case_id']:<42} "
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
            f"Profile: {profile}",
            "Object presets: "
            + ", ".join(describe_object_preset(preset) for preset in object_presets),
            "Position cases: "
            + ", ".join(
                f"{case.name}/{case.quadrant}/xy={case.xy}"
                for case in position_cases
            ),
            "PickUp approach cases: " + ", ".join(approaches),
            f"CPU memory backend: {CPU_MEMORY_BACKEND}",
            f"n_sample: {1000 if profile == 'smoke' else args.n_sample}",
            "Replay videos: " + (", ".join(video_paths) if video_paths else "disabled"),
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
