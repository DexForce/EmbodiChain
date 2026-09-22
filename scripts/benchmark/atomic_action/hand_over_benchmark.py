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

"""Benchmark HandOver atomic action on the dual-arm tutorial scene.

Reports the four-level success ladder (planning_success, motion_valid,
execution_success, task_success). Task success requires the object to reach the
commanded delivery pose without ever being dropped during the transfer.
Run: embodichain benchmark atomic-action --action hand_over
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from scripts.benchmark.atomic_action.common import (
    CPU_MEMORY_BACKEND,
    DEFAULT_REPLAY_STEPS_PER_WAYPOINT,
    REPLAY_TRACKING_TOLERANCE_RAD,
    SuccessLadder,
    add_common_benchmark_args,
    add_grasp_benchmark_args,
    build_ladder_leaderboard,
    build_video_output_path,
    check_motion_valid,
    ensure_repo_root,
    ensure_torch,
    format_float,
    object_position_tuple,
    replay_and_track_scalar,
    replay_trajectory_with_recording,
    reset_rigid_object,
    reset_robot,
    resolve_profile,
    should_record_case,
    summarize_video_recording,
    timed_call,
    warmup_planning,
    write_markdown_report,
    xyz_distance_m,
)

DUAL_ROBOT_TYPE = "ur5"
GRIPPER_TCP_Z = 0.15
HAND_CLOSE_QPOS = 0.04
HANDOVER_SAMPLE_INTERVAL = 220
HANDOVER_HAND_INTERP_STEPS = 10
HANDOVER_PRE_GRASP_DISTANCE = 0.08
HANDOVER_LIFT_HEIGHT = 0.15
SUPPORT_SURFACE_Z = 0.50
FINAL_OBJECT_XYZ = (0.0, -0.2, 0.6)
SETTLE_ITERATIONS = 50
REPLAY_HOLD_STEPS = 60
# Delivery tolerance on the object's final position. Matches the magnitude of
# the existing MoveHeldObject placement tolerance (0.12 m) used by this suite.
HANDOVER_DELIVERY_TOLERANCE_M = 0.12
# The object is dropped if it ever falls well below the support surface it was
# lifted from; 0.15 m under the surface clears normal transfer dips.
HANDOVER_DROP_Z_M = SUPPORT_SURFACE_Z - 0.15


@dataclass(frozen=True)
class HandOverCase:
    """HandOver object-orientation benchmark case."""

    name: str
    is_horizontal: bool


HAND_OVER_CASES: dict[str, HandOverCase] = {
    "vertical_can": HandOverCase("vertical_can", False),
    "horizontal_pencil": HandOverCase("horizontal_pencil", True),
}
DEFAULT_HAND_OVER_CASES = tuple(HAND_OVER_CASES.keys())
SMOKE_HAND_OVER_CASE = "vertical_can"


def add_benchmark_args(parser: argparse.ArgumentParser) -> None:
    """Add HandOver benchmark CLI arguments."""
    parser.add_argument(
        "--hand_over_cases",
        nargs="+",
        choices=(*HAND_OVER_CASES.keys(), "all"),
        default=list(DEFAULT_HAND_OVER_CASES),
        help="Object-orientation cases to benchmark. Use 'all' for every case.",
    )
    add_grasp_benchmark_args(parser)
    add_common_benchmark_args(parser)


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark HandOver over dual-arm transfer cases."
    )
    add_benchmark_args(parser)
    return parser.parse_args()


def _select_cases(case_names: list[str]) -> list[HandOverCase]:
    """Resolve selected handover case names."""
    if "all" in case_names:
        return list(HAND_OVER_CASES.values())
    return [HAND_OVER_CASES[name] for name in case_names]


def _build_invocation(atomic_engine, semantics, device, physics_dt):
    """Build the HandOver invocation and planning context."""
    torch = ensure_torch()
    from embodichain.lab.sim.atomic_actions import (
        HandOverGoal,
        HandOverOptions,
        MotionPolicy,
    )

    final_pose = torch.eye(4, dtype=torch.float32, device=device)
    final_pose[:3, 3] = torch.as_tensor(
        FINAL_OBJECT_XYZ, dtype=torch.float32, device=device
    )
    invocation = atomic_engine.make_invocation(
        "hand_over",
        HandOverGoal(semantics, target_pose=final_pose),
        control_parts={
            "source": {"motion": "left_arm", "grasp": "left_hand"},
            "destination": {"motion": "right_arm", "grasp": "right_hand"},
        },
        motion_policy=MotionPolicy(
            strategy="motion_gen", sample_count=HANDOVER_SAMPLE_INTERVAL
        ),
        skill_options=HandOverOptions(
            pre_grasp_distance=HANDOVER_PRE_GRASP_DISTANCE,
            lift_height=HANDOVER_LIFT_HEIGHT,
            hand_interp_steps=HANDOVER_HAND_INTERP_STEPS,
        ),
    )
    return invocation, atomic_engine.initial_context(control_dt=physics_dt)


def _run_case(
    sim,
    robot,
    obj,
    atomic_engine,
    semantics,
    initial_qpos,
    initial_object_pose,
    case: HandOverCase,
    repeat: int,
    args: argparse.Namespace,
    recorded_count: int,
) -> dict[str, object]:
    """Run one HandOver case through the full four-level success ladder."""
    ladder = SuccessLadder()
    reset_robot(robot, initial_qpos)
    reset_rigid_object(obj, initial_object_pose)
    sim.update(step=10)

    invocation, context = _build_invocation(
        atomic_engine, semantics, sim.device, sim.sim_config.physics_dt
    )
    try:
        elapsed, mem_delta, peak_gpu, result = timed_call(
            lambda: atomic_engine.compile((invocation,), context)
        )
    except Exception as exc:
        print(f"    planner exception: {type(exc).__name__}: {exc}")
        ladder.fail("planning_success", "planner_exception")
        return _case_result(case, repeat, ladder, 0.0, None, None, None, None, "")

    ladder.planning_success = bool(result.plan_success.all().item())
    traj = result.trajectory.positions
    if not ladder.planning_success:
        for plan in result.action_plans:
            if not plan.plan_success.all():
                messages = plan.diagnostics.messages or ("planning failed",)
                print(f"    plan failure [{plan.skill_id}]: {'; '.join(messages)}")
        ladder.fail("planning_success", "planner_reported_failure")
        return _case_result(
            case, repeat, ladder, elapsed, mem_delta, None, None, None, "", peak_gpu
        )

    motion_valid, motion_reason = check_motion_valid(traj, robot)
    ladder.motion_valid = motion_valid
    if not motion_valid:
        ladder.fail("motion_valid", motion_reason)
        return _case_result(
            case,
            repeat,
            ladder,
            elapsed,
            mem_delta,
            None,
            None,
            None,
            "",
            peak_gpu,
            traj,
        )

    state: dict[str, float] = {"min_z": float("inf")}

    def read_delivery_distance(waypoint_index: int) -> float:
        del waypoint_index
        position = object_position_tuple(obj)
        state["min_z"] = min(state["min_z"], position[2])
        return xyz_distance_m(position, FINAL_OBJECT_XYZ)

    try:
        trace = replay_and_track_scalar(
            sim=sim,
            robot=robot,
            traj=traj,
            read_value=read_delivery_distance,
            steps_per_waypoint=DEFAULT_REPLAY_STEPS_PER_WAYPOINT,
            hold_steps=REPLAY_HOLD_STEPS,
        )
        ladder.execution_success = trace is not None
    except Exception as exc:
        print(f"    replay exception: {type(exc).__name__}: {exc}")
        trace = None
        ladder.fail("execution_success", "controller_tracking_failure")

    min_z = state["min_z"] if state["min_z"] != float("inf") else None
    dropped = min_z is not None and min_z < HANDOVER_DROP_Z_M
    if trace is not None and (
        trace.max_tracking_error_rad > REPLAY_TRACKING_TOLERANCE_RAD
    ):
        ladder.execution_success = False
        ladder.fail("execution_success", "controller_tracking_failure")
    elif ladder.execution_success and trace is not None:
        delivered = trace.settled_position <= HANDOVER_DELIVERY_TOLERANCE_M
        ladder.task_success = delivered and not dropped
        if not ladder.task_success:
            ladder.fail(
                "task_success", "object_dropped" if dropped else "task_goal_miss"
            )
    elif not ladder.failure_stage:
        ladder.fail("execution_success", "controller_tracking_failure")

    video_path = ""
    if should_record_case(args, recorded_count, ladder.task_success):
        reset_robot(robot, initial_qpos)
        reset_rigid_object(obj, initial_object_pose)
        recorded = replay_trajectory_with_recording(
            sim=sim,
            robot=robot,
            traj=traj,
            args=args,
            video_path=build_video_output_path(
                args, "atomic_action_hand_over", f"{case.name}_r{repeat}"
            ),
        )
        video_path = str(recorded) if recorded is not None else ""

    return _case_result(
        case,
        repeat,
        ladder,
        elapsed,
        mem_delta,
        trace,
        min_z,
        dropped,
        video_path,
        peak_gpu,
        traj,
    )


def _release_simulation(sim) -> None:
    """Tear down one case's simulation so the next case can build its own."""
    from embodichain.lab.sim import SimulationManager

    if not SimulationManager.is_instantiated():
        return
    if not getattr(sim, "_is_constructed", False):
        SimulationManager.reset(getattr(sim, "instance_id", 0))
        return
    if sim.is_window_recording():
        sim.stop_window_record()
    sim.wait_window_record_saves()
    sim.destroy(exit_process=False)
    SimulationManager.flush_cleanup_queue()


def _case_result(
    case: HandOverCase,
    repeat: int,
    ladder: SuccessLadder,
    elapsed: float,
    mem_delta: dict[str, float] | None,
    trace,
    min_object_z: float | None,
    dropped: bool | None,
    video_path: str,
    peak_gpu: float = 0.0,
    traj=None,
) -> dict[str, object]:
    """Assemble one HandOver case result row."""
    mem_delta = mem_delta or {"cpu_mb": 0.0, "gpu_mb": 0.0}
    return {
        "case_id": f"{case.name}:r{repeat}",
        "hand_over_case": case.name,
        "repeat": repeat,
        "ladder": ladder,
        "cost_time_ms": elapsed * 1000.0,
        "cpu_delta_mb": mem_delta["cpu_mb"],
        "gpu_delta_mb": mem_delta["gpu_mb"],
        "peak_gpu_mb": peak_gpu,
        "max_tracking_error_rad": trace.max_tracking_error_rad if trace else None,
        "initial_delivery_distance_m": trace.initial_position if trace else None,
        "final_delivery_distance_m": trace.settled_position if trace else None,
        "min_object_z_m": min_object_z,
        "object_dropped": bool(dropped) if dropped is not None else None,
        "trajectory_waypoints": (
            int(traj.shape[1]) if traj is not None and traj.ndim >= 3 else 0
        ),
        "success": ladder.task_success,
        "video_path": video_path,
    }


def _build_rows(results: list[dict[str, object]]):
    """Convert case results into report row dicts."""
    perf_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    for result in results:
        perf_rows.append(
            {
                "sample_size": 1,
                "impl": "hand_over",
                "case_id": result["case_id"],
                "hand_over_case": result["hand_over_case"],
                "repeat": result["repeat"],
                "cost_time_ms": format_float(result["cost_time_ms"], precision=2),
                "cpu_delta_mb": format_float(result["cpu_delta_mb"], precision=1),
                "gpu_delta_mb": format_float(result["gpu_delta_mb"], precision=1),
                "peak_gpu_mb": format_float(result["peak_gpu_mb"], precision=1),
                "trajectory_waypoints": result["trajectory_waypoints"],
            }
        )
        ladder: SuccessLadder = result["ladder"]  # type: ignore[assignment]
        metric_rows.append(
            {
                "sample_size": 1,
                "impl": "hand_over",
                "case_id": result["case_id"],
                "hand_over_case": result["hand_over_case"],
                "success_rate": f"{float(ladder.task_success):.6f}",
                **ladder.as_row_fields(),
                "max_tracking_error_rad": format_float(
                    result["max_tracking_error_rad"], 4
                ),
                "initial_delivery_distance_m": format_float(
                    result["initial_delivery_distance_m"], 4
                ),
                "final_delivery_distance_m": format_float(
                    result["final_delivery_distance_m"], 4
                ),
                "min_object_z_m": format_float(result["min_object_z_m"], 4),
                "object_dropped": result["object_dropped"],
            }
        )
    return perf_rows, metric_rows


def run_all_benchmarks(args: argparse.Namespace | None = None) -> Path:
    """Run HandOver benchmark and write a markdown report."""
    args = _parse_args() if args is None else args
    if args.repeat < 1:
        raise ValueError("--repeat must be at least 1.")
    profile = resolve_profile(args)
    ensure_repo_root()
    ensure_torch()
    from embodichain.lab.sim.atomic_actions import ControlPartCommandProfile
    from embodichain.lab.sim.atomic_actions.sim_adapter import (
        create_simulation_atomic_action_engine,
    )
    from scripts.tutorials.atomic_action.hand_over import (
        create_dual_robot,
        create_handover_object,
        create_support_surface,
    )
    from scripts.tutorials.atomic_action.scenario_utils import settle_object
    from scripts.tutorials.atomic_action.tutorial_utils import (
        clone_local_pose_from_first_env,
        create_antipodal_semantics,
        create_parallel_jaw_grasp_pose_generator,
        create_toppra_motion_generator,
        create_tutorial_simulation,
        get_hand_open_close_qpos,
    )

    cases = _select_cases(args.hand_over_cases)
    repeat = 1 if profile == "smoke" else args.repeat
    if profile == "smoke":
        cases = [HAND_OVER_CASES[SMOKE_HAND_OVER_CASE]]

    print("=" * 60)
    print("HandOver Atomic Action Benchmark")
    print("=" * 60)
    print(
        f"Coverage: profile={profile}, {len(cases)} handover case(s) x "
        f"{repeat} repeat(s)"
    )
    if len(cases) > 1:
        print(
            "Note: object orientation is fixed at scene construction, so each "
            "case is run in its own simulation."
        )

    report_results: list[dict[str, object]] = []
    video_paths: list[str] = []
    n_sample = 1000 if profile == "smoke" else args.n_sample

    # The handover object's mesh and orientation are chosen when the scene is
    # built, so each case owns a simulation rather than resetting a shared one.
    for case in cases:
        sim = create_tutorial_simulation(
            argparse.Namespace(
                num_envs=getattr(args, "num_envs", 1),
                device=getattr(args, "device", "cpu"),
                renderer=getattr(args, "renderer", "auto"),
                headless=True,
            ),
            arena_space=3.0,
        )
        try:
            robot = create_dual_robot(sim, DUAL_ROBOT_TYPE)
            create_support_surface(sim)
            obj = create_handover_object(
                sim, argparse.Namespace(is_horizontal=case.is_horizontal)
            )
            sim.prepare()
            settle_object(sim, obj, step=0)
            clone_local_pose_from_first_env(obj)
            obj.clear_dynamics()
            # The tutorial lets the object fall onto the support surface before
            # planning; reproduce that settling.
            for _ in range(SETTLE_ITERATIONS):
                sim.update(step=10)

            semantics = create_antipodal_semantics(obj, label="handover")
            left_open, left_close = get_hand_open_close_qpos(
                robot, hand_control_part="left_hand", close_qpos=HAND_CLOSE_QPOS
            )
            right_open, right_close = get_hand_open_close_qpos(
                robot, hand_control_part="right_hand", close_qpos=HAND_CLOSE_QPOS
            )
            grasp_pose_generator = create_parallel_jaw_grasp_pose_generator(
                n_sample=n_sample, force_refresh=args.force_reannotate
            )
            atomic_engine = create_simulation_atomic_action_engine(
                motion_generator=create_toppra_motion_generator(robot),
                scene_entities=(obj,),
                control_profiles={
                    "left_hand": ControlPartCommandProfile.joint_positions(
                        open=left_open, grasp=left_close
                    ),
                    "right_hand": ControlPartCommandProfile.joint_positions(
                        open=right_open, grasp=right_close
                    ),
                },
                grasp_pose_generators={
                    "left_hand": grasp_pose_generator,
                    "right_hand": grasp_pose_generator,
                },
            )
            initial_qpos = robot.get_qpos().clone()
            initial_object_pose = obj.get_local_pose(to_matrix=True).clone()

            print(f"\n=== HandOver case {case.name} ===")
            print("Warm-up: running one discarded compile...")
            warmup_invocation, warmup_context = _build_invocation(
                atomic_engine, semantics, sim.device, sim.sim_config.physics_dt
            )
            warmup_planning(
                lambda: atomic_engine.compile((warmup_invocation,), warmup_context)
            )
            reset_robot(robot, initial_qpos)
            reset_rigid_object(obj, initial_object_pose)

            for repeat_index in range(repeat):
                result = _run_case(
                    sim,
                    robot,
                    obj,
                    atomic_engine,
                    semantics,
                    initial_qpos,
                    initial_object_pose,
                    case,
                    repeat_index,
                    args,
                    len(video_paths),
                )
                report_results.append(result)
                if result["video_path"]:
                    video_paths.append(str(result["video_path"]))
                ladder: SuccessLadder = result["ladder"]  # type: ignore[assignment]
                print(
                    f"  {result['case_id']:<26} "
                    f"time={result['cost_time_ms']:>9.2f} ms | "
                    f"plan={ladder.planning_success} valid={ladder.motion_valid} "
                    f"exec={ladder.execution_success} task={ladder.task_success} "
                    f"dist={format_float(result['final_delivery_distance_m'], 4)} "
                    f"min_z={format_float(result['min_object_z_m'], 3)} "
                    f"[{ladder.failure_reason or 'ok'}]"
                )
        finally:
            # SimulationManager is a singleton, so a multi-case sweep must tear
            # the current scene down before the next case builds its own.
            _release_simulation(sim)

    perf_rows, metric_rows = _build_rows(report_results)
    leaderboard_rows = build_ladder_leaderboard("hand_over", report_results)
    report_path = write_markdown_report(
        benchmark_name="atomic_action_hand_over",
        perf_rows=perf_rows,
        metric_rows=metric_rows,
        leaderboard_rows=leaderboard_rows,
        notes=[
            f"Profile: {profile}",
            f"CPU memory backend: {CPU_MEMORY_BACKEND}",
            f"Grasp samples: {n_sample}",
            f"Dual-arm robot: {DUAL_ROBOT_TYPE}, source=left_arm/left_hand, "
            "destination=right_arm/right_hand.",
            "task_success requires the object to settle within "
            f"{HANDOVER_DELIVERY_TOLERANCE_M:.3f} m of the commanded delivery "
            f"pose {FINAL_OBJECT_XYZ} and never to fall below "
            f"{HANDOVER_DROP_Z_M:.3f} m during the transfer.",
            "object_dropped uses the minimum object height observed across the "
            "whole replay, so a mid-transfer drop is caught even if the object "
            "later comes to rest near the target.",
            "Planning time excludes a discarded warm-up compile and includes "
            "grasp-pose generation.",
            *summarize_video_recording(args, report_results, video_paths),
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
    from scripts.tutorials.atomic_action.tutorial_utils import run_tutorial

    run_tutorial(main)


__all__ = ["add_benchmark_args", "run_all_benchmarks"]
