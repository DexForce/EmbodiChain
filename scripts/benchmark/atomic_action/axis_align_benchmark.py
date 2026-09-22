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

"""Benchmark AxisAlign atomic action on the tutorial cube.

Reports the ordered stages defined for AxisAlign in ``BENCHMARK_STANDARD.md``:
grasped, aligned, held. The angle between the object's internal axis and the
commanded target axis is scored at the end of the commanded alignment segment,
and the object must still be held, and not dropped, after the settle.
Run: embodichain benchmark atomic-action --action axis_align
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

from scripts.benchmark.atomic_action.common import (
    CPU_MEMORY_BACKEND,
    DEFAULT_REPLAY_STEPS_PER_WAYPOINT,
    ENGAGED_MIN_DISPLACEMENT,
    SKILL_STAGES,
    StageLadder,
    TASK_ROTATION_TOLERANCE_RAD,
    add_common_benchmark_args,
    add_grasp_benchmark_args,
    build_stage_leaderboard,
    build_video_output_path,
    check_motion_valid,
    ensure_repo_root,
    ensure_torch,
    dropped_below_support,
    format_float,
    object_position_tuple,
    replay_and_track_channels,
    replay_trajectory_with_recording,
    reset_rigid_object,
    reset_robot,
    resolve_profile,
    should_record_case,
    summarize_video_recording,
    timed_call,
    warmup_planning,
    write_markdown_report,
)

INTERNAL_AXIS = (1.0, 0.0, 0.0)
ALIGN_SAMPLE_INTERVAL = 180
HAND_INTERP_STEPS = 12
PRE_GRASP_DISTANCE = 0.15
LIFT_HEIGHT = 0.16
REPLAY_HOLD_STEPS = 60


@dataclass(frozen=True)
class AxisAlignCase:
    """AxisAlign target-axis benchmark case."""

    name: str
    target_axis: tuple[float, float, float]


AXIS_ALIGN_CASES: dict[str, AxisAlignCase] = {
    "upright": AxisAlignCase("upright", (0.0, 0.0, 1.0)),
    "horizontal": AxisAlignCase("horizontal", (0.0, 1.0, 0.0)),
}
DEFAULT_AXIS_ALIGN_CASES = tuple(AXIS_ALIGN_CASES.keys())
SMOKE_AXIS_ALIGN_CASE = "upright"


def add_benchmark_args(parser: argparse.ArgumentParser) -> None:
    """Add AxisAlign benchmark CLI arguments."""
    parser.add_argument(
        "--align_cases",
        nargs="+",
        choices=(*AXIS_ALIGN_CASES.keys(), "all"),
        default=list(DEFAULT_AXIS_ALIGN_CASES),
        help="Target-axis alignment cases to benchmark. Use 'all' for every case.",
    )
    add_grasp_benchmark_args(parser)
    add_common_benchmark_args(parser)


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark AxisAlign over target-axis alignment cases."
    )
    add_benchmark_args(parser)
    return parser.parse_args()


def _select_cases(case_names: list[str]) -> list[AxisAlignCase]:
    """Resolve selected alignment case names."""
    if "all" in case_names:
        return list(AXIS_ALIGN_CASES.values())
    return [AXIS_ALIGN_CASES[name] for name in case_names]


def object_axis_angle_rad(obj, internal_axis, target_axis) -> float:
    """Return the angle between the object's internal axis and a target axis.

    Args:
        obj: Rigid object whose world rotation carries the internal axis.
        internal_axis: Object-local axis tensor with shape ``(3,)``.
        target_axis: World-frame target axis tensor with shape ``(3,)``.

    Returns:
        Angle in radians, in ``[0, pi]``.
    """
    torch = ensure_torch()
    rotation = obj.get_local_pose(to_matrix=True)[0, :3, :3]
    world_axis = rotation.to(internal_axis.dtype) @ internal_axis
    world_axis = world_axis / torch.linalg.vector_norm(world_axis)
    target = target_axis / torch.linalg.vector_norm(target_axis)
    sin_angle = torch.linalg.vector_norm(torch.linalg.cross(world_axis, target))
    cos_angle = torch.dot(world_axis, target)
    return float(torch.atan2(sin_angle, cos_angle))


def _build_invocation(
    atomic_engine, semantics, case: AxisAlignCase, device, physics_dt
):
    """Build the AxisAlign invocation and planning context for one case."""
    torch = ensure_torch()
    from embodichain.lab.sim.atomic_actions import (
        AxisAlignGoal,
        AxisAlignOptions,
        MotionPolicy,
    )

    invocation = atomic_engine.make_invocation(
        "axis_align",
        AxisAlignGoal(semantics),
        control_parts={"primary": {"motion": "arm", "grasp": "hand"}},
        motion_policy=MotionPolicy(
            strategy="motion_gen", sample_count=ALIGN_SAMPLE_INTERVAL
        ),
        skill_options=AxisAlignOptions(
            target_axis=torch.tensor(
                case.target_axis, dtype=torch.float32, device=device
            ),
            approach_direction=torch.tensor(
                [0.0, 0.0, -1.0], dtype=torch.float32, device=device
            ),
            pre_grasp_distance=PRE_GRASP_DISTANCE,
            lift_height=LIFT_HEIGHT,
            hand_interp_steps=HAND_INTERP_STEPS,
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
    case: AxisAlignCase,
    repeat: int,
    args: argparse.Namespace,
    recorded_count: int,
) -> dict[str, object]:
    """Run one AxisAlign case through its stages."""
    torch = ensure_torch()
    ladder = StageLadder(stages=SKILL_STAGES["axis_align"])
    reset_robot(robot, initial_qpos)
    reset_rigid_object(obj, initial_object_pose)
    sim.update(step=10)

    internal_axis = torch.tensor(INTERNAL_AXIS, dtype=torch.float32, device=sim.device)
    target_axis = torch.tensor(case.target_axis, dtype=torch.float32, device=sim.device)
    initial_angle = object_axis_angle_rad(obj, internal_axis, target_axis)

    invocation, context = _build_invocation(
        atomic_engine, semantics, case, sim.device, sim.sim_config.physics_dt
    )
    try:
        elapsed, mem_delta, peak_gpu, result = timed_call(
            lambda: atomic_engine.compile((invocation,), context)
        )
    except Exception as exc:
        print(f"    planner exception: {type(exc).__name__}: {exc}")
        ladder.fail("grasped", "planner_exception")
        return _case_result(case, repeat, ladder, 0.0, None, None, initial_angle, "")

    plan_success = bool(result.plan_success.all().item())
    traj = result.trajectory.positions
    if not plan_success:
        ladder.fail("grasped", "planner_reported_failure")
        return _case_result(
            case, repeat, ladder, elapsed, mem_delta, None, initial_angle, "", peak_gpu
        )

    motion_valid, motion_detail = check_motion_valid(traj, robot)
    ladder.motion_valid = motion_valid
    ladder.motion_detail = motion_detail
    if motion_detail == "non_finite_trajectory":
        ladder.fail("grasped", "planner_reported_failure")
        return _case_result(
            case,
            repeat,
            ladder,
            elapsed,
            mem_delta,
            None,
            initial_angle,
            "",
            peak_gpu,
            traj,
        )

    try:
        measure_waypoint = result.segment(0, "manipulate").stop - 1
    except (KeyError, AttributeError, IndexError):
        measure_waypoint = int(traj.shape[1]) - 1

    try:
        traces = replay_and_track_channels(
            sim=sim,
            robot=robot,
            traj=traj,
            readers={
                "angle": lambda _index: object_axis_angle_rad(
                    obj, internal_axis, target_axis
                ),
                "height": lambda _index: object_position_tuple(obj)[2],
            },
            measure_waypoint=measure_waypoint,
            steps_per_waypoint=DEFAULT_REPLAY_STEPS_PER_WAYPOINT,
            hold_steps=REPLAY_HOLD_STEPS,
        )
    except Exception as exc:
        print(f"    replay exception: {type(exc).__name__}: {exc}")
        traces = None

    trace = None if traces is None else traces["angle"]
    if trace is None:
        ladder.fail("grasped", "invalid_case")
    else:
        height = traces["height"]
        ladder.max_tracking_error_rad = trace.max_tracking_error_rad
        # The object can only be reoriented while the hand holds it, so the
        # axis having moved off its initial angle is the evidence of a grasp.
        ladder.record(
            "grasped",
            abs(trace.measured_displacement) > ENGAGED_MIN_DISPLACEMENT,
            "object_not_grasped",
        )
        # The tracked scalar is already the axis angle error to the target.
        ladder.record(
            "aligned",
            trace.measured_position <= TASK_ROTATION_TOLERANCE_RAD,
            "task_goal_miss",
        )
        dropped = dropped_below_support(height.min_position, height.initial_position)
        ladder.record(
            "held",
            (not dropped) and trace.settled_position <= TASK_ROTATION_TOLERANCE_RAD,
            "object_dropped" if dropped else "task_goal_miss",
        )

    video_path = ""
    if should_record_case(args, recorded_count, ladder.success):
        reset_robot(robot, initial_qpos)
        reset_rigid_object(obj, initial_object_pose)
        recorded = replay_trajectory_with_recording(
            sim=sim,
            robot=robot,
            traj=traj,
            args=args,
            video_path=build_video_output_path(
                args, "atomic_action_axis_align", f"{case.name}_r{repeat}"
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
        initial_angle,
        video_path,
        peak_gpu,
        traj,
    )


def _case_result(
    case: AxisAlignCase,
    repeat: int,
    ladder: StageLadder,
    elapsed: float,
    mem_delta: dict[str, float] | None,
    trace,
    initial_angle: float,
    video_path: str,
    peak_gpu: float = 0.0,
    traj=None,
) -> dict[str, object]:
    """Assemble one AxisAlign case result row."""
    mem_delta = mem_delta or {"cpu_mb": 0.0, "gpu_mb": 0.0}
    return {
        "case_id": f"{case.name}:r{repeat}",
        "align_case": case.name,
        "repeat": repeat,
        "ladder": ladder,
        "target_axis": str(case.target_axis),
        "initial_axis_angle_rad": initial_angle,
        "cost_time_ms": elapsed * 1000.0,
        "cpu_delta_mb": mem_delta["cpu_mb"],
        "gpu_delta_mb": mem_delta["gpu_mb"],
        "peak_gpu_mb": peak_gpu,
        "max_tracking_error_rad": trace.max_tracking_error_rad if trace else None,
        "final_axis_angle_rad": trace.measured_position if trace else None,
        "settled_axis_angle_rad": trace.settled_position if trace else None,
        "trajectory_waypoints": (
            int(traj.shape[1]) if traj is not None and traj.ndim >= 3 else 0
        ),
        "success": ladder.success,
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
                "impl": "axis_align",
                "case_id": result["case_id"],
                "align_case": result["align_case"],
                "repeat": result["repeat"],
                "cost_time_ms": format_float(result["cost_time_ms"], precision=2),
                "cpu_delta_mb": format_float(result["cpu_delta_mb"], precision=1),
                "gpu_delta_mb": format_float(result["gpu_delta_mb"], precision=1),
                "peak_gpu_mb": format_float(result["peak_gpu_mb"], precision=1),
                "trajectory_waypoints": result["trajectory_waypoints"],
            }
        )
        ladder: StageLadder = result["ladder"]  # type: ignore[assignment]
        metric_rows.append(
            {
                "sample_size": 1,
                "impl": "axis_align",
                "case_id": result["case_id"],
                "align_case": result["align_case"],
                "target_axis": result["target_axis"],
                **ladder.as_row_fields(),
                "max_tracking_error_rad": format_float(
                    result["max_tracking_error_rad"], 4
                ),
                "initial_axis_angle_rad": format_float(
                    result["initial_axis_angle_rad"], 4
                ),
                "final_axis_angle_rad": format_float(result["final_axis_angle_rad"], 4),
                "settled_axis_angle_rad": format_float(
                    result["settled_axis_angle_rad"], 4
                ),
            }
        )
    return perf_rows, metric_rows


def run_all_benchmarks(args: argparse.Namespace | None = None) -> Path:
    """Run AxisAlign benchmark and write a markdown report."""
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
    from scripts.tutorials.atomic_action.axis_align import (
        create_align_object,
        create_axis_align_semantics,
    )
    from scripts.tutorials.atomic_action.tutorial_utils import (
        add_ur5_gripper_robot,
        create_parallel_jaw_grasp_pose_generator,
        create_toppra_motion_generator,
        get_hand_open_close_qpos,
        initialize_benchmark_simulation,
        initialize_pre_pick_robot_pose,
    )

    cases = _select_cases(args.align_cases)
    repeat = 1 if profile == "smoke" else args.repeat
    if profile == "smoke":
        cases = [AXIS_ALIGN_CASES[SMOKE_AXIS_ALIGN_CASE]]

    print("=" * 60)
    print("AxisAlign Atomic Action Benchmark")
    print("=" * 60)
    print(
        f"Coverage: profile={profile}, {len(cases)} align case(s) x {repeat} repeat(s)"
    )

    sim = initialize_benchmark_simulation(args)
    robot = add_ur5_gripper_robot(sim, tcp_z=0.15)
    obj = create_align_object(sim)
    sim.prepare()
    sim.update(step=10)
    hand_open, hand_close = get_hand_open_close_qpos(robot)
    # AxisAlign plans from the tutorial's pre-pick posture.
    initialize_pre_pick_robot_pose(robot, obj, hand_open)
    initial_qpos = robot.get_qpos().clone()
    initial_object_pose = obj.get_local_pose(to_matrix=True).clone()

    n_sample = 1000 if profile == "smoke" else args.n_sample
    # AxisAlign resolves the object pose from the scene by entity id, so the
    # engine must be bound to the simulation entity rather than constructed bare.
    atomic_engine = create_simulation_atomic_action_engine(
        motion_generator=create_toppra_motion_generator(robot),
        scene_entities=(obj,),
        control_profiles={
            "hand": ControlPartCommandProfile.joint_positions(
                open=hand_open, grasp=hand_close
            )
        },
        grasp_pose_generators={
            "hand": create_parallel_jaw_grasp_pose_generator(
                n_sample=n_sample, force_refresh=args.force_reannotate
            )
        },
    )
    semantics = create_axis_align_semantics(obj, INTERNAL_AXIS)

    print("Warm-up: running one discarded compile to exclude first-call cost...")
    warmup_invocation, warmup_context = _build_invocation(
        atomic_engine, semantics, cases[0], sim.device, sim.sim_config.physics_dt
    )
    warmup_planning(lambda: atomic_engine.compile((warmup_invocation,), warmup_context))
    reset_robot(robot, initial_qpos)
    reset_rigid_object(obj, initial_object_pose)

    results: list[dict[str, object]] = []
    video_paths: list[str] = []
    print("\n=== AxisAlign Target-Axis Sweep ===")
    for case in cases:
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
            results.append(result)
            if result["video_path"]:
                video_paths.append(str(result["video_path"]))
            ladder: StageLadder = result["ladder"]  # type: ignore[assignment]
            stages = " ".join(
                f"{stage}={'ok' if ladder.passed[stage] else 'FAIL'}"
                for stage in ladder.stages
                if stage in ladder.passed
            )
            print(
                f"  {result['case_id']:<22} "
                f"time={result['cost_time_ms']:>9.2f} ms | "
                f"{stages} "
                f"angle0={format_float(result['initial_axis_angle_rad'], 4)} "
                f"angle={format_float(result['final_axis_angle_rad'], 4)} "
                f"[{ladder.failure_reason or 'ok'}]"
            )

    perf_rows, metric_rows = _build_rows(results)
    leaderboard_rows = build_stage_leaderboard("axis_align", results)
    report_path = write_markdown_report(
        benchmark_name="atomic_action_axis_align",
        perf_rows=perf_rows,
        metric_rows=metric_rows,
        leaderboard_rows=leaderboard_rows,
        notes=[
            f"Profile: {profile}",
            f"CPU memory backend: {CPU_MEMORY_BACKEND}",
            f"Grasp samples: {n_sample}",
            f"Object internal axis: {INTERNAL_AXIS}.",
            "Stages: grasped (the object's axis left its initial angle), "
            "aligned (the angle between the object's internal axis and the "
            f"commanded target axis is at most {TASK_ROTATION_TOLERANCE_RAD:.3f} "
            f"rad, {math.degrees(TASK_ROTATION_TOLERANCE_RAD):.1f} deg, at the "
            "end of the commanded 'manipulate' segment), held (it is still "
            "within that angle after the settle and was never dropped).",
            "motion_valid and max_tracking_error_rad are diagnostics; neither "
            "fails a stage.",
            "initial_axis_angle_rad is the angle before the skill runs; a case "
            "only demonstrates alignment when it starts meaningfully misaligned.",
            "Planning time excludes a discarded warm-up compile and includes "
            "grasp-pose generation.",
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
    from scripts.tutorials.atomic_action.tutorial_utils import run_tutorial

    run_tutorial(main)


__all__ = ["add_benchmark_args", "object_axis_angle_rad", "run_all_benchmarks"]
