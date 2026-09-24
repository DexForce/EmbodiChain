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

"""Benchmark Pour atomic action as a PickUp-then-Pour sequence.

Reports the ordered stage defined for Pour in ``BENCHMARK_STANDARD.md``:
poured. Pour rotates a held object about its own internal axis and returns it,
so the stage is measured as the peak signed rotation the object actually
turned about that axis during the commanded pour segment.
Run: embodichain benchmark atomic-action --action pour
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
    format_float,
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
)

POUR_INTERNAL_AXIS = (1.0, 0.0, 0.0)
APPROACH_DIRECTION = (-0.707, 0.0, -0.707)
OBJ_POSITION = (-0.5, 0.0, 0.0)
PICK_SAMPLE_INTERVAL = 120
POUR_SAMPLE_INTERVAL = 80
HAND_INTERP_STEPS = 12
PRE_GRASP_DISTANCE = 0.15
LIFT_HEIGHT = 0.16
REPLAY_HOLD_STEPS = 60
# Commanded-versus-achieved pour rotation tolerance. Uses the same angular
# tolerance as the other rotation skills; see BENCHMARK_STANDARD.md.


@dataclass(frozen=True)
class PourCase:
    """Pour commanded-rotation benchmark case."""

    name: str
    rotate_angle_rad: float


POUR_CASES: dict[str, PourCase] = {
    "pour_30": PourCase("pour_30", math.radians(30.0)),
    "pour_45": PourCase("pour_45", math.radians(45.0)),
    "pour_90": PourCase("pour_90", math.radians(90.0)),
}
DEFAULT_POUR_CASES = tuple(POUR_CASES.keys())
SMOKE_POUR_CASE = "pour_45"


def add_benchmark_args(parser: argparse.ArgumentParser) -> None:
    """Add Pour benchmark CLI arguments."""
    parser.add_argument(
        "--pour_cases",
        nargs="+",
        choices=(*POUR_CASES.keys(), "all"),
        default=list(DEFAULT_POUR_CASES),
        help="Commanded pour-rotation cases to benchmark. Use 'all' for every case.",
    )
    add_grasp_benchmark_args(parser)
    add_common_benchmark_args(parser)


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark Pour over commanded rotation angles."
    )
    add_benchmark_args(parser)
    return parser.parse_args()


def _select_cases(case_names: list[str]) -> list[PourCase]:
    """Resolve selected pour case names."""
    if "all" in case_names:
        return list(POUR_CASES.values())
    return [POUR_CASES[name] for name in case_names]


def signed_rotation_about_axis_rad(rotation, reference_rotation, axis) -> float:
    """Return the signed rotation from a reference orientation about an axis.

    Pour turns the held object about the object's own internal axis, which that
    rotation leaves invariant. Measuring the axis direction is therefore
    useless; the achieved pour is the rotation angle of the relative transform
    projected onto the axis.

    Args:
        rotation: Current object rotation matrix with shape ``(3, 3)``.
        reference_rotation: Rotation captured when the pour segment started.
        axis: World-frame pour axis with shape ``(3,)``.

    Returns:
        Signed rotation in radians about ``axis``.
    """
    torch = ensure_torch()
    delta = rotation @ reference_rotation.transpose(0, 1)
    cos_angle = torch.clamp((torch.diagonal(delta).sum() - 1.0) / 2.0, -1.0, 1.0)
    angle = torch.arccos(cos_angle)
    # Rotation-vector direction recovers the sign about the requested axis.
    vector = torch.stack(
        [
            delta[2, 1] - delta[1, 2],
            delta[0, 2] - delta[2, 0],
            delta[1, 0] - delta[0, 1],
        ]
    )
    unit_axis = axis / torch.linalg.vector_norm(axis)
    sign = torch.sign(torch.dot(vector, unit_axis.to(vector.dtype)))
    if float(sign) == 0.0:
        sign = torch.ones_like(sign)
    return float(angle * sign)


def _build_invocations(atomic_engine, semantics, case: PourCase, device, physics_dt):
    """Build the PickUp-then-Pour invocations and planning context."""
    torch = ensure_torch()
    from embodichain.lab.sim.atomic_actions import (
        GraspGoal,
        MotionPolicy,
        PickUpOptions,
        PourGoal,
        PourOptions,
    )

    control_parts = {"primary": {"motion": "arm", "grasp": "hand"}}
    pick = atomic_engine.make_invocation(
        "pick_up",
        GraspGoal(semantics),
        control_parts=control_parts,
        # Leave the strategy at its default. Forcing strategy="motion_gen"
        # here makes PickUp fail to plan on this scene, which is how the Pour
        # tutorial's own motion policy breaks.
        motion_policy=MotionPolicy(sample_count=PICK_SAMPLE_INTERVAL),
        skill_options=PickUpOptions(
            approach_direction=torch.tensor(
                APPROACH_DIRECTION, dtype=torch.float32, device=device
            ),
            pre_grasp_distance=PRE_GRASP_DISTANCE,
            lift_height=LIFT_HEIGHT,
            hand_interp_steps=HAND_INTERP_STEPS,
        ),
    )
    pour = atomic_engine.make_invocation(
        "pour",
        PourGoal(),
        control_parts=control_parts,
        motion_policy=MotionPolicy(sample_count=POUR_SAMPLE_INTERVAL),
        skill_options=PourOptions(rotate_angle=case.rotate_angle_rad),
    )
    return (pick, pour), atomic_engine.initial_context(control_dt=physics_dt)


def _run_case(
    sim,
    robot,
    obj,
    atomic_engine,
    semantics,
    initial_qpos,
    initial_object_pose,
    case: PourCase,
    repeat: int,
    args: argparse.Namespace,
    recorded_count: int,
) -> dict[str, object]:
    """Run one Pour case through its stages."""
    torch = ensure_torch()
    ladder = StageLadder(stages=SKILL_STAGES["pour"])
    reset_robot(robot, initial_qpos)
    reset_rigid_object(obj, initial_object_pose)
    sim.update(step=10)

    invocations, context = _build_invocations(
        atomic_engine, semantics, case, sim.device, sim.sim_config.physics_dt
    )
    try:
        elapsed, mem_delta, peak_gpu, result = timed_call(
            lambda: atomic_engine.compile(invocations, context)
        )
    except Exception as exc:
        print(f"    planner exception: {type(exc).__name__}: {exc}")
        ladder.fail("poured", "planner_exception")
        return _case_result(case, repeat, ladder, 0.0, None, None, "")

    plan_success = bool(result.plan_success.all().item())
    traj = result.trajectory.positions
    if not plan_success:
        for plan in result.action_plans:
            if not plan.plan_success.all():
                messages = plan.diagnostics.messages or ("planning failed",)
                print(f"    plan failure [{plan.skill_id}]: {'; '.join(messages)}")
        ladder.fail("poured", "planner_reported_failure")
        return _case_result(
            case, repeat, ladder, elapsed, mem_delta, None, "", peak_gpu
        )

    motion_valid, motion_detail = check_motion_valid(traj, robot)
    ladder.motion_valid = motion_valid
    ladder.motion_detail = motion_detail
    if motion_detail == "non_finite_trajectory":
        ladder.fail("poured", "planner_reported_failure")
        return _case_result(
            case, repeat, ladder, elapsed, mem_delta, None, "", peak_gpu, traj
        )

    # Pour is the second invocation; score at the end of its own segment.
    pour_segment = result.segment(1, "pour")
    pour_start = pour_segment.start
    measure_waypoint = pour_segment.stop - 1

    internal_axis = torch.tensor(
        POUR_INTERNAL_AXIS, dtype=torch.float32, device=sim.device
    )
    state: dict[str, object] = {"reference": None, "axis": None, "pour_start_z": None}

    def read_pour_rotation(waypoint_index: int) -> float:
        rotation = obj.get_local_pose(to_matrix=True)[0, :3, :3]
        if state["reference"] is None or waypoint_index < pour_start:
            # Capture the orientation the pour starts from, so the PickUp
            # rotation that precedes it is not counted as pour rotation.
            state["reference"] = rotation.clone()
            state["axis"] = (rotation.to(internal_axis.dtype) @ internal_axis).clone()
            state["pour_start_z"] = float(obj.get_local_pose(to_matrix=True)[0, 2, 3])
            return 0.0
        return signed_rotation_about_axis_rad(
            rotation, state["reference"], state["axis"]
        )

    try:
        trace = replay_and_track_scalar(
            sim=sim,
            robot=robot,
            traj=traj,
            read_value=read_pour_rotation,
            measure_waypoint=measure_waypoint,
            steps_per_waypoint=DEFAULT_REPLAY_STEPS_PER_WAYPOINT,
            hold_steps=REPLAY_HOLD_STEPS,
        )
    except Exception as exc:
        print(f"    replay exception: {type(exc).__name__}: {exc}")
        trace = None

    rotation_error = None
    if trace is None:
        ladder.fail("poured", "invalid_case")
    else:
        ladder.max_tracking_error_rad = trace.max_tracking_error_rad
        # Pour rotates to the poured pose and then returns to the pose it
        # started from, all inside one "pour" segment, so the end of the
        # segment is back at zero rotation by construction. The achieved pour
        # is the peak signed rotation reached during the segment.
        rotation_error = abs(
            abs(trace.peak_signed_displacement) - abs(case.rotate_angle_rad)
        )
        ladder.record(
            "poured",
            rotation_error <= TASK_ROTATION_TOLERANCE_RAD,
            "task_goal_miss",
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
                args, "atomic_action_pour", f"{case.name}_r{repeat}"
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
        video_path,
        peak_gpu,
        traj,
        rotation_error,
        state["pour_start_z"],
        float(obj.get_local_pose(to_matrix=True)[0, 2, 3]),
    )


def _case_result(
    case: PourCase,
    repeat: int,
    ladder: StageLadder,
    elapsed: float,
    mem_delta: dict[str, float] | None,
    trace,
    video_path: str,
    peak_gpu: float = 0.0,
    traj=None,
    rotation_error: float | None = None,
    pour_start_object_z: float | None = None,
    final_object_z: float | None = None,
) -> dict[str, object]:
    """Assemble one Pour case result row."""
    mem_delta = mem_delta or {"cpu_mb": 0.0, "gpu_mb": 0.0}
    return {
        "case_id": f"{case.name}:r{repeat}",
        "pour_case": case.name,
        "repeat": repeat,
        "ladder": ladder,
        "commanded_rotation_rad": case.rotate_angle_rad,
        "cost_time_ms": elapsed * 1000.0,
        "cpu_delta_mb": mem_delta["cpu_mb"],
        "gpu_delta_mb": mem_delta["gpu_mb"],
        "peak_gpu_mb": peak_gpu,
        "max_tracking_error_rad": trace.max_tracking_error_rad if trace else None,
        "achieved_rotation_rad": trace.measured_position if trace else None,
        "peak_rotation_rad": trace.peak_signed_displacement if trace else None,
        "settled_rotation_rad": trace.settled_position if trace else None,
        "rotation_error_rad": rotation_error,
        "pour_start_object_z_m": pour_start_object_z,
        "final_object_z_m": final_object_z,
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
                "impl": "pour",
                "case_id": result["case_id"],
                "pour_case": result["pour_case"],
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
                "impl": "pour",
                "case_id": result["case_id"],
                "pour_case": result["pour_case"],
                **ladder.as_row_fields(),
                "commanded_rotation_rad": format_float(
                    result["commanded_rotation_rad"], 4
                ),
                "max_tracking_error_rad": format_float(
                    result["max_tracking_error_rad"], 4
                ),
                "achieved_rotation_rad": format_float(
                    result["achieved_rotation_rad"], 4
                ),
                "peak_rotation_rad": format_float(result["peak_rotation_rad"], 4),
                "settled_rotation_rad": format_float(result["settled_rotation_rad"], 4),
                "rotation_error_rad": format_float(result["rotation_error_rad"], 4),
                "pour_start_object_z_m": format_float(
                    result["pour_start_object_z_m"], 4
                ),
                "final_object_z_m": format_float(result["final_object_z_m"], 4),
            }
        )
    return perf_rows, metric_rows


def run_all_benchmarks(args: argparse.Namespace | None = None) -> Path:
    """Run Pour benchmark and write a markdown report."""
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

    cases = _select_cases(args.pour_cases)
    repeat = 1 if profile == "smoke" else args.repeat
    if profile == "smoke":
        cases = [POUR_CASES[SMOKE_POUR_CASE]]

    print("=" * 60)
    print("Pour Atomic Action Benchmark")
    print("=" * 60)
    print(
        f"Coverage: profile={profile}, {len(cases)} pour case(s) x {repeat} repeat(s)"
    )

    sim = initialize_benchmark_simulation(args)
    robot = add_ur5_gripper_robot(sim, tcp_z=0.15)
    obj = create_align_object(sim, obj_position=OBJ_POSITION)
    sim.prepare()
    sim.update(step=10)
    hand_open, hand_close = get_hand_open_close_qpos(robot)
    initialize_pre_pick_robot_pose(robot, obj, hand_open)
    initial_qpos = robot.get_qpos().clone()
    initial_object_pose = obj.get_local_pose(to_matrix=True).clone()

    n_sample = 1000 if profile == "smoke" else args.n_sample
    # Pour resolves the held object from the scene, so bind the engine to it.
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
    semantics = create_axis_align_semantics(obj, POUR_INTERNAL_AXIS)

    print("Warm-up: running one discarded compile to exclude first-call cost...")
    warmup_invocations, warmup_context = _build_invocations(
        atomic_engine, semantics, cases[0], sim.device, sim.sim_config.physics_dt
    )
    warmup_planning(lambda: atomic_engine.compile(warmup_invocations, warmup_context))
    reset_robot(robot, initial_qpos)
    reset_rigid_object(obj, initial_object_pose)

    results: list[dict[str, object]] = []
    video_paths: list[str] = []
    print("\n=== Pour Commanded-Rotation Sweep ===")
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
                f"peak={format_float(result['peak_rotation_rad'], 4)} "
                f"z0={format_float(result['pour_start_object_z_m'], 3)} "
                f"err={format_float(result['rotation_error_rad'], 4)} "
                f"[{ladder.failure_reason or 'ok'}]"
            )

    perf_rows, metric_rows = _build_rows(results)
    leaderboard_rows = build_stage_leaderboard("pour", results)
    report_path = write_markdown_report(
        benchmark_name="atomic_action_pour",
        perf_rows=perf_rows,
        metric_rows=metric_rows,
        leaderboard_rows=leaderboard_rows,
        notes=[
            f"Profile: {profile}",
            f"CPU memory backend: {CPU_MEMORY_BACKEND}",
            f"Grasp samples: {n_sample}",
            "Sequence: PickUp then Pour, compiled as one episode. Planning time "
            "covers both invocations.",
            "task_success requires the peak signed rotation of the object "
            "about its own internal axis during the commanded 'pour' segment "
            f"to match the commanded angle within {TASK_ROTATION_TOLERANCE_RAD:.3f} rad.",
            "Pour rotates to the poured pose and returns to its starting pose "
            "inside a single 'pour' segment, so the end-of-segment rotation is "
            "zero by construction; the peak is the achieved pour.",
            "The rotation reference is captured at the pour segment start, so "
            "the preceding PickUp rotation is not credited to Pour.",
            "Planning time excludes a discarded warm-up compile.",
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


__all__ = [
    "add_benchmark_args",
    "run_all_benchmarks",
    "signed_rotation_about_axis_rad",
]
