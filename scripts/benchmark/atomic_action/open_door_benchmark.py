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

"""Benchmark OpenDoor atomic action across microwave hinge opening targets.

Reports the four-level success ladder (planning_success, motion_valid,
execution_success, task_success) defined in
``scripts/benchmark/motion_generation/BENCHMARK_DESIGN.md``. Task success
requires the physically replayed hinge to reach the commanded opening angle.
Run: embodichain benchmark atomic-action --action open_door
"""

from __future__ import annotations

import argparse
import math
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
    replay_and_track_joint,
    replay_trajectory_with_recording,
    reset_robot,
    resolve_profile,
    should_record_case,
    summarize_video_recording,
    timed_call,
    warmup_planning,
    write_markdown_report,
)


@dataclass(frozen=True)
class DoorCase:
    """OpenDoor hinge target benchmark case."""

    name: str
    open_angle_deg: float


DOOR_CASES: dict[str, DoorCase] = {
    "open_30": DoorCase("open_30", 30.0),
    "open_60": DoorCase("open_60", 60.0),
    "open_90": DoorCase("open_90", 90.0),
}
DEFAULT_DOOR_CASES = tuple(DOOR_CASES.keys())
SMOKE_DOOR_CASE = "open_60"
TRAJECTORY_SAMPLE_COUNT = 120
HAND_INTERP_STEPS = 12
DOOR_WAYPOINT_COUNT = 50
REPLAY_HOLD_STEPS = 60
# Measured hinge error of a healthy replay is ~0.02 rad; 0.15 rad (8.6 deg)
# keeps that comfortably inside while still rejecting a door that never moved.
HINGE_SUCCESS_TOLERANCE_RAD = 0.15


def add_benchmark_args(parser: argparse.ArgumentParser) -> None:
    """Add OpenDoor benchmark CLI arguments."""
    parser.add_argument(
        "--door_cases",
        nargs="+",
        choices=(*DOOR_CASES.keys(), "all"),
        default=list(DEFAULT_DOOR_CASES),
        help="Hinge opening cases to benchmark. Use 'all' for every case.",
    )
    add_grasp_benchmark_args(parser)
    add_common_benchmark_args(parser)


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark OpenDoor over microwave hinge opening targets."
    )
    add_benchmark_args(parser)
    return parser.parse_args()


def _select_cases(case_names: list[str]) -> list[DoorCase]:
    """Resolve selected door case names."""
    if "all" in case_names:
        return list(DOOR_CASES.values())
    return [DOOR_CASES[name] for name in case_names]


def _reset_scene(robot, microwave, initial_qpos, initial_hinge_qpos) -> None:
    """Restore the robot and the microwave hinge to their initial state."""
    reset_robot(robot, initial_qpos)
    microwave.set_qpos(initial_hinge_qpos, target=False)
    microwave.set_qpos(initial_hinge_qpos, target=True)
    microwave.clear_dynamics()


def _build_invocation(atomic_engine, microwave, case: DoorCase, physics_dt: float):
    """Build the OpenDoor invocation and planning context for one case."""
    from embodichain.lab.sim.atomic_actions import (
        MotionPolicy,
        ObservedArticulationJointState,
        OpenDoorGoal,
        OpenDoorOptions,
        SceneSnapshot,
    )
    from scripts.tutorials.atomic_action.open_door import (
        HANDLE_LINK_NAME,
        MICROWAVE_SCENE_ENTITY_ID,
        create_door_handle_semantics,
    )

    semantics = create_door_handle_semantics(microwave)
    affordance = semantics.affordance
    lower_limit, upper_limit = affordance.joint_limits
    closed_position = lower_limit if affordance.opening_direction > 0 else upper_limit
    open_position = upper_limit if affordance.opening_direction > 0 else lower_limit
    open_angle = math.radians(case.open_angle_deg)
    open_fraction = (open_angle - closed_position) / (open_position - closed_position)

    handle_pose = microwave.get_link_pose(HANDLE_LINK_NAME, to_matrix=True)
    hinge_joint_index = microwave.joint_names.index(affordance.joint_name)
    hinge_position = microwave.get_qpos(target=False)[
        :, hinge_joint_index : hinge_joint_index + 1
    ]

    invocation = atomic_engine.make_invocation(
        "open_door",
        OpenDoorGoal(semantics, handle_pose, open_fraction=open_fraction),
        control_parts={"primary": {"motion": "arm", "grasp": "hand"}},
        motion_policy=MotionPolicy(sample_count=TRAJECTORY_SAMPLE_COUNT),
        skill_options=OpenDoorOptions(
            hand_interp_steps=HAND_INTERP_STEPS,
            door_waypoint_count=DOOR_WAYPOINT_COUNT,
        ),
    )
    # OpenDoor resolves the hinge goal against the observed joint state, so the
    # planning context must carry it exactly as the tutorial does.
    context = atomic_engine.initial_context(
        scene=SceneSnapshot(
            timestamp=0.0,
            version=0,
            articulation_joints={
                (
                    MICROWAVE_SCENE_ENTITY_ID,
                    affordance.joint_name,
                ): ObservedArticulationJointState(hinge_position)
            },
        ),
        control_dt=physics_dt,
    )
    return invocation, context, hinge_joint_index, open_angle


def _door_measure_waypoint(compiled, traj) -> int:
    """Return the last waypoint of the commanded door-actuation segment."""
    try:
        return compiled.segment(0, "open").stop - 1
    except (KeyError, AttributeError, IndexError):
        return int(traj.shape[1]) - 1


def _run_case(
    sim,
    robot,
    microwave,
    atomic_engine,
    initial_qpos,
    initial_hinge_qpos,
    case: DoorCase,
    repeat: int,
    args: argparse.Namespace,
    recorded_count: int,
) -> dict[str, object]:
    """Run one OpenDoor case through the full four-level success ladder."""
    ladder = SuccessLadder()
    _reset_scene(robot, microwave, initial_qpos, initial_hinge_qpos)
    sim.update(step=4)

    invocation, context, hinge_joint_index, open_angle = _build_invocation(
        atomic_engine, microwave, case, sim.sim_config.physics_dt
    )

    try:
        elapsed, mem_delta, peak_gpu, result = timed_call(
            lambda: atomic_engine.compile((invocation,), context)
        )
    except Exception as exc:
        print(f"    planner exception: {type(exc).__name__}: {exc}")
        ladder.fail("planning_success", "planner_exception")
        return _case_result(case, repeat, ladder, 0.0, None, None, open_angle, "")

    ladder.planning_success = bool(result.plan_success.all().item())
    traj = result.trajectory.positions
    if not ladder.planning_success:
        ladder.fail("planning_success", "planner_reported_failure")
        return _case_result(
            case, repeat, ladder, elapsed, mem_delta, None, open_angle, "", peak_gpu
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
            open_angle,
            "",
            peak_gpu,
            traj,
        )

    # Score the hinge at the end of the commanded door-actuation segment. After
    # the release and retract segments the hand is open and this microwave's
    # hinge is undriven, so it swings freely and its resting angle measures the
    # asset's dynamics rather than what OpenDoor achieved.
    measure_waypoint = _door_measure_waypoint(result, traj)

    try:
        trace = replay_and_track_joint(
            sim=sim,
            robot=robot,
            traj=traj,
            articulation=microwave,
            joint_index=hinge_joint_index,
            measure_waypoint=measure_waypoint,
            # This door swing needs a slower replay than the shared default.
            # At 16 the arm trails the plan by 0.66 rad and the tracking
            # gate rejects the run; 64 keeps it inside the gate in both
            # profiles and lands the hinge within 0.001 rad of the command.
            steps_per_waypoint=4 * DEFAULT_REPLAY_STEPS_PER_WAYPOINT,
            hold_steps=REPLAY_HOLD_STEPS,
        )
        ladder.execution_success = trace is not None
    except Exception as exc:
        print(f"    replay exception: {type(exc).__name__}: {exc}")
        trace = None
        ladder.fail("execution_success", "controller_tracking_failure")

    if trace is not None and (
        trace.max_tracking_error_rad > REPLAY_TRACKING_TOLERANCE_RAD
    ):
        # The arm did not follow the plan, so the hinge reading describes the
        # drive rather than OpenDoor. Do not score task success.
        ladder.execution_success = False
        ladder.fail("execution_success", "controller_tracking_failure")
    elif ladder.execution_success and trace is not None:
        # The hinge target is an absolute angle, so compare the replayed
        # absolute position rather than the displacement.
        hinge_error = abs(trace.measured_position - open_angle)
        ladder.task_success = hinge_error <= HINGE_SUCCESS_TOLERANCE_RAD
        if not ladder.task_success:
            ladder.fail("task_success", "task_goal_miss")
    elif not ladder.failure_stage:
        ladder.fail("execution_success", "controller_tracking_failure")

    video_path = ""
    if should_record_case(args, recorded_count, ladder.task_success):
        _reset_scene(robot, microwave, initial_qpos, initial_hinge_qpos)
        recorded = replay_trajectory_with_recording(
            sim=sim,
            robot=robot,
            traj=traj,
            args=args,
            video_path=build_video_output_path(
                args, "atomic_action_open_door", f"{case.name}_r{repeat}"
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
        open_angle,
        video_path,
        peak_gpu,
        traj,
    )


def _case_result(
    case: DoorCase,
    repeat: int,
    ladder: SuccessLadder,
    elapsed: float,
    mem_delta: dict[str, float] | None,
    trace,
    open_angle: float,
    video_path: str,
    peak_gpu: float = 0.0,
    traj=None,
) -> dict[str, object]:
    """Assemble one OpenDoor case result row."""
    mem_delta = mem_delta or {"cpu_mb": 0.0, "gpu_mb": 0.0}
    hinge_error = None
    if trace is not None:
        hinge_error = abs(trace.measured_position - open_angle)
    return {
        "case_id": f"{case.name}:r{repeat}",
        "door_case": case.name,
        "repeat": repeat,
        "ladder": ladder,
        "target_angle_rad": math.radians(case.open_angle_deg),
        "cost_time_ms": elapsed * 1000.0,
        "cpu_delta_mb": mem_delta["cpu_mb"],
        "gpu_delta_mb": mem_delta["gpu_mb"],
        "peak_gpu_mb": peak_gpu,
        "max_tracking_error_rad": trace.max_tracking_error_rad if trace else None,
        "hinge_initial_rad": trace.initial_position if trace else None,
        "hinge_measured_rad": trace.measured_position if trace else None,
        "hinge_settled_rad": trace.settled_position if trace else None,
        "hinge_peak_signed_rad": trace.peak_signed_displacement if trace else None,
        "hinge_error_rad": hinge_error,
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
                "impl": "open_door",
                "case_id": result["case_id"],
                "door_case": result["door_case"],
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
                "impl": "open_door",
                "case_id": result["case_id"],
                "door_case": result["door_case"],
                "success_rate": f"{float(ladder.task_success):.6f}",
                **ladder.as_row_fields(),
                "target_angle_rad": format_float(result["target_angle_rad"], 4),
                "max_tracking_error_rad": format_float(
                    result["max_tracking_error_rad"], 4
                ),
                "hinge_initial_rad": format_float(result["hinge_initial_rad"], 4),
                "hinge_measured_rad": format_float(result["hinge_measured_rad"], 4),
                "hinge_peak_signed_rad": format_float(
                    result["hinge_peak_signed_rad"], 4
                ),
                "hinge_settled_rad": format_float(result["hinge_settled_rad"], 4),
                "hinge_error_rad": format_float(result["hinge_error_rad"], 4),
            }
        )
    return perf_rows, metric_rows


def run_all_benchmarks(args: argparse.Namespace | None = None) -> Path:
    """Run OpenDoor benchmark and write a markdown report."""
    args = _parse_args() if args is None else args
    if args.repeat < 1:
        raise ValueError("--repeat must be at least 1.")
    profile = resolve_profile(args)
    ensure_repo_root()
    ensure_torch()
    from embodichain.lab.sim.atomic_actions import (
        AtomicActionEngine,
        ControlPartCommandProfile,
    )
    from scripts.tutorials.atomic_action.open_door import create_microwave
    from scripts.tutorials.atomic_action.tutorial_utils import (
        add_ur5_gripper_robot,
        create_parallel_jaw_grasp_pose_generator,
        create_toppra_motion_generator,
        get_hand_open_close_qpos,
        initialize_benchmark_simulation,
    )

    cases = _select_cases(args.door_cases)
    repeat = 1 if profile == "smoke" else args.repeat
    if profile == "smoke":
        cases = [DOOR_CASES[SMOKE_DOOR_CASE]]

    print("=" * 60)
    print("OpenDoor Atomic Action Benchmark")
    print("=" * 60)
    print(
        f"Coverage: profile={profile}, {len(cases)} door case(s) x {repeat} repeat(s)"
    )

    sim = initialize_benchmark_simulation(args)
    robot = add_ur5_gripper_robot(
        sim,
        init_qpos=[0.0, -1.57, 1.57, -3.14, -1.57, 0.0, 0.0, 0.0],
        tcp_z=0.15,
    )
    microwave = create_microwave(sim)
    sim.prepare()
    sim.update(step=10)
    initial_qpos = robot.get_qpos().clone()
    initial_hinge_qpos = microwave.get_qpos(target=False).clone()
    hand_open, hand_close = get_hand_open_close_qpos(robot, close_qpos=0.04)

    n_sample = 1000 if profile == "smoke" else args.n_sample
    atomic_engine = AtomicActionEngine(
        motion_generator=create_toppra_motion_generator(robot),
        control_profiles={
            "hand": ControlPartCommandProfile.joint_positions(
                open=hand_open,
                grasp=hand_close,
            )
        },
        grasp_pose_generators={
            "hand": create_parallel_jaw_grasp_pose_generator(
                n_sample=n_sample,
                force_refresh=args.force_reannotate,
            )
        },
    )

    warmup_case = cases[0]
    warmup_invocation, warmup_context, _, _ = _build_invocation(
        atomic_engine, microwave, warmup_case, sim.sim_config.physics_dt
    )
    print("Warm-up: running one discarded compile to exclude first-call cost...")
    warmup_planning(lambda: atomic_engine.compile((warmup_invocation,), warmup_context))
    _reset_scene(robot, microwave, initial_qpos, initial_hinge_qpos)

    results: list[dict[str, object]] = []
    video_paths: list[str] = []
    print("\n=== OpenDoor Hinge Target Sweep ===")
    for case in cases:
        for repeat_index in range(repeat):
            result = _run_case(
                sim,
                robot,
                microwave,
                atomic_engine,
                initial_qpos,
                initial_hinge_qpos,
                case,
                repeat_index,
                args,
                len(video_paths),
            )
            results.append(result)
            if result["video_path"]:
                video_paths.append(str(result["video_path"]))
            ladder: SuccessLadder = result["ladder"]  # type: ignore[assignment]
            print(
                f"  {result['case_id']:<20} "
                f"time={result['cost_time_ms']:>9.2f} ms | "
                f"plan={ladder.planning_success} valid={ladder.motion_valid} "
                f"exec={ladder.execution_success} task={ladder.task_success} "
                f"hinge_err={format_float(result['hinge_error_rad'], 4)} "
                f"[{ladder.failure_reason or 'ok'}]"
            )

    perf_rows, metric_rows = _build_rows(results)
    leaderboard_rows = build_ladder_leaderboard("open_door", results)
    report_path = write_markdown_report(
        benchmark_name="atomic_action_open_door",
        perf_rows=perf_rows,
        metric_rows=metric_rows,
        leaderboard_rows=leaderboard_rows,
        notes=[
            f"Profile: {profile}",
            f"CPU memory backend: {CPU_MEMORY_BACKEND}",
            f"Grasp samples: {n_sample}",
            "task_success requires the physically replayed hinge angle to reach "
            f"the commanded target within {HINGE_SUCCESS_TOLERANCE_RAD} rad, "
            "measured at the last waypoint of the commanded door-actuation "
            "('open') segment.",
            "hinge_settled_rad is the resting angle after release and retract. "
            "This microwave hinge is undriven (drive_type='none'), so it swings "
            "freely once released; that value is diagnostic, not a success gate.",
            "Planning time excludes a discarded warm-up compile and includes "
            "handle grasp-pose generation.",
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


__all__ = ["add_benchmark_args", "run_all_benchmarks"]
