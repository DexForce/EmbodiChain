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

"""Benchmark Twist atomic action on the microwave power knob.

Reports the four-level success ladder (planning_success, motion_valid,
execution_success, task_success). Task success requires the knob's revolute
joint to reach the commanded twist angle during the commanded twist segment.
Run: embodichain benchmark atomic-action --action twist
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

from scripts.benchmark.atomic_action.common import (
    CPU_MEMORY_BACKEND,
    SuccessLadder,
    add_common_benchmark_args,
    build_ladder_leaderboard,
    build_video_output_path,
    ensure_repo_root,
    ensure_torch,
    format_float,
    replay_trajectory_with_recording,
    reset_robot,
    resolve_profile,
    run_articulated_contact_case,
    should_record_case,
    summarize_video_recording,
    warmup_planning,
    write_markdown_report,
)

KNOB_JOINT_NAME = "power_knob_rotation"
TWIST_SAMPLE_INTERVAL = 140
HAND_INTERP_STEPS = 12
PRE_GRASP_DISTANCE = 0.12
REPLAY_HOLD_STEPS = 60
# Commanded-versus-achieved knob rotation tolerance. Calibrated from the
# measured twist error on this asset; see BENCHMARK_STANDARD.md.
TWIST_TOLERANCE_RAD = 0.15


@dataclass(frozen=True)
class TwistCase:
    """Twist commanded-rotation benchmark case."""

    name: str
    twist_angle_rad: float


TWIST_CASES: dict[str, TwistCase] = {
    "twist_ccw_30": TwistCase("twist_ccw_30", math.radians(-30.0)),
    "twist_ccw_45": TwistCase("twist_ccw_45", math.radians(-45.0)),
    "twist_ccw_90": TwistCase("twist_ccw_90", math.radians(-90.0)),
}
DEFAULT_TWIST_CASES = tuple(TWIST_CASES.keys())
SMOKE_TWIST_CASE = "twist_ccw_45"


def add_benchmark_args(parser: argparse.ArgumentParser) -> None:
    """Add Twist benchmark CLI arguments."""
    parser.add_argument(
        "--twist_cases",
        nargs="+",
        choices=(*TWIST_CASES.keys(), "all"),
        default=list(DEFAULT_TWIST_CASES),
        help="Commanded knob-rotation cases to benchmark. Use 'all' for every case.",
    )
    add_common_benchmark_args(parser)


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark Twist over commanded knob rotation angles."
    )
    add_benchmark_args(parser)
    return parser.parse_args()


def _select_cases(case_names: list[str]) -> list[TwistCase]:
    """Resolve selected twist case names."""
    if "all" in case_names:
        return list(TWIST_CASES.values())
    return [TWIST_CASES[name] for name in case_names]


def _reset_scene(robot, microwave, initial_qpos, initial_knob_qpos) -> None:
    """Restore the robot and the microwave joints to their initial state."""
    reset_robot(robot, initial_qpos)
    microwave.set_qpos(initial_knob_qpos, target=False)
    microwave.set_qpos(initial_knob_qpos, target=True)
    microwave.clear_dynamics()


def _build_invocation(atomic_engine, microwave, case: TwistCase, physics_dt: float):
    """Build the Twist invocation and planning context for one case."""
    from embodichain.lab.sim.atomic_actions import MotionPolicy, TwistGoal, TwistOptions
    from scripts.tutorials.atomic_action.twist import create_knob_semantics

    semantics, target_pose = create_knob_semantics(microwave)
    invocation = atomic_engine.make_invocation(
        "twist",
        TwistGoal(semantics, target_pose),
        control_parts={"primary": {"motion": "arm", "grasp": "hand"}},
        motion_policy=MotionPolicy(
            strategy="motion_gen", sample_count=TWIST_SAMPLE_INTERVAL
        ),
        skill_options=TwistOptions(
            hand_interp_steps=HAND_INTERP_STEPS,
            pre_grasp_distance=PRE_GRASP_DISTANCE,
            twist_angle=case.twist_angle_rad,
        ),
    )
    return invocation, atomic_engine.initial_context(control_dt=physics_dt)


def _build_rows(results: list[dict[str, object]]):
    """Convert case results into report row dicts."""
    perf_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    for result in results:
        perf_rows.append(
            {
                "sample_size": 1,
                "impl": "twist",
                "case_id": result["case_id"],
                "twist_case": result["twist_case"],
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
                "impl": "twist",
                "case_id": result["case_id"],
                "twist_case": result["twist_case"],
                "success_rate": f"{float(ladder.task_success):.6f}",
                **ladder.as_row_fields(),
                "commanded_angle_rad": format_float(result["commanded_angle_rad"], 4),
                "max_tracking_error_rad": format_float(
                    result["max_tracking_error_rad"], 4
                ),
                "knob_initial_rad": format_float(result["knob_initial_rad"], 4),
                "knob_measured_disp_rad": format_float(
                    result["knob_measured_disp_rad"], 4
                ),
                "knob_peak_signed_rad": format_float(result["knob_peak_signed_rad"], 4),
                "knob_settled_disp_rad": format_float(
                    result["knob_settled_disp_rad"], 4
                ),
                "angle_error_rad": format_float(result["angle_error_rad"], 4),
            }
        )
    return perf_rows, metric_rows


def run_all_benchmarks(args: argparse.Namespace | None = None) -> Path:
    """Run Twist benchmark and write a markdown report."""
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
    from scripts.tutorials.atomic_action.tutorial_utils import (
        add_ur5_gripper_robot,
        create_toppra_motion_generator,
        get_hand_open_close_qpos,
        initialize_benchmark_simulation,
    )
    from scripts.tutorials.atomic_action.twist import create_microwave

    cases = _select_cases(args.twist_cases)
    repeat = 1 if profile == "smoke" else args.repeat
    if profile == "smoke":
        cases = [TWIST_CASES[SMOKE_TWIST_CASE]]

    print("=" * 60)
    print("Twist Atomic Action Benchmark")
    print("=" * 60)
    print(
        f"Coverage: profile={profile}, {len(cases)} twist case(s) x {repeat} repeat(s)"
    )

    sim = initialize_benchmark_simulation(args)
    robot = add_ur5_gripper_robot(
        sim, init_qpos=[0.0, -1.57, 1.57, -3.14, -1.57, 0.0, 0.0, 0.0]
    )
    microwave = create_microwave(sim)
    sim.prepare()
    sim.update(step=10)
    initial_qpos = robot.get_qpos().clone()
    initial_knob_qpos = microwave.get_qpos(target=False).clone()
    hand_open, hand_close = get_hand_open_close_qpos(robot)

    knob_joint_index = microwave.joint_names.index(KNOB_JOINT_NAME)
    print(
        f"Knob joint {KNOB_JOINT_NAME!r} (link cap_1), "
        f"tolerance={TWIST_TOLERANCE_RAD:.3f} rad"
    )

    atomic_engine = AtomicActionEngine(
        motion_generator=create_toppra_motion_generator(robot),
        control_profiles={
            "hand": ControlPartCommandProfile.joint_positions(
                open=hand_open, grasp=hand_close
            )
        },
    )

    print("Warm-up: running one discarded compile to exclude first-call cost...")
    warmup_invocation, warmup_context = _build_invocation(
        atomic_engine, microwave, cases[0], sim.sim_config.physics_dt
    )
    warmup_planning(lambda: atomic_engine.compile((warmup_invocation,), warmup_context))
    _reset_scene(robot, microwave, initial_qpos, initial_knob_qpos)

    results: list[dict[str, object]] = []
    video_paths: list[str] = []
    print("\n=== Twist Commanded-Rotation Sweep ===")
    for case in cases:
        for repeat_index in range(repeat):
            _reset_scene(robot, microwave, initial_qpos, initial_knob_qpos)
            sim.update(step=4)

            def evaluate(trace, case=case):
                error = abs(
                    abs(trace.measured_displacement) - abs(case.twist_angle_rad)
                )
                return (error <= TWIST_TOLERANCE_RAD, "task_goal_miss")

            outcome = run_articulated_contact_case(
                sim=sim,
                robot=robot,
                articulation=microwave,
                atomic_engine=atomic_engine,
                build_invocation=lambda case=case: _build_invocation(
                    atomic_engine, microwave, case, sim.sim_config.physics_dt
                ),
                joint_index=knob_joint_index,
                actuation_segment="twist",
                evaluate=evaluate,
                hold_steps=REPLAY_HOLD_STEPS,
            )
            trace = outcome.trace
            angle_error = None
            if trace is not None:
                angle_error = abs(
                    abs(trace.measured_displacement) - abs(case.twist_angle_rad)
                )

            video_path = ""
            if should_record_case(args, len(video_paths), outcome.ladder.task_success):
                _reset_scene(robot, microwave, initial_qpos, initial_knob_qpos)
                recorded = replay_trajectory_with_recording(
                    sim=sim,
                    robot=robot,
                    traj=outcome.trajectory,
                    args=args,
                    video_path=build_video_output_path(
                        args, "atomic_action_twist", f"{case.name}_r{repeat_index}"
                    ),
                )
                video_path = str(recorded) if recorded is not None else ""
                video_paths.append(video_path)

            result = {
                "case_id": f"{case.name}:r{repeat_index}",
                "twist_case": case.name,
                "repeat": repeat_index,
                "ladder": outcome.ladder,
                "commanded_angle_rad": case.twist_angle_rad,
                "cost_time_ms": outcome.elapsed_s * 1000.0,
                "cpu_delta_mb": outcome.cpu_delta_mb,
                "gpu_delta_mb": outcome.gpu_delta_mb,
                "peak_gpu_mb": outcome.peak_gpu_mb,
                "max_tracking_error_rad": (
                    trace.max_tracking_error_rad if trace else None
                ),
                "knob_initial_rad": trace.initial_position if trace else None,
                "knob_measured_disp_rad": (
                    trace.measured_displacement if trace else None
                ),
                "knob_peak_signed_rad": (
                    trace.peak_signed_displacement if trace else None
                ),
                "knob_settled_disp_rad": trace.settled_displacement if trace else None,
                "angle_error_rad": angle_error,
                "trajectory_waypoints": outcome.trajectory_waypoints,
                "success": outcome.ladder.task_success,
                "video_path": video_path,
            }
            results.append(result)
            ladder = outcome.ladder
            print(
                f"  {result['case_id']:<22} "
                f"time={result['cost_time_ms']:>9.2f} ms | "
                f"plan={ladder.planning_success} valid={ladder.motion_valid} "
                f"exec={ladder.execution_success} task={ladder.task_success} "
                f"rot={format_float(result['knob_measured_disp_rad'], 4)} "
                f"err={format_float(angle_error, 4)} "
                f"[{ladder.failure_reason or 'ok'}]"
            )

    perf_rows, metric_rows = _build_rows(results)
    leaderboard_rows = build_ladder_leaderboard("twist", results)
    report_path = write_markdown_report(
        benchmark_name="atomic_action_twist",
        perf_rows=perf_rows,
        metric_rows=metric_rows,
        leaderboard_rows=leaderboard_rows,
        notes=[
            f"Profile: {profile}",
            f"CPU memory backend: {CPU_MEMORY_BACKEND}",
            f"Knob joint: {KNOB_JOINT_NAME} (target link cap_1).",
            "task_success requires the replayed knob rotation at the end of the "
            "commanded 'twist' segment to match the commanded angle within "
            f"{TWIST_TOLERANCE_RAD:.3f} rad.",
            "knob_settled_disp_rad is the resting rotation after release and "
            "retract; it is diagnostic, not a success gate.",
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


__all__ = ["add_benchmark_args", "run_all_benchmarks"]
