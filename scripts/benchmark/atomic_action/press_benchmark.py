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

"""Benchmark Press atomic action on the microwave start button.

Reports the ordered stages defined for Press in ``BENCHMARK_STANDARD.md``:
contacted, pressed. The button's prismatic joint must reach the configured
fraction of its stroke during the commanded press segment; the peak signed
displacement is used, so a button that rebounds afterwards still counts.
Run: embodichain benchmark atomic-action --action press
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from scripts.benchmark.atomic_action.common import (
    CPU_MEMORY_BACKEND,
    DEFAULT_REPLAY_STEPS_PER_WAYPOINT,
    ENGAGED_MIN_DISPLACEMENT,
    PHYSICAL_PRESS_MIN_STROKE_RATIO,
    SKILL_STAGES,
    StageLadder,
    add_common_benchmark_args,
    build_stage_leaderboard,
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

BUTTON_JOINT_NAME = "start_button_press"
PRESS_SAMPLE_INTERVAL = 140
HAND_INTERP_STEPS = 12
APPROACH_DISTANCE = 0.12
REPLAY_HOLD_STEPS = 60
# The microwave button joint is a 6 mm prismatic stroke. Requiring 80% of the
# asset's own usable stroke keeps the gate asset-relative instead of hard-coding
# a millimetre figure that would not transfer to another button.


@dataclass(frozen=True)
class PressCase:
    """Press commanded-travel benchmark case."""

    name: str
    press_distance_m: float


PRESS_CASES: dict[str, PressCase] = {
    # Sensitivity case, kept out of the default set. Measured result: commanding
    # 3 mm still drives the button to 5.66 mm of its 6 mm stroke, so on this
    # asset the commanded travel does not control the achieved depth. The
    # button joint has near-zero stiffness (1e-3), so it offers no resistance
    # and bottoms out once contact is made.
    "press_3mm": PressCase("press_3mm", 0.003),
    "press_10mm": PressCase("press_10mm", 0.010),
    "press_20mm": PressCase("press_20mm", 0.020),
    "press_30mm": PressCase("press_30mm", 0.030),
}
DEFAULT_PRESS_CASES = ("press_10mm", "press_20mm", "press_30mm")
SMOKE_PRESS_CASE = "press_30mm"


def add_benchmark_args(parser: argparse.ArgumentParser) -> None:
    """Add Press benchmark CLI arguments."""
    parser.add_argument(
        "--press_cases",
        nargs="+",
        choices=(*PRESS_CASES.keys(), "all"),
        default=list(DEFAULT_PRESS_CASES),
        help="Commanded press-travel cases to benchmark. Use 'all' for every case.",
    )
    add_common_benchmark_args(parser)


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark Press over commanded button travel distances."
    )
    add_benchmark_args(parser)
    return parser.parse_args()


def _select_cases(case_names: list[str]) -> list[PressCase]:
    """Resolve selected press case names."""
    if "all" in case_names:
        return list(PRESS_CASES.values())
    return [PRESS_CASES[name] for name in case_names]


def _reset_scene(robot, microwave, initial_qpos, initial_button_qpos) -> None:
    """Restore the robot and the microwave joints to their initial state."""
    reset_robot(robot, initial_qpos)
    microwave.set_qpos(initial_button_qpos, target=False)
    microwave.set_qpos(initial_button_qpos, target=True)
    microwave.clear_dynamics()


def _build_invocation(atomic_engine, microwave, case: PressCase, physics_dt: float):
    """Build the Press invocation and planning context for one case."""
    from embodichain.lab.sim.atomic_actions import MotionPolicy, PressGoal, PressOptions
    from scripts.tutorials.atomic_action.press import create_button_semantics

    semantics, target_pose = create_button_semantics(microwave)
    invocation = atomic_engine.make_invocation(
        "press",
        PressGoal(semantics, target_pose),
        control_parts={"primary": {"motion": "arm", "grasp": "hand"}},
        motion_policy=MotionPolicy(
            strategy="ik_interp", sample_count=PRESS_SAMPLE_INTERVAL
        ),
        skill_options=PressOptions(
            hand_interp_steps=HAND_INTERP_STEPS,
            approach_distance=APPROACH_DISTANCE,
            press_distance=case.press_distance_m,
        ),
    )
    context = atomic_engine.initial_context(control_dt=physics_dt)
    return invocation, context


def _press_measure_waypoint(compiled, traj) -> int:
    """Return the last waypoint of the commanded press segment."""
    try:
        return compiled.segment(0, "press").stop - 1
    except (KeyError, AttributeError, IndexError):
        return int(traj.shape[1]) - 1


def _run_case(
    sim,
    robot,
    microwave,
    atomic_engine,
    initial_qpos,
    initial_button_qpos,
    button_joint_index: int,
    button_stroke_m: float,
    case: PressCase,
    repeat: int,
    args: argparse.Namespace,
    recorded_count: int,
) -> dict[str, object]:
    """Run one Press case through its stages."""
    ladder = StageLadder(stages=SKILL_STAGES["press"])
    _reset_scene(robot, microwave, initial_qpos, initial_button_qpos)
    sim.update(step=4)

    invocation, context = _build_invocation(
        atomic_engine, microwave, case, sim.sim_config.physics_dt
    )
    required_travel_m = PHYSICAL_PRESS_MIN_STROKE_RATIO * button_stroke_m

    try:
        elapsed, mem_delta, peak_gpu, result = timed_call(
            lambda: atomic_engine.compile((invocation,), context)
        )
    except Exception as exc:
        print(f"    planner exception: {type(exc).__name__}: {exc}")
        ladder.fail("contacted", "planner_exception")
        return _case_result(
            case, repeat, ladder, 0.0, None, None, required_travel_m, ""
        )

    plan_success = bool(result.plan_success.all().item())
    traj = result.trajectory.positions
    if not plan_success:
        ladder.fail("contacted", "planner_reported_failure")
        return _case_result(
            case,
            repeat,
            ladder,
            elapsed,
            mem_delta,
            None,
            required_travel_m,
            "",
            peak_gpu,
        )

    motion_valid, motion_detail = check_motion_valid(traj, robot)
    ladder.motion_valid = motion_valid
    ladder.motion_detail = motion_detail
    if motion_detail == "non_finite_trajectory":
        ladder.fail("contacted", "planner_reported_failure")
        return _case_result(
            case,
            repeat,
            ladder,
            elapsed,
            mem_delta,
            None,
            required_travel_m,
            "",
            peak_gpu,
            traj,
        )

    try:
        trace = replay_and_track_joint(
            sim=sim,
            robot=robot,
            traj=traj,
            articulation=microwave,
            joint_index=button_joint_index,
            measure_waypoint=_press_measure_waypoint(result, traj),
            steps_per_waypoint=DEFAULT_REPLAY_STEPS_PER_WAYPOINT,
            hold_steps=REPLAY_HOLD_STEPS,
        )
    except Exception as exc:
        print(f"    replay exception: {type(exc).__name__}: {exc}")
        trace = None

    if trace is None:
        ladder.fail("contacted", "invalid_case")
    else:
        ladder.max_tracking_error_rad = trace.max_tracking_error_rad
        ladder.record(
            "contacted",
            abs(trace.peak_signed_displacement) > ENGAGED_MIN_DISPLACEMENT,
            "task_goal_miss",
        )
        # The button travels along +qpos; use the peak signed displacement so a
        # button that reaches its stroke and rebounds still counts.
        ladder.record(
            "pressed",
            abs(trace.peak_signed_displacement) >= required_travel_m,
            "task_goal_miss",
        )

    video_path = ""
    if should_record_case(args, recorded_count, ladder.success):
        _reset_scene(robot, microwave, initial_qpos, initial_button_qpos)
        recorded = replay_trajectory_with_recording(
            sim=sim,
            robot=robot,
            traj=traj,
            args=args,
            video_path=build_video_output_path(
                args, "atomic_action_press", f"{case.name}_r{repeat}"
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
        required_travel_m,
        video_path,
        peak_gpu,
        traj,
    )


def _case_result(
    case: PressCase,
    repeat: int,
    ladder: StageLadder,
    elapsed: float,
    mem_delta: dict[str, float] | None,
    trace,
    required_travel_m: float,
    video_path: str,
    peak_gpu: float = 0.0,
    traj=None,
) -> dict[str, object]:
    """Assemble one Press case result row."""
    mem_delta = mem_delta or {"cpu_mb": 0.0, "gpu_mb": 0.0}
    return {
        "case_id": f"{case.name}:r{repeat}",
        "press_case": case.name,
        "repeat": repeat,
        "ladder": ladder,
        "commanded_press_m": case.press_distance_m,
        "required_travel_m": required_travel_m,
        "cost_time_ms": elapsed * 1000.0,
        "cpu_delta_mb": mem_delta["cpu_mb"],
        "gpu_delta_mb": mem_delta["gpu_mb"],
        "peak_gpu_mb": peak_gpu,
        "max_tracking_error_rad": trace.max_tracking_error_rad if trace else None,
        "button_initial_m": trace.initial_position if trace else None,
        "button_measured_disp_m": trace.measured_displacement if trace else None,
        "button_peak_signed_m": trace.peak_signed_displacement if trace else None,
        "button_settled_disp_m": trace.settled_displacement if trace else None,
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
                "impl": "press",
                "case_id": result["case_id"],
                "press_case": result["press_case"],
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
                "impl": "press",
                "case_id": result["case_id"],
                "press_case": result["press_case"],
                **ladder.as_row_fields(),
                "commanded_press_m": format_float(result["commanded_press_m"], 4),
                "required_travel_m": format_float(result["required_travel_m"], 4),
                "max_tracking_error_rad": format_float(
                    result["max_tracking_error_rad"], 4
                ),
                "button_initial_m": format_float(result["button_initial_m"], 5),
                "button_measured_disp_m": format_float(
                    result["button_measured_disp_m"], 5
                ),
                "button_peak_signed_m": format_float(result["button_peak_signed_m"], 5),
                "button_settled_disp_m": format_float(
                    result["button_settled_disp_m"], 5
                ),
            }
        )
    return perf_rows, metric_rows


def run_all_benchmarks(args: argparse.Namespace | None = None) -> Path:
    """Run Press benchmark and write a markdown report."""
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
    from scripts.tutorials.atomic_action.press import create_microwave
    from scripts.tutorials.atomic_action.tutorial_utils import (
        add_ur5_gripper_robot,
        create_toppra_motion_generator,
        get_hand_open_close_qpos,
        initialize_benchmark_simulation,
    )

    cases = _select_cases(args.press_cases)
    repeat = 1 if profile == "smoke" else args.repeat
    if profile == "smoke":
        cases = [PRESS_CASES[SMOKE_PRESS_CASE]]

    print("=" * 60)
    print("Press Atomic Action Benchmark")
    print("=" * 60)
    print(
        f"Coverage: profile={profile}, {len(cases)} press case(s) x {repeat} repeat(s)"
    )

    sim = initialize_benchmark_simulation(args)
    robot = add_ur5_gripper_robot(
        sim, init_qpos=[0.0, -1.57, 1.57, -3.14, -1.57, 0.0, 0.0, 0.0]
    )
    microwave = create_microwave(sim)
    sim.prepare()
    sim.update(step=10)
    initial_qpos = robot.get_qpos().clone()
    initial_button_qpos = microwave.get_qpos(target=False).clone()
    hand_open, hand_close = get_hand_open_close_qpos(robot, close_qpos=0.040)

    button_joint_index = microwave.joint_names.index(BUTTON_JOINT_NAME)
    button_limits = microwave.get_qpos_limits()[0, button_joint_index]
    button_stroke_m = float(button_limits[1] - button_limits[0])
    print(
        f"Button joint {BUTTON_JOINT_NAME!r}: stroke={button_stroke_m * 1000.0:.2f} mm, "
        f"required travel={PHYSICAL_PRESS_MIN_STROKE_RATIO * button_stroke_m * 1000.0:.2f} mm"
    )

    atomic_engine = AtomicActionEngine(
        motion_generator=create_toppra_motion_generator(robot),
        control_profiles={
            "hand": ControlPartCommandProfile.joint_positions(
                open=hand_open,
                grasp=hand_close,
            )
        },
    )

    warmup_invocation, warmup_context = _build_invocation(
        atomic_engine, microwave, cases[0], sim.sim_config.physics_dt
    )
    print("Warm-up: running one discarded compile to exclude first-call cost...")
    warmup_planning(lambda: atomic_engine.compile((warmup_invocation,), warmup_context))
    _reset_scene(robot, microwave, initial_qpos, initial_button_qpos)

    results: list[dict[str, object]] = []
    video_paths: list[str] = []
    print("\n=== Press Commanded-Travel Sweep ===")
    for case in cases:
        for repeat_index in range(repeat):
            result = _run_case(
                sim,
                robot,
                microwave,
                atomic_engine,
                initial_qpos,
                initial_button_qpos,
                button_joint_index,
                button_stroke_m,
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
                f"track={format_float(ladder.max_tracking_error_rad, 4)} "
                f"peak={format_float(result['button_peak_signed_m'], 5)} "
                f"[{ladder.failure_reason or 'ok'}]"
            )

    perf_rows, metric_rows = _build_rows(results)
    leaderboard_rows = build_stage_leaderboard("press", results)
    report_path = write_markdown_report(
        benchmark_name="atomic_action_press",
        perf_rows=perf_rows,
        metric_rows=metric_rows,
        leaderboard_rows=leaderboard_rows,
        notes=[
            f"Profile: {profile}",
            f"CPU memory backend: {CPU_MEMORY_BACKEND}",
            f"Button joint: {BUTTON_JOINT_NAME}, usable stroke "
            f"{button_stroke_m * 1000.0:.2f} mm.",
            "Stages: contacted (the button moved at all), pressed (the peak "
            "signed button displacement during replay reached "
            f"{PHYSICAL_PRESS_MIN_STROKE_RATIO:.0%} of that stroke, "
            f"{PHYSICAL_PRESS_MIN_STROKE_RATIO * button_stroke_m * 1000.0:.2f} "
            "mm). Rebound after the press segment is allowed.",
            "This criterion only shows that the button moved: commanding 3 mm "
            "still drives it to 5.66 mm of its 6 mm stroke, so the ratio does "
            "not discriminate commanded depth on this asset.",
            "motion_valid and max_tracking_error_rad are diagnostics; neither "
            "fails a stage.",
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
