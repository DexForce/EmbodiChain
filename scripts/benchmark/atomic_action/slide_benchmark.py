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

"""Benchmark Slide atomic action on the drawer prismatic joint.

Reports the four-level success ladder (planning_success, motion_valid,
execution_success, task_success). Task success requires the drawer joint to
reach the commanded translation distance during the commanded pull segment.
Run: embodichain benchmark atomic-action --action slide
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from scripts.benchmark.atomic_action.common import (
    CPU_MEMORY_BACKEND,
    SuccessLadder,
    add_common_benchmark_args,
    add_grasp_benchmark_args,
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

DRAWER_JOINT_NAME = "cabinet_to_drawer"
TRAJECTORY_SAMPLE_COUNT = 140
HAND_INTERP_STEPS = 12
APPROACH_DISTANCE = 0.10
REPLAY_HOLD_STEPS = 60
# Commanded-versus-achieved drawer travel tolerance. Calibrated from the
# measured pull error on this asset; see BENCHMARK_STANDARD.md.
SLIDE_TOLERANCE_M = 0.03


@dataclass(frozen=True)
class SlideCase:
    """Slide commanded-travel benchmark case."""

    name: str
    translation_distance_m: float


SLIDE_CASES: dict[str, SlideCase] = {
    "pull_100mm": SlideCase("pull_100mm", 0.10),
    "pull_180mm": SlideCase("pull_180mm", 0.18),
    "pull_250mm": SlideCase("pull_250mm", 0.25),
}
DEFAULT_SLIDE_CASES = tuple(SLIDE_CASES.keys())
SMOKE_SLIDE_CASE = "pull_180mm"


def add_benchmark_args(parser: argparse.ArgumentParser) -> None:
    """Add Slide benchmark CLI arguments."""
    parser.add_argument(
        "--slide_cases",
        nargs="+",
        choices=(*SLIDE_CASES.keys(), "all"),
        default=list(DEFAULT_SLIDE_CASES),
        help="Commanded drawer-travel cases to benchmark. Use 'all' for every case.",
    )
    add_grasp_benchmark_args(parser)
    add_common_benchmark_args(parser)


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark Slide over commanded drawer travel distances."
    )
    add_benchmark_args(parser)
    return parser.parse_args()


def _select_cases(case_names: list[str]) -> list[SlideCase]:
    """Resolve selected slide case names."""
    if "all" in case_names:
        return list(SLIDE_CASES.values())
    return [SLIDE_CASES[name] for name in case_names]


def _reset_scene(robot, drawer, initial_qpos, initial_drawer_qpos) -> None:
    """Restore the robot and the drawer to their initial closed state."""
    reset_robot(robot, initial_qpos)
    drawer.set_qpos(initial_drawer_qpos, target=False)
    drawer.set_qpos(initial_drawer_qpos, target=True)
    drawer.clear_dynamics()


def _build_invocation(atomic_engine, drawer, semantics, case: SlideCase, physics_dt):
    """Build the Slide pull invocation and planning context for one case."""
    from embodichain.lab.sim.atomic_actions import MotionPolicy, SlideGoal, SlideOptions
    from scripts.tutorials.atomic_action.slide import HANDLE_LINK_NAME

    handle_pose = drawer.get_link_pose(HANDLE_LINK_NAME, to_matrix=True)
    invocation = atomic_engine.make_invocation(
        "slide",
        SlideGoal(semantics, handle_pose),
        control_parts={"primary": {"motion": "arm", "grasp": "hand"}},
        motion_policy=MotionPolicy(
            strategy="ik_interp", sample_count=TRAJECTORY_SAMPLE_COUNT
        ),
        skill_options=SlideOptions(
            direction="pull",
            hand_interp_steps=HAND_INTERP_STEPS,
            approach_distance=APPROACH_DISTANCE,
            translation_distance=case.translation_distance_m,
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
                "impl": "slide",
                "case_id": result["case_id"],
                "slide_case": result["slide_case"],
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
                "impl": "slide",
                "case_id": result["case_id"],
                "slide_case": result["slide_case"],
                "success_rate": f"{float(ladder.task_success):.6f}",
                **ladder.as_row_fields(),
                "commanded_travel_m": format_float(result["commanded_travel_m"], 4),
                "max_tracking_error_rad": format_float(
                    result["max_tracking_error_rad"], 4
                ),
                "drawer_initial_m": format_float(result["drawer_initial_m"], 4),
                "drawer_measured_disp_m": format_float(
                    result["drawer_measured_disp_m"], 4
                ),
                "drawer_peak_signed_m": format_float(result["drawer_peak_signed_m"], 4),
                "drawer_settled_disp_m": format_float(
                    result["drawer_settled_disp_m"], 4
                ),
                "travel_error_m": format_float(result["travel_error_m"], 4),
            }
        )
    return perf_rows, metric_rows


def run_all_benchmarks(args: argparse.Namespace | None = None) -> Path:
    """Run Slide benchmark and write a markdown report."""
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
    from scripts.tutorials.atomic_action.slide import (
        create_drawer,
        create_drawer_semantics,
    )
    from scripts.tutorials.atomic_action.tutorial_utils import (
        add_ur5_gripper_robot,
        create_parallel_jaw_grasp_pose_generator,
        create_toppra_motion_generator,
        get_hand_open_close_qpos,
        initialize_benchmark_simulation,
    )

    cases = _select_cases(args.slide_cases)
    repeat = 1 if profile == "smoke" else args.repeat
    if profile == "smoke":
        cases = [SLIDE_CASES[SMOKE_SLIDE_CASE]]

    print("=" * 60)
    print("Slide Atomic Action Benchmark")
    print("=" * 60)
    print(
        f"Coverage: profile={profile}, {len(cases)} slide case(s) x {repeat} repeat(s)"
    )

    sim = initialize_benchmark_simulation(args)
    robot = add_ur5_gripper_robot(
        sim, init_qpos=[0.0, -1.57, 1.57, -3.14, -1.57, 0.0, 0.0, 0.0], tcp_z=0.15
    )
    drawer = create_drawer(sim)
    sim.prepare()
    sim.update(step=10)
    initial_qpos = robot.get_qpos().clone()
    initial_drawer_qpos = drawer.get_qpos(target=False).clone()
    hand_open, hand_close = get_hand_open_close_qpos(robot)

    drawer_joint_index = drawer.joint_names.index(DRAWER_JOINT_NAME)
    drawer_limits = drawer.get_qpos_limits()[0, drawer_joint_index]
    drawer_stroke_m = float(drawer_limits[1] - drawer_limits[0])
    print(
        f"Drawer joint {DRAWER_JOINT_NAME!r}: stroke={drawer_stroke_m:.3f} m, "
        f"tolerance={SLIDE_TOLERANCE_M:.3f} m"
    )

    n_sample = 1000 if profile == "smoke" else args.n_sample
    atomic_engine = AtomicActionEngine(
        motion_generator=create_toppra_motion_generator(robot),
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
    semantics = create_drawer_semantics(drawer)

    print("Warm-up: running one discarded compile to exclude first-call cost...")
    warmup_planning(
        lambda: atomic_engine.compile(
            (
                _build_invocation(
                    atomic_engine,
                    drawer,
                    semantics,
                    cases[0],
                    sim.sim_config.physics_dt,
                )[0],
            ),
            atomic_engine.initial_context(control_dt=sim.sim_config.physics_dt),
        )
    )
    _reset_scene(robot, drawer, initial_qpos, initial_drawer_qpos)

    results: list[dict[str, object]] = []
    video_paths: list[str] = []
    print("\n=== Slide Commanded-Travel Sweep ===")
    for case in cases:
        for repeat_index in range(repeat):
            _reset_scene(robot, drawer, initial_qpos, initial_drawer_qpos)
            sim.update(step=4)

            def evaluate(trace, case=case):
                error = abs(
                    abs(trace.measured_displacement) - case.translation_distance_m
                )
                return (error <= SLIDE_TOLERANCE_M, "task_goal_miss")

            outcome = run_articulated_contact_case(
                sim=sim,
                robot=robot,
                articulation=drawer,
                atomic_engine=atomic_engine,
                build_invocation=lambda case=case: _build_invocation(
                    atomic_engine, drawer, semantics, case, sim.sim_config.physics_dt
                ),
                joint_index=drawer_joint_index,
                actuation_segment="pull",
                evaluate=evaluate,
                hold_steps=REPLAY_HOLD_STEPS,
            )
            trace = outcome.trace
            travel_error = None
            if trace is not None:
                travel_error = abs(
                    abs(trace.measured_displacement) - case.translation_distance_m
                )

            video_path = ""
            if should_record_case(args, len(video_paths), outcome.ladder.task_success):
                _reset_scene(robot, drawer, initial_qpos, initial_drawer_qpos)
                recorded = replay_trajectory_with_recording(
                    sim=sim,
                    robot=robot,
                    traj=outcome.trajectory,
                    args=args,
                    video_path=build_video_output_path(
                        args, "atomic_action_slide", f"{case.name}_r{repeat_index}"
                    ),
                )
                video_path = str(recorded) if recorded is not None else ""
                video_paths.append(video_path)

            result = {
                "case_id": f"{case.name}:r{repeat_index}",
                "slide_case": case.name,
                "repeat": repeat_index,
                "ladder": outcome.ladder,
                "commanded_travel_m": case.translation_distance_m,
                "cost_time_ms": outcome.elapsed_s * 1000.0,
                "cpu_delta_mb": outcome.cpu_delta_mb,
                "gpu_delta_mb": outcome.gpu_delta_mb,
                "peak_gpu_mb": outcome.peak_gpu_mb,
                "max_tracking_error_rad": (
                    trace.max_tracking_error_rad if trace else None
                ),
                "drawer_initial_m": trace.initial_position if trace else None,
                "drawer_measured_disp_m": (
                    trace.measured_displacement if trace else None
                ),
                "drawer_peak_signed_m": (
                    trace.peak_signed_displacement if trace else None
                ),
                "drawer_settled_disp_m": trace.settled_displacement if trace else None,
                "travel_error_m": travel_error,
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
                f"travel={format_float(result['drawer_measured_disp_m'], 4)} "
                f"err={format_float(travel_error, 4)} "
                f"[{ladder.failure_reason or 'ok'}]"
            )

    perf_rows, metric_rows = _build_rows(results)
    leaderboard_rows = build_ladder_leaderboard("slide", results)
    report_path = write_markdown_report(
        benchmark_name="atomic_action_slide",
        perf_rows=perf_rows,
        metric_rows=metric_rows,
        leaderboard_rows=leaderboard_rows,
        notes=[
            f"Profile: {profile}",
            f"CPU memory backend: {CPU_MEMORY_BACKEND}",
            f"Grasp samples: {n_sample}",
            f"Drawer joint: {DRAWER_JOINT_NAME}, usable stroke "
            f"{drawer_stroke_m:.3f} m.",
            "task_success requires the replayed drawer travel at the end of the "
            "commanded 'pull' segment to match the commanded translation within "
            f"{SLIDE_TOLERANCE_M:.3f} m.",
            "drawer_settled_disp_m is the resting travel after release and "
            "retract. The drawer joint is undriven (drive_type='none'), so that "
            "value is diagnostic, not a success gate.",
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
