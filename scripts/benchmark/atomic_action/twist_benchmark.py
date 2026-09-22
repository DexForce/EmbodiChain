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

Reports the ordered stages defined for Twist in ``BENCHMARK_STANDARD.md``:
grasped, twisted, released. The twisted stage scores the executed end-effector
rotation about the knob axis; the knob joint's own rotation is a diagnostic,
because this knob offers almost no resistance and over-rotates.
Run: embodichain benchmark atomic-action --action twist
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

from scripts.benchmark.atomic_action.common import (
    CPU_MEMORY_BACKEND,
    ENGAGED_MIN_DISPLACEMENT,
    SKILL_STAGES,
    StageLadder,
    TASK_ROTATION_TOLERANCE_RAD,
    add_common_benchmark_args,
    build_stage_leaderboard,
    build_video_output_path,
    ensure_repo_root,
    ensure_torch,
    format_float,
    hand_is_released,
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
# Angle used to read the knob joint's axis off the asset itself.
AXIS_PROBE_ANGLE_RAD = 0.05


def _knob_axis_world(sim, microwave, joint_index: int):
    """Return the knob joint's rotation axis in world coordinates.

    The axis is measured from the asset rather than assumed: the knob joint is
    perturbed by a small angle, the resulting change in the knob link's
    orientation is a rotation about that joint's axis, and its direction is
    read back out of that rotation. The joint is restored afterwards.

    Args:
        sim: Simulation manager, stepped so the probe reaches the link pose.
        microwave: Articulation owning the knob joint.
        joint_index: Index of the knob joint.

    Returns:
        Unit axis in world coordinates with shape ``(3,)``.
    """
    from scripts.tutorials.atomic_action.twist import KNOB_LINK_NAME

    torch = ensure_torch()
    initial_qpos = microwave.get_qpos(target=False).clone()
    before = microwave.get_link_pose(KNOB_LINK_NAME, to_matrix=True)[0, :3, :3].clone()
    probed = initial_qpos.clone()
    probed[:, joint_index] = probed[:, joint_index] + AXIS_PROBE_ANGLE_RAD
    microwave.set_qpos(probed, target=False)
    microwave.set_qpos(probed, target=True)
    sim.update(step=1)
    after = microwave.get_link_pose(KNOB_LINK_NAME, to_matrix=True)[0, :3, :3].clone()
    microwave.set_qpos(initial_qpos, target=False)
    microwave.set_qpos(initial_qpos, target=True)
    microwave.clear_dynamics()
    sim.update(step=1)

    relative = before.transpose(0, 1) @ after
    axis = torch.stack(
        [
            relative[2, 1] - relative[1, 2],
            relative[0, 2] - relative[2, 0],
            relative[1, 0] - relative[0, 1],
        ]
    )
    norm = torch.linalg.vector_norm(axis)
    if float(norm) <= 1.0e-6:
        raise RuntimeError("The knob joint did not rotate its link; axis unknown.")
    return before @ (axis / norm)


def _make_twist_rotation_reader(robot, axis_world, reference_waypoint: int):
    """Return a reader for the end-effector rotation about the knob axis.

    The standard scores Twist on what the hand did, not on where the knob
    ended: this knob offers almost no resistance, so the gripper wedges against
    it and drags it past the command. The reference orientation is captured at
    the start of the twist segment, after the approach has already reoriented
    the hand.

    Args:
        robot: Robot whose arm pose is read.
        axis_world: Unit knob axis in world coordinates.
        reference_waypoint: Waypoint index at which the reference orientation
            is captured.

    Returns:
        Callable returning the signed rotation in radians, zero until the
        reference is captured.
    """
    torch = ensure_torch()
    state: dict[str, object] = {}

    def perpendicular_component(rotation):
        for column in (0, 1):
            direction = rotation[:, column]
            projected = direction - axis_world * torch.dot(direction, axis_world)
            norm = torch.linalg.vector_norm(projected)
            if float(norm) > 1.0e-6:
                return projected / norm
        raise RuntimeError("The end-effector frame is degenerate about the knob axis.")

    def read(waypoint_index: int) -> float:
        pose = robot.compute_fk(robot.get_qpos(name="arm"), name="arm", to_matrix=True)
        current = perpendicular_component(pose[0, :3, :3])
        if waypoint_index <= reference_waypoint:
            state["reference"] = current
            return 0.0
        reference = state.get("reference")
        if reference is None:
            return 0.0
        sine = torch.dot(axis_world, torch.linalg.cross(reference, current))
        cosine = torch.dot(reference, current)
        return float(torch.atan2(sine, cosine))

    return read


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
        ladder: StageLadder = result["ladder"]  # type: ignore[assignment]
        metric_rows.append(
            {
                "sample_size": 1,
                "impl": "twist",
                "case_id": result["case_id"],
                "twist_case": result["twist_case"],
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
                "ee_rotation_rad": format_float(result["ee_rotation_rad"], 4),
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
    knob_axis_world = _knob_axis_world(sim, microwave, knob_joint_index)
    print(
        f"Knob joint {KNOB_JOINT_NAME!r} (link cap_1), "
        f"tolerance={TASK_ROTATION_TOLERANCE_RAD:.3f} rad"
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

            def build_readers(compiled):
                try:
                    reference = compiled.segment(0, "twist").start - 1
                except (KeyError, AttributeError, IndexError):
                    reference = -1
                return {
                    "ee_rotation": _make_twist_rotation_reader(
                        robot, knob_axis_world, reference
                    )
                }

            def evaluate(ladder, traces, case=case):
                knob = traces["joint"]
                rotation = traces["ee_rotation"]
                # A knob cannot turn unless the hand is on it, so the knob
                # leaving its initial angle is the evidence that the grasp
                # took. Where the knob ended is then only a diagnostic.
                ladder.record(
                    "grasped",
                    abs(knob.measured_displacement) > ENGAGED_MIN_DISPLACEMENT,
                    "object_not_grasped",
                )
                error = abs(abs(rotation.measured_position) - abs(case.twist_angle_rad))
                ladder.record(
                    "twisted", error <= TASK_ROTATION_TOLERANCE_RAD, "task_goal_miss"
                )
                ladder.record(
                    "released",
                    hand_is_released(robot, "hand", hand_open, hand_close),
                    "release_failure",
                )

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
                stages=SKILL_STAGES["twist"],
                evaluate=evaluate,
                build_extra_readers=build_readers,
                hold_steps=REPLAY_HOLD_STEPS,
            )
            trace = outcome.trace
            rotation = outcome.channels.get("ee_rotation")
            angle_error = None
            if rotation is not None:
                angle_error = abs(
                    abs(rotation.measured_position) - abs(case.twist_angle_rad)
                )

            video_path = ""
            if should_record_case(args, len(video_paths), outcome.ladder.success):
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
                "ee_rotation_rad": (
                    rotation.measured_position if rotation is not None else None
                ),
                "angle_error_rad": angle_error,
                "trajectory_waypoints": outcome.trajectory_waypoints,
                "success": outcome.ladder.success,
                "video_path": video_path,
            }
            results.append(result)
            ladder = outcome.ladder
            stages = " ".join(
                f"{stage}={'ok' if ladder.passed[stage] else 'FAIL'}"
                for stage in ladder.stages
                if stage in ladder.passed
            )
            print(
                f"  {result['case_id']:<22} "
                f"time={result['cost_time_ms']:>9.2f} ms | "
                f"{stages} "
                f"rot={format_float(result['knob_measured_disp_rad'], 4)} "
                f"ee_rot={format_float(result['ee_rotation_rad'], 4)} "
                f"err={format_float(angle_error, 4)} "
                f"[{ladder.failure_reason or 'ok'}]"
            )

    perf_rows, metric_rows = _build_rows(results)
    leaderboard_rows = build_stage_leaderboard("twist", results)
    report_path = write_markdown_report(
        benchmark_name="atomic_action_twist",
        perf_rows=perf_rows,
        metric_rows=metric_rows,
        leaderboard_rows=leaderboard_rows,
        notes=[
            f"Profile: {profile}",
            f"CPU memory backend: {CPU_MEMORY_BACKEND}",
            f"Knob joint: {KNOB_JOINT_NAME} (target link cap_1).",
            "Stages: grasped (the knob left its initial angle), twisted (the "
            "executed end-effector rotation about the knob axis matched the "
            f"command within {TASK_ROTATION_TOLERANCE_RAD:.3f} rad at the end "
            "of the 'twist' segment), released (the hand ended nearer its open "
            "command than its closed one).",
            "The knob joint's own rotation is a diagnostic. It offers almost "
            "no resistance, so the gripper wedges against it and drags it past "
            "the command; knob_settled_disp_rad is the resting value after "
            "release and retract.",
            "The knob axis is measured from the asset by probing the joint, "
            "not assumed from the affordance default.",
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
