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

"""Benchmark OpenDoor atomic action across hinge opening targets.

Measures planning latency, memory usage, plan success, and the physically
replayed hinge angle error for the tutorial microwave door.
Run: embodichain benchmark atomic-action --action open_door
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

from scripts.benchmark.atomic_action.common import (
    add_common_benchmark_args,
    add_grasp_benchmark_args,
    build_single_action_leaderboard,
    build_video_output_path,
    ensure_repo_root,
    ensure_torch,
    format_float,
    replay_trajectory_with_recording,
    reset_robot,
    resolve_profile,
    should_record_case,
    timed_call,
    write_markdown_report,
)


@dataclass(frozen=True)
class DoorCase:
    """OpenDoor hinge target benchmark case."""

    name: str
    open_angle_deg: float


DOOR_CASES = {
    "open_30": DoorCase("open_30", 30.0),
    "open_60": DoorCase("open_60", 60.0),
    "open_90": DoorCase("open_90", 90.0),
}
DEFAULT_DOOR_CASES = tuple(DOOR_CASES.keys())
TRAJECTORY_SAMPLE_COUNT = 120
HAND_INTERP_STEPS = 12
DOOR_WAYPOINT_COUNT = 50
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


def _hinge_position(microwave, hinge_joint_index: int):
    """Read the current hinge position, shape ``(num_envs, 1)``."""
    return microwave.get_qpos(target=False)[
        :, hinge_joint_index : hinge_joint_index + 1
    ]


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
):
    """Run one OpenDoor case: plan, replay physically, measure hinge error."""
    torch = ensure_torch()
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

    _reset_scene(robot, microwave, initial_qpos, initial_hinge_qpos)
    sim.update(step=4)

    semantics = create_door_handle_semantics(microwave)
    affordance = semantics.affordance
    lower_limit, upper_limit = affordance.joint_limits
    closed_position = lower_limit if affordance.opening_direction > 0 else upper_limit
    open_position = upper_limit if affordance.opening_direction > 0 else lower_limit
    open_angle = math.radians(case.open_angle_deg)
    open_fraction = (open_angle - closed_position) / (open_position - closed_position)
    handle_pose = microwave.get_link_pose(HANDLE_LINK_NAME, to_matrix=True)
    hinge_joint_index = microwave.joint_names.index(affordance.joint_name)
    hinge_position = _hinge_position(microwave, hinge_joint_index)

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
    # OpenDoor requires the observed hinge state in the planning context,
    # exactly as the tutorial provides it.
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
        control_dt=sim.sim_config.physics_dt,
    )
    elapsed, mem_delta, peak_gpu, result = timed_call(
        lambda: atomic_engine.compile((invocation,), context)
    )
    is_success = bool(result.plan_success.all().item())
    traj = result.trajectory.positions

    hinge_error_rad = None
    door_opened = False
    if is_success and traj is not None and traj.shape[1] > 0:
        for waypoint_index in range(traj.shape[1]):
            robot.set_qpos(traj[:, waypoint_index, :])
            sim.update(step=4)
        for _ in range(60):
            robot.set_qpos(traj[:, -1, :])
            sim.update(step=2)
        final_hinge = _hinge_position(microwave, hinge_joint_index)
        hinge_error_rad = float((final_hinge[0, 0] - open_angle).abs())
        door_opened = hinge_error_rad <= HINGE_SUCCESS_TOLERANCE_RAD

    video_path = None
    if should_record_case(args, recorded_count, bool(is_success)):
        _reset_scene(robot, microwave, initial_qpos, initial_hinge_qpos)
        video_path = replay_trajectory_with_recording(
            sim=sim,
            robot=robot,
            traj=traj,
            args=args,
            video_path=build_video_output_path(
                args, "atomic_action_open_door", f"{case.name}_r{repeat}"
            ),
        )

    return {
        "case_id": f"{case.name}:r{repeat}",
        "door_case": case.name,
        "repeat": repeat,
        "planning_success": bool(is_success),
        "door_opened": door_opened,
        "cost_time_ms": elapsed * 1000.0,
        "cpu_delta_mb": mem_delta["cpu_mb"],
        "gpu_delta_mb": mem_delta["gpu_mb"],
        "peak_gpu_mb": peak_gpu,
        "hinge_error_rad": hinge_error_rad,
        "trajectory_waypoints": int(traj.shape[1]) if traj is not None else 0,
        "failure_reason": "" if door_opened else "door_not_opened",
        "video_path": str(video_path) if video_path is not None else "",
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
                "cost_time_ms": f"{result['cost_time_ms']:.2f}",
                "cpu_delta_mb": f"{result['cpu_delta_mb']:.1f}",
                "gpu_delta_mb": f"{result['gpu_delta_mb']:.1f}",
                "peak_gpu_mb": f"{result['peak_gpu_mb']:.1f}",
            }
        )
        metric_rows.append(
            {
                "sample_size": 1,
                "impl": "open_door",
                "case_id": result["case_id"],
                "door_case": result["door_case"],
                "success_rate": f"{float(result['door_opened']):.6f}",
                "planning_success_rate": f"{float(result['planning_success']):.6f}",
                "hinge_error_rad": format_float(result["hinge_error_rad"]),
                "trajectory_waypoints": result["trajectory_waypoints"],
                "failure_reason": result["failure_reason"] or "N/A",
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
        cases = [DOOR_CASES["open_60"]]

    print("=" * 60)
    print("OpenDoor Atomic Action Benchmark")
    print("=" * 60)
    print(
        "Coverage: "
        f"profile={profile}, {len(cases)} door case(s) x {repeat} repeat(s)"
    )

    sim = initialize_benchmark_simulation(args)
    robot = add_ur5_gripper_robot(
        sim,
        init_qpos=[0.0, -1.57, 1.57, -3.14, -1.57, 0.0, 0.0, 0.0],
        tcp_z=0.15,
    )
    microwave = create_microwave(sim)
    sim.update(step=10)
    initial_qpos = robot.get_qpos().clone()
    initial_hinge_qpos = microwave.get_qpos(target=False).clone()
    hand_open, hand_close = get_hand_open_close_qpos(robot, close_qpos=0.04)

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
                n_sample=1000 if profile == "smoke" else args.n_sample,
                force_refresh=args.force_reannotate,
            )
        },
    )

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
            print(
                f"  {result['case_id']:<24} "
                f"time={result['cost_time_ms']:>10.2f} ms | "
                f"opened={result['door_opened']} "
                f"hinge_err={format_float(result['hinge_error_rad'], precision=4)}"
            )

    perf_rows, metric_rows = _build_rows(results)
    leaderboard_rows = build_single_action_leaderboard("open_door", metric_rows)
    report_path = write_markdown_report(
        benchmark_name="atomic_action_open_door",
        perf_rows=perf_rows,
        metric_rows=metric_rows,
        leaderboard_rows=leaderboard_rows,
        notes=[
            "Success requires the physically replayed hinge angle to reach the "
            f"target within {HINGE_SUCCESS_TOLERANCE_RAD} rad.",
            "Planning time includes handle grasp-pose generation and NMS.",
        ],
    )
    print(f"Markdown report saved: {report_path}")
    return report_path


def main() -> None:
    """CLI entry point."""
    run_all_benchmarks()


if __name__ == "__main__":
    from scripts.tutorials.atomic_action.tutorial_utils import run_tutorial

    run_tutorial(main)
