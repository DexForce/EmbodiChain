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

"""Benchmark MoveJoints atomic action across joint-space target sequences.

Measures planning latency, memory usage, trajectory success, and final arm joint
error for named and explicit joint-position targets.
Run: python -m scripts.benchmark.atomic_action.move_joints_benchmark
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from scripts.benchmark.atomic_action.common import (
    CPU_MEMORY_BACKEND,
    add_common_benchmark_args,
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
class JointSequenceCase:
    """MoveJoints target sequence benchmark case."""

    name: str
    sequence: tuple[str, ...]


READY_QPOS = (0.35, -1.20, 1.30, -1.65, -1.57, 0.20)
HOME_QPOS = (0.0, -1.57, 1.57, -1.57, -1.57, 0.0)
EXTENDED_QPOS = (0.55, -1.05, 1.10, -1.45, -1.35, 0.35)
JOINT_TARGETS = {
    "ready": READY_QPOS,
    "home": HOME_QPOS,
    "extended": EXTENDED_QPOS,
}
SEQUENCE_CASES = {
    "ready_home": JointSequenceCase("ready_home", ("ready", "home")),
    "extended_home": JointSequenceCase("extended_home", ("extended", "home")),
    "ready_extended_home": JointSequenceCase(
        "ready_extended_home", ("ready", "extended", "home")
    ),
}
DEFAULT_SEQUENCE_CASES = tuple(SEQUENCE_CASES.keys())
MOVE_JOINTS_SAMPLE_INTERVAL = 80
SUCCESS_TOLERANCE_RAD = 1e-4


def add_benchmark_args(parser: argparse.ArgumentParser) -> None:
    """Add MoveJoints benchmark CLI arguments."""
    parser.add_argument(
        "--sequence_cases",
        nargs="+",
        choices=(*SEQUENCE_CASES.keys(), "all"),
        default=list(DEFAULT_SEQUENCE_CASES),
        help="Joint sequence cases to benchmark. Use 'all' for every case.",
    )
    add_common_benchmark_args(parser)


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark MoveJoints over named and explicit qpos targets."
    )
    add_benchmark_args(parser)
    return parser.parse_args()


def _select_cases(case_names: list[str]) -> list[JointSequenceCase]:
    """Resolve selected sequence case names."""
    if "all" in case_names:
        return list(SEQUENCE_CASES.values())
    return [SEQUENCE_CASES[name] for name in case_names]


def _qpos(values, device):
    """Create a torch qpos tensor."""
    torch = ensure_torch()
    return torch.tensor(values, dtype=torch.float32, device=device)


def _targets_for_sequence(sequence_case: JointSequenceCase, device):
    """Build typed MoveJoints targets for a sequence case."""
    from embodichain.lab.sim.atomic_actions import (
        JointPositionTarget,
        NamedJointPositionTarget,
    )

    targets = []
    for index, name in enumerate(sequence_case.sequence):
        if index == 0 and name == "ready":
            targets.append(("move_joints", NamedJointPositionTarget(name="ready")))
        else:
            targets.append(
                (
                    "move_joints",
                    JointPositionTarget(qpos=_qpos(JOINT_TARGETS[name], device)),
                )
            )
    return targets


def _run_case(
    sim,
    robot,
    atomic_engine,
    initial_qpos,
    case: JointSequenceCase,
    repeat: int,
    args: argparse.Namespace,
    recorded_count: int,
):
    """Run one MoveJoints case."""
    torch = ensure_torch()
    reset_robot(robot, initial_qpos)
    steps = _targets_for_sequence(case, sim.device)
    elapsed, mem_delta, peak_gpu, result = timed_call(lambda: atomic_engine.run(steps))
    is_success, traj, _ = result
    video_path = None
    if should_record_case(args, recorded_count, bool(is_success)):
        reset_robot(robot, initial_qpos)
        video_path = replay_trajectory_with_recording(
            sim=sim,
            robot=robot,
            traj=traj,
            args=args,
            video_path=build_video_output_path(
                args, "atomic_action_move_joints", f"{case.name}_r{repeat}"
            ),
        )
        reset_robot(robot, initial_qpos)

    final_error_rad = None
    if is_success and traj.shape[1] > 0:
        arm_joint_ids = robot.get_joint_ids(name="arm")
        final_qpos = traj[:, -1, arm_joint_ids][0]
        expected = _qpos(JOINT_TARGETS[case.sequence[-1]], sim.device)
        final_error_rad = float(torch.linalg.norm(final_qpos - expected))
    target_reached = bool(
        is_success
        and final_error_rad is not None
        and final_error_rad <= SUCCESS_TOLERANCE_RAD
    )
    return {
        "case_id": f"{case.name}:r{repeat}",
        "sequence_case": case.name,
        "repeat": repeat,
        "planning_success": bool(is_success),
        "target_reached": target_reached,
        "cost_time_ms": elapsed * 1000.0,
        "cpu_delta_mb": mem_delta["cpu_mb"],
        "gpu_delta_mb": mem_delta["gpu_mb"],
        "peak_gpu_mb": peak_gpu,
        "final_error_rad": final_error_rad,
        "trajectory_waypoints": int(traj.shape[1]) if traj.ndim >= 2 else 0,
        "failure_reason": "" if target_reached else "target_not_reached",
        "video_path": str(video_path) if video_path is not None else "",
    }


def _build_rows(results: list[dict[str, object]]):
    """Build report rows for MoveJoints."""
    perf_rows = []
    metric_rows = []
    for result in results:
        perf_rows.append(
            {
                "sample_size": 1,
                "impl": "move_joints",
                "case_id": result["case_id"],
                "sequence_case": result["sequence_case"],
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
                "impl": "move_joints",
                "case_id": result["case_id"],
                "sequence_case": result["sequence_case"],
                "success_rate": f"{float(result['target_reached']):.6f}",
                "planning_success_rate": f"{float(result['planning_success']):.6f}",
                "joint_err_rad": format_float(result["final_error_rad"]),
                "trajectory_waypoints": result["trajectory_waypoints"],
                "failure_reason": result["failure_reason"] or "N/A",
            }
        )
    return perf_rows, metric_rows


def run_all_benchmarks(args: argparse.Namespace | None = None) -> Path:
    """Run MoveJoints benchmark and write a markdown report."""
    args = _parse_args() if args is None else args
    if args.repeat < 1:
        raise ValueError("--repeat must be at least 1.")
    profile = resolve_profile(args)

    ensure_repo_root()
    ensure_torch()
    from embodichain.lab.sim.atomic_actions import (
        AtomicActionEngine,
        MoveJoints,
        MoveJointsCfg,
    )
    from embodichain.lab.sim.planners import MotionGenerator, MotionGenCfg
    from embodichain.lab.sim.planners import ToppraPlannerCfg
    from scripts.tutorials.atomic_action.move_joints import (
        create_robot,
        initialize_simulation,
    )

    cases = _select_cases(args.sequence_cases)
    repeat = 1 if profile == "smoke" else args.repeat
    if profile == "smoke":
        cases = [SEQUENCE_CASES["ready_home"]]

    print("=" * 60)
    print("MoveJoints Atomic Action Benchmark")
    print("=" * 60)
    print(
        "Coverage: "
        f"profile={profile}, {len(cases)} sequence case(s) x "
        f"{repeat} repeat(s)"
    )

    sim = initialize_simulation(args)
    robot = create_robot(sim)
    initial_qpos = robot.get_qpos().clone()
    motion_gen = MotionGenerator(
        cfg=MotionGenCfg(planner_cfg=ToppraPlannerCfg(robot_uid=robot.uid))
    )
    ready_qpos = _qpos(READY_QPOS, sim.device)
    atomic_engine = AtomicActionEngine(motion_generator=motion_gen)
    atomic_engine.register(
        MoveJoints(
            motion_gen,
            cfg=MoveJointsCfg(
                control_part="arm",
                sample_interval=MOVE_JOINTS_SAMPLE_INTERVAL,
                named_joint_positions={"ready": ready_qpos},
            ),
        )
    )

    results: list[dict[str, object]] = []
    video_paths: list[str] = []
    print("\n=== MoveJoints Sequence Sweep ===")
    for case in cases:
        for repeat_index in range(repeat):
            result = _run_case(
                sim,
                robot,
                atomic_engine,
                initial_qpos,
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
                f"success={result['target_reached']} "
                f"err={format_float(result['final_error_rad'], precision=6)}"
            )

    perf_rows, metric_rows = _build_rows(results)
    leaderboard_rows = build_single_action_leaderboard("move_joints", metric_rows)
    report_path = write_markdown_report(
        benchmark_name="atomic_action_move_joints",
        perf_rows=perf_rows,
        metric_rows=metric_rows,
        leaderboard_rows=leaderboard_rows,
        notes=[
            f"Profile: {profile}",
            f"CPU memory backend: {CPU_MEMORY_BACKEND}",
            f"Success tolerance: {SUCCESS_TOLERANCE_RAD} rad joint error.",
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
