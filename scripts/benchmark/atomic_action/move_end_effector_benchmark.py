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

"""Benchmark MoveEndEffector atomic action across target poses.

Measures planning latency, memory usage, trajectory success, and final TCP
translation error for several reachable pose targets.
Run: python -m scripts.benchmark.atomic_action.move_end_effector_benchmark
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
class PoseCase:
    """End-effector target pose benchmark case."""

    name: str
    xyz: tuple[float, float, float]


POSE_CASES = {
    "front_left": PoseCase("front_left", (0.30, -0.20, 0.36)),
    "front_right": PoseCase("front_right", (0.30, 0.20, 0.36)),
    "near_center": PoseCase("near_center", (0.18, 0.00, 0.42)),
    "far_center": PoseCase("far_center", (0.42, 0.00, 0.34)),
}
DEFAULT_POSE_CASES = tuple(POSE_CASES.keys())
MOVE_SAMPLE_INTERVAL = 80
SUCCESS_TOLERANCE_M = 0.01


def add_benchmark_args(parser: argparse.ArgumentParser) -> None:
    """Add MoveEndEffector benchmark CLI arguments."""
    parser.add_argument(
        "--pose_cases",
        nargs="+",
        choices=(*POSE_CASES.keys(), "all"),
        default=list(DEFAULT_POSE_CASES),
        help="Target pose cases to benchmark. Use 'all' for every case.",
    )
    add_common_benchmark_args(parser)


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark MoveEndEffector over reachable target poses."
    )
    add_benchmark_args(parser)
    return parser.parse_args()


def _select_pose_cases(case_names: list[str]) -> list[PoseCase]:
    """Resolve selected pose case names."""
    if "all" in case_names:
        return list(POSE_CASES.values())
    return [POSE_CASES[name] for name in case_names]


def _make_pose(device, xyz: tuple[float, float, float]):
    """Create a top-down target pose."""
    torch = ensure_torch()
    pose = torch.eye(4, dtype=torch.float32, device=device)
    pose[:3, :3] = torch.tensor(
        [
            [-0.0539, -0.9985, -0.0022],
            [-0.9977, 0.0540, -0.0401],
            [0.0401, 0.0000, -0.9992],
        ],
        dtype=torch.float32,
        device=device,
    )
    pose[:3, 3] = torch.tensor(xyz, dtype=torch.float32, device=device)
    return pose


def _run_case(
    sim,
    robot,
    atomic_engine,
    initial_qpos,
    pose_case: PoseCase,
    repeat: int,
    args: argparse.Namespace,
    recorded_count: int,
):
    """Run one MoveEndEffector case."""
    torch = ensure_torch()
    from embodichain.lab.sim.atomic_actions import EndEffectorPoseTarget

    reset_robot(robot, initial_qpos)
    target_pose = _make_pose(sim.device, pose_case.xyz)

    elapsed, mem_delta, peak_gpu, result = timed_call(
        lambda: atomic_engine.run(
            steps=[("move_end_effector", EndEffectorPoseTarget(xpos=target_pose))]
        )
    )
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
                args, "atomic_action_move_end_effector", f"{pose_case.name}_r{repeat}"
            ),
        )
        reset_robot(robot, initial_qpos)

    final_error_m = None
    if is_success and traj.shape[1] > 0:
        arm_joint_ids = robot.get_joint_ids(name="arm")
        final_arm_qpos = traj[:, -1, arm_joint_ids]
        final_pose = robot.compute_fk(qpos=final_arm_qpos, name="arm", to_matrix=True)[
            0
        ]
        final_error_m = float(torch.linalg.norm(final_pose[:3, 3] - target_pose[:3, 3]))
    target_reached = bool(
        is_success
        and final_error_m is not None
        and final_error_m <= SUCCESS_TOLERANCE_M
    )
    return {
        "case_id": f"{pose_case.name}:r{repeat}",
        "target_case": pose_case.name,
        "repeat": repeat,
        "planning_success": bool(is_success),
        "target_reached": target_reached,
        "cost_time_ms": elapsed * 1000.0,
        "cpu_delta_mb": mem_delta["cpu_mb"],
        "gpu_delta_mb": mem_delta["gpu_mb"],
        "peak_gpu_mb": peak_gpu,
        "final_error_m": final_error_m,
        "trajectory_waypoints": int(traj.shape[1]) if traj.ndim >= 2 else 0,
        "failure_reason": "" if target_reached else "target_not_reached",
        "video_path": str(video_path) if video_path is not None else "",
    }


def _build_rows(results: list[dict[str, object]]):
    """Build report rows for MoveEndEffector."""
    perf_rows = []
    metric_rows = []
    for result in results:
        perf_rows.append(
            {
                "sample_size": 1,
                "impl": "move_end_effector",
                "case_id": result["case_id"],
                "target_case": result["target_case"],
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
                "impl": "move_end_effector",
                "case_id": result["case_id"],
                "target_case": result["target_case"],
                "success_rate": f"{float(result['target_reached']):.6f}",
                "planning_success_rate": f"{float(result['planning_success']):.6f}",
                "translation_err_m": format_float(result["final_error_m"]),
                "trajectory_waypoints": result["trajectory_waypoints"],
                "failure_reason": result["failure_reason"] or "N/A",
            }
        )
    return perf_rows, metric_rows


def run_all_benchmarks(args: argparse.Namespace | None = None) -> Path:
    """Run MoveEndEffector benchmark and write a markdown report."""
    args = _parse_args() if args is None else args
    if args.repeat < 1:
        raise ValueError("--repeat must be at least 1.")
    profile = resolve_profile(args)

    ensure_repo_root()
    ensure_torch()
    from embodichain.lab.sim.atomic_actions import (
        AtomicActionEngine,
        MoveEndEffector,
        MoveEndEffectorCfg,
    )
    from embodichain.lab.sim.planners import MotionGenerator, MotionGenCfg
    from embodichain.lab.sim.planners import ToppraPlannerCfg
    from scripts.tutorials.atomic_action.move_end_effector import (
        create_robot,
        initialize_simulation,
    )

    pose_cases = _select_pose_cases(args.pose_cases)
    repeat = 1 if profile == "smoke" else args.repeat
    if profile == "smoke":
        pose_cases = [POSE_CASES["front_left"]]

    print("=" * 60)
    print("MoveEndEffector Atomic Action Benchmark")
    print("=" * 60)
    print(
        "Coverage: "
        f"profile={profile}, {len(pose_cases)} pose case(s) x "
        f"{repeat} repeat(s)"
    )

    sim = initialize_simulation(args)
    robot = create_robot(sim)
    initial_qpos = robot.get_qpos().clone()
    motion_gen = MotionGenerator(
        cfg=MotionGenCfg(planner_cfg=ToppraPlannerCfg(robot_uid=robot.uid))
    )
    atomic_engine = AtomicActionEngine(motion_generator=motion_gen)
    atomic_engine.register(
        MoveEndEffector(
            motion_gen,
            cfg=MoveEndEffectorCfg(
                control_part="arm", sample_interval=MOVE_SAMPLE_INTERVAL
            ),
        )
    )

    results: list[dict[str, object]] = []
    video_paths: list[str] = []
    print("\n=== MoveEndEffector Target Sweep ===")
    for pose_case in pose_cases:
        for repeat_index in range(repeat):
            result = _run_case(
                sim,
                robot,
                atomic_engine,
                initial_qpos,
                pose_case,
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
                f"err={format_float(result['final_error_m'], precision=4)}"
            )

    perf_rows, metric_rows = _build_rows(results)
    leaderboard_rows = build_single_action_leaderboard("move_end_effector", metric_rows)
    report_path = write_markdown_report(
        benchmark_name="atomic_action_move_end_effector",
        perf_rows=perf_rows,
        metric_rows=metric_rows,
        leaderboard_rows=leaderboard_rows,
        notes=[
            f"Profile: {profile}",
            f"CPU memory backend: {CPU_MEMORY_BACKEND}",
            f"Success tolerance: {SUCCESS_TOLERANCE_M} m translation error.",
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
