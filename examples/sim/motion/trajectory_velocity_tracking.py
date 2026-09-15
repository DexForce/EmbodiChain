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

"""Compare DexSim position tracking with zero and planned velocity targets.

Both trials restore the same Franka initial state and execute the same timed
reference at the same control and physics cadence. The position-only trial
explicitly writes zero target velocity at every command. Results are descriptive
for this robot, drive configuration, and cadence; velocity feed-forward is not
expected to improve every controller or trajectory.

Joint-space tracking errors and forward-kinematics end-effector errors are
reported for both trials. Translation error is a distance in meters; rotation
error is the geodesic angle between reference and measured orientations.

Run from the repository root::

    python examples/sim/motion/trajectory_velocity_tracking.py --headless
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from embodichain.cli.sim import add_sim_args_to_parser

__all__ = [
    "compute_pose_errors",
    "compute_tracking_metrics",
    "generate_reference",
    "main",
]


def generate_reference(
    start_qpos: torch.Tensor,
    *,
    duration: float,
    control_dt: float,
    amplitude: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a smooth closed joint trajectory and its timed derivatives.

    Args:
        start_qpos: Initial joint positions shaped ``(B, DOF)``.
        duration: Reference duration in seconds.
        control_dt: Uniform reference interval in seconds.
        amplitude: Peak joint displacement in radians.

    Returns:
        Position samples, velocity samples, and arrival intervals.
    """
    if start_qpos.dim() != 2:
        raise ValueError("start_qpos must have shape (B, DOF).")
    if duration <= 0.0 or control_dt <= 0.0:
        raise ValueError("duration and control_dt must be positive.")
    steps = round(duration / control_dt)
    if steps < 2 or abs(steps * control_dt - duration) > 1e-9:
        raise ValueError(
            "duration must be an exact multiple of control_dt with at least two steps."
        )

    phase = torch.linspace(
        0.0,
        torch.pi,
        steps + 1,
        dtype=start_qpos.dtype,
        device=start_qpos.device,
    )
    signs = torch.where(
        torch.arange(start_qpos.shape[1], device=start_qpos.device) % 2 == 0,
        1.0,
        -1.0,
    ).to(start_qpos.dtype)
    positions = (
        start_qpos[:, None, :]
        + amplitude * torch.sin(phase)[None, :, None] ** 2 * signs
    )
    dt = torch.full(
        (start_qpos.shape[0], steps + 1),
        control_dt,
        dtype=start_qpos.dtype,
        device=start_qpos.device,
    )
    dt[:, 0] = 0.0
    velocities = differentiate_positions(positions, dt)
    velocities[:, 0] = 0.0
    velocities[:, -1] = 0.0
    return positions, velocities, dt


def compute_tracking_metrics(
    reference: torch.Tensor, measured: torch.Tensor
) -> dict[str, float]:
    """Return scalar absolute joint-error metrics over all samples and joints.

    Args:
        reference: Reference joint samples.
        measured: Measured joint samples with the same shape.

    Returns:
        RMSE, 95th-percentile absolute error, and maximum absolute error.
    """
    if reference.shape != measured.shape or reference.numel() == 0:
        raise ValueError("reference and measured must have the same non-empty shape.")
    return _compute_error_metrics((measured - reference).abs())


def compute_pose_errors(
    reference: torch.Tensor, measured: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute end-effector translation and rotation errors.

    Args:
        reference: Reference homogeneous transforms shaped ``(..., 4, 4)``.
        measured: Measured homogeneous transforms with the same shape.

    Returns:
        Translation error in meters and geodesic rotation error in radians,
        each shaped ``reference.shape[:-2]``.
    """
    if (
        reference.shape != measured.shape
        or reference.numel() == 0
        or reference.shape[-2:] != (4, 4)
    ):
        raise ValueError(
            "reference and measured must have the same non-empty (..., 4, 4) shape."
        )
    translation_error = torch.linalg.vector_norm(
        measured[..., :3, 3] - reference[..., :3, 3], dim=-1
    )
    relative_rotation = torch.matmul(
        reference[..., :3, :3].transpose(-1, -2), measured[..., :3, :3]
    )
    trace = relative_rotation.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    rotation_error = torch.acos(torch.clamp((trace - 1.0) * 0.5, -1.0, 1.0))
    return translation_error, rotation_error


def _compute_error_metrics(error: torch.Tensor) -> dict[str, float]:
    """Summarize a non-empty scalar error tensor."""
    error = error.reshape(-1).float()
    return {
        "rmse": float(torch.sqrt(torch.mean(error.square())).item()),
        "p95": float(torch.quantile(error, 0.95).item()),
        "max": float(error.max().item()),
    }


def _compute_fk_trajectory(
    robot: Robot, qpos: torch.Tensor, *, control_part: str
) -> torch.Tensor:
    """Compute one end-effector pose for every joint trajectory sample."""
    if qpos.dim() != 3 or qpos.shape[1] == 0:
        raise ValueError("qpos must have shape (B, N, DOF) with N > 0.")
    poses = [
        robot.compute_fk(
            qpos=qpos[:, sample], name=control_part, to_matrix=True
        ).clone()
        for sample in range(qpos.shape[1])
    ]
    return torch.stack(poses, dim=1)


def build_parser() -> argparse.ArgumentParser:
    """Build CLI options without importing simulation dependencies."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_sim_args_to_parser(parser)
    parser.set_defaults(device="cpu", num_envs=1, arena_space=2.0)
    parser.add_argument("--physics-dt", type=float, default=0.01)
    parser.add_argument("--control-dt", type=float, default=0.04)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--amplitude", type=float, default=0.2)
    parser.add_argument("--settle-seconds", type=float, default=0.5)
    parser.add_argument("--terminal-settle-seconds", type=float, default=0.5)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("trajectory_velocity_results")
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the trajectory tracking example."""
    parser = build_parser()
    return parser.parse_args() if argv is None else parser.parse_args(argv)


if __name__ == "__main__":
    # Parse before importing optional simulation dependencies.
    _cli_args = parse_args()


import torch

from embodichain.compute.trajectory import differentiate_positions
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import RenderCfg, physics_cfg_for_backend
from embodichain.lab.sim.objects import Robot
from embodichain.lab.sim.robots import FrankaPandaCfg
from embodichain.lab.visualization import visualization_cfg_from_args


def _validate_args(args: argparse.Namespace) -> None:
    values = {
        "physics_dt": args.physics_dt,
        "control_dt": args.control_dt,
        "duration": args.duration,
        "amplitude": args.amplitude,
        "settle_seconds": args.settle_seconds,
        "terminal_settle_seconds": args.terminal_settle_seconds,
    }
    if any(not torch.isfinite(torch.tensor(value)) for value in values.values()):
        raise ValueError("Timing and amplitude arguments must be finite.")
    if args.physics_dt <= 0 or args.control_dt <= 0 or args.duration <= 0:
        raise ValueError("physics_dt, control_dt, and duration must be positive.")
    if (
        args.amplitude < 0
        or args.settle_seconds < 0
        or args.terminal_settle_seconds < 0
    ):
        raise ValueError("amplitude and settling durations must be non-negative.")


def _run_trial(
    args: argparse.Namespace,
    sim: SimulationManager,
    robot: Robot,
    common_initial_qpos: torch.Tensor,
    use_velocity: bool,
) -> dict[str, torch.Tensor]:
    ratio = args.control_dt / args.physics_dt
    physics_steps = round(ratio)
    if physics_steps < 1 or abs(physics_steps - ratio) > 1e-9:
        raise ValueError("control_dt must be an exact positive multiple of physics_dt.")

    all_zero_velocity = torch.zeros_like(common_initial_qpos)
    robot.set_qvel(all_zero_velocity, target=False)
    robot.set_qvel(all_zero_velocity, target=True)
    robot.reset()
    robot.set_qpos(common_initial_qpos, target=False)
    robot.set_qpos(common_initial_qpos, target=True)
    robot.set_qvel(all_zero_velocity, target=False)
    robot.set_qvel(all_zero_velocity, target=True)
    joint_ids = robot.get_joint_ids("arm")
    initial = common_initial_qpos[:, joint_ids].clone()
    zeros = torch.zeros_like(initial)
    settle_steps = round(args.settle_seconds / args.physics_dt)
    if settle_steps > 0:
        sim.update(step=settle_steps)

    reference, velocity, dt = generate_reference(
        initial,
        duration=args.duration,
        control_dt=args.control_dt,
        amplitude=args.amplitude,
    )
    reference_pose = _compute_fk_trajectory(robot, reference, control_part="arm")
    robot.set_qpos(reference[:, 0], joint_ids=joint_ids)
    robot.set_qvel(velocity[:, 0] if use_velocity else zeros, joint_ids=joint_ids)
    measured = [robot.get_qpos()[:, joint_ids].clone()]
    measured_pose = [
        robot.compute_fk(qpos=measured[0], name="arm", to_matrix=True).clone()
    ]
    timestamps = [0.0]
    for index in range(reference.shape[1] - 1):
        if index > 0:
            robot.set_qpos(reference[:, index], joint_ids=joint_ids)
            target_velocity = velocity[:, index] if use_velocity else zeros
            robot.set_qvel(target_velocity, joint_ids=joint_ids)
        sim.update(step=physics_steps)
        timestamps.append(timestamps[-1] + float(dt[0, index + 1]))
        measured.append(robot.get_qpos()[:, joint_ids].clone())
        measured_pose.append(
            robot.compute_fk(qpos=measured[-1], name="arm", to_matrix=True).clone()
        )

    robot.set_qpos(reference[:, -1], joint_ids=joint_ids)
    robot.set_qvel(zeros, joint_ids=joint_ids)
    terminal_steps = round(args.terminal_settle_seconds / args.physics_dt)
    if terminal_steps > 0:
        sim.update(step=terminal_steps)
    return {
        "time": torch.tensor(timestamps),
        "reference": reference.cpu(),
        "measured": torch.stack(measured, dim=1).cpu(),
        "reference_pose": reference_pose.cpu(),
        "measured_pose": torch.stack(measured_pose, dim=1).cpu(),
        "velocity": velocity.cpu(),
    }


def _write_artifacts(
    output_dir: Path, trials: dict[str, dict[str, torch.Tensor]]
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "tracking.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["mode", "time_s", "joint", "reference_rad", "measured_rad", "error_rad"]
        )
        for mode, trial in trials.items():
            for sample, timestamp in enumerate(trial["time"].tolist()):
                for joint, (reference, measured) in enumerate(
                    zip(
                        trial["reference"][0, sample].tolist(),
                        trial["measured"][0, sample].tolist(),
                    )
                ):
                    writer.writerow(
                        [
                            mode,
                            timestamp,
                            joint,
                            reference,
                            measured,
                            measured - reference,
                        ]
                    )

    pose_csv_path = output_dir / "pose_tracking.csv"
    with pose_csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["mode", "time_s", "translation_error_m", "rotation_error_rad"])
        for mode, trial in trials.items():
            translation_error, rotation_error = compute_pose_errors(
                trial["reference_pose"], trial["measured_pose"]
            )
            for timestamp, translation, rotation in zip(
                trial["time"].tolist(),
                translation_error[0].tolist(),
                rotation_error[0].tolist(),
            ):
                writer.writerow([mode, timestamp, translation, rotation])

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN]: matplotlib is unavailable; wrote CSV only.")
        return
    figure, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    for mode, trial in trials.items():
        joint_error = torch.sqrt(
            torch.mean((trial["measured"] - trial["reference"]) ** 2, dim=(0, 2))
        )
        translation_error, rotation_error = compute_pose_errors(
            trial["reference_pose"], trial["measured_pose"]
        )
        axes[0].plot(trial["time"], joint_error, label=mode)
        axes[1].plot(trial["time"], translation_error[0], label=mode)
        axes[2].plot(trial["time"], rotation_error[0], label=mode)
    axes[0].set(ylabel="joint RMSE (rad)", title="DexSim trajectory tracking")
    axes[1].set(ylabel="FK translation error (m)")
    axes[2].set(xlabel="measurement time (s)", ylabel="FK rotation error (rad)")
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "tracking.png", dpi=160)
    plt.close(figure)


def main(args: argparse.Namespace | None = None) -> None:
    """Run both tracking modes and write their measurements."""
    args = parse_args() if args is None else args
    _validate_args(args)
    sim = SimulationManager(
        SimulationManagerCfg(
            headless=args.headless,
            physics_dt=args.physics_dt,
            sim_device=args.device,
            num_envs=1,
            arena_space=args.arena_space,
            render_cfg=RenderCfg(renderer=args.renderer),
            physics_cfg=physics_cfg_for_backend(args.physics),
            visualization=visualization_cfg_from_args(args),
        )
    )
    try:
        robot: Robot = sim.add_robot(
            FrankaPandaCfg.from_dict({"uid": "tracking_franka"})
        )
        if sim.is_use_gpu_physics:
            sim.init_gpu_physics()
        if not args.headless:
            sim.open_window()
        common_initial_qpos = robot.get_qpos().clone()
        trials = {
            "qpos_only_zero_qvel": _run_trial(
                args, sim, robot, common_initial_qpos, use_velocity=False
            ),
            "qpos_plus_qvel": _run_trial(
                args, sim, robot, common_initial_qpos, use_velocity=True
            ),
        }
        torch.testing.assert_close(
            trials["qpos_only_zero_qvel"]["reference"],
            trials["qpos_plus_qvel"]["reference"],
            rtol=0.0,
            atol=0.0,
        )
        _write_artifacts(args.output_dir, trials)
        for mode, trial in trials.items():
            joint_metrics = compute_tracking_metrics(
                trial["reference"], trial["measured"]
            )
            translation_error, rotation_error = compute_pose_errors(
                trial["reference_pose"], trial["measured_pose"]
            )
            translation_metrics = _compute_error_metrics(translation_error)
            rotation_metrics = _compute_error_metrics(rotation_error)
            print(
                f"{mode}:\n"
                f"  joint: RMSE={joint_metrics['rmse']:.6f} rad, "
                f"P95={joint_metrics['p95']:.6f} rad, "
                f"max={joint_metrics['max']:.6f} rad\n"
                f"  FK translation: RMSE={translation_metrics['rmse']:.6f} m, "
                f"P95={translation_metrics['p95']:.6f} m, "
                f"max={translation_metrics['max']:.6f} m\n"
                f"  FK rotation: RMSE={rotation_metrics['rmse']:.6f} rad, "
                f"P95={rotation_metrics['p95']:.6f} rad, "
                f"max={rotation_metrics['max']:.6f} rad",
                flush=True,
            )
        print(f"Artifacts: {args.output_dir.resolve()}", flush=True)
    finally:
        sim.destroy()


if __name__ == "__main__":
    main(_cli_args)
