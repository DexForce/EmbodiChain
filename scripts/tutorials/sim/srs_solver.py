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

"""Move a W1 end effector along a Cartesian straight line with SRS IK."""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.objects import Robot
from embodichain.lab.sim.robots import DexforceW1Cfg
from embodichain.lab.visualization import (
    VisualizationCfg,
    add_viser_args_to_parser,
    visualization_cfg_from_args,
)


def main(
    device: str,
    num_steps: int,
    line_offset: tuple[float, float, float],
    physics_steps: int,
    visualization: VisualizationCfg | None = None,
) -> None:
    """Run sequential SRS IK targets along a fixed-orientation straight line.

    Args:
        device: Simulation and SRS solver device, either ``cpu`` or ``cuda``.
        num_steps: Number of Cartesian waypoints, including both endpoints.
        line_offset: End-point translation relative to the initial TCP pose, in meters.
        physics_steps: Number of simulation steps displayed per waypoint.
        visualization: Optional visualization configuration.
    """
    if num_steps < 2:
        raise ValueError("num_steps must be at least 2")
    if physics_steps < 1:
        raise ValueError("physics_steps must be at least 1")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but PyTorch cannot access a CUDA device"
        )

    np.set_printoptions(precision=5, suppress=True)
    torch.set_printoptions(precision=5, sci_mode=False)
    sim = SimulationManager(
        SimulationManagerCfg(
            headless=False,
            sim_device=device,
            width=2200,
            height=1200,
            visualization=visualization or VisualizationCfg(),
        )
    )
    # Keep pose conversion, IK, and FK error measurement deterministic. In automatic
    # mode the engine thread may advance the robot base between these operations.
    sim.set_manual_update(True)

    try:
        robot: Robot = sim.add_robot(
            cfg=DexforceW1Cfg.from_dict({"uid": "dexforce_w1"})
        )
        arm_name = "left_arm"
        joint_ids = robot.get_joint_ids(arm_name)
        qpos_seed = torch.tensor(
            [[0.0, 0.0, 0.0, -np.pi / 2, 0.0, 0.0, 0.0]],
            dtype=torch.float32,
            device=sim.device,
        )
        robot.set_qpos(qpos_seed, joint_ids=joint_ids)
        sim.update(step=physics_steps)

        start_pose = robot.compute_fk(qpos=qpos_seed, name=arm_name, to_matrix=True)
        target_poses = start_pose.repeat(num_steps, 1, 1)
        offset = torch.tensor(
            line_offset, dtype=start_pose.dtype, device=start_pose.device
        )
        interpolation = torch.linspace(
            0.0, 1.0, num_steps, dtype=start_pose.dtype, device=start_pose.device
        )
        target_poses[:, :3, 3] += interpolation.unsqueeze(1) * offset

        # Warm up lazy FK compilation and the selected SRS backend outside timing.
        robot.compute_ik(
            pose=target_poses[:1],
            joint_seed=qpos_seed,
            name=arm_name,
            return_all_solutions=False,
        )
        if device == "cuda":
            torch.cuda.synchronize()

        solve_times_ms: list[float] = []
        translation_errors_mm: list[float] = []
        solved_waypoints = 0
        for waypoint, target_pose in enumerate(target_poses):
            start_time = time.perf_counter()
            success, solution = robot.compute_ik(
                pose=target_pose.unsqueeze(0),
                joint_seed=qpos_seed,
                name=arm_name,
                return_all_solutions=False,
            )
            if device == "cuda":
                torch.cuda.synchronize()
            solve_times_ms.append((time.perf_counter() - start_time) * 1000.0)

            if not bool(success[0]):
                print(f"Waypoint {waypoint + 1}/{num_steps}: IK failed")
                break

            qpos_seed = solution.reshape(1, 7)
            robot.set_qpos(qpos_seed, joint_ids=joint_ids)
            # Measure the solver reconstruction before advancing physics so that
            # movement of the simulated base is not counted as analytical IK error.
            actual_pose = robot.compute_fk(
                qpos=qpos_seed, name=arm_name, to_matrix=True
            )
            error_mm = float(
                torch.linalg.vector_norm(
                    actual_pose[0, :3, 3] - target_pose[:3, 3]
                ).item()
                * 1000.0
            )
            translation_errors_mm.append(error_mm)
            solved_waypoints += 1
            print(
                f"Waypoint {waypoint + 1:03d}/{num_steps}: "
                f"solve={solve_times_ms[-1]:.3f} ms, error={error_mm:.4f} mm"
            )
            sim.update(step=physics_steps)

        print(
            f"SRS {device.upper()} summary: {solved_waypoints}/{num_steps} solved, "
            f"mean solve={np.mean(solve_times_ms):.3f} ms, "
            f"max translation error="
            f"{max(translation_errors_mm, default=float('nan')):.4f} mm"
        )
        sim.capture_visualization(force=True)
    finally:
        sim.destroy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", choices=("cpu", "cuda"), default="cpu", help="SRS backend."
    )
    parser.add_argument(
        "--num-steps", type=int, default=50, help="Number of line waypoints."
    )
    parser.add_argument(
        "--line-offset",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=(0.0, 0.10, 0.0),
        help="TCP line displacement in meters (default: 0 0.10 0).",
    )
    parser.add_argument(
        "--physics-steps",
        type=int,
        default=2,
        help="Simulation steps displayed per waypoint.",
    )
    add_viser_args_to_parser(parser)
    args = parser.parse_args()
    main(
        device=args.device,
        num_steps=args.num_steps,
        line_offset=tuple(args.line_offset),
        physics_steps=args.physics_steps,
        visualization=visualization_cfg_from_args(args),
    )
