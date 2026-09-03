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

"""
This script demonstrates how to create and simulate a robot using SimulationManager.
It shows how to load a robot from URDF, set up control parts, and run basic simulation.
"""

from __future__ import annotations

import argparse
import numpy as np
import time
import torch

torch.set_printoptions(precision=4, sci_mode=False)

from scipy.spatial.transform import Rotation as R

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.visualization import visualization_cfg_from_args
from embodichain.lab.sim.objects import Robot
from embodichain.lab.sim.cfg import (
    RenderCfg,
    physics_cfg_for_backend,
    JointDrivePropertiesCfg,
    RobotCfg,
    URDFCfg,
)
from embodichain.data import get_data_path
from embodichain.lab.gym.utils.gym_utils import add_env_launcher_args_to_parser

ACTION_SWITCH_INTERVAL = 100
ACTION_CYCLE_STEPS = 4 * ACTION_SWITCH_INTERVAL


def main():
    """Main function to demonstrate robot simulation."""

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Create and simulate a robot in SimulationManager"
    )
    add_env_launcher_args_to_parser(parser)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Stop after this many physics steps (default: run until interrupted).",
    )
    args = parser.parse_args()
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("--max-steps must be at least 1")

    # Initialize simulation
    print("Creating simulation...")
    config = SimulationManagerCfg(
        headless=True,
        device=args.device,
        arena_space=3.0,
        render_cfg=RenderCfg(renderer=args.renderer),
        physics_cfg=physics_cfg_for_backend(args.physics),
        physics_dt=1.0 / 100.0,
        num_envs=args.num_envs,
        visualization=visualization_cfg_from_args(args),
    )
    sim = SimulationManager(config)

    # Create robot configuration
    robot = create_robot(sim)

    # Materialize the declared scene before accessing robot metadata.
    sim.prepare()
    print(f"Robot created successfully with {robot.dof} joints")

    # Open visualization window if not headless
    if not args.headless:
        sim.open_window()

    # Run simulation loop
    run_simulation(sim, robot, max_steps=args.max_steps)


def create_robot(sim):
    """Create and configure a robot in the simulation."""

    print("Loading robot...")

    # Get SR5 arm URDF path
    sr5_urdf_path = get_data_path("Rokae/SR5/SR5.urdf")

    # Get hand URDF path
    hand_urdf_path = get_data_path(
        "BrainCoHandRevo1/BrainCoLeftHand/BrainCoLeftHand.urdf"
    )

    # Define control parts for the robot
    # Joint names in control_parts can be regex patterns
    CONTROL_PARTS = {
        "arm": [
            "joint[1-6]",  # Matches JOINT1, JOINT2, ..., JOINT6
        ],
        "hand": ["LEFT_.*"],  # Matches all joints starting with L_
    }

    # Define transformation for hand attachment
    hand_attach_xpos = np.eye(4)
    hand_attach_xpos[:3, :3] = R.from_rotvec([90, 0, 0], degrees=True).as_matrix()
    hand_attach_xpos[2, 3] = 0.02

    cfg = RobotCfg(
        uid="sr5_with_brainco",
        urdf_cfg=URDFCfg(
            components=[
                {
                    "component_type": "arm",
                    "urdf_path": sr5_urdf_path,
                },
                {
                    "component_type": "hand",
                    "urdf_path": hand_urdf_path,
                    "transform": hand_attach_xpos,
                },
            ]
        ),
        control_parts=CONTROL_PARTS,
        joint_drive_props=JointDrivePropertiesCfg(
            drive_type="force",
            stiffness={"joint[1-6]": 1e4, "LEFT_.*": 1e3},
            damping={"joint[1-6]": 1.5e3, "LEFT_.*": 1e2},
            max_effort={"joint[1-6]": 1e4, "LEFT_.*": 1e4},
        ),
    )

    # Add robot to simulation
    robot: Robot = sim.add_robot(cfg=cfg)

    return robot


def _expand_mimic_targets(
    robot: Robot, joint_ids: list[int], joint_targets: torch.Tensor
) -> torch.Tensor:
    """Expand active-joint targets into mimic-consistent articulation targets."""

    targets = robot.get_qpos(target=True).clone()
    targets[:, joint_ids] = joint_targets

    for mimic_id, parent_id, multiplier, offset in zip(
        robot.mimic_ids,
        robot.mimic_parents,
        robot.mimic_multipliers,
        robot.mimic_offsets,
    ):
        if mimic_id is None or parent_id is None:
            continue
        targets[:, mimic_id] = offset + multiplier * targets[:, parent_id]

    limits = robot.body_data.qpos_limits
    return targets.clamp(min=limits[..., 0], max=limits[..., 1])


def run_simulation(
    sim: SimulationManager, robot: Robot, max_steps: int | None = None
) -> None:
    """Run the simulation loop with robot control."""

    print("Starting simulation...")
    print("Robot will move through different poses")
    print("Press Ctrl+C to stop")

    step_count = 0

    arm_joint_ids = robot.get_joint_ids("arm")
    # Define some target joint positions for demonstration
    arm_position1 = (
        torch.tensor(
            [0.0, -0.5, 0.5, -1.0, 0.5, 0.0], dtype=torch.float32, device=sim.device
        )
        .unsqueeze_(0)
        .repeat(sim.num_envs, 1)
    )

    arm_position2 = (
        torch.tensor(
            [0.5, 0.0, -0.5, 0.5, -0.5, 0.5], dtype=torch.float32, device=sim.device
        )
        .unsqueeze_(0)
        .repeat(sim.num_envs, 1)
    )

    # Get joint IDs for the hand.
    hand_joint_ids = robot.get_joint_ids("hand")
    active_hand_joint_ids = robot.get_joint_ids("hand", remove_mimic=True)
    # Drive mimic joints toward the pose implied by their active parent instead of
    # sending each joint to its independent limit. Newton keeps drives on mimic
    # joints, so inconsistent targets otherwise compete with the mimic constraints.
    hand_position_open = _expand_mimic_targets(
        robot,
        active_hand_joint_ids,
        robot.body_data.qpos_limits[:, active_hand_joint_ids, 1],
    )[:, hand_joint_ids]
    hand_position_close = _expand_mimic_targets(
        robot,
        active_hand_joint_ids,
        robot.body_data.qpos_limits[:, active_hand_joint_ids, 0],
    )[:, hand_joint_ids]

    # The reset pose is zero for every DOF, but this hand has non-zero mimic
    # offsets. Start from a valid closed pose so the initial state and drive
    # targets satisfy the same mimic equations.
    robot.set_qpos(qpos=hand_position_close, joint_ids=hand_joint_ids, target=False)
    robot.set_qpos(qpos=hand_position_close, joint_ids=hand_joint_ids)

    try:
        while max_steps is None or step_count < max_steps:
            cycle_step = step_count % ACTION_CYCLE_STEPS

            if cycle_step == 0:
                robot.set_qpos(qpos=arm_position1, joint_ids=arm_joint_ids)
                print(f"Moving to arm position 1")

            if cycle_step == ACTION_SWITCH_INTERVAL:
                robot.set_qpos(qpos=arm_position2, joint_ids=arm_joint_ids)
                print(f"Moving to arm position 2")

            if cycle_step == 2 * ACTION_SWITCH_INTERVAL:
                robot.set_qpos(qpos=hand_position_close, joint_ids=hand_joint_ids)
                print(f"Closing hand")

            if cycle_step == 3 * ACTION_SWITCH_INTERVAL:
                robot.set_qpos(qpos=hand_position_open, joint_ids=hand_joint_ids)
                print(f"Opening hand")

            # Apply commands before advancing physics so both backends observe the
            # target change on the same simulation step.
            sim.update(step=1)
            step_count += 1

    except KeyboardInterrupt:
        print("Stopping simulation...")
    finally:
        print("Cleaning up...")
        sim.destroy()


if __name__ == "__main__":
    main()
