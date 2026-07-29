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
import torch

from scipy.spatial.transform import Rotation as R

from embodichain.lab.sim import SimulationManager
from embodichain.lab.sim.objects import Robot
from embodichain.lab.sim.cfg import (
    JointDrivePropertiesCfg,
    RobotCfg,
    URDFCfg,
)
from embodichain.data import get_data_path
from embodichain.lab.sim.utility.demo_utils import (
    DemoRecording,
    add_demo_args,
    create_default_sim,
    maybe_init_gpu_physics,
    maybe_open_window,
    resolve_demo_steps,
    run_simulation_loop,
    setup_print_options,
    shutdown_sim,
)

ACTION_SWITCH_INTERVAL = 100
ACTION_CYCLE_STEPS = 4 * ACTION_SWITCH_INTERVAL


def main():
    """Main function to demonstrate robot simulation."""

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Create and simulate a robot in SimulationManager"
    )
    add_demo_args(parser)
    args = parser.parse_args()
    setup_print_options()

    # Initialize simulation
    print("Creating simulation...")
    sim = create_default_sim(
        args,
        arena_space=3.0,
        num_envs=args.num_envs,
        add_default_light=False,
    )

    # Create robot configuration
    robot = create_robot(sim)

    # Initialize GPU physics if using CUDA
    maybe_init_gpu_physics(sim)

    # Open visualization window if not headless
    maybe_open_window(sim, args)

    # Run simulation loop
    run_simulation(sim, robot, args)


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
        drive_pros=JointDrivePropertiesCfg(
            stiffness={"joint[1-6]": 1e4, "LEFT_.*": 1e3},
            damping={"joint[1-6]": 1e3, "LEFT_.*": 1e2},
        ),
    )

    # Add robot to simulation
    robot: Robot = sim.add_robot(cfg=cfg)

    print(f"Robot created successfully with {robot.dof} joints")

    return robot


def run_simulation(
    sim: SimulationManager,
    robot: Robot,
    args: argparse.Namespace,
) -> None:
    """Run the simulation loop with robot control."""

    print("Starting simulation...")
    print("Robot will move through different poses")
    print("Press Ctrl+C to stop")

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
    # Define hand open and close positions based on joint limits.
    hand_position_open = robot.body_data.qpos_limits[:, hand_joint_ids, 1]
    hand_position_close = robot.body_data.qpos_limits[:, hand_joint_ids, 0]

    def update_target(step: int) -> None:
        """Switch arm and hand targets at fixed simulation intervals."""
        cycle_step = (step - 1) % ACTION_CYCLE_STEPS

        if cycle_step == 0:
            robot.set_qpos(qpos=arm_position1, joint_ids=arm_joint_ids)
            print("Moving to arm position 1")

        if cycle_step == ACTION_SWITCH_INTERVAL:
            robot.set_qpos(qpos=arm_position2, joint_ids=arm_joint_ids)
            print("Moving to arm position 2")

        if cycle_step == 2 * ACTION_SWITCH_INTERVAL:
            robot.set_qpos(qpos=hand_position_close, joint_ids=hand_joint_ids)
            print("Closing hand")

        if cycle_step == 3 * ACTION_SWITCH_INTERVAL:
            robot.set_qpos(qpos=hand_position_open, joint_ids=hand_joint_ids)
            print("Opening hand")

    try:
        with DemoRecording(sim, args, prefix="create_robot"):
            run_simulation_loop(
                sim,
                max_steps=resolve_demo_steps(args),
                on_step=update_target,
            )
    finally:
        print("Cleaning up...")
        shutdown_sim(sim)


if __name__ == "__main__":
    main()
