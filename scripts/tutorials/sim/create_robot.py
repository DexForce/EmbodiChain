# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
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

import argparse
import numpy as np
import time
import torch

torch.set_printoptions(precision=4, sci_mode=False)

from scipy.spatial.transform import Rotation as R

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.objects import Robot
from embodichain.lab.sim.cfg import (
    JointDrivePropertiesCfg,
    RobotCfg,
    URDFCfg,
)
from embodichain.data import get_data_path


def main():
    """Main function to demonstrate robot simulation."""

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Create and simulate a robot in SimulationManager"
    )
    parser.add_argument(
        "--num_envs", type=int, default=1, help="Number of environments to simulate"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run simulation on",
    )
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    parser.add_argument(
        "--enable_rt", action="store_true", help="Enable ray tracing rendering"
    )
    args = parser.parse_args()

    # Initialize simulation
    print("Creating simulation...")
    config = SimulationManagerCfg(
        headless=True,
        sim_device=args.device,
        arena_space=3.0,
        enable_rt=args.enable_rt,
        physics_dt=1.0 / 100.0,
    )
    sim = SimulationManager(config)

    # Build multiple environments if requested
    if args.num_envs > 1:
        sim.build_multiple_arenas(args.num_envs)

    # Create robot configuration
    robot = create_robot(sim)

    # Initialize GPU physics if using CUDA
    if sim.is_use_gpu_physics:
        sim.init_gpu_physics()

    # Open visualization window if not headless
    if not args.headless:
        sim.open_window()

    # Run simulation loop
    run_simulation(sim, robot)


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
            "JOINT[1-6]",  # Matches JOINT1, JOINT2, ..., JOINT6
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
            stiffness={"JOINT[1-6]": 1e4, "LEFT_.*": 1e3},
            damping={"JOINT[1-6]": 1e3, "LEFT_.*": 1e2},
        ),
    )

    # Add robot to simulation
    robot: Robot = sim.add_robot(cfg=cfg)

    print(f"Robot created successfully with {robot.dof} joints")

    return robot


def run_simulation(sim: SimulationManager, robot: Robot):
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
    # Define hand open and close positions based on joint limits.
    hand_position_open = robot.body_data.qpos_limits[:, hand_joint_ids, 1]
    hand_position_close = robot.body_data.qpos_limits[:, hand_joint_ids, 0]

    try:
        while True:
            # Update physics
            sim.update(step=1)

            if step_count % 4000 == 0:
                robot.set_qpos(qpos=arm_position1, joint_ids=arm_joint_ids)
                print(f"Moving to arm position 1")

            if step_count % 4000 == 1000:
                robot.set_qpos(qpos=arm_position2, joint_ids=arm_joint_ids)
                print(f"Moving to arm position 2")

            if step_count % 4000 == 2000:
                robot.set_qpos(qpos=hand_position_close, joint_ids=hand_joint_ids)
                print(f"Closing hand")

            if step_count % 4000 == 3000:
                robot.set_qpos(qpos=hand_position_open, joint_ids=hand_joint_ids)
                print(f"Opening hand")

            step_count += 1

    except KeyboardInterrupt:
        print("Stopping simulation...")
    finally:
        print("Cleaning up...")
        sim.destroy()


if __name__ == "__main__":
    main()
