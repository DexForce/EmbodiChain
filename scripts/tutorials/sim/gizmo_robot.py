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
Gizmo-Robot Example: Test Gizmo class on a robot (UR10)
"""

import time
import torch
import numpy as np
import argparse

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import (
    RobotCfg,
    URDFCfg,
    JointDrivePropertiesCfg,
)

from embodichain.lab.sim.solvers import PinkSolverCfg
from embodichain.data import get_data_path
from embodichain.utils import logger


def main():
    """Main function to create and run the simulation scene."""

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Create a simulation scene with SimulationManager"
    )
    parser.add_argument(
        "--num_envs", type=int, default=1, help="Number of parallel environments"
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Simulation device (cuda or cpu)"
    )
    parser.add_argument(
        "--enable_rt",
        action="store_true",
        default=False,
        help="Enable ray tracing for better visuals",
    )
    args = parser.parse_args()

    # Configure the simulation
    sim_cfg = SimulationManagerCfg(
        width=1920,
        height=1080,
        physics_dt=1.0 / 100.0,
        sim_device=args.device,
        enable_rt=args.enable_rt,
    )

    sim = SimulationManager(sim_cfg)
    sim.set_manual_update(False)

    # Get UR10 URDF path
    urdf_path = get_data_path("UniversalRobots/UR10/UR10.urdf")

    # Create UR10 robot
    robot_cfg = RobotCfg(
        uid="ur10_gizmo_test",
        urdf_cfg=URDFCfg(
            components=[{"component_type": "arm", "urdf_path": urdf_path}]
        ),
        control_parts={"arm": ["Joint[1-6]"]},
        solver_cfg={
            "arm": PinkSolverCfg(
                urdf_path=urdf_path,
                end_link_name="ee_link",
                root_link_name="base_link",
                pos_eps=1e-2,
                rot_eps=5e-2,
                max_iterations=300,
                dt=0.1,
            )
        },
        drive_pros=JointDrivePropertiesCfg(
            stiffness={"Joint[1-6]": 1e4},
            damping={"Joint[1-6]": 1e3},
        ),
    )
    robot = sim.add_robot(cfg=robot_cfg)

    # Set initial joint positions
    initial_qpos = torch.tensor(
        [[0, -np.pi / 2, np.pi / 2, 0.0, np.pi / 2, 0.0]],
        dtype=torch.float32,
        device="cpu",
    )
    joint_ids = robot.get_joint_ids("arm")
    robot.set_qpos(qpos=initial_qpos, joint_ids=joint_ids)

    time.sleep(0.2)  # Wait for a moment to ensure everything is set up

    # Enable gizmo using the new API
    sim.enable_gizmo(uid="ur10_gizmo_test", control_part="arm")
    if not sim.has_gizmo("ur10_gizmo_test", control_part="arm"):
        logger.log_error("Failed to enable gizmo!")
        return

    sim.open_window()

    logger.log_info("Gizmo-Robot example started!")
    logger.log_info("Use the gizmo to drag the robot end-effector (EE)")
    logger.log_info("Press Ctrl+C to stop the simulation")

    run_simulation(sim)


def run_simulation(sim: SimulationManager):
    step_count = 0
    try:
        last_time = time.time()
        last_step = 0
        while True:
            time.sleep(0.033)  # 30Hz
            # Update all gizmos managed by sim
            sim.update_gizmos()
            step_count += 1

            if step_count % 100 == 0:
                current_time = time.time()
                elapsed = current_time - last_time
                fps = (
                    sim.num_envs * (step_count - last_step) / elapsed
                    if elapsed > 0
                    else 0
                )
                logger.log_info(f"Simulation step: {step_count}, FPS: {fps:.2f}")
                last_time = current_time
                last_step = step_count
    except KeyboardInterrupt:
        logger.log_info("\nStopping simulation...")
    finally:
        sim.destroy()
        logger.log_info("Simulation terminated successfully")


if __name__ == "__main__":
    main()
