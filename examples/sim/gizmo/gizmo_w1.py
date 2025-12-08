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

    # Build multiple arenas if requested
    if args.num_envs > 1:
        sim.build_multiple_arenas(args.num_envs, space=3.0)

    # Get DexForce W1 URDF path
    urdf_path = get_data_path(
        "DexforceW1V021_INDUSTRIAL_DH_PGC_GRIPPER_M/DexforceW1V021.urdf"
    )

    # Create DexForce W1 robot
    robot_cfg = RobotCfg(
        uid="w1_gizmo_test",
        urdf_cfg=URDFCfg(
            components=[{"component_type": "humanoid", "urdf_path": urdf_path}]
        ),
        control_parts={"left_arm": ["LEFT_J[1-7]"], "right_arm": ["RIGHT_J[1-7]"]},
        solver_cfg={
            "left_arm": PinkSolverCfg(
                urdf_path=urdf_path,
                end_link_name="left_ee",
                root_link_name="left_arm_base",
                pos_eps=1e-2,
                rot_eps=5e-2,
                max_iterations=300,
                dt=0.1,
            ),
            "right_arm": PinkSolverCfg(
                urdf_path=urdf_path,
                end_link_name="right_ee",
                root_link_name="right_arm_base",
                pos_eps=1e-2,
                rot_eps=5e-2,
                max_iterations=300,
                dt=0.1,
            ),
        },
        drive_pros=JointDrivePropertiesCfg(
            stiffness={"(LEFT|RIGHT)_J[1-7]": 1e4, "(ANKLE|KNEE|BUTTOCK|WAIST)": 1e7},
            damping={"(LEFT|RIGHT)_J[1-7]": 1e3, "(ANKLE|KNEE|BUTTOCK|WAIST)": 1e4},
            max_effort={"(LEFT|RIGHT)_J[1-7]": 1e5, "(ANKLE|KNEE|BUTTOCK|WAIST)": 1e10},
        ),
    )
    robot = sim.add_robot(cfg=robot_cfg)

    # Set initial joint positions for both arms
    # Left arm: 8 joints (WAIST + 7 LEFT_J), Right arm: 8 joints (WAIST + 7 RIGHT_J)
    left_arm_qpos = torch.tensor(
        [
            [0, 0, -np.pi / 4, np.pi / 4, -np.pi / 2, 0.0, np.pi / 4, 0.0]
        ],  # WAIST + LEFT_J[1-7]
        dtype=torch.float32,
        device="cpu",
    )
    right_arm_qpos = torch.tensor(
        [
            [0, 0, np.pi / 4, -np.pi / 4, np.pi / 2, 0.0, -np.pi / 4, 0.0]
        ],  # WAIST + RIGHT_J[1-7]
        dtype=torch.float32,
        device="cpu",
    )

    left_joint_ids = robot.get_joint_ids("left_arm")
    right_joint_ids = robot.get_joint_ids("right_arm")

    robot.set_qpos(qpos=left_arm_qpos, joint_ids=left_joint_ids)
    robot.set_qpos(qpos=right_arm_qpos, joint_ids=right_joint_ids)

    time.sleep(0.2)  # Wait for a moment to ensure everything is set up

    # Enable gizmo for both arms using the new API
    sim.enable_gizmo(uid="w1_gizmo_test", control_part="left_arm")
    if not sim.has_gizmo("w1_gizmo_test", control_part="left_arm"):
        logger.log_error("Failed to enable left arm gizmo!")
        return

    sim.enable_gizmo(uid="w1_gizmo_test", control_part="right_arm")
    if not sim.has_gizmo("w1_gizmo_test", control_part="right_arm"):
        logger.log_error("Failed to enable right arm gizmo!")
        return

    sim.open_window()

    logger.log_info("Gizmo-DexForce W1 example started!")
    logger.log_info("Use the gizmos to drag both robot arms' end-effectors")
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
