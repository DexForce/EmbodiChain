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
Gizmo-Robot Example: Test Gizmo class on a robot (UR10)
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from embodichain.lab.sim import SimulationManager
from embodichain.lab.sim.solvers import PytorchSolverCfg
from embodichain.lab.sim.cfg import (
    JointDrivePropertiesCfg,
    RobotCfg,
    URDFCfg,
)
from embodichain.data import get_data_path
from embodichain.lab.sim.utility.demo_utils import (
    add_demo_args,
    create_default_sim,
    maybe_open_window,
    resolve_demo_steps,
    shutdown_sim,
)
from embodichain.utils import logger


def main() -> None:
    """Main function to create and run the simulation scene."""

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Create a simulation scene with SimulationManager"
    )
    add_demo_args(parser)
    args = parser.parse_args()

    sim = create_default_sim(
        args,
        width=1920,
        height=1080,
        physics_dt=1.0 / 100.0,
        add_default_light=False,
    )
    sim.set_manual_update(False)

    # Get UR10 URDF path
    ur10_urdf_path = get_data_path("UniversalRobots/UR10/UR10.urdf")
    gripper_urdf_path = get_data_path("DH_PGC_140_50_M/DH_PGC_140_50_M.urdf")

    # Create UR10 robot
    robot_cfg = RobotCfg(
        uid="ur10_gizmo_test",
        urdf_cfg=URDFCfg(
            components=[
                {"component_type": "arm", "urdf_path": ur10_urdf_path},
                {"component_type": "hand", "urdf_path": gripper_urdf_path},
            ]
        ),
        control_parts={
            "arm": ["JOINT[0-9]"],
            "hand": ["FINGER[1-2]"],
        },
        solver_cfg={
            "arm": PytorchSolverCfg(
                end_link_name="ee_link",
                root_link_name="base_link",
                tcp=[
                    [0.0, 1.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.12],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                num_samples=30,
            )
        },
        drive_pros=JointDrivePropertiesCfg(
            stiffness={"JOINT[0-9]": 1e4, "FINGER[1-2]": 1e2},
            damping={"JOINT[0-9]": 1e3, "FINGER[1-2]": 1e1},
            max_effort={"JOINT[0-9]": 1e5, "FINGER[1-2]": 1e3},
            drive_type="force",
        ),
        init_qpos=[0.0, -np.pi / 2, -np.pi / 2, np.pi / 2, -np.pi / 2, 0.0, 0.0, 0.0],
    )
    robot = sim.add_robot(cfg=robot_cfg)

    # Set initial joint positions
    initial_qpos = torch.tensor(
        [[0.0, -np.pi / 2, -np.pi / 2, np.pi / 2, -np.pi / 2, 0.0]],
        dtype=torch.float32,
        device=robot.device,
    )
    joint_ids = robot.get_joint_ids("arm")
    robot.set_qpos(qpos=initial_qpos, joint_ids=joint_ids)

    time.sleep(0.2)  # Wait for a moment to ensure everything is set up

    # Enable gizmo using the new API
    if not args.headless:
        sim.enable_gizmo(uid="ur10_gizmo_test", control_part="arm")
        if not sim.has_gizmo("ur10_gizmo_test", control_part="arm"):
            shutdown_sim(sim)
            raise RuntimeError("Failed to enable gizmo.")

    maybe_open_window(sim, args)

    logger.log_info("Gizmo-Robot example started!")
    logger.log_info("Use the gizmo to drag the robot end-effector (EE)")
    logger.log_info("Press Ctrl+C to stop the simulation")

    run_simulation(sim, max_steps=resolve_demo_steps(args))


def run_simulation(sim: SimulationManager, max_steps: int | None = None) -> None:
    step_count = 0
    try:
        last_time = time.time()
        last_step = 0
        while max_steps is None or step_count < max_steps:
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
        shutdown_sim(sim)
        logger.log_info("Simulation terminated successfully")


if __name__ == "__main__":
    main()
