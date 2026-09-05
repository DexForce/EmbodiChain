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
"""Control a UR10 end effector with a Gizmo and manual physics stepping."""

from __future__ import annotations

import time
import torch
import numpy as np
import argparse

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.objects import GizmoCfg
from embodichain.lab.visualization import visualization_cfg_from_args
from embodichain.lab.gym.utils.gym_utils import add_env_launcher_args_to_parser
from embodichain.lab.sim.cfg import (
    RenderCfg,
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
    add_env_launcher_args_to_parser(parser)
    args = parser.parse_args()

    # Configure the simulation
    sim_cfg = SimulationManagerCfg(
        width=1920,
        height=1080,
        headless=True,
        physics_dt=1.0 / 100.0,
        sim_device=args.device,
        render_cfg=RenderCfg(renderer=args.renderer),
        visualization=visualization_cfg_from_args(args),
        robot_ik_gizmo=GizmoCfg(ik_start_enabled=True),
    )

    sim = SimulationManager(sim_cfg)
    sim.set_manual_update(True)

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
    if sim.is_use_gpu_physics:
        sim.init_gpu_physics()

    # Set initial joint positions
    initial_qpos = torch.tensor(
        [[0, -np.pi / 2, np.pi / 2, 0.0, np.pi / 2, 0.0]],
        dtype=torch.float32,
        device=sim.device,
    )
    joint_ids = robot.get_joint_ids("arm")
    robot.set_qpos(qpos=initial_qpos, joint_ids=joint_ids, target=False)
    robot.set_qpos(qpos=initial_qpos, joint_ids=joint_ids)

    sim.update(step=1)  # Refresh link poses before creating the IK target.

    native_window_opened = False
    if not args.headless:
        native_window_opened = sim.open_window()

    if not native_window_opened and not args.viser:
        logger.log_warning(
            "Gizmo interaction is disabled in headless mode without Viser."
        )

    logger.log_info("Gizmo-Robot example started!")
    if native_window_opened or args.viser:
        logger.log_info("Use the gizmo to drag the robot end-effector (EE)")
    if native_window_opened:
        logger.log_info(
            "Native robot IK Gizmo starts enabled; press I to show or hide it"
        )
    logger.log_info("Press Ctrl+C to stop the simulation")

    run_simulation(sim)


def run_simulation(sim: SimulationManager) -> None:
    """Advance physics; the manager owns native and Viser IK interaction."""
    step_count = 0
    try:
        last_time = time.perf_counter()
        last_step = 0
        while True:
            frame_start = time.perf_counter()
            # update() owns IK interaction, physics stepping, and Viser capture.
            sim.update(step=1)
            step_count += 1

            if step_count % 100 == 0:
                current_time = time.perf_counter()
                elapsed = current_time - last_time
                fps = (
                    sim.num_envs * (step_count - last_step) / elapsed
                    if elapsed > 0
                    else 0
                )
                logger.log_info(f"Simulation step: {step_count}, FPS: {fps:.2f}")
                last_time = current_time
                last_step = step_count

            elapsed = time.perf_counter() - frame_start
            time.sleep(max(0.0, sim.sim_config.physics_dt - elapsed))
    except KeyboardInterrupt:
        logger.log_info("\nStopping simulation...")
    finally:
        sim.destroy()
        logger.log_info("Simulation terminated successfully")


if __name__ == "__main__":
    main()
