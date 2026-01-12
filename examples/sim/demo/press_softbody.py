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
This script demonstrates the creation and simulation of a robot with a soft object,
and performs a pressing task in a simulated environment.
"""

import argparse
import numpy as np
import time
import torch

from dexsim.utility.path import get_resources_data_path

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.objects import Robot, SoftObject
from embodichain.lab.sim.utility.action_utils import interpolate_with_distance_warp
from embodichain.lab.sim.shapes import MeshCfg
from embodichain.lab.sim.solvers import PytorchSolverCfg
from embodichain.data import get_data_path
from embodichain.utils import logger
from embodichain.lab.sim.cfg import (
    RobotCfg,
    LightCfg,
    SoftObjectCfg,
    SoftbodyVoxelAttributesCfg,
    SoftbodyPhysicalAttributesCfg,
    URDFCfg,
)
from embodichain.lab.sim.shapes import MeshCfg


def parse_arguments():
    """
    Parse command-line arguments to configure the simulation.

    Returns:
        argparse.Namespace: Parsed arguments including number of environments, device, and rendering options.
    """
    parser = argparse.ArgumentParser(
        description="Create and simulate a robot in SimulationManager"
    )
    parser.add_argument(
        "--enable_rt", action="store_true", help="Enable ray tracing rendering"
    )
    return parser.parse_args()


def initialize_simulation(args):
    """
    Initialize the simulation environment based on the provided arguments.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        SimulationManager: Configured simulation manager instance.
    """
    config = SimulationManagerCfg(
        headless=True,
        sim_device="cuda",
        enable_rt=args.enable_rt,
        physics_dt=1.0 / 100.0,
    )
    sim = SimulationManager(config)

    light = sim.add_light(
        cfg=LightCfg(uid="main_light", intensity=50.0, init_pos=(0, 0, 2.0))
    )

    return sim


def create_robot(sim: SimulationManager):
    """
    Create and configure a robot with an arm and a dexterous hand in the simulation.

    Args:
        sim (SimulationManager): The simulation manager instance.

    Returns:
        Robot: The configured robot instance added to the simulation.
    """
    # Retrieve URDF paths for the robot arm and hand
    ur10_urdf_path = get_data_path("UniversalRobots/UR10/UR10.urdf")

    # Configure the robot with its components and control properties
    cfg = RobotCfg(
        uid="UR10",
        urdf_cfg=URDFCfg(
            components=[{"component_type": "arm", "urdf_path": ur10_urdf_path}]
        ),
        control_parts={
            "arm": ["Joint[0-9]"],
        },
        solver_cfg={
            "arm": PytorchSolverCfg(
                end_link_name="ee_link",
                root_link_name="base_link",
                tcp=np.eye(4),
            )
        },
        init_qpos=[
            0.0,
            -np.pi / 2,
            -np.pi / 2,
            np.pi / 2,
            -np.pi / 2,
            0.0,
        ],
    )
    return sim.add_robot(cfg=cfg)


def create_soft_cow(sim: SimulationManager) -> SoftObject:
    """create soft cow object in the simulation

    Args:
        sim (SimulationManager): The simulation manager instance.

    Returns:
        SoftObject: soft cow object
    """
    cow: SoftObject = sim.add_soft_object(
        cfg=SoftObjectCfg(
            uid="cow",
            shape=MeshCfg(
                fpath=get_resources_data_path("Model", "cow", "cow2.obj"),
            ),
            init_rot=[0, 90, 0],
            init_pos=[0.45, -0.1, 0.12],
            voxel_attr=SoftbodyVoxelAttributesCfg(
                simulation_mesh_resolution=8,
                maximal_edge_length=0.5,
            ),
            physical_attr=SoftbodyPhysicalAttributesCfg(
                youngs=5e3,
                poissons=0.45,
                density=100,
                dynamic_friction=0.1,
            ),
        ),
    )
    return cow


def press_cow(sim: SimulationManager, robot: Robot):
    """robot press cow softbody with its end link

    Args:
        sim (SimulationManager): The simulation manager instance.
        robot (Robot): The robot instance to be controlled.
    """
    start_qpos = robot.get_qpos()
    arm_ids = robot.get_joint_ids("arm")
    arm_start_qpos = start_qpos[:, arm_ids]

    arm_start_xpos = robot.compute_fk(arm_start_qpos, name="arm", to_matrix=True)
    press_xpos = arm_start_xpos.clone()
    press_xpos[:, :3, 3] = torch.tensor([0.5, -0.1, 0.005], device=press_xpos.device)

    approach_xpos = press_xpos.clone()
    approach_xpos[:, 2, 3] += 0.05

    is_success, approach_qpos = robot.compute_ik(
        approach_xpos, joint_seed=arm_start_qpos, name="arm"
    )
    is_success, press_qpos = robot.compute_ik(
        approach_xpos, joint_seed=arm_start_qpos, name="arm"
    )

    arm_trajectory = torch.concatenate([arm_start_qpos, approach_qpos, press_qpos])
    interp_trajectory = interpolate_with_distance_warp(
        trajectory=arm_trajectory[None, :, :], interp_num=50, device=sim.device
    )
    interp_trajectory = interp_trajectory[0]
    for qpos in interp_trajectory:
        robot.set_qpos(qpos.unsqueeze(0), joint_ids=arm_ids)
        sim.update(step=5)


def main():
    """
    Main function to demonstrate robot simulation.

    This function initializes the simulation, creates the robot and other objects,
    and performs the press softbody task.
    """
    args = parse_arguments()
    sim = initialize_simulation(args)

    robot = create_robot(sim)
    soft_cow = create_soft_cow(sim)
    sim.init_gpu_physics()
    sim.open_window()

    press_cow(sim, robot)

    logger.log_info("\n Press Ctrl+C to exit simulation loop.")
    try:
        while True:
            sim.update(step=10)
    except KeyboardInterrupt:
        logger.log_info("\n Exit")


if __name__ == "__main__":
    main()
