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
This script demonstrates the creation and simulation of UR10 robot with gripper,
and performs an open cabinet task in a simulated environment.
"""

import argparse
import os
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.objects import Robot, Articulation
from embodichain.lab.sim.utility.action_utils import interpolate_with_distance_warp
from embodichain.lab.sim.solvers import PytorchSolverCfg
from embodichain.data import get_data_path
from embodichain.utils import logger
from embodichain.lab.sim.cfg import (
    RobotCfg,
    LightCfg,
    ArticulationCfg,
    JointDrivePropertiesCfg,
    RigidBodyAttributesCfg,
    URDFCfg,
)


def parse_arguments():
    """
    Parse command-line arguments to configure the simulation.

    Returns:
        argparse.Namespace: Parsed arguments including rendering options and cabinet path.
    """
    parser = argparse.ArgumentParser(
        description="Create and simulate a robot to open cabinet in SimulationManager"
    )
    parser.add_argument(
        "--enable_rt", action="store_true", help="Enable ray tracing rendering"
    )
    parser.add_argument(
        "--cabinet_path", type=str, default=None, help="Path to cabinet URDF file"
    )
    parser.add_argument("--headless", action="store_true", help="Enable headless mode")
    return parser.parse_args()


def initialize_simulation(args) -> SimulationManager:
    """
    Initialize the simulation environment based on the provided arguments.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        SimulationManager: Configured simulation manager instance.
    """
    config = SimulationManagerCfg(
        headless=args.headless,
        sim_device="cpu",
        enable_rt=args.enable_rt,
        physics_dt=1.0 / 100.0,
    )
    sim = SimulationManager(config)
    sim.add_light(cfg=LightCfg(uid="main_light", intensity=50.0, init_pos=(0, 0, 2.0)))
    return sim


def create_robot(sim: SimulationManager) -> Robot:
    """
    Create and configure a robot with an arm and a gripper in the simulation.

    Args:
        sim (SimulationManager): The simulation manager instance.

    Returns:
        Robot: The configured robot instance added to the simulation.
    """
    ur10_urdf_path = get_data_path("UniversalRobots/UR10/UR10.urdf")
    gripper_urdf_path = get_data_path("DH_PGC_140_50_M/DH_PGC_140_50_M.urdf")

    gripper_attach_xpos = np.eye(4)
    # rotate gripper 90 degrees around Z axis for installation orientation
    gripper_attach_xpos[:3, :3] = R.from_rotvec([0, 0, 90], degrees=True).as_matrix()

    return sim.add_robot(
        cfg=RobotCfg(
            uid="UR10_with_gripper",
            urdf_cfg=URDFCfg(
                components=[
                    {"component_type": "arm", "urdf_path": ur10_urdf_path},
                    {
                        "component_type": "hand",
                        "urdf_path": gripper_urdf_path,
                        "transform": gripper_attach_xpos,
                    },
                ]
            ),
            control_parts={"arm": ["JOINT[0-9]"], "hand": ["FINGER[1-2]"]},
            drive_pros=JointDrivePropertiesCfg(
                stiffness={"JOINT[0-9]": 1e4, "FINGER[1-2]": 1e1},
                damping={"JOINT[0-9]": 1e3, "FINGER[1-2]": 1e0},
                max_effort={"JOINT[0-9]": 1e5, "FINGER[1-2]": 1e2},
                drive_type="force",
            ),
            solver_cfg={
                "arm": PytorchSolverCfg(
                    end_link_name="ee_link", root_link_name="base_link", tcp=np.eye(4)
                )
            },
            init_qpos=[0.0, -np.pi / 2, -np.pi / 2, 0.0, np.pi / 2, 0.0, 0.0, 0.0],
            init_pos=[0.0, 0.0, 0.0],
        )
    )


def create_cabinet(sim: SimulationManager, cabinet_path: str = None) -> Articulation:
    """
    Create a cabinet articulated object in the simulation.

    Args:
        sim (SimulationManager): The simulation manager instance.
        cabinet_path (str, optional): Path to cabinet URDF file. If None, uses default path.

    Returns:
        Articulation: The cabinet object added to the simulation.
    """

    cabinet_urdf_path = "/home/dex/桌面/forvisualization_examples_PM_datasets/StorageFurniture@45661@18/object_fromJson2urdf_decomposed.urdf"

    return sim.add_articulation(
        cfg=ArticulationCfg(
            uid="cabinet",
            fpath=cabinet_urdf_path,
            init_pos=[1.55, 0.0, 0.425],
            init_rot=[90, -90, 0],
            body_scale=(0.5, 0.5, 0.5),
            attrs=RigidBodyAttributesCfg(mass=1.0),
            drive_pros=JointDrivePropertiesCfg(
                stiffness=1e-3, damping=1e-4, max_effort=1e-2, drive_type="force"
            ),
            init_qpos=[0.0, 0.0, 0.0],
            fix_base=True,
        )
    )


def open_cabinet(sim: SimulationManager, robot: Robot, cabinet: Articulation):
    """
    Execute the trajectory to drive the robot to open the cabinet.

    Args:
        sim (SimulationManager): The simulation manager instance.
        robot (Robot): The robot instance.
        cabinet (Articulation): The cabinet object.
    """
    arm_ids = robot.get_joint_ids("arm")
    gripper_ids = robot.get_joint_ids("hand")
    arm_start_qpos = robot.get_qpos()[:, arm_ids]

    joint_names = cabinet.joint_names
    drawer_joint_id = next(
        (i for i, name in enumerate(joint_names) if "drawer" in name.lower()),
        len(joint_names) - 1,
    )

    if drawer_joint_id is not None:
        cabinet_qpos = cabinet.get_qpos()
        cabinet_qpos[:, drawer_joint_id] = 0.0
        cabinet.set_qpos(cabinet_qpos, target=False)
        sim.update(step=20)

    cabinet_position = cabinet.get_local_pose(to_matrix=True)[:, :3, 3]
    arm_start_xpos = robot.compute_fk(arm_start_qpos, name="arm", to_matrix=True)

    # grasp cabinet handle waypoint generation
    grasp_xpos = arm_start_xpos.clone()
    grasp_pos = cabinet_position + torch.tensor([-0.39, -0.16, -0.1], device=sim.device)
    grasp_xpos[:, :3, 3] = grasp_pos
    pull_xpos = grasp_xpos.clone()
    pull_xpos[:, 0, 3] -= 0.25

    # compute ik for each waypoint
    _, grasp_qpos = robot.compute_ik(grasp_xpos, joint_seed=arm_start_qpos, name="arm")
    _, pull_qpos = robot.compute_ik(pull_xpos, joint_seed=grasp_qpos, name="arm")

    approach_trajectory = interpolate_with_distance_warp(
        trajectory=torch.stack([arm_start_qpos[0], grasp_qpos[0]])[None, :, :],
        interp_num=50,
        device=sim.device,
    )[0]

    pull_trajectory = interpolate_with_distance_warp(
        trajectory=torch.stack([grasp_qpos[0], pull_qpos[0]])[None, :, :],
        interp_num=50,
        device=sim.device,
    )[0]

    retract_trajectory = interpolate_with_distance_warp(
        trajectory=torch.stack([pull_qpos[0], arm_start_qpos[0]])[None, :, :],
        interp_num=50,
        device=sim.device,
    )[0]

    # execute approach trajectory
    for qpos in approach_trajectory:
        robot.set_qpos(qpos.unsqueeze(0), joint_ids=arm_ids)
        sim.update(step=20)

    # close gripper
    robot.set_qpos(
        torch.tensor([0.025, 0.025], device=sim.device)
        .unsqueeze(0)
        .repeat(sim.num_envs, 1),
        joint_ids=gripper_ids,
    )
    sim.update(step=100)

    # execute pull trajectory
    for qpos in pull_trajectory:
        robot.set_qpos(qpos.unsqueeze(0), joint_ids=arm_ids, target=False)
        sim.update(step=20)

    # open gripper
    robot.set_qpos(
        torch.tensor([0.0, 0.0], device=sim.device)
        .unsqueeze(0)
        .repeat(sim.num_envs, 1),
        joint_ids=gripper_ids,
    )
    sim.update(step=100)

    # execute retract trajectory
    for qpos in retract_trajectory:
        robot.set_qpos(qpos.unsqueeze(0), joint_ids=arm_ids, target=False)
        sim.update(step=20)


def main():
    """
    Main function to demonstrate robot simulation.

    Initializes the simulation, creates the robot and cabinet, and performs the open cabinet task.
    """
    args = parse_arguments()
    sim = initialize_simulation(args)
    robot = create_robot(sim)
    cabinet = create_cabinet(sim, args.cabinet_path)

    sim.init_gpu_physics()
    if not args.headless:
        sim.open_window()

    sim.update(step=100)
    open_cabinet(sim, robot, cabinet)

    logger.log_info("\n Press Ctrl+C to exit simulation loop.")
    try:
        while True:
            sim.update(step=10)
    except KeyboardInterrupt:
        logger.log_info("\n Exit")


if __name__ == "__main__":
    main()
