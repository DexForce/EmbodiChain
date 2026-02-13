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
and performs an open drawer task in a simulated environment.
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
import time


def parse_arguments():
    """
    Parse command-line arguments to configure the simulation.

    Returns:
        argparse.Namespace: Parsed arguments including rendering options and cabinet path.
    """
    parser = argparse.ArgumentParser(
        description="Create and simulate a robot to open drawer in SimulationManager"
    )
    parser.add_argument(
        "--enable_rt", action="store_true", help="Enable ray tracing rendering"
    )
    parser.add_argument(
        "--cabinet_path", type=str, default="", help="Path to cabinet URDF file"
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
        enable_rt=True,
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
    gripper_urdf_path = get_data_path("DH_PGI_140_80/DH_PGI_140_80.urdf")

    gripper_attach_xpos = np.eye(4)
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
            control_parts={
                "arm": ["JOINT[0-9]"],
                "hand": ["GRIPPER_FINGER[1-2]_JOINT_1"],
            },
            drive_pros=JointDrivePropertiesCfg(
                stiffness={"JOINT[0-9]": 1e4, "GRIPPER_FINGER[1-2]_JOINT_1": 1e4},
                damping={"JOINT[0-9]": 1e3, "GRIPPER_FINGER[1-2]_JOINT_1": 1e3},
                max_effort={"JOINT[0-9]": 1e5, "GRIPPER_FINGER[1-2]_JOINT_1": 1e3},
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


def create_cabinet(
    sim: SimulationManager,
    cabinet_urdf_path: str,
    cabinet_posi: list[float],
    cabinet_rota: list[float],
) -> Articulation:
    """
    Create a cabinet articulated object in the simulation.

    Args:
        sim (SimulationManager): The simulation manager instance.
        cabinet_path (str, optional): Path to cabinet URDF file. If None, uses default path.

    Returns:
        Articulation: The cabinet object added to the simulation.
    """
    cabinet = sim.add_articulation(
        cfg=ArticulationCfg(
            uid="cabinet",
            fpath=cabinet_urdf_path,
            init_pos=cabinet_posi,
            init_rot=cabinet_rota,
            init_qpos=[0, 0, 0],
            drive_pros=JointDrivePropertiesCfg(
                stiffness=1e-2, damping=1e-1, max_effort=1e-1, drive_type="force"
            ),
            fix_base=True,
            body_scale=[0.75, 0.75, 0.75],
        )
    )

    return cabinet


def open_drawer(
    sim: SimulationManager,
    robot: Robot,
    cabinet: Articulation,
    grasp_relative_posi: list[float],
):
    """
    Execute the trajectory to drive the robot to open the cabinet drawer.

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

    # grasp drawer handle waypoint generation
    grasp_xpos = arm_start_xpos.clone()
    grasp_posi = cabinet_position + torch.tensor(grasp_relative_posi, device=sim.device)
    grasp_xpos[:, :3, 3] = grasp_posi
    approach_xpos = grasp_xpos.clone()
    approach_xpos[:, 0, 3] -= 0.3
    pull_xpos = grasp_xpos.clone()
    pull_xpos[:, 0, 3] -= 0.5
    open_qpos = torch.tensor(
        [
            [0.0, 0.0],
        ],
        device=sim.device,
    )
    close_qpos = torch.tensor(
        [
            [0.08, 0.08],
        ],
        device=sim.device,
    )

    # compute ik for each waypoint
    _, approach_qpos = robot.compute_ik(
        approach_xpos, joint_seed=arm_start_qpos, name="arm"
    )
    _, grasp_qpos = robot.compute_ik(grasp_xpos, joint_seed=approach_qpos, name="arm")
    _, pull_qpos = robot.compute_ik(pull_xpos, joint_seed=grasp_qpos, name="arm")

    arm_traj = torch.concatenate(
        [
            arm_start_qpos,
            approach_qpos,
            grasp_qpos,
            grasp_qpos,
            approach_qpos,
            approach_qpos,
            pull_qpos,
        ]
    )

    hand_traj = torch.concatenate(
        [
            open_qpos,
            open_qpos,
            open_qpos,
            close_qpos,
            close_qpos,
            open_qpos,
            open_qpos,
        ]
    )

    all_trajectory = torch.hstack([arm_traj, hand_traj])

    n_interp = 400
    interp_trajectory = interpolate_with_distance_warp(
        trajectory=all_trajectory[None, :, :],
        interp_num=n_interp,
        device=sim.device,
    )
    # from IPython import embed; embed()
    for i in range(n_interp):
        robot.set_qpos(interp_trajectory[:, i, :]),
        sim.update(step=5)
        time.sleep(0.01)

    from IPython import embed

    embed()


def move_circular(robot: Robot, sim: SimulationManager):

    qpos_start = robot.get_qpos()
    qpos_end = qpos_start.clone()
    qpos_end[:, 0] -= 0.5

    all_trajectory = torch.concatenate([qpos_start, qpos_end])

    n_interp = 400
    interp_trajectory = interpolate_with_distance_warp(
        trajectory=all_trajectory[None, :, :],
        interp_num=n_interp,
        device=sim.device,
    )
    # from IPython import embed; embed()
    for i in range(n_interp):
        robot.set_qpos(interp_trajectory[:, i, :]),
        sim.update(step=5)
        time.sleep(0.01)


if __name__ == "__main__":

    # table 0
    cabinet_posi = [1.4, 0.0, 0.7]
    cabinet_rpy = [90, -93, 0]
    grasp_relative_posi = [-0.59, 0.00, -0.015]
    cabinet_path = (
        "/home/dex/Downloads/demo_assets/cabinet2/obj.urdf"
    )

    # # table 1
    # cabinet_posi = [1.55, 0.2, 0.55]
    # cabinet_rpy = [90, -90, 0]
    # grasp_relative_posi = [-0.68, -0.6, 0.055]
    # cabinet_path = (
    #     "/home/dex/Downloads/demo_assets/table/obj.urdf"
    # )

    # # # table 2
    # cabinet_posi = [1.6, -0.2, 0.55]
    # cabinet_rpy = [90, -90, 0]
    # grasp_relative_posi = [-0.65, 0.0, 0.12]
    # cabinet_path = (
    #     "/home/dex/Downloads/demo_assets/cabinet/obj.urdf"
    # )

    # cabinet_posi = [1.4, -0.2, 0.85]
    # cabinet_rpy = [90, -65, 0]
    # grasp_relative_posi = [-0.65, 0.0, 0.12]
    # cabinet_path = "/home/dex/Downloads/demo_assets/drawer/obj.urdf"

    args = parse_arguments()
    sim = initialize_simulation(args)
    robot = create_robot(sim)
    cabinet = create_cabinet(sim, cabinet_path, cabinet_posi, cabinet_rpy)
    # cabinet._entities[0].create_physical_visible_node(np.array([0.9, 0.1, 0.1, 0.9]))

    if not args.headless:
        sim.open_window()

    # cabinet.set_qpos(torch.tensor([[0.3, 0]]))

    # sim.update(step=5000)
    from IPython import embed

    embed()
    # move_circular(robot, sim)
    sim.update(step=100)
    open_drawer(sim, robot, cabinet, grasp_relative_posi)

    logger.log_info("\n Press Ctrl+C to exit simulation loop.")
    try:
        while True:
            sim.update(step=10)
    except KeyboardInterrupt:
        logger.log_info("\n Exit")
