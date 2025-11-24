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
This script demonstrates the creation and simulation of dexforce w1 robot,
and performs a grasp cup to coffee machine task in a simulated environment.
"""

import argparse
import numpy as np
import torch
from tqdm import tqdm
from typing import Union
from scipy.spatial.transform import Rotation as R
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.objects import Robot, RigidObject
from embodichain.lab.sim.cfg import (
    LightCfg,
    JointDrivePropertiesCfg,
    RigidObjectCfg,
    RigidBodyAttributesCfg,
    ArticulationCfg,
)
from embodichain.lab.sim.utility.action_utils import interpolate_with_distance_warp
from embodichain.lab.sim.shapes import MeshCfg
from embodichain.data import get_data_path
from embodichain.utils import logger

from embodichain.lab.sim.robots.dexforce_w1.cfg import DexforceW1Cfg


def parse_arguments():
    """
    Parse command-line arguments to configure the simulation.

    Returns:
        argparse.Namespace: Parsed arguments including number of environments and rendering options.
    """
    parser = argparse.ArgumentParser(
        description="Create and simulate a robot in SimulationManager"
    )
    parser.add_argument(
        "--num_envs", type=int, default=9, help="Number of parallel environments"
    )
    parser.add_argument(
        "--enable_rt", action="store_true", help="Enable ray tracing rendering"
    )
    parser.add_argument("--headless", action="store_true", help="Enable headless mode")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="device to run the environment on, e.g., 'cpu' or 'cuda'",
    )
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
        headless=True,
        sim_device=args.device,
        enable_rt=args.enable_rt,
        physics_dt=1.0 / 100.0,
    )
    sim = SimulationManager(config)

    sim.build_multiple_arenas(args.num_envs, space=2.5)

    if args.enable_rt:
        light = sim.add_light(
            cfg=LightCfg(
                uid="main_light",
                color=(0.6, 0.6, 0.6),
                intensity=30.0,
                init_pos=(1.0, 0, 3.0),
            )
        )

    return sim


def create_robot(sim: SimulationManager) -> Robot:
    """
    Create and configure a robot with an arm and a dexterous hand in the simulation.

    Args:
        sim (SimulationManager): The simulation manager instance.

    Returns:
        Robot: The configured robot instance added to the simulation.
    """
    cfg = DexforceW1Cfg.from_dict(
        {
            "uid": "dexforce_w1",
            "init_pos": [0.4, -0.5, 0.0],
        }
    )
    cfg.solver_cfg["left_arm"].tcp = np.array(
        [
            [1.0, 0.0, 0.0, 0.012],
            [0.0, 1.0, 0.0, 0.04],
            [0.0, 0.0, 1.0, 0.11],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    cfg.solver_cfg["right_arm"].tcp = np.array(
        [
            [1.0, 0.0, 0.0, 0.012],
            [0.0, 1.0, 0.0, -0.04],
            [0.0, 0.0, 1.0, 0.11],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    cfg.init_qpos = [
        1.0000e00,
        -2.0000e00,
        1.0000e00,
        0.0000e00,
        -2.6921e-05,
        -2.6514e-03,
        -1.5708e00,
        1.4575e00,
        -7.8540e-01,
        1.2834e-01,
        1.5708e00,
        -2.2310e00,
        -7.8540e-01,
        1.4461e00,
        -1.5708e00,
        1.6716e00,
        7.8540e-01,
        7.6745e-01,
        0.0000e00,
        3.8108e-01,
        0.0000e00,
        0.0000e00,
        0.0000e00,
        0.0000e00,
        1.5000e00,
        0.0000e00,
        0.0000e00,
        0.0000e00,
        0.0000e00,
        1.5000e00,
        6.9974e-02,
        7.3950e-02,
        6.6574e-02,
        6.0923e-02,
        0.0000e00,
        6.7342e-02,
        7.0862e-02,
        6.3684e-02,
        5.7822e-02,
        0.0000e00,
    ]
    return sim.add_robot(cfg=cfg)


def create_table(sim: SimulationManager) -> RigidObject:
    """
    Create a table rigid object in the simulation.

    Args:
        sim (SimulationManager): The simulation manager instance.

    Returns:
        RigidObject: The table object added to the simulation.
    """
    scoop_cfg = RigidObjectCfg(
        uid="table",
        shape=MeshCfg(
            fpath=get_data_path("MultiW1Data/table_a.obj"),
        ),
        attrs=RigidBodyAttributesCfg(
            mass=0.5,
        ),
        max_convex_hull_num=8,
        body_type="kinematic",
        init_pos=[1.1, -0.5, 0.08],
        init_rot=[0.0, 0.0, 0.0],
    )
    scoop = sim.add_rigid_object(cfg=scoop_cfg)
    return scoop


def create_caffe(sim: SimulationManager) -> Robot:
    """
    Create a caffe (container) articulated object in the simulation.

    Args:
        sim (SimulationManager): The simulation manager instance.

    Returns:
        Robot: The caffe object added to the simulation.
    """
    container_cfg = ArticulationCfg(
        uid="caffe",
        fpath=get_data_path("MultiW1Data/cafe/cafe.urdf"),
        init_pos=[1.05, -0.5, 0.79],
        init_rot=[0, 0, -30],
        attrs=RigidBodyAttributesCfg(
            mass=1.0,
        ),
        drive_pros=JointDrivePropertiesCfg(
            stiffness=1.0, damping=0.1, max_effort=100.0, drive_type="force"
        ),
    )
    container = sim.add_articulation(cfg=container_cfg)
    return container


def create_cup(sim: SimulationManager) -> RigidObject:
    """
    Create a cup rigid object in the simulation.

    Args:
        sim (SimulationManager): The simulation manager instance.

    Returns:
        RigidObject: The cup object added to the simulation.
    """
    scoop_cfg = RigidObjectCfg(
        uid="cup",
        shape=MeshCfg(
            fpath=get_data_path("MultiW1Data/paper_cup_2.obj"),
        ),
        attrs=RigidBodyAttributesCfg(
            mass=0.3,
        ),
        max_convex_hull_num=1,
        body_type="dynamic",
        init_pos=[0.86, -0.76, 0.841],
        init_rot=[0.0, 0.0, 0.0],
    )
    scoop = sim.add_rigid_object(cfg=scoop_cfg)
    return scoop


def create_trajectory(
    sim: SimulationManager, robot: Robot, cup: RigidObject, caffe: Robot
) -> torch.Tensor:
    """
    Generate a trajectory for the right arm to grasp the cup and move it to the caffe.

    Args:
        sim (SimulationManager): The simulation manager instance.
        robot (Robot): The robot instance.
        cup (RigidObject): The cup object.
        caffe (Robot): The caffe object.

    Returns:
        torch.Tensor: Interpolated trajectory of shape [n_envs, n_waypoint, dof].
    """
    right_arm_ids = robot.get_joint_ids("right_arm")
    hand_open_qpos = torch.tensor(
        [0.0, 1.5, 0.0, 0.0, 0.0, 0.0],
        dtype=torch.float32,
        device=sim.device,
    )
    hand_close_qpos = torch.tensor(
        [0.1, 1.5, 0.3, 0.2, 0.3, 0.3],
        dtype=torch.float32,
        device=sim.device,
    )

    cup_position = cup.get_local_pose(to_matrix=True)[:, :3, 3]

    # grasp cup waypoint generation
    rest_right_qpos = robot.get_qpos()[:, right_arm_ids]  # [n_envs, dof]
    right_arm_xpos = robot.compute_fk(
        qpos=rest_right_qpos, name="right_arm", to_matrix=True
    )
    approach_cup_relative_position = torch.tensor(
        [-0.05, -0.06, 0.025], dtype=torch.float32, device=sim.device
    )
    pick_cup_relative_position = torch.tensor(
        [-0.03, -0.028, 0.021], dtype=torch.float32, device=sim.device
    )

    approach_xpos = right_arm_xpos.clone()
    approach_xpos[:, :3, 3] = cup_position + approach_cup_relative_position

    pick_xpos = right_arm_xpos.clone()
    pick_xpos[:, :3, 3] = cup_position + pick_cup_relative_position

    lift_xpos = pick_xpos.clone()
    lift_xpos[:, 2, 3] += 0.07

    # place cup to caffe waypoint generation
    caffe_position = caffe.get_local_pose(to_matrix=True)[:, :3, 3]
    place_cup_up_relative_position = torch.tensor(
        [-0.14, -0.18, 0.13], dtype=torch.float32, device=sim.device
    )
    place_cup_down_relative_position = torch.tensor(
        [-0.14, -0.18, 0.09], dtype=torch.float32, device=sim.device
    )

    place_cup_up_pose = lift_xpos.clone()
    place_cup_up_pose[:, :3, 3] = caffe_position + place_cup_up_relative_position
    place_down_pose = lift_xpos.clone()
    place_down_pose[:, :3, 3] = caffe_position + place_cup_down_relative_position
    # compute ik for each waypoint
    is_success, approach_qpos = robot.compute_ik(
        pose=approach_xpos, joint_seed=rest_right_qpos, name="right_arm"
    )
    is_success, pick_qpos = robot.compute_ik(
        pose=pick_xpos, joint_seed=approach_qpos, name="right_arm"
    )
    is_success, lift_qpos = robot.compute_ik(
        pose=lift_xpos, joint_seed=pick_qpos, name="right_arm"
    )
    is_success, place_up_qpos = robot.compute_ik(
        pose=place_cup_up_pose, joint_seed=lift_qpos, name="right_arm"
    )
    is_success, place_down_qpos = robot.compute_ik(
        pose=place_down_pose, joint_seed=place_up_qpos, name="right_arm"
    )

    n_envs = sim.num_envs

    # combine hand and arm trajectory
    arm_trajectory = torch.cat(
        [
            rest_right_qpos[:, None, :],
            approach_qpos[:, None, :],
            pick_qpos[:, None, :],
            pick_qpos[:, None, :],
            lift_qpos[:, None, :],
            place_up_qpos[:, None, :],
            place_down_qpos[:, None, :],
            place_down_qpos[:, None, :],
            lift_qpos[:, None, :],
            rest_right_qpos[:, None, :],
        ],
        dim=1,
    )
    hand_trajectory = torch.cat(
        [
            hand_open_qpos[None, None, :].repeat(n_envs, 1, 1),
            hand_open_qpos[None, None, :].repeat(n_envs, 1, 1),
            hand_open_qpos[None, None, :].repeat(n_envs, 1, 1),
            hand_close_qpos[None, None, :].repeat(n_envs, 1, 1),
            hand_close_qpos[None, None, :].repeat(n_envs, 1, 1),
            hand_close_qpos[None, None, :].repeat(n_envs, 1, 1),
            hand_close_qpos[None, None, :].repeat(n_envs, 1, 1),
            hand_open_qpos[None, None, :].repeat(n_envs, 1, 1),
            hand_open_qpos[None, None, :].repeat(n_envs, 1, 1),
            hand_open_qpos[None, None, :].repeat(n_envs, 1, 1),
        ],
        dim=1,
    )
    all_trajectory = torch.cat([arm_trajectory, hand_trajectory], dim=-1)
    # trajetory with shape [n_envs, n_waypoint, dof]
    interp_trajectory = interpolate_with_distance_warp(
        trajectory=all_trajectory, interp_num=150, device=sim.device
    )
    return interp_trajectory


def run_simulation(
    sim: SimulationManager, robot: Robot, cup: RigidObject, caffe: Robot
):
    """
    Execute the generated trajectory to drive the robot to complete the grasp and place task.

    Args:
        sim (SimulationManager): The simulation manager instance.
        robot (Robot): The robot instance.
        cup (RigidObject): The cup object.
        caffe (Robot): The caffe object.
    """
    # [n_envs, n_waypoint, dof]
    interp_trajectory = create_trajectory(sim, robot, cup, caffe)

    right_arm_ids = robot.get_joint_ids("right_arm")
    right_hand_ids = robot.get_joint_ids("right_eef")
    combine_ids = np.concatenate([right_arm_ids, right_hand_ids])
    n_waypoints = interp_trajectory.shape[1]
    logger.log_info(f"Executing trajectory...")
    for i in tqdm(range(n_waypoints)):
        robot.set_qpos(interp_trajectory[:, i, :], joint_ids=combine_ids)
        sim.update(step=10)


def apply_random_xy_perturbation(
    item: Union[RigidObject, Robot], max_perturbation: float = 0.02
):
    """
    Apply random perturbation to the object's XY position.

    Args:
        item (Union[RigidObject, Robot]): The object to perturb.
        max_perturbation (float): Maximum perturbation magnitude.
    """
    item_pose = item.get_local_pose(to_matrix=True)
    item_xy = item_pose[:, :2, 3].to("cpu").numpy()
    perturbation = np.random.uniform(
        low=-max_perturbation, high=max_perturbation, size=item_xy.shape
    )
    new_xy = item_xy + perturbation
    item_pose[:, :2, 3] = torch.tensor(
        new_xy, dtype=torch.float32, device=item_pose.device
    )
    item.set_local_pose(item_pose)


def main():
    """
    Main function to demonstrate robot simulation.

    Initializes the simulation, creates the robot and objects, and performs the grasp and place task.
    """
    args = parse_arguments()
    sim = initialize_simulation(args)

    robot = create_robot(sim)
    table = create_table(sim)
    caffe = create_caffe(sim)
    cup = create_cup(sim)

    # apply random perturbation
    apply_random_xy_perturbation(cup, max_perturbation=0.05)
    apply_random_xy_perturbation(caffe, max_perturbation=0.05)

    if not args.headless:
        sim.open_window()

    run_simulation(sim, robot, cup, caffe)

    logger.log_info("\n Press Ctrl+C to exit simulation loop.")
    try:
        counter = 0
        while True:
            counter += 1
            sim.update(step=10)
            if counter % 10 == 0:
                pass

    except KeyboardInterrupt:
        logger.log_info("\n Exit")


if __name__ == "__main__":
    main()
