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
This script demonstrates the creation and simulation of a robot with dexterous hands,
and performs a scoop ice task in a simulated environment.
"""

import argparse
import numpy as np
import time
import torch
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.objects import Robot, RigidObject, RigidObjectGroup
from embodichain.lab.sim.cfg import (
    JointDrivePropertiesCfg,
    RobotCfg,
    URDFCfg,
    RigidObjectCfg,
    RigidBodyAttributesCfg,
    ArticulationCfg,
    RigidObjectGroupCfg,
    LightCfg,
)
from embodichain.lab.sim.material import VisualMaterialCfg
from embodichain.lab.sim.utility.action_utils import interpolate_with_distance_warp
from embodichain.lab.sim.shapes import MeshCfg, CubeCfg
from embodichain.lab.sim.solvers import PytorchSolverCfg
from embodichain.data import get_data_path
from embodichain.utils import logger


def initialize_simulation():
    """
    Initialize the simulation environment based on the provided arguments.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        SimulationManager: Configured simulation manager instance.
    """
    config = SimulationManagerCfg(
        headless=True,
        sim_device="cpu",
        enable_rt=True,
        physics_dt=1.0 / 100.0,
    )
    sim = SimulationManager(config)

    light = sim.add_light(
        cfg=LightCfg(uid="main_light", intensity=50.0, init_pos=(0, 0, 2.0))
    )

    return sim


def randomize_ice_positions(sim, ice_cubes):
    """
    Randomly drop ice cubes into the container within a specified range.

    Args:
        sim (SimulationManager): The simulation manager instance.
        ice_cubes (RigidObjectGroup): Group of ice cube objects to be randomized.
    """
    num_objs = ice_cubes.num_objects
    position_low = np.array([0.65, -0.45, 0.5])
    position_high = np.array([0.55, -0.35, 0.5])
    position_random = np.random.uniform(
        low=position_low, high=position_high, size=(num_objs, 3)
    )
    random_drop_pose_np = np.eye(4)[None, :, :].repeat(num_objs, axis=0)
    random_drop_pose_np[:, :3, 3] = position_random

    # Assign random positions to each ice cube
    for i in tqdm(range(num_objs), desc="Dropping ice cubes"):
        ice_cubes.set_local_pose(
            pose=torch.tensor(
                random_drop_pose_np[i][None, None, :, :],
                dtype=torch.float32,
                device=sim.device,
            ),
            obj_ids=[i],
        )
        sim.update(step=10)


def create_robot(sim):
    """
    Create and configure a robot with an arm and a dexterous hand in the simulation.

    Args:
        sim (SimulationManager): The simulation manager instance.

    Returns:
        Robot: The configured robot instance added to the simulation.
    """
    # Retrieve URDF paths for the robot arm and hand
    ur10_urdf_path = get_data_path("UniversalRobots/UR10/UR10.urdf")
    hand_urdf_path = get_data_path(
        "BrainCoHandRevo1/BrainCoLeftHand/BrainCoLeftHand.urdf"
    )

    # Define transformation for attaching the hand to the arm
    hand_attach_xpos = np.eye(4)
    hand_attach_xpos[:3, :3] = R.from_rotvec([90, 0, 0], degrees=True).as_matrix()

    # Configure the robot with its components and control properties
    cfg = RobotCfg(
        uid="ur10_with_brainco",
        urdf_cfg=URDFCfg(
            components=[
                {"component_type": "arm", "urdf_path": ur10_urdf_path},
                {
                    "component_type": "hand",
                    "urdf_path": hand_urdf_path,
                    "transform": hand_attach_xpos,
                },
            ]
        ),
        control_parts={
            "arm": ["JOINT[0-9]"],
            "hand": [
                "LEFT_HAND_THUMB1",
                "LEFT_HAND_THUMB2",
                "LEFT_HAND_INDEX",
                "LEFT_HAND_MIDDLE",
                "LEFT_HAND_RING",
                "LEFT_HAND_PINKY",
            ],
        },
        drive_pros=JointDrivePropertiesCfg(
            stiffness={"JOINT[0-9]": 1e4, "LEFT_[A-Z|_]+[0-9]?": 1e2},
            damping={"JOINT[0-9]": 1e3, "LEFT_[A-Z|_]+[0-9]?": 1e1},
            max_effort={"JOINT[0-9]": 1e5, "LEFT_[A-Z|_]+[0-9]?": 1e3},
            drive_type="force",
        ),
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
            2.5,
            -np.pi / 2,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.5,
            -0.00016,
            -0.00010,
            -0.00013,
            -0.00009,
            0.0,
        ],
    )

    return sim.add_robot(cfg=cfg)


def create_scoop(sim: SimulationManager):
    """Create a scoop rigid object in the simulation."""
    scoop_cfg = RigidObjectCfg(
        uid="scoop",
        shape=MeshCfg(
            fpath=get_data_path("ScoopIceNewEnv/scoop.ply"),
        ),
        attrs=RigidBodyAttributesCfg(
            mass=0.5,
            static_friction=0.95,
            dynamic_friction=0.9,
            restitution=0.01,
            min_position_iters=32,
            min_velocity_iters=8,
        ),
        max_convex_hull_num=12,
        body_type="dynamic",
        init_pos=[0.6, 0.0, 0.09],
        init_rot=[0.0, 0.0, 0.0],
    )
    scoop = sim.add_rigid_object(cfg=scoop_cfg)
    return scoop


def create_heave_ice(sim: SimulationManager):
    """Create a heave ice rigid object in the simulation. Make sure that"""
    heave_ice_cfg = RigidObjectCfg(
        uid="heave_ice",
        shape=MeshCfg(
            fpath=get_data_path("ScoopIceNewEnv/ice_mesh_small/ice_000.obj"),
        ),
        attrs=RigidBodyAttributesCfg(
            mass=0.5,
            static_friction=0.95,
            dynamic_friction=0.9,
            restitution=0.01,
            min_position_iters=32,
            min_velocity_iters=8,
        ),
        body_type="dynamic",
        init_pos=[10, 10, 0.08],
        init_rot=[0.0, 0.0, 0.0],
    )
    heave_ice = sim.add_rigid_object(cfg=heave_ice_cfg)
    return heave_ice


def create_padding_box(sim: SimulationManager):
    padding_box_cfg = RigidObjectCfg(
        uid="padding_box",
        shape=CubeCfg(
            size=[0.1, 0.16, 0.05],
        ),
        attrs=RigidBodyAttributesCfg(
            mass=1.0,
            static_friction=0.95,
            dynamic_friction=0.9,
            restitution=0.01,
            min_position_iters=32,
            min_velocity_iters=8,
        ),
        body_type="kinematic",
        init_pos=[0.6, 0.15, 0.025],
        init_rot=[0.0, 0.0, 0.0],
    )
    heave_ice = sim.add_rigid_object(cfg=padding_box_cfg)
    return heave_ice


def create_container(sim: SimulationManager):
    container_cfg = ArticulationCfg(
        uid="container",
        fpath=get_data_path("ScoopIceNewEnv/IceContainer/ice_container.urdf"),
        init_pos=[0.7, -0.4, 0.21],
        init_rot=[0, 0, -90],
        attrs=RigidBodyAttributesCfg(
            mass=1.0,
            static_friction=0.95,
            dynamic_friction=0.9,
            restitution=0.01,
            min_position_iters=32,
            min_velocity_iters=8,
        ),
        drive_pros=JointDrivePropertiesCfg(
            stiffness=1.0, damping=0.1, max_effort=100.0, drive_type="force"
        ),
    )
    container = sim.add_articulation(cfg=container_cfg)
    return container


def create_ice_cubes(sim: SimulationManager):
    ice_cubes_path = get_data_path("ScoopIceNewEnv/ice_mesh_small")
    cfg_dict = {
        "uid": "ice_cubes",
        "max_num": 300,
        "folder_path": ice_cubes_path,
        "ext": ".obj",
        "rigid_objects": {
            "obj": {
                "attrs": {
                    "mass": 0.003,
                    "contact_offset": 0.001,
                    "rest_offset": 0,
                    "dynamic_friction": 0.05,
                    "static_friction": 0.1,
                    "restitution": 0.01,
                    "min_position_iters": 32,
                    "min_velocity_iters": 4,
                    "max_depenetration_velocity": 1.0,
                },
                "shape": {"shape_type": "Mesh"},
                "init_pos": [20.0, 0, 1.0],
            }
        },
    }

    ice_cubes_cfg = RigidObjectGroupCfg.from_dict(cfg_dict)
    ice_cubes: RigidObjectGroup = sim.add_rigid_object_group(cfg=ice_cubes_cfg)

    # Set visual material for ice cubes.
    # The material below only works for ray tracing backend.
    # Set ior to 1.31 and material type to "BSDF" for better ice appearance.
    ice_mat = sim.create_visual_material(
        cfg=VisualMaterialCfg(
            base_color=[1.0, 1.0, 1.0, 1.0],
            ior=1.31,
            roughness=0.05,
            material_type="BSDF",
        )
    )
    ice_cubes.set_visual_material(mat=ice_mat)

    return ice_cubes


def scoop_grasp(
    sim: SimulationManager,
    robot: Robot,
    scoop: RigidObject,
    heave_ice: RigidObject,
    padding_box: RigidObject,
):
    """
    Control the robot to grasp the scoop object and position the heave ice for scooping.

    Args:
        sim (SimulationManager): The simulation manager instance.
        robot (Robot): The robot instance to be controlled.
        scoop (RigidObject): The scoop object to be grasped.
        heave_ice (RigidObject): The heave ice object to be positioned.
        padding_box (RigidObject): The padding box object used as a reference for positioning.
    """
    rest_qpos = robot.get_qpos()
    arm_ids = robot.get_joint_ids("arm")
    hand_ids = robot.get_joint_ids("hand")
    hand_open_qpos = torch.tensor([0.0, 1.5, 0.4, 0.4, 0.4, 0.4])
    hand_close_qpos = torch.tensor([0.4, 1.5, 1.0, 1.1, 1.1, 0.9])
    arm_rest_qpos = rest_qpos[:, arm_ids]

    # Calculate and set the drop pose for the scoop object
    padding_box_pose = padding_box.get_local_pose(to_matrix=True)
    scoop_drop_relative_pose = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.115],
            [0.0, 0.0, 1.0, 0.065],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
        device=sim.device,
    )
    scoop_drop_pose = torch.bmm(
        padding_box_pose,
        scoop_drop_relative_pose[None, :, :].repeat(sim.num_envs, 1, 1),
    )
    scoop.set_local_pose(scoop_drop_pose)

    scoop_pose = scoop.get_local_pose(to_matrix=True)

    # tricky implementation
    heave_ice_relative = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, -0.13],
            [0.0, 0.0, 1.0, 0.04],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
        device=sim.device,
    )[None, :, :].repeat(sim.num_envs, 1, 1)
    heave_ice_pose = torch.bmm(scoop_pose, heave_ice_relative)
    heave_ice.set_local_pose(heave_ice_pose)
    sim.update(step=200)

    # move hand to grasp scoop
    scoop_pose = scoop.get_local_pose(to_matrix=True)
    grasp_scoop_pose_relative = torch.tensor(
        [
            [0.00522967, 0.6788424, 0.7342653, -0.05885637],
            [0.99054945, 0.0971214, -0.09684561, 0.0301468],
            [-0.13705578, 0.72783256, -0.6719191, 0.1040391],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
        device=sim.device,
    )[None, :, :].repeat(sim.num_envs, 1, 1)

    grasp_scoop_pose = torch.bmm(scoop_pose, grasp_scoop_pose_relative)
    pregrasp_scoop_pose = grasp_scoop_pose.clone()
    pregrasp_scoop_pose[:, 2, 3] += 0.1
    is_success, pre_grasp_scoop_qpos = robot.compute_ik(
        pregrasp_scoop_pose, joint_seed=arm_rest_qpos, name="arm"
    )

    is_success, grasp_scoop_qpos = robot.compute_ik(
        grasp_scoop_pose, joint_seed=arm_rest_qpos, name="arm"
    )
    robot.set_qpos(pre_grasp_scoop_qpos, joint_ids=arm_ids)
    sim.update(step=100)
    robot.set_qpos(grasp_scoop_qpos, joint_ids=arm_ids)
    sim.update(step=100)

    # close hand
    robot.set_qpos(hand_close_qpos[None, :].repeat(sim.num_envs, 1), joint_ids=hand_ids)
    sim.update(step=100)

    # remove heave ice
    remove_heave_ice_pose = torch.tensor(
        [
            [1.0, 0.0, 0.0, 10.0],
            [0.0, 1.0, 0.0, 10.0],
            [0.0, 0.0, 1.0, 0.04],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
        device=sim.device,
    )
    heave_ice.set_local_pose(remove_heave_ice_pose[None, :, :])


def scoop_ice(sim: SimulationManager, robot: Robot, scoop: RigidObject):
    """
    Control the robot to perform the scoop ice task, including lifting, scooping,
    and placing the ice.

    Args:
        sim (SimulationManager): The simulation manager instance.
        robot (Robot): The robot instance to be controlled.
        scoop (RigidObject): The scoop object used for scooping ice.
    """
    start_qpos = robot.get_qpos()
    arm_ids = robot.get_joint_ids("arm")
    hand_ids = robot.get_joint_ids("hand")
    hand_open_qpos = torch.tensor([0.0, 1.5, 0.4, 0.4, 0.4, 0.4])
    hand_close_qpos = torch.tensor([0.4, 1.5, 1.0, 1.1, 1.1, 0.9])
    arm_start_qpos = start_qpos[:, arm_ids]

    # lift
    arm_start_xpos = robot.compute_fk(arm_start_qpos, name="arm", to_matrix=True)
    arm_lift_xpos = arm_start_xpos.clone()
    arm_lift_xpos[:, 2, 3] += 0.45
    is_success, arm_lift_qpos = robot.compute_ik(
        arm_lift_xpos, joint_seed=arm_start_qpos, name="arm"
    )

    # apply 45 degree wrist rotation
    wrist_rotation = R.from_euler("X", 45, degrees=True).as_matrix()
    arm_lift_rotation = arm_lift_xpos[0, :3, :3].to("cpu").numpy()
    new_rotation = wrist_rotation @ arm_lift_rotation
    arm_lift_xpos_rotated = arm_lift_xpos.clone()
    arm_lift_xpos_rotated[:, :3, :3] = torch.tensor(
        new_rotation, dtype=torch.float32, device=sim.device
    )
    arm_lift_xpos_rotated[:, :3, 3] = torch.tensor(
        [0.5, -0.2, 0.55], dtype=torch.float32, device=sim.device
    )
    is_success, arm_lift_qpos_rotated = robot.compute_ik(
        arm_lift_xpos_rotated, joint_seed=arm_lift_qpos, name="arm"
    )

    # into container
    scoop_dis = 0.252
    scoop_offset = scoop_dis * torch.tensor(
        [0.0, -0.58123819, -0.81373347], dtype=torch.float32, device=sim.device
    )
    arm_into_container_xpos = arm_lift_xpos_rotated.clone()
    arm_into_container_xpos[:, :3, 3] = arm_into_container_xpos[:, :3, 3] + scoop_offset
    is_success, arm_into_container_qpos = robot.compute_ik(
        arm_into_container_xpos, joint_seed=arm_lift_qpos_rotated, name="arm"
    )

    # apply -60 degree wrist rotation
    arm_into_container_rotation = arm_into_container_xpos[0, :3, :3].to("cpu").numpy()
    wrist_rotation = R.from_euler("X", -60, degrees=True).as_matrix()
    new_rotation = wrist_rotation @ arm_into_container_rotation
    arm_scoop_xpos = arm_into_container_xpos.clone()
    arm_scoop_xpos[:, :3, :3] = torch.tensor(
        new_rotation, dtype=torch.float32, device=sim.device
    )
    is_success, arm_scoop_qpos = robot.compute_ik(
        arm_scoop_xpos, joint_seed=arm_into_container_qpos, name="arm"
    )

    # minor lift
    arm_scoop_xpos[:, 2, 3] += 0.15
    is_success, arm_scoop_lift_qpos = robot.compute_ik(
        arm_scoop_xpos, joint_seed=arm_scoop_qpos, name="arm"
    )

    # pack arm and hand trajectory
    arm_trajectory = torch.concatenate(
        [
            arm_start_qpos,
            arm_lift_qpos,
            arm_lift_qpos_rotated,
            arm_into_container_qpos,
            arm_scoop_qpos,
            arm_scoop_lift_qpos,
        ]
    )

    hand_trajectory = torch.vstack(
        [
            hand_close_qpos,
            hand_close_qpos,
            hand_close_qpos,
            hand_close_qpos,
            hand_close_qpos,
            hand_close_qpos,
        ]
    )

    all_trajectory = torch.hstack([arm_trajectory, hand_trajectory])
    interp_trajectory = interpolate_with_distance_warp(
        trajectory=all_trajectory[None, :, :], interp_num=200, device=sim.device
    )
    interp_trajectory = interp_trajectory[0]
    # run trajectory
    arm_ids = robot.get_joint_ids("arm")
    hand_ids = robot.get_joint_ids("hand")
    combine_ids = np.concatenate([arm_ids, hand_ids])
    for qpos in interp_trajectory:
        robot.set_qpos(qpos.unsqueeze(0), joint_ids=combine_ids)
        sim.update(step=10)


def main():
    """
    Main function to demonstrate robot simulation.

    This function initializes the simulation, creates the robot and other objects,
    and performs the scoop ice task.
    """
    sim = initialize_simulation()

    # Create simulation objects
    robot = create_robot(sim)
    container = create_container(sim)
    padding_box = create_padding_box(sim)
    scoop = create_scoop(sim)
    heave_ice = create_heave_ice(sim)
    ice_cubes = create_ice_cubes(sim)

    sim.open_window()

    # Randomize ice positions
    randomize_ice_positions(sim, ice_cubes)

    # Perform tasks
    scoop_grasp(sim, robot, scoop, heave_ice, padding_box)
    scoop_ice(sim, robot, scoop)

    logger.log_info("\n Press Ctrl+C to exit simulation loop.")
    try:
        while True:
            # sim.update(step=10)
            time.sleep(1e-2)
    except KeyboardInterrupt:
        logger.log_info("\n Exit")


if __name__ == "__main__":
    main()
