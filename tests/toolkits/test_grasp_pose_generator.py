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
This script demonstrates the creation and simulation of a robot that grasps a rigid mug
in a simulated environment using the SimulationManager and grasp planning utilities.
"""

import argparse
import numpy as np
import time
import torch

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.objects import Robot, RigidObject
from embodichain.lab.sim.utility.action_utils import interpolate_with_distance
from embodichain.lab.sim.shapes import MeshCfg
from embodichain.lab.sim.solvers import PytorchSolverCfg
from embodichain.data import get_data_path
from embodichain.lab.gym.utils.gym_utils import add_env_launcher_args_to_parser
from embodichain.utils import logger
from embodichain.lab.sim.cfg import (
    RenderCfg,
    JointDrivePropertiesCfg,
    RobotCfg,
    LightCfg,
    RigidBodyAttributesCfg,
    RigidObjectCfg,
    URDFCfg,
)
from embodichain.toolkits.graspkit.pg_grasp.antipodal_generator import (
    GraspGenerator,
    GraspGeneratorCfg,
    AntipodalSamplerCfg,
)
from embodichain.toolkits.graspkit.pg_grasp.gripper_collision_checker import (
    GripperCollisionCfg,
)


def initialize_simulation() -> SimulationManager:
    """
    Initialize the simulation environment based on the provided arguments.

    Returns:
        SimulationManager: Configured simulation manager instance.
    """
    config = SimulationManagerCfg(
        headless=True,
        sim_device=torch.device("cuda"),
        render_cfg=RenderCfg(renderer="auto"),
        physics_dt=1.0 / 100.0,
        arena_space=2.5,
    )
    sim = SimulationManager(config)

    light = sim.add_light(
        cfg=LightCfg(
            uid="main_light",
            color=(0.6, 0.6, 0.6),
            intensity=30.0,
            init_pos=(1.0, 0, 3.0),
        )
    )

    return sim


def create_robot(sim: SimulationManager, position=[0.0, 0.0, 0.0]) -> Robot:
    """
    Create and configure a robot with an arm and a dexterous hand in the simulation.

    Args:
        sim (SimulationManager): The simulation manager instance.

    Returns:
        Robot: The configured robot instance added to the simulation.
    """
    # Retrieve URDF paths for the robot arm and hand
    ur10_urdf_path = get_data_path("UniversalRobots/UR10/UR10.urdf")
    gripper_urdf_path = get_data_path("DH_PGC_140_50_M/DH_PGC_140_50_M.urdf")
    # Configure the robot with its components and control properties
    cfg = RobotCfg(
        uid="UR10",
        urdf_cfg=URDFCfg(
            components=[
                {"component_type": "arm", "urdf_path": ur10_urdf_path},
                {"component_type": "hand", "urdf_path": gripper_urdf_path},
            ]
        ),
        drive_pros=JointDrivePropertiesCfg(
            stiffness={"Joint[0-9]": 1e4, "FINGER[1-2]": 1e3},
            damping={"Joint[0-9]": 1e3, "FINGER[1-2]": 1e2},
            max_effort={"Joint[0-9]": 1e5, "FINGER[1-2]": 1e4},
            drive_type="force",
        ),
        control_parts={
            "arm": ["Joint[0-9]"],
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
            )
        },
        init_qpos=[0.0, -np.pi / 2, -np.pi / 2, np.pi / 2, -np.pi / 2, 0.0, 0.0, 0.0],
        init_pos=position,
    )
    return sim.add_robot(cfg=cfg)


def create_mug(sim: SimulationManager):
    mug_cfg = RigidObjectCfg(
        uid="table",
        shape=MeshCfg(
            fpath=get_data_path("CoffeeCup/cup.ply"),
        ),
        attrs=RigidBodyAttributesCfg(
            mass=0.01,
            dynamic_friction=0.97,
            static_friction=0.99,
        ),
        max_convex_hull_num=16,
        init_pos=[0.55, 0.0, 0.01],
        init_rot=[0.0, 0.0, -90],
        body_scale=(4, 4, 4),
    )
    mug = sim.add_rigid_object(cfg=mug_cfg)
    return mug


def get_grasp_traj(sim: SimulationManager, robot: Robot, grasp_xpos: torch.Tensor):
    n_envs = sim.num_envs
    rest_arm_qpos = robot.get_qpos("arm")

    approach_xpos = grasp_xpos.clone()
    approach_xpos[:, 2, 3] += 0.1

    _, qpos_approach = robot.compute_ik(
        pose=approach_xpos, joint_seed=rest_arm_qpos, name="arm"
    )
    _, qpos_grasp = robot.compute_ik(
        pose=grasp_xpos, joint_seed=qpos_approach, name="arm"
    )
    hand_open_qpos = torch.tensor([0.00, 0.00], dtype=torch.float32, device=sim.device)
    hand_close_qpos = torch.tensor(
        [0.025, 0.025], dtype=torch.float32, device=sim.device
    )

    arm_trajectory = torch.cat(
        [
            rest_arm_qpos[:, None, :],
            qpos_approach[:, None, :],
            qpos_grasp[:, None, :],
            qpos_grasp[:, None, :],
            qpos_approach[:, None, :],
            rest_arm_qpos[:, None, :],
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
        ],
        dim=1,
    )
    all_trajectory = torch.cat([arm_trajectory, hand_trajectory], dim=-1)
    interp_trajectory = interpolate_with_distance(
        trajectory=all_trajectory, interp_num=200, device=sim.device
    )
    return interp_trajectory


def test_grasp_pose_generator():

    sim = initialize_simulation()
    robot = create_robot(sim, position=[0.0, 0.0, 0.0])
    mug = create_mug(sim)

    # get mug grasp pose
    grasp_cfg = GraspGeneratorCfg(
        viser_port=11801,
        antipodal_sampler_cfg=AntipodalSamplerCfg(
            n_sample=10000, max_length=0.088, min_length=0.003
        ),
        is_partial_annotate=False,
        is_filter_ground_collision=True,
        n_top_grasps=30,
    )

    gripper_collision_cfg = GripperCollisionCfg(
        max_open_length=0.088, finger_length=0.078, point_sample_dense=0.012
    )

    vertices = mug.get_vertices(env_ids=[0], scale=True)[0]
    triangles = mug.get_triangles(env_ids=[0])[0]
    grasp_generator = GraspGenerator(
        vertices=vertices,
        triangles=triangles,
        cfg=grasp_cfg,
        gripper_collision_cfg=gripper_collision_cfg,
    )

    # Annotate grasp region (populates internal antipodal point pairs)
    grasp_generator.annotate()

    # Compute grasp poses per environment
    approach_direction = torch.tensor(
        [0, 0, -1], dtype=torch.float32, device=sim.device
    )
    obj_poses = mug.get_local_pose(to_matrix=True)
    grasp_xpos_list = []

    rest_xpos = robot.compute_fk(
        qpos=robot.get_qpos("arm"), name="arm", to_matrix=True
    )[0]
    for i, obj_pose in enumerate(obj_poses):
        is_success, grasp_pose, open_length = grasp_generator.get_grasp_poses(
            obj_pose,
            approach_direction,
            visualize_collision=False,
            visualize_pose=False,
        )
        if is_success:
            grasp_xpos_list.append(grasp_pose.unsqueeze(0))
        else:
            logger.log_warning(f"No valid grasp pose found for {i}-th object.")
            grasp_xpos_list.append(rest_xpos.unsqueeze(0))

    grasp_xpos = torch.cat(grasp_xpos_list, dim=0)
    grab_traj = get_grasp_traj(sim, robot, grasp_xpos)
    assert grasp_xpos.shape == torch.Size([1, 4, 4])
    assert grab_traj.shape == torch.Size([1, 200, 8])


if __name__ == "__main__":
    test_grasp_pose_generator()
