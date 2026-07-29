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

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from embodichain.lab.sim import SimulationManager
from embodichain.lab.sim.objects import Robot, RigidObject
from embodichain.lab.sim.utility.action_utils import interpolate_with_distance
from embodichain.lab.sim.shapes import MeshCfg
from embodichain.lab.sim.solvers import URSolverCfg
from embodichain.data import get_data_path
from dexsim.utility.path import get_resources_data_path
from embodichain.utils import logger
from embodichain.lab.sim.cfg import (
    JointDrivePropertiesCfg,
    RobotCfg,
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
from embodichain.lab.sim.utility.demo_utils import (
    DemoRecording,
    add_demo_args,
    create_default_sim,
    maybe_open_window,
    maybe_wait_for_user,
    replay_trajectory,
    shutdown_sim,
)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments to configure the simulation.

    Returns:
        argparse.Namespace: Parsed arguments including number of environments and rendering options.
    """
    parser = argparse.ArgumentParser(
        description="Create and simulate a robot in SimulationManager"
    )
    add_demo_args(parser)
    parser.add_argument(
        "--n_sample",
        "--n-sample",
        type=int,
        default=10000,
        help="Number of antipodal grasp samples.",
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
    return create_default_sim(
        args,
        num_envs=args.num_envs,
        arena_space=2.5,
    )


def create_robot(
    sim: SimulationManager,
    position: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Robot:
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
            "arm": URSolverCfg(
                ur_type="ur10",
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


def create_obj(sim: SimulationManager) -> RigidObject:
    mug_cfg = RigidObjectCfg(
        uid="mug",
        shape=MeshCfg(
            fpath=get_resources_data_path("Model", "BakeTexture", "hdr_color_mesh.ply"),
        ),
        attrs=RigidBodyAttributesCfg(
            mass=0.01,
            dynamic_friction=0.97,
            static_friction=0.99,
        ),
        max_convex_hull_num=16,
        acd_method="vhacd",
        init_pos=[0.55, 0.0, 0.08],
        init_rot=[0.0, 0.0, 0.0],
    )
    return sim.add_rigid_object(cfg=mug_cfg)


def get_grasp_traj(
    sim: SimulationManager,
    robot: Robot,
    grasp_xpos: torch.Tensor,
) -> torch.Tensor:
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


def main() -> None:
    """Plan and replay a mug grasp."""
    args = parse_arguments()
    sim = initialize_simulation(args)
    try:
        robot = create_robot(sim)
        obj = create_obj(sim)
        maybe_open_window(sim, args)

        grasp_cfg = GraspGeneratorCfg(
            viser_port=11801,
            antipodal_sampler_cfg=AntipodalSamplerCfg(
                n_sample=args.n_sample,
                max_length=0.088,
                min_length=0.003,
            ),
            is_partial_annotate=False,
            is_filter_ground_collision=True,
            n_top_grasps=30,
        )
        started_at = time.perf_counter()
        grasp_generator = GraspGenerator(
            vertices=obj.get_vertices(env_ids=[0], scale=True)[0],
            triangles=obj.get_triangles(env_ids=[0])[0],
            cfg=grasp_cfg,
            gripper_collision_cfg=GripperCollisionCfg(
                max_open_length=0.088,
                finger_length=0.078,
                point_sample_dense=0.012,
            ),
        )

        # The first run opens Viser for selecting the mug's graspable region;
        # later runs reuse the cached annotation.
        grasp_generator.annotate()

        approach_direction = torch.tensor(
            [0, 0, -1],
            dtype=torch.float32,
            device=sim.device,
        )
        rest_pose = robot.compute_fk(
            qpos=robot.get_qpos("arm"),
            name="arm",
            to_matrix=True,
        )[0]
        grasp_poses = []
        for env_id, obj_pose in enumerate(obj.get_local_pose(to_matrix=True)):
            success, grasp_pose, _ = grasp_generator.get_grasp_poses(
                obj_pose,
                approach_direction,
                visualize_collision=False,
                visualize_pose=True,
            )
            if not success:
                logger.log_warning(
                    f"No valid grasp pose found for environment {env_id}."
                )
                grasp_pose = rest_pose
            grasp_poses.append(grasp_pose.unsqueeze(0))

        logger.log_info(
            f"Grasp pose generation took {time.perf_counter() - started_at:.2f} seconds"
        )
        trajectory = get_grasp_traj(sim, robot, torch.cat(grasp_poses, dim=0))
        maybe_wait_for_user(args, "Press Enter to start the mug grasp...")
        with DemoRecording(sim, args, prefix="grasp_generator"):
            replay_trajectory(
                sim,
                robot,
                trajectory,
                post_steps=0,
                step_size=4,
            )
        maybe_wait_for_user(args, "Press Enter to exit...")
    finally:
        shutdown_sim(sim)


if __name__ == "__main__":
    main()
