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
This script demonstrates the creation and simulation of a robot with a soft object,
and performs a pressing task in a simulated environment.
"""

import argparse
import numpy as np
import time
import open3d as o3d
import torch

from dexsim.utility.path import get_resources_data_path

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.objects import Robot, RigidObject
from embodichain.lab.sim.utility.action_utils import interpolate_with_distance
from embodichain.lab.sim.shapes import MeshCfg
from embodichain.lab.sim.solvers import PytorchSolverCfg
from embodichain.data import get_data_path
from embodichain.utils import logger
from embodichain.lab.sim.cfg import (
    JointDrivePropertiesCfg,
    RobotCfg,
    RigidObjectCfg,
    RigidBodyAttributesCfg,
    LightCfg,
    ClothObjectCfg,
    ClothPhysicalAttributesCfg,
    URDFCfg,
)
from embodichain.lab.sim.planners import MotionGenerator, MotionGenCfg, ToppraPlannerCfg
import os
from embodichain.lab.sim.shapes import MeshCfg, CubeCfg

from embodichain.toolkits.graspkit.pg_grasp.gripper_collision_checker import (
    GripperCollisionCfg,
)
from embodichain.toolkits.graspkit.pg_grasp.antipodal_generator import (
    GraspGenerator,
    GraspGeneratorCfg,
    AntipodalSamplerCfg,
)
from embodichain.lab.sim.atomic_actions.engine import (
    AtomicActionEngine,
    register_action,
)
from embodichain.lab.sim.atomic_actions.core import ObjectSemantics, AntipodalAffordance
from embodichain.lab.sim.atomic_actions.actions import PickUpActionCfg, PickUpAction


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
    parser.add_argument(
        "--num_envs", type=int, default=1, help="Number of parallel environments"
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
        num_envs=args.num_envs,
    )
    sim = SimulationManager(config)

    light = sim.add_light(
        cfg=LightCfg(uid="main_light", intensity=50.0, init_pos=(0, 0, 2.0))
    )

    return sim


def create_robot(sim: SimulationManager, position=[0.0, 0.0, 0.0]):
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
            stiffness={"JOINT[0-9]": 1e4, "FINGER[1-2]": 1e2},
            damping={"JOINT[0-9]": 1e3, "FINGER[1-2]": 1e1},
            max_effort={"JOINT[0-9]": 1e5, "FINGER[1-2]": 1e3},
            drive_type="force",
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
            )
        },
        init_qpos=[0.0, -np.pi / 2, -np.pi / 2, np.pi / 2, -np.pi / 2, 0.0, 0.0, 0.0],
        init_pos=position,
    )
    return sim.add_robot(cfg=cfg)


def create_mug(sim: SimulationManager) -> RigidObject:
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
    approach_xpos[:, 2, 3] += 0.04
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
        trajectory=all_trajectory, interp_num=220, device=sim.device
    )
    return interp_trajectory


def main():
    """
    Main function to demonstrate robot simulation.

    This function initializes the simulation, creates the robot and other objects,
    and performs the press softbody task.
    """
    args = parse_arguments()
    sim = initialize_simulation(args)

    robot = create_robot(sim)
    mug = create_mug(sim)

    motion_gen = MotionGenerator(
        cfg=MotionGenCfg(planner_cfg=ToppraPlannerCfg(robot_uid=robot.uid))
    )

    register_action(
        name="pick_up",
        action_class=PickUpAction,
    )
    pickup_cfg = PickUpActionCfg(
        hand_open_qpos=torch.tensor(
            [0.00, 0.00], dtype=torch.float32, device=sim.device
        ),
        hand_close_qpos=torch.tensor(
            [0.025, 0.025], dtype=torch.float32, device=sim.device
        ),
        control_part="arm",
        hand_control_part="hand",
        approach_direction=torch.tensor(
            [0.0, 0.0, -1.0], dtype=torch.float32, device=sim.device
        ),
    )
    atom_engine = AtomicActionEngine(
        robot=robot,
        motion_generator=motion_gen,
        device=sim.device,
        actions_cfg_dict={"pick_up": pickup_cfg},
    )

    sim.init_gpu_physics()
    sim.open_window()

    # Define object semantics and affordances for the mug
    gripper_collision_cfg = GripperCollisionCfg(
        max_open_length=0.088, finger_length=0.078, point_sample_dense=0.012
    )
    generator_cfg = GraspGeneratorCfg(
        viser_port=11801,
        antipodal_sampler_cfg=AntipodalSamplerCfg(
            n_sample=20000, max_length=0.088, min_length=0.003
        ),
    )
    mug_grasp_affordance = AntipodalAffordance(
        object_label="mug",
        force_reannotate=False,
        custom_config={
            "gripper_collision_cfg": gripper_collision_cfg,
            "generator_cfg": generator_cfg,
        },
    )
    mug_semantics = ObjectSemantics(
        label="mug",
        geometry={
            "mesh_vertices": mug.get_vertices(env_ids=[0], scale=True)[0],
            "mesh_triangles": mug.get_triangles(env_ids=[0])[0],
        },
        affordance=mug_grasp_affordance,
        entity=mug,  # in order to fetch object pose
    )
    result = atom_engine.execute(
        action_name="pick_up",
        target=mug_semantics,
        control_part="arm",
    )


if __name__ == "__main__":
    main()
