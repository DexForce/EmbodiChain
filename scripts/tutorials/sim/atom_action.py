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
from embodichain.lab.sim.atomic_actions.actions import (
    PickUpActionCfg,
    PickUpAction,
    PlaceActionCfg,
    PlaceAction,
    MoveActionCfg,
    MoveAction,
)


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


def run_trajactory(
    robot: Robot,
    trajectory: torch.Tensor,
    joint_ids: list[float],
    sim: SimulationManager,
):
    n_waypoint = trajectory.shape[1]
    for i in range(n_waypoint):
        robot.set_qpos(trajectory[:, i, :], joint_ids=joint_ids)
        sim.update(step=4)
        time.sleep(1e-2)


def main():
    """
    Main function to demonstrate robot simulation.

    This function initializes the simulation, creates the robot and other objects,
    and performs the press softbody task.
    """
    args = parse_arguments()
    sim: SimulationManager = initialize_simulation(args)
    robot = create_robot(sim)
    mug = create_mug(sim)

    motion_gen = MotionGenerator(
        cfg=MotionGenCfg(planner_cfg=ToppraPlannerCfg(robot_uid=robot.uid))
    )

    register_action(
        name="pick_up",
        action_class=PickUpAction,
    )
    register_action(
        name="place",
        action_class=PlaceAction,
    )
    register_action(
        name="move",
        action_class=MoveAction,
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
        pre_grasp_distance=0.15,
        lift_height=0.15,
    )

    place_cfg = PlaceActionCfg(
        hand_open_qpos=torch.tensor(
            [0.00, 0.00], dtype=torch.float32, device=sim.device
        ),
        hand_close_qpos=torch.tensor(
            [0.025, 0.025], dtype=torch.float32, device=sim.device
        ),
        control_part="arm",
        hand_control_part="hand",
        lift_height=0.15,
    )

    move_cfg = MoveActionCfg(
        control_part="arm",
    )

    atom_engine = AtomicActionEngine(
        robot=robot,
        motion_generator=motion_gen,
        device=sim.device,
        actions_cfg_dict={"pick_up": pickup_cfg, "place": place_cfg, "move": move_cfg},
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
    start_qpos = robot.get_qpos(name="arm")

    target_grasp_xpos = torch.tensor(
        [
            [-0.0539, -0.9985, -0.0022, 0.4489],
            [-0.9977, 0.0540, -0.0401, -0.0030],
            [0.0401, 0.0000, -0.9992, 0.1400],
            [0.0000, 0.0000, 0.0000, 1.0000],
        ],
        dtype=torch.float32,
        device=sim.device,
    )

    is_success, pick_trajectory, joint_ids = atom_engine.execute(
        start_qpos=start_qpos,
        action_name="pick_up",
        target=mug_semantics,
        # target=target_grasp_xpos,
        control_part="arm",
    )
    arm_joint_ids = robot.get_joint_ids("arm")
    place_start_qpos = pick_trajectory[:, -1, arm_joint_ids]
    place_xpos = target_grasp_xpos.clone()
    place_xpos[:3, 3] += torch.tensor([-0.2, 0.4, 0.1], device=sim.device)
    is_success, place_trajectory, joint_ids = atom_engine.execute(
        start_qpos=place_start_qpos,
        action_name="place",
        target=place_xpos,
        control_part="arm",
    )
    rest_xpos = target_grasp_xpos.clone()
    rest_xpos[:3, 3] = torch.tensor([0.5, 0.0, 0.5], device=sim.device)
    move_start_qpos = place_trajectory[:, -1, arm_joint_ids]
    is_success, move_trajectory, arm_joint_ids = atom_engine.execute(
        start_qpos=move_start_qpos,
        action_name="move",
        target=rest_xpos,
        control_part="arm",
    )

    run_trajactory(robot, pick_trajectory, joint_ids, sim)
    run_trajactory(robot, place_trajectory, joint_ids, sim)
    run_trajactory(robot, move_trajectory, arm_joint_ids, sim)

    input("Press Enter to exit...")


if __name__ == "__main__":
    main()
