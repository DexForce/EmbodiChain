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
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.objects import Robot, RigidObject
from embodichain.lab.sim.utility.action_utils import interpolate_with_distance
from embodichain.lab.sim.shapes import MeshCfg
from embodichain.lab.sim.solvers import PytorchSolverCfg
from embodichain.lab.sim.planners import MotionGenerator, MotionGenCfg, ToppraPlannerCfg
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
from embodichain.lab.sim.atomic_actions import (
    AntipodalAffordance,
    ObjectSemantics,
    UprightAction,
    UprightActionCfg,
)
from embodichain.toolkits.graspkit.pg_grasp.antipodal_generator import (
    GraspGenerator,
    GraspGeneratorCfg,
    AntipodalSamplerCfg,
)
from embodichain.toolkits.graspkit.pg_grasp.gripper_collision_checker import (
    GripperCollisionCfg,
)


def parse_arguments():
    """
    Parse command-line arguments to configure the simulation.

    Returns:
        argparse.Namespace: Parsed arguments including number of environments and rendering options.
    """
    parser = argparse.ArgumentParser(
        description="Create and simulate a robot in SimulationManager"
    )
    add_env_launcher_args_to_parser(parser)
    parser.add_argument(
        "--demo_mode",
        type=str,
        choices=["grasp", "upright"],
        default="grasp",
        help="Select the tutorial scenario to run.",
    )
    parser.add_argument(
        "--object_kind",
        type=str,
        choices=["cup", "bottle"],
        default="bottle",
        help="Object to use in upright mode.",
    )
    parser.add_argument(
        "--n_sample",
        type=int,
        default=10000,
        help="Number of surface samples for antipodal grasp generation.",
    )
    parser.add_argument(
        "--n_top_grasps",
        type=int,
        default=30,
        help="Number of top-ranked grasp poses to keep.",
    )
    parser.add_argument(
        "--force_reannotate",
        action="store_true",
        help="Force grasp region re-annotation instead of reusing cached antipodal pairs.",
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
        render_cfg=RenderCfg(renderer=args.renderer),
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
            stiffness={"JOINT[0-9]": 1e4, "FINGER[1-2]": 1e3},
            damping={"JOINT[0-9]": 1e3, "FINGER[1-2]": 1e2},
            max_effort={"JOINT[0-9]": 1e5, "FINGER[1-2]": 1e4},
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


def create_fallen_cup(sim: SimulationManager) -> RigidObject:
    cup_cfg = RigidObjectCfg(
        uid="cup",
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
        init_rot=[90.0, 0.0, -90.0],
        body_scale=(4, 4, 4),
    )
    return sim.add_rigid_object(cfg=cup_cfg)


def create_fallen_bottle(sim: SimulationManager) -> RigidObject:
    # Use a slightly smaller and closer bottle for the UR10 gripper demo.
    bottle_scale = 0.0008
    bottle_cfg = RigidObjectCfg(
        uid="bottle",
        shape=MeshCfg(
            fpath=get_data_path("ScannedBottle/yibao.ply"),
        ),
        attrs=RigidBodyAttributesCfg(
            mass=0.02,
            dynamic_friction=0.97,
            static_friction=0.99,
        ),
        max_convex_hull_num=16,
        init_pos=[0.4294, 0.0825, -0.0997],
        init_rot=[90.0, 0.0, 0.0],
        body_scale=(bottle_scale, bottle_scale, bottle_scale),
    )
    return sim.add_rigid_object(cfg=bottle_cfg)


def build_grasp_generator_cfg(args: argparse.Namespace) -> GraspGeneratorCfg:
    return GraspGeneratorCfg(
        viser_port=11801,
        antipodal_sampler_cfg=AntipodalSamplerCfg(
            n_sample=args.n_sample,
            max_length=0.088,
            min_length=0.003,
        ),
        is_partial_annotate=False,
        is_filter_ground_collision=True,
        n_top_grasps=args.n_top_grasps,
    )


def build_gripper_collision_cfg() -> GripperCollisionCfg:
    return GripperCollisionCfg(
        max_open_length=0.088,
        finger_length=0.078,
        point_sample_dense=0.012,
    )


def create_object_semantics(
    obj: RigidObject, label: str, args: argparse.Namespace
) -> ObjectSemantics:
    return ObjectSemantics(
        label=label,
        geometry={
            "mesh_vertices": obj.get_vertices(env_ids=[0], scale=True)[0],
            "mesh_triangles": obj.get_triangles(env_ids=[0])[0],
        },
        affordance=AntipodalAffordance(
            object_label=label,
            force_reannotate=args.force_reannotate,
            custom_config={
                "gripper_collision_cfg": build_gripper_collision_cfg(),
                "generator_cfg": build_grasp_generator_cfg(args),
            },
        ),
        entity=obj,
    )


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


def run_grasp_demo(
    args: argparse.Namespace, sim: SimulationManager, robot: Robot
) -> None:
    mug = create_mug(sim)
    grasp_cfg = build_grasp_generator_cfg(args)
    sim.open_window()

    # Annotate part of the mug to be grasped by following the instructions in the visualization window:
    # 1. View grasp object in browser (e.g http://localhost:11801)
    # 2. press 'Rect Select Region', select grasp region
    # 3. press 'Confirm Selection' to finish grasp region selection.

    start_time = time.time()

    vertices = mug.get_vertices(env_ids=[0], scale=True)[0]
    triangles = mug.get_triangles(env_ids=[0])[0]
    grasp_generator = GraspGenerator(
        vertices=vertices,
        triangles=triangles,
        cfg=grasp_cfg,
        gripper_collision_cfg=build_gripper_collision_cfg(),
    )

    grasp_generator.annotate()

    approach_direction = torch.tensor(
        [0, 0, -1], dtype=torch.float32, device=sim.device
    )
    obj_poses = mug.get_local_pose(to_matrix=True)
    grasp_xpos_list = []

    rest_xpos = robot.compute_fk(
        qpos=robot.get_qpos("arm"), name="arm", to_matrix=True
    )[0]
    for i, obj_pose in enumerate(obj_poses):
        is_success, grasp_pose, _ = grasp_generator.get_grasp_poses(
            obj_pose,
            approach_direction,
            visualize_collision=False,
            visualize_pose=True,
        )
        if is_success:
            grasp_xpos_list.append(grasp_pose.unsqueeze(0))
        else:
            logger.log_warning(f"No valid grasp pose found for {i}-th object.")
            grasp_xpos_list.append(rest_xpos.unsqueeze(0))

    grasp_xpos = torch.cat(grasp_xpos_list, dim=0)
    cost_time = time.time() - start_time
    logger.log_info(f"Get grasp pose cost time: {cost_time:.2f} seconds")

    grab_traj = get_grasp_traj(sim, robot, grasp_xpos)
    input("Press Enter to start the grab mug demo...")
    for i in range(grab_traj.shape[1]):
        robot.set_qpos(grab_traj[:, i, :])
        sim.update(step=4)
        time.sleep(1e-2)
    input("Press Enter to exit the simulation...")


def run_upright_demo(
    args: argparse.Namespace, sim: SimulationManager, robot: Robot
) -> None:
    if args.object_kind == "cup":
        obj = create_fallen_cup(sim)
    else:
        obj = create_fallen_bottle(sim)

    semantics = create_object_semantics(obj, args.object_kind, args)
    motion_gen = MotionGenerator(
        cfg=MotionGenCfg(planner_cfg=ToppraPlannerCfg(robot_uid=robot.uid))
    )
    hand_open = torch.tensor([0.00, 0.00], dtype=torch.float32, device=sim.device)
    hand_close = torch.tensor([0.025, 0.025], dtype=torch.float32, device=sim.device)
    upright_action = UprightAction(
        motion_generator=motion_gen,
        cfg=UprightActionCfg(
            control_part="arm",
            hand_control_part="hand",
            hand_open_qpos=hand_open,
            hand_close_qpos=hand_close,
            approach_direction=torch.tensor(
                [0.0, 0.0, -1.0], dtype=torch.float32, device=sim.device
            ),
            pre_grasp_distance=0.15,
            lift_height=0.15,
            upright_axis_sign=-1.0 if args.object_kind == "bottle" else 1.0,
        ),
    )

    sim.init_gpu_physics()
    sim.open_window()
    input(f"Inspect the fallen {args.object_kind}, then press Enter to plan upright...")

    start_time = time.time()
    is_success, traj, _ = upright_action.execute(semantics)
    cost_time = time.time() - start_time
    logger.log_info(f"Plan upright trajectory cost time: {cost_time:.2f} seconds")
    if not is_success:
        logger.log_warning("Failed to plan upright trajectory.")
        return

    input("Press Enter to start the upright demo...")
    for i in range(traj.shape[1]):
        robot.set_qpos(traj[:, i, :])
        sim.update(step=4)
        time.sleep(1e-2)
    input("Press Enter to exit the simulation...")


def main() -> None:
    args = parse_arguments()
    sim = initialize_simulation(args)
    robot = create_robot(sim, position=[0.0, 0.0, 0.0])

    if args.demo_mode == "upright":
        run_upright_demo(args, sim, robot)
    else:
        run_grasp_demo(args, sim, robot)


if __name__ == "__main__":
    main()
