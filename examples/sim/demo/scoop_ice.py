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
This script demonstrates the creation and simulation of a robot with dexterous hands,
and performs a scoop ice task in a simulated environment.
"""

from __future__ import annotations

import argparse
import time
from embodichain.cli.sim import (
    add_sim_args_to_parser,
    add_seed_arg_to_parser,
    resolve_seed,
)


def build_parser() -> argparse.ArgumentParser:
    """Build CLI options without initializing simulation resources."""
    parser = argparse.ArgumentParser(description="Scoop ice task simulation")
    add_sim_args_to_parser(parser)
    add_seed_arg_to_parser(parser, default=0, scope="ice placement")
    return parser


if __name__ == "__main__":
    # Parse before importing optional simulation/planning dependencies.
    _cli_args = build_parser().parse_args()


import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.visualization import visualization_cfg_from_args
from embodichain.lab.sim.objects import Robot, RigidObject, RigidObjectGroup
from embodichain.lab.sim.cfg import (
    RenderCfg,
    physics_cfg_for_backend,
    RigidObjectCfg,
    RigidBodyPhysicsCfg,
    ArticulationCfg,
    RigidObjectGroupCfg,
    JointDrivePropertiesCfg,
    LightCfg,
)
from embodichain.lab.sim.material import VisualMaterialCfg
from embodichain.lab.sim.shapes import CubeCfg, MeshCfg, MeshCollisionCfg
from embodichain.data import get_data_path
from embodichain.utils import logger
from embodichain.lab.sim.robots import URRobotCfg


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
        device=args.device,
        render_cfg=RenderCfg(renderer=args.renderer),
        physics_cfg=physics_cfg_for_backend(args.physics),
        physics_dt=1.0 / 100.0,
        visualization=visualization_cfg_from_args(args),
    )
    sim = SimulationManager(config)

    light = sim.add_light(
        cfg=LightCfg(uid="main_light", intensity=10.0, init_pos=(0, 0, 2.0))
    )

    return sim


def randomize_ice_positions(
    sim: SimulationManager, ice_cubes: RigidObjectGroup, *, seed: int = 0
) -> None:
    """Place separated ice cubes inside the bin before advancing physics.

    The meshes fit inside 29 mm boxes. A 32 mm lattice with at most 0.5 mm
    jitter leaves clearance for the collision contact offsets as well.

    Args:
        sim: Prepared simulation containing the ice container.
        ice_cubes: Ice group to reset before settling.
        seed: Local random seed for the small position perturbations.
    """
    indices = np.arange(ice_cubes.num_objects)
    positions = np.column_stack(
        [
            0.144 - ((indices // 5) % 10) * 0.032,
            -0.20 + (indices % 5) * 0.032,
            -0.16 + (indices // 50) * 0.032,
        ]
    )
    positions += np.random.default_rng(seed).uniform(-0.0005, 0.0005, positions.shape)
    poses = torch.eye(4, device=sim.device).repeat(sim.num_envs, len(indices), 1, 1)
    poses[:, :, :3, 3] = torch.as_tensor(
        positions, dtype=torch.float32, device=sim.device
    )
    container_pose = sim.get_articulation("container").get_local_pose(to_matrix=True)
    ice_cubes.set_local_pose(container_pose[:, None] @ poses)
    ice_cubes.clear_dynamics()
    sim.update(step=300)


def create_robot(sim):
    """
    Create and configure a robot with an arm and a dexterous hand in the simulation.

    Args:
        sim (SimulationManager): The simulation manager instance.

    Returns:
        Robot: The configured robot instance added to the simulation.
    """
    hand_urdf_path = get_data_path(
        "BrainCoHandRevo1/BrainCoLeftHand/BrainCoLeftHand.urdf"
    )

    # Define transformation for attaching the hand to the arm
    hand_attach_xpos = np.eye(4)
    hand_attach_xpos[:3, :3] = R.from_rotvec([90, 0, 0], degrees=True).as_matrix()

    cfg = URRobotCfg.from_dict(
        {
            "robot_type": "ur10",
            "uid": "ur10_with_brainco",
            "urdf_cfg": {
                "components": [
                    {
                        "component_type": "hand",
                        "urdf_path": hand_urdf_path,
                        "transform": hand_attach_xpos,
                    },
                ]
            },
            "control_parts": {
                "hand": [
                    "LEFT_HAND_THUMB1",
                    "LEFT_HAND_THUMB2",
                    "LEFT_HAND_INDEX",
                    "LEFT_HAND_MIDDLE",
                    "LEFT_HAND_RING",
                    "LEFT_HAND_PINKY",
                ],
            },
            "joint_drive_props": {
                "stiffness": {"LEFT_[A-Z|_]+[0-9]?": 1e2},
                "damping": {"LEFT_[A-Z|_]+[0-9]?": 1e1},
                "max_effort": {"LEFT_[A-Z|_]+[0-9]?": 1e3},
                "drive_type": "force",
            },
            "solver_cfg": {"arm": {"tcp": np.eye(4)}},
            "init_qpos": [
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
        }
    )

    return sim.add_robot(cfg=cfg)


def create_scoop(sim: SimulationManager):
    """Create a lightweight (150 g) scoop for the hand's friction grasp."""
    scoop_cfg = RigidObjectCfg(
        uid="scoop",
        shape=MeshCfg(
            fpath=get_data_path("ScoopIceNewEnv/scoop.ply"),
            collision=MeshCollisionCfg(
                approximation="convex_decomposition",
                max_hulls=12,
            ),
        ),
        attrs=RigidBodyPhysicsCfg.from_dict(
            {
                "mass_props": {"mass": 0.15},
                "rigid_props": {"min_position_iters": 32, "min_velocity_iters": 8},
                "material_props": {
                    "static_friction": 0.95,
                    "dynamic_friction": 0.9,
                    "restitution": 0.01,
                },
            }
        ),
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
        attrs=RigidBodyPhysicsCfg.from_dict(
            {
                "mass_props": {"mass": 0.5},
                "rigid_props": {"min_position_iters": 32, "min_velocity_iters": 8},
                "material_props": {
                    "static_friction": 0.95,
                    "dynamic_friction": 0.9,
                    "restitution": 0.01,
                },
            }
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
        attrs=RigidBodyPhysicsCfg.from_dict(
            {
                "mass_props": {"mass": 1.0},
                "rigid_props": {"min_position_iters": 32, "min_velocity_iters": 8},
                "material_props": {
                    "static_friction": 0.95,
                    "dynamic_friction": 0.9,
                    "restitution": 0.01,
                },
            }
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
        attrs=RigidBodyPhysicsCfg.from_dict(
            {
                "mass_props": {"mass": 1.0},
                "rigid_props": {"min_position_iters": 32, "min_velocity_iters": 8},
                "material_props": {
                    "static_friction": 0.95,
                    "dynamic_friction": 0.9,
                    "restitution": 0.01,
                },
            }
        ),
        joint_drive_props=JointDrivePropertiesCfg(
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
                    "mass_props": {"mass": 0.003},
                    "rigid_props": {
                        "min_position_iters": 32,
                        "min_velocity_iters": 4,
                        "max_depenetration_velocity": 1.0,
                    },
                    "collision_props": {
                        "contact_offset": 0.001,
                        "rest_offset": 0,
                    },
                    "material_props": {
                        "dynamic_friction": 0.05,
                        "static_friction": 0.1,
                        "restitution": 0.01,
                    },
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
            roughness=0.2,
            material_type="BSDF",
        )
    )
    sim.prepare()
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
    pre_grasp_scoop_qpos = _solve_arm_ik(robot, pregrasp_scoop_pose, arm_rest_qpos)

    grasp_scoop_qpos = _solve_arm_ik(robot, grasp_scoop_pose, pre_grasp_scoop_qpos)
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


def _solve_arm_ik(
    robot: Robot, pose: torch.Tensor, joint_seed: torch.Tensor
) -> torch.Tensor:
    """Reject unreachable targets before sending them to the arm."""
    success, qpos = robot.compute_ik(pose, joint_seed=joint_seed, name="arm")
    if not bool(torch.all(success)) or not bool(torch.isfinite(qpos).all()):
        raise RuntimeError("Scoop trajectory contains an unreachable arm pose.")
    return qpos


def scoop_ice(sim: SimulationManager, robot: Robot, scoop: RigidObject) -> None:
    """Scoop below the ice surface, curl the bowl upward, then lift it out.

    Targets describe the scoop, using the measured grasp transform to account
    for the tool's settled orientation in the fingers. Bin-local positions
    keep the insertion depth and wall clearance tied to the container.

    Args:
        sim: Prepared simulation containing the ice container.
        robot: Robot holding the scoop with its hand closed.
        scoop: Grasped scoop whose current pose defines the tool transform.
    """
    arm_ids = robot.get_joint_ids("arm")
    qpos = robot.get_qpos()[:, arm_ids]
    tcp_pose = robot.compute_fk(qpos, name="arm", to_matrix=True)
    scoop_pose = scoop.get_local_pose(to_matrix=True)
    scoop_to_tcp = torch.linalg.inv(scoop_pose) @ tcp_pose
    container_pose = sim.get_articulation("container").get_local_pose(to_matrix=True)

    lift_pose = scoop_pose.clone()
    lift_pose[:, 2, 3] += 0.35
    targets = [lift_pose]
    # Container local -X points toward the approach side; local -Y crosses the bin.
    # The bowl extends along scoop -Y, 19 cm beyond the grasp origin.
    for position, pitch in [
        ((-0.24, -0.15, 0.19), 0.0),
        ((-0.16, -0.15, 0.09), 30.0),
        ((-0.10, -0.15, -0.06), 30.0),
        ((-0.03, -0.15, -0.11), -10.0),
        ((-0.03, -0.15, 0.19), -10.0),
    ]:
        relative = torch.eye(4, dtype=torch.float32, device=sim.device)
        relative[:3, :3] = torch.as_tensor(
            R.from_euler("z", 90, degrees=True).as_matrix()
            @ R.from_euler("x", pitch, degrees=True).as_matrix(),
            dtype=torch.float32,
            device=sim.device,
        )
        relative[:3, 3] = torch.as_tensor(position, device=sim.device)
        targets.append(container_pose @ relative)

    for target in targets:
        end_qpos = _solve_arm_ik(robot, target @ scoop_to_tcp, qpos)
        # Small target increments avoid impulsive loads on the friction grasp.
        for alpha in torch.linspace(0.0, 1.0, 100, device=sim.device):
            robot.set_qpos(qpos + alpha * (end_qpos - qpos), joint_ids=arm_ids)
            sim.update(step=4)
        qpos = robot.get_qpos()[:, arm_ids]
    sim.update(step=100)


def main(args: argparse.Namespace | None = None) -> None:
    parser = build_parser()
    if args is None:
        args = parser.parse_args()

    """
    Main function to demonstrate robot simulation.

    This function initializes the simulation, creates the robot and other objects,
    and performs the scoop ice task.
    """
    sim = initialize_simulation(args)

    # Create simulation objects
    robot = create_robot(sim)
    container = create_container(sim)
    padding_box = create_padding_box(sim)
    scoop = create_scoop(sim)
    heave_ice = create_heave_ice(sim)
    ice_cubes = create_ice_cubes(sim)
    sim.prepare()

    if not args.headless:
        sim.open_window()

    # Randomize ice positions
    seed = resolve_seed(args.seed)
    logger.log_info(f"Ice placement seed: {seed}")
    randomize_ice_positions(sim, ice_cubes, seed=seed)

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
    main(_cli_args)
