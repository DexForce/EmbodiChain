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

"""Press a soft cow with a UR10 using the Newton MJVBD solver."""

from __future__ import annotations

import argparse

import numpy as np
import torch
from dexsim.utility.path import get_resources_data_path

from embodichain.lab.gym.utils.gym_utils import add_env_launcher_args_to_parser
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import (
    NewtonCollisionPipelineCfg,
    NewtonPhysicsCfg,
    RenderCfg,
    SoftObjectCfg,
    SoftbodyPhysicalAttributesCfg,
    SoftbodyVoxelAttributesCfg,
)
from embodichain.lab.sim.objects import Robot, SoftObject
from embodichain.lab.sim.robots import URRobotCfg
from embodichain.lab.sim.shapes import MeshCfg
from embodichain.lab.sim.utility.action_utils import interpolate_with_nums
from embodichain.lab.visualization import visualization_cfg_from_args
from embodichain.utils import logger

PHYSICS_DT = 1.0 / 100.0
NUM_SUBSTEPS = 6
SOLVER_ITERATIONS = 8
SETTLE_STEPS = 100

SOFT_CONTACT_MARGIN = 0.003
SOFT_CONTACT_KE = 5.0e4
SOFT_CONTACT_KD = 1.0e-3
SOFT_CONTACT_MU = 1.0

COW_POSITION = (0.45, -0.1, 0.12)
PRESS_POSITION = (0.5, -0.1, 0.04)
APPROACH_HEIGHT = 0.015


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the demo."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_env_launcher_args_to_parser(parser)
    parser.set_defaults(device="cuda", physics="newton")
    args = parser.parse_args()
    if args.physics != "newton":
        parser.error("Soft bodies require --physics newton.")
    if not str(args.device).startswith("cuda"):
        parser.error("EmbodiChain soft bodies currently require a CUDA device.")
    return args


def initialize_simulation(args: argparse.Namespace) -> SimulationManager:
    """Create the Newton MJVBD simulation manager."""
    config = SimulationManagerCfg(
        width=1920,
        height=1080,
        headless=args.headless,
        device=args.device,
        gpu_id=args.gpu_id,
        num_envs=args.num_envs,
        arena_space=args.arena_space,
        render_cfg=RenderCfg(renderer=args.renderer),
        physics_cfg=NewtonPhysicsCfg(
            physics_dt=PHYSICS_DT,
            device=args.device,
            num_substeps=NUM_SUBSTEPS,
            solver_cfg={
                "solver_type": "mjvbd",
                "iterations": SOLVER_ITERATIONS,
                "particle_enable_self_contact": False,
                "particle_self_contact_radius": 0.005,
                "particle_self_contact_margin": 0.005,
                "particle_topological_contact_filter_threshold": 3,
                "particle_enable_tile_solve": True,
                "rigid_body_particle_contact_buffer_size": 512,
                "rigid_contact_k_start": SOFT_CONTACT_KE,
                "rigid_contact_max": 0,
                "soft_contact_margin": SOFT_CONTACT_MARGIN,
                "soft_contact_ke": SOFT_CONTACT_KE,
                "soft_contact_kd": SOFT_CONTACT_KD,
                "soft_contact_mu": SOFT_CONTACT_MU,
                # The registered trajectory updates the robot kinematically at
                # every Newton substep; MJVBD only needs to solve the soft body.
                "step_rigid_bodies": False,
            },
            collision_cfg=NewtonCollisionPipelineCfg(
                soft_contact_margin=SOFT_CONTACT_MARGIN,
            ),
        ),
        visualization=visualization_cfg_from_args(args),
    )
    return SimulationManager(config)


def create_robot(sim: SimulationManager) -> Robot:
    """Add the UR10 and enable particle contact on its pressing flange."""
    cfg = URRobotCfg.from_dict(
        {
            "robot_type": "ur10",
            "uid": "UR10",
            "solver_cfg": {"arm": {"tcp": np.eye(4)}},
            "link_attrs": {
                "pressing_flange": {
                    # ee_link has no collision geometry in the UR10 asset;
                    # Link6 is the physical flange immediately above it.
                    "link_names_expr": ["Link6"],
                    "attrs": {
                        "collision_props": {
                            "collision_enabled": True,
                            "contact_offset": 0.004,
                            "rest_offset": 0.001,
                        },
                        "material_props": {
                            "dynamic_friction": SOFT_CONTACT_MU,
                        },
                        "newton_props": {
                            "collision_props": {
                                "has_particle_collision": True,
                            },
                            "material_props": {
                                "ke": SOFT_CONTACT_KE,
                                "kd": SOFT_CONTACT_KD,
                            },
                        },
                    },
                },
            },
            "init_qpos": [
                0.0,
                -np.pi / 2,
                -np.pi / 2,
                np.pi / 2,
                -np.pi / 2,
                0.0,
            ],
        }
    )
    robot = sim.add_robot(cfg=cfg)
    if robot is None:
        raise RuntimeError("Failed to add the UR10 robot.")
    return robot


def create_soft_cow(sim: SimulationManager) -> SoftObject:
    """Add the tetrahedral soft cow used by the pressing task."""
    return sim.add_soft_object(
        cfg=SoftObjectCfg(
            uid="cow",
            shape=MeshCfg(
                fpath=get_resources_data_path("Model", "cow", "cow2.obj"),
            ),
            init_rot=[0.0, 90.0, 0.0],
            init_pos=COW_POSITION,
            particle_radius=0.005,
            voxel_attr=SoftbodyVoxelAttributesCfg(
                triangle_remesh_resolution=24,
                simulation_mesh_resolution=16,
                voxel_num_relaxation_iters=5,
            ),
            physical_attr=SoftbodyPhysicalAttributesCfg(
                youngs=5.0e3,
                poissons=0.45,
                density=100.0,
                elasticity_damping=0.1,
            ),
        )
    )


def build_press_trajectory(sim: SimulationManager, robot: Robot) -> torch.Tensor:
    """Compute the batched approach-and-press trajectory before preparation."""
    arm_joint_names = robot.cfg.control_parts["arm"]
    initial_qpos = torch.as_tensor(
        robot.cfg.init_qpos,
        dtype=torch.float32,
        device=sim.device,
    ).reshape(1, -1)
    initial_qpos = initial_qpos.repeat(sim.num_envs, 1)

    solver_cfg = robot.cfg.solver_cfg["arm"]
    solver_cfg.joint_names = list(arm_joint_names)
    if solver_cfg.urdf_path is None:
        solver_cfg.urdf_path = robot.cfg.fpath
    solver = solver_cfg.init_solver(device=sim.device)

    approach_pose = solver.get_fk(initial_qpos)
    approach_pose[:, :3, 3] = torch.tensor(
        [
            PRESS_POSITION[0],
            PRESS_POSITION[1],
            PRESS_POSITION[2] + APPROACH_HEIGHT,
        ],
        dtype=torch.float32,
        device=sim.device,
    )
    press_pose = approach_pose.clone()
    press_pose[:, :3, 3] = torch.tensor(
        PRESS_POSITION,
        dtype=torch.float32,
        device=sim.device,
    )

    approach_success, approach_qpos = solver.get_ik(
        target_xpos=approach_pose,
        qpos_seed=initial_qpos,
    )
    press_success, press_qpos = solver.get_ik(
        target_xpos=press_pose,
        qpos_seed=approach_qpos,
    )
    success = approach_success & press_success
    if not bool(torch.all(success)):
        failed_envs = torch.nonzero(~success, as_tuple=False).flatten().tolist()
        raise RuntimeError(f"IK failed for environment indices {failed_envs}.")

    keyframes = torch.stack(
        [initial_qpos, approach_qpos, press_qpos],
        dim=1,
    )
    # Move to the cow over 1.5 seconds, then press by 1.5 cm over 1 second.
    return interpolate_with_nums(
        trajectory=keyframes,
        interp_nums=torch.tensor([150, 100]),
        device=sim.device,
    )


def register_press_trajectory(
    sim: SimulationManager,
    trajectory: torch.Tensor,
) -> None:
    """Register the press after an initial soft-body settling interval."""
    hold = trajectory[:, :1, :].repeat(1, SETTLE_STEPS + 1, 1)
    playback = torch.cat([hold, trajectory[:, 1:, :]], dim=1)
    sim.register_kinematic_joint_trajectory("UR10", playback)


def main() -> None:
    """Create the scene, settle the cow, and execute the pressing motion."""
    args = parse_arguments()
    sim = initialize_simulation(args)

    try:
        robot = create_robot(sim)
        create_soft_cow(sim)
        trajectory = build_press_trajectory(sim, robot)
        register_press_trajectory(sim, trajectory)

        sim.prepare()
        if not args.headless:
            sim.open_window()

        sim.update(step=SETTLE_STEPS)
        if not args.headless:
            input("Press Enter to press the soft body...")
        sim.update(step=trajectory.shape[1] - 1)

        logger.log_info("\nPress Ctrl+C to exit the simulation loop.")
        while True:
            sim.update(step=10)
    except KeyboardInterrupt:
        logger.log_info("\nExit")
    finally:
        sim.destroy()


if __name__ == "__main__":
    main()
