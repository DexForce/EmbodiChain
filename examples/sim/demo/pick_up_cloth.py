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

"""Pick up a cloth with a UR10 gripper using the Newton MJVBD solver."""

from __future__ import annotations

import argparse
import os
import tempfile
from collections.abc import Sequence

import numpy as np
import open3d as o3d
import torch
from scipy.spatial.transform import Rotation

from embodichain.data import get_data_path
from embodichain.lab.gym.utils.gym_utils import add_env_launcher_args_to_parser
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import (
    ClothObjectCfg,
    ClothPhysicalAttributesCfg,
    MassPropertiesCfg,
    NewtonCollisionPropertiesCfg,
    NewtonPhysicsCfg,
    NewtonRigidBodyMaterialCfg,
    NewtonRigidBodyPhysicsCfg,
    RigidBodyMaterialCfg,
    RigidBodyPhysicsCfg,
    RigidObjectCfg,
    RenderCfg,
)
from embodichain.lab.sim.objects import ClothObject, RigidObject, Robot
from embodichain.lab.sim.robots import URRobotCfg
from embodichain.lab.sim.shapes import CubeCfg, MeshCfg
from embodichain.lab.sim.utility.action_utils import interpolate_with_nums
from embodichain.lab.visualization import visualization_cfg_from_args

CLOTH_SIZE = 0.3
CLOTH_GRID_CELLS = 50
CLOTH_PARTICLE_RADIUS = 0.003
CLOTH_RIGID_CONTACT_KE = 1.0e6
CLOTH_RIGID_CONTACT_KD = 5.0e-2
CLOTH_RIGID_CONTACT_MU = 0.5


def create_robot(
    sim: SimulationManager,
    position: Sequence[float] = (0.0, 0.0, 0.0),
) -> Robot:
    """
    Create and configure a robot with an arm and a dexterous hand in the simulation.

    Args:
        sim (SimulationManager): The simulation manager instance.

    Returns:
        Robot: The configured robot instance added to the simulation.
    """
    gripper_urdf_path = get_data_path("DH_PGC_140_50_M/DH_PGC_140_50_M.urdf")
    cfg = URRobotCfg.from_dict(
        {
            "robot_type": "ur10",
            "uid": "UR10",
            "urdf_cfg": {
                "components": [
                    {"component_type": "hand", "urdf_path": gripper_urdf_path},
                ]
            },
            "drive_pros": {
                "stiffness": {"FINGER[1-2]": 1e2},
                "damping": {"FINGER[1-2]": 1e1},
                "max_effort": {"FINGER[1-2]": 1e3},
                "drive_type": "force",
                "target_mode": "position_velocity",
            },
            "link_attrs": {
                "gripper_collision": {
                    "link_names_expr": ["hand_base_link", "finger[1-2]"],
                    "attrs": {
                        "collision_props": {
                            "collision_enabled": True,
                            # Detect the thin cloth before it reaches the mesh.
                            "contact_offset": 0.008,
                            "rest_offset": 0.002,
                        },
                        "material_props": {
                            "dynamic_friction": 2.0,
                        },
                        "newton_props": {
                            "collision_props": {
                                "has_particle_collision": True,
                            },
                            "material_props": {
                                "ke": 3.0e5,
                                "kd": 1.0e-4,
                            },
                        },
                    },
                },
            },
            "control_parts": {
                "hand": ["FINGER[1-2]"],
            },
            "solver_cfg": {
                "arm": {
                    "tcp": [
                        [0.0, 1.0, 0.0, 0.0],
                        [-1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.12],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                }
            },
            "init_qpos": [
                0.0,
                -np.pi / 2,
                -np.pi / 2,
                np.pi / 2,
                -np.pi / 2,
                0.0,
                0.0,
                0.0,
            ],
            "init_pos": position,
        }
    )
    return sim.add_robot(cfg=cfg)


def create_padding_box(sim: SimulationManager) -> RigidObject:
    padding_box_cfg = RigidObjectCfg(
        uid="padding_box",
        shape=CubeCfg(
            size=[0.02, 0.07, 0.05],
        ),
        attrs=RigidBodyPhysicsCfg(
            mass_props=MassPropertiesCfg(mass=1.0),
            material_props=RigidBodyMaterialCfg(
                static_friction=CLOTH_RIGID_CONTACT_MU,
                dynamic_friction=CLOTH_RIGID_CONTACT_MU,
                restitution=0.01,
            ),
            newton_props=NewtonRigidBodyPhysicsCfg(
                collision_props=NewtonCollisionPropertiesCfg(
                    has_particle_collision=True,
                ),
                material_props=NewtonRigidBodyMaterialCfg(
                    ke=CLOTH_RIGID_CONTACT_KE,
                    kd=CLOTH_RIGID_CONTACT_KD,
                ),
            ),
        ),
        body_type="kinematic",
        init_pos=[0.5, 0.0, 0.026],
        init_rot=[0.0, 0.0, 0.0],
    )
    return sim.add_rigid_object(cfg=padding_box_cfg)


def create_2d_grid_mesh(
    width: float, height: float, nx: int = 1, ny: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a flat rectangle in the XY plane centered at `origin`.

    The rectangle is subdivided into an `nx` by `ny` grid (cells) and
    triangulated. `nx=1, ny=1` yields the simple two-triangle rectangle.

    Returns an vertices and triangles.
    """
    w = float(width)
    h = float(height)
    if nx < 1 or ny < 1:
        raise ValueError("nx and ny must be >= 1")

    # Vectorized vertex positions using PyTorch
    x_lin = torch.linspace(-w / 2.0, w / 2.0, steps=nx + 1, dtype=torch.float64)
    y_lin = torch.linspace(-h / 2.0, h / 2.0, steps=ny + 1, dtype=torch.float64)
    yy, xx = torch.meshgrid(y_lin, x_lin, indexing="ij")
    xx_flat = xx.reshape(-1)
    yy_flat = yy.reshape(-1)
    zz_flat = torch.full_like(xx_flat, 0, dtype=torch.float64)
    verts = torch.stack([xx_flat, yy_flat, zz_flat], dim=1)  # (Nverts, 3)

    # Vectorized triangle indices
    idx = torch.arange((nx + 1) * (ny + 1), dtype=torch.int64).reshape(ny + 1, nx + 1)
    v0 = idx[:-1, :-1].reshape(-1)
    v1 = idx[:-1, 1:].reshape(-1)
    v2 = idx[1:, :-1].reshape(-1)
    v3 = idx[1:, 1:].reshape(-1)
    tri1 = torch.stack([v0, v1, v3], dim=1)
    tri2 = torch.stack([v0, v3, v2], dim=1)
    faces = torch.cat([tri1, tri2], dim=0).to(torch.int32)
    return verts, faces


def create_cloth(sim: SimulationManager) -> ClothObject:
    cloth_verts, cloth_faces = create_2d_grid_mesh(
        width=CLOTH_SIZE,
        height=CLOTH_SIZE,
        nx=CLOTH_GRID_CELLS,
        ny=CLOTH_GRID_CELLS,
    )
    cloth_mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(cloth_verts.to("cpu").numpy()),
        triangles=o3d.utility.Vector3iVector(cloth_faces.to("cpu").numpy()),
    )
    cloth_save_path = os.path.join(tempfile.gettempdir(), "cloth_mesh.ply")
    o3d.io.write_triangle_mesh(cloth_save_path, cloth_mesh)

    cloth = sim.add_cloth_object(
        cfg=ClothObjectCfg(
            uid="cloth",
            shape=MeshCfg(fpath=cloth_save_path),
            init_pos=[0.5, 0.0, 0.3],
            init_rot=[0, 0, 0],
            # Keep the collision shell close to the rendered surface.  A large
            # radius can make the gripper carry cloth while visibly separated.
            particle_radius=CLOTH_PARTICLE_RADIUS,
            physical_attr=ClothPhysicalAttributesCfg(
                # Give gravity enough authority to produce fabric-like drape.
                # Stretch remains firmer than bending so the cloth folds
                # instead of behaving like an elastic sheet.
                density=0.05,
                tri_ke=2.0e2,
                tri_ka=2.0e2,
                tri_kd=1.0e-5,
                edge_ke=0.005,
                edge_kd=0.01,
            ),
        )
    )
    return cloth


def get_grasp_traj(
    sim: SimulationManager,
    robot: Robot,
    grasp_xpos: torch.Tensor,
) -> torch.Tensor:
    """Build the full robot trajectory without materializing the scene.

    MJVBD cloth manipulation uses an external kinematic articulation.  The
    trajectory therefore has to be registered before ``sim.prepare()`` so the
    Newton runtime can interpolate joint poses and velocities at every
    substep.
    """
    num_envs = sim.num_envs
    arm_joint_names = robot.cfg.control_parts["arm"]
    arm_dof = len(arm_joint_names)
    initial_qpos = torch.as_tensor(
        robot.cfg.init_qpos,
        dtype=torch.float32,
        device=sim.device,
    ).reshape(1, -1)
    hand_dof = initial_qpos.shape[1] - arm_dof
    if hand_dof <= 0:
        raise ValueError("The robot trajectory requires at least one hand DOF.")
    rest_arm_qpos = initial_qpos[:, :arm_dof].repeat(num_envs, 1)

    solver_cfg = robot.cfg.solver_cfg["arm"]
    solver_cfg.joint_names = list(arm_joint_names)
    if solver_cfg.urdf_path is None:
        solver_cfg.urdf_path = robot.cfg.fpath
    solver = solver_cfg.init_solver(device=sim.device)

    root_pose_value = robot.cfg.init_local_pose
    if root_pose_value is None:
        root_pose_value = np.eye(4, dtype=np.float32)
        root_pose_value[:3, :3] = Rotation.from_euler(
            "xyz", robot.cfg.init_rot, degrees=True
        ).as_matrix()
        root_pose_value[:3, 3] = np.asarray(robot.cfg.init_pos, dtype=np.float32)
    root_pose = torch.as_tensor(
        root_pose_value,
        dtype=torch.float32,
        device=sim.device,
    ).reshape(1, 4, 4)
    root_pose = root_pose.repeat(num_envs, 1, 1)
    root_pose_inv = torch.linalg.inv(root_pose)

    approach_xpos = grasp_xpos.clone()
    approach_xpos[:, 2, 3] += 0.06
    approach_xpos = torch.bmm(root_pose_inv, approach_xpos)
    local_grasp_xpos = torch.bmm(root_pose_inv, grasp_xpos)
    approach_success, qpos_approach = solver.get_ik(
        target_xpos=approach_xpos,
        qpos_seed=rest_arm_qpos,
    )
    grasp_success, qpos_grasp = solver.get_ik(
        target_xpos=local_grasp_xpos,
        qpos_seed=qpos_approach,
    )
    if not bool(torch.all(approach_success & grasp_success)):
        failed_envs = torch.nonzero(
            ~(approach_success & grasp_success), as_tuple=False
        ).flatten()
        raise RuntimeError(f"IK failed for environment indices {failed_envs.tolist()}.")

    hand_open_qpos = initial_qpos[:, arm_dof : arm_dof + hand_dof].repeat(num_envs, 1)
    # First close around the cloth ridge while the padding box supports it.
    # After lifting clear of the box, close once more so the fingers—not the
    # support reaction—provide the sustained normal force for the grasp.
    hand_pregrasp_qpos = torch.full(
        (num_envs, hand_dof),
        0.012,
        dtype=torch.float32,
        device=sim.device,
    )
    hand_grasp_qpos = torch.full(
        (num_envs, hand_dof),
        0.024,
        dtype=torch.float32,
        device=sim.device,
    )

    arm_trajectory = torch.cat(
        [
            rest_arm_qpos[:, None, :],
            qpos_approach[:, None, :],
            qpos_grasp[:, None, :],
            qpos_grasp[:, None, :],
            qpos_approach[:, None, :],
            qpos_approach[:, None, :],
            rest_arm_qpos[:, None, :],
        ],
        dim=1,
    )
    hand_trajectory = torch.cat(
        [
            hand_open_qpos[:, None, :],
            hand_open_qpos[:, None, :],
            hand_open_qpos[:, None, :],
            hand_pregrasp_qpos[:, None, :],
            hand_pregrasp_qpos[:, None, :],
            hand_grasp_qpos[:, None, :],
            hand_grasp_qpos[:, None, :],
        ],
        dim=1,
    )
    all_trajectory = torch.cat([arm_trajectory, hand_trajectory], dim=-1)
    # Keep the original wall-clock timing while supplying one waypoint per
    # 100 Hz frame. Newton further interpolates each frame over 12 substeps.
    interp_trajectory = interpolate_with_nums(
        trajectory=all_trajectory,
        interp_nums=torch.tensor([180, 90, 180, 90, 90, 180]),
        device=sim.device,
    )
    return interp_trajectory


def register_kinematic_trajectory(
    sim: SimulationManager,
    trajectory: torch.Tensor,
    settle_steps: int,
) -> None:
    """Register the grasp trajectory after an initial settling hold.

    Args:
        sim: Simulation manager that owns the target robot.
        trajectory: Batched robot joint positions with shape
            ``(num_envs, frames, dof)``.
        settle_steps: Number of initial physics frames held at row zero.
    """
    if settle_steps < 0:
        raise ValueError("settle_steps must be non-negative.")

    # Row zero is the initial state and each simulation frame advances to the
    # next row. Keep settle_steps + 1 identical rows so settling consumes no
    # part of the grasp motion.
    hold = trajectory[:, :1, :].repeat(1, settle_steps + 1, 1)
    playback = torch.cat([hold, trajectory[:, 1:, :]], dim=1)

    # SimulationManager owns Spawn path expansion and the pre-prepare runtime
    # control lifecycle. DexSim supplies the substep q/qdot interpolation and FK.
    sim.register_kinematic_joint_trajectory("UR10", playback)


def main() -> None:
    """
    Main function to demonstrate robot simulation.

    This function initializes the simulation, creates the robot and cloth,
    and executes the pick-up trajectory.
    """
    parser = argparse.ArgumentParser(
        description="Create a simulation scene with SimulationManager"
    )
    add_env_launcher_args_to_parser(parser)
    parser.set_defaults(device="cuda", physics="newton")
    args = parser.parse_args()
    if args.physics != "newton":
        parser.error("Cloth requires --physics newton.")
    if not str(args.device).startswith("cuda"):
        parser.error("Newton MJVBD cloth simulation requires a CUDA device.")
    # Configure the simulation
    sim_cfg = SimulationManagerCfg(
        width=1920,
        height=1080,
        num_envs=args.num_envs,
        arena_space=args.arena_space,
        gpu_id=args.gpu_id,
        headless=args.headless,
        physics_dt=1.0 / 100.0,  # Physics timestep (100 Hz)
        device=args.device,
        render_cfg=RenderCfg(
            renderer=args.renderer
        ),  # Enable ray tracing for better visuals
        physics_cfg=NewtonPhysicsCfg(
            num_substeps=12,
            solver_cfg={
                "solver_type": "mjvbd",
                "iterations": 24,
                "particle_enable_self_contact": True,
                "particle_self_contact_radius": 0.002,
                "particle_self_contact_margin": 0.002,
                "particle_topological_contact_filter_threshold": 1,
                "particle_rest_shape_contact_exclusion_radius": 0.005,
                "particle_vertex_contact_buffer_size": 96,
                "particle_edge_contact_buffer_size": 128,
                "particle_collision_detection_interval": -1,
                "particle_enable_tile_solve": True,
                "soft_contact_margin": 0.008,
                "soft_contact_ke": CLOTH_RIGID_CONTACT_KE,
                "soft_contact_kd": CLOTH_RIGID_CONTACT_KD,
                "soft_contact_mu": CLOTH_RIGID_CONTACT_MU,
                # Use the mixed material stiffness immediately instead of
                # ramping new contacts from the low MJVBD default.
                "rigid_contact_k_start": CLOTH_RIGID_CONTACT_KE,
                "rigid_body_particle_contact_buffer_size": 512,
                "rigid_contact_max": 0,
                # The registered runtime control advances and interpolates the
                # robot kinematically at every Newton substep.
                "step_rigid_bodies": False,
                "self_contact_bvh_rebuild_interval_frames": 1,
            },
        ),
        visualization=visualization_cfg_from_args(args),
    )

    # Create the simulation instance
    sim = SimulationManager(sim_cfg)

    try:
        robot = create_robot(sim)
        create_cloth(sim)
        create_padding_box(sim)

        grasp_xpos = torch.tensor(
            [
                [
                    [-1, 0, 0, 0.5],
                    [0, 1, 0, 0],
                    [0, 0, -1, 0.075],
                    [0, 0, 0, 1],
                ],
            ],
            dtype=torch.float32,
            device=sim.device,
        )
        grasp_xpos = grasp_xpos.repeat(sim.num_envs, 1, 1)
        grab_traj = get_grasp_traj(sim, robot, grasp_xpos)
        settle_steps = 100
        register_kinematic_trajectory(sim, grab_traj, settle_steps)

        sim.prepare()
        if not args.headless:
            sim.open_window()
        sim.update(step=settle_steps)
        input("Press Enter to start grabbing the cloth...")

        # The initial trajectory row is already active after settling.
        sim.update(step=grab_traj.shape[1] - 1)
        input("Press Enter to exit the simulation...")
    finally:
        sim.destroy()


if __name__ == "__main__":
    main()
