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
from embodichain.lab.sim.objects import Robot, SoftObject
from embodichain.lab.sim.utility.action_utils import interpolate_with_distance
from embodichain.lab.sim.shapes import MeshCfg
from embodichain.lab.sim.solvers import PytorchSolverCfg
from embodichain.data import get_data_path
from embodichain.utils import logger
from embodichain.lab.sim.cfg import (
    RenderCfg,
    JointDrivePropertiesCfg,
    RobotCfg,
    RigidObjectCfg,
    RigidBodyAttributesCfg,
    LightCfg,
    ClothObjectCfg,
    ClothPhysicalAttributesCfg,
    URDFCfg,
)
import os
from embodichain.lab.sim.shapes import MeshCfg, CubeCfg
import tempfile
from embodichain.lab.gym.utils.gym_utils import add_env_launcher_args_to_parser


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


def create_padding_box(sim: SimulationManager):
    padding_box_cfg = RigidObjectCfg(
        uid="padding_box",
        shape=CubeCfg(
            size=[0.01, 0.04, 0.03],
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
        init_pos=[0.5, 0.0, 0.01],
        init_rot=[0.0, 0.0, 0.0],
    )
    padding_box = sim.add_rigid_object(cfg=padding_box_cfg)
    return padding_box


def create_2d_grid_mesh(width: float, height: float, nx: int = 1, ny: int = 1):
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
    yy, xx = torch.meshgrid(y_lin, x_lin)  # shapes: (ny+1, nx+1)
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


def create_cloth(sim: SimulationManager):
    cloth_verts, cloth_faces = create_2d_grid_mesh(width=0.3, height=0.3, nx=12, ny=12)
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
            physical_attr=ClothPhysicalAttributesCfg(
                mass=0.01,
                youngs=1e10,
                poissons=0.4,
                thickness=0.04,
                bending_stiffness=0.01,
                bending_damping=0.1,
                dynamic_friction=0.95,
                min_position_iters=30,
            ),
        )
    )
    return cloth


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
    parser = argparse.ArgumentParser(
        description="Create a simulation scene with SimulationManager"
    )
    add_env_launcher_args_to_parser(parser)
    args = parser.parse_args()
    # Configure the simulation
    sim_cfg = SimulationManagerCfg(
        width=1920,
        height=1080,
        num_envs=args.num_envs,
        headless=True,
        physics_dt=1.0 / 100.0,  # Physics timestep (100 Hz)
        sim_device=args.device,
        render_cfg=RenderCfg(
            renderer=args.renderer
        ),  # Enable ray tracing for better visuals
    )

    # Create the simulation instance
    sim = SimulationManager(sim_cfg)

    robot = create_robot(sim)
    cloth = create_cloth(sim)
    padding_box = create_padding_box(sim)
    sim.init_gpu_physics()
    sim.open_window()
    sim.update(step=10)  # Let the cloth settle before interaction

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
    input("Press Enter to start grabing cloth...")

    n_waypoint = grab_traj.shape[1]
    for i in range(n_waypoint):
        robot.set_qpos(grab_traj[:, i, :])
        sim.update(step=4)
        time.sleep(1e-2)
    input("Press Enter to exit the simulation...")


if __name__ == "__main__":
    main()
