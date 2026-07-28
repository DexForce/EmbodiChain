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
This script demonstrates the creation and simulation of a robot with a cloth object,
and performs a pick-up task in a simulated environment.
"""

from __future__ import annotations

import argparse
import os
import tempfile

import numpy as np
import open3d as o3d
import torch

from embodichain.data import get_data_path
from embodichain.lab.sim.cfg import (
    ClothObjectCfg,
    ClothPhysicalAttributesCfg,
    RigidBodyAttributesCfg,
    RigidObjectCfg,
)
from embodichain.lab.sim.demo_base import DemoBase
from embodichain.lab.sim.objects import Robot
from embodichain.lab.sim.robots import URRobotCfg
from embodichain.lab.sim.shapes import CubeCfg, MeshCfg
from embodichain.lab.sim.utility.action_utils import interpolate_with_distance
from embodichain.lab.sim.utility.demo_utils import (
    add_demo_args,
    create_default_sim,
    maybe_init_gpu_physics,
    maybe_open_window,
    maybe_wait_for_user,
    setup_print_options,
)


class PickUpClothDemo(DemoBase):
    """Pick up a cloth with a UR10 arm and a parallel gripper."""

    def setup(self) -> None:
        """Create the simulation, robot, cloth and padding box, then settle."""
        self.sim = create_default_sim(
            self.args,
            arena_space=5.0,
            num_envs=self.args.num_envs,
            add_default_light=False,
        )
        self.robot = self._create_robot()
        self.cloth = self._create_cloth()
        self.padding_box = self._create_padding_box()
        maybe_init_gpu_physics(self.sim)
        maybe_open_window(self.sim, self.args)
        # Let the cloth settle before interaction.
        self.sim.update(step=10)

    def run(self) -> None:
        """Plan and replay the cloth pick-up trajectory."""
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
            device=self.sim.device,
        )
        grasp_xpos = grasp_xpos.repeat(self.sim.num_envs, 1, 1)
        grab_traj = self._get_grasp_traj(grasp_xpos)

        maybe_wait_for_user(self.args, "Press Enter to start grabbing cloth...")

        n_waypoint = grab_traj.shape[1]
        for i in range(n_waypoint):
            self.robot.set_qpos(grab_traj[:, i, :])
            self.sim.update(step=3)

        maybe_wait_for_user(self.args, "Press Enter to exit the simulation...")

    def _create_robot(self, position=[0.0, 0.0, 0.0]) -> Robot:
        """Create and configure a UR10 robot with a parallel gripper.

        Args:
            position: Initial root position of the robot.

        Returns:
            The configured robot instance added to the simulation.
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
        return self.sim.add_robot(cfg=cfg)

    def _create_padding_box(self):
        """Create a kinematic padding box used as a grasp reference."""
        padding_box_cfg = RigidObjectCfg(
            uid="padding_box",
            shape=CubeCfg(
                size=[0.02, 0.07, 0.05],
            ),
            attrs=RigidBodyAttributesCfg(
                mass=1.0,
                static_friction=0.01,
                dynamic_friction=0.00,
                restitution=0.01,
                min_position_iters=32,
                min_velocity_iters=8,
            ),
            body_type="kinematic",
            init_pos=[0.5, 0.0, 0.026],
            init_rot=[0.0, 0.0, 0.0],
        )
        padding_box = self.sim.add_rigid_object(cfg=padding_box_cfg)
        return padding_box

    def _create_2d_grid_mesh(
        self, width: float, height: float, nx: int = 1, ny: int = 1
    ):
        """Create a flat rectangle in the XY plane centered at the origin.

        The rectangle is subdivided into an ``nx`` by ``ny`` grid (cells) and
        triangulated. ``nx=1, ny=1`` yields the simple two-triangle rectangle.

        Returns vertices and triangles.
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
        idx = torch.arange((nx + 1) * (ny + 1), dtype=torch.int64).reshape(
            ny + 1, nx + 1
        )
        v0 = idx[:-1, :-1].reshape(-1)
        v1 = idx[:-1, 1:].reshape(-1)
        v2 = idx[1:, :-1].reshape(-1)
        v3 = idx[1:, 1:].reshape(-1)
        tri1 = torch.stack([v0, v1, v3], dim=1)
        tri2 = torch.stack([v0, v3, v2], dim=1)
        faces = torch.cat([tri1, tri2], dim=0).to(torch.int32)
        return verts, faces

    def _create_cloth(self):
        """Create the cloth object from a generated grid mesh."""
        cloth_verts, cloth_faces = self._create_2d_grid_mesh(
            width=0.3, height=0.3, nx=12, ny=12
        )
        cloth_mesh = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(cloth_verts.to("cpu").numpy()),
            triangles=o3d.utility.Vector3iVector(cloth_faces.to("cpu").numpy()),
        )
        cloth_save_path = os.path.join(tempfile.gettempdir(), "cloth_mesh.ply")
        o3d.io.write_triangle_mesh(cloth_save_path, cloth_mesh)

        cloth = self.sim.add_cloth_object(
            cfg=ClothObjectCfg(
                uid="cloth",
                shape=MeshCfg(fpath=cloth_save_path),
                init_pos=[0.5, 0.0, 0.3],
                init_rot=[0, 0, 0],
                physical_attr=ClothPhysicalAttributesCfg(
                    mass=0.01,
                    youngs=1e10,
                    poissons=0.4,
                    thickness=0.06,
                    bending_stiffness=0.01,
                    bending_damping=0.1,
                    dynamic_friction=0.95,
                    min_position_iters=30,
                ),
            ),
        )
        return cloth

    def _get_grasp_traj(self, grasp_xpos: torch.Tensor):
        """Compute the interpolated arm+hand trajectory for grasping the cloth."""
        n_envs = self.sim.num_envs
        rest_arm_qpos = self.robot.get_qpos("arm")

        approach_xpos = grasp_xpos.clone()
        approach_xpos[:, 2, 3] += 0.04
        _, qpos_approach = self.robot.compute_ik(
            pose=approach_xpos, joint_seed=rest_arm_qpos, name="arm"
        )
        _, qpos_grasp = self.robot.compute_ik(
            pose=grasp_xpos, joint_seed=qpos_approach, name="arm"
        )
        hand_open_qpos = torch.tensor(
            [0.00, 0.00], dtype=torch.float32, device=self.sim.device
        )
        hand_close_qpos = torch.tensor(
            [0.025, 0.025], dtype=torch.float32, device=self.sim.device
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
            trajectory=all_trajectory, interp_num=220, device=self.sim.device
        )
        return interp_trajectory


def main() -> None:
    """Entry point for the pick-up-cloth demo."""
    setup_print_options()
    parser = argparse.ArgumentParser(
        description="Create a simulation scene with SimulationManager"
    )
    parser = add_demo_args(parser)
    # Cloth simulation requires GPU physics; default to CUDA.
    parser.set_defaults(device="cuda")
    args = parser.parse_args()
    PickUpClothDemo(args).main()


if __name__ == "__main__":
    main()
