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
This script demonstrates how to create a simulation scene using SimulationManager.
It shows the basic setup of simulation context, adding objects, lighting, and sensors.
"""

import argparse
import os
import tempfile
import time
import torch
import open3d as o3d
from dexsim.utility.path import get_resources_data_path
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.gym.utils.gym_utils import add_env_launcher_args_to_parser
from embodichain.lab.sim.cfg import (
    RenderCfg,
    RigidObjectCfg,
    RigidBodyAttributesCfg,
    ClothObjectCfg,
    ClothPhysicalAttributesCfg,
)
from embodichain.lab.sim.shapes import MeshCfg, CubeCfg
from embodichain.lab.sim.objects import ClothObject


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


def main():
    """Main function to create and run the simulation scene."""

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Create a simulation scene with SimulationManager"
    )
    add_env_launcher_args_to_parser(parser)
    args = parser.parse_args()

    # Configure the simulation
    sim_cfg = SimulationManagerCfg(
        width=1920,
        height=1080,
        headless=True,
        physics_dt=1.0 / 100.0,  # Physics timestep (100 Hz)
        sim_device="cuda",  # soft simulation only supports cuda device
        render_cfg=RenderCfg(renderer=args.renderer),
    )

    # Create the simulation instance
    sim = SimulationManager(sim_cfg)

    print("[INFO]: Scene setup complete!")

    cloth_verts, cloth_faces = create_2d_grid_mesh(width=0.3, height=0.3, nx=12, ny=12)
    cloth_mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(cloth_verts.to("cpu").numpy()),
        triangles=o3d.utility.Vector3iVector(cloth_faces.to("cpu").numpy()),
    )
    cloth_save_path = os.path.join(tempfile.gettempdir(), "cloth_mesh.ply")
    o3d.io.write_triangle_mesh(cloth_save_path, cloth_mesh)
    # add cloth to the scene
    cloth = sim.add_cloth_object(
        cfg=ClothObjectCfg(
            uid="cloth",
            shape=MeshCfg(fpath=cloth_save_path),
            init_pos=[0.5, 0.0, 0.3],
            init_rot=[0, 0, 0],
            physical_attr=ClothPhysicalAttributesCfg(
                mass=0.01,
                youngs=1e9,
                poissons=0.4,
                thickness=0.04,
                bending_stiffness=0.01,
                bending_damping=0.1,
                dynamic_friction=0.95,
                min_position_iters=30,
            ),
        )
    )
    padding_box_cfg = RigidObjectCfg(
        uid="padding_box",
        shape=CubeCfg(
            size=[0.1, 0.1, 0.06],
        ),
        attrs=RigidBodyAttributesCfg(
            mass=1.0,
            static_friction=0.95,
            dynamic_friction=0.9,
            restitution=0.01,
            min_position_iters=32,
            min_velocity_iters=8,
        ),
        body_type="dynamic",
        init_pos=[0.5, 0.0, 0.04],
        init_rot=[0.0, 0.0, 0.0],
    )
    padding_box = sim.add_rigid_object(cfg=padding_box_cfg)
    print("[INFO]: Add soft object complete!")

    # Open window when the scene has been set up
    if not args.headless:
        sim.open_window()

    print(f"[INFO]: Running simulation with {args.num_envs} environment(s)")
    print("[INFO]: Press Ctrl+C to stop the simulation")

    # Run the simulation
    run_simulation(sim, cloth)


def run_simulation(sim: SimulationManager, cloth: ClothObject) -> None:
    """Run the simulation loop.

    Args:
        sim: The SimulationManager instance to run
        soft_obj: soft object
    """

    # Initialize GPU physics
    sim.init_gpu_physics()

    step_count = 0

    try:
        last_time = time.time()
        last_step = 0
        while True:
            # Update physics simulation
            sim.update(step=1)
            step_count += 1

            # Print FPS every second
            if step_count % 100 == 0:
                current_time = time.time()
                elapsed = current_time - last_time
                fps = (
                    sim.num_envs * (step_count - last_step) / elapsed
                    if elapsed > 0
                    else 0
                )
                print(f"[INFO]: Simulation step: {step_count}, FPS: {fps:.2f}")
                last_time = current_time
                last_step = step_count
                if step_count % 500 == 0:
                    cloth.reset()

    except KeyboardInterrupt:
        print("\n[INFO]: Stopping simulation...")
    finally:
        # Clean up resources
        sim.destroy()
        print("[INFO]: Simulation terminated successfully")


if __name__ == "__main__":
    main()
