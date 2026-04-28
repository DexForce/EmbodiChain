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

import os
from dexsim.utility.path import get_resources_data_path
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import ClothPhysicalAttributesCfg
from embodichain.lab.sim.shapes import MeshCfg
from embodichain.lab.sim.objects import ClothObjectCfg, ClothObject
import open3d as o3d
import pytest
import torch
import tempfile


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


class BaseSoftObjectTest:
    def setup_simulation(self):
        sim_cfg = SimulationManagerCfg(
            width=1920,
            height=1080,
            headless=True,
            physics_dt=1.0 / 100.0,  # Physics timestep (100 Hz)
            sim_device="cuda",
            num_envs=4,
            arena_space=3.0,
        )

        # Create the simulation instance
        self.sim = SimulationManager(sim_cfg)

        # Enable manual physics update for precise control
        self.n_envs = 4

        cloth_verts, cloth_faces = create_2d_grid_mesh(
            width=0.3, height=0.3, nx=12, ny=12
        )
        cloth_mesh = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(cloth_verts.to("cpu").numpy()),
            triangles=o3d.utility.Vector3iVector(cloth_faces.to("cpu").numpy()),
        )
        cloth_save_path = os.path.join(tempfile.gettempdir(), "cloth_mesh.ply")
        o3d.io.write_triangle_mesh(cloth_save_path, cloth_mesh)
        # add softbody to the scene
        self.cloth: ClothObject = self.sim.add_cloth_object(
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

    def test_run_simulation(self):
        self.sim.init_gpu_physics()
        for _ in range(100):
            self.sim.update(step=1)
        self.cloth.reset()
        for _ in range(100):
            self.sim.update(step=1)

    def test_remove(self):
        self.sim.remove_asset(self.cloth.uid)
        assert (
            self.cloth.uid not in self.sim._soft_objects
        ), "Cow UID still present after removal"

    def test_get_current_vertex_positions(self):
        vertex_positions = self.cloth.get_current_vertex_position()
        assert vertex_positions.shape == (
            self.sim.num_envs,
            self.cloth._data.n_vertices,
            3,
        ), "Vertex positions shape mismatch"

    def teardown_method(self):
        """Clean up resources after each test method."""
        self.sim.destroy()
        import embodichain.lab.sim as om
        om.SimulationManager.flush_cleanup_queue()
        self.__dict__.clear()
        import gc; gc.collect()


class TestSoftObjectCUDA(BaseSoftObjectTest):
    def setup_method(self):
        self.setup_simulation()


if __name__ == "__main__":
    test = TestSoftObjectCUDA()
    test.setup_method()
    test.test_run_simulation()
    test.test_get_current_vertex_positions()
