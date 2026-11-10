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

"""Explicit tetrahedral soft bodies and direct nodal-state writes."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import NewtonPhysicsCfg, VolumeDeformablePhysicsCfg
from embodichain.lab.sim.objects import (
    VolumeDeformableObject,
    VolumeDeformableObjectCfg,
)


def _cube_tet_mesh(cells: int, size: float) -> tuple[np.ndarray, np.ndarray]:
    """Structured cube with six tetrahedra per cell, vertices in x-fastest order."""
    n = cells + 1
    axis = np.linspace(-0.5 * size, 0.5 * size, n, dtype=np.float32)
    z, y, x = np.meshgrid(axis, axis, axis, indexing="ij")
    vertices = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)

    def vid(i: int, j: int, k: int) -> int:
        return i + n * (j + n * k)

    tets = []
    for k in range(cells):
        for j in range(cells):
            for i in range(cells):
                c = [
                    vid(i + dx, j + dy, k + dz)
                    for dz in (0, 1)
                    for dy in (0, 1)
                    for dx in (0, 1)
                ]
                for a, b in ((1, 3), (3, 2), (2, 6), (6, 4), (4, 5), (5, 1)):
                    tets.append([c[0], c[a], c[b], c[7]])
    return vertices, np.asarray(tets, dtype=np.int32)


class TestExplicitTetSoftBodyCUDA:
    def setup_method(self) -> None:
        self.vertices, self.tets = _cube_tet_mesh(cells=3, size=0.06)
        self.init_pos = np.array([0.0, 0.0, 0.2], dtype=np.float32)
        self.sim = SimulationManager(
            SimulationManagerCfg(
                headless=True,
                physics_dt=1.0 / 60.0,
                device="cuda",
                num_envs=1,
                physics_cfg=NewtonPhysicsCfg(solver_cfg={"solver_type": "vbd"}),
            )
        )
        self.block: VolumeDeformableObject = self.sim.add_deformable_object(
            cfg=VolumeDeformableObjectCfg(
                uid="block",
                shape={"vertices": self.vertices, "triangles": None},
                tetrahedra=self.tets,
                init_pos=self.init_pos.tolist(),
                attrs=VolumeDeformablePhysicsCfg(
                    youngs=3e4, poissons=0.35, density=250.0
                ),
            )
        )
        self.sim.prepare()

    def test_particles_follow_authored_vertex_order(self) -> None:
        positions = self.block.data.nodal_pos_w[0].cpu().numpy()
        assert positions.shape == self.vertices.shape
        np.testing.assert_allclose(positions, self.vertices + self.init_pos, atol=1e-6)

    def test_set_nodal_state_writes_positions_and_velocities(self) -> None:
        rest = self.block.data.nodal_pos_w
        squeezed = rest.clone()
        center_z = float(self.init_pos[2])
        squeezed[..., 2] = center_z + 0.9 * (rest[..., 2] - center_z)
        velocity = torch.zeros_like(rest)
        velocity[..., 0] = 0.01

        self.block.set_nodal_state(torch.cat((squeezed, velocity), dim=-1))

        torch.testing.assert_close(self.block.data.nodal_pos_w, squeezed)
        torch.testing.assert_close(self.block.data.nodal_vel_w, velocity)
        for _ in range(10):
            self.sim.update(step=1)
        assert torch.isfinite(self.block.data.nodal_pos_w).all()

    def test_set_nodal_state_rejects_bad_shape(self) -> None:
        with pytest.raises(ValueError, match="num_nodes, 6"):
            self.block.set_nodal_state(self.block.data.nodal_pos_w)

    def teardown_method(self) -> None:
        self.sim.destroy()
        SimulationManager.flush_cleanup_queue()
        self.__dict__.clear()
        import gc

        gc.collect()
