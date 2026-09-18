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

from __future__ import annotations

import os
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import (
    NewtonPhysicsCfg,
    VolumeDeformableMeshingCfg,
    VolumeDeformablePhysicsCfg,
)
from embodichain.lab.sim.shapes import MeshCfg
from embodichain.lab.sim.objects import (
    DeformableObject,
    DeformableObjectData,
    VolumeDeformableObject,
    VolumeDeformableObjectCfg,
)
from embodichain.data import get_data_path
import pytest
import torch

COW_PATH = get_data_path("Cow/cow2.obj")


class BaseSoftObjectTest:
    def setup_simulation(self):
        sim_cfg = SimulationManagerCfg(
            width=1920,
            height=1080,
            headless=True,
            physics_dt=1.0 / 100.0,  # Physics timestep (100 Hz)
            device="cuda",
            # DexSim 0.5 currently changes the cow render topology while
            # cloning it; keep the functional volume test single-instance.
            num_envs=1,
            arena_space=3.0,
            physics_cfg=NewtonPhysicsCfg(
                solver_cfg={"solver_type": "vbd"},
            ),
        )

        # Create the simulation instance
        self.sim = SimulationManager(sim_cfg)

        assert os.path.isfile(COW_PATH)

        self.num_envs = 1

        # add softbody to the scene
        self.cow: VolumeDeformableObject = self.sim.add_deformable_object(
            cfg=VolumeDeformableObjectCfg(
                uid="cow",
                shape=MeshCfg(
                    fpath=get_data_path("Cow/cow2.obj"),
                ),
                init_pos=[0.0, 0.0, 3.0],
                meshing=VolumeDeformableMeshingCfg(
                    simulation_mesh_resolution=8,
                ),
                attrs=VolumeDeformablePhysicsCfg(
                    youngs=1e6,
                    poissons=0.45,
                    density=100,
                    elasticity_damping=0.1,
                ),
            ),
        )
        self.sim.prepare()

    def test_run_simulation(self):
        for _ in range(100):
            self.sim.update(step=1)
        self.cow.reset()
        for _ in range(100):
            self.sim.update(step=1)

    def test_get_deformable_mesh_geometry(self):
        """Test current collision vertices and matching surface triangles."""
        vertices = self.cow.data.nodal_pos_w
        triangles = self.cow.get_collision_surface_triangles(env_ids=[0])

        assert vertices.ndim == 3 and vertices.shape[0] == self.sim.num_envs
        assert triangles.ndim == 3 and triangles.shape[0] == 1
        assert int(triangles.max()) < vertices.shape[1]

    def test_set_local_pose_updates_selected_particle_batch(self):
        """Setting one instance pose writes only its packed simulation nodes."""
        before = self.cow.data.nodal_pos_w
        translation = torch.tensor([0.5, 0.0, 0.0], device=self.cow.device)
        pose = torch.eye(
            4,
            dtype=torch.float32,
            device=self.cow.device,
        ).unsqueeze(0)
        pose[:, :3, 3] = (
            torch.as_tensor(
                self.cow.cfg.init_pos,
                dtype=torch.float32,
                device=self.cow.device,
            )
            + translation
        )

        self.cow.set_local_pose(pose, env_ids=[0])
        after = self.cow.data.nodal_pos_w

        torch.testing.assert_close(after[0], before[0] + translation)
        torch.testing.assert_close(after[1:], before[1:])

    def test_unified_deformable_contract(self):
        assert isinstance(self.cow, DeformableObject)
        assert isinstance(self.cow, VolumeDeformableObject)
        assert self.cow.deformable_type == "volume"
        assert self.sim.get_deformable_object("cow") is self.cow
        assert self.sim.get_deformable_object_uid_list() == ["cow"]

        assert type(self.cow.data) is DeformableObjectData
        positions = self.cow.data.nodal_pos_w
        velocities = self.cow.data.nodal_vel_w
        state = self.cow.data.nodal_state_w
        default_state = self.cow.data.default_nodal_state_w
        assert positions.shape[-1] == 3
        assert velocities.shape == positions.shape
        assert state.shape == (*positions.shape[:-1], 6)
        assert default_state.shape == state.shape
        render_vertices = self.cow.get_surface_vertices()
        render_triangles = self.cow.get_surface_triangles(env_ids=[0])
        assert render_vertices.shape[0] == self.sim.num_envs
        assert int(render_triangles.max()) < render_vertices.shape[1]

    def test_remove(self):
        with pytest.raises(NotImplementedError, match="pending removal"):
            self.sim.remove_asset(self.cow.uid)
        assert self.sim.get_deformable_object(self.cow.uid) is self.cow

    def teardown_method(self):
        """Clean up resources after each test method."""
        self.sim.destroy()
        import embodichain.lab.sim as om

        om.SimulationManager.flush_cleanup_queue()
        self.__dict__.clear()
        import gc

        gc.collect()


class TestSoftObjectCUDA(BaseSoftObjectTest):
    def setup_method(self):
        self.setup_simulation()


if __name__ == "__main__":
    test = TestSoftObjectCUDA()
    test.setup_method()
    test.test_run_simulation()
