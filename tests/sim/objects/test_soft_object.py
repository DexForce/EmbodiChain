# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
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
from embodichain.lab.sim.cfg import (
    SoftbodyVoxelAttributesCfg,
    SoftbodyPhysicalAttributesCfg,
)
from embodichain.lab.sim.shapes import MeshCfg
from embodichain.lab.sim.objects import (
    SoftObject,
    SoftObjectCfg,
)
import pytest

COW_PATH = get_resources_data_path("Model", "cow", "cow.obj")


class BaseSoftObjectTest:
    def setup_simulation(self):
        sim_cfg = SimulationManagerCfg(
            width=1920,
            height=1080,
            headless=True,
            physics_dt=1.0 / 100.0,  # Physics timestep (100 Hz)
            sim_device="cuda",
            enable_rt=False,  # Enable ray tracing for better visuals
        )

        # Create the simulation instance
        self.sim = SimulationManager(sim_cfg)

        assert os.path.isfile(COW_PATH)

        # Enable manual physics update for precise control
        self.n_envs = 4
        # Build multiple arenas if requested
        self.sim.build_multiple_arenas(self.n_envs, space=3.0)
        # add softbody to the scene
        self.cow: SoftObject = self.sim.add_soft_object(
            cfg=SoftObjectCfg(
                uid="cow",
                shape=MeshCfg(
                    fpath=get_resources_data_path("Model", "cow", "cow.obj"),
                ),
                init_pos=[0.0, 0.0, 3.0],
                voxel_attr=SoftbodyVoxelAttributesCfg(
                    simulation_mesh_resolution=8,
                    maximal_edge_length=0.5,
                ),
                physical_attr=SoftbodyPhysicalAttributesCfg(
                    youngs=1e6,
                    poissons=0.45,
                    density=100,
                    dynamic_friction=0.1,
                    min_position_iters=30,
                ),
            ),
        )

    def test_run_simulation(self):
        self.sim.init_gpu_physics()
        for _ in range(100):
            self.sim.update(step=1)
        self.cow.reset()
        for _ in range(100):
            self.sim.update(step=1)

    def test_remove(self):
        self.sim.remove_asset(self.cow.uid)
        assert (
            self.cow.uid not in self.sim._soft_objects
        ), "Cow UID still present after removal"

    def teardown_method(self):
        """Clean up resources after each test method."""
        self.sim.destroy()


@pytest.mark.skip(reason="Skipping SoftObject test now")
class TestSoftObjectCUDA(BaseSoftObjectTest):
    def setup_method(self):
        self.setup_simulation()


if __name__ == "__main__":
    test = TestSoftObjectCUDA()
    test.setup_method()
    test.test_run_simulation()
