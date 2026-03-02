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

import pytest

from embodichain.lab.sim import (
    SimulationManager,
    SimulationManagerCfg,
)
from embodichain.lab.sim.objects import Articulation, RigidObject
from embodichain.lab.sim.cfg import ArticulationCfg, RigidObjectCfg
from embodichain.lab.sim.shapes import MeshCfg
from embodichain.data import get_data_path

NUM_ARENAS = 1


class BaseUsdTest:
    """Shared test logic for CPU and CUDA."""

    def setup_simulation(self, sim_device):
        config = SimulationManagerCfg(
            headless=True, sim_device=sim_device, num_envs=NUM_ARENAS, enable_rt=False
        )
        self.sim = SimulationManager(config)

        if sim_device == "cuda" and getattr(self.sim, "is_use_gpu_physics", False):
            self.sim.init_gpu_physics()

    def test_import_rigid(self):
        sugar_box_path = get_data_path("SugarBox/sugar_box_usd/sugar_box.usda")
        sugar_box: RigidObject = self.sim.add_rigid_object(
            cfg=RigidObjectCfg(
                uid="sugar_box",
                shape=MeshCfg(fpath=sugar_box_path),
                body_type="dynamic",
                use_usd_properties=True,
                init_pos=[0.0, 0.0, 0.1],
            )
        )

    def test_import_articulation(self):
        h1_path = get_data_path("UnitreeH1Usd/H1_usd/h1.usd")
        h1: Articulation = self.sim.add_articulation(
            cfg=ArticulationCfg(
                uid="h1",
                fpath=h1_path,
                build_pk_chain=False,
                use_usd_properties=False,
                init_pos=[0.0, 0.0, 1.2],
            )
        )

    def teardown_method(self):
        """Clean up resources after each test method."""
        self.sim.destroy()


class TestUsdCPU(BaseUsdTest):
    def setup_method(self):
        self.setup_simulation("cpu")


@pytest.mark.skip(reason="Skipping CUDA tests temporarily")
class TestUsdCUDA(BaseUsdTest):
    def setup_method(self):
        self.setup_simulation("cuda")


if __name__ == "__main__":
    test = TestUsdCPU()
    test.setup_method()
    test.test_import_rigid()
    test.test_import_articulation()

    from IPython import embed

    embed()
