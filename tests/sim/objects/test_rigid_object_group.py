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
import torch
import pytest

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.objects import RigidObjectGroup
from embodichain.lab.sim.cfg import RigidObjectGroupCfg, RigidObjectCfg
from embodichain.lab.sim.shapes import MeshCfg
from embodichain.data import get_data_path
from dexsim.types import ActorType

DUCK_PATH = "ToyDuck/toy_duck.glb"
TABLE_PATH = "ShopTableSimple/shop_table_simple.ply"
NUM_ARENAS = 4
Z_TRANSLATION = 2.0


class BaseRigidObjectGroupTest:
    """Shared test logic for CPU and CUDA."""

    def setup_simulation(self, sim_device):
        config = SimulationManagerCfg(
            headless=True, sim_device=sim_device, num_envs=NUM_ARENAS
        )
        self.sim = SimulationManager(config)

        duck_path = get_data_path(DUCK_PATH)
        assert os.path.isfile(duck_path)
        table_path = get_data_path(TABLE_PATH)
        assert os.path.isfile(table_path)

        cfg_dict = {
            "uid": "group",
            "rigid_objects": {
                "duck1": {
                    "shape": {
                        "shape_type": "Mesh",
                        "fpath": duck_path,
                    },
                },
                "duck2": {
                    "shape": {
                        "shape_type": "Mesh",
                        "fpath": duck_path,
                    },
                },
            },
        }
        self.obj_group: RigidObjectGroup = self.sim.add_rigid_object_group(
            cfg=RigidObjectGroupCfg.from_dict(cfg_dict)
        )

        if sim_device == "cuda" and self.sim.is_use_gpu_physics:
            self.sim.init_gpu_physics()

        self.sim.enable_physics(True)

    def test_local_pose_behavior(self):

        # Set initial poses
        pose_duck1 = torch.eye(4, device=self.sim.device)
        pose_duck1[2, 3] = Z_TRANSLATION
        pose_duck1 = pose_duck1.unsqueeze(0).repeat(NUM_ARENAS, 1, 1)

        pose_duck2 = torch.eye(4, device=self.sim.device)
        pose_duck2[2, 3] = Z_TRANSLATION
        pose_duck2 = pose_duck2.unsqueeze(0).repeat(NUM_ARENAS, 1, 1)

        combined_pose = torch.stack([pose_duck1, pose_duck2], dim=1)

        self.obj_group.set_local_pose(combined_pose)
        group_pos = self.obj_group.get_local_pose()[..., :3]
        assert torch.allclose(
            group_pos,
            combined_pose[..., :3, 3],
            atol=1e-5,
        ), "FAIL: Local poses do not match after setting."

    def test_get_user_ids(self):
        """Test get_user_ids method."""
        user_ids = self.obj_group.get_user_ids()

        assert user_ids.shape == (NUM_ARENAS, self.obj_group.num_objects), (
            f"Unexpected user_ids shape: {user_ids.shape}, "
            f"expected ({NUM_ARENAS}, {self.obj_group.num_objects})"
        )

    def test_remove(self):
        self.sim.remove_asset(self.obj_group.uid)

        assert (
            self.obj_group.uid not in self.sim.asset_uids
        ), "Object group UID still present after removal"

    def test_set_physical_visible(self):
        self.obj_group.set_physical_visible(visible=True)
        self.obj_group.set_physical_visible(visible=False)

    def test_set_visible(self):
        self.obj_group.set_visible(visible=True)
        self.obj_group.set_visible(visible=False)

    def teardown_method(self):
        """Clean up resources after each test method."""
        self.sim.destroy()


class TestRigidObjectGroupCPU(BaseRigidObjectGroupTest):
    def setup_method(self):
        self.setup_simulation("cpu")


# TODO: Fix CUDA tests issue.
@pytest.mark.skip(reason="Skipping CUDA tests temporarily")
class TestRigidObjectGroupCUDA(BaseRigidObjectGroupTest):
    def setup_method(self):
        self.setup_simulation("cuda")


if __name__ == "__main__":
    # pytest.main(["-s", __file__])
    test = TestRigidObjectGroupCPU()
    test.setup_method()
    test.test_local_pose_behavior()
