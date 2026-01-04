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

from embodichain.lab.sim import (
    SimulationManager,
    SimulationManagerCfg,
    VisualMaterialCfg,
)
from embodichain.lab.sim.objects import Articulation
from embodichain.lab.sim.cfg import ArticulationCfg
from embodichain.data import get_data_path
from dexsim.types import ActorType

ART_PATH = "SlidingBoxDrawer/SlidingBoxDrawer.urdf"
NUM_ARENAS = 10


class BaseArticulationTest:
    """Shared test logic for CPU and CUDA."""

    def setup_simulation(self, sim_device):
        config = SimulationManagerCfg(
            headless=True, sim_device=sim_device, num_envs=NUM_ARENAS
        )
        self.sim = SimulationManager(config)

        art_path = get_data_path(ART_PATH)
        assert os.path.isfile(art_path)

        cfg_dict = {"fpath": art_path}
        self.art: Articulation = self.sim.add_articulation(
            cfg=ArticulationCfg.from_dict(cfg_dict)
        )

        if sim_device == "cuda" and getattr(self.sim, "is_use_gpu_physics", False):
            self.sim.init_gpu_physics()

    def test_local_pose_behavior(self):
        """Test set_local_pose and get_local_pose:
        - Drawer pose is correctly set
        """

        # Set initial poses
        pose = torch.eye(4, device=self.sim.device)
        pose[2, 3] = 1.0
        pose = pose.unsqueeze(0).repeat(NUM_ARENAS, 1, 1)

        self.art.set_local_pose(pose, env_ids=None)

        # --- Check poses immediately after setting
        xyz = self.art.get_local_pose()[0, :3]

        expected_pos = torch.tensor(
            [0.0, 0.0, 1.0], device=self.sim.device, dtype=torch.float32
        )
        assert torch.allclose(
            xyz, expected_pos, atol=1e-5
        ), f"FAIL: Drawer pose not set correctly: {xyz.tolist()}"

    def test_control_api(self):
        """Test control API for setting and getting joint positions."""
        # Set initial joint positions
        qpos_zero = torch.zeros(
            (NUM_ARENAS, self.art.dof), dtype=torch.float32, device=self.sim.device
        )
        qpos = qpos_zero.clone()
        qpos[:, -1] = 0.1

        # Test setting joint positions directly.
        self.art.set_qpos(qpos, env_ids=None, target=False)
        target_qpos = self.art.body_data.qpos
        assert torch.allclose(
            target_qpos, qpos, atol=1e-5
        ), f"FAIL: Joint positions not set correctly: {target_qpos.tolist()}"

        self.art.set_qpos(qpos=qpos_zero, env_ids=None, target=False)

        # Test setting joint positions with target=True
        self.art.set_qpos(qpos, env_ids=None, target=True)
        self.sim.update(step=100)
        target_qpos = self.art.body_data.qpos
        assert torch.allclose(
            target_qpos, qpos, atol=1e-5
        ), f"FAIL: Joint positions not set correctly with target=True: {target_qpos.tolist()}"

        self.art.set_qpos(qpos=qpos_zero, env_ids=None, target=False)
        self.art.clear_dynamics()

        # Test setting joint forces
        qf = torch.ones(
            (NUM_ARENAS, self.art.dof), dtype=torch.float32, device=self.sim.device
        )
        self.art.set_qf(qf, env_ids=None)
        target_qf = self.art.body_data.qf
        assert torch.allclose(
            target_qf, qf, atol=1e-5
        ), f"FAIL: Joint forces not set correctly: {target_qf.tolist()}"
        print("Applying joint forces...")
        print(f"qpos before applying force: {qpos_zero.tolist()}")
        print(f"qf before applying force: {qf.tolist()}")

        self.sim.update(step=100)
        target_qpos = self.art.body_data.qpos
        print(f"target_qpos: {target_qpos}")
        print(f"qpos_zero: {qpos_zero}")
        print("qpos diff:", target_qpos - qpos_zero)
        # check target_qpos is greater than qpos
        assert torch.any(
            (target_qpos - qpos_zero).abs() > 1e-4
        ), f"FAIL: Target qpos did not change after applying force: {target_qpos.tolist()}"

    def test_set_visual_material(self):
        """Test setting visual material properties."""
        # Create blue material
        blue_mat = self.sim.create_visual_material(
            cfg=VisualMaterialCfg(base_color=[0.0, 0.0, 1.0, 1.0])
        )

        self.art.set_visual_material(blue_mat, link_names=["outer_box", "handle_xpos"])

        mat_insts = self.art.get_visual_material_inst()

        assert (
            len(mat_insts) == 10
        ), f"FAIL: Expected 10 material instances, got {len(mat_insts)}"
        assert (
            "outer_box" in mat_insts[0]
        ), "FAIL: 'outer_box' not in material instances"
        assert (
            "handle_xpos" in mat_insts[0]
        ), "FAIL: 'handle_xpos' not in material instances"
        assert mat_insts[0]["outer_box"].base_color == [
            0.0,
            0.0,
            1.0,
            1.0,
        ], f"FAIL: 'outer_box' base color not set correctly: {mat_insts[0]['outer_box'].base_color}"
        assert mat_insts[0]["handle_xpos"].base_color == [
            0.0,
            0.0,
            1.0,
            1.0,
        ], f"FAIL: 'handle_xpos' base color not set correctly: {mat_insts[0]['handle_xpos'].base_color}"

    # TODO: Open this test will cause segfault in CI env
    # def test_get_link_pose(self):
    #     """Test getting link poses."""
    #     poses = self.art.get_link_pose(link_name="handle_xpos", to_matrix=False)
    #     assert poses.shape == (
    #         NUM_ARENAS,
    #         7,
    #     ), f"FAIL: Expected poses shape {(NUM_ARENAS, 7)}, got {poses.shape}"

    def test_remove_articulation(self):
        """Test removing an articulation from the simulation."""
        self.sim.remove_asset(self.art.uid)
        assert (
            self.art.uid not in self.sim.asset_uids
        ), "FAIL: Articulation UID still present after removal"

    def test_set_physical_visible(self):
        self.art.set_physical_visible(
            visible=True,
            rgba=(0.1, 0.1, 0.9, 0.4),
        )
        self.art.set_physical_visible(visible=False)
        all_link_names = self.art.link_names
        self.art.set_physical_visible(visible=True, link_names=all_link_names[:3])

    def teardown_method(self):
        """Clean up resources after each test method."""
        self.sim.destroy()


class TestArticulationCPU(BaseArticulationTest):
    def setup_method(self):
        self.setup_simulation("cpu")


@pytest.mark.skip(reason="Skipping CUDA tests temporarily")
class TestArticulationCUDA(BaseArticulationTest):
    def setup_method(self):
        self.setup_simulation("cuda")


if __name__ == "__main__":
    test = TestArticulationCPU()
    test.setup_method()
    test.test_set_visual_material()
