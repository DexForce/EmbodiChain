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
from embodichain.lab.sim.objects import RigidObject
from embodichain.lab.sim.cfg import RigidObjectCfg
from embodichain.lab.sim.shapes import MeshCfg
from embodichain.data import get_data_path
from dexsim.types import ActorType

DUCK_PATH = "ToyDuck/toy_duck.glb"
TABLE_PATH = "ShopTableSimple/shop_table_simple.ply"
CHAIR_PATH = "Chair/chair.glb"
NUM_ARENAS = 2
Z_TRANSLATION = 2.0


class BaseRigidObjectTest:
    """Shared test logic for CPU and CUDA."""

    def setup_simulation(self, sim_device):
        config = SimulationManagerCfg(headless=True, sim_device=sim_device)
        self.sim = SimulationManager(config)
        self.sim.build_multiple_arenas(NUM_ARENAS)

        duck_path = get_data_path(DUCK_PATH)
        assert os.path.isfile(duck_path)
        table_path = get_data_path(TABLE_PATH)
        assert os.path.isfile(table_path)
        chair_path = get_data_path(CHAIR_PATH)
        assert os.path.isfile(chair_path)

        cfg_dict = {
            "uid": "duck",
            "shape": {
                "shape_type": "Mesh",
                "fpath": duck_path,
            },
            "body_type": "dynamic",
        }
        self.duck: RigidObject = self.sim.add_rigid_object(
            cfg=RigidObjectCfg.from_dict(cfg_dict),
        )
        self.table: RigidObject = self.sim.add_rigid_object(
            cfg=RigidObjectCfg(
                uid="table", shape=MeshCfg(fpath=table_path), body_type="static"
            ),
        )

        self.chair: RigidObject = self.sim.add_rigid_object(
            cfg=RigidObjectCfg(
                uid="chair", shape=MeshCfg(fpath=chair_path), body_type="kinematic"
            ),
        )

        if sim_device == "cuda" and getattr(self.sim, "is_use_gpu_physics", False):
            self.sim.init_gpu_physics()

        self.sim.enable_physics(True)

    def test_is_static(self):
        """Test the is_static() method of duck, table, and chair objects."""
        assert not self.duck.is_static, "Duck should be dynamic but is marked static"
        assert self.table.is_static, "Table should be static but is marked dynamic"
        assert (
            not self.chair.is_static
        ), "Chair should be kinematic but is marked static"

    def test_local_pose_behavior(self):
        """Test set_local_pose and get_local_pose:
        - duck pose is correctly set
        - duck falls after physics update
        - table stays in place throughout
        - chair is kinematic and does not move
        """

        # Set initial poses
        pose_duck = torch.eye(4, device=self.sim.device)
        pose_duck[2, 3] = Z_TRANSLATION
        pose_duck = pose_duck.unsqueeze(0).repeat(NUM_ARENAS, 1, 1)

        pose_table = torch.eye(4, device=self.sim.device)
        pose_table = pose_table.unsqueeze(0).repeat(NUM_ARENAS, 1, 1)

        pose_chair = torch.eye(4, device=self.sim.device)
        pose_chair[0, 3] = 1.0
        pose_chair[1, 3] = 2.0
        pose_chair = pose_chair.unsqueeze(0).repeat(NUM_ARENAS, 1, 1)

        self.duck.set_local_pose(pose_duck)
        self.table.set_local_pose(pose_table)
        self.chair.set_local_pose(pose_chair)

        # --- Check poses immediately after setting
        duck_xyz = self.duck.get_local_pose()[0, :3]
        table_xyz = self.table.get_local_pose()[0, :3]
        chair_xyz = self.chair.get_local_pose()[0, :3]

        expected_duck_pos = torch.tensor(
            [0.0, 0.0, Z_TRANSLATION], device=self.sim.device, dtype=torch.float32
        )
        expected_table_pos = torch.tensor(
            [0.0, 0.0, 0.0], device=self.sim.device, dtype=torch.float32
        )
        expected_chair_pos = torch.tensor(
            [1.0, 2.0, 0.0], device=self.sim.device, dtype=torch.float32
        )

        assert torch.allclose(
            duck_xyz, expected_duck_pos, atol=1e-5
        ), f"FAIL: Duck pose not set correctly: {duck_xyz.tolist()}"
        assert torch.allclose(
            table_xyz, expected_table_pos, atol=1e-5
        ), f"FAIL: Table pose not set correctly: {table_xyz.tolist()}"
        assert torch.allclose(
            chair_xyz, expected_chair_pos, atol=1e-5
        ), f"FAIL: Chair pose not set correctly: {chair_xyz.tolist()}"

        # --- Step simulation
        for _ in range(10):
            self.sim.update(0.01)

        # --- Post-update checks
        duck_z_after = self.duck.get_local_pose()[0, 2].item()
        table_xyz_after = self.table.get_local_pose()[0, :3].tolist()
        chair_xyz_after = self.chair.get_local_pose()[0, :3]

        assert (
            duck_z_after < Z_TRANSLATION
        ), f"FAIL: Duck did not fall: z = {duck_z_after:.3f}"
        assert all(
            abs(x) < 1e-5 for x in table_xyz_after
        ), f"FAIL: Table moved unexpectedly: {table_xyz_after}"
        assert torch.allclose(
            chair_xyz_after, expected_chair_pos, atol=1e-5
        ), f"FAIL: Chair pose changed unexpectedly: {chair_xyz_after.tolist()}"

    def test_add_force_torque(self):
        """Test that add_force applies force correctly to the duck object."""

        pose_before = self.duck.get_local_pose()

        force = (
            torch.tensor([10.0, 0.0, 0], device=self.sim.device)
            .unsqueeze(0)
            .repeat(NUM_ARENAS, 1)
        )
        self.duck.add_force_torque(force)

        # Update simulation to apply the force
        self.sim.update(0.01)

        # Check if the duck's z position has changed
        pose_after = self.duck.get_local_pose()
        assert not torch.allclose(
            pose_before, pose_after
        ), "FAIL: Duck pose did not change after applying force"

        pose_before = self.duck.get_local_pose()
        torque = (
            torch.tensor([0.0, 10.0, 0.0], device=self.sim.device)
            .unsqueeze(0)
            .repeat(NUM_ARENAS, 1)
        )
        self.duck.add_force_torque(None, torque=torque)

        # Update simulation to apply the torque
        self.sim.update(0.01)

        pose_after = self.duck.get_local_pose()
        assert not torch.allclose(
            pose_before, pose_after
        ), "FAIL: Duck pose did not change after applying torque"

        # Test clear dynamics
        self.duck.clear_dynamics()

    def test_set_visual_material(self):
        """Test that set_material correctly assigns the material to the duck."""

        # Create blue material
        blue_mat = self.sim.create_visual_material(
            cfg=VisualMaterialCfg(base_color=[0.0, 0.0, 1.0, 1.0])
        )

        # Set it to the duck
        self.duck.set_visual_material(blue_mat)

        # # # Get material instances
        material_list = self.duck.get_visual_material_inst()

        # # Check correctness
        assert isinstance(material_list, list), "get_material() did not return a list"
        assert (
            len(material_list) == NUM_ARENAS
        ), f"Expected {NUM_ARENAS} materials, got {len(material_list)}"
        for mat_inst in material_list:
            assert mat_inst.base_color == [
                0.0,
                0.0,
                1.0,
                1.0,
            ], f"Material base color incorrect: {mat_inst.base_color}"

    def test_add_cube(self):
        cfg_dict = {
            "uid": "cube",
            "shape": {
                "shape_type": "Cube",
            },
            "body_type": "dynamic",
        }
        cube: RigidObject = self.sim.add_rigid_object(
            cfg=RigidObjectCfg.from_dict(cfg_dict),
        )

    def test_add_sphere(self):
        cfg_dict = {
            "uid": "sphere",
            "shape": {
                "shape_type": "Sphere",
            },
            "body_type": "dynamic",
        }
        sphere: RigidObject = self.sim.add_rigid_object(
            cfg=RigidObjectCfg.from_dict(cfg_dict),
        )

    def test_remove(self):
        self.sim.remove_asset(self.duck.uid)

        assert (
            self.duck.uid not in self.sim.asset_uids
        ), "Duck UID still present after removal"

    def test_set_physical_visible(self):
        self.table.set_physical_visible(
            visible=True,
            rgba=(0.1, 0.1, 0.9, 0.4),
        )
        self.table.set_physical_visible(visible=True)
        self.table.set_physical_visible(visible=False)

    def test_set_visible(self):
        self.table.set_visible(visible=True)
        self.table.set_visible(visible=False)

    def teardown_method(self):
        """Clean up resources after each test method."""
        self.sim.destroy()


class TestRigidObjectCPU(BaseRigidObjectTest):
    def setup_method(self):
        self.setup_simulation("cpu")


@pytest.mark.skip(reason="Skipping CUDA tests temporarily")
class TestRigidObjectCUDA(BaseRigidObjectTest):
    def setup_method(self):
        self.setup_simulation("cuda")


if __name__ == "__main__":
    # pytest.main(["-s", __file__])
    test = TestRigidObjectCPU()
    test.setup_method()
    test.test_set_visual_material()
    from IPython import embed

    embed()
