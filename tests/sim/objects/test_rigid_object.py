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
import torch
import pytest

from embodichain.lab.sim import (
    SimulationManager,
    SimulationManagerCfg,
    VisualMaterialCfg,
)
from embodichain.lab.sim.objects import RigidObject
from embodichain.lab.sim.cfg import RigidObjectCfg, RigidBodyAttributesCfg
from embodichain.lab.sim.shapes import MeshCfg
from embodichain.data import get_data_path
from dexsim.types import ActorType

from embodichain.lab.sim.cfg import (
    RenderCfg,
    RigidObjectCfg
)

DUCK_PATH = "ToyDuck/toy_duck.glb"
TABLE_PATH = "ShopTableSimple/shop_table_simple.ply"
CHAIR_PATH = "Chair/chair.glb"
NUM_ARENAS = 2
Z_TRANSLATION = 2.0


class BaseRigidObjectTest:
    """Shared test logic for CPU and CUDA."""

    def setup_simulation(self, sim_device):
        config = SimulationManagerCfg(render_cfg=RenderCfg(renderer="hybrid"),
            headless=True, sim_device=sim_device, num_envs=NUM_ARENAS
        )
        self.sim = SimulationManager(config)

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
            "attrs": {
                "mass": 1.0,
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

    def test_set_velocity(self):
        """Test that set_velocity correctly assigns linear and angular velocity to the duck."""

        lin_vel = (
            torch.tensor([0.0, 5.0, 0.0], device=self.sim.device)
            .unsqueeze(0)
            .repeat(NUM_ARENAS, 1)
        )
        ang_vel = (
            torch.tensor([0.0, 0.0, 5.0], device=self.sim.device)
            .unsqueeze(0)
            .repeat(NUM_ARENAS, 1)
        )
        self.duck.set_velocity(lin_vel=lin_vel, ang_vel=ang_vel)

        # Read back velocities from the duck and verify they match the values set.
        duck_lin_vel = self.duck.body_data.lin_vel
        duck_ang_vel = self.duck.body_data.ang_vel

        assert duck_lin_vel.shape == lin_vel.shape, (
            f"Linear velocity shape mismatch: expected {lin_vel.shape}, "
            f"got {duck_lin_vel.shape}"
        )
        assert duck_ang_vel.shape == ang_vel.shape, (
            f"Angular velocity shape mismatch: expected {ang_vel.shape}, "
            f"got {duck_ang_vel.shape}"
        )

        assert torch.allclose(
            duck_lin_vel, lin_vel
        ), f"Linear velocity not set correctly: expected {lin_vel}, got {duck_lin_vel}"
        assert torch.allclose(
            duck_ang_vel, ang_vel
        ), f"Angular velocity not set correctly: expected {ang_vel}, got {duck_ang_vel}"

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

    def test_add_sdf_mesh(self):
        duck_path = get_data_path(DUCK_PATH)
        sdf = self.sim.add_rigid_object(
            cfg=RigidObjectCfg(
                uid="duck_sdf",
                shape=MeshCfg(fpath=duck_path),
                sdf_resolution=128,
                body_type="dynamic",
            )
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

    def test_body_data(self):
        """Test the body_data property for dynamic objects."""
        # Dynamic object should have body_data
        assert self.duck.body_data is not None, "Dynamic duck should have body_data"

        # Static object should return None with warning
        assert self.table.body_data is None, "Static table should not have body_data"

        # Kinematic object should have body_data
        assert self.chair.body_data is not None, "Kinematic chair should have body_data"

    def test_physical_attributes(self):
        """Test getting and setting physical attributes and body states."""
        # 1. Body state
        lin_vel = (
            torch.tensor([1.0, 0.0, 0.0], device=self.sim.device)
            .unsqueeze(0)
            .repeat(NUM_ARENAS, 1)
        )
        ang_vel = (
            torch.tensor([0.0, 0.0, 1.0], device=self.sim.device)
            .unsqueeze(0)
            .repeat(NUM_ARENAS, 1)
        )
        self.duck.set_velocity(lin_vel=lin_vel, ang_vel=ang_vel)

        body_state = self.duck.body_state
        assert body_state.shape == (
            NUM_ARENAS,
            13,
        ), f"Body state shape should be (NUM_ARENAS, 13), got {body_state.shape}"
        assert torch.allclose(
            body_state[:, 7:10], lin_vel, atol=1e-5
        ), "Linear velocity in body_state doesn't match"
        assert torch.allclose(
            body_state[:, 10:13], ang_vel, atol=1e-5
        ), "Angular velocity in body_state doesn't match"

        table_state = self.table.body_state
        assert torch.allclose(
            table_state[:, 7:], torch.zeros_like(table_state[:, 7:])
        ), "Static object should have zero velocities in body_state"

        # 2. is_non_dynamic
        assert not self.duck.is_non_dynamic, "Dynamic duck should not be is_non_dynamic"
        assert self.table.is_non_dynamic, "Static table should be is_non_dynamic"
        assert self.chair.is_non_dynamic, "Kinematic chair should be is_non_dynamic"

        # 3. body_type
        assert self.duck.body_type == "dynamic"
        self.duck.set_body_type("kinematic")
        assert self.duck.body_type == "kinematic"
        self.duck.set_body_type("dynamic")
        assert self.duck.body_type == "dynamic"

        assert self.chair.body_type == "kinematic"
        self.chair.set_body_type("dynamic")
        assert self.chair.body_type == "dynamic"
        self.chair.set_body_type("kinematic")
        assert self.chair.body_type == "kinematic"

        # 4. attrs
        new_attrs = RigidBodyAttributesCfg(mass=2.5, density=1000.0)
        self.duck.set_attrs(new_attrs)
        masses = self.duck.get_mass()
        assert torch.allclose(
            masses, torch.tensor([2.5] * NUM_ARENAS, device=self.sim.device)
        ), f"Mass not set correctly: {masses.tolist()}"

        partial_attrs = RigidBodyAttributesCfg(mass=3.0)
        self.duck.set_attrs(partial_attrs, env_ids=[0])
        masses = self.duck.get_mass()
        assert torch.allclose(
            masses[0], torch.tensor(3.0, device=self.sim.device)
        ), "Mass for env_id 0 should be 3.0"

        # 5. mass, friction, damping, inertia, scale
        new_mass = (
            torch.tensor([1.5, 2.5], device=self.sim.device)
            if NUM_ARENAS == 2
            else torch.ones(NUM_ARENAS, device=self.sim.device) * 2.0
        )
        self.duck.set_mass(new_mass)
        assert torch.allclose(self.duck.get_mass(), new_mass), f"Mass not set correctly"

        new_friction = (
            torch.tensor([0.5, 0.7], device=self.sim.device)
            if NUM_ARENAS == 2
            else torch.ones(NUM_ARENAS, device=self.sim.device) * 0.6
        )
        self.duck.set_friction(new_friction)
        assert torch.allclose(
            self.duck.get_friction(), new_friction, atol=1e-5
        ), f"Friction not set correctly"

        new_damping = (
            torch.tensor([[0.1, 0.2], [0.3, 0.4]], device=self.sim.device)
            if NUM_ARENAS == 2
            else torch.ones(NUM_ARENAS, 2, device=self.sim.device) * 0.15
        )
        self.duck.set_damping(new_damping)
        assert torch.allclose(
            self.duck.get_damping()[:, 0], new_damping[:, 0], atol=1e-5
        ), "Linear damping not set correctly"

        new_inertia = (
            torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device=self.sim.device)
            if NUM_ARENAS == 2
            else torch.ones(NUM_ARENAS, 3, device=self.sim.device) * 1.0
        )
        self.duck.set_inertia(new_inertia)
        assert torch.allclose(
            self.duck.get_inertia(), new_inertia, atol=1e-5
        ), f"Inertia not set correctly"

        new_scale = (
            torch.tensor([[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]], device=self.sim.device)
            if NUM_ARENAS == 2
            else torch.ones(NUM_ARENAS, 3, device=self.sim.device) * 2.0
        )
        self.duck.set_body_scale(new_scale)
        assert torch.allclose(
            self.duck.get_body_scale(), new_scale
        ), f"Body scale not set correctly"

        # 6. COM pose
        com_pose = torch.zeros((NUM_ARENAS, 7), device=self.sim.device)
        com_pose[:, 3] = 1.0  # Unit quaternion
        com_pose[0, :3] = torch.tensor([0.1, 0.1, 0.1], device=self.sim.device)

        self.duck.set_com_pose(com_pose)

        # Static object should not be able to set COM pose
        self.table.set_com_pose(com_pose)  # Should log warning but not crash

        assert self.duck.body_data is not None
        assert self.duck.body_data.default_com_pose is not None
        assert self.duck.body_data.default_com_pose.shape == (
            NUM_ARENAS,
            7,
        ), f"Default COM pose should have shape (NUM_ARENAS, 7)"

        com_pose = self.duck.body_data.com_pose
        assert isinstance(com_pose, torch.Tensor), "com_pose should be a torch.Tensor"
        assert com_pose.shape == (
            NUM_ARENAS,
            7,
        ), f"COM pose should have shape (NUM_ARENAS, 7), got {com_pose.shape}"

    def test_misc_properties(self):
        """Test miscellaneous properties like collision filter, vertices, and visual materials."""
        # 1. Collision filter
        filter_data = torch.zeros((NUM_ARENAS, 4), dtype=torch.int32)
        for i in range(NUM_ARENAS):
            filter_data[i, 0] = i + 10  # Set arena id
            filter_data[i, 1] = 1

        self.duck.set_collision_filter(filter_data)

        # 2. Vertices
        vertices = self.duck.get_vertices()

        assert isinstance(
            vertices, torch.Tensor
        ), "get_vertices should return a torch.Tensor"
        assert (
            len(vertices.shape) == 3
        ), f"Vertices should have shape (N, num_verts, 3), got {vertices.shape}"
        assert (
            vertices.shape[0] == NUM_ARENAS
        ), f"First dimension should be {NUM_ARENAS}, got {vertices.shape[0]}"
        assert (
            vertices.shape[2] == 3
        ), f"Last dimension should be 3, got {vertices.shape[2]}"

        partial_vertices = self.duck.get_vertices(env_ids=[0])
        assert partial_vertices.shape[0] == 1, "Should get vertices for 1 instance"

        # 3. User IDs
        user_ids = self.duck.get_user_ids()

        assert isinstance(
            user_ids, torch.Tensor
        ), "get_user_ids should return a torch.Tensor"
        assert user_ids.shape == (
            NUM_ARENAS,
        ), f"User IDs should have shape ({NUM_ARENAS},), got {user_ids.shape}"
        assert (
            user_ids.dtype == torch.int32
        ), f"User IDs should be int32, got {user_ids.dtype}"

        # 4. Share material
        blue_mat = self.sim.create_visual_material(
            cfg=VisualMaterialCfg(base_color=[0.0, 0.0, 1.0, 1.0])
        )
        self.duck.set_visual_material(blue_mat)

        duck_materials = self.duck.get_visual_material_inst()

        cfg_dict = {
            "uid": "test_cube",
            "shape": {"shape_type": "Cube"},
            "body_type": "dynamic",
        }
        cube = self.sim.add_rigid_object(
            cfg=RigidObjectCfg.from_dict(cfg_dict),
        )

        cube.share_visual_material_inst(duck_materials)

        cube_materials = cube.get_visual_material_inst()
        assert (
            len(cube_materials) == NUM_ARENAS
        ), f"Cube should have {NUM_ARENAS} material instances"
        for i in range(NUM_ARENAS):
            assert cube_materials[i].base_color == [
                0.0,
                0.0,
                1.0,
                1.0,
            ], f"Material {i} base color incorrect"

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
