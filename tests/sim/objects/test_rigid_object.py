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

import pytest
import torch

from embodichain.lab.sim import (
    SimulationManager,
    SimulationManagerCfg,
    VisualMaterialCfg,
)
from embodichain.data import get_data_path
from embodichain.lab.sim.cfg import RigidObjectCfg, physics_cfg_for_backend
from embodichain.lab.sim.cfg import RigidBodyAttributesCfg
from embodichain.lab.sim.cfg import NewtonCollisionAttributesCfg
from embodichain.lab.sim.objects import RigidObject
from embodichain.lab.sim.shapes import MeshCfg

DUCK_PATH = "ToyDuck/toy_duck.glb"
TABLE_PATH = "ShopTableSimple/shop_table_simple.ply"
CHAIR_PATH = "Chair/chair.glb"
NUM_ARENAS = 2
Z_TRANSLATION = 2.0


def _make_test_com_pose(device: torch.device) -> torch.Tensor:
    """Create per-env COM poses using EmbodiChain xyzw quaternion convention."""
    return torch.tensor(
        [
            [0.04, -0.02, 0.03, 0.0, 0.0, 0.0, 1.0],
            [-0.01, 0.05, 0.02, 0.0, 0.0, 0.70710677, 0.70710677],
        ],
        device=device,
        dtype=torch.float32,
    )


def _teardown_newton_physics() -> None:
    from dexsim.engine.newton_physics import teardown_newton_physics

    teardown_newton_physics()


class BaseRigidObjectTest:
    """Shared test logic for CPU and CUDA."""

    def setup_simulation(self, device: str, physics: str = "default"):
        config = SimulationManagerCfg(
            headless=True,
            device=device,
            num_envs=NUM_ARENAS,
            physics_cfg=physics_cfg_for_backend(physics),
        )
        self.sim = SimulationManager(config)
        self.physics = physics
        self.sim.enable_physics(False)
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

        if (
            physics == "default"
            and device == "cuda"
            and getattr(self.sim, "is_use_gpu_physics", False)
        ):
            self.sim.init_gpu_physics()

        self.sim.enable_physics(True)
        if physics == "newton":
            self.sim.finalize_newton_physics()

    def test_is_static(self):
        """Test the is_static() method of duck, table, and chair objects."""
        assert not self.duck.is_static, "Duck should be dynamic but is marked static"
        assert self.table.is_static, "Table should be static but is marked dynamic"
        assert (
            not self.chair.is_static
        ), "Chair should be kinematic but is marked static"

    def test_spawn_clones_distinct_entities(self):
        """Multi-env rigid objects are spawned via prototype + clone_actor_to."""
        assert len(self.duck._entities) == NUM_ARENAS
        handles = {entity.get_native_handle() for entity in self.duck._entities}
        assert len(handles) == NUM_ARENAS, "Each arena clone must be a distinct actor"
        assert self.duck._entities[0].get_name() == "duck_0"
        assert self.duck._entities[1].get_name() == "duck_1"

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
        if self.physics != "newton":
            assert torch.allclose(
                chair_xyz_after, expected_chair_pos, atol=1e-5
            ), f"FAIL: Chair pose changed unexpectedly: {chair_xyz_after.tolist()}"
        # Newton: kinematic bodies are not pose-locked yet (DexSim TODO).

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

    def test_get_acceleration(self):
        """Test that lin_acc, ang_acc, and acc return correct shapes and values."""

        # Apply a force to generate non-zero acceleration
        force = (
            torch.tensor([10.0, 0.0, 0.0], device=self.sim.device)
            .unsqueeze(0)
            .repeat(NUM_ARENAS, 1)
        )
        self.duck.add_force_torque(force=force)
        self.sim.update(0.01)

        # Read back accelerations
        duck_lin_acc = self.duck.body_data.lin_acc
        duck_ang_acc = self.duck.body_data.ang_acc
        duck_acc = self.duck.body_data.acc

        assert duck_lin_acc.shape == (
            NUM_ARENAS,
            3,
        ), f"Linear acceleration shape mismatch: expected ({NUM_ARENAS}, 3), got {duck_lin_acc.shape}"
        assert duck_ang_acc.shape == (
            NUM_ARENAS,
            3,
        ), f"Angular acceleration shape mismatch: expected ({NUM_ARENAS}, 3), got {duck_ang_acc.shape}"
        assert duck_acc.shape == (
            NUM_ARENAS,
            6,
        ), f"Concatenated acceleration shape mismatch: expected ({NUM_ARENAS}, 6), got {duck_acc.shape}"

        # Verify concatenated acceleration matches individual components
        assert torch.allclose(
            duck_acc[:, :3], duck_lin_acc
        ), "First 3 columns of acc should match lin_acc"
        assert torch.allclose(
            duck_acc[:, 3:], duck_ang_acc
        ), "Last 3 columns of acc should match ang_acc"

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

        if self.physics == "newton":
            expected_mass = torch.ones(NUM_ARENAS, device=self.sim.device)
            expected_friction = torch.full(
                (NUM_ARENAS,),
                self.duck.cfg.attrs.dynamic_friction,
                device=self.sim.device,
            )
            expected_damping = torch.tensor(
                [
                    self.duck.cfg.attrs.linear_damping,
                    self.duck.cfg.attrs.angular_damping,
                ],
                device=self.sim.device,
            ).repeat(NUM_ARENAS, 1)
            expected_inertia = self.duck.get_inertia()
            assert expected_inertia.shape == (NUM_ARENAS, 3)
            assert (
                expected_inertia >= 0
            ).all(), "Initial inertia should be non-negative"

            assert torch.allclose(self.duck.get_mass(), expected_mass)
            assert torch.allclose(self.duck.get_friction(), expected_friction)
            assert torch.allclose(self.duck.get_damping(), expected_damping)

            # set_attrs applies the Newton-supported subset (mass, friction,
            # restitution, contact_offset) at runtime and mirrors the rest.
            self.duck.set_attrs(
                RigidBodyAttributesCfg(mass=2.5, dynamic_friction=0.7, restitution=0.4)
            )
            assert torch.allclose(
                self.duck.get_mass(),
                torch.full((NUM_ARENAS,), 2.5, device=self.sim.device),
                atol=1e-5,
            ), "Newton set_attrs(mass) did not apply via batch API"
            assert torch.allclose(
                self.duck.get_friction(),
                torch.full((NUM_ARENAS,), 0.7, device=self.sim.device),
                atol=1e-5,
            ), "Newton set_attrs(dynamic_friction) did not apply via batch API"

            # set_body_type is a runtime no-op on Newton (body type is fixed at
            # registration); the call must not change body_type.
            self.duck.set_body_type("kinematic")
            assert self.duck.body_type == "dynamic"

            # Mass: set and verify round-trip
            new_mass = torch.full((NUM_ARENAS,), 2.5, device=self.sim.device)
            self.duck.set_mass(new_mass)
            assert torch.allclose(
                self.duck.get_mass(), new_mass, atol=1e-5
            ), f"Newton set_mass round-trip failed: {self.duck.get_mass()}"

            # Friction: set and verify round-trip
            new_friction = torch.full((NUM_ARENAS,), 0.7, device=self.sim.device)
            self.duck.set_friction(new_friction)
            assert torch.allclose(
                self.duck.get_friction(), new_friction, atol=1e-5
            ), f"Newton set_friction round-trip failed: {self.duck.get_friction()}"

            # Inertia: set and verify round-trip
            new_inertia = torch.full((NUM_ARENAS, 3), 0.3, device=self.sim.device)
            self.duck.set_inertia(new_inertia)
            assert torch.allclose(
                self.duck.get_inertia(), new_inertia, atol=1e-5
            ), f"Newton set_inertia round-trip failed: {self.duck.get_inertia()}"

            # Damping is a runtime no-op on Newton (not modelled per body) but
            # mirrors onto metadata so get_damping stays consistent.
            new_damping = torch.full((NUM_ARENAS, 2), 0.2, device=self.sim.device)
            self.duck.set_damping(new_damping)
            assert torch.allclose(
                self.duck.get_damping(), new_damping, atol=1e-5
            ), "Newton set_damping should mirror onto metadata for get_damping"

            self.table.get_mass()
            self.table.get_friction()
            self.table.get_damping()
            self.table.get_inertia()
            return

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

    def test_set_com_pose(self):
        """Test setting full and partial center-of-mass poses."""
        assert self.duck.body_data is not None
        assert self.duck.body_data.default_com_pose is not None
        assert self.duck.body_data.default_com_pose.shape == (
            NUM_ARENAS,
            7,
        ), "Default COM pose should have shape (NUM_ARENAS, 7)"

        com_pose = _make_test_com_pose(self.sim.device)

        self.duck.set_com_pose(com_pose)

        actual_com_pose = self.duck.body_data.com_pose
        assert isinstance(
            actual_com_pose, torch.Tensor
        ), "com_pose should be a torch.Tensor"
        assert actual_com_pose.shape == (
            NUM_ARENAS,
            7,
        ), f"COM pose should have shape (NUM_ARENAS, 7), got {actual_com_pose.shape}"
        assert torch.allclose(actual_com_pose, com_pose, atol=1e-5), (
            "COM pose did not match after full set: "
            f"expected {com_pose.tolist()}, got {actual_com_pose.tolist()}"
        )

        partial_com_pose = torch.tensor(
            [[0.07, -0.03, 0.04, 0.0, 0.38268343, 0.0, 0.9238795]],
            device=self.sim.device,
            dtype=torch.float32,
        )
        expected_com_pose = com_pose.clone()
        expected_com_pose[1] = partial_com_pose[0]

        self.duck.set_com_pose(partial_com_pose, env_ids=[1])

        actual_com_pose = self.duck.body_data.com_pose
        assert torch.allclose(actual_com_pose, expected_com_pose, atol=1e-5), (
            "COM pose did not preserve untouched envs after partial set: "
            f"expected {expected_com_pose.tolist()}, got {actual_com_pose.tolist()}"
        )

        assert self.chair.body_data is not None
        chair_com_pose_before = self.chair.body_data.com_pose.clone()
        self.chair.set_com_pose(com_pose)
        assert torch.allclose(
            self.chair.body_data.com_pose, chair_com_pose_before, atol=1e-5
        ), "Kinematic rigid object COM pose should not change"

        # Static object should not be able to set COM pose.
        self.table.set_com_pose(com_pose)

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

    def test_geometry_data(self):
        """Test mesh-level read APIs: get_triangles and scaled get_vertices.

        Covers:
        - ``get_triangles`` — shape ``(N, num_tris, 3)``, int32, partial env_ids.
        - ``get_vertices(scale=True)`` — scaled vertices differ from unscaled.
        """
        # --- get_triangles (full) ---
        triangles = self.duck.get_triangles()
        assert isinstance(
            triangles, torch.Tensor
        ), "get_triangles should return a torch.Tensor"
        assert triangles.ndim == 3, "Triangles tensor should be 3-D (N, num_tris, 3)"
        assert (
            triangles.shape[0] == NUM_ARENAS
        ), f"First dim should be {NUM_ARENAS}, got {triangles.shape[0]}"
        assert triangles.shape[2] == 3, "Last dim should be 3 (vertex indices)"
        assert (
            triangles.dtype == torch.int32
        ), f"Triangles dtype should be int32, got {triangles.dtype}"

        # --- get_triangles (partial) ---
        partial_tris = self.duck.get_triangles(env_ids=[0])
        assert (
            partial_tris.shape[0] == 1
        ), "Partial get_triangles should return 1 instance"

        # --- get_vertices(scale=True) ---
        new_scale = torch.full(
            (NUM_ARENAS, 3), 2.0, device=self.sim.device, dtype=torch.float32
        )
        self.duck.set_body_scale(new_scale)

        verts_raw = self.duck.get_vertices()
        verts_scaled = self.duck.get_vertices(scale=True)
        assert torch.allclose(
            verts_scaled, verts_raw * 2.0, atol=1e-5
        ), "Scaled vertices should be 2x the raw vertices"

    def test_enable_collision(self):
        """Test enable_collision toggle for individual arenas.

        Covers:
        - ``enable_collision`` with ``enable=False`` (per-instance mask).
        - ``enable_collision`` with ``enable=True`` (restore).
        - partial ``env_ids`` subset.
        """
        # Disable collision for all arenas and re-enable — no exception should be raised.
        disable = torch.zeros(NUM_ARENAS, dtype=torch.bool, device=self.sim.device)
        self.duck.enable_collision(disable)

        enable = torch.ones(NUM_ARENAS, dtype=torch.bool, device=self.sim.device)
        self.duck.enable_collision(enable)

        # Partial: disable only env 0.
        partial_disable = torch.zeros(1, dtype=torch.bool, device=self.sim.device)
        self.duck.enable_collision(partial_disable, env_ids=[0])

        # Restore env 0.
        partial_enable = torch.ones(1, dtype=torch.bool, device=self.sim.device)
        self.duck.enable_collision(partial_enable, env_ids=[0])

    def test_reset(self):
        """Test reset() restores initial pose and clears dynamics.

        Covers:
        - ``reset()`` — all envs returned to ``cfg.init_pos`` (default origin).
        - Velocities cleared to zero after reset.
        - Partial ``env_ids`` reset: only the specified instance is restored.
        """
        # Move duck far from origin and give it velocity.
        pose_far = (
            torch.eye(4, device=self.sim.device).unsqueeze(0).repeat(NUM_ARENAS, 1, 1)
        )
        pose_far[:, 2, 3] = 5.0
        self.duck.set_local_pose(pose_far)

        lin_vel = (
            torch.tensor([3.0, 0.0, 0.0], device=self.sim.device)
            .unsqueeze(0)
            .repeat(NUM_ARENAS, 1)
        )
        self.duck.set_velocity(lin_vel=lin_vel)

        # Full reset.
        self.duck.reset()

        pos_after = self.duck.get_local_pose()[:, :3]
        origin = torch.zeros(NUM_ARENAS, 3, device=self.sim.device)
        assert torch.allclose(
            pos_after, origin, atol=1e-4
        ), f"Duck should be at origin after reset, got {pos_after.tolist()}"

        # Velocities should be zero after reset.
        assert self.duck.body_data is not None
        lin_vel_after = self.duck.body_data.lin_vel
        assert torch.allclose(
            lin_vel_after, torch.zeros_like(lin_vel_after), atol=1e-5
        ), f"Linear velocity should be zero after reset, got {lin_vel_after.tolist()}"

        # --- Partial reset: move duck again, reset only env 0 ---
        self.duck.set_local_pose(pose_far)
        self.duck.reset(env_ids=[0])

        pos_partial = self.duck.get_local_pose()[:, :3]
        assert torch.allclose(
            pos_partial[0], origin[0], atol=1e-4
        ), f"Env 0 should be at origin after partial reset, got {pos_partial[0].tolist()}"
        # Env 1 was not reset — it should still be displaced.
        assert (
            pos_partial[1, 2].item() > 1.0
        ), f"Env 1 should remain displaced after partial reset, got z={pos_partial[1, 2].item()}"

    def test_local_pose_matrix(self):
        """Test ``get_local_pose(to_matrix=True)`` returns correct shape and values.

        Covers:
        - Shape ``(N, 4, 4)`` output.
        - Rotation and translation columns are consistent with the 7-vec form.
        - Partial ``env_ids``.
        """
        pose_7 = torch.eye(4, device=self.sim.device)
        pose_7[0, 3] = 1.0
        pose_7[1, 3] = 2.0
        pose_7[2, 3] = 3.0
        pose_mat_input = pose_7.unsqueeze(0).repeat(NUM_ARENAS, 1, 1)
        self.duck.set_local_pose(pose_mat_input)

        # 7-vec form
        pose_vec = self.duck.get_local_pose(to_matrix=False)
        assert pose_vec.shape == (
            NUM_ARENAS,
            7,
        ), f"7-vec pose shape should be ({NUM_ARENAS}, 7), got {pose_vec.shape}"

        # Matrix form
        pose_mat = self.duck.get_local_pose(to_matrix=True)
        assert pose_mat.shape == (
            NUM_ARENAS,
            4,
            4,
        ), f"Matrix pose shape should be ({NUM_ARENAS}, 4, 4), got {pose_mat.shape}"

        # Translation columns must match.
        assert torch.allclose(
            pose_mat[:, :3, 3], pose_vec[:, :3], atol=1e-5
        ), "Matrix translation column should match 7-vec xyz"

        # Last row must be [0, 0, 0, 1].
        last_row = (
            torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.sim.device)
            .unsqueeze(0)
            .repeat(NUM_ARENAS, 1)
        )
        assert torch.allclose(
            pose_mat[:, 3, :], last_row, atol=1e-5
        ), "Last row of pose matrix should be [0, 0, 0, 1]"

        # Rotation matrix must be orthogonal (R @ R.T ≈ I).
        R = pose_mat[:, :3, :3]
        eye = torch.eye(3, device=self.sim.device).unsqueeze(0).repeat(NUM_ARENAS, 1, 1)
        assert torch.allclose(
            torch.bmm(R, R.transpose(1, 2)), eye, atol=1e-5
        ), "Rotation sub-matrix should be orthogonal"

        # Partial env_ids.
        pose_mat_partial = self.duck.get_local_pose(to_matrix=True)
        assert pose_mat_partial.shape[0] == NUM_ARENAS

    def test_body_data_vel_clear(self):
        """Test ``body_data.vel``, partial ``clear_dynamics``, and verify dynamics reset.

        Covers:
        - ``body_data.vel`` — shape ``(N, 6)`` concatenated lin+ang vel.
        - ``clear_dynamics()`` — verifies all velocities become zero (not just called).
        - ``clear_dynamics(env_ids=[0])`` — partial clear; only env 0 is zeroed.
        """
        assert self.duck.body_data is not None

        lin_vel = (
            torch.tensor([2.0, 0.0, 0.0], device=self.sim.device)
            .unsqueeze(0)
            .repeat(NUM_ARENAS, 1)
        )
        ang_vel = (
            torch.tensor([0.0, 3.0, 0.0], device=self.sim.device)
            .unsqueeze(0)
            .repeat(NUM_ARENAS, 1)
        )
        self.duck.set_velocity(lin_vel=lin_vel, ang_vel=ang_vel)

        # --- body_data.vel ---
        vel = self.duck.body_data.vel
        assert vel.shape == (
            NUM_ARENAS,
            6,
        ), f"vel shape should be ({NUM_ARENAS}, 6), got {vel.shape}"
        assert torch.allclose(
            vel[:, :3], lin_vel, atol=1e-5
        ), f"First 3 columns of vel should match lin_vel"
        assert torch.allclose(
            vel[:, 3:], ang_vel, atol=1e-5
        ), f"Last 3 columns of vel should match ang_vel"

        # --- clear_dynamics() full — verify velocities go to zero ---
        self.duck.clear_dynamics()
        vel_after_clear = self.duck.body_data.vel
        assert torch.allclose(
            vel_after_clear, torch.zeros_like(vel_after_clear), atol=1e-5
        ), f"Velocities should be zero after clear_dynamics, got {vel_after_clear.tolist()}"

        # --- clear_dynamics(env_ids=[0]) partial ---
        # Give env 1 non-zero velocity again.
        self.duck.set_velocity(lin_vel=lin_vel, ang_vel=ang_vel)
        self.duck.clear_dynamics(env_ids=[0])
        vel_partial = self.duck.body_data.vel
        assert torch.allclose(
            vel_partial[0], torch.zeros(6, device=self.sim.device), atol=1e-5
        ), f"Env 0 should be zeroed after partial clear_dynamics, got {vel_partial[0].tolist()}"
        assert not torch.allclose(
            vel_partial[1], torch.zeros(6, device=self.sim.device), atol=1e-5
        ), "Env 1 should still have non-zero velocity after partial clear_dynamics"

    def teardown_method(self):
        """Clean up resources after each test method."""
        self.sim.destroy()
        import embodichain.lab.sim as om

        om.SimulationManager.flush_cleanup_queue()
        self.__dict__.clear()
        import gc

        gc.collect()


class TestRigidObjectCPU(BaseRigidObjectTest):
    def setup_method(self):
        self.setup_simulation("cpu")


class TestRigidObjectCUDA(BaseRigidObjectTest):
    def setup_method(self):
        self.setup_simulation("cuda")


class TestRigidObjectNewton(BaseRigidObjectTest):
    """Full rigid-object coverage on the DexSim Newton physics backend."""

    def setup_method(self):
        self.setup_simulation("cuda", physics="newton")

    def teardown_method(self):
        super().teardown_method()
        _teardown_newton_physics()

    def test_physical_attributes(self):
        """Newton getters and setters for mass, friction, inertia work via batch API."""
        super().test_physical_attributes()

    def test_newton_native_attrs_desc_native_spawn(self):
        """RigidObject with attrs.newton spawns via the desc-native path on Newton.

        Setting ``attrs.newton`` routes spawn through
        ``register_mesh_object_to_newton_patch`` (bypassing legacy PhysicalAttr),
        so Newton-native contact/shape params reach the model. Verifies the
        body is registered with the Newton manager after finalize.
        """
        duck_path = get_data_path(DUCK_PATH)
        cfg = RigidObjectCfg(
            uid="duck_newton_native",
            shape=MeshCfg(fpath=duck_path),
            body_type="dynamic",
            attrs=RigidBodyAttributesCfg(
                mass=1.0,
                dynamic_friction=0.5,
                restitution=0.1,
                newton=NewtonCollisionAttributesCfg(ke=1e3, kd=50.0, margin=0.01),
            ),
        )
        obj: RigidObject = self.sim.add_rigid_object(cfg=cfg)
        self.sim.finalize_newton_physics()

        assert obj.num_instances == NUM_ARENAS
        assert obj.body_type == "dynamic"
        # The body must be registered with the Newton manager post-finalize.
        mgr = self.sim.newton_manager
        assert mgr is not None
        assert mgr.registered_body_count() > 0
        # Common fields round-trip via the batch view (mass applied live).
        assert torch.allclose(
            obj.get_mass(),
            torch.full((NUM_ARENAS,), 1.0, device=self.sim.device),
            atol=1e-5,
        )

    @pytest.mark.skip(
        reason="TODO: DexSim Newton SDF rigidbody path is not validated in EmbodiChain yet."
    )
    def test_add_sdf_mesh(self):
        super().test_add_sdf_mesh()


if __name__ == "__main__":
    # pytest.main(["-s", __file__])
    test = TestRigidObjectCPU()
    test.setup_method()
    test.test_set_visual_material()
    from IPython import embed

    embed()
