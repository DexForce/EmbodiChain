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
from embodichain.lab.sim.objects import Articulation
from embodichain.lab.sim.cfg import (
    ArticulationCfg,
    JointDrivePropertiesCfg,
    LinkPhysicsOverrideCfg,
    RigidBodyAttributesCfg,
    RigidBodyAttributesOverrideCfg,
)
from embodichain.lab.sim.utility.sim_utils import _resolve_link_physics_groups
from embodichain.data import get_data_path
from dexsim.types import ActorType

ART_PATH = "SlidingBoxDrawer/SlidingBoxDrawer.urdf"
NUM_ARENAS = 10


def _link_static_friction(art: Articulation, link_name: str, env_idx: int = 0) -> float:
    return art._entities[env_idx].get_physical_attr(link_name).static_friction


class TestRigidBodyAttributesOverride:
    """Pure-Python tests for per-link physics config merging."""

    def test_merge_with_applies_only_set_fields(self):
        base = RigidBodyAttributesCfg(
            static_friction=0.3,
            dynamic_friction=0.25,
            linear_damping=0.5,
        )
        override = RigidBodyAttributesOverrideCfg(static_friction=0.85)
        merged = override.merge_with(base)
        assert abs(merged.static_friction - 0.85) < 1e-6
        assert abs(merged.dynamic_friction - 0.25) < 1e-6
        assert abs(merged.linear_damping - 0.5) < 1e-6

    def test_resolve_link_physics_overlap_raises(self):
        link_names = ["outer_box", "handle_xpos", "inner_drawer"]
        link_attrs = {
            "box": LinkPhysicsOverrideCfg(
                link_names_expr=["outer_box", "handle_xpos"],
                attrs=RigidBodyAttributesOverrideCfg(static_friction=0.9),
            ),
            "handle": LinkPhysicsOverrideCfg(
                link_names_expr=["handle_xpos"],
                attrs=RigidBodyAttributesOverrideCfg(static_friction=0.8),
            ),
        }
        with pytest.raises(ValueError, match="multiple link_attrs groups"):
            _resolve_link_physics_groups(link_names, link_attrs)


class BaseArticulationTest:
    """Shared test logic for CPU and CUDA."""

    def setup_simulation(self, sim_device):
        config = SimulationManagerCfg(
            headless=True, sim_device=sim_device, num_envs=NUM_ARENAS
        )
        self.sim = SimulationManager(config)

        art_path = get_data_path(ART_PATH)
        assert os.path.isfile(art_path)

        cfg_dict = {"fpath": art_path, "drive_pros": {"drive_type": "force"}}
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

    def test_setter_methods(self):
        """Test setter methods for articulation properties."""
        # Test setting fix_base
        self.art.set_fix_base(True)
        self.art.set_fix_base(False)

        self.art.set_self_collision(False)
        self.art.set_self_collision(True)

    def test_get_joint_drive_with_joint_ids(self):
        """Test get_joint_drive supports joint_ids and env_ids filtering."""
        (
            all_stiffness,
            all_damping,
            all_max_effort,
            all_max_velocity,
            all_friction,
            all_armature,
        ) = self.art.get_joint_drive()

        assert all_stiffness.shape == (
            NUM_ARENAS,
            self.art.dof,
        ), f"FAIL: Expected full stiffness shape {(NUM_ARENAS, self.art.dof)}, got {all_stiffness.shape}"

        if self.art.dof >= 2:
            joint_ids = [0, self.art.dof - 1]
        else:
            joint_ids = [0]

        env_ids = [0, 2, 4] if NUM_ARENAS >= 5 else [0]

        (
            stiffness,
            damping,
            max_effort,
            max_velocity,
            friction,
            armature,
        ) = self.art.get_joint_drive(joint_ids=joint_ids, env_ids=env_ids)

        expected_stiffness = all_stiffness[env_ids][:, joint_ids]
        expected_damping = all_damping[env_ids][:, joint_ids]
        expected_max_effort = all_max_effort[env_ids][:, joint_ids]
        expected_max_velocity = all_max_velocity[env_ids][:, joint_ids]
        expected_friction = all_friction[env_ids][:, joint_ids]
        expected_armature = all_armature[env_ids][:, joint_ids]

        expected_shape = (len(env_ids), len(joint_ids))
        assert (
            stiffness.shape == expected_shape
        ), f"FAIL: Expected stiffness shape {expected_shape}, got {stiffness.shape}"
        assert torch.allclose(
            stiffness, expected_stiffness, atol=1e-5
        ), "FAIL: stiffness does not match expected filtered values"
        assert torch.allclose(
            damping, expected_damping, atol=1e-5
        ), "FAIL: damping does not match expected filtered values"
        assert torch.allclose(
            max_effort, expected_max_effort, atol=1e-5
        ), "FAIL: max_effort does not match expected filtered values"
        assert torch.allclose(
            max_velocity, expected_max_velocity, atol=1e-5
        ), "FAIL: max_velocity does not match expected filtered values"
        assert torch.allclose(
            friction, expected_friction, atol=1e-5
        ), "FAIL: friction does not match expected filtered values"
        assert torch.allclose(
            armature, expected_armature, atol=1e-5
        ), "FAIL: armature does not match expected filtered values"

    def teardown_method(self):
        """Clean up resources after each test method."""
        self.sim.destroy()
        import embodichain.lab.sim as om

        om.SimulationManager.flush_cleanup_queue()
        self.__dict__.clear()
        import gc

        gc.collect()


class BaseArticulationLinkPhysicsTest:
    """Tests for per-link physics configuration (isolated sim per test)."""

    def setup_simulation(self, sim_device: str) -> None:
        config = SimulationManagerCfg(headless=True, sim_device=sim_device, num_envs=2)
        self.sim = SimulationManager(config)
        self.art_path = get_data_path(ART_PATH)
        assert os.path.isfile(self.art_path)

    def teardown_method(self):
        self.sim.destroy()
        import embodichain.lab.sim as om

        om.SimulationManager.flush_cleanup_queue()
        self.__dict__.clear()
        import gc

        gc.collect()

    def test_global_attrs_applied_to_all_links(self):
        """Default attrs should set the same static friction on every link."""
        global_friction = 0.31
        cfg = ArticulationCfg(
            uid="drawer_global_attrs",
            fpath=self.art_path,
            drive_pros=JointDrivePropertiesCfg(drive_type="force"),
            attrs=RigidBodyAttributesCfg(static_friction=global_friction),
        )
        art: Articulation = self.sim.add_articulation(cfg=cfg)
        for link_name in art.link_names:
            assert abs(_link_static_friction(art, link_name) - global_friction) < 1e-3

    def test_link_attrs_override_selected_links(self):
        """link_attrs should override friction only on matched links."""
        global_friction = 0.31
        handle_friction = 0.87
        cfg = ArticulationCfg(
            uid="drawer_link_attrs",
            fpath=self.art_path,
            drive_pros=JointDrivePropertiesCfg(drive_type="force"),
            attrs=RigidBodyAttributesCfg(static_friction=global_friction),
            link_attrs={
                "handle": LinkPhysicsOverrideCfg(
                    link_names_expr=["handle_xpos"],
                    attrs=RigidBodyAttributesOverrideCfg(
                        static_friction=handle_friction
                    ),
                ),
            },
        )
        art: Articulation = self.sim.add_articulation(cfg=cfg)
        assert abs(_link_static_friction(art, "handle_xpos") - handle_friction) < 1e-3
        for link_name in art.link_names:
            if link_name == "handle_xpos":
                continue
            assert abs(_link_static_friction(art, link_name) - global_friction) < 1e-3

    def test_link_attrs_from_dict(self):
        """ArticulationCfg.from_dict should parse nested link_attrs."""
        cfg = ArticulationCfg.from_dict(
            {
                "uid": "drawer_link_attrs_dict",
                "fpath": self.art_path,
                "drive_pros": {"drive_type": "force"},
                "attrs": {"static_friction": 0.4},
                "link_attrs": {
                    "handle": {
                        "link_names_expr": ["handle_xpos"],
                        "attrs": {"static_friction": 0.77},
                    }
                },
            }
        )
        art: Articulation = self.sim.add_articulation(cfg=cfg)
        assert abs(_link_static_friction(art, "handle_xpos") - 0.77) < 1e-3
        assert abs(_link_static_friction(art, "outer_box") - 0.4) < 1e-3

    def test_set_link_physical_attr_runtime(self):
        """Runtime API should update selected links without affecting others."""
        cfg = ArticulationCfg(
            uid="drawer_runtime_attrs",
            fpath=self.art_path,
            drive_pros=JointDrivePropertiesCfg(drive_type="force"),
        )
        art: Articulation = self.sim.add_articulation(cfg=cfg)
        handle_friction = 0.66
        art.set_link_physical_attr(
            RigidBodyAttributesOverrideCfg(static_friction=handle_friction),
            link_names=["handle_xpos"],
        )
        assert abs(_link_static_friction(art, "handle_xpos") - handle_friction) < 1e-3
        for link_name in art.link_names:
            if link_name == "handle_xpos":
                continue
            assert abs(_link_static_friction(art, link_name) - 0.5) < 1e-3


class TestArticulationLinkPhysicsCPU(BaseArticulationLinkPhysicsTest):
    def setup_method(self):
        self.setup_simulation("cpu")


class TestArticulationLinkPhysicsCUDA(BaseArticulationLinkPhysicsTest):
    def setup_method(self):
        self.setup_simulation("cuda")


class TestArticulationCPU(BaseArticulationTest):
    def setup_method(self):
        self.setup_simulation("cpu")


class TestArticulationCUDA(BaseArticulationTest):
    def setup_method(self):
        self.setup_simulation("cuda")


if __name__ == "__main__":
    test = TestArticulationCPU()
    test.setup_method()
    test.test_set_visual_material()
