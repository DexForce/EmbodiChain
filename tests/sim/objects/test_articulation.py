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
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from embodichain.lab.sim import (
    SimulationManager,
    SimulationManagerCfg,
    VisualMaterialCfg,
)
from embodichain.lab.sim.objects import Articulation, ArticulationJointKinematics
from embodichain.lab.sim.cfg import (
    ArticulationCfg,
    JointDrivePropertiesCfg,
    LinkPhysicsOverrideCfg,
    MassPropertiesCfg,
    physics_cfg_for_backend,
    RigidBodyPhysicsCfg,
)
from embodichain.data import get_data_path
from dexsim.types import ActorType, DriveType

ART_PATH = "SlidingBoxDrawer/SlidingBoxDrawer.urdf"
NUM_ARENAS = 10
NEWTON_EFFORT_TARGET_MODE = 4
DRIVE_TEST_STIFFNESS = 12.0
DRIVE_TEST_DAMPING = 4.0


def _teardown_newton_physics() -> None:
    from dexsim.engine.newton_physics import teardown_newton_physics

    teardown_newton_physics()


def test_get_qf_returns_all_articulation_joint_efforts():
    expected_qf = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
    articulation = object.__new__(Articulation)
    articulation._data = SimpleNamespace(qf=expected_qf)

    actual_qf = articulation.get_qf()

    assert torch.equal(actual_qf, expected_qf)


@pytest.mark.no_sim
def test_get_parent_joint_chain_returns_backend_neutral_child_to_root_values():
    articulation = object.__new__(Articulation)
    articulation._data = SimpleNamespace(link_names=["body", "door", "door_handle"])
    fixed_origin = torch.eye(4).numpy()
    fixed = SimpleNamespace(
        name="handle_fixed",
        joint_type=SimpleNamespace(name="FIXED"),
        parent_link_name="door",
        child_link_name="door_handle",
        origin_pose=fixed_origin,
        axis=torch.zeros(3).numpy(),
        lower_limit=0.0,
        upper_limit=0.0,
    )
    hinge = SimpleNamespace(
        name="door_hinge",
        joint_type=SimpleNamespace(name="REVOLUTE"),
        parent_link_name="body",
        child_link_name="door",
        origin_pose=torch.eye(4).numpy(),
        axis=torch.tensor([0.0, 0.0, 1.0]).numpy(),
        lower_limit=0.0,
        upper_limit=2.0,
    )
    joint_infos = {fixed.name: fixed, hinge.name: hinge}
    articulation._entities = [
        SimpleNamespace(
            get_joint_names=lambda: [fixed.name, hinge.name],
            get_joint_info=joint_infos.get,
        )
    ]

    chain = articulation.get_parent_joint_chain("door_handle")
    fixed_origin[0, 3] = 9.0

    assert all(isinstance(joint, ArticulationJointKinematics) for joint in chain)
    assert [joint.name for joint in chain] == ["handle_fixed", "door_hinge"]
    assert [joint.joint_type for joint in chain] == ["fixed", "revolute"]
    assert chain[1].joint_limits == (0.0, 2.0)
    assert chain[0].origin_pose[0, 3].item() == 0.0


@pytest.mark.no_sim
def test_get_parent_joint_chain_uses_newton_joint_descriptors_when_needed():
    articulation = object.__new__(Articulation)
    articulation._data = SimpleNamespace(
        link_names=["body", "door", "door_handle"],
        is_newton_backend=True,
    )
    fixed = SimpleNamespace(
        name="handle_fixed",
        joint_type=SimpleNamespace(name="FIXED"),
        parent_link_name="door",
        child_link_name="door_handle",
        origin_pose=np.eye(4, dtype=np.float32),
        axis=np.zeros(3, dtype=np.float32),
        lower_limit=np.asarray([0.0], dtype=np.float32),
        upper_limit=np.asarray([0.0], dtype=np.float32),
    )
    hinge = SimpleNamespace(
        name="door_hinge",
        joint_type=SimpleNamespace(name="REVOLUTE"),
        parent_link_name="body",
        child_link_name="door",
        origin_pose=np.eye(4, dtype=np.float32),
        axis=np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
        lower_limit=np.asarray([0.0], dtype=np.float32),
        upper_limit=np.asarray([2.0], dtype=np.float32),
    )
    joint_descs = {fixed.name: fixed, hinge.name: hinge}
    entity = SimpleNamespace(
        get_joint_names=lambda: [fixed.name, hinge.name],
        get_joint_info=lambda _: None,
        get_joint_desc=joint_descs.__getitem__,
    )
    articulation._entities = [entity]

    chain = articulation.get_parent_joint_chain("door_handle")

    assert [joint.name for joint in chain] == ["handle_fixed", "door_hinge"]
    assert [joint.joint_type for joint in chain] == ["fixed", "revolute"]
    assert chain[1].joint_limits == (0.0, 2.0)


def _link_static_friction(art: Articulation, link_name: str, env_idx: int = 0) -> float:
    return art.get_link_physical_attr(link_names=[link_name], env_ids=[env_idx])[
        0
    ].static_friction


class _EntityMethodOverride:
    """Delegate every entity method except one overridden setter."""

    def __init__(self, entity, method_name: str, override):
        self._entity = entity
        self._method_name = method_name
        self._override = override

    def __getattr__(self, name: str):
        if name == self._method_name:
            return self._override
        return getattr(self._entity, name)


class TestLinkPhysicsOverrideCfg:
    """Pure-Python tests for per-link physics config merging."""

    def test_grouped_override_applies_only_configured_fields(self):
        base = RigidBodyPhysicsCfg.from_dict(
            {
                "rigid_props": {"linear_damping": 0.5},
                "material_props": {
                    "static_friction": 0.3,
                    "dynamic_friction": 0.25,
                },
            }
        )
        override = RigidBodyPhysicsCfg.from_dict(
            {"material_props": {"static_friction": 0.85}}
        )
        merged = override.to_dexsim_physical_attr(base=base.to_dexsim_physical_attr())
        assert abs(merged.static_friction - 0.85) < 1e-6
        assert abs(merged.dynamic_friction - 0.25) < 1e-6
        assert abs(merged.linear_damping - 0.5) < 1e-6


class BaseArticulationTest:
    """Shared test logic for CPU and CUDA."""

    def setup_simulation(self, device, physics: str = "default"):
        physics_cfg = physics_cfg_for_backend(physics)
        if physics == "newton":
            physics_cfg.solver_cfg = {
                "solver_type": "mujoco_warp",
                "njmax": 8192,
                "nconmax": 8192,
            }
        config = SimulationManagerCfg(
            headless=True,
            device=device,
            num_envs=NUM_ARENAS,
            physics_cfg=physics_cfg,
        )
        self.sim = SimulationManager(config)
        self.physics = physics

        art_path = get_data_path(ART_PATH)
        assert os.path.isfile(art_path)

        cfg_dict = {
            "fpath": art_path,
            "asset_physics_mode": "overlay",
            "joint_drive_props": {"drive_type": "force"},
        }
        self.art: Articulation = self.sim.add_articulation(
            cfg=ArticulationCfg.from_dict(cfg_dict)
        )

        self.sim.prepare()

    def test_local_pose_behavior(self):
        """Test set_local_pose and get_local_pose:
        - Drawer pose is correctly set
        """

        # Set initial poses
        distinct_xyzw = torch.tensor(
            [1.0, 2.0, 3.0, 4.0], device=self.sim.device
        ) / torch.sqrt(torch.tensor(30.0, device=self.sim.device))
        pose = torch.zeros(NUM_ARENAS, 7, device=self.sim.device)
        pose[:, 2] = 1.0
        pose[:, 3:7] = distinct_xyzw

        self.art.set_local_pose(pose, env_ids=None)

        # --- Check poses immediately after setting
        actual_pose = self.art.get_local_pose()
        xyz = actual_pose[0, :3]
        expected_pos = torch.tensor(
            [0.0, 0.0, 1.0], device=self.sim.device, dtype=torch.float32
        )
        assert torch.allclose(
            xyz, expected_pos, atol=1e-5
        ), f"FAIL: Drawer pose not set correctly: {xyz.tolist()}"
        torch.testing.assert_close(
            actual_pose[:, 3:7],
            distinct_xyzw.unsqueeze(0).expand(NUM_ARENAS, -1),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_replicated_link_shapes_are_isolated_by_environment(self):
        """Every articulation link shape should use its environment group."""
        for env_index, entity in enumerate(self.art._entities):
            if self.physics == "newton":
                shape_ids = [
                    shape_id
                    for link in entity.physics_articulation.links
                    for shape_id in link.shape_ids
                ]
                assert shape_ids
                groups = (
                    entity.physics_articulation.runtime.model.shape_collision_group.numpy()
                )
                assert {int(groups[shape_id]) for shape_id in shape_ids} == {
                    env_index + 1
                }
                continue

            expected = np.asarray([env_index, 1, 0, 0], dtype=np.uint32)
            physical_links = [
                link
                for link in entity.articulation_desc.links
                if link.rigid_body is not None
            ]
            assert physical_links
            for link in physical_links:
                np.testing.assert_array_equal(
                    link.rigid_body.collision_filter_data,
                    expected,
                )

    def test_body_data_exposes_link_mass_properties(self):
        """Current and initialization-time link mass properties share one layout."""
        data = self.art.body_data

        assert data.mass.shape == (NUM_ARENAS, self.art.num_links)
        assert data.inertia.shape == (NUM_ARENAS, self.art.num_links, 3)
        assert data.com_pose.shape == (NUM_ARENAS, self.art.num_links, 7)
        assert data.default_mass.shape == data.mass.shape
        assert data.default_inertia.shape == data.inertia.shape
        assert data.default_com_pose.shape == data.com_pose.shape
        assert torch.allclose(self.art.default_link_masses, data.default_mass)

    def test_reset_restores_default_link_mass_properties(self):
        """Partial reset restores mass, inertia, and COM only for selected rows."""
        data = self.art.body_data
        link_name = self.art.link_names[0]
        link_id = self.art.link_names.index(link_name)
        env_ids = [0, 1]
        default_mass = data.default_mass[env_ids, link_id : link_id + 1].clone()
        default_inertia = data.default_inertia[env_ids, link_id : link_id + 1].clone()
        default_com_pose = data.default_com_pose[env_ids, link_id : link_id + 1].clone()
        changed_mass = default_mass + 0.5
        changed_inertia = default_inertia * 1.25
        changed_com_pose = default_com_pose.clone()
        changed_com_pose[..., 0] += 0.02
        changed_com_pose[..., 3:7] = torch.tensor(
            [1.0, 2.0, 3.0, 4.0], device=self.sim.device
        ) / torch.sqrt(torch.tensor(30.0, device=self.sim.device))

        self.art.set_mass(changed_mass, link_names=[link_name], env_ids=env_ids)
        self.art.set_inertia(
            changed_inertia,
            link_names=[link_name],
            env_ids=env_ids,
        )
        self.art.set_com_pose(
            changed_com_pose,
            link_names=[link_name],
            env_ids=env_ids,
        )
        self.sim.prepare()

        assert torch.allclose(
            data.default_mass[env_ids, link_id : link_id + 1], default_mass
        )
        assert torch.allclose(
            data.default_inertia[env_ids, link_id : link_id + 1], default_inertia
        )
        assert torch.allclose(
            data.default_com_pose[env_ids, link_id : link_id + 1], default_com_pose
        )

        self.art.reset(env_ids=[env_ids[0]])
        self.sim.prepare()
        mass_after_partial = self.art.get_mass(link_names=[link_name], env_ids=env_ids)
        inertia_after_partial = self.art.get_inertia(
            link_names=[link_name], env_ids=env_ids
        )
        com_after_partial = self.art.get_com_pose(
            link_names=[link_name], env_ids=env_ids
        )

        assert torch.allclose(mass_after_partial[0], default_mass[0], atol=1e-5)
        assert torch.allclose(mass_after_partial[1], changed_mass[1], atol=1e-5)
        assert torch.allclose(inertia_after_partial[0], default_inertia[0], atol=1e-5)
        assert torch.allclose(inertia_after_partial[1], changed_inertia[1], atol=1e-5)
        assert torch.allclose(com_after_partial[0], default_com_pose[0], atol=1e-5)
        assert torch.allclose(com_after_partial[1], changed_com_pose[1], atol=1e-5)

        self.art.reset(env_ids=[env_ids[1]])
        self.sim.prepare()
        assert torch.allclose(
            self.art.get_mass(link_names=[link_name], env_ids=env_ids),
            default_mass,
            atol=1e-5,
        )
        assert torch.allclose(
            self.art.get_inertia(link_names=[link_name], env_ids=env_ids),
            default_inertia,
            atol=1e-5,
        )
        assert torch.allclose(
            self.art.get_com_pose(link_names=[link_name], env_ids=env_ids),
            default_com_pose,
            atol=1e-5,
        )

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

    def test_explicit_passive_drive_after_construction(self):
        """An explicit passive overlay disables backend joint drives."""
        passive_articulation = self.sim.add_articulation(
            cfg=ArticulationCfg(
                uid="passive_drawer",
                fpath=get_data_path(ART_PATH),
                asset_physics_mode="overlay",
                joint_drive_props=JointDrivePropertiesCfg(drive_type="none"),
            )
        )

        expected_drive_types = [
            [DriveType.NONE] * passive_articulation.dof for _ in range(NUM_ARENAS)
        ]
        assert passive_articulation.get_joint_drive_type() == expected_drive_types

        if self.sim.is_newton_backend:
            expected_target_modes = [
                [0] * passive_articulation.dof for _ in range(NUM_ARENAS)
            ]
            assert passive_articulation.get_joint_target_mode() == expected_target_modes

    def test_preserve_mode_ignores_urdf_physics_overrides(self):
        """Preserve mode keeps source-resolved URDF link and joint physics."""
        source = self.sim.add_articulation(
            cfg=ArticulationCfg(
                uid="source_drawer",
                fpath=get_data_path(ART_PATH),
                asset_physics_mode="preserve",
                init_pos=(-1.0, 0.0, 0.0),
            )
        )
        preserved = self.sim.add_articulation(
            cfg=ArticulationCfg(
                uid="preserved_drawer",
                fpath=get_data_path(ART_PATH),
                asset_physics_mode="preserve",
                init_pos=(1.0, 0.0, 0.0),
                attrs=RigidBodyPhysicsCfg(mass_props=MassPropertiesCfg(mass=123.0)),
                joint_drive_props=JointDrivePropertiesCfg(
                    drive_type="none",
                    stiffness=987.0,
                    damping=654.0,
                    max_effort=321.0,
                    max_velocity=123.0,
                ),
                qpos_limits={".*": [-0.01, 0.01]},
            )
        )

        assert torch.allclose(preserved.body_data.mass, source.body_data.mass)
        assert torch.allclose(
            preserved.body_data.qpos_limits,
            source.body_data.qpos_limits,
        )
        for preserved_value, source_value in zip(
            preserved.get_joint_drive(), source.get_joint_drive()
        ):
            assert torch.allclose(preserved_value, source_value)

    def test_joint_limit_getters_support_env_and_joint_filters(self):
        """Test joint limit getters support joint_ids and env_ids filtering."""
        all_qpos_limits = self.art.body_data.qpos_limits
        (
            _stiffness,
            _damping,
            all_qf_limits,
            all_qvel_limits,
            _friction,
            _armature,
        ) = self.art.get_joint_drive()

        joint_ids = [0, self.art.dof - 1] if self.art.dof >= 2 else [0]
        env_ids = [0, 2, 4] if NUM_ARENAS >= 5 else [0]

        qpos_limits = self.art.get_qpos_limits(joint_ids=joint_ids, env_ids=env_ids)
        qvel_limits = self.art.get_qvel_limits(joint_ids=joint_ids, env_ids=env_ids)
        qf_limits = self.art.get_qf_limits(joint_ids=joint_ids, env_ids=env_ids)

        expected_qpos_limits = all_qpos_limits[env_ids][:, joint_ids, :]
        expected_qvel_limits = all_qvel_limits[env_ids][:, joint_ids]
        expected_qf_limits = all_qf_limits[env_ids][:, joint_ids]

        expected_qpos_shape = (len(env_ids), len(joint_ids), 2)
        expected_joint_shape = (len(env_ids), len(joint_ids))

        assert torch.allclose(
            self.art.body_data.qvel_limits, all_qvel_limits, atol=1e-5
        ), "FAIL: qvel_limits backing tensor does not match post-init joint drive state"
        assert torch.allclose(
            self.art.body_data.qf_limits, all_qf_limits, atol=1e-5
        ), "FAIL: qf_limits backing tensor does not match post-init joint drive state"

        assert (
            qpos_limits.shape == expected_qpos_shape
        ), f"FAIL: Expected qpos_limits shape {expected_qpos_shape}, got {qpos_limits.shape}"
        assert (
            qvel_limits.shape == expected_joint_shape
        ), f"FAIL: Expected qvel_limits shape {expected_joint_shape}, got {qvel_limits.shape}"
        assert (
            qf_limits.shape == expected_joint_shape
        ), f"FAIL: Expected qf_limits shape {expected_joint_shape}, got {qf_limits.shape}"

        assert torch.allclose(
            qpos_limits, expected_qpos_limits, atol=1e-5
        ), "FAIL: qpos_limits does not match expected filtered values"
        assert torch.allclose(
            qvel_limits, expected_qvel_limits, atol=1e-5
        ), "FAIL: qvel_limits does not match expected filtered values"
        assert torch.allclose(
            qf_limits, expected_qf_limits, atol=1e-5
        ), "FAIL: qf_limits does not match expected filtered values"

    def test_joint_limit_cache_tracks_set_joint_drive_updates(self):
        """Test qvel/qf limit caches stay aligned with set_joint_drive writes."""
        (
            _stiffness_before,
            _damping_before,
            all_qf_limits_before,
            all_qvel_limits_before,
            _friction_before,
            _armature_before,
        ) = self.art.get_joint_drive()

        joint_ids = [0, self.art.dof - 1] if self.art.dof >= 2 else [0]
        env_ids = [0, 2, 4] if NUM_ARENAS >= 5 else [0]
        env_ids_tensor = torch.as_tensor(
            env_ids, dtype=torch.long, device=self.sim.device
        )
        joint_ids_tensor = torch.as_tensor(
            joint_ids, dtype=torch.long, device=self.sim.device
        )

        new_qvel_limits = torch.full(
            (len(env_ids), len(joint_ids)),
            321.0,
            dtype=torch.float32,
            device=self.sim.device,
        )
        new_qf_limits = torch.full(
            (len(env_ids), len(joint_ids)),
            654.0,
            dtype=torch.float32,
            device=self.sim.device,
        )

        self.art.set_joint_drive(
            max_effort=new_qf_limits,
            max_velocity=new_qvel_limits,
            joint_ids=joint_ids,
            env_ids=env_ids,
        )

        (
            _stiffness_after,
            _damping_after,
            all_qf_limits_after,
            all_qvel_limits_after,
            _friction_after,
            _armature_after,
        ) = self.art.get_joint_drive()
        qvel_limits = self.art.get_qvel_limits(joint_ids=joint_ids, env_ids=env_ids)
        qf_limits = self.art.get_qf_limits(joint_ids=joint_ids, env_ids=env_ids)

        expected_qvel_limits = all_qvel_limits_before.clone()
        expected_qvel_limits[env_ids_tensor[:, None], joint_ids_tensor] = (
            new_qvel_limits
        )
        expected_qf_limits = all_qf_limits_before.clone()
        expected_qf_limits[env_ids_tensor[:, None], joint_ids_tensor] = new_qf_limits

        assert torch.allclose(
            self.art.body_data.qvel_limits, expected_qvel_limits, atol=1e-5
        ), "FAIL: qvel_limits backing tensor did not track set_joint_drive max_velocity"
        assert torch.allclose(
            self.art.body_data.qf_limits, expected_qf_limits, atol=1e-5
        ), "FAIL: qf_limits backing tensor did not track set_joint_drive max_effort"
        assert torch.allclose(
            all_qvel_limits_after, expected_qvel_limits, atol=1e-5
        ), "FAIL: live qvel limits did not match expected post-write state"
        assert torch.allclose(
            all_qf_limits_after, expected_qf_limits, atol=1e-5
        ), "FAIL: live qf limits did not match expected post-write state"
        assert torch.allclose(
            qvel_limits, new_qvel_limits, atol=1e-5
        ), "FAIL: filtered qvel_limits did not return the updated max_velocity values"
        assert torch.allclose(
            qf_limits, new_qf_limits, atol=1e-5
        ), "FAIL: filtered qf_limits did not return the updated max_effort values"

    def test_joint_limit_setters_update_selected_envs_and_body_data(self):
        """Test joint limit setters update selected envs and cached body data."""
        joint_ids = [0, self.art.dof - 1] if self.art.dof >= 2 else [0]
        env_ids = [0, 2] if NUM_ARENAS >= 3 else [0]

        original_qpos_limits = self.art.body_data.qpos_limits.clone()
        original_qvel_limits = self.art.body_data.qvel_limits.clone()
        original_qf_limits = self.art.body_data.qf_limits.clone()

        qpos_limits = self.art.get_qpos_limits(
            joint_ids=joint_ids, env_ids=env_ids
        ).clone()
        scale = torch.arange(
            1,
            len(env_ids) * len(joint_ids) + 1,
            dtype=torch.float32,
            device=self.sim.device,
        ).reshape(len(env_ids), len(joint_ids))
        tighten = torch.minimum(
            0.001 * scale,
            0.25 * (qpos_limits[:, :, 1] - qpos_limits[:, :, 0]),
        )
        qpos_limits[:, :, 0] += tighten
        qpos_limits[:, :, 1] -= tighten
        qvel_limits = 0.5 + 0.05 * scale
        qf_limits = 1.0 + 0.25 * scale

        self.art.set_qpos_limits(qpos_limits, joint_ids=joint_ids, env_ids=env_ids)
        self.art.set_qvel_limits(qvel_limits, joint_ids=joint_ids, env_ids=env_ids)
        self.art.set_qf_limits(qf_limits, joint_ids=joint_ids, env_ids=env_ids)

        updated_qpos_limits = self.art.get_qpos_limits(
            joint_ids=joint_ids, env_ids=env_ids
        )
        updated_qvel_limits = self.art.get_qvel_limits(
            joint_ids=joint_ids, env_ids=env_ids
        )
        updated_qf_limits = self.art.get_qf_limits(joint_ids=joint_ids, env_ids=env_ids)

        assert torch.allclose(
            updated_qpos_limits, qpos_limits, atol=1e-5
        ), "FAIL: filtered qpos_limits did not return the written values"
        assert torch.allclose(
            updated_qvel_limits, qvel_limits, atol=1e-5
        ), "FAIL: filtered qvel_limits did not return the written values"
        assert torch.allclose(
            updated_qf_limits, qf_limits, atol=1e-5
        ), "FAIL: filtered qf_limits did not return the written values"
        assert torch.allclose(
            self.art.body_data.qpos_limits[env_ids][:, joint_ids, :],
            qpos_limits,
            atol=1e-5,
        ), "FAIL: body_data qpos_limits cache did not update for the selected slice"
        assert torch.allclose(
            self.art.body_data.qvel_limits[env_ids][:, joint_ids],
            qvel_limits,
            atol=1e-5,
        ), "FAIL: body_data qvel_limits cache did not update for the selected slice"
        assert torch.allclose(
            self.art.body_data.qf_limits[env_ids][:, joint_ids],
            qf_limits,
            atol=1e-5,
        ), "FAIL: body_data qf_limits cache did not update for the selected slice"

        non_selected_joint_ids = [
            joint_id for joint_id in range(self.art.dof) if joint_id not in joint_ids
        ]
        if non_selected_joint_ids:
            assert torch.allclose(
                self.art.body_data.qpos_limits[env_ids][:, non_selected_joint_ids, :],
                original_qpos_limits[env_ids][:, non_selected_joint_ids, :],
                atol=1e-5,
            ), "FAIL: qpos_limits changed for non-selected joints in targeted environments"
            assert torch.allclose(
                self.art.body_data.qvel_limits[env_ids][:, non_selected_joint_ids],
                original_qvel_limits[env_ids][:, non_selected_joint_ids],
                atol=1e-5,
            ), "FAIL: qvel_limits changed for non-selected joints in targeted environments"
            assert torch.allclose(
                self.art.body_data.qf_limits[env_ids][:, non_selected_joint_ids],
                original_qf_limits[env_ids][:, non_selected_joint_ids],
                atol=1e-5,
            ), "FAIL: qf_limits changed for non-selected joints in targeted environments"

        untouched_env_ids = [
            env_id for env_id in range(NUM_ARENAS) if env_id not in env_ids
        ]
        if untouched_env_ids:
            assert torch.allclose(
                self.art.body_data.qpos_limits[untouched_env_ids],
                original_qpos_limits[untouched_env_ids],
                atol=1e-5,
            ), "FAIL: qpos_limits changed for untouched environments"
            assert torch.allclose(
                self.art.body_data.qvel_limits[untouched_env_ids],
                original_qvel_limits[untouched_env_ids],
                atol=1e-5,
            ), "FAIL: qvel_limits changed for untouched environments"
            assert torch.allclose(
                self.art.body_data.qf_limits[untouched_env_ids],
                original_qf_limits[untouched_env_ids],
                atol=1e-5,
            ), "FAIL: qf_limits changed for untouched environments"

    def test_joint_limit_setters_accept_single_env_convenience_shapes(self):
        """Test single-env convenience shapes for joint limit setters."""
        env_ids = [0]
        joint_ids = [0, self.art.dof - 1] if self.art.dof >= 2 else [0]

        original_qpos_limits = self.art.body_data.qpos_limits.clone()
        original_qvel_limits = self.art.body_data.qvel_limits.clone()
        original_qf_limits = self.art.body_data.qf_limits.clone()

        qpos_limits = self.art.get_qpos_limits(joint_ids=joint_ids, env_ids=env_ids)[
            0
        ].clone()
        scale = torch.arange(
            1,
            len(joint_ids) + 1,
            dtype=torch.float32,
            device=self.sim.device,
        )
        tighten = torch.minimum(
            0.001 * scale,
            0.25 * (qpos_limits[:, 1] - qpos_limits[:, 0]),
        )
        qpos_limits[:, 0] += tighten
        qpos_limits[:, 1] -= tighten
        qvel_limits = 0.6 + 0.05 * scale
        qf_limits = 1.5 + 0.1 * scale

        self.art.set_qpos_limits(qpos_limits, joint_ids=joint_ids, env_ids=env_ids)
        self.art.set_qvel_limits(qvel_limits, joint_ids=joint_ids, env_ids=env_ids)
        self.art.set_qf_limits(qf_limits, joint_ids=joint_ids, env_ids=env_ids)

        assert torch.allclose(
            self.art.get_qpos_limits(joint_ids=joint_ids, env_ids=env_ids),
            qpos_limits.unsqueeze(0),
            atol=1e-5,
        ), "FAIL: single-env qpos convenience shape did not write expected values"
        assert torch.allclose(
            self.art.get_qvel_limits(joint_ids=joint_ids, env_ids=env_ids),
            qvel_limits.unsqueeze(0),
            atol=1e-5,
        ), "FAIL: single-env qvel convenience shape did not write expected values"
        assert torch.allclose(
            self.art.get_qf_limits(joint_ids=joint_ids, env_ids=env_ids),
            qf_limits.unsqueeze(0),
            atol=1e-5,
        ), "FAIL: single-env qf convenience shape did not write expected values"

        non_selected_joint_ids = [
            joint_id for joint_id in range(self.art.dof) if joint_id not in joint_ids
        ]
        if non_selected_joint_ids:
            assert torch.allclose(
                self.art.body_data.qpos_limits[env_ids][:, non_selected_joint_ids, :],
                original_qpos_limits[env_ids][:, non_selected_joint_ids, :],
                atol=1e-5,
            ), "FAIL: single-env qpos write changed non-selected joints"
            assert torch.allclose(
                self.art.body_data.qvel_limits[env_ids][:, non_selected_joint_ids],
                original_qvel_limits[env_ids][:, non_selected_joint_ids],
                atol=1e-5,
            ), "FAIL: single-env qvel write changed non-selected joints"
            assert torch.allclose(
                self.art.body_data.qf_limits[env_ids][:, non_selected_joint_ids],
                original_qf_limits[env_ids][:, non_selected_joint_ids],
                atol=1e-5,
            ), "FAIL: single-env qf write changed non-selected joints"

        untouched_env_ids = [
            env_id for env_id in range(NUM_ARENAS) if env_id not in env_ids
        ]
        if untouched_env_ids:
            assert torch.allclose(
                self.art.body_data.qpos_limits[untouched_env_ids],
                original_qpos_limits[untouched_env_ids],
                atol=1e-5,
            ), "FAIL: single-env qpos write changed untouched environments"
            assert torch.allclose(
                self.art.body_data.qvel_limits[untouched_env_ids],
                original_qvel_limits[untouched_env_ids],
                atol=1e-5,
            ), "FAIL: single-env qvel write changed untouched environments"
            assert torch.allclose(
                self.art.body_data.qf_limits[untouched_env_ids],
                original_qf_limits[untouched_env_ids],
                atol=1e-5,
            ), "FAIL: single-env qf write changed untouched environments"

    def test_set_qpos_limits_failure_does_not_update_cache(self):
        """Test a failed DexSim qpos limit write leaves the Python cache unchanged."""
        env_ids = [0]
        joint_ids = [0, self.art.dof - 1] if self.art.dof >= 2 else [0]
        original_qpos_limits = self.art.body_data.qpos_limits.clone()

        qpos_limits = self.art.get_qpos_limits(
            joint_ids=joint_ids, env_ids=env_ids
        ).clone()
        scale = torch.arange(
            1,
            len(joint_ids) + 1,
            dtype=torch.float32,
            device=self.sim.device,
        ).unsqueeze(0)
        tighten = torch.minimum(
            0.001 * scale,
            0.25 * (qpos_limits[:, :, 1] - qpos_limits[:, :, 0]),
        )
        qpos_limits[:, :, 0] += tighten
        qpos_limits[:, :, 1] -= tighten

        original_entity = self.art._entities[0]

        def _fail_set_joint_limits(_limits, _joint_ids):
            return -1

        self.art._entities[0] = _EntityMethodOverride(
            original_entity,
            "set_joint_position_limits",
            _fail_set_joint_limits,
        )
        try:
            with pytest.raises(RuntimeError, match="set_joint_position_limits failed"):
                self.art.set_qpos_limits(
                    qpos_limits, joint_ids=joint_ids, env_ids=env_ids
                )
        finally:
            self.art._entities[0] = original_entity

        assert torch.allclose(
            self.art.body_data.qpos_limits,
            original_qpos_limits,
            atol=1e-5,
        ), "FAIL: qpos_limits cache changed after a failed DexSim write"

    def test_set_qpos_clamps_against_updated_qpos_limits(self):
        """Test set_qpos clamps to the selected environments' exact updated limits."""
        env_ids = [0, 1] if NUM_ARENAS >= 2 else [0]
        joint_ids = [0]
        if len(env_ids) == 2:
            qpos_limits = torch.tensor(
                [[[-0.02, 0.02]], [[-0.05, 0.01]]],
                dtype=torch.float32,
                device=self.sim.device,
            )
            requested_qpos = torch.tensor(
                [[0.5], [-0.5]],
                dtype=torch.float32,
                device=self.sim.device,
            )
            expected_qpos = torch.tensor(
                [[0.02], [-0.05]],
                dtype=torch.float32,
                device=self.sim.device,
            )
        else:
            qpos_limits = torch.tensor(
                [[[-0.02, 0.02]]],
                dtype=torch.float32,
                device=self.sim.device,
            )
            requested_qpos = torch.tensor(
                [[0.5]],
                dtype=torch.float32,
                device=self.sim.device,
            )
            expected_qpos = torch.tensor(
                [[0.02]],
                dtype=torch.float32,
                device=self.sim.device,
            )

        self.art.set_qpos_limits(qpos_limits, joint_ids=joint_ids, env_ids=env_ids)

        self.art.set_qpos(
            requested_qpos,
            joint_ids=joint_ids,
            env_ids=env_ids,
            target=False,
        )

        clamped_qpos = self.art.get_qpos()[env_ids][:, joint_ids]

        assert torch.allclose(
            clamped_qpos, expected_qpos, atol=1e-5
        ), f"FAIL: qpos did not clamp to the per-env updated limits: {clamped_qpos.tolist()}"

    def test_qpos_limits_from_cfg_dict_can_tighten(self):
        """Test qpos_limits can be set from ArticulationCfg with a regex dictionary."""
        from embodichain.lab.sim.cfg import ArticulationCfg

        cfg = ArticulationCfg(
            uid="drawer_cfg_qpos_limits",
            fpath=get_data_path(ART_PATH),
            asset_physics_mode="overlay",
            joint_drive_props=JointDrivePropertiesCfg(drive_type="force"),
            qpos_limits={".*": [-0.05, 0.05]},
        )
        art: Articulation = self.sim.add_articulation(cfg=cfg)
        limits = art.get_qpos_limits()
        assert torch.all(
            limits[..., 0] >= -0.0501
        ), "FAIL: cfg qpos_limits lower bound not applied"
        assert torch.all(
            limits[..., 1] <= 0.0501
        ), "FAIL: cfg qpos_limits upper bound not applied"

    def test_qpos_limits_from_cfg_can_expand(self):
        """Test qpos_limits from ArticulationCfg can expand joint limits."""
        from embodichain.lab.sim.cfg import ArticulationCfg

        joint_name = self.art.joint_names[0]
        asset_limits = self.art.get_qpos_limits()[:, 0, :]
        expanded_lower = asset_limits[:, 0].min().item() - 0.1
        expanded_upper = asset_limits[:, 1].max().item() + 0.1

        cfg = ArticulationCfg(
            uid="drawer_expanded_limits",
            fpath=get_data_path(ART_PATH),
            asset_physics_mode="overlay",
            joint_drive_props=JointDrivePropertiesCfg(drive_type="force"),
            qpos_limits={joint_name: [expanded_lower, expanded_upper]},
        )
        art: Articulation = self.sim.add_articulation(cfg=cfg)
        limits = art.get_qpos_limits()[:, 0, :]
        assert torch.allclose(
            limits,
            torch.tensor(
                [expanded_lower, expanded_upper],
                device=self.sim.device,
                dtype=torch.float32,
            ),
            atol=1e-4,
        ), f"FAIL: cfg qpos_limits not applied: {limits.tolist()}"

        # set_qpos should clamp to the expanded limits, not the asset limits.
        requested_qpos = torch.full(
            (NUM_ARENAS, 1), expanded_upper, device=self.sim.device
        )
        art.set_qpos(requested_qpos, joint_ids=[0], target=False)
        actual_qpos = art.get_qpos()[:, 0]
        assert torch.allclose(
            actual_qpos,
            torch.full_like(actual_qpos, expanded_upper),
            atol=1e-4,
        ), f"FAIL: set_qpos did not use expanded limits: {actual_qpos.tolist()}"

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

    def setup_simulation(self, device: str) -> None:
        config = SimulationManagerCfg(headless=True, device=device, num_envs=2)
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
            asset_physics_mode="overlay",
            joint_drive_props=JointDrivePropertiesCfg(drive_type="force"),
            attrs=RigidBodyPhysicsCfg.from_dict(
                {"material_props": {"static_friction": global_friction}}
            ),
        )
        art: Articulation = self.sim.add_articulation(cfg=cfg)
        self.sim.prepare()
        for link_name in art.link_names:
            assert abs(_link_static_friction(art, link_name) - global_friction) < 1e-3

    def test_link_attrs_override_selected_links(self):
        """link_attrs should override friction only on matched links."""
        global_friction = 0.31
        handle_friction = 0.87
        cfg = ArticulationCfg(
            uid="drawer_link_attrs",
            fpath=self.art_path,
            asset_physics_mode="overlay",
            joint_drive_props=JointDrivePropertiesCfg(drive_type="force"),
            attrs=RigidBodyPhysicsCfg.from_dict(
                {"material_props": {"static_friction": global_friction}}
            ),
            link_attrs={
                "handle": LinkPhysicsOverrideCfg(
                    link_names_expr=["handle_xpos"],
                    attrs=RigidBodyPhysicsCfg.from_dict(
                        {"material_props": {"static_friction": handle_friction}}
                    ),
                ),
            },
        )
        art: Articulation = self.sim.add_articulation(cfg=cfg)
        self.sim.prepare()
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
                "asset_physics_mode": "overlay",
                "joint_drive_props": {"drive_type": "force"},
                "attrs": {"material_props": {"static_friction": 0.4}},
                "link_attrs": {
                    "handle": {
                        "link_names_expr": ["handle_xpos"],
                        "attrs": {"material_props": {"static_friction": 0.77}},
                    }
                },
            }
        )
        art: Articulation = self.sim.add_articulation(cfg=cfg)
        self.sim.prepare()
        assert abs(_link_static_friction(art, "handle_xpos") - 0.77) < 1e-3
        assert abs(_link_static_friction(art, "outer_box") - 0.4) < 1e-3

    def test_set_link_physical_attr_runtime(self):
        """Runtime API should update selected links without affecting others."""
        cfg = ArticulationCfg(
            uid="drawer_runtime_attrs",
            fpath=self.art_path,
            asset_physics_mode="overlay",
            joint_drive_props=JointDrivePropertiesCfg(drive_type="force"),
        )
        art: Articulation = self.sim.add_articulation(cfg=cfg)
        self.sim.prepare()
        source_friction = {
            link_name: _link_static_friction(art, link_name)
            for link_name in art.link_names
        }
        handle_friction = 0.66
        art.set_link_physical_attr(
            RigidBodyPhysicsCfg.from_dict(
                {"material_props": {"static_friction": handle_friction}}
            ),
            link_names=["handle_xpos"],
        )
        self.sim.prepare()
        assert abs(_link_static_friction(art, "handle_xpos") - handle_friction) < 1e-3
        for link_name in art.link_names:
            if link_name == "handle_xpos":
                continue
            assert (
                abs(_link_static_friction(art, link_name) - source_friction[link_name])
                < 1e-3
            )


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


class TestArticulationNewton(BaseArticulationTest):
    """Articulation coverage on the DexSim Newton physics backend."""

    def setup_method(self):
        self.setup_simulation("cuda", physics="newton")

    def teardown_method(self):
        self.sim.destroy()
        import embodichain.lab.sim as om

        om.SimulationManager.flush_cleanup_queue()
        _teardown_newton_physics()
        import gc

        gc.collect()

    def test_control_api(self):
        """Newton articulation direct state and control buffers round-trip."""
        qpos_zero = torch.zeros(
            (NUM_ARENAS, self.art.dof), dtype=torch.float32, device=self.sim.device
        )
        qpos = qpos_zero.clone()
        qpos[:, -1] = 0.1

        self.art.set_qpos(qpos, env_ids=None, target=False)
        assert torch.allclose(self.art.body_data.qpos, qpos, atol=1e-5)

        self.art.set_qpos(qpos_zero, env_ids=None, target=False)
        self.art.set_qpos(qpos, env_ids=None, target=True)
        assert torch.allclose(self.art.body_data.target_qpos, qpos, atol=1e-5)

        qvel = torch.full(
            (NUM_ARENAS, self.art.dof),
            0.2,
            dtype=torch.float32,
            device=self.sim.device,
        )
        self.art.set_qvel(qvel, env_ids=None, target=False)
        assert torch.allclose(self.art.body_data.qvel, qvel, atol=1e-5)

        qf = torch.ones(
            (NUM_ARENAS, self.art.dof), dtype=torch.float32, device=self.sim.device
        )
        self.art.set_qf(qf, env_ids=None)
        assert torch.allclose(self.art.body_data.qf, qf, atol=1e-5)

        self.art.clear_dynamics()
        assert torch.allclose(self.art.body_data.qvel, qpos_zero, atol=1e-5)
        assert torch.allclose(self.art.body_data.qf, qpos_zero, atol=1e-5)

    @pytest.mark.gpu
    def test_runtime_effort_drive_mode(self):
        """Newton authors effort mode and removes effective PD gains."""
        shape = (NUM_ARENAS, self.art.dof)
        self.art.set_joint_drive(
            stiffness=torch.full(
                shape,
                DRIVE_TEST_STIFFNESS,
                dtype=torch.float32,
                device=self.sim.device,
            ),
            damping=torch.full(
                shape,
                DRIVE_TEST_DAMPING,
                dtype=torch.float32,
                device=self.sim.device,
            ),
            drive_type="force",
            target_mode="effort",
        )

        assert self.art.get_joint_target_mode() == [
            [NEWTON_EFFORT_TARGET_MODE] * self.art.dof for _ in range(NUM_ARENAS)
        ]
        stiffness, damping, *_ = self.art.get_joint_drive()
        assert torch.count_nonzero(stiffness) == 0
        assert torch.count_nonzero(damping) == 0

    @pytest.mark.skip(
        reason="DexSim Newton articulation visual-material helpers are render-Skeleton only."
    )
    def test_set_visual_material(self):
        super().test_set_visual_material()

    @pytest.mark.skip(
        reason="DexSim Newton articulation physical-visible helpers are render-Skeleton only."
    )
    def test_set_physical_visible(self):
        super().test_set_physical_visible()

    def test_set_mass_rebuilds_mass_on_newton(self):
        """A retained Newton per-link mass takes effect at prepare()."""
        link_name = self.art.link_names[0]
        original = self.art.get_mass(link_names=[link_name])[0, 0].item()
        new_mass = original + 1.5
        self.art.set_mass(
            torch.full(
                (NUM_ARENAS, 1),
                new_mass,
                dtype=torch.float32,
                device=self.sim.device,
            ),
            link_names=[link_name],
        )
        self.sim.prepare()
        live_mass = self.art.get_mass(link_names=[link_name])[0, 0].item()
        assert (
            abs(live_mass - new_mass) < 1e-3
        ), f"per-link mass {new_mass} not applied after Newton rebuild (got {live_mass})"


if __name__ == "__main__":
    test = TestArticulationCPU()
    test.setup_method()
    test.test_set_visual_material()
