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

"""Tests for solver-aware Spawn descriptor translation."""

from __future__ import annotations

import pytest

from embodichain.lab.sim.cfg import (
    ArticulationCfg,
    JointDrivePropertiesCfg,
    LinkPhysicsOverrideCfg,
    RigidBodyAttributesCfg,
    RigidBodyAttributesOverrideCfg,
    RigidObjectCfg,
)
from embodichain.lab.sim.shapes import CubeCfg, LoadOption, MeshCfg
from embodichain.lab.sim.spawn.descriptors import (
    articulation_desc_from_cfg,
    rigid_desc_from_cfg,
)

pytestmark = pytest.mark.no_sim

RESTITUTION = 0.25


@pytest.mark.parametrize(
    ("solver_type", "expected_restitution"),
    [
        ("mujoco_warp", None),
        ("semi_implicit", None),
        ("featherstone", None),
        ("xpbd", RESTITUTION),
        (None, RESTITUTION),
    ],
)
def test_rigid_descriptor_projects_restitution_only_to_supported_solvers(
    solver_type: str | None,
    expected_restitution: float | None,
) -> None:
    cfg = RigidObjectCfg(
        uid="cube",
        shape=CubeCfg(size=(0.1, 0.1, 0.1)),
        attrs=RigidBodyAttributesCfg(restitution=RESTITUTION),
    )

    descriptor, _ = rigid_desc_from_cfg(
        cfg,
        newton_solver_type=solver_type,
    )

    assert descriptor.collisions[0].newton.restitution == expected_restitution


def test_rigid_descriptor_preserves_default_backend_restitution() -> None:
    cfg = RigidObjectCfg(
        uid="cube",
        shape=CubeCfg(size=(0.1, 0.1, 0.1)),
        attrs=RigidBodyAttributesCfg(restitution=RESTITUTION),
    )

    descriptor, _ = rigid_desc_from_cfg(
        cfg,
        newton_solver_type="mujoco_warp",
    )

    assert descriptor.collisions[0].dexsim.restitution == RESTITUTION


def test_rigid_descriptor_authors_mass_or_density_exclusively() -> None:
    cfg = RigidObjectCfg(
        uid="cube",
        shape=CubeCfg(size=(0.1, 0.1, 0.1)),
        attrs=RigidBodyAttributesCfg(mass=1.0, density=1.0),
    )

    descriptor, _ = rigid_desc_from_cfg(cfg)

    assert descriptor.physics.mass == 1.0
    assert descriptor.physics.density is None


@pytest.mark.parametrize("body_type", ["static", "kinematic"])
def test_non_dynamic_rigid_descriptor_omits_mass_properties(body_type: str) -> None:
    cfg = RigidObjectCfg(
        uid="cube",
        shape=CubeCfg(size=(0.1, 0.1, 0.1)),
        body_type=body_type,
        attrs=RigidBodyAttributesCfg(mass=2.0, density=3.0),
    )

    descriptor, _ = rigid_desc_from_cfg(cfg)

    assert descriptor.physics.mass is None
    assert descriptor.physics.density is None


def test_mesh_descriptor_passes_load_options_to_spawn() -> None:
    cfg = RigidObjectCfg(
        uid="mesh",
        shape=MeshCfg(
            fpath="mesh.glb",
            load_option=LoadOption(
                rebuild_normals=True,
                rebuild_tangent=True,
                rebuild_3rdnormal=False,
                rebuild_3rdtangent=False,
                smooth=45.0,
            ),
        ),
    )

    descriptor, _ = rigid_desc_from_cfg(cfg)

    option = descriptor.renders[0].load_option
    assert option is not None
    assert option.rebuild_normals is True
    assert option.rebuild_tangent is True
    assert option.rebuild_3rdnormal is False
    assert option.rebuild_3rdtangent is False
    assert option.smooth == 45.0


def test_articulation_descriptor_omits_restitution_for_mujoco_warp() -> None:
    cfg = ArticulationCfg(
        uid="robot",
        fpath="robot.urdf",
        attrs=RigidBodyAttributesCfg(restitution=RESTITUTION),
    )

    descriptor = articulation_desc_from_cfg(
        cfg,
        newton_solver_type="mujoco_warp",
    )

    assert descriptor.newton_collision.restitution is None


def test_articulation_descriptor_rejects_newton_acceleration_drive() -> None:
    cfg = ArticulationCfg(
        uid="robot",
        fpath="robot.urdf",
        drive_pros=JointDrivePropertiesCfg(drive_type="acceleration"),
    )

    with pytest.raises(NotImplementedError, match="acceleration-drive"):
        articulation_desc_from_cfg(cfg, newton_solver_type="mujoco_warp")


def test_articulation_descriptor_compiles_source_resolved_overrides() -> None:
    cfg = ArticulationCfg(
        uid="robot",
        fpath="robot.urdf",
        attrs=RigidBodyAttributesCfg(mass=1.0, dynamic_friction=0.4),
        link_attrs={
            "fingers": LinkPhysicsOverrideCfg(
                link_names_expr=["finger_.*"],
                attrs=RigidBodyAttributesOverrideCfg(
                    mass=2.0,
                    dynamic_friction=0.8,
                ),
                replace_inertial=True,
            )
        },
        drive_pros=JointDrivePropertiesCfg(
            drive_type="force",
            stiffness={"arm_.*": 10.0},
            damping=3.0,
            max_effort=20.0,
            max_velocity=4.0,
            friction=0.1,
            armature=0.2,
        ),
        qpos_limits={"arm_.*": [-1.0, 1.0]},
    )

    descriptor = articulation_desc_from_cfg(cfg)

    assert descriptor.link_defaults.rigid_body.mass == 1.0
    assert descriptor.link_defaults.collision.dexsim.dynamic_friction == 0.4
    assert len(descriptor.link_overrides) == 1
    link_rule = descriptor.link_overrides[0]
    assert link_rule.name == "fingers"
    assert link_rule.patterns == ("finger_.*",)
    assert link_rule.rigid_body.mass == 2.0
    assert link_rule.collision.newton.mu == 0.8
    assert link_rule.replace_inertial

    assert descriptor.joint_defaults.dexsim.damping == 3.0
    assert descriptor.joint_defaults.newton.target_kd == 3.0
    assert descriptor.joint_defaults.armature == 0.2
    assert len(descriptor.joint_overrides) == 1
    joint_rule = descriptor.joint_overrides[0]
    assert joint_rule.patterns == ("arm_.*",)
    assert joint_rule.dexsim.stiffness == 10.0
    assert joint_rule.newton.target_ke == 10.0
    assert joint_rule.lower_limit == -1.0
    assert joint_rule.upper_limit == 1.0
