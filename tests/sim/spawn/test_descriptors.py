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

import copy
from dataclasses import fields, is_dataclass
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

import dexsim
from dexsim.spawn import (
    ArticulationDesc,
    CollisionDesc,
    JointDesc,
    LinkDesc,
    RigidBodyPhysicsDesc,
)

from embodichain.lab.sim.cfg import (
    ArticulationCfg,
    JointDrivePropertiesCfg,
    LinkPhysicsOverrideCfg,
    RigidBodyAttributesCfg,
    RigidBodyAttributesOverrideCfg,
    RigidObjectCfg,
)
from embodichain.lab.sim.shapes import CubeCfg, LoadOption, MeshCfg
from embodichain.lab.sim.objects import Articulation
from embodichain.lab.sim.spawn.descriptors import (
    articulation_desc_from_cfg,
    configure_articulation_desc,
    rigid_desc_from_cfg,
)
from embodichain.lab.sim.spawn.usd import articulation_desc_from_usd

pytestmark = pytest.mark.no_sim

RESTITUTION = 0.25


def _resolved_articulation_desc() -> ArticulationDesc:
    source_inertia = np.ones(3, dtype=np.float32)
    base = LinkDesc(
        "base",
        "",
        np.eye(4, dtype=np.float32),
        collisions=[CollisionDesc()],
        rigid_body=RigidBodyPhysicsDesc.dynamic(
            mass=0.5,
            inertia=source_inertia,
        ),
    )
    finger = LinkDesc(
        "finger_left",
        "base",
        np.eye(4, dtype=np.float32),
        collisions=[CollisionDesc()],
        rigid_body=RigidBodyPhysicsDesc.dynamic(
            mass=0.25,
            inertia=source_inertia,
        ),
    )
    return ArticulationDesc(
        name="robot",
        links=[base, finger],
        joints=[
            JointDesc(
                "arm_joint",
                "base",
                "finger_left",
                dexsim.engine.JointType.REVOLUTE,
            )
        ],
        root_link_name="base",
    )


def _assert_property_tree_equal(actual: object, expected: object) -> None:
    if isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif is_dataclass(expected):
        assert type(actual) is type(expected)
        for field in fields(expected):
            _assert_property_tree_equal(
                getattr(actual, field.name),
                getattr(expected, field.name),
            )
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key, value in expected.items():
            _assert_property_tree_equal(actual[key], value)
    elif isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected)
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            _assert_property_tree_equal(actual_item, expected_item)
    else:
        assert actual == expected


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


def test_articulation_constructor_defers_newton_properties_until_configure() -> None:
    cfg = ArticulationCfg(
        uid="robot",
        fpath="robot.urdf",
        attrs=RigidBodyAttributesCfg(restitution=RESTITUTION),
    )

    descriptor = articulation_desc_from_cfg(
        cfg,
        newton_solver_type="mujoco_warp",
    )

    assert descriptor.newton_collision is None
    assert descriptor.newton_drive is None

    descriptor.links = _resolved_articulation_desc().links
    descriptor.joints = _resolved_articulation_desc().joints
    configure_articulation_desc(
        descriptor,
        cfg,
        newton_solver_type="mujoco_warp",
    )
    assert descriptor.links[0].collisions[0].newton.restitution is None


def test_articulation_descriptor_rejects_newton_acceleration_drive() -> None:
    cfg = ArticulationCfg(
        uid="robot",
        fpath="robot.urdf",
        drive_pros=JointDrivePropertiesCfg(drive_type="acceleration"),
    )

    descriptor = articulation_desc_from_cfg(cfg, newton_solver_type="mujoco_warp")
    descriptor.links = _resolved_articulation_desc().links
    descriptor.joints = _resolved_articulation_desc().joints

    with pytest.raises(NotImplementedError, match="acceleration-drive"):
        configure_articulation_desc(
            descriptor,
            cfg,
            newton_solver_type="mujoco_warp",
        )


def test_articulation_config_applies_to_exact_source_resolved_names() -> None:
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
    assert descriptor.links == []
    assert descriptor.joints == []

    resolved = _resolved_articulation_desc()
    descriptor.links = resolved.links
    descriptor.joints = resolved.joints
    descriptor.root_link_name = resolved.root_link_name

    with (
        patch.object(
            descriptor,
            "set_link_properties",
            wraps=descriptor.set_link_properties,
        ) as set_link_properties,
        patch.object(
            descriptor,
            "set_joint_properties",
            wraps=descriptor.set_joint_properties,
        ) as set_joint_properties,
    ):
        configure_articulation_desc(descriptor, cfg)

    assert set_link_properties.call_count == len(descriptor.links)
    assert set_joint_properties.call_count == len(descriptor.joints)

    base = descriptor.get_link_desc("base")
    finger = descriptor.get_link_desc("finger_left")
    assert base.rigid_body.mass == 1.0
    assert base.collisions[0].dexsim.dynamic_friction == 0.4
    np.testing.assert_array_equal(
        base.rigid_body.inertia,
        np.ones(3, dtype=np.float32),
    )
    assert finger.rigid_body.mass == 2.0
    assert finger.collisions[0].newton.mu == 0.8
    assert finger.rigid_body.inertia is None
    assert finger.replace_inertial

    joint = descriptor.get_joint_desc("arm_joint")
    assert joint.dexsim.damping == 3.0
    assert joint.newton.target_kd == 3.0
    assert joint.armature == 0.2
    assert joint.dexsim.stiffness == 10.0
    assert joint.newton.target_ke == 10.0
    assert joint.lower_limit == -1.0
    assert joint.upper_limit == 1.0


@pytest.mark.parametrize(
    ("cfg", "error_type"),
    [
        (
            ArticulationCfg(
                uid="robot",
                fpath="robot.urdf",
                link_attrs={
                    "missing": LinkPhysicsOverrideCfg(
                        link_names_expr=["missing_.*"],
                    )
                },
            ),
            ValueError,
        ),
        (
            ArticulationCfg(
                uid="robot",
                fpath="robot.urdf",
                link_attrs={
                    "first": LinkPhysicsOverrideCfg(
                        link_names_expr=["finger_.*"],
                        attrs=RigidBodyAttributesOverrideCfg(mass=2.0),
                        replace_inertial=True,
                    ),
                    "second": LinkPhysicsOverrideCfg(
                        link_names_expr=["finger_left"],
                        attrs=RigidBodyAttributesOverrideCfg(mass=3.0),
                    ),
                },
            ),
            ValueError,
        ),
        (
            ArticulationCfg(
                uid="robot",
                fpath="robot.urdf",
                drive_pros=JointDrivePropertiesCfg(stiffness={"missing_.*": 10.0}),
            ),
            ValueError,
        ),
        (
            ArticulationCfg(
                uid="robot",
                fpath="robot.urdf",
                drive_pros=JointDrivePropertiesCfg(
                    stiffness={"arm_.*": "not-a-number"}
                ),
            ),
            TypeError,
        ),
        (
            ArticulationCfg(
                uid="robot",
                fpath="robot.urdf",
                qpos_limits={"arm_.*": [1.0, -1.0]},
            ),
            ValueError,
        ),
    ],
    ids=[
        "unmatched-link",
        "overlapping-link-groups",
        "unmatched-joint",
        "non-numeric-joint-property",
        "invalid-qpos-limit",
    ],
)
def test_articulation_config_validation_failure_is_atomic(
    cfg: ArticulationCfg,
    error_type: type[Exception],
) -> None:
    descriptor = _resolved_articulation_desc()
    before = copy.deepcopy(descriptor)

    with pytest.raises(error_type):
        configure_articulation_desc(descriptor, cfg)

    _assert_property_tree_equal(descriptor, before)
    finger = descriptor.get_link_desc("finger_left")
    np.testing.assert_array_equal(
        finger.rigid_body.inertia,
        np.ones(3, dtype=np.float32),
    )


def test_usd_articulation_uses_the_same_exact_name_configuration() -> None:
    cfg = ArticulationCfg(
        uid="robot",
        fpath="robot.usd",
        use_usd_properties=False,
        attrs=RigidBodyAttributesCfg(mass=1.0),
        link_attrs={
            "fingers": LinkPhysicsOverrideCfg(
                link_names_expr=["finger_.*"],
                attrs=RigidBodyAttributesOverrideCfg(mass=2.0),
            )
        },
        drive_pros=JointDrivePropertiesCfg(stiffness={"arm_.*": 10.0}),
    )
    source = ArticulationDesc(
        name="source",
        links=[
            LinkDesc(
                "finger_left",
                "",
                np.eye(4, dtype=np.float32),
                collisions=[CollisionDesc()],
                rigid_body=RigidBodyPhysicsDesc.dynamic(mass=0.5),
            )
        ],
        joints=[
            JointDesc(
                "arm_joint",
                "finger_left",
                "tip",
                dexsim.engine.JointType.REVOLUTE,
            )
        ],
    )

    with patch(
        "embodichain.lab.sim.spawn.usd._parse_singleton",
        return_value=(SimpleNamespace(materials={}), source),
    ):
        descriptor, _ = articulation_desc_from_usd(
            cfg,
            newton_solver_type="mujoco_warp",
        )

    configure_articulation_desc(
        descriptor,
        cfg,
        newton_solver_type="mujoco_warp",
    )

    assert descriptor.get_link_desc("finger_left").rigid_body.mass == 2.0
    joint = descriptor.get_joint_desc("arm_joint")
    assert joint.dexsim.stiffness == 10.0
    assert joint.newton.target_ke == 10.0


def test_spawn_post_config_only_applies_render_uv() -> None:
    render_body = Mock()
    entity = Mock()
    entity.get_render_body.return_value = render_body
    articulation = object.__new__(Articulation)
    articulation.cfg = SimpleNamespace(compute_uv=True)
    articulation._entities = [entity]
    articulation.__dict__["link_names"] = ["base"]
    articulation._set_default_joint_drive = Mock()

    articulation._apply_spawn_config()

    articulation._set_default_joint_drive.assert_not_called()
    entity.get_render_body.assert_called_once_with("base")
    render_body.set_projective_uv.assert_called_once_with()
