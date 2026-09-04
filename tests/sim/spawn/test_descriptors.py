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
import warnings
from dataclasses import fields, is_dataclass
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

import dexsim
from dexsim.types import DriveType
from dexsim.spawn import (
    ArticulationDesc,
    ClothDesc,
    CollisionDesc,
    CollisionApproximation,
    DexsimCollisionDesc,
    DexsimClothPhysicsDesc,
    DexsimJointDesc,
    DexsimPhysicsDesc,
    DexsimSoftBodyPhysicsDesc,
    JointDesc,
    LinkDesc,
    NewtonCollisionDesc,
    NewtonJointDesc,
    ObjectDesc,
    RigidBodyPhysicsDesc,
    SoftBodyDesc,
)

from embodichain.lab.sim.cfg import (
    ArticulationCfg,
    ArticulationRootPropertiesCfg,
    ClothObjectCfg,
    ClothPhysicalAttributesCfg,
    CollisionPropertiesCfg,
    DefaultCollisionPropertiesCfg,
    DefaultRigidBodyPropertiesCfg,
    JointDrivePropertiesCfg,
    LinkPhysicsOverrideCfg,
    MassPropertiesCfg,
    MeshCollisionCfg,
    NewtonCollisionPropertiesCfg,
    NewtonJointDrivePropertiesCfg,
    NewtonRigidBodyMaterialCfg,
    RigidBodyMaterialCfg,
    RigidBodyPhysicsCfg,
    RigidObjectCfg,
    RobotCfg,
    SoftbodyPhysicalAttributesCfg,
    SoftbodyVoxelAttributesCfg,
    SoftObjectCfg,
)
from embodichain.lab.sim.shapes import CubeCfg, LoadOption, MeshCfg
from embodichain.lab.sim.objects import Articulation
from embodichain.lab.sim.spawn.descriptors import (
    articulation_desc_from_cfg,
    cloth_desc_from_cfg,
    configure_articulation_desc,
    rigid_desc_from_cfg,
    soft_desc_from_cfg,
)
from embodichain.lab.sim.spawn.usd import (
    articulation_desc_from_usd,
    rigid_desc_from_usd,
)

pytestmark = pytest.mark.no_sim

RESTITUTION = 0.25
DEFORMABLE_MESH_PATH = "/assets/deformable.obj"


def test_soft_descriptor_projects_current_dexsim_particle_schema() -> None:
    youngs = 1.0e5
    poissons = 0.4
    density = 75.0
    dynamic_friction = 0.2
    min_position_iters = 8
    simplify_target = 40
    remesh_resolution = 12
    voxel_resolution = 16
    cfg = SoftObjectCfg(
        uid="soft",
        shape=MeshCfg(fpath=DEFORMABLE_MESH_PATH),
        voxel_attr=SoftbodyVoxelAttributesCfg(
            triangle_remesh_resolution=remesh_resolution,
            triangle_simplify_target=simplify_target,
            simulation_mesh_resolution=voxel_resolution,
        ),
        physical_attr=SoftbodyPhysicalAttributesCfg(
            youngs=youngs,
            poissons=poissons,
            density=density,
            dynamic_friction=dynamic_friction,
            min_position_iters=min_position_iters,
        ),
    )

    descriptor, materials = soft_desc_from_cfg(cfg, per_env=False)

    assert isinstance(descriptor, SoftBodyDesc)
    assert descriptor.mesh.file_path == DEFORMABLE_MESH_PATH
    assert descriptor.per_env is False
    assert descriptor.meshing is not None
    assert descriptor.meshing.proxy_simplify_target == simplify_target
    assert descriptor.meshing.proxy_remesh_resolution == remesh_resolution
    assert descriptor.meshing.voxel_resolution == voxel_resolution
    assert descriptor.physics.volume_density == density
    assert descriptor.physics.k_mu == pytest.approx(youngs / (2.0 * (1.0 + poissons)))
    assert descriptor.physics.k_lambda == pytest.approx(
        youngs * poissons / ((1.0 + poissons) * (1.0 - 2.0 * poissons))
    )
    assert isinstance(descriptor.physics.dexsim, DexsimSoftBodyPhysicsDesc)
    assert descriptor.physics.dexsim.dynamic_friction == dynamic_friction
    assert descriptor.physics.dexsim.min_position_iters == min_position_iters
    assert materials == {}


def test_cloth_descriptor_projects_current_dexsim_particle_schema() -> None:
    density = 2.5
    mass = 0.05
    thickness = 0.02
    bending_stiffness = 0.1
    cfg = ClothObjectCfg(
        uid="cloth",
        shape=MeshCfg(fpath=DEFORMABLE_MESH_PATH),
        physical_attr=ClothPhysicalAttributesCfg(
            density=density,
            mass=mass,
            thickness=thickness,
            bending_stiffness=bending_stiffness,
        ),
    )

    descriptor, materials = cloth_desc_from_cfg(cfg, per_env=False)

    assert isinstance(descriptor, ClothDesc)
    assert descriptor.mesh.file_path == DEFORMABLE_MESH_PATH
    assert descriptor.per_env is False
    assert descriptor.physics.surface_density == density
    assert isinstance(descriptor.physics.dexsim, DexsimClothPhysicsDesc)
    assert descriptor.physics.dexsim.mass == mass
    assert descriptor.physics.dexsim.thickness == thickness
    assert descriptor.physics.dexsim.bending_stiffness == bending_stiffness
    assert materials == {}


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
        attrs=RigidBodyPhysicsCfg(
            material_props=RigidBodyMaterialCfg(restitution=RESTITUTION)
        ),
    )

    descriptor, _ = rigid_desc_from_cfg(
        cfg,
        newton_solver_type=solver_type,
    )

    newton = descriptor.collisions[0].newton
    assert newton is not None
    assert newton.margin == pytest.approx(0.001)
    assert newton.gap == pytest.approx(0.001)
    assert newton.restitution == expected_restitution


def test_rigid_descriptor_preserves_default_backend_restitution() -> None:
    cfg = RigidObjectCfg(
        uid="cube",
        shape=CubeCfg(size=(0.1, 0.1, 0.1)),
        attrs=RigidBodyPhysicsCfg(
            material_props=RigidBodyMaterialCfg(restitution=RESTITUTION)
        ),
    )

    descriptor, _ = rigid_desc_from_cfg(
        cfg,
        newton_solver_type="mujoco_warp",
    )

    assert descriptor.collisions[0].dexsim.restitution == RESTITUTION


def test_flat_rigid_physics_is_rejected_at_config_boundary() -> None:
    with pytest.raises(ValueError, match="Removed flat rigid-body attrs fields"):
        RigidObjectCfg.from_dict(
            {
                "uid": "cube",
                "shape": {"shape_type": "Cube", "size": [0.1, 0.1, 0.1]},
                "attrs": {"mass": 2.0},
            }
        )


def test_rigid_descriptor_authors_mass_or_density_exclusively() -> None:
    cfg = RigidObjectCfg(
        uid="cube",
        shape=CubeCfg(size=(0.1, 0.1, 0.1)),
        attrs=RigidBodyPhysicsCfg.from_dict(
            {"mass_props": {"mass": 1.0, "density": 1.0}}
        ),
    )

    descriptor, _ = rigid_desc_from_cfg(cfg)

    assert descriptor.physics.mass == 1.0
    assert descriptor.physics.density is None


def test_rigid_descriptor_forwards_explicit_mass_properties() -> None:
    cfg = RigidObjectCfg(
        uid="cube",
        shape=CubeCfg(size=(0.1, 0.1, 0.1)),
        attrs=RigidBodyPhysicsCfg(
            mass_props=MassPropertiesCfg(
                mass=2.0,
                inertia=[1.0, 2.0, 3.0],
                com_position=[0.1, 0.2, 0.3],
                com_quaternion=[1.0, 2.0, 3.0, 4.0],
            ),
        ),
    )

    descriptor, _ = rigid_desc_from_cfg(cfg)

    np.testing.assert_array_equal(descriptor.physics.inertia, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(
        descriptor.physics.com_position,
        [0.1, 0.2, 0.3],
    )
    np.testing.assert_allclose(
        descriptor.physics.com_quaternion,
        np.array([4.0, 1.0, 2.0, 3.0]) / np.sqrt(30.0),
    )


@pytest.mark.parametrize(
    ("attrs", "error_match"),
    [
        (
            RigidBodyPhysicsCfg.from_dict(
                {"mass_props": {"mass": 0.0, "inertia": [1.0, 2.0, 3.0]}}
            ),
            "density is required when mass is zero",
        ),
        (
            RigidBodyPhysicsCfg.from_dict(
                {"mass_props": {"mass": 1.0, "inertia": [1.0, 2.0]}}
            ),
            "inertia must contain",
        ),
        (
            RigidBodyPhysicsCfg.from_dict(
                {"mass_props": {"mass": 1.0, "com_quaternion": [0.0, 0.0, 0.0, 0.0]}}
            ),
            "com_quaternion cannot be zero",
        ),
    ],
    ids=["inertia-without-mass", "invalid-inertia-shape", "zero-com-quaternion"],
)
def test_rigid_descriptor_rejects_invalid_mass_properties(
    attrs: RigidBodyPhysicsCfg,
    error_match: str,
) -> None:
    cfg = RigidObjectCfg(
        uid="cube",
        shape=CubeCfg(size=(0.1, 0.1, 0.1)),
        attrs=attrs,
    )

    with pytest.raises(ValueError, match=error_match):
        rigid_desc_from_cfg(cfg)


def test_static_rigid_descriptor_omits_mass_properties() -> None:
    cfg = RigidObjectCfg(
        uid="cube",
        shape=CubeCfg(size=(0.1, 0.1, 0.1)),
        body_type="static",
        attrs=RigidBodyPhysicsCfg.from_dict(
            {
                "mass_props": {
                    "mass": 2.0,
                    "density": 3.0,
                    "inertia": [1.0, 2.0, 3.0],
                    "com_position": [0.1, 0.2, 0.3],
                }
            }
        ),
    )

    descriptor, _ = rigid_desc_from_cfg(cfg)

    assert descriptor.physics.mass is None
    assert descriptor.physics.density is None
    assert descriptor.physics.inertia is None
    assert descriptor.physics.com_position is None


def test_kinematic_rigid_descriptor_honors_mass_priority() -> None:
    cfg = RigidObjectCfg(
        uid="cube",
        shape=CubeCfg(size=(0.1, 0.1, 0.1)),
        body_type="kinematic",
        attrs=RigidBodyPhysicsCfg.from_dict(
            {"mass_props": {"mass": 2.0, "density": 3.0}}
        ),
    )

    descriptor, _ = rigid_desc_from_cfg(cfg)

    assert descriptor.physics.mass == 2.0
    assert descriptor.physics.density is None


def test_grouped_rigid_physics_routes_common_and_backend_properties() -> None:
    cfg = RigidObjectCfg(
        uid="cube",
        shape=CubeCfg(size=(0.1, 0.1, 0.1)),
        attrs=RigidBodyPhysicsCfg(
            mass_props=MassPropertiesCfg(mass=2.0),
            rigid_props=DefaultRigidBodyPropertiesCfg(linear_damping=0.2),
            collision_props=NewtonCollisionPropertiesCfg(
                collision_enabled=False,
                margin=0.01,
            ),
            material_props=NewtonRigidBodyMaterialCfg(
                dynamic_friction=0.4,
                ke=1000.0,
                torsional_friction=0.02,
            ),
        ),
    )

    descriptor, _ = rigid_desc_from_cfg(
        cfg,
        newton_solver_type="mujoco_warp",
    )

    assert descriptor.physics.mass == 2.0
    assert descriptor.physics.dexsim.linear_damping == 0.2
    assert descriptor.physics.dexsim.angular_damping is None
    collision = descriptor.collisions[0]
    assert collision.enable_collision is False
    assert collision.dexsim.dynamic_friction == 0.4
    assert collision.dexsim.static_friction is None
    assert collision.dexsim.contact_offset is None
    assert collision.dexsim.rest_offset is None
    assert collision.newton.margin == 0.01
    assert collision.newton.mu == 0.4
    assert collision.newton.ke == 1000.0
    assert collision.newton.mu_torsional == 0.02


def test_portable_collision_envelope_compiles_to_both_backends() -> None:
    cfg = RigidObjectCfg(
        uid="cube",
        shape=CubeCfg(size=(0.1, 0.1, 0.1)),
        attrs=RigidBodyPhysicsCfg(
            collision_props=CollisionPropertiesCfg(
                contact_offset=0.015,
                rest_offset=0.005,
            )
        ),
    )

    descriptor, _ = rigid_desc_from_cfg(
        cfg,
        newton_solver_type="mujoco_warp",
    )

    collision = descriptor.collisions[0]
    assert collision.dexsim.contact_offset == pytest.approx(0.015)
    assert collision.dexsim.rest_offset == pytest.approx(0.005)
    assert collision.newton.margin == pytest.approx(0.005)
    assert collision.newton.gap == pytest.approx(0.01)


def test_procedural_rigid_collision_defaults_compile_to_both_backends() -> None:
    cfg = RigidObjectCfg(
        uid="cube",
        shape=CubeCfg(size=(0.1, 0.1, 0.1)),
    )

    descriptor, _ = rigid_desc_from_cfg(
        cfg,
        newton_solver_type="mujoco_warp",
    )

    collision = descriptor.collisions[0]
    assert collision.dexsim.contact_offset == pytest.approx(0.002)
    assert collision.dexsim.rest_offset == pytest.approx(0.001)
    assert collision.newton.margin == pytest.approx(0.001)
    assert collision.newton.gap == pytest.approx(0.001)


def test_newton_native_collision_envelope_overrides_portable_translation() -> None:
    cfg = RigidObjectCfg(
        uid="cube",
        shape=CubeCfg(size=(0.1, 0.1, 0.1)),
        attrs=RigidBodyPhysicsCfg(
            collision_props=NewtonCollisionPropertiesCfg(
                contact_offset=0.015,
                rest_offset=0.005,
                margin=0.007,
                gap=0.004,
            )
        ),
    )

    descriptor, _ = rigid_desc_from_cfg(
        cfg,
        newton_solver_type="mujoco_warp",
    )

    collision = descriptor.collisions[0]
    assert collision.dexsim.contact_offset == pytest.approx(0.015)
    assert collision.dexsim.rest_offset == pytest.approx(0.005)
    assert collision.newton.margin == pytest.approx(0.007)
    assert collision.newton.gap == pytest.approx(0.004)


def test_portable_collision_envelope_rejects_invalid_ordering() -> None:
    cfg = RigidObjectCfg(
        uid="cube",
        shape=CubeCfg(size=(0.1, 0.1, 0.1)),
        attrs=RigidBodyPhysicsCfg(
            collision_props=CollisionPropertiesCfg(
                contact_offset=0.001,
                rest_offset=0.002,
            )
        ),
    )

    with pytest.raises(ValueError, match="no smaller than rest_offset"):
        rigid_desc_from_cfg(cfg, newton_solver_type="mujoco_warp")


def test_newton_fills_missing_portable_rest_offset_from_the_default_profile() -> None:
    cfg = RigidObjectCfg(
        uid="cube",
        shape=CubeCfg(size=(0.1, 0.1, 0.1)),
        attrs=RigidBodyPhysicsCfg(
            collision_props=CollisionPropertiesCfg(contact_offset=0.003)
        ),
    )

    descriptor, _ = rigid_desc_from_cfg(cfg, newton_solver_type="mujoco_warp")

    collision = descriptor.collisions[0]
    assert collision.dexsim.contact_offset == pytest.approx(0.003)
    assert collision.dexsim.rest_offset == pytest.approx(0.001)
    assert collision.newton.margin == pytest.approx(0.001)
    assert collision.newton.gap == pytest.approx(0.002)


def test_procedural_collision_defaults_apply_when_only_collision_is_enabled() -> None:
    cfg = RigidObjectCfg(
        uid="cube",
        shape=CubeCfg(size=(0.1, 0.1, 0.1)),
        attrs=RigidBodyPhysicsCfg(
            collision_props=CollisionPropertiesCfg(collision_enabled=True)
        ),
    )

    descriptor, _ = rigid_desc_from_cfg(cfg)

    assert descriptor.physics.dexsim is None
    assert descriptor.physics.newton is None
    assert descriptor.collisions[0].enable_collision is True
    assert descriptor.collisions[0].dexsim.contact_offset == pytest.approx(0.002)
    assert descriptor.collisions[0].dexsim.rest_offset == pytest.approx(0.001)
    assert descriptor.collisions[0].newton.margin == pytest.approx(0.001)
    assert descriptor.collisions[0].newton.gap == pytest.approx(0.001)


def test_grouped_rigid_physics_overlays_usd_without_erasing_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = ObjectDesc(
        name="source",
        physics=RigidBodyPhysicsDesc.dynamic(
            mass=7.0,
            inertia=np.array([1.0, 2.0, 3.0], dtype=np.float32),
            dexsim=DexsimPhysicsDesc(
                linear_damping=0.6,
                angular_damping=0.8,
            ),
        ),
        collisions=[
            CollisionDesc(
                enable_collision=False,
                dexsim=DexsimCollisionDesc(
                    dynamic_friction=0.9,
                    contact_offset=0.05,
                ),
                newton=NewtonCollisionDesc(margin=0.03, gap=0.07),
            )
        ],
    )
    scene = SimpleNamespace(materials={})

    def parse_singleton(path, collection, label):
        return scene, source

    monkeypatch.setattr(
        "embodichain.lab.sim.spawn.usd._parse_singleton",
        parse_singleton,
    )
    cfg = RigidObjectCfg(
        uid="cube",
        shape=MeshCfg(fpath="cube.usd"),
        asset_physics_mode="overlay",
        attrs=RigidBodyPhysicsCfg(
            rigid_props=DefaultRigidBodyPropertiesCfg(linear_damping=0.2),
            collision_props=NewtonCollisionPropertiesCfg(margin=0.01),
            material_props=RigidBodyMaterialCfg(dynamic_friction=0.4),
        ),
    )

    descriptor, _ = rigid_desc_from_usd(cfg)

    assert descriptor.physics.mass == 7.0
    np.testing.assert_array_equal(descriptor.physics.inertia, [1.0, 2.0, 3.0])
    assert descriptor.physics.dexsim.linear_damping == 0.2
    assert descriptor.physics.dexsim.angular_damping == 0.8
    collision = descriptor.collisions[0]
    assert collision.enable_collision is False
    assert collision.dexsim.dynamic_friction == 0.4
    assert collision.dexsim.contact_offset == 0.05
    assert collision.newton.margin == 0.01
    assert collision.newton.gap == 0.07


def test_rigid_usd_can_recompute_source_inertia(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = ObjectDesc(
        name="source",
        physics=RigidBodyPhysicsDesc.dynamic(
            mass=7.0,
            inertia=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        ),
        collisions=[CollisionDesc()],
    )
    monkeypatch.setattr(
        "embodichain.lab.sim.spawn.usd._parse_singleton",
        lambda path, collection, label: (SimpleNamespace(materials={}), source),
    )
    cfg = RigidObjectCfg(
        uid="cube",
        shape=MeshCfg(fpath="cube.usd"),
        asset_physics_mode="overlay",
        attrs=RigidBodyPhysicsCfg(
            mass_props=MassPropertiesCfg(
                mass=2.0,
                recompute_inertia=True,
            )
        ),
    )

    descriptor, _ = rigid_desc_from_usd(cfg)

    assert descriptor.physics.mass == 2.0
    assert descriptor.physics.inertia is None


def test_rigid_usd_preserves_asset_physics_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_mass = 7.0
    source_scale = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    source = ObjectDesc(
        name="source",
        physics=RigidBodyPhysicsDesc.dynamic(mass=source_mass),
        collisions=[CollisionDesc(enable_collision=False)],
        body_scale=source_scale,
    )
    monkeypatch.setattr(
        "embodichain.lab.sim.spawn.usd._parse_singleton",
        lambda path, collection, label: (SimpleNamespace(materials={}), source),
    )
    cfg = RigidObjectCfg(
        uid="cube",
        shape=MeshCfg(fpath="cube.usd"),
        body_type="static",
        body_scale=(1.0, 1.0, 1.0),
        attrs=RigidBodyPhysicsCfg(mass_props=MassPropertiesCfg(mass=1.0)),
    )

    descriptor, _ = rigid_desc_from_usd(cfg)

    assert descriptor.physics.mass == source_mass
    assert descriptor.physics.actor_type == dexsim.types.ActorType.DYNAMIC
    np.testing.assert_array_equal(descriptor.body_scale, source_scale)
    assert descriptor.collisions[0].enable_collision is False
    assert cfg.body_type == "dynamic"
    assert cfg.body_scale == tuple(source_scale)


def test_rigid_descriptor_forwards_newton_sdf_options() -> None:
    cfg = RigidObjectCfg(
        uid="mesh",
        shape=MeshCfg(
            fpath="mesh.glb",
            collision=MeshCollisionCfg(
                approximation="sdf",
                sdf_padding=0.02,
            ),
        ),
    )

    descriptor, _ = rigid_desc_from_cfg(cfg)

    assert descriptor.collisions[0].newton.force_sdf is True
    assert descriptor.collisions[0].newton.sdf_padding == pytest.approx(0.02)


def test_mesh_collision_and_backend_property_slots_compile_independently() -> None:
    cfg = RigidObjectCfg(
        uid="mesh",
        shape=MeshCfg(
            fpath="mesh.glb",
            collision=MeshCollisionCfg(
                approximation="sdf",
                sdf_target_voxel_size=0.005,
                sdf_padding=0.02,
            ),
        ),
        attrs=RigidBodyPhysicsCfg(
            rigid_props=DefaultRigidBodyPropertiesCfg(linear_damping=0.2),
            collision_props=NewtonCollisionPropertiesCfg(margin=0.04),
        ),
    )

    descriptor, _ = rigid_desc_from_cfg(cfg)
    collision = descriptor.collisions[0]

    assert descriptor.physics.dexsim.linear_damping == pytest.approx(0.2)
    assert collision.approximation == CollisionApproximation.SDF
    assert collision.decomp_max_hulls == 1
    assert collision.newton.margin == pytest.approx(0.04)
    assert collision.newton.sdf_target_voxel_size == pytest.approx(0.005)
    assert collision.newton.sdf_max_resolution is None
    assert collision.newton.sdf_padding == pytest.approx(0.02)


def test_mesh_cfg_legacy_collision_fields_normalize_before_compilation() -> None:
    with pytest.warns(DeprecationWarning):
        cfg = RigidObjectCfg.from_dict(
            {
                "uid": "mesh",
                "shape": {
                    "shape_type": "Mesh",
                    "fpath": "mesh.glb",
                    "max_convex_hull_num": 3,
                    "acd_method": "coacd",
                },
            }
        )

    descriptor, _ = rigid_desc_from_cfg(cfg)

    assert (
        descriptor.collisions[0].approximation
        == CollisionApproximation.CONVEX_DECOMPOSITION
    )
    assert descriptor.collisions[0].decomp_max_hulls == 3


def test_static_triangle_mesh_collision_compiles_without_convex_cooking() -> None:
    cfg = RigidObjectCfg(
        uid="mesh",
        body_type="static",
        shape=MeshCfg(
            fpath="mesh.glb",
            collision=MeshCollisionCfg(approximation="triangle_mesh"),
        ),
    )

    descriptor, _ = rigid_desc_from_cfg(cfg)

    assert descriptor.collisions[0].approximation == CollisionApproximation.NONE


def test_dynamic_triangle_mesh_collision_is_rejected_before_spawn() -> None:
    cfg = RigidObjectCfg(
        uid="mesh",
        shape=MeshCfg(
            fpath="mesh.glb",
            collision=MeshCollisionCfg(approximation="triangle_mesh"),
        ),
    )

    with pytest.raises(ValueError, match="only for static"):
        rigid_desc_from_cfg(cfg)


def test_spawn_rejects_unsupported_convex_decomposition_method() -> None:
    cfg = RigidObjectCfg(
        uid="mesh",
        shape=MeshCfg(
            fpath="mesh.glb",
            collision=MeshCollisionCfg(
                approximation="convex_decomposition",
                max_hulls=4,
                acd_method="vhacd",
            ),
        ),
    )

    with pytest.raises(ValueError, match="only acd_method='coacd'"):
        rigid_desc_from_cfg(cfg)


def test_default_collision_solver_fields_compile_from_collision_slot() -> None:
    cfg = RigidObjectCfg(
        uid="cube",
        shape=CubeCfg(size=(0.1, 0.1, 0.1)),
        attrs=RigidBodyPhysicsCfg(
            collision_props=DefaultCollisionPropertiesCfg(
                contact_offset=0.01,
                torsional_patch_radius=0.02,
                min_torsional_patch_radius=0.005,
                disable_strong_friction=True,
            )
        ),
    )

    descriptor, _ = rigid_desc_from_cfg(cfg)

    default_collision = descriptor.collisions[0].dexsim
    assert default_collision.contact_offset == pytest.approx(0.01)
    assert default_collision.torsional_patch_radius == pytest.approx(0.02)
    assert default_collision.min_torsional_patch_radius == pytest.approx(0.005)
    assert default_collision.disable_strong_friction is True


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
        asset_physics_mode="overlay",
        attrs=RigidBodyPhysicsCfg(
            material_props=RigidBodyMaterialCfg(restitution=RESTITUTION)
        ),
    )

    descriptor = articulation_desc_from_cfg(
        cfg,
        newton_solver_type="mujoco_warp",
    )

    assert descriptor.newton_collision is None
    assert descriptor.newton_drive is None
    assert descriptor.urdf_read_inertia is True

    descriptor.links = _resolved_articulation_desc().links
    descriptor.joints = _resolved_articulation_desc().joints
    configure_articulation_desc(
        descriptor,
        cfg,
        newton_solver_type="mujoco_warp",
    )
    assert descriptor.links[0].collisions[0].newton is None


def test_flat_articulation_physics_is_rejected_at_config_boundary() -> None:
    with pytest.raises(ValueError, match="Removed flat rigid-body attrs fields"):
        ArticulationCfg.from_dict(
            {
                "uid": "robot",
                "fpath": "robot.urdf",
                "attrs": {"mass": 2.0},
            }
        )


def test_articulation_root_properties_compile_to_common_descriptor() -> None:
    cfg = ArticulationCfg(
        uid="robot",
        fpath="robot.urdf",
        root_props=ArticulationRootPropertiesCfg(
            fixed_base=False,
            self_collision_enabled=True,
        ),
    )

    descriptor = articulation_desc_from_cfg(cfg)

    assert descriptor.fixed_base is False
    assert descriptor.urdf_fix_root_link is False
    assert descriptor.enable_self_collision is True


def test_articulation_root_defaults_are_resolved_at_import_boundary() -> None:
    descriptor = articulation_desc_from_cfg(
        ArticulationCfg(uid="robot", fpath="robot.urdf")
    )

    assert descriptor.fixed_base is True
    assert descriptor.urdf_fix_root_link is True
    assert descriptor.enable_self_collision is False


def test_explicit_root_properties_override_usd_in_preserve_mode() -> None:
    source = ArticulationDesc(
        name="source",
        fixed_base=False,
        enable_self_collision=True,
    )
    cfg = ArticulationCfg(
        uid="robot",
        fpath="robot.usd",
        asset_physics_mode="preserve",
        root_props=ArticulationRootPropertiesCfg(
            fixed_base=True,
            self_collision_enabled=False,
        ),
    )

    with patch(
        "embodichain.lab.sim.spawn.usd._parse_singleton",
        return_value=(SimpleNamespace(materials={}), source),
    ):
        descriptor, _ = articulation_desc_from_usd(cfg)

    assert descriptor.fixed_base is True
    assert descriptor.enable_self_collision is False


def test_default_root_properties_override_usd_values() -> None:
    source = ArticulationDesc(
        name="source",
        fixed_base=False,
        enable_self_collision=True,
    )
    cfg = ArticulationCfg(
        uid="robot",
        fpath="robot.usd",
        asset_physics_mode="preserve",
    )

    with patch(
        "embodichain.lab.sim.spawn.usd._parse_singleton",
        return_value=(SimpleNamespace(materials={}), source),
    ):
        descriptor, _ = articulation_desc_from_usd(cfg)

    assert descriptor.fixed_base is True
    assert descriptor.enable_self_collision is False


def test_explicit_none_root_properties_preserve_usd_values() -> None:
    source = ArticulationDesc(
        name="source",
        fixed_base=False,
        enable_self_collision=True,
    )
    cfg = ArticulationCfg(
        uid="robot",
        fpath="robot.usd",
        asset_physics_mode="preserve",
        root_props=ArticulationRootPropertiesCfg(
            fixed_base=None,
            self_collision_enabled=None,
        ),
    )

    with patch(
        "embodichain.lab.sim.spawn.usd._parse_singleton",
        return_value=(SimpleNamespace(materials={}), source),
    ):
        descriptor, _ = articulation_desc_from_usd(cfg)

    assert descriptor.fixed_base is False
    assert descriptor.enable_self_collision is True


def test_articulation_descriptor_rejects_newton_acceleration_drive() -> None:
    cfg = ArticulationCfg(
        uid="robot",
        fpath="robot.urdf",
        asset_physics_mode="overlay",
        joint_drive_props=JointDrivePropertiesCfg(
            drive_type="acceleration",
        ),
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


@pytest.mark.parametrize(
    (
        "target_mode",
        "expected_default_mode",
        "expected_newton_mode",
        "expected_stiffness",
        "expected_damping",
    ),
    [
        ("none", DriveType.NONE, 0, 0.0, 0.0),
        ("position", DriveType.FORCE, 1, 12.0, 4.0),
        ("velocity", DriveType.FORCE, 2, 0.0, 4.0),
        ("position_velocity", DriveType.FORCE, 3, 12.0, 4.0),
        ("effort", DriveType.NONE, 4, 0.0, 0.0),
    ],
)
def test_portable_joint_target_modes_compile_for_both_backends(
    target_mode: str,
    expected_default_mode: DriveType,
    expected_newton_mode: int,
    expected_stiffness: float,
    expected_damping: float,
) -> None:
    cfg = ArticulationCfg(
        uid="robot",
        fpath="robot.urdf",
        asset_physics_mode="overlay",
        joint_drive_props=JointDrivePropertiesCfg(
            drive_type="force",
            target_mode=target_mode,  # type: ignore[arg-type]
            stiffness=12.0,
            damping=4.0,
        ),
    )
    descriptor = _resolved_articulation_desc()

    configure_articulation_desc(
        descriptor,
        cfg,
        newton_solver_type="mujoco_warp",
    )

    joint = descriptor.get_joint_desc("arm_joint")
    assert joint.dexsim.drive_mode == expected_default_mode
    assert joint.newton.target_mode == expected_newton_mode
    assert joint.dexsim.stiffness == pytest.approx(expected_stiffness)
    assert joint.dexsim.damping == pytest.approx(expected_damping)
    assert joint.newton.target_ke == pytest.approx(expected_stiffness)
    assert joint.newton.target_kd == pytest.approx(expected_damping)


def test_force_drive_defaults_newton_target_to_position_velocity() -> None:
    cfg = ArticulationCfg(
        uid="robot",
        fpath="robot.urdf",
        asset_physics_mode="overlay",
        joint_drive_props=JointDrivePropertiesCfg(drive_type="force"),
    )
    descriptor = _resolved_articulation_desc()

    configure_articulation_desc(
        descriptor,
        cfg,
        newton_solver_type="mujoco_warp",
    )

    joint = descriptor.get_joint_desc("arm_joint")
    assert joint.dexsim.drive_mode == DriveType.FORCE
    assert joint.newton.target_mode == 3


@pytest.mark.parametrize(
    ("target_mode", "expected_ke", "expected_kd"),
    [
        ("none", 0.0, 0.0),
        ("velocity", 0.0, 4.0),
        ("effort", 0.0, 0.0),
    ],
)
def test_non_mode_aware_newton_solver_uses_gain_fallbacks(
    target_mode: str,
    expected_ke: float,
    expected_kd: float,
) -> None:
    cfg = ArticulationCfg(
        uid="robot",
        fpath="robot.urdf",
        asset_physics_mode="overlay",
        joint_drive_props=JointDrivePropertiesCfg(
            target_mode=target_mode,  # type: ignore[arg-type]
            stiffness=12.0,
            damping=4.0,
        ),
    )
    descriptor = _resolved_articulation_desc()

    configure_articulation_desc(descriptor, cfg, newton_solver_type="xpbd")

    joint = descriptor.get_joint_desc("arm_joint")
    assert joint.newton.target_ke == pytest.approx(expected_ke)
    assert joint.newton.target_kd == pytest.approx(expected_kd)


def test_non_mode_aware_newton_position_fallback_is_explicit() -> None:
    cfg = ArticulationCfg(
        uid="robot",
        fpath="robot.urdf",
        asset_physics_mode="overlay",
        joint_drive_props=JointDrivePropertiesCfg(
            target_mode="position",
            stiffness=12.0,
            damping=4.0,
        ),
    )
    descriptor = _resolved_articulation_desc()

    with pytest.warns(UserWarning, match="POSITION is emulated"):
        configure_articulation_desc(descriptor, cfg, newton_solver_type="xpbd")


def test_auto_solver_defers_position_mode_compatibility_warning() -> None:
    cfg = ArticulationCfg(
        uid="robot",
        fpath="robot.urdf",
        asset_physics_mode="overlay",
        joint_drive_props=JointDrivePropertiesCfg(
            target_mode="position",
            stiffness=12.0,
            damping=4.0,
        ),
    )
    descriptor = _resolved_articulation_desc()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        configure_articulation_desc(descriptor, cfg, newton_solver_type="auto")

    assert not caught


def test_default_articulation_body_properties_compile_per_link() -> None:
    cfg = ArticulationCfg(
        uid="robot",
        fpath="robot.urdf",
        asset_physics_mode="overlay",
        attrs=RigidBodyPhysicsCfg(
            rigid_props=DefaultRigidBodyPropertiesCfg(
                sleep_threshold=0.002,
                min_position_iters=8,
                min_velocity_iters=2,
            )
        ),
    )
    descriptor = _resolved_articulation_desc()

    configure_articulation_desc(
        descriptor,
        cfg,
        newton_solver_type="mujoco_warp",
    )

    for link in descriptor.links:
        assert link.rigid_body.dexsim.sleep_threshold == pytest.approx(0.002)
        assert link.rigid_body.dexsim.min_position_iters == 8
        assert link.rigid_body.dexsim.min_velocity_iters == 2
        assert link.rigid_body.newton is None


def test_articulation_config_applies_to_exact_source_resolved_names() -> None:
    cfg = ArticulationCfg(
        uid="robot",
        fpath="robot.urdf",
        asset_physics_mode="overlay",
        attrs=RigidBodyPhysicsCfg(
            mass_props=MassPropertiesCfg(mass=1.0),
            material_props=RigidBodyMaterialCfg(dynamic_friction=0.4),
        ),
        link_attrs={
            "fingers": LinkPhysicsOverrideCfg(
                link_names_expr=["finger_.*"],
                attrs=RigidBodyPhysicsCfg(
                    mass_props=MassPropertiesCfg(
                        mass=2.0,
                        recompute_inertia=True,
                    ),
                    material_props=RigidBodyMaterialCfg(dynamic_friction=0.8),
                ),
            )
        },
        joint_drive_props=JointDrivePropertiesCfg(
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
    assert joint.effort_limit == 20.0
    assert joint.velocity_limit == 4.0
    assert joint.lower_limit == -1.0
    assert joint.upper_limit == 1.0


def test_joint_drive_properties_compile_joint_dynamics() -> None:
    cfg = ArticulationCfg(
        uid="robot",
        fpath="robot.urdf",
        asset_physics_mode="overlay",
        joint_drive_props=JointDrivePropertiesCfg(
            stiffness=10.0,
            max_effort=20.0,
            max_velocity=2.0,
            friction=0.4,
            armature=0.7,
        ),
    )
    descriptor = _resolved_articulation_desc()

    configure_articulation_desc(descriptor, cfg)

    joint = descriptor.get_joint_desc("arm_joint")
    assert joint.dexsim.stiffness == pytest.approx(10.0)
    assert joint.effort_limit == pytest.approx(20.0)
    assert joint.velocity_limit == pytest.approx(2.0)
    assert joint.dexsim.joint_friction == pytest.approx(0.4)
    assert joint.newton.friction == pytest.approx(0.4)
    assert joint.armature == pytest.approx(0.7)


def test_articulation_array_qpos_limits_compile_before_backend_build() -> None:
    cfg = ArticulationCfg(
        uid="robot",
        fpath="robot.urdf",
        asset_physics_mode="overlay",
        qpos_limits=np.array([[-0.5, 0.75]], dtype=np.float32),
    )
    descriptor = _resolved_articulation_desc()

    configure_articulation_desc(descriptor, cfg)

    joint = descriptor.get_joint_desc("arm_joint")
    assert joint.lower_limit == pytest.approx(-0.5)
    assert joint.upper_limit == pytest.approx(0.75)


def test_robot_control_part_drive_rule_expands_before_spawn() -> None:
    cfg = RobotCfg(
        uid="robot",
        fpath="robot.urdf",
        control_parts={"arm": ["arm_joint"]},
        joint_drive_props=JointDrivePropertiesCfg(
            drive_type="force",
            stiffness={"arm": 10.0, "arm_joint": 20.0},
        ),
    )
    descriptor = _resolved_articulation_desc()

    configure_articulation_desc(descriptor, cfg)

    joint = descriptor.get_joint_desc("arm_joint")
    assert joint.dexsim.stiffness == 20.0
    assert joint.newton.target_ke == 20.0


def test_newton_joint_compatibility_subclass_uses_portable_target_mode() -> None:
    cfg = ArticulationCfg(
        uid="robot",
        fpath="robot.urdf",
        asset_physics_mode="overlay",
        joint_drive_props=NewtonJointDrivePropertiesCfg(
            drive_type="force",
            stiffness={"arm_.*": 12.0},
            damping=4.0,
            friction=0.5,
            armature=0.7,
            target_mode={"arm_.*": "velocity"},
        ),
    )
    descriptor = _resolved_articulation_desc()

    configure_articulation_desc(descriptor, cfg)

    joint = descriptor.get_joint_desc("arm_joint")
    assert joint.dexsim.stiffness == 0.0
    assert joint.dexsim.damping == 4.0
    assert joint.dexsim.joint_friction == 0.5
    assert joint.armature == 0.7
    assert joint.newton.target_ke == 0.0
    assert joint.newton.target_kd == 4.0
    assert joint.newton.friction == 0.5
    assert joint.newton.armature is None
    assert joint.newton.target_mode == 2


def test_grouped_link_physics_overrides_compose_after_source_resolution() -> None:
    cfg = ArticulationCfg(
        uid="robot",
        fpath="robot.urdf",
        asset_physics_mode="overlay",
        attrs=RigidBodyPhysicsCfg(
            mass_props=MassPropertiesCfg(mass=1.0),
            material_props=RigidBodyMaterialCfg(dynamic_friction=0.4),
        ),
        link_attrs={
            "fingers": LinkPhysicsOverrideCfg(
                link_names_expr=["finger_.*"],
                attrs=RigidBodyPhysicsCfg(
                    mass_props=MassPropertiesCfg(
                        mass=2.0,
                        recompute_inertia=True,
                    ),
                    material_props=RigidBodyMaterialCfg(dynamic_friction=0.8),
                ),
            )
        },
    )
    descriptor = _resolved_articulation_desc()

    configure_articulation_desc(descriptor, cfg)

    base = descriptor.get_link_desc("base")
    finger = descriptor.get_link_desc("finger_left")
    assert base.rigid_body.mass == 1.0
    assert base.collisions[0].newton.mu == 0.4
    assert finger.rigid_body.mass == 2.0
    assert finger.collisions[0].newton.mu == 0.8
    assert finger.rigid_body.inertia is None


def test_global_articulation_mass_properties_can_recompute_inertia() -> None:
    cfg = ArticulationCfg(
        uid="robot",
        fpath="robot.urdf",
        asset_physics_mode="overlay",
        attrs=RigidBodyPhysicsCfg(mass_props=MassPropertiesCfg(recompute_inertia=True)),
    )
    descriptor = _resolved_articulation_desc()

    configure_articulation_desc(descriptor, cfg)

    for link in descriptor.links:
        assert link.rigid_body.inertia is None
        assert link.replace_inertial is True
        assert link._embodichain_apply_physics


def test_invalid_source_inertia_uses_geometry_fallback_without_an_overlay() -> None:
    """All-zero asset inertia is not a physical value to preserve."""
    descriptor = _resolved_articulation_desc()
    for link in descriptor.links:
        link._embodichain_source_inertia_valid = False
        link._embodichain_has_collision_geometry = True
        link.rigid_body.inertia = None
        link.rigid_body.com_position = None
        link.rigid_body.com_quaternion = None
    cfg = ArticulationCfg(
        uid="robot",
        fpath="robot.urdf",
        asset_physics_mode="overlay",
    )

    configure_articulation_desc(descriptor, cfg)

    for link in descriptor.links:
        assert link._embodichain_apply_physics
        assert link.replace_inertial is True
        assert link.rigid_body.inertia is None


def test_joint_only_overlay_does_not_author_link_physics() -> None:
    """A robot drive overlay must not trigger native inertia derivation."""
    descriptor = _resolved_articulation_desc()
    before = copy.deepcopy(descriptor)
    cfg = ArticulationCfg(
        uid="robot",
        fpath="robot.urdf",
        asset_physics_mode="overlay",
        joint_drive_props=JointDrivePropertiesCfg(stiffness=10.0),
    )

    configure_articulation_desc(descriptor, cfg)

    for link, source_link in zip(descriptor.links, before.links, strict=True):
        assert not link._embodichain_apply_physics
        _assert_property_tree_equal(link.rigid_body, source_link.rigid_body)


def test_density_override_requires_explicit_source_inertia_recomputation() -> None:
    descriptor = _resolved_articulation_desc()
    for link in descriptor.links:
        link._embodichain_source_inertia_valid = True
    cfg = ArticulationCfg(
        uid="robot",
        fpath="robot.urdf",
        asset_physics_mode="overlay",
        attrs=RigidBodyPhysicsCfg(mass_props=MassPropertiesCfg(density=1000.0)),
    )

    with pytest.raises(ValueError, match="Density override.*recompute_inertia=True"):
        configure_articulation_desc(descriptor, cfg)


def test_per_link_mass_properties_can_preserve_global_source_inertia() -> None:
    cfg = ArticulationCfg(
        uid="robot",
        fpath="robot.urdf",
        asset_physics_mode="overlay",
        attrs=RigidBodyPhysicsCfg(mass_props=MassPropertiesCfg(recompute_inertia=True)),
        link_attrs={
            "fingers": LinkPhysicsOverrideCfg(
                link_names_expr=["finger_.*"],
                attrs=RigidBodyPhysicsCfg(
                    mass_props=MassPropertiesCfg(recompute_inertia=False)
                ),
            )
        },
    )
    descriptor = _resolved_articulation_desc()

    configure_articulation_desc(descriptor, cfg)

    base = descriptor.get_link_desc("base")
    finger = descriptor.get_link_desc("finger_left")
    assert base.rigid_body.inertia is None
    assert base.replace_inertial is True
    np.testing.assert_array_equal(
        finger.rigid_body.inertia,
        np.ones(3, dtype=np.float32),
    )
    assert finger.replace_inertial is False


def test_explicit_and_recomputed_inertia_are_mutually_exclusive() -> None:
    cfg = ArticulationCfg(
        uid="robot",
        fpath="robot.urdf",
        asset_physics_mode="overlay",
        attrs=RigidBodyPhysicsCfg(
            mass_props=MassPropertiesCfg(
                mass=2.0,
                inertia=[1.0, 2.0, 3.0],
                recompute_inertia=True,
            )
        ),
    )

    with pytest.raises(ValueError, match="recompute_inertia"):
        configure_articulation_desc(_resolved_articulation_desc(), cfg)


def test_recompute_inertia_rejects_non_boolean_values() -> None:
    cfg = ArticulationCfg(
        uid="robot",
        fpath="robot.urdf",
        asset_physics_mode="overlay",
        attrs=RigidBodyPhysicsCfg(
            mass_props=MassPropertiesCfg(
                recompute_inertia="yes",  # type: ignore[arg-type]
            )
        ),
    )

    with pytest.raises(TypeError, match="recompute_inertia"):
        configure_articulation_desc(_resolved_articulation_desc(), cfg)


def test_grouped_link_zero_mass_falls_back_to_inherited_density() -> None:
    cfg = ArticulationCfg(
        uid="robot",
        fpath="robot.urdf",
        asset_physics_mode="overlay",
        attrs=RigidBodyPhysicsCfg(
            mass_props=MassPropertiesCfg(mass=1.0, density=500.0)
        ),
        link_attrs={
            "fingers": LinkPhysicsOverrideCfg(
                link_names_expr=["finger_.*"],
                attrs=RigidBodyPhysicsCfg(mass_props=MassPropertiesCfg(mass=0.0)),
            )
        },
    )
    descriptor = _resolved_articulation_desc()

    configure_articulation_desc(descriptor, cfg)

    assert descriptor.get_link_desc("base").rigid_body.mass == 1.0
    finger_physics = descriptor.get_link_desc("finger_left").rigid_body
    assert finger_physics.mass is None
    assert finger_physics.density == 500.0


@pytest.mark.parametrize("source_path", ["robot.urdf", "robot.usd"])
def test_articulation_preserve_mode_keeps_source_physics(source_path: str) -> None:
    descriptor = _resolved_articulation_desc()
    source_joint = descriptor.get_joint_desc("arm_joint")
    source_joint.lower_limit = -2.0
    source_joint.upper_limit = 2.0
    source_joint.effort_limit = 321.0
    source_joint.dexsim = DexsimJointDesc(stiffness=123.0, damping=456.0)
    before = copy.deepcopy(descriptor)
    cfg = ArticulationCfg(
        uid="robot",
        fpath=source_path,
        asset_physics_mode="preserve",
        attrs=RigidBodyPhysicsCfg(mass_props=MassPropertiesCfg(mass=9.0)),
        joint_drive_props=JointDrivePropertiesCfg(
            drive_type="force",
            stiffness=10.0,
            damping=20.0,
        ),
        qpos_limits={"arm_.*": [-1.0, 1.0]},
    )

    with pytest.warns(
        UserWarning,
        match="preserve.*attrs, joint_drive_props, qpos_limits",
    ):
        configure_articulation_desc(descriptor, cfg)

    _assert_property_tree_equal(descriptor, before)


def test_articulation_drive_overlay_preserves_unspecified_source_fields() -> None:
    source_stiffness = 123.0
    source_damping = 456.0
    configured_stiffness = 10.0
    descriptor = _resolved_articulation_desc()
    joint = descriptor.get_joint_desc("arm_joint")
    joint.effort_limit = 321.0
    joint.dexsim = DexsimJointDesc(
        stiffness=source_stiffness,
        damping=source_damping,
        drive_mode=DriveType.FORCE,
    )
    joint.newton = NewtonJointDesc(
        target_ke=source_stiffness,
        target_kd=source_damping,
        target_mode=2,
    )
    cfg = ArticulationCfg(
        uid="robot",
        fpath="robot.urdf",
        asset_physics_mode="overlay",
        joint_drive_props=JointDrivePropertiesCfg(stiffness=configured_stiffness),
    )

    configure_articulation_desc(descriptor, cfg)

    joint = descriptor.get_joint_desc("arm_joint")
    assert joint.dexsim.stiffness == configured_stiffness
    assert joint.dexsim.damping == source_damping
    assert joint.dexsim.drive_mode == DriveType.FORCE
    assert joint.newton.target_ke == configured_stiffness
    assert joint.newton.target_kd == source_damping
    assert joint.newton.target_mode == 2
    assert joint.effort_limit == 321.0


def test_articulation_overlay_does_not_invent_collision_geometry() -> None:
    descriptor = _resolved_articulation_desc()
    collisionless_link = LinkDesc(
        "imu_link",
        "base",
        np.eye(4, dtype=np.float32),
        rigid_body=RigidBodyPhysicsDesc.dynamic(mass=0.1),
    )
    descriptor.links.append(collisionless_link)
    cfg = ArticulationCfg(
        uid="robot",
        fpath="robot.urdf",
        asset_physics_mode="overlay",
        attrs=RigidBodyPhysicsCfg(
            material_props=RigidBodyMaterialCfg(dynamic_friction=0.5)
        ),
    )

    configure_articulation_desc(descriptor, cfg)

    assert descriptor.get_link_desc("imu_link").collisions == []


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
                        attrs=RigidBodyPhysicsCfg(
                            mass_props=MassPropertiesCfg(mass=2.0)
                        ),
                    ),
                    "second": LinkPhysicsOverrideCfg(
                        link_names_expr=["finger_left"],
                        attrs=RigidBodyPhysicsCfg(
                            mass_props=MassPropertiesCfg(mass=3.0)
                        ),
                    ),
                },
            ),
            ValueError,
        ),
        (
            ArticulationCfg(
                uid="robot",
                fpath="robot.urdf",
                joint_drive_props=JointDrivePropertiesCfg(
                    stiffness={"missing_.*": 10.0}
                ),
            ),
            ValueError,
        ),
        (
            ArticulationCfg(
                uid="robot",
                fpath="robot.urdf",
                joint_drive_props=JointDrivePropertiesCfg(
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
        (
            ArticulationCfg(
                uid="robot",
                fpath="robot.urdf",
                qpos_limits=np.zeros((2, 2), dtype=np.float32),
            ),
            ValueError,
        ),
        (
            ArticulationCfg(
                uid="robot",
                fpath="robot.urdf",
                joint_drive_props=NewtonJointDrivePropertiesCfg(
                    target_mode={"arm_.*": "servo"}
                ),
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
        "invalid-array-qpos-shape",
        "invalid-newton-target-mode",
    ],
)
def test_articulation_config_validation_failure_is_atomic(
    cfg: ArticulationCfg,
    error_type: type[Exception],
) -> None:
    cfg.asset_physics_mode = "overlay"
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
        asset_physics_mode="overlay",
        attrs=RigidBodyPhysicsCfg(mass_props=MassPropertiesCfg(mass=1.0)),
        link_attrs={
            "fingers": LinkPhysicsOverrideCfg(
                link_names_expr=["finger_.*"],
                attrs=RigidBodyPhysicsCfg(mass_props=MassPropertiesCfg(mass=2.0)),
            )
        },
        joint_drive_props=JointDrivePropertiesCfg(stiffness={"arm_.*": 10.0}),
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
    entity.joint_dof_layout = []
    entity.get_render_body.return_value = render_body
    articulation = object.__new__(Articulation)
    articulation.cfg = SimpleNamespace(compute_uv=True)
    articulation._entities = [entity]
    articulation.__dict__["link_names"] = ["base"]
    articulation._prepared_default_root_topology_revision = -1
    articulation._mimic_info = SimpleNamespace(
        mimic_id=np.array([], dtype=np.int32),
        mimic_parent=np.array([], dtype=np.int32),
    )
    articulation._set_default_joint_drive = Mock()

    articulation._apply_spawn_config()

    articulation._set_default_joint_drive.assert_not_called()
    entity.get_render_body.assert_called_once_with("base")
    render_body.set_projective_uv.assert_called_once_with()


def test_spawn_post_config_applies_default_only_root_properties() -> None:
    native_articulation = Mock()
    entity = SimpleNamespace(
        _physics_binding=native_articulation,
        joint_dof_layout=[],
    )
    articulation = object.__new__(Articulation)
    articulation.cfg = ArticulationCfg(
        root_props=ArticulationRootPropertiesCfg(
            sleep_threshold=0.005,
            min_position_iters=8,
            min_velocity_iters=2,
        )
    )
    articulation._spawn_result = SimpleNamespace(backend="dexsim", topology_revision=0)
    articulation._entities = [entity]
    articulation._prepared_default_root_topology_revision = -1
    articulation._mimic_info = SimpleNamespace(
        mimic_id=np.array([], dtype=np.int32),
        mimic_parent=np.array([], dtype=np.int32),
    )

    articulation._apply_spawn_config()

    native_articulation.set_sleep_threshold.assert_called_once_with(0.005)
    native_articulation.set_solver_iteration_counts.assert_called_once_with(
        min_position_iters=8,
        min_velocity_iters=2,
    )


def test_newton_skips_default_only_articulation_root_properties() -> None:
    native_articulation = Mock()
    entity = SimpleNamespace(
        _physics_binding=native_articulation,
        joint_dof_layout=[],
    )
    articulation = object.__new__(Articulation)
    articulation.cfg = ArticulationCfg(
        root_props=ArticulationRootPropertiesCfg(
            sleep_threshold=0.005,
            min_position_iters=8,
            min_velocity_iters=2,
        )
    )
    articulation._spawn_result = SimpleNamespace(backend="newton", topology_revision=0)
    articulation._entities = [entity]
    articulation._prepared_default_root_topology_revision = -1
    articulation._mimic_info = SimpleNamespace(
        mimic_id=np.array([], dtype=np.int32),
        mimic_parent=np.array([], dtype=np.int32),
    )

    articulation._apply_spawn_config()

    native_articulation.set_sleep_threshold.assert_not_called()
    native_articulation.set_solver_iteration_counts.assert_not_called()
