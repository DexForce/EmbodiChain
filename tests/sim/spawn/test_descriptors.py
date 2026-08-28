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
from dexsim.types import DriveType
from dexsim.spawn import (
    ArticulationDesc,
    ClothDesc,
    CollisionDesc,
    DexsimCollisionDesc,
    DexsimJointDesc,
    DexsimPhysicsDesc,
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
    ClothObjectCfg,
    ClothPhysicalAttributesCfg,
    CollisionPropertiesCfg,
    DexsimRigidBodyPropertiesCfg,
    JointDrivePropertiesCfg,
    LinkPhysicsOverrideCfg,
    MassPropertiesCfg,
    NewtonArticulationRootPropertiesCfg,
    NewtonCollisionPropertiesCfg,
    NewtonJointDrivePropertiesCfg,
    NewtonRigidBodyMaterialCfg,
    RigidBodyAttributesCfg,
    RigidBodyAttributesOverrideCfg,
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


def test_soft_descriptor_uses_newton_particle_schema() -> None:
    youngs = 1.0e5
    poissons = 0.4
    cfg = SoftObjectCfg(
        uid="soft",
        shape=MeshCfg(fpath=DEFORMABLE_MESH_PATH),
        particle_radius=0.02,
        validate_mesh=True,
        voxel_attr=SoftbodyVoxelAttributesCfg(
            triangle_remesh_resolution=12,
            triangle_simplify_target=40,
            simulation_mesh_resolution=16,
            voxel_num_relaxation_iters=7,
            voxel_rel_min_tet_volume=0.08,
            voxel_surface_dist_ratio=0.3,
            embedding_impl="dexsim_exact_cpu",
        ),
        physical_attr=SoftbodyPhysicalAttributesCfg(
            youngs=youngs,
            poissons=poissons,
            density=75.0,
            elasticity_damping=0.2,
            surface_tri_ke=1.0,
            surface_edge_ke=2.0,
        ),
    )

    descriptor, materials = soft_desc_from_cfg(cfg, per_env=False)

    assert isinstance(descriptor, SoftBodyDesc)
    assert descriptor.mesh.file_path == DEFORMABLE_MESH_PATH
    assert descriptor.particle_radius == pytest.approx(0.02)
    assert descriptor.validate_mesh is True
    assert descriptor.per_env is False
    assert descriptor.physics.volume_density == pytest.approx(75.0)
    assert descriptor.physics.k_mu == pytest.approx(youngs / (2.0 * (1.0 + poissons)))
    assert descriptor.physics.k_lambda == pytest.approx(
        youngs * poissons / ((1.0 + poissons) * (1.0 - 2.0 * poissons))
    )
    assert descriptor.physics.k_damp == pytest.approx(0.2)
    assert descriptor.physics.surface_tri_ke == pytest.approx(1.0)
    assert descriptor.physics.surface_edge_ke == pytest.approx(2.0)
    assert descriptor.physics.dexsim is None
    assert descriptor.meshing.proxy_simplify_target == 40
    assert descriptor.meshing.proxy_remesh_resolution == 12
    assert descriptor.meshing.voxel_resolution == 16
    assert descriptor.meshing.voxel_num_relaxation_iters == 7
    assert descriptor.meshing.voxel_rel_min_tet_volume == pytest.approx(0.08)
    assert descriptor.meshing.voxel_surface_dist_ratio == pytest.approx(0.3)
    assert descriptor.meshing.embedding_impl == "dexsim_exact_cpu"
    assert materials == {}


def test_cloth_descriptor_uses_newton_particle_schema() -> None:
    cfg = ClothObjectCfg(
        uid="cloth",
        shape=MeshCfg(fpath=DEFORMABLE_MESH_PATH),
        particle_radius=0.01,
        validate_mesh=True,
        physical_attr=ClothPhysicalAttributesCfg(
            density=2.5,
            tri_ke=100.0,
            tri_ka=90.0,
            tri_kd=5.0,
            edge_ke=20.0,
            edge_kd=2.0,
            add_springs=True,
            spring_ke=30.0,
            spring_kd=3.0,
        ),
    )

    descriptor, materials = cloth_desc_from_cfg(cfg, per_env=False)

    assert isinstance(descriptor, ClothDesc)
    assert descriptor.mesh.file_path == DEFORMABLE_MESH_PATH
    assert descriptor.particle_radius == pytest.approx(0.01)
    assert descriptor.validate_mesh is True
    assert descriptor.per_env is False
    assert descriptor.physics.surface_density == pytest.approx(2.5)
    assert descriptor.physics.tri_ke == pytest.approx(100.0)
    assert descriptor.physics.tri_ka == pytest.approx(90.0)
    assert descriptor.physics.tri_kd == pytest.approx(5.0)
    assert descriptor.physics.edge_ke == pytest.approx(20.0)
    assert descriptor.physics.edge_kd == pytest.approx(2.0)
    assert descriptor.physics.add_springs is True
    assert descriptor.physics.spring_ke == pytest.approx(30.0)
    assert descriptor.physics.spring_kd == pytest.approx(3.0)
    assert descriptor.physics.dexsim is None
    assert materials == {}


def test_soft_descriptor_rejects_invalid_poisson_ratio() -> None:
    cfg = SoftObjectCfg(
        uid="soft",
        shape=MeshCfg(fpath=DEFORMABLE_MESH_PATH),
        physical_attr=SoftbodyPhysicalAttributesCfg(poissons=0.5),
    )

    with pytest.raises(ValueError, match="poissons"):
        soft_desc_from_cfg(cfg)


@pytest.mark.parametrize("particle_radius", [0.0, float("nan")])
def test_cloth_descriptor_rejects_invalid_particle_radius(
    particle_radius: float,
) -> None:
    cfg = ClothObjectCfg(
        uid="cloth",
        shape=MeshCfg(fpath=DEFORMABLE_MESH_PATH),
        particle_radius=particle_radius,
    )

    with pytest.raises(ValueError, match="particle_radius"):
        cloth_desc_from_cfg(cfg)


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
    if expected_restitution is None:
        assert newton is None
    else:
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


def test_newton_backend_rejects_legacy_flat_rigid_physics() -> None:
    cfg = RigidObjectCfg(
        uid="cube",
        shape=CubeCfg(size=(0.1, 0.1, 0.1)),
        attrs=RigidBodyAttributesCfg(mass=2.0),
    )

    with pytest.raises(TypeError, match="Default-backend-only"):
        rigid_desc_from_cfg(cfg, newton_solver_type="xpbd")


def test_rigid_descriptor_authors_mass_or_density_exclusively() -> None:
    cfg = RigidObjectCfg(
        uid="cube",
        shape=CubeCfg(size=(0.1, 0.1, 0.1)),
        attrs=RigidBodyAttributesCfg(mass=1.0, density=1.0),
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
            RigidBodyAttributesCfg(mass=0.0, inertia=[1.0, 2.0, 3.0]),
            "requires a positive mass",
        ),
        (
            RigidBodyAttributesCfg(mass=1.0, inertia=[1.0, 2.0]),
            "inertia must contain",
        ),
        (
            RigidBodyAttributesCfg(
                mass=1.0,
                com_quaternion=[0.0, 0.0, 0.0, 0.0],
            ),
            "com_quaternion cannot be zero",
        ),
    ],
    ids=["inertia-without-mass", "invalid-inertia-shape", "zero-com-quaternion"],
)
def test_rigid_descriptor_rejects_invalid_mass_properties(
    attrs: RigidBodyAttributesCfg,
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
        attrs=RigidBodyAttributesCfg(
            mass=2.0,
            density=3.0,
            inertia=[1.0, 2.0, 3.0],
            com_position=[0.1, 0.2, 0.3],
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
        attrs=RigidBodyAttributesCfg(mass=2.0, density=3.0),
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
            rigid_props=DexsimRigidBodyPropertiesCfg(linear_damping=0.2),
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
    assert collision.newton.margin == 0.01
    assert collision.newton.mu == 0.4
    assert collision.newton.ke == 1000.0
    assert collision.newton.mu_torsional == 0.02


def test_grouped_rigid_physics_keeps_unset_backend_blocks_absent() -> None:
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
    assert descriptor.collisions[0].dexsim is None
    assert descriptor.collisions[0].newton is None


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
            rigid_props=DexsimRigidBodyPropertiesCfg(linear_damping=0.2),
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
        uid="cube",
        shape=CubeCfg(size=(0.1, 0.1, 0.1)),
        attrs=RigidBodyPhysicsCfg(
            collision_props=NewtonCollisionPropertiesCfg(
                force_sdf=True,
                sdf_padding=0.02,
            )
        ),
    )

    descriptor, _ = rigid_desc_from_cfg(cfg)

    assert descriptor.collisions[0].newton.force_sdf is True
    assert descriptor.collisions[0].newton.sdf_padding == pytest.approx(0.02)


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


def test_newton_backend_rejects_legacy_flat_articulation_physics() -> None:
    cfg = ArticulationCfg(
        uid="robot",
        fpath="robot.urdf",
        asset_physics_mode="overlay",
        attrs=RigidBodyAttributesCfg(mass=2.0),
    )

    with pytest.raises(TypeError, match="Default-backend-only"):
        articulation_desc_from_cfg(cfg, newton_solver_type="xpbd")


def test_grouped_articulation_root_properties_override_legacy_aliases() -> None:
    cfg = ArticulationCfg(
        uid="robot",
        fpath="robot.urdf",
        fix_base=True,
        disable_self_collision=True,
        articulation_props=NewtonArticulationRootPropertiesCfg(
            fixed_base=False,
            self_collision_enabled=True,
        ),
    )

    descriptor = articulation_desc_from_cfg(cfg)

    assert descriptor.fixed_base is False
    assert descriptor.urdf_fix_root_link is False
    assert descriptor.enable_self_collision is True


def test_articulation_descriptor_rejects_newton_acceleration_drive() -> None:
    cfg = ArticulationCfg(
        uid="robot",
        fpath="robot.urdf",
        asset_physics_mode="overlay",
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


def test_newton_articulation_solver_iterations_do_not_warn() -> None:
    cfg = ArticulationCfg(
        uid="robot",
        fpath="robot.urdf",
        asset_physics_mode="overlay",
        min_position_iters=8,
        min_velocity_iters=2,
    )
    descriptor = _resolved_articulation_desc()

    with patch(
        "embodichain.lab.sim.spawn.descriptors.logger.log_warning"
    ) as log_warning:
        configure_articulation_desc(
            descriptor,
            cfg,
            newton_solver_type="mujoco_warp",
        )

    log_warning.assert_not_called()


def test_articulation_config_applies_to_exact_source_resolved_names() -> None:
    cfg = ArticulationCfg(
        uid="robot",
        fpath="robot.urdf",
        asset_physics_mode="overlay",
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
    assert joint.effort_limit == 20.0
    assert joint.velocity_limit == 4.0
    assert joint.lower_limit == -1.0
    assert joint.upper_limit == 1.0


def test_robot_control_part_drive_rule_expands_before_spawn() -> None:
    cfg = RobotCfg(
        uid="robot",
        fpath="robot.urdf",
        control_parts={"arm": ["arm_joint"]},
        drive_pros=JointDrivePropertiesCfg(
            drive_type="force",
            stiffness={"arm": 10.0, "arm_joint": 20.0},
        ),
    )
    descriptor = _resolved_articulation_desc()

    configure_articulation_desc(descriptor, cfg)

    joint = descriptor.get_joint_desc("arm_joint")
    assert joint.dexsim.stiffness == 20.0
    assert joint.newton.target_ke == 20.0


def test_articulation_config_applies_newton_joint_subclass() -> None:
    cfg = ArticulationCfg(
        uid="robot",
        fpath="robot.urdf",
        asset_physics_mode="overlay",
        drive_pros=NewtonJointDrivePropertiesCfg(
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
    assert joint.dexsim.stiffness == 12.0
    assert joint.dexsim.damping == 4.0
    assert joint.dexsim.joint_friction == 0.5
    assert joint.armature == 0.7
    assert joint.newton.target_ke == 12.0
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
                    mass_props=MassPropertiesCfg(mass=2.0),
                    material_props=RigidBodyMaterialCfg(dynamic_friction=0.8),
                ),
                replace_inertial=True,
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
        drive_pros=JointDrivePropertiesCfg(
            drive_type="force",
            stiffness=10.0,
            damping=20.0,
        ),
        qpos_limits={"arm_.*": [-1.0, 1.0]},
    )

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
        drive_pros=JointDrivePropertiesCfg(stiffness=configured_stiffness),
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
        (
            ArticulationCfg(
                uid="robot",
                fpath="robot.urdf",
                drive_pros=NewtonJointDrivePropertiesCfg(
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
