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

from dataclasses import fields

import dexsim
import pytest

import embodichain.lab.sim.cfg as sim_cfg

from dexsim.engine.newton_physics import (
    NewtonCollisionPipelineCfg as SpawnNewtonCollisionPipelineCfg,
)
from dexsim.spawn import DexsimCollisionDesc, DexsimPhysicsDesc, NewtonCollisionDesc
from dexsim.types import DenoiserType, Renderer, ToneMappingType

from embodichain.lab.sim.cfg import (
    ArticulationCfg,
    ArticulationRootPropertiesCfg,
    CollisionPropertiesCfg,
    DefaultCollisionPropertiesCfg,
    DefaultPhysicsCfg,
    DefaultRigidBodyPropertiesCfg,
    JointDrivePropertiesCfg,
    LinkPhysicsOverrideCfg,
    MassPropertiesCfg,
    MeshCollisionCfg,
    NewtonCollisionPipelineCfg,
    NewtonCollisionPropertiesCfg,
    NewtonJointDrivePropertiesCfg,
    NewtonPhysicsCfg,
    NewtonRigidBodyMaterialCfg,
    PhysicsBackendCfg,
    RenderCfg,
    RigidBodyMaterialCfg,
    RigidBodyPhysicsCfg,
    RigidObjectCfg,
    RobotCfg,
    RobotPresetCfg,
    physics_cfg_for_backend,
)
from embodichain.lab.sim.utility.cfg_utils import merge_robot_cfg
from embodichain.utils import configclass


def test_cfg_package_preserves_the_public_facade() -> None:
    from embodichain.lab.sim.cfg.rigid import (
        RigidBodyPhysicsCfg as LeafRigidBodyPhysicsCfg,
    )
    from embodichain.lab.sim.cfg.robot import RobotCfg as LeafRobotCfg

    assert hasattr(sim_cfg, "__path__")
    assert not hasattr(sim_cfg, "PhysicsCfg")
    assert sim_cfg.RigidBodyPhysicsCfg is LeafRigidBodyPhysicsCfg
    assert sim_cfg.RobotCfg is LeafRobotCfg


def test_articulation_cfg_defaults_to_preserving_asset_physics() -> None:
    """Generic articulations do not author source drive properties."""
    articulation_cfg = ArticulationCfg()

    assert articulation_cfg.joint_drive_props is None
    assert articulation_cfg.resolve_asset_physics_mode() == "preserve"


def test_articulation_cfg_uses_grouped_physics_fields_only() -> None:
    field_names = {item.name for item in fields(ArticulationCfg)}
    root_props = ArticulationCfg().root_props

    assert {
        "fix_base",
        "disable_self_collision",
        "sleep_threshold",
        "min_position_iters",
        "min_velocity_iters",
        "articulation_props",
        "drive_pros",
        "joint_props",
    }.isdisjoint(field_names)
    assert root_props == ArticulationRootPropertiesCfg()
    assert root_props.fixed_base is True
    assert root_props.self_collision_enabled is False


@pytest.mark.parametrize(
    "field_name",
    [
        "fix_base",
        "disable_self_collision",
        "sleep_threshold",
        "min_position_iters",
        "min_velocity_iters",
        "articulation_props",
        "drive_pros",
        "joint_props",
    ],
)
def test_removed_articulation_fields_fail_with_migration_target(
    field_name: str,
) -> None:
    with pytest.raises(ValueError, match=f"{field_name} ->"):
        ArticulationCfg.from_dict({field_name: True})

    with pytest.raises(ValueError, match=f"{field_name} ->"):
        merge_robot_cfg(RobotCfg(), {field_name: True})


def test_physics_cfg_factory_rejects_noncanonical_backend_names() -> None:
    with pytest.raises(ValueError, match="expected 'default' or 'newton'"):
        physics_cfg_for_backend("alternate")  # type: ignore[arg-type]


def test_articulation_cfg_parses_sparse_drive_overrides() -> None:
    """Unspecified drive fields remain source-owned."""
    articulation_cfg = ArticulationCfg.from_dict(
        {"joint_drive_props": {"stiffness": 0.0, "damping": 0.0}}
    )

    assert articulation_cfg.joint_drive_props.drive_type is None
    assert articulation_cfg.joint_drive_props.stiffness == 0.0
    assert articulation_cfg.joint_drive_props.damping == 0.0
    assert articulation_cfg.joint_drive_props.max_effort is None


def test_robot_cfg_defaults_to_portable_position_velocity_drive() -> None:
    """The original force drive resolves to position+velocity targets."""
    robot_cfg = RobotCfg()

    assert robot_cfg.joint_drive_props.drive_type == "force"
    assert robot_cfg.joint_drive_props.target_mode is None
    assert robot_cfg.joint_drive_props._resolve_modes() == (
        "position_velocity",
        "force",
    )
    assert robot_cfg.resolve_asset_physics_mode() == "overlay"


def test_robot_cfg_partial_drive_properties_preserve_portable_drive() -> None:
    """Partial robot drive overrides retain the original force mode."""
    robot_cfg = RobotCfg.from_dict(
        {"joint_drive_props": {"stiffness": 0.0, "damping": 0.0}}
    )

    assert robot_cfg.joint_drive_props.drive_type == "force"
    assert robot_cfg.joint_drive_props.target_mode is None
    assert robot_cfg.joint_drive_props._resolve_modes() == (
        "position_velocity",
        "force",
    )


def test_drive_type_override_replaces_robot_force_default() -> None:
    override = {"joint_drive_props": {"drive_type": "none"}}
    robot_cfg = RobotCfg.from_dict(override)
    merged_cfg = merge_robot_cfg(RobotCfg(), override)

    for cfg in (robot_cfg, merged_cfg):
        assert cfg.joint_drive_props.target_mode is None
        assert cfg.joint_drive_props.drive_type == "none"
        assert cfg.joint_drive_props._resolve_modes() == ("none", "none")


def test_common_target_mode_does_not_require_newton_subclass() -> None:
    articulation_cfg = ArticulationCfg.from_dict(
        {
            "joint_drive_props": {
                "target_mode": "effort",
                "drive_type": "force",
            }
        }
    )

    assert type(articulation_cfg.joint_drive_props) is JointDrivePropertiesCfg
    assert articulation_cfg.joint_drive_props.target_mode == "effort"
    assert articulation_cfg.joint_drive_props.drive_type == "force"


def test_asset_physics_policy_uses_explicit_modes() -> None:
    rigid_cfg = RigidObjectCfg()
    articulation_cfg = ArticulationCfg()
    robot_cfg = RobotCfg()
    overlay_cfg = ArticulationCfg(asset_physics_mode="overlay")

    assert rigid_cfg.asset_physics_mode == "preserve"
    assert rigid_cfg.resolve_asset_physics_mode() == "preserve"
    assert articulation_cfg.asset_physics_mode == "preserve"
    assert articulation_cfg.resolve_asset_physics_mode() == "preserve"
    assert robot_cfg.asset_physics_mode == "overlay"
    assert robot_cfg.resolve_asset_physics_mode() == "overlay"
    assert overlay_cfg.resolve_asset_physics_mode() == "overlay"

    invalid_cfg = RigidObjectCfg(asset_physics_mode="replace")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must be 'preserve' or 'overlay'"):
        invalid_cfg.resolve_asset_physics_mode()


def test_articulation_cfg_parses_polymorphic_newton_joint_drive() -> None:
    articulation_cfg = ArticulationCfg.from_dict(
        {
            "joint_drive_props": {
                "backend": "newton",
                "stiffness": {"arm_.*": 25.0},
                "target_mode": "position",
            }
        }
    )

    assert articulation_cfg.joint_drive_props.drive_type is None
    assert isinstance(articulation_cfg.joint_drive_props, NewtonJointDrivePropertiesCfg)
    assert articulation_cfg.joint_drive_props.stiffness == {"arm_.*": 25.0}
    assert articulation_cfg.joint_drive_props.target_mode == "position"


def test_joint_drive_from_dict_preserves_newton_subclass_defaults() -> None:
    defaults = NewtonJointDrivePropertiesCfg(
        stiffness=10.0,
        target_mode="position",
    )

    cfg = JointDrivePropertiesCfg.from_dict(
        {"damping": 4.0},
        defaults=defaults,
    )

    assert isinstance(cfg, NewtonJointDrivePropertiesCfg)
    assert cfg.stiffness == 10.0
    assert cfg.damping == 4.0
    assert cfg.target_mode == "position"


def test_robot_cfg_merge_preserves_typed_backend_property_configs() -> None:
    base = RobotCfg(
        joint_drive_props=NewtonJointDrivePropertiesCfg(
            stiffness=10.0,
            target_mode="position",
        ),
        attrs=RigidBodyPhysicsCfg(
            collision_props=NewtonCollisionPropertiesCfg(margin=0.01),
            material_props=NewtonRigidBodyMaterialCfg(ke=1000.0),
        ),
    )

    merged = merge_robot_cfg(
        base,
        {
            "joint_drive_props": {"backend": "newton", "damping": 4.0},
            "attrs": {"material_props": {"backend": "newton", "kd": 50.0}},
        },
    )

    assert isinstance(merged.joint_drive_props, NewtonJointDrivePropertiesCfg)
    assert merged.joint_drive_props.stiffness == 10.0
    assert merged.joint_drive_props.damping == 4.0
    assert merged.joint_drive_props.target_mode == "position"
    assert isinstance(merged.attrs, RigidBodyPhysicsCfg)
    assert isinstance(merged.attrs.material_props, NewtonRigidBodyMaterialCfg)
    assert merged.attrs.material_props.ke == 1000.0
    assert merged.attrs.material_props.kd == 50.0


def test_rigid_physics_uses_one_slot_per_physical_concept() -> None:
    """Backend blocks and geometry cooking are not parallel physics owners."""
    assert {item.name for item in fields(RigidBodyPhysicsCfg)} == {
        "mass_props",
        "rigid_props",
        "collision_props",
        "material_props",
    }
    assert issubclass(DefaultCollisionPropertiesCfg, CollisionPropertiesCfg)
    assert issubclass(NewtonCollisionPropertiesCfg, CollisionPropertiesCfg)
    assert issubclass(NewtonRigidBodyMaterialCfg, RigidBodyMaterialCfg)
    assert issubclass(NewtonJointDrivePropertiesCfg, JointDrivePropertiesCfg)
    for removed_name in (
        "DefaultRigidBodyPhysicsCfg",
        "NewtonRigidBodyPhysicsCfg",
        "MeshCollisionPropertiesCfg",
        "NewtonMeshCollisionPropertiesCfg",
    ):
        assert not hasattr(sim_cfg, removed_name)


def test_backend_property_groups_track_dexsim_spawn_descriptors() -> None:
    def names(config_type: type) -> set[str]:
        return {item.name for item in fields(config_type)}

    assert names(DefaultRigidBodyPropertiesCfg) == names(DexsimPhysicsDesc)
    default_collision_fields = (
        (names(CollisionPropertiesCfg) - {"collision_enabled"})
        | (names(DefaultCollisionPropertiesCfg) - names(CollisionPropertiesCfg))
        | names(RigidBodyMaterialCfg)
    )
    assert default_collision_fields == names(DexsimCollisionDesc)

    newton_fields = (
        names(NewtonCollisionPropertiesCfg) - names(CollisionPropertiesCfg)
    ) | (names(NewtonRigidBodyMaterialCfg) - names(RigidBodyMaterialCfg))
    newton_fields.remove("torsional_friction")
    newton_fields.remove("rolling_friction")
    newton_fields.update(
        {
            "mu",
            "restitution",
            "mu_torsional",
            "mu_rolling",
            "is_hydroelastic",
            "sdf_narrow_band_range",
            "sdf_target_voxel_size",
            "sdf_max_resolution",
            "sdf_texture_format",
            "force_sdf",
            "sdf_padding",
        }
    )
    intentionally_unowned_shape_fields = {
        "is_solid",
        "collision_group",
        "collision_filter_parent",
        "has_particle_collision",
        "is_visible",
        "is_site",
    }
    assert (
        newton_fields == names(NewtonCollisionDesc) - intentionally_unowned_shape_fields
    )

    assert names(NewtonCollisionPipelineCfg) == names(
        SpawnNewtonCollisionPipelineCfg
    ) - {"requires_grad"}


def test_rigid_physics_from_dict_selects_backend_subclasses() -> None:
    cfg = RigidBodyPhysicsCfg.from_dict(
        {
            "mass_props": {"mass": 2.0},
            "rigid_props": {"backend": "default", "has_gravity": False},
            "collision_props": {"backend": "newton", "margin": 0.01},
            "material_props": {
                "backend": "newton",
                "dynamic_friction": 0.4,
                "ke": 1000.0,
            },
        }
    )

    assert isinstance(cfg.mass_props, MassPropertiesCfg)
    assert isinstance(cfg.rigid_props, DefaultRigidBodyPropertiesCfg)
    assert isinstance(cfg.collision_props, NewtonCollisionPropertiesCfg)
    assert isinstance(cfg.material_props, NewtonRigidBodyMaterialCfg)


def test_recompute_inertia_is_a_mass_property() -> None:
    cfg = RigidBodyPhysicsCfg.from_dict(
        {"mass_props": {"mass": 2.0, "recompute_inertia": True}}
    )

    assert "replace_inertial" not in {
        item.name for item in fields(LinkPhysicsOverrideCfg)
    }
    assert cfg.mass_props.recompute_inertia is True
    assert cfg.to_dict()["mass_props"]["recompute_inertia"] is True

    link_cfg = LinkPhysicsOverrideCfg.from_dict(
        {
            "link_names_expr": ["finger_.*"],
            "attrs": {"mass_props": {"recompute_inertia": True}},
        }
    )
    assert link_cfg.attrs.mass_props.recompute_inertia is True

    with pytest.raises(ValueError, match="attrs.mass_props.recompute_inertia"):
        LinkPhysicsOverrideCfg.from_dict(
            {
                "link_names_expr": ["finger_.*"],
                "replace_inertial": True,
            }
        )


@pytest.mark.parametrize(
    ("removed_field", "replacement"),
    [
        ("default_props", "polymorphic property slot"),
        ("newton_props", "polymorphic property slot"),
        ("mesh_collision_props", "MeshCfg.collision"),
    ],
)
def test_rigid_physics_rejects_removed_parallel_owners(
    removed_field: str,
    replacement: str,
) -> None:
    with pytest.raises(ValueError, match=replacement):
        RigidBodyPhysicsCfg.from_dict({removed_field: {}})


def test_articulation_cfg_parses_joint_drive_and_dynamics() -> None:
    cfg = ArticulationCfg.from_dict(
        {
            "joint_drive_props": {
                "stiffness": 12.0,
                "max_effort": 20.0,
                "friction": {"arm_.*": 0.2},
            },
        }
    )

    assert cfg.joint_drive_props.stiffness == pytest.approx(12.0)
    assert cfg.joint_drive_props.max_effort == pytest.approx(20.0)
    assert cfg.joint_drive_props.friction == {"arm_.*": 0.2}


def test_robot_cfg_merge_composes_single_slot_and_joint_drive_properties() -> None:
    base = RobotCfg(
        attrs=RigidBodyPhysicsCfg(
            rigid_props=DefaultRigidBodyPropertiesCfg(linear_damping=0.1),
        ),
        joint_drive_props=JointDrivePropertiesCfg(
            max_effort={"arm": 10.0},
            friction=0.1,
        ),
    )

    merged = merge_robot_cfg(
        base,
        {
            "attrs": {
                "rigid_props": {
                    "backend": "default",
                    "angular_damping": 0.2,
                },
            },
            "joint_drive_props": {
                "max_effort": {"wrist": 20.0},
                "armature": 0.3,
            },
        },
    )

    assert merged.attrs.rigid_props.linear_damping == pytest.approx(0.1)
    assert merged.attrs.rigid_props.angular_damping == pytest.approx(0.2)
    assert merged.joint_drive_props.max_effort == {"arm": 10.0, "wrist": 20.0}
    assert merged.joint_drive_props.friction == pytest.approx(0.1)
    assert merged.joint_drive_props.armature == pytest.approx(0.3)


def test_portable_collision_envelope_round_trips_as_common_config() -> None:
    cfg = RigidBodyPhysicsCfg.from_dict(
        {
            "collision_props": {
                "collision_enabled": True,
                "contact_offset": 0.01,
                "rest_offset": 0.002,
            }
        }
    )

    assert type(cfg.collision_props) is CollisionPropertiesCfg
    assert cfg.to_dict()["collision_props"] == {
        "collision_enabled": True,
        "contact_offset": 0.01,
        "rest_offset": 0.002,
    }


@configclass
class _RobotPhysicsPresetCfg(RobotPresetCfg):
    default: RobotCfg = RobotCfg(uid="default")
    newton: RobotCfg = RobotCfg(uid="newton")
    newton_xpbd: RobotCfg = RobotCfg(uid="newton_xpbd")


def test_robot_preset_selects_complete_backend_and_solver_variants() -> None:
    preset = _RobotPhysicsPresetCfg()

    default_cfg = preset.resolve(DefaultPhysicsCfg())
    newton_cfg = preset.resolve(NewtonPhysicsCfg())
    xpbd_cfg = preset.resolve(NewtonPhysicsCfg(solver_cfg={"solver_type": "xpbd"}))

    assert default_cfg.uid == "default"
    assert newton_cfg.uid == "newton"
    assert xpbd_cfg.uid == "newton_xpbd"
    assert default_cfg is not preset.default


@configclass
class _CommonRobotPresetCfg(RobotPresetCfg):
    default: RobotCfg = RobotCfg(uid="portable")


def test_robot_preset_falls_back_to_one_portable_definition() -> None:
    preset = _CommonRobotPresetCfg()

    assert preset.resolve(DefaultPhysicsCfg()).uid == "portable"
    assert preset.resolve(NewtonPhysicsCfg()).uid == "portable"


@configclass
class _NewtonSolverAliasRobotPresetCfg(RobotPresetCfg):
    default: RobotCfg = RobotCfg(uid="fallback")
    newton_mjwarp: RobotCfg = RobotCfg(uid="mjwarp")


def test_robot_preset_accepts_newton_solver_alias() -> None:
    preset = _NewtonSolverAliasRobotPresetCfg()

    assert preset.resolve(DefaultPhysicsCfg()).uid == "fallback"
    assert preset.resolve(NewtonPhysicsCfg()).uid == "fallback"
    assert (
        preset.resolve(NewtonPhysicsCfg(solver_cfg={"solver_type": "mjwarp"})).uid
        == "mjwarp"
    )


@configclass
class _UnsupportedRobotPresetCfg(RobotPresetCfg):
    default: RobotCfg = RobotCfg(uid="default")
    alternate: RobotCfg = RobotCfg(uid="alternate")


def test_robot_preset_rejects_noncanonical_backend_names() -> None:
    with pytest.raises(TypeError, match="unsupported preset name"):
        _UnsupportedRobotPresetCfg().resolve(DefaultPhysicsCfg())


def test_backend_property_configs_round_trip_without_losing_subclasses() -> None:
    cfg = RigidBodyPhysicsCfg(
        collision_props=NewtonCollisionPropertiesCfg(margin=0.01),
        material_props=NewtonRigidBodyMaterialCfg(ke=1000.0),
    )

    serialized = cfg.to_dict()
    restored = RigidBodyPhysicsCfg.from_dict(serialized)

    assert serialized["rigid_props"] is None
    assert serialized["collision_props"]["backend"] == "newton"
    assert serialized["material_props"]["backend"] == "newton"
    assert isinstance(restored.collision_props, NewtonCollisionPropertiesCfg)
    assert isinstance(restored.material_props, NewtonRigidBodyMaterialCfg)


def test_default_property_configs_use_the_default_discriminator() -> None:
    cfg = RigidBodyPhysicsCfg(
        rigid_props=DefaultRigidBodyPropertiesCfg(linear_damping=0.2),
        collision_props=DefaultCollisionPropertiesCfg(
            contact_offset=0.01,
            disable_strong_friction=True,
        ),
        material_props=RigidBodyMaterialCfg(dynamic_friction=0.5),
    )

    serialized = cfg.to_dict()
    restored = RigidBodyPhysicsCfg.from_dict(serialized)

    assert serialized["rigid_props"]["backend"] == "default"
    assert serialized["collision_props"]["backend"] == "default"
    assert "backend" not in serialized["material_props"]
    assert isinstance(restored.rigid_props, DefaultRigidBodyPropertiesCfg)
    assert isinstance(restored.collision_props, DefaultCollisionPropertiesCfg)
    assert type(restored.material_props) is RigidBodyMaterialCfg


def test_backend_property_parser_infers_unique_fields_without_discriminator() -> None:
    cfg = RigidBodyPhysicsCfg.from_dict(
        {
            "rigid_props": {"linear_damping": 0.2},
            "collision_props": {"margin": 0.01},
            "material_props": {"rolling_friction": 0.03},
        }
    )

    assert isinstance(cfg.rigid_props, DefaultRigidBodyPropertiesCfg)
    assert isinstance(cfg.collision_props, NewtonCollisionPropertiesCfg)
    assert isinstance(cfg.material_props, NewtonRigidBodyMaterialCfg)


def test_mesh_collision_cfg_requires_explicit_strategy_fields() -> None:
    collision = MeshCollisionCfg(
        approximation="convex_decomposition",
        max_hulls=8,
        acd_method="coacd",
    )

    assert collision.max_hulls == 8
    with pytest.raises(ValueError, match="valid only for convex_decomposition"):
        MeshCollisionCfg(approximation="convex_hull", acd_method="coacd")
    with pytest.raises(ValueError, match="only one"):
        MeshCollisionCfg(
            approximation="sdf",
            sdf_resolution=64,
            sdf_target_voxel_size=0.005,
        )


@pytest.mark.parametrize(
    ("legacy_max_hulls", "expected_approximation", "expected_max_hulls"),
    [
        (1, "convex_hull", None),
        (4, "convex_decomposition", 4),
    ],
)
def test_mesh_collision_cfg_accepts_deprecated_hull_count_alias(
    legacy_max_hulls: int,
    expected_approximation: str,
    expected_max_hulls: int | None,
) -> None:
    with pytest.warns(DeprecationWarning):
        collision = MeshCollisionCfg(max_convex_hull_num=legacy_max_hulls)

    assert collision.approximation == expected_approximation
    assert collision.max_hulls == expected_max_hulls
    assert "max_convex_hull_num" not in collision.to_dict()


def test_mesh_collision_cfg_deprecated_hull_count_view_uses_canonical_value() -> None:
    collision = MeshCollisionCfg(
        approximation="convex_decomposition",
        max_hulls=4,
    )

    assert collision.max_convex_hull_num == 4


def test_mesh_collision_cfg_rejects_both_hull_count_names() -> None:
    with (
        pytest.warns(DeprecationWarning),
        pytest.raises(ValueError, match="cannot both be configured"),
    ):
        MeshCollisionCfg(
            approximation="convex_decomposition",
            max_hulls=4,
            max_convex_hull_num=8,
        )


@pytest.mark.parametrize(
    "collision_kwargs",
    [
        {"approximation": "convex_decomposition", "max_hulls": 2.5},
        {"approximation": "sdf", "sdf_resolution": 64.5},
        {"approximation": "sdf", "sdf_padding": float("nan")},
        {"approximation": "sdf", "sdf_texture_format": "invalid"},
    ],
)
def test_mesh_collision_cfg_rejects_invalid_numeric_types_and_values(
    collision_kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        MeshCollisionCfg(**collision_kwargs)


def test_mesh_cfg_legacy_collision_fields_normalize_to_nested_config() -> None:
    with pytest.warns(DeprecationWarning):
        cfg = RigidObjectCfg.from_dict(
            {
                "uid": "mesh",
                "shape": {
                    "shape_type": "Mesh",
                    "fpath": "mesh.obj",
                    "max_convex_hull_num": 4,
                    "acd_method": "coacd",
                },
            }
        )

    assert cfg.shape.collision == MeshCollisionCfg(
        approximation="convex_decomposition",
        max_hulls=4,
        acd_method="coacd",
    )
    serialized_shape = cfg.shape.to_dict()
    assert "max_convex_hull_num" not in serialized_shape
    assert serialized_shape["collision"]["approximation"] == "convex_decomposition"


def test_rigid_object_legacy_physics_mesh_collision_moves_to_shape() -> None:
    with pytest.warns(DeprecationWarning):
        cfg = RigidObjectCfg.from_dict(
            {
                "uid": "mesh",
                "shape": {"shape_type": "Mesh", "fpath": "mesh.obj"},
                "attrs": {
                    "mesh_collision_props": {"max_convex_hull_num": 4},
                },
            }
        )

    assert cfg.shape.collision.approximation == "convex_decomposition"
    assert cfg.shape.collision.max_hulls == 4
    assert "mesh_collision_props" not in cfg.attrs.to_dict()


def test_legacy_mesh_collision_physics_rejects_non_mesh_shape() -> None:
    with pytest.raises(ValueError, match="only to a MeshCfg"):
        RigidObjectCfg.from_dict(
            {
                "uid": "cube",
                "shape": {"shape_type": "Cube", "size": [1.0, 1.0, 1.0]},
                "attrs": {
                    "mesh_collision_props": {"max_convex_hull_num": 4},
                },
            }
        )


def test_backend_joint_and_articulation_configs_round_trip() -> None:
    drive = NewtonJointDrivePropertiesCfg(target_mode=None)
    root = ArticulationRootPropertiesCfg(fixed_base=False)

    restored_drive = JointDrivePropertiesCfg.from_dict(drive.to_dict())
    restored_root = ArticulationRootPropertiesCfg.from_dict(root.to_dict())

    assert isinstance(restored_drive, NewtonJointDrivePropertiesCfg)
    assert root.to_dict() == {
        "fixed_base": False,
        "self_collision_enabled": False,
        "sleep_threshold": None,
        "min_position_iters": None,
        "min_velocity_iters": None,
    }
    assert type(restored_root) is ArticulationRootPropertiesCfg


def test_articulation_root_config_rejects_backend_discriminator() -> None:
    with pytest.raises(TypeError, match="backend"):
        ArticulationRootPropertiesCfg.from_dict(
            {"backend": "newton", "fixed_base": False}
        )


def test_articulation_root_config_round_trip() -> None:
    root = ArticulationRootPropertiesCfg(
        fixed_base=True,
        sleep_threshold=0.005,
        min_position_iters=8,
        min_velocity_iters=2,
    )

    restored = ArticulationRootPropertiesCfg.from_dict(root.to_dict())

    assert type(restored) is ArticulationRootPropertiesCfg
    assert restored == root
    assert "backend" not in root.to_dict()


def test_articulation_root_requires_both_solver_iteration_counts() -> None:
    with pytest.raises(ValueError, match="must be configured together"):
        ArticulationRootPropertiesCfg(min_position_iters=8)


def test_robot_cfg_round_trip_preserves_grouped_backend_types() -> None:
    cfg = RobotCfg(
        attrs=RigidBodyPhysicsCfg(
            collision_props=NewtonCollisionPropertiesCfg(margin=0.01),
            material_props=NewtonRigidBodyMaterialCfg(ke=1000.0),
        ),
        joint_drive_props=NewtonJointDrivePropertiesCfg(target_mode="position"),
        root_props=ArticulationRootPropertiesCfg(fixed_base=False),
    )

    restored = RobotCfg.from_dict(cfg.to_dict())

    assert isinstance(restored.attrs, RigidBodyPhysicsCfg)
    assert isinstance(restored.attrs.collision_props, NewtonCollisionPropertiesCfg)
    assert isinstance(restored.attrs.material_props, NewtonRigidBodyMaterialCfg)
    assert isinstance(restored.joint_drive_props, NewtonJointDrivePropertiesCfg)
    assert type(restored.root_props) is ArticulationRootPropertiesCfg


def test_rigid_physics_from_dict_rejects_unknown_fields() -> None:
    with pytest.raises((KeyError, TypeError)):
        RigidBodyPhysicsCfg.from_dict({"collision_props": {"margn": 0.01}})


def test_robot_cfg_merge_preserves_grouped_overrides() -> None:
    base = RobotCfg(
        attrs=RigidBodyPhysicsCfg(
            material_props=RigidBodyMaterialCfg(dynamic_friction=0.8)
        )
    )

    merged = merge_robot_cfg(base, {"attrs": {"mass_props": {"mass": 2.0}}})

    assert isinstance(merged.attrs, RigidBodyPhysicsCfg)
    assert merged.attrs.mass_props.mass == 2.0
    assert merged.attrs.material_props.dynamic_friction == 0.8


def test_newton_physics_inherits_common_gravity_and_collision_config() -> None:
    cfg = NewtonPhysicsCfg(
        gravity=[0.0, 0.0, -1.5],
        collision_cfg=NewtonCollisionPipelineCfg(
            broad_phase="sap",
            rigid_contact_max=1234,
        ),
    )

    assert isinstance(cfg, PhysicsBackendCfg)
    dexsim_cfg = cfg.to_dexsim_cfg(gpu_id=0)
    assert dexsim_cfg.gravity == [0.0, 0.0, -1.5]
    assert dexsim_cfg.collision_pipeline_cfg.broad_phase == "sap"
    assert dexsim_cfg.collision_pipeline_cfg.rigid_contact_max == 1234


def test_newton_physics_normalizes_mapping_collision_config() -> None:
    cfg = NewtonPhysicsCfg(
        collision_cfg={"broad_phase": "sap", "rigid_contact_max": 12}
    )

    assert isinstance(cfg.collision_cfg, NewtonCollisionPipelineCfg)
    assert cfg.collision_cfg.broad_phase == "sap"
    assert cfg.collision_cfg.rigid_contact_max == 12


def test_default_physics_accepts_the_same_gravity_input_shape() -> None:
    cfg = DefaultPhysicsCfg(gravity=[0.0, 0.0, -1.5])

    assert cfg.to_dexsim_args()["gravity"] == [0.0, 0.0, -1.5]
    assert DefaultPhysicsCfg().to_dexsim_args()["gravity"] == [0.0, 0.0, -9.81]

    with pytest.raises(ValueError, match="three finite values"):
        DefaultPhysicsCfg(gravity=[0.0, -9.81]).to_dexsim_args()


def test_default_physics_cfg_does_not_expose_fixed_solver_options() -> None:
    """Fixed solver implementation details are not part of the public config."""
    physics_cfg = DefaultPhysicsCfg()

    assert not hasattr(physics_cfg, "enable_enhanced_determinism")
    assert not hasattr(physics_cfg, "enable_friction_every_iteration")


def test_default_physics_cfg_applies_fixed_solver_defaults() -> None:
    """Removed solver options retain the Default backend's established values."""
    physics_args = DefaultPhysicsCfg(enable_ccd=True).to_dexsim_args()

    assert physics_args["enable_ccd"] is True
    assert physics_args["enable_enhanced_determinism"] is False
    assert physics_args["enable_friction_every_iteration"] is True


def test_render_cfg_applies_default_denoiser() -> None:
    """Rendering always uses the default OptiX denoiser."""
    world_config = dexsim.WorldConfig()

    RenderCfg(renderer="hybrid").apply_to_dexsim_config(world_config)

    assert world_config.raytrace_config.open_denoise is True
    assert world_config.raytrace_config.denoiser_type == DenoiserType.OPTIX


def test_render_cfg_does_not_expose_denoiser_options() -> None:
    """Denoiser implementation details are not part of EmbodiChain's API."""
    render_cfg = RenderCfg()

    assert not hasattr(render_cfg, "denoiser_enabled")
    assert not hasattr(render_cfg, "denoiser_type")


def test_render_cfg_applies_tone_mapping_and_fixed_exposure() -> None:
    """Tone mapping forwards its curve and fixed exposure to DexSim."""
    expected_exposure = 1.25
    world_config = dexsim.WorldConfig()
    render_cfg = RenderCfg(
        renderer="rt",
        tone_mapping_enabled=True,
        tone_mapping_exposure=expected_exposure,
    )

    render_cfg.apply_to_dexsim_config(world_config)

    assert world_config.postprocess_config.tone_mapping_enabled is True
    assert (
        world_config.postprocess_config.tone_mapping_type
        == ToneMappingType.MODIFIED_REINHARD
    )
    assert world_config.postprocess_config.tone_mapping_exposure == expected_exposure


def test_render_cfg_applies_renderer_and_sample_count() -> None:
    """The consolidated conversion preserves existing renderer settings."""
    expected_spp = 8
    world_config = dexsim.WorldConfig()

    RenderCfg(renderer="fast-rt", spp=expected_spp).apply_to_dexsim_config(world_config)

    assert world_config.renderer == Renderer.FASTRT
    assert world_config.raytrace_config.render_iterations_per_frame == expected_spp


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("tone_mapping_exposure", -0.1),
        ("spp", 0),
    ],
)
def test_render_cfg_rejects_invalid_image_processing_settings(
    field_name: str, invalid_value: object
) -> None:
    """Invalid image-processing values fail at configuration construction."""
    with pytest.raises(ValueError):
        RenderCfg(**{field_name: invalid_value})
