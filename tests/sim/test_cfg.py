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

from dexsim.engine.newton_physics import (
    NewtonCollisionPipelineCfg as DexsimNewtonCollisionPipelineCfg,
)
from dexsim.spawn import DexsimCollisionDesc, DexsimPhysicsDesc, NewtonCollisionDesc
from dexsim.types import DenoiserType, Renderer, ToneMappingType

from embodichain.lab.sim.cfg import (
    ArticulationCfg,
    ArticulationRootPropertiesCfg,
    CollisionPropertiesCfg,
    DexsimCollisionPropertiesCfg,
    DexsimRigidBodyMaterialCfg,
    DexsimRigidBodyPropertiesCfg,
    JointDrivePropertiesCfg,
    MassPropertiesCfg,
    NewtonArticulationRootPropertiesCfg,
    NewtonCollisionPipelineCfg,
    NewtonCollisionPropertiesCfg,
    NewtonJointDrivePropertiesCfg,
    NewtonPhysicsCfg,
    NewtonRigidBodyMaterialCfg,
    NewtonRigidBodyPropertiesCfg,
    PhysicsBackendCfg,
    PhysicsCfg,
    RenderCfg,
    RigidBodyAttributesCfg,
    RigidBodyMaterialCfg,
    RigidBodyPhysicsCfg,
    RigidBodyPropertiesCfg,
    RigidObjectCfg,
    RobotCfg,
)
from embodichain.lab.sim.utility.cfg_utils import merge_robot_cfg


def test_articulation_cfg_defaults_to_preserving_asset_physics() -> None:
    """Generic articulations do not author source drive properties."""
    articulation_cfg = ArticulationCfg()

    assert articulation_cfg.drive_pros is None
    assert articulation_cfg.resolve_asset_physics_mode() == "preserve"


def test_articulation_cfg_parses_sparse_drive_overrides() -> None:
    """Unspecified drive fields remain source-owned."""
    articulation_cfg = ArticulationCfg.from_dict(
        {"drive_pros": {"stiffness": 0.0, "damping": 0.0}}
    )

    assert articulation_cfg.drive_pros.drive_type is None
    assert articulation_cfg.drive_pros.stiffness == 0.0
    assert articulation_cfg.drive_pros.damping == 0.0
    assert articulation_cfg.drive_pros.max_effort is None


def test_robot_cfg_defaults_to_force_joint_drive() -> None:
    """Robots retain force-based joint drives by default."""
    robot_cfg = RobotCfg()

    assert robot_cfg.drive_pros.drive_type == "force"
    assert robot_cfg.resolve_asset_physics_mode() == "overlay"


def test_robot_cfg_partial_drive_properties_preserve_force_drive() -> None:
    """Partial robot drive overrides retain the force-drive default."""
    robot_cfg = RobotCfg.from_dict({"drive_pros": {"stiffness": 0.0, "damping": 0.0}})

    assert robot_cfg.drive_pros.drive_type == "force"


def test_asset_physics_policy_supports_legacy_alias_and_conflict_checks() -> None:
    rigid_cfg = RigidObjectCfg()
    articulation_cfg = ArticulationCfg(use_usd_properties=False)

    assert rigid_cfg.resolve_asset_physics_mode() == "preserve"
    with pytest.warns(DeprecationWarning, match="use_usd_properties"):
        assert articulation_cfg.resolve_asset_physics_mode() == "overlay"

    conflicting_cfg = ArticulationCfg(
        asset_physics_mode="preserve",
        use_usd_properties=False,
    )
    with pytest.raises(ValueError, match="conflicts"):
        conflicting_cfg.resolve_asset_physics_mode()

    invalid_cfg = RigidObjectCfg(asset_physics_mode="replace")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must be 'preserve' or 'overlay'"):
        invalid_cfg.resolve_asset_physics_mode()


def test_articulation_cfg_parses_polymorphic_newton_joint_drive() -> None:
    articulation_cfg = ArticulationCfg.from_dict(
        {
            "drive_pros": {
                "backend": "newton",
                "stiffness": {"arm_.*": 25.0},
                "target_mode": "position",
            }
        }
    )

    assert articulation_cfg.drive_pros.drive_type is None
    assert isinstance(articulation_cfg.drive_pros, NewtonJointDrivePropertiesCfg)
    assert articulation_cfg.drive_pros.stiffness == {"arm_.*": 25.0}
    assert articulation_cfg.drive_pros.target_mode == "position"


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
        drive_pros=NewtonJointDrivePropertiesCfg(
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
            "drive_pros": {"backend": "newton", "damping": 4.0},
            "attrs": {"material_props": {"backend": "newton", "kd": 50.0}},
        },
    )

    assert isinstance(merged.drive_pros, NewtonJointDrivePropertiesCfg)
    assert merged.drive_pros.stiffness == 10.0
    assert merged.drive_pros.damping == 4.0
    assert merged.drive_pros.target_mode == "position"
    assert isinstance(merged.attrs, RigidBodyPhysicsCfg)
    assert isinstance(merged.attrs.material_props, NewtonRigidBodyMaterialCfg)
    assert merged.attrs.material_props.ke == 1000.0
    assert merged.attrs.material_props.kd == 50.0


def test_rigid_physics_property_groups_have_single_backend_roots() -> None:
    """Backend configs extend one logical property root without duplication."""
    assert issubclass(DexsimRigidBodyPropertiesCfg, RigidBodyPropertiesCfg)
    assert issubclass(NewtonRigidBodyPropertiesCfg, RigidBodyPropertiesCfg)
    assert issubclass(DexsimCollisionPropertiesCfg, CollisionPropertiesCfg)
    assert issubclass(NewtonCollisionPropertiesCfg, CollisionPropertiesCfg)
    assert issubclass(NewtonRigidBodyMaterialCfg, RigidBodyMaterialCfg)
    assert issubclass(NewtonJointDrivePropertiesCfg, JointDrivePropertiesCfg)
    assert issubclass(
        NewtonArticulationRootPropertiesCfg,
        ArticulationRootPropertiesCfg,
    )


def test_backend_property_groups_track_dexsim_spawn_descriptors() -> None:
    def names(config_type: type) -> set[str]:
        return {item.name for item in fields(config_type)}

    assert names(DexsimRigidBodyPropertiesCfg) == names(DexsimPhysicsDesc)
    assert (names(DexsimCollisionPropertiesCfg) - {"collision_enabled"}) | names(
        DexsimRigidBodyMaterialCfg
    ) == names(DexsimCollisionDesc)

    newton_fields = (names(NewtonCollisionPropertiesCfg) - {"collision_enabled"}) | (
        names(NewtonRigidBodyMaterialCfg) - names(RigidBodyMaterialCfg)
    )
    newton_fields.remove("torsional_friction")
    newton_fields.remove("rolling_friction")
    newton_fields.update({"mu", "restitution", "mu_torsional", "mu_rolling"})
    assert newton_fields == names(NewtonCollisionDesc)

    assert names(NewtonCollisionPipelineCfg) == names(
        DexsimNewtonCollisionPipelineCfg
    ) - {"requires_grad"}


def test_rigid_physics_from_dict_selects_backend_subclasses() -> None:
    cfg = RigidBodyPhysicsCfg.from_dict(
        {
            "mass_props": {"mass": 2.0},
            "rigid_props": {"backend": "dexsim", "has_gravity": False},
            "collision_props": {"backend": "newton", "margin": 0.01},
            "material_props": {
                "backend": "newton",
                "dynamic_friction": 0.4,
                "ke": 1000.0,
            },
        }
    )

    assert isinstance(cfg.mass_props, MassPropertiesCfg)
    assert isinstance(cfg.rigid_props, DexsimRigidBodyPropertiesCfg)
    assert isinstance(cfg.collision_props, NewtonCollisionPropertiesCfg)
    assert isinstance(cfg.material_props, NewtonRigidBodyMaterialCfg)


def test_backend_property_configs_round_trip_without_losing_subclasses() -> None:
    cfg = RigidBodyPhysicsCfg(
        rigid_props=NewtonRigidBodyPropertiesCfg(),
        collision_props=NewtonCollisionPropertiesCfg(margin=0.01),
        material_props=NewtonRigidBodyMaterialCfg(ke=1000.0),
    )

    serialized = cfg.to_dict()
    restored = RigidBodyPhysicsCfg.from_dict(serialized)

    assert serialized["rigid_props"]["backend"] == "newton"
    assert serialized["collision_props"]["backend"] == "newton"
    assert serialized["material_props"]["backend"] == "newton"
    assert isinstance(restored.rigid_props, NewtonRigidBodyPropertiesCfg)
    assert isinstance(restored.collision_props, NewtonCollisionPropertiesCfg)
    assert isinstance(restored.material_props, NewtonRigidBodyMaterialCfg)


def test_backend_property_parser_infers_unique_fields_without_discriminator() -> None:
    cfg = RigidBodyPhysicsCfg.from_dict(
        {
            "rigid_props": {"linear_damping": 0.2},
            "collision_props": {"margin": 0.01},
            "material_props": {"rolling_friction": 0.03},
        }
    )

    assert isinstance(cfg.rigid_props, DexsimRigidBodyPropertiesCfg)
    assert isinstance(cfg.collision_props, NewtonCollisionPropertiesCfg)
    assert isinstance(cfg.material_props, NewtonRigidBodyMaterialCfg)


def test_backend_joint_and_articulation_configs_round_trip() -> None:
    drive = NewtonJointDrivePropertiesCfg(target_mode=None)
    root = NewtonArticulationRootPropertiesCfg(fixed_base=False)

    restored_drive = JointDrivePropertiesCfg.from_dict(drive.to_dict())
    restored_root = ArticulationRootPropertiesCfg.from_dict(root.to_dict())

    assert isinstance(restored_drive, NewtonJointDrivePropertiesCfg)
    assert isinstance(restored_root, NewtonArticulationRootPropertiesCfg)


def test_robot_cfg_round_trip_preserves_grouped_backend_types() -> None:
    cfg = RobotCfg(
        attrs=RigidBodyPhysicsCfg(
            collision_props=NewtonCollisionPropertiesCfg(margin=0.01),
            material_props=NewtonRigidBodyMaterialCfg(ke=1000.0),
        ),
        drive_pros=NewtonJointDrivePropertiesCfg(target_mode="position"),
        articulation_props=NewtonArticulationRootPropertiesCfg(fixed_base=False),
    )

    restored = RobotCfg.from_dict(cfg.to_dict())

    assert isinstance(restored.attrs, RigidBodyPhysicsCfg)
    assert isinstance(restored.attrs.collision_props, NewtonCollisionPropertiesCfg)
    assert isinstance(restored.attrs.material_props, NewtonRigidBodyMaterialCfg)
    assert isinstance(restored.drive_pros, NewtonJointDrivePropertiesCfg)
    assert isinstance(
        restored.articulation_props,
        NewtonArticulationRootPropertiesCfg,
    )


def test_rigid_physics_from_dict_rejects_unknown_fields() -> None:
    with pytest.raises((KeyError, TypeError)):
        RigidBodyPhysicsCfg.from_dict({"collision_props": {"margn": 0.01}})


def test_robot_cfg_merge_keeps_flat_override_as_default_only_legacy_cfg() -> None:
    base = RobotCfg(
        attrs=RigidBodyPhysicsCfg(
            material_props=RigidBodyMaterialCfg(dynamic_friction=0.8)
        )
    )

    merged = merge_robot_cfg(base, {"attrs": {"mass": 2.0}})

    assert isinstance(merged.attrs, RigidBodyAttributesCfg)
    assert merged.attrs.mass == 2.0
    assert merged.attrs.dynamic_friction == 0.8


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
    cfg = PhysicsCfg(gravity=[0.0, 0.0, -1.5])

    assert cfg.to_dexsim_args()["gravity"] == [0.0, 0.0, -1.5]
    assert PhysicsCfg().to_dexsim_args()["gravity"] == [0.0, 0.0, -9.81]

    with pytest.raises(ValueError, match="three finite values"):
        PhysicsCfg(gravity=[0.0, -9.81]).to_dexsim_args()


def test_physics_cfg_does_not_expose_fixed_solver_options() -> None:
    """Fixed solver implementation details are not part of the public config."""
    physics_cfg = PhysicsCfg()

    assert not hasattr(physics_cfg, "enable_enhanced_determinism")
    assert not hasattr(physics_cfg, "enable_friction_every_iteration")


def test_physics_cfg_applies_fixed_solver_defaults() -> None:
    """Removed solver options retain their established DexSim defaults."""
    physics_args = PhysicsCfg(enable_ccd=True).to_dexsim_args()

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
