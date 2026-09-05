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

import dexsim
import pytest

from dexsim.types import DenoiserType, Renderer, ToneMappingType

from embodichain.lab.sim.cfg import (
    ArticulationCfg,
    DLSSCfg,
    PhysicsCfg,
    RenderCfg,
    RobotCfg,
)


def test_articulation_cfg_defaults_to_no_joint_drive() -> None:
    """Generic articulations are passive unless a drive is requested."""
    articulation_cfg = ArticulationCfg()

    assert articulation_cfg.drive_pros.drive_type == "none"


def test_articulation_cfg_partial_drive_properties_preserve_no_drive() -> None:
    """Partial articulation drive overrides retain the passive default."""
    articulation_cfg = ArticulationCfg.from_dict(
        {"drive_pros": {"stiffness": 0.0, "damping": 0.0}}
    )

    assert articulation_cfg.drive_pros.drive_type == "none"


def test_articulation_cfg_enables_gravity_by_default() -> None:
    """Articulations opt into gravity unless explicitly configured otherwise."""
    assert ArticulationCfg().enable_gravity is True


def test_articulation_cfg_parses_disabled_gravity() -> None:
    """Dictionary configuration can disable articulation gravity."""
    articulation_cfg = ArticulationCfg.from_dict({"enable_gravity": False})

    assert articulation_cfg.enable_gravity is False


def test_robot_cfg_defaults_to_force_joint_drive() -> None:
    """Robots retain force-based joint drives by default."""
    robot_cfg = RobotCfg()

    assert robot_cfg.drive_pros.drive_type == "force"


def test_robot_cfg_partial_drive_properties_preserve_force_drive() -> None:
    """Partial robot drive overrides retain the force-drive default."""
    robot_cfg = RobotCfg.from_dict({"drive_pros": {"stiffness": 0.0, "damping": 0.0}})

    assert robot_cfg.drive_pros.drive_type == "force"


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


def test_dlss_defaults_enable_offscreen_and_preserve_quality_resolution() -> None:
    """Offscreen DLSS defaults on while quality and resolution retain native defaults."""
    native = dexsim.DLSSConfig()
    converted = DLSSCfg().to_dexsim_cfg(1920, 1080)

    for name in DLSSCfg().to_dict():
        if name not in ("upsample_ratio", "offscreen_dlss_enabled"):
            assert getattr(converted, name) == getattr(native, name), name
    assert converted.render_width == converted.render_height == 0
    assert converted.dlss_quality == 2
    assert converted.offscreen_dlss_enabled is True


@pytest.mark.parametrize("rr_enabled", [False, True])
@pytest.mark.parametrize("sr_enabled", [False, True])
def test_dlss_features_are_independent(rr_enabled: bool, sr_enabled: bool) -> None:
    """RR and SR selections survive conversion independently."""
    converted = DLSSCfg(
        rayreconstruction_enabled=rr_enabled,
        upscale_enabled=sr_enabled,
    ).to_dexsim_cfg(1920, 1080)

    assert converted.rayreconstruction_enabled is rr_enabled
    assert converted.upscale_enabled is sr_enabled


@pytest.mark.parametrize("quality", range(-1, 6))
def test_dlss_quality_presets_keep_automatic_render_dimensions(quality: int) -> None:
    """EmbodiChain forwards quality instead of hard-coding an internal scale."""
    converted = DLSSCfg(dlss_quality=quality).to_dexsim_cfg(1920, 1080)

    assert converted.dlss_quality == quality
    assert converted.render_width == converted.render_height == 0


@pytest.mark.parametrize(
    ("render_size", "expected_size"),
    [
        ((0, 0), (960, 540)),
        ((1280, 0), (1280, 540)),
        ((0, 720), (960, 720)),
        ((1280, 720), (1280, 720)),
    ],
)
def test_dlss_ratio_uses_window_size_and_preserves_explicit_dimensions(
    render_size: tuple[int, int], expected_size: tuple[int, int]
) -> None:
    """The ratio fills only zero dimensions using the actual window target."""
    converted = DLSSCfg(
        upsample_ratio=2.0,
        render_width=render_size[0],
        render_height=render_size[1],
        target_width=3840,
        target_height=2160,
        exposure_compensation=1.5,
    ).to_dexsim_cfg(1920, 1080)

    assert (converted.render_width, converted.render_height) == expected_size
    assert (converted.target_width, converted.target_height) == (3840, 2160)
    assert converted.exposure_compensation == pytest.approx(1.5)


def test_dlss_ratio_clamps_small_internal_dimensions() -> None:
    """An explicit ratio cannot produce an invalid zero-pixel render target."""
    converted = DLSSCfg(upsample_ratio=8.0).to_dexsim_cfg(4, 2)

    assert converted.render_width == converted.render_height == 1


def test_render_cfg_instances_do_not_share_dlss_settings() -> None:
    """Changing one rendering configuration does not alter another."""
    first, second = RenderCfg(), RenderCfg()
    first.dlss.dlss_enabled = False

    assert second.dlss.dlss_enabled is True


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("dlss_quality", -2),
        ("dlss_quality", 6),
        ("dlss_quality", 2.5),
        ("render_width", -1),
        ("render_height", -1),
        ("target_width", -1),
        ("target_height", -1),
        ("render_width", 1.5),
        ("upsample_ratio", 0.5),
        ("upsample_ratio", float("inf")),
        ("upsample_ratio", float("nan")),
        ("exposure_compensation", 0.0),
        ("exposure_compensation", -1.0),
        ("exposure_compensation", float("inf")),
        ("exposure_compensation", float("nan")),
    ],
)
def test_dlss_rejects_invalid_settings(field_name: str, invalid_value: object) -> None:
    """Invalid DLSS values fail before native configuration or rendering."""
    with pytest.raises(ValueError, match=field_name):
        DLSSCfg(**{field_name: invalid_value})


def test_dlss_conversion_revalidates_mutated_settings() -> None:
    """Mutable config edits are checked before entering the native binding."""
    config = DLSSCfg()
    config.upsample_ratio = 0.0

    with pytest.raises(ValueError, match="upsample_ratio"):
        config.to_dexsim_cfg(1920, 1080)
