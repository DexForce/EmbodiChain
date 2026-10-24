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

"""Material identity, native state, and configuration-lowering contracts."""

from __future__ import annotations

from unittest.mock import MagicMock
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from embodichain.lab.sim.material import (
    VisualMaterial,
    VisualMaterialCfg,
    VisualMaterialInst,
)
from embodichain.lab.sim.material import _set_render_material
from embodichain.lab.sim.sim_manager import SimulationManager
from embodichain.lab.sim.spawn.descriptors import _compile_visual_material
from embodichain.lab.sim.spawn.scene import SpawnScene
from embodichain.lab.sim.objects.rigid_object_group import RigidObjectGroup
from embodichain.lab.gym.envs.managers.cfg import SceneEntityCfg
from embodichain.lab.gym.envs.managers.randomization.visual import (
    set_rigid_object_group_visual_material,
)

pytestmark = pytest.mark.no_sim


def test_material_assignment_uses_native_identity_even_when_names_match():
    original = MagicMock(name="original_handle")
    replacement = MagicMock(name="replacement_handle")
    for native in (original, replacement):
        native.get_name.return_value = "same_name"
        native.get_template.return_value.get_name.return_value = "same_template"
    original.get_material_id.return_value = 1
    replacement.get_material_id.return_value = 2
    body = MagicMock()
    body.get_material.return_value = original

    _set_render_material(body, 0, replacement)

    body.set_material.assert_called_once_with(0, replacement)


def _native_template() -> MagicMock:
    template = MagicMock(name="template")
    instances = {}

    def get_instance(uid=""):
        if uid not in instances:
            state = {
                "base_color": [0.5, 0.5, 0.5, 1.0],
                "metallic": 0.0,
                "roughness": 0.7,
                "ior": 1.5,
                "emissive": [0.0, 0.0, 0.0, 1.0],
            }
            native = MagicMock(name=uid)
            native.get_name.return_value = uid
            native.get_template.return_value = template
            native.get_base_color_map.return_value = None
            for field in state:
                getattr(native, f"get_{field}").side_effect = lambda key=field: state[
                    key
                ]
                getattr(native, f"set_{field}").side_effect = (
                    lambda value, key=field: state.__setitem__(key, value)
                )
            native.set_pbr_param.side_effect = lambda key, value: state.__setitem__(
                key, value
            )
            instances[uid] = native
        return instances[uid]

    template.get_inst.side_effect = get_instance
    return template


def test_repeated_instance_creation_preserves_identity_properties_and_capacity():
    material = VisualMaterial(VisualMaterialCfg(uid="paint"), _native_template())
    instance = material.create_instance("object")
    instance.set_roughness(0.2)
    instance.mat.reset_mock()

    for _ in range(1025):  # Exceeds the historical bounded-texture stress case.
        assert material.create_instance("object") is instance
        assert material.get_instance("object") is instance

    assert instance.roughness == pytest.approx(0.2)
    assert len(material._instances) == 2  # Template default plus one object instance.
    instance.mat.set_roughness.assert_not_called()


def test_instance_properties_read_current_native_values():
    material = VisualMaterial(VisualMaterialCfg(uid="paint"), _native_template())
    instance = material.create_instance("object")
    instance.mat.set_roughness(0.9)
    instance.mat.set_metallic(0.6)
    instance.mat.set_base_color([0.1, 0.2, 0.3, 1.0])
    instance.mat.set_pbr_param("ior", 2.0)

    queried = material.get_instance("object")

    assert queried.roughness == pytest.approx(0.9)
    assert queried.metallic == pytest.approx(0.6)
    assert queried.ior == pytest.approx(2.0)
    assert queried.base_color == pytest.approx([0.1, 0.2, 0.3])


@pytest.mark.parametrize(
    ("setter", "value", "native_setter"),
    [
        ("set_base_color", [0.1, 0.2, 0.3, 1.0], "set_base_color"),
        ("set_metallic", 0.8, "set_metallic"),
        ("set_roughness", 0.2, "set_roughness"),
        ("set_emissive", [0.1, 0.2, 0.3], "set_emissive"),
        ("set_ior", 2.0, "set_pbr_param"),
        ("set_base_color_texture", "color.png", "set_base_color_map"),
        ("set_metallic_texture", "metal.png", "set_metallic_map"),
        ("set_roughness_texture", "rough.png", "set_roughness_map"),
        ("set_normal_texture", "normal.png", "set_normal_map"),
        ("set_ao_texture", "ao.png", "set_ao_map"),
    ],
)
def test_asset_wrapper_setters_use_exact_existing_handle(setter, value, native_setter):
    template = _native_template()
    original = template.get_inst("parsed")
    wrapper = VisualMaterialInst.from_existing(original)
    template.reset_mock()

    getattr(wrapper, setter)(value)

    assert wrapper.mat is original
    template.get_inst.assert_not_called()
    getattr(original, native_setter).assert_called_once()


@pytest.mark.parametrize("intensity", [0.0, 1.0, 10.0])
def test_runtime_and_spawn_lower_identical_material_properties(intensity):
    cfg = VisualMaterialCfg(
        uid="paint",
        base_color=[0.1, 0.2, 0.3, 0.5],
        metallic=0.8,
        roughness=0.2,
        emissive=[0.2, 0.3, 0.4],
        emissive_intensity=intensity,
        ior=2.0,
    )
    _, (_, descriptor) = _compile_visual_material("object", cfg)
    material = VisualMaterial(cfg, _native_template())
    native = material.inst.mat

    assert material.inst.base_color == pytest.approx(descriptor.base_color[:3])
    assert material.inst.metallic == pytest.approx(descriptor.metallic)
    assert material.inst.roughness == pytest.approx(descriptor.roughness)
    assert material.inst.ior == pytest.approx(descriptor.ior)
    np.testing.assert_allclose(
        native.set_emissive.call_args.args[0], descriptor.emissive
    )
    assert descriptor.material_type is None
    native.update_pbr_material_type.assert_not_called()
    assert descriptor.has_alpha is True


def test_emissive_intensity_can_be_zero_then_restored_without_losing_rgb():
    instance = VisualMaterial(
        VisualMaterialCfg(uid="glow", emissive=[0.2, 0.3, 0.4]), _native_template()
    ).inst

    instance.set_emissive_intensity(0.0)
    np.testing.assert_allclose(instance.mat.get_emissive(), [0.0, 0.0, 0.0, 1.0])
    instance.set_emissive_intensity(2.0)

    np.testing.assert_allclose(instance.mat.get_emissive(), [0.4, 0.6, 0.8, 1.0])
    assert instance.emissive == pytest.approx([0.2, 0.3, 0.4])


@pytest.mark.parametrize(
    "properties",
    [
        {"base_color": [1.0, 0.0, 0.0]},
        {"base_color": [1.0, 0.0, 0.0, 2.0]},
        {"emissive": [0.0, 0.0]},
        {"metallic": float("nan")},
        {"roughness": 1.1},
        {"ior": 0.0},
        {"emissive_intensity": -1.0},
        {"emissive": [1e308, 0.0, 0.0], "emissive_intensity": 1e308},
    ],
)
def test_invalid_material_config_is_rejected_before_native_instance_creation(
    properties,
):
    cfg = VisualMaterialCfg(uid="invalid", **properties)
    template = _native_template()

    with pytest.raises(ValueError):
        VisualMaterial(cfg, template)
    with pytest.raises(ValueError):
        _compile_visual_material("object", cfg)

    template.get_inst.assert_not_called()


def test_unnamed_inline_materials_have_independent_object_scoped_names():
    red_ref, (_, red) = _compile_visual_material(
        "red", VisualMaterialCfg(base_color=[1, 0, 0, 1])
    )
    blue_ref, (_, blue) = _compile_visual_material(
        "blue", VisualMaterialCfg(base_color=[0, 0, 1, 1])
    )
    definitions = {red_ref: red, blue_ref: blue}

    assert len(definitions) == 2
    assert definitions[red_ref].base_color == (1.0, 0.0, 0.0, 1.0)
    assert definitions[blue_ref].base_color == (0.0, 0.0, 1.0, 1.0)


def _material_manager():
    manager = SimulationManager.__new__(SimulationManager)
    manager._visual_materials = {}
    manager._require_native_renderer = MagicMock()
    manager._env = MagicMock()
    manager._env.find_material.return_value = None
    manager._env.create_pbr_material.side_effect = lambda *args: _native_template()
    scene = SpawnScene.__new__(SpawnScene)
    scene.builder = SimpleNamespace(materials={})
    scene._usd_texture_files = SimpleNamespace(resolve=lambda materials: materials)
    manager._spawn_scene = scene
    return manager


def test_explicit_runtime_uid_reuses_only_equivalent_definitions():
    manager = _material_manager()
    cfg = VisualMaterialCfg(uid="shared")

    first = manager.create_visual_material(cfg)
    assert manager.create_visual_material(cfg.copy()) is first
    with pytest.raises(ValueError, match="Conflicting visual material"):
        manager.create_visual_material(cfg.replace(roughness=0.1))

    manager._env.create_pbr_material.assert_called_once()


def test_unnamed_runtime_materials_receive_distinct_ids_without_mutating_config():
    manager = _material_manager()
    cfg = VisualMaterialCfg()

    first = manager.create_visual_material(cfg)
    second = manager.create_visual_material(cfg)

    assert first.uid != second.uid
    assert cfg.uid is None


@pytest.mark.parametrize("runtime_first", [False, True])
def test_explicit_uid_conflicts_are_rejected_across_creation_paths(runtime_first):
    manager = _material_manager()
    runtime_cfg = VisualMaterialCfg(uid="shared", base_color=[1, 0, 0, 1])
    _, (uid, inline_desc) = _compile_visual_material(
        "object", VisualMaterialCfg(uid="shared", base_color=[0, 0, 1, 1])
    )
    scene = manager._spawn_scene
    if runtime_first:
        manager.create_visual_material(runtime_cfg)
        with pytest.raises(ValueError, match="Conflicting visual material"):
            scene._resolve_materials({uid: inline_desc})
    else:
        scene.builder.materials.update(scene._resolve_materials({uid: inline_desc}))
        with pytest.raises(ValueError, match="Conflicting visual material"):
            manager.create_visual_material(runtime_cfg)
        manager._env.create_pbr_material.assert_not_called()


def test_equivalent_inline_material_uses_configured_native_default_after_runtime_creation():
    from dexsim.spawn.adapters.common import _create_render_material

    manager = _material_manager()
    cfg = VisualMaterialCfg(uid="shared", base_color=[1, 0, 0, 1])
    runtime = manager.create_visual_material(cfg)
    manager._env.find_material.return_value = runtime.mat
    _, (_, descriptor) = _compile_visual_material("object", cfg)
    manager._spawn_scene._resolve_materials({"shared": descriptor})

    native = _create_render_material(
        SimpleNamespace(get_env=lambda: manager._env), descriptor
    )

    assert native is runtime.inst.mat
    assert list(native.get_base_color())[:3] == [1.0, 0.0, 0.0]


def test_equivalent_runtime_alias_preserves_native_state_from_inline_material():
    manager = _material_manager()
    cfg = VisualMaterialCfg(uid="shared", roughness=0.2)
    _, (_, descriptor) = _compile_visual_material("object", cfg)
    manager._spawn_scene.builder.materials["shared"] = descriptor
    native = _native_template()
    native.get_inst().set_roughness(0.8)
    native.get_inst().reset_mock()
    manager._env.find_material.return_value = native

    alias = manager.create_visual_material(cfg)

    assert alias.inst.roughness == pytest.approx(0.8)
    alias.inst.mat.set_roughness.assert_not_called()
    manager._env.create_pbr_material.assert_not_called()


def test_cleaning_materials_preserves_world_lifetime_definition_identity():
    manager = _material_manager()
    cfg = VisualMaterialCfg(uid="reserved", roughness=0.2)
    manager.create_visual_material(cfg)

    manager.clean_materials()

    assert manager.get_visual_material("reserved") is None
    manager._env.clean_materials.assert_called_once()
    with pytest.raises(ValueError, match="Conflicting visual material"):
        manager.create_visual_material(cfg.replace(roughness=0.9))
    assert manager.create_visual_material(cfg).cfg.roughness == 0.2


def test_implicit_names_allow_replacement_and_reuse_equal_material_definitions():
    cfg = VisualMaterialCfg(base_color=[1, 0, 0, 1])
    first_ref, first_entry = _compile_visual_material("object", cfg)
    second_ref, second_entry = _compile_visual_material(
        "object", cfg.replace(base_color=[0, 0, 1, 1])
    )
    repeated_ref, _ = _compile_visual_material("object", cfg.copy())
    scene = _material_manager()._spawn_scene
    scene.builder.materials.update(scene._resolve_materials(dict([first_entry])))
    scene.builder.materials.update(scene._resolve_materials(dict([second_entry])))

    assert first_ref != second_ref
    assert repeated_ref == first_ref
    assert len(scene.builder.materials) == 2


def test_deterministic_group_reset_reapplies_properties_only_to_selected_environments():
    material = VisualMaterial(
        VisualMaterialCfg(uid="fixed", roughness=0.2), _native_template()
    )
    group = RigidObjectGroup.__new__(RigidObjectGroup)
    group.uid = "group"
    group._entities = [[MagicMock()], [MagicMock()]]
    group._selected_indices = lambda env_ids: (env_ids, None, None)
    sim = SimpleNamespace(
        get_rigid_object_group_uid_list=lambda: ["group"],
        get_rigid_object_group=lambda uid: group,
        create_visual_material=lambda cfg: material,
    )
    env = SimpleNamespace(sim=sim, num_envs=2)
    set_rigid_object_group_visual_material(
        env, None, SceneEntityCfg(uid="group"), material.cfg
    )
    first = material.get_instance("fixed_group_0")
    second = material.get_instance("fixed_group_1")
    first.set_roughness(0.9)
    second.set_roughness(0.9)

    set_rigid_object_group_visual_material(
        env, torch.tensor([1]), SceneEntityCfg(uid="group"), material.cfg
    )

    assert first.roughness == pytest.approx(0.9)
    assert second.roughness == pytest.approx(0.2)


def test_preset_inheritance_and_yaml_roundtrip_keep_explicit_defaults():
    import yaml
    from embodichain.lab.sim.material import _material_desc_from_cfg
    from embodichain.lab.sim.shapes import ShapeCfg

    cfg = VisualMaterialCfg.from_preset("GlassClear", uid="glass")
    assert cfg.ior is None
    explicit = cfg.replace(ior=1.5, roughness=0.0)
    explicit.parameters = {"thin_walled": False, "transmission": 0.0}
    restored = VisualMaterialCfg.from_dict(
        yaml.safe_load(yaml.safe_dump(explicit.to_dict()))
    )
    assert restored.to_dict() == explicit.to_dict()
    shape = ShapeCfg.from_dict(
        yaml.safe_load(
            yaml.safe_dump(
                {
                    "shape_type": "Sphere",
                    "radius": 0.25,
                    "visual_material": explicit.to_dict(),
                }
            )
        )
    )
    assert shape.visual_material.to_dict() == explicit.to_dict()
    assert restored.copy().to_dict() == explicit.to_dict()
    assert restored.replace(ior=None).ior is None
    assert restored.replace(ior=None).roughness == 0.0
    inherited_desc = _material_desc_from_cfg(cfg, "glass")
    explicit_desc = _material_desc_from_cfg(explicit, "glass")
    assert inherited_desc != explicit_desc
    assert VisualMaterialCfg.from_dict({"preset": "GlassClear"}).ior is None
    assert VisualMaterialCfg().ior == 1.5
    assert VisualMaterialCfg().roughness == 0.7
    assert "material_type" not in VisualMaterialCfg().to_dict()
    assert (
        _material_desc_from_cfg(VisualMaterialCfg.from_dict(cfg.to_dict()), "glass")
        == inherited_desc
    )


@pytest.mark.parametrize(
    "properties",
    [
        {"ior": 1.5, "parameters": {"ior": 1.6}},
        {"emissive_intensity": 0.0, "parameters": {"emission_intensity": 1.0}},
        {"parameters": {"emission_intensity": 1.0, "emissive_intensity": 2.0}},
        {"parameters": {"thin_walled": 0}},
        {"parameters": {"volume_attenuation_distance": 0.0}},
        {"parameters": {"iridescence_thickness": [520, 250]}},
        {"parameters": {"transmission": float("nan")}},
        {"parameters": {"transmission": None}},
        {"parameters": {"misspelled": 1.0}},
        {"ior": 0.0},
        {"roughness": 1.1},
        {"base_color_texture": "override.png"},
    ],
)
def test_preset_invalid_overrides_fail_before_download(properties):
    with pytest.raises(ValueError):
        VisualMaterialCfg.from_preset("GlassClear", **properties)


@pytest.mark.parametrize("preset", [None, "GlassClear"])
def test_removed_scattering_type_is_rejected_without_silent_conversion(preset):
    with pytest.raises(TypeError, match="material_type"):
        VisualMaterialCfg(preset=preset, material_type="BSDF")
    for legacy in (None, "BRDF", "BTDF", "BSDF"):
        with pytest.raises(ValueError, match="material_type has been removed"):
            VisualMaterialCfg.from_dict({"preset": preset, "material_type": legacy})
    if preset is not None:
        with pytest.raises(TypeError, match="material_type"):
            VisualMaterialCfg.from_preset(preset, material_type="BSDF")


def test_manual_optical_parameters_load_without_an_explicit_scattering_type():
    import json
    from pathlib import Path
    from embodichain.lab.sim._material_presets import _MaterialResources
    from embodichain.lab.sim.material import _material_desc_from_cfg

    cfg = VisualMaterialCfg(
        uid="ice",
        base_color=[1, 1, 1, 1],
        ior=1.31,
        roughness=0.2,
        parameters={"transmission": 1.0},
    )
    _, (_, descriptor) = _compile_visual_material("ice_object", cfg)
    assert descriptor == _material_desc_from_cfg(cfg, "ice")
    world = MagicMock()
    template = _native_template()
    world.get_env.return_value.load_material.return_value = template
    resources = _MaterialResources(world, "rt")
    try:
        resources.resolve(descriptor)
        directory = Path(world.get_env.return_value.load_material.call_args.args[0])
        manifest = json.loads((directory / "material.json").read_text())
        assert manifest["parameters"]["transmission"] == 1.0
        assert manifest["parameters"]["ior"] == 1.31
        assert manifest["parameters"]["roughness"] == 0.2
        template.get_inst().update_pbr_material_type.assert_not_called()
        assert "material_type" not in cfg.to_dict()
    finally:
        resources.close()


def test_resource_wrapper_preserves_authored_template_and_resets_full_instance():
    template = _native_template()
    default = template.get_inst()
    default.set_pbr_param("ior", 1.52)
    child = template.get_inst("private_child")
    child.set_pbr_param("ior", 1.52)
    template.reset_mock()
    cfg = VisualMaterialCfg.from_preset("GlassClear", uid="glass")
    material = VisualMaterial(cfg, template, initialize_default=False, mat_inst=child)
    assert material.inst.mat is child
    child.set_roughness.assert_not_called()
    child.update_pbr_material_type.assert_not_called()
    assert default.get_ior() == pytest.approx(1.52)
    material.set_default_properties(material.inst, cfg)
    template.reset_inst.assert_called_once_with(child)
    with pytest.raises(ValueError, match="original definition"):
        material.set_default_properties(material.inst, cfg.replace(ior=1.6))


def test_preset_backend_preflight_and_no_render_declarations():
    from embodichain.lab.sim._material_presets import _MaterialResources
    from embodichain.lab.sim.material import _material_desc_from_cfg

    manager = _material_manager()
    scene = manager._spawn_scene
    world = MagicMock()
    scene._material_resources = _MaterialResources(world, "hybrid")
    scene._resource_material_definitions = {}
    glass = _material_desc_from_cfg(
        VisualMaterialCfg.from_preset("GlassClear"), "glass"
    )
    with pytest.raises(ValueError, match="OfflineRT"):
        scene._resolve_materials({"glass": glass})
    advanced = _material_desc_from_cfg(
        VisualMaterialCfg.from_preset("BrickRed", parameters={"clearcoat": 0.0}),
        "brick",
    )
    with pytest.raises(ValueError, match="OfflineRT"):
        scene._resolve_materials({"brick": advanced})
    world.get_env.assert_not_called()
    scene._material_resources.renderer = "no-render"
    assert scene._resolve_materials({"glass": glass}) == {}
    assert scene._material_definition("glass") == glass
    world.get_env.assert_not_called()
    scene._material_resources.close()


def test_native_manifest_variants_share_images_and_templates_without_mutation(
    tmp_path, monkeypatch
):
    import json
    from pathlib import Path
    from embodichain.lab.sim._material_presets import _MaterialResources
    from embodichain.lab.sim.material import _material_desc_from_cfg

    source = tmp_path / "Material/GlassClear"
    source.mkdir(parents=True)
    (source / "Color.png").write_bytes(b"shared original image")
    manifest = {
        "schema_version": 1,
        "parameters": {
            "ior": 1.52,
            "transmission": 1.0,
            "volume_attenuation_color": [0.995, 0.998, 1.0],
            "iridescence_thickness": [250, 520],
        },
        "textures": {"base_color": "Color.png"},
    }
    original = json.dumps(manifest)
    (source / "material.json").write_text(original)
    dataset = SimpleNamespace(get_material_path=lambda name: str(source))
    monkeypatch.setattr(
        "embodichain.data.assets.materials.DefaultMaterials", lambda: dataset
    )
    world = MagicMock()
    world.get_env.return_value.load_material.side_effect = (
        lambda *args: _native_template()
    )
    resources = _MaterialResources(world, "rt")
    cfg = VisualMaterialCfg.from_preset(
        "GlassClear",
        ior=1.5,
        parameters={"volume_attenuation_distance": 8.0, "thin_walled": False},
    )
    try:
        first = resources.resolve(_material_desc_from_cfg(cfg, "one"))
        second = resources.resolve(_material_desc_from_cfg(cfg, "two"))
        assert first[0] is second[0]
        assert first[1] is not second[1]
        assert resources.resolve(_material_desc_from_cfg(cfg, "one")) is first
        world.get_env.return_value.load_material.assert_called_once()
        directory = Path(world.get_env.return_value.load_material.call_args.args[0])
        staged = json.loads((directory / "material.json").read_text())
        assert staged["parameters"] == {
            **manifest["parameters"],
            "ior": 1.5,
            "volume_attenuation_distance": 8.0,
            "thin_walled": False,
        }
        assert Path(staged["textures"]["base_color"]) == source / "Color.png"
        assert list(directory.iterdir()) == [directory / "material.json"]
        assert (source / "material.json").read_text() == original
    finally:
        resources.close()
    assert not directory.exists()


def test_common_parameter_aliases_canonicalize_before_replace_restores_inheritance():
    from embodichain.lab.sim.material import _material_desc_from_cfg

    cfg = VisualMaterialCfg.from_dict(
        {
            "preset": "GlassClear",
            "uid": "glass",
            "parameters": {"ior": 1.5, "emission_intensity": 0.0, "thin_walled": False},
        }
    )
    expected = VisualMaterialCfg.from_preset(
        "GlassClear",
        uid="glass",
        ior=1.5,
        emissive_intensity=0.0,
        parameters={"thin_walled": False},
    )
    assert cfg.to_dict() == expected.to_dict()
    inherited = cfg.replace(ior=None, emissive_intensity=None)
    assert inherited.ior is None and inherited.emissive_intensity is None
    assert inherited.parameters == {"thin_walled": False}
    assert _material_desc_from_cfg(inherited, "glass") == _material_desc_from_cfg(
        VisualMaterialCfg.from_preset("GlassClear", parameters={"thin_walled": False}),
        "glass",
    )


def test_builtin_presets_resolve_without_resource_download_and_keep_overrides(
    monkeypatch,
):
    import json
    from pathlib import Path
    from embodichain.data import get_default_material_info
    from embodichain.lab.sim._material_presets import _MaterialResources
    from embodichain.lab.sim.material import _material_desc_from_cfg

    def forbidden(*args, **kwargs):
        raise AssertionError("Built-in presets must not construct DefaultMaterials")

    monkeypatch.setattr("embodichain.data.assets.materials.DefaultMaterials", forbidden)
    world = MagicMock()
    world.get_env.return_value.load_material.side_effect = (
        lambda *args: _native_template()
    )
    resources = _MaterialResources(world, "rt")
    cfg = VisualMaterialCfg.from_preset("PlasticMatteBlue", roughness=0.0)
    try:
        resources.resolve(_material_desc_from_cfg(cfg, "blue"))
        path = (
            Path(world.get_env.return_value.load_material.call_args.args[0])
            / "material.json"
        )
        manifest = json.loads(path.read_text())
        assert manifest["parameters"]["base_color"] == [0.05, 0.22, 0.6, 1]
        assert manifest["parameters"]["roughness"] == 0.0
        assert manifest["parameters"]["transmission"] == 0.0
        assert manifest["textures"] == {}
        assert (
            get_default_material_info("PlasticMatteBlue")["parameters"]["roughness"]
            == 0.65
        )
        resources.resolve(
            _material_desc_from_cfg(cfg.replace(roughness=None), "inherited")
        )
        path = (
            Path(world.get_env.return_value.load_material.call_args.args[0])
            / "material.json"
        )
        assert json.loads(path.read_text())["parameters"]["roughness"] == 0.65
        assert resources._dataset is None
    finally:
        resources.close()
