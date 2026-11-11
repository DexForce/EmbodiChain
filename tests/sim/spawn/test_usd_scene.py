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

from copy import deepcopy
from pathlib import Path
import zipfile

import numpy as np
import pytest
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils

from embodichain.lab.sim.cfg import UsdImportCfg, UsdSourceCfg
from embodichain.lab.sim.spawn._usd_scene import (
    _assert_usd_scene_current,
    _inspect_usd_scene,
    _open_usd_scene_source,
)


def _registry() -> dict:
    """A fixture-only inventory declaration, without runtime certification."""
    rule = "TEST-INVENTORY"
    return {
        "schema_version": "2.0-draft",
        "registry_id": "test-inventory",
        "version": "1",
        "rules": [
            {
                "rule_id": rule,
                "version": "1",
                "domain": "ci",
                "title": "Test inventory",
                "requirement": "Preserve fixture bytes",
                "capability_id": "inventory",
                "source": {
                    "namespace": "embodichain",
                    "document_uri": "urn:embodichain:test:inventory",
                    "version": "1",
                    "retrieved_at": "2026-11-11T00:00:00Z",
                    "document_digest": None,
                    "requirement_id": rule,
                    "mapping": "none",
                },
                "usd_binding": None,
                "applicability": {
                    "asset_kinds": [],
                    "formats": ["usd", "usda", "usdc"],
                    "required_declarations": [],
                    "subject_scope": "asset",
                    "condition_description": "Test fixture",
                },
                "checker": {
                    "id": "test-inventory",
                    "version": "1",
                    "method": "static",
                    "parameters_schema": {"type": "object"},
                    "implementation": "partial",
                },
                "acceptance": "Inventory only",
                "default_severity": "error",
                "evidence_kinds": ["log"],
            }
        ],
        "capabilities": [
            {
                "capability_id": "inventory",
                "version": "1",
                "domain": "ci",
                "requirement_ids": [rule],
            }
        ],
        "features": [
            {
                "feature_id": "inventory",
                "version": "1",
                "required_rules": [{"rule_id": rule, "version": "1"}],
                "runtime_test_ids": [],
                "dependencies": [],
            }
        ],
        "profiles": [
            {
                "profile_id": "inventory",
                "version": "1",
                "namespace": "embodichain",
                "source": None,
                "features": [{"feature_id": "inventory", "version": "1"}],
                "required_runtime_tests": [],
                "asset_kinds": [],
                "target_runtimes": [],
                "policy": {
                    "policy_version": "1",
                    "blocking_rule_ids": [rule],
                    "allow_manual_evidence": False,
                    "required_evaluation_methods": ["static"],
                    "parameters": {},
                },
            }
        ],
    }


@pytest.fixture
def scene_files(tmp_path: Path) -> dict[str, Path]:
    from PIL import Image
    from dexsim.kit.simready._usd_package import export_processed_usd

    source = tmp_path / "source"
    source.mkdir()
    path = source / "scene.usda"
    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1)
    physics = UsdPhysics.Scene.Define(stage, "/World/physics")
    physics.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    physics.CreateGravityMagnitudeAttr(9.81)
    frame = UsdGeom.Xform.Define(stage, "/World/frame")
    frame.AddTranslateOp().Set(Gf.Vec3d(1, 2, 0))
    frame.AddRotateZOp().Set(90)
    body = UsdGeom.Cube.Define(stage, "/World/frame/box")
    body.CreateSizeAttr(0.1)
    body.AddTranslateOp().Set(Gf.Vec3d(0.3, 0.4, 0.5))
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    UsdPhysics.CollisionAPI.Apply(body.GetPrim())
    UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr(0.25)
    Image.new("RGB", (2, 2), (20, 80, 160)).save(source / "texture.png")
    material = UsdShade.Material.Define(stage, "/World/material")
    shader = UsdShade.Shader.Define(stage, "/World/material/surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    texture = UsdShade.Shader.Define(stage, "/World/material/texture")
    texture.CreateIdAttr("UsdUVTexture")
    texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("texture.png")
    )
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        texture.ConnectableAPI(), "rgb"
    )
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(body.GetPrim()).Bind(material)

    articulation = UsdGeom.Xform.Define(stage, "/World/drawer")
    UsdPhysics.ArticulationRootAPI.Apply(articulation.GetPrim())
    for name, pos in (("base", (0, 0, 0.1)), ("slide", (0.2, 0, 0.1))):
        link = UsdGeom.Xform.Define(stage, f"/World/drawer/{name}")
        link.AddTranslateOp().Set(Gf.Vec3d(*pos))
        UsdPhysics.RigidBodyAPI.Apply(link.GetPrim())
        UsdPhysics.MassAPI.Apply(link.GetPrim()).CreateMassAttr(0.1)
        shape = UsdGeom.Cube.Define(stage, f"/World/drawer/{name}/shape")
        shape.CreateSizeAttr(0.1)
        UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
    anchor = UsdPhysics.FixedJoint.Define(stage, "/World/drawer/anchor")
    anchor.CreateBody1Rel().SetTargets(["/World/drawer/base"])
    joint = UsdPhysics.PrismaticJoint.Define(stage, "/World/drawer/travel")
    joint.CreateBody0Rel().SetTargets(["/World/drawer/base"])
    joint.CreateBody1Rel().SetTargets(["/World/drawer/slide"])
    joint.CreateAxisAttr("X")
    joint.CreateLowerLimitAttr(-0.02)
    joint.CreateUpperLimitAttr(0.05)
    stage.GetRootLayer().Save()
    archive = tmp_path / "scene.usdz"
    assert UsdUtils.CreateNewUsdzPackage(Sdf.AssetPath(str(path)), str(archive))
    package = tmp_path / "scene.zip"
    export_processed_usd(
        path,
        package,
        source_root=source,
        catalog={
            "name": "Import test",
            "contributor": "EmbodiChain tests",
            "contributor_origin": "declared",
            "annotation": "Fixture, no runtime certification",
            "version_number": 0,
            "version": "v0",
            "created_at": "2026-11-11T00:00:00Z",
            "published_at": None,
            "permission": "private",
            "tags": ["test"],
        },
        registry=_registry(),
    )
    return {"usd": path, "usdz": archive, "zip": package}


def _recipe(path: Path) -> UsdImportCfg:
    prefix = "/Asset" if path.suffix == ".zip" else ""
    return UsdImportCfg(
        name="environment",
        source=UsdSourceCfg(file_path=str(path)),
        asset_cfgs={
            prefix + "/World/frame/box": {"uid": "target"},
            prefix + "/World/drawer": {"uid": "drawer"},
        },
    )


@pytest.mark.parametrize("kind", ["usd", "usdz", "zip"])
def test_inventory_matches_runtime_compiler_without_world(scene_files, kind):
    inventory = _inspect_usd_scene(_recipe(scene_files[kind]))
    assert set(inventory["resource_uids"].values()) == {"target", "drawer"}
    assert set(inventory["resource_types"].values()) == {"rigid_object", "articulation"}
    assert list(inventory["joints"].values()) == [["travel"]]
    from embodichain.lab.sim.spawn.usd_import import _compile_usd_import

    with _open_usd_scene_source(_recipe(scene_files[kind])) as recipe:
        plan = _compile_usd_import(None, recipe)
        box = next(item for item in plan.source.mesh_objects if item.name == "target")
        assert box.pose[:3, 3] == pytest.approx([0.6, 2.3, 0.5])
        assert box.physics.mass == pytest.approx(0.25)
        assert plan.source.articulations[0].joints[0].lower_limit == pytest.approx(
            -0.02
        )
        assert plan.source.articulations[0].joints[0].upper_limit == pytest.approx(0.05)
        assert any(
            material.base_color_map for material in plan.source.materials.values()
        )


def test_inventory_rejects_missing_target_uid_conflict_and_gravity(scene_files):
    recipe = _recipe(scene_files["usd"])
    with pytest.raises(ValueError, match="already exists"):
        _inspect_usd_scene(recipe, reserved_uids=("target",))
    invalid = deepcopy(recipe)
    invalid.asset_cfgs["/World/missing"] = {"uid": "missing"}
    with pytest.raises(ValueError, match="missing"):
        _inspect_usd_scene(invalid)
    with pytest.raises(ValueError, match="gravity.*conflicts"):
        _inspect_usd_scene(recipe, gravity=(0, 0, -1))


def test_source_changes_and_corrupt_package_are_rejected(scene_files, tmp_path):
    recipe = _recipe(scene_files["usd"])
    inventory = _inspect_usd_scene(recipe)
    recipe_path = scene_files["usd"]
    with recipe_path.open("a") as stream:
        stream.write("\n# changed\n")
    with pytest.raises(ValueError, match="changed after preflight"):
        _assert_usd_scene_current(inventory)
    broken = tmp_path / "broken.zip"
    with (
        zipfile.ZipFile(scene_files["zip"]) as archive,
        zipfile.ZipFile(broken, "w") as output,
    ):
        for item in archive.infolist():
            content = archive.read(item.filename)
            output.writestr(
                item,
                content + b"\n# changed" if item.filename == "asset.usda" else content,
            )
    with pytest.raises(ValueError, match="digest|inventory"):
        _inspect_usd_scene(_recipe(broken))


def test_plain_zip_is_not_a_standard_scene(tmp_path):
    path = tmp_path / "vendor.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("scene.usda", "#usda 1.0")
    with pytest.raises(ValueError, match="standard AssetRevision"):
        with _open_usd_scene_source(_recipe(path)):
            pass


def test_component_paths_binding_types_and_default_ground(
    scene_files, tmp_path, monkeypatch
):
    import yaml
    from embodichain.lab.gym.utils.composition import resolve_gym_components
    from embodichain.lab.gym.utils.gym_utils import config_to_cfg

    owner = tmp_path / "component"
    owner.mkdir()
    environment = {
        "environment_id": "usd_scene",
        "physics": "default",
        "simulation": {"usd_scene": _recipe(scene_files["usd"]).to_dict()},
        "env": {},
    }
    environment["simulation"]["usd_scene"]["source"][
        "file_path"
    ] = "../source/scene.usda"
    (owner / "env.yaml").write_text(yaml.safe_dump(environment))
    shadow = tmp_path / "shadow"
    shadow.mkdir()
    monkeypatch.chdir(shadow)
    config = {
        "id": "EmbodiedEnv-v1",
        "environment": {"component": "component/env.yaml"},
        "robot": {"uid": "robot"},
    }
    resolved = resolve_gym_components(config, base_dir=tmp_path)
    assert resolved.usd_scene_inventory["source"] == str(scene_files["usd"])
    resolved.validate_scene_bindings(
        {
            "rigid_objects": [{"entity_id": "box", "simulation_uid": "target"}],
            "articulations": [{"entity_id": "cabinet", "simulation_uid": "drawer"}],
        }
    )
    with pytest.raises(ValueError, match="no native link"):
        resolved.validate_scene_bindings(
            {
                "articulations": [{"entity_id": "cabinet", "simulation_uid": "drawer"}],
                "links": [
                    {
                        "entity_id": "panel",
                        "articulation_id": "cabinet",
                        "native_link_name": "missing",
                    }
                ],
            }
        )
    with pytest.raises(ValueError, match="not declared"):
        resolved.validate_scene_bindings(
            {"rigid_objects": [{"entity_id": "cabinet", "simulation_uid": "drawer"}]}
        )
    cfg = config_to_cfg(config, source_path=tmp_path / "task.yaml")
    assert cfg.usd_scene.source.file_path == str(scene_files["usd"])
    assert cfg.sim_cfg.default_ground is False
    assert cfg.background == [] and cfg.rigid_object == [] and cfg.articulation == []
    config["default_ground"] = True
    assert (
        config_to_cfg(config, source_path=tmp_path / "task.yaml").sim_cfg.default_ground
        is True
    )
    del environment["simulation"]["usd_scene"]["asset_cfgs"]["/World/frame/box"]
    (owner / "env.yaml").write_text(yaml.safe_dump(environment))
    unnamed = resolve_gym_components(config, base_dir=tmp_path)
    with pytest.raises(ValueError, match="explicit asset_cfgs UID"):
        unnamed.validate_scene_bindings({"rigid_objects": [{"entity_id": "box"}]})


@pytest.mark.parametrize("up,length,magnitude", [("Z", 1.0, None), ("Y", 0.01, 981.0)])
def test_gravity_fallback_and_robot_binding_policy(scene_files, up, length, magnitude):
    recipe = _recipe(scene_files["usd"])
    stage = Usd.Stage.Open(str(scene_files["usd"]))
    scene = UsdPhysics.Scene(stage.GetPrimAtPath("/World/physics"))
    scene.GetGravityDirectionAttr().Set(Gf.Vec3f(0, 0, 0))
    UsdGeom.SetStageUpAxis(stage, up)
    UsdGeom.SetStageMetersPerUnit(stage, length)
    if magnitude is None:
        scene.GetGravityMagnitudeAttr().Clear()
    else:
        scene.GetGravityMagnitudeAttr().Set(magnitude)
    stage.GetRootLayer().Save()
    _inspect_usd_scene(recipe)
    from embodichain.lab.sim.cfg import RobotCfg

    del recipe.asset_cfgs["/World/drawer"]
    recipe.robot_cfgs = {"/World/drawer": RobotCfg(uid="robot")}
    with pytest.raises(ValueError, match="robots must come from embodiment"):
        _inspect_usd_scene(recipe)


def test_changed_source_rejects_environment_before_world(scene_files, monkeypatch):
    from unittest.mock import Mock
    from embodichain.lab.gym.envs.embodied_env import EmbodiedEnv
    from embodichain.lab.gym.utils.gym_utils import config_to_cfg

    cfg = config_to_cfg(
        {
            "id": "EmbodiedEnv-v1",
            "physics": "default",
            "env": {},
            "robot": {"uid": "robot"},
            "usd_scene": _recipe(scene_files["usd"]).to_dict(),
        }
    )
    with scene_files["usd"].open("a") as stream:
        stream.write("\n# changed after configuration\n")
    factory = Mock(side_effect=AssertionError("World must not be allocated"))
    monkeypatch.setattr("embodichain.lab.gym.envs.base_env.SimulationManager", factory)
    with pytest.raises(ValueError, match="changed after preflight"):
        EmbodiedEnv(cfg)
    factory.assert_not_called()


@pytest.mark.requires_sim
@pytest.mark.gpu
@pytest.mark.parametrize("kind", ["usd", "usdz", "zip"])
def test_native_import_reset_and_package_lifetime(scene_files, kind):
    from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
    from embodichain.lab.sim.cfg import RenderCfg

    sim = SimulationManager(
        SimulationManagerCfg(
            headless=True,
            default_ground=False,
            render_cfg=RenderCfg(renderer="no-render"),
            num_envs=2,
        )
    )
    imported = target = drawer = None
    resolved_path = None
    try:
        recipe = _recipe(scene_files[kind])
        target_uid = "default_plane" if kind == "zip" else "target"
        if kind == "zip":
            recipe.asset_cfgs["/Asset/World/frame/box"]["uid"] = target_uid
        imported = sim.add_usd(recipe)
        sim.prepare()
        target = sim.get_rigid_object(target_uid)
        drawer = sim.get_articulation("drawer")
        assert set(imported.resource_types.values()) == {"rigid_object", "articulation"}
        initial = target.get_local_pose().clone()
        assert initial[:, :3].cpu().numpy() == pytest.approx(
            np.array([[0.6, 2.3, 0.5]] * 2)
        )
        assert target.get_mass().cpu().tolist() == pytest.approx([0.25, 0.25])
        assert drawer.get_qpos_limits()[0].cpu().numpy().reshape(-1) == pytest.approx(
            [-0.02, 0.05]
        )
        resolved_path = Path(target.cfg.source.file_path)
        assert resolved_path.is_file()
        sim.update(step=5)
        assert bool((target.get_local_pose()[:, 2] < initial[:, 2]).all())
        fallen = target.get_local_pose().clone()
        sim.reset_objects_state(env_ids=[0])
        assert np.allclose(
            target.get_local_pose()[0].cpu().numpy(),
            initial[0].cpu().numpy(),
            atol=1e-5,
        )
        assert np.allclose(
            target.get_local_pose()[1].cpu().numpy(), fallen[1].cpu().numpy(), atol=1e-5
        )
        sim.reset_objects_state()
        assert np.allclose(
            target.get_local_pose().cpu().numpy(), initial.cpu().numpy(), atol=1e-5
        )
        assert set(sim.asset_uids) == set(imported.resource_uids.values())
        if kind == "zip":
            assert resolved_path != scene_files[kind]
            assert (resolved_path.parent / "texture.png").is_file()
            sim.remove_asset(target_uid)
            assert target_uid not in sim.asset_uids
    finally:
        imported = target = drawer = None
        sim.destroy(exit_process=False)
        sim = None
        SimulationManager.flush_cleanup_queue()
    if kind == "zip":
        assert not resolved_path.exists()
