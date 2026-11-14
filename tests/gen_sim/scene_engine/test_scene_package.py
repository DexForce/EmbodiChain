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

import hashlib
from importlib.resources import files
import json
from pathlib import Path
import shutil

import dexsim
from dexsim.kit.simready import open_asset_package, validate_asset_revision
from jsonschema import Draft202012Validator, FormatChecker
from pxr import Sdf, Usd, UsdGeom, UsdPhysics
import pytest

from embodichain.gen_sim.scene_engine.pipeline.utils.scene_package import (
    _export_scene_package,
)
from embodichain.gen_sim.scene_engine.pipeline.utils.usd_scene import (
    USD_SCENE_SCHEMA,
    UsdSceneIndex,
)


def test_scene_package_schema_bytes_and_relocated_joint_targets(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    layer = Usd.Stage.CreateNew(str(source / "bodies.usda"))
    world = UsdGeom.Xform.Define(layer, "/World").GetPrim()
    world.SetCustomDataByKey("embodichain:scene_schema", USD_SCENE_SCHEMA)
    layer.SetDefaultPrim(world)
    UsdGeom.SetStageUpAxis(layer, "Z")
    UsdGeom.SetStageMetersPerUnit(layer, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(layer, 1.0)
    for name in ("parent", "child"):
        body = UsdGeom.Cube.Define(layer, f"/World/{name}").GetPrim()
        body.SetCustomDataByKey("embodichain:uid", name)
        body.SetCustomDataByKey("embodichain:kind", "rigid")
        UsdPhysics.RigidBodyAPI.Apply(body)
        UsdPhysics.CollisionAPI.Apply(body)
        UsdPhysics.MassAPI.Apply(body).CreateMassAttr(0.25)
    joint = UsdPhysics.FixedJoint.Define(layer, "/World/joint")
    joint.CreateBody0Rel().SetTargets(["/World/parent"])
    joint.CreateBody1Rel().SetTargets(["/World/child"])
    layer.GetRootLayer().Save()
    stage = Usd.Stage.CreateNew(str(source / "scene.usda"))
    stage.GetRootLayer().subLayerPaths = ["bodies.usda"]
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    stage.GetRootLayer().Save()
    # Unreferenced compatibility assets and old snapshots are not deliverables.
    (source / "stale.gltf").write_text("obsolete")
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    destination = tmp_path / "scene_package.zip"
    destination.write_bytes(b"old scene")

    _export_scene_package(scene_usd_path=source / "scene.usda", destination=destination)

    assert {p.name: p.read_bytes() for p in source.iterdir()} == before
    with open_asset_package(destination, workspace=tmp_path) as extracted:
        relocated = tmp_path / "relocated package"
        shutil.copytree(extracted, relocated)
    parsed = validate_asset_revision(relocated)
    assert parsed is not None
    asset = json.loads((relocated / "asset.json").read_text())
    schema = json.loads(
        files("dexsim.kit.simready.schemas").joinpath("asset.schema.json").read_text()
    )
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(asset)
    assert asset["schema_version"] == "2.0-draft"
    assert asset["asset_kind"] is None  # A scene is not one rigidbody/articulation.
    assert asset["capabilities"] == []
    assert asset["certification"] is None
    assert asset["quality_report"]["summary"]["status"] == "incomplete"
    assert asset["quality_report"]["runtime_tests"] == []
    assert not (relocated / "stale.gltf").exists()
    for representation in asset["representations"]:
        content = (relocated / representation["relative_path"]).read_bytes()
        assert (
            representation["digest"] == "sha256:" + hashlib.sha256(content).hexdigest()
        )
        assert representation["size_bytes"] == len(content)
    packaged = Usd.Stage.Open(str(relocated / "asset.usda"))
    index = UsdSceneIndex.load(relocated / "asset.usda", require_schema=True)
    assert index.get("child").prim_path == "/Asset/World/child"
    assert str(packaged.GetDefaultPrim().GetPath()) == "/Asset"
    assert [str(p.GetPath()) for p in packaged.GetPseudoRoot().GetChildren()] == [
        "/Asset"
    ]
    joint = UsdPhysics.FixedJoint(packaged.GetPrimAtPath("/Asset/World/joint"))
    assert joint.GetBody0Rel().GetTargets() == [Sdf.Path("/Asset/World/parent")]
    assert joint.GetBody1Rel().GetTargets() == [Sdf.Path("/Asset/World/child")]
    assert UsdPhysics.MassAPI(
        packaged.GetPrimAtPath("/Asset/World/child")
    ).GetMassAttr().Get() == pytest.approx(0.25)
    # JSON structure alone cannot detect altered artifact bytes.
    with (relocated / "bodies.usda").open("a") as stream:
        stream.write("\n# changed bytes\n")
    with pytest.raises(ValueError, match="inventory or digest mismatch"):
        validate_asset_revision(relocated)


def test_scene_package_removes_stale_delivery_on_dependency_failure(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    stage = Usd.Stage.CreateNew(str(source / "scene.usda"))
    UsdGeom.Xform.Define(stage, "/World")
    stage.GetRootLayer().subLayerPaths = ["missing.usda"]
    stage.GetRootLayer().Save()
    destination = tmp_path / "scene_package.zip"
    destination.write_bytes(b"stale scene")

    with pytest.raises(ValueError, match="unresolved dependencies"):
        _export_scene_package(
            scene_usd_path=source / "scene.usda", destination=destination
        )

    assert not destination.exists()
