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
from zipfile import ZipFile

import dexsim
from dexsim.kit.simready import open_asset_package, validate_asset_revision
from jsonschema import Draft202012Validator, FormatChecker
from PIL import Image
from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
import pytest
import trimesh

from embodichain.gen_sim.scene_engine.core.scene_graph import SceneGraph, SceneGraphNode
from embodichain.gen_sim.scene_engine.pipeline.utils.scene_exporter import (
    SceneExporter,
)
from embodichain.gen_sim.scene_engine.pipeline.utils.scene_importer import (
    SceneExportImporter,
)
from embodichain.gen_sim.scene_engine.pipeline.utils.scene_package import (
    _export_scene_package,
    _restore_scene_export,
)
from embodichain.gen_sim.scene_engine.pipeline.utils.scene_usd import (
    _prepare_runtime_assets,
)
from embodichain.gen_sim.scene_engine.pipeline.utils.usd_scene import (
    USD_SCENE_SCHEMA,
    UsdSceneIndex,
)
from .test_randomization_baseline import _export


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


def test_editable_scene_package_retains_both_articulation_models(
    tmp_path: Path,
) -> None:
    export = tmp_path / "scene_export"
    table_glb = Path("mesh_assets/table/table.glb")
    laptop_glb = Path("mesh_assets/laptop/laptop.glb")
    for path in (table_glb, laptop_glb):
        (export / path).parent.mkdir(parents=True)
        trimesh.creation.box().export(export / path)

    articulation_dir = Path("articulated_assets/laptop")
    (export / articulation_dir).mkdir(parents=True)
    runtime = articulation_dir / "laptop.usdc"
    original = articulation_dir / "laptop.source.usdc"
    runtime_dependency = articulation_dir / "runtime_geometry.usda"
    original_dependency = articulation_dir / "source_geometry.usda"
    texture = articulation_dir / "albedo.png"
    Image.new("RGB", (2, 2), "red").save(export / texture)
    for model, dependency, bottom in (
        (runtime, runtime_dependency, -0.25),
        (original, original_dependency, -1.0),
    ):
        geometry = Usd.Stage.CreateNew(str(export / dependency))
        body = UsdGeom.Xform.Define(geometry, "/Body").GetPrim()
        UsdPhysics.ArticulationRootAPI.Apply(body)
        UsdPhysics.RigidBodyAPI.Apply(body)
        collider = UsdGeom.Mesh.Define(geometry, "/Body/collision")
        collider.CreatePointsAttr([(0, 0, bottom), (1, 0, 0), (0, 1, 0), (0, 0, 1)])
        collider.CreateFaceVertexCountsAttr([3, 3, 3, 3])
        collider.CreateFaceVertexIndicesAttr([0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3])
        UsdPhysics.CollisionAPI.Apply(collider.GetPrim())
        shader = UsdShade.Shader.Define(geometry, "/Body/Albedo")
        shader.CreateIdAttr("UsdUVTexture")
        shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
            Sdf.AssetPath(texture.name)
        )
        geometry.GetRootLayer().Save()
        model_stage = Usd.Stage.CreateNew(str(export / model))
        model_stage.GetRootLayer().subLayerPaths = [dependency.name]
        model_stage.SetDefaultPrim(model_stage.GetPrimAtPath("/Body"))
        UsdGeom.SetStageUpAxis(model_stage, "Z")
        UsdGeom.SetStageMetersPerUnit(model_stage, 1.0)
        model_stage.GetRootLayer().Save()

    config = {
        "format": "embodichain.scene-export/v1",
        "scene_id": "laptop_scene",
        "background": [
            {
                "uid": "table",
                "is_articulated": False,
                "shape": {"shape_type": "Mesh", "fpath": table_glb.as_posix()},
            }
        ],
        "articulation": [
            {
                "uid": "laptop",
                "is_articulated": True,
                "fpath": runtime.as_posix(),
                "source_usdc_fpath": original.as_posix(),
                "proxy_glb_fpath": laptop_glb.as_posix(),
                "init_pos": [0, 0, 0],
                "init_rot": [0, 0, 0],
                "body_scale": [1.0, 1.0, 1.0],
                "proxy_body_scale": [1.0, 1.0, 1.0],
            }
        ],
    }
    (export / "scene_config.json").write_text(json.dumps(config), encoding="utf-8")
    graph = SceneGraph(
        nodes=[SceneGraphNode("table", None), SceneGraphNode("laptop", "table", "on")],
        relations=[],
    )
    (export / "scene_graph.json").write_text(
        json.dumps(graph.to_dict()), encoding="utf-8"
    )

    scene_usd = tmp_path / "scene.usda"
    stage = Usd.Stage.CreateNew(str(scene_usd))
    world = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    world.SetCustomDataByKey("embodichain:scene_schema", USD_SCENE_SCHEMA)
    stage.SetDefaultPrim(world)
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    for uid, kind in (("table", "rigid"), ("laptop", "articulation")):
        prim = UsdGeom.Cube.Define(stage, f"/World/{uid}").GetPrim()
        prim.SetCustomDataByKey("embodichain:uid", uid)
        prim.SetCustomDataByKey("embodichain:kind", kind)
    stage.GetRootLayer().Save()

    destination = tmp_path / "scene_package.zip"
    _export_scene_package(
        scene_usd_path=scene_usd, destination=destination, scene_export_root=export
    )

    with open_asset_package(destination, workspace=tmp_path) as root:
        for path in (runtime, original, runtime_dependency, original_dependency):
            assert (root / "scene_export" / path).read_bytes() == (
                export / path
            ).read_bytes()
        asset = json.loads((root / "asset.json").read_text(encoding="utf-8"))
        laptop = asset["extensions"]["embodichain:scene_export"]["instances"]["laptop"]
        representations = {
            rep["representation_id"]: rep for rep in asset["representations"]
        }
        assert (
            representations[laptop["runtime_representation_id"]]["relative_path"]
            == f"scene_export/{runtime.as_posix()}"
        )

    reopened = tmp_path / "reopened"
    reopened.mkdir()
    shutil.copyfile(destination, reopened / destination.name)
    scene, graph = SceneExportImporter(output_root=reopened).import_scene_and_graph(
        write_scene_json=False
    )
    refreshed = SceneExporter(scene=scene, scene_graph=graph, output_root=reopened)
    refreshed_config = json.loads(refreshed.export().read_text(encoding="utf-8"))
    (laptop,) = refreshed_config["articulation"]
    # The table top is 0.5 m, the original collision bottom is -1 m,
    # and the legacy positioning leaves 1 mm of clearance.
    assert laptop["init_pos"][2] == pytest.approx(1.501)
    assert laptop["source_usdc_fpath"] == original.as_posix()
    runtime_assets = _prepare_runtime_assets(
        scene_config=refreshed_config,
        config_dir=reopened / "scene_export",
        runtime_assets_root=reopened / "runtime",
    )
    staged = Usd.Stage.Open(str(runtime_assets["laptop"]))
    assert staged.GetDefaultPrim().GetPath() == Sdf.Path("/Body")
    collision = UsdGeom.Mesh.Get(staged, "/Body/collision")
    assert collision.GetPrim().HasAPI(UsdPhysics.CollisionAPI)
    assert min(point[2] for point in collision.GetPointsAttr().Get()) == -1.0
    albedo = UsdShade.Shader.Get(staged, "/Body/Albedo").GetInput("file").Get()
    assert Path(albedo.resolvedPath).is_relative_to(runtime_assets["laptop"].parent)
    assert Path(albedo.resolvedPath).read_bytes() == (export / texture).read_bytes()
    for saved, expected in (
        (runtime, original),
        (original, original),
        (original_dependency, original_dependency),
        (texture, texture),
    ):
        assert (reopened / "scene_export" / saved).read_bytes() == (
            export / expected
        ).read_bytes()


def test_scene_package_projects_floor_support_and_restores_legacy_metadata(
    tmp_path: Path,
) -> None:
    export, _ = _export(tmp_path, floor_subtree=True)
    scene_usd = tmp_path / "scene.usda"
    stage = Usd.Stage.CreateNew(str(scene_usd))
    world = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    world.SetCustomDataByKey("embodichain:scene_schema", USD_SCENE_SCHEMA)
    stage.SetDefaultPrim(world)
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    for uid in ("table", "tray", "cup", "block", "bottle"):
        prim = UsdGeom.Cube.Define(stage, f"/World/{uid}").GetPrim()
        prim.SetCustomDataByKey("embodichain:uid", uid)
        prim.SetCustomDataByKey("embodichain:kind", "rigid")
    stage.GetRootLayer().Save()

    current = tmp_path / "current"
    current.mkdir()
    package = current / "scene_package.zip"
    _export_scene_package(
        scene_usd_path=scene_usd, destination=package, scene_export_root=export
    )
    with ZipFile(package) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    asset = json.loads(members["asset.json"])
    relations = asset["semantics"]["relations"]
    assert {
        (row["subject"]["instance_id"], row["relation"], row["object"]["instance_id"])
        for row in relations
    } == {("cup", "on", "tray"), ("block", "on", "cup"), ("bottle", "on", "table")}
    attributes = asset["semantics"]["attributes"]
    assert [
        (row["subject"]["instance_id"], row["value"])
        for row in attributes
        if row["name"] == "placement_support"
    ] == [("tray", "floor")]

    # Earlier packages published the structural table-parent edge and no floor
    # attribute. Reconstruct that on-disk metadata without calling its projector.
    asset["semantics"]["attributes"] = [
        row for row in attributes if row["name"] != "placement_support"
    ]
    legacy_edge = {
        "relation_id": "relation-"
        + hashlib.sha256(b'["tray","on","table"]').hexdigest(),
        "subject": {"kind": "instance", "instance_id": "tray"},
        "object": {"kind": "instance", "instance_id": "table"},
        "relation": "on",
        "conditions": [],
        "namespace": "embodichain",
        "taxonomy": "scene-graph",
        "taxonomy_version": "1",
        "origin": "unknown",
        "evidence_refs": [],
    }
    relations.insert(0, legacy_edge)
    legacy = tmp_path / "legacy"
    tampered = tmp_path / "tampered"
    for workspace in (legacy, tampered):
        workspace.mkdir()
        if workspace == tampered:
            asset["semantics"]["labels"][0]["value"] = "unrelated category"
        members["asset.json"] = json.dumps(asset).encode("utf-8")
        with ZipFile(workspace / "scene_package.zip", "w") as archive:
            for name, content in members.items():
                archive.writestr(name, content)

    for workspace in (current, legacy):
        _restore_scene_export(workspace)
        for filename in ("scene_config.json", "scene_graph.json"):
            assert (workspace / "scene_export" / filename).read_bytes() == (
                export / filename
            ).read_bytes()
    with pytest.raises(ValueError, match="metadata disagrees with its editable export"):
        _restore_scene_export(tampered)
    assert not (tampered / "scene_export").exists()


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
