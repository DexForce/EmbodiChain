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

import json
from pathlib import Path

import numpy as np
import pytest

from embodichain.gen_sim.scene_engine.core.scene import Scene
from embodichain.gen_sim.scene_engine.core.scene_graph import (
    SceneGraph,
    SceneGraphNode,
)
from embodichain.gen_sim.scene_engine.core.scene_object import (
    ObjectPhysics,
    SceneObject,
)
from embodichain.gen_sim.scene_engine.cli.preview import (
    _add_articulations,
    _setup_viser_joint_control,
)
from embodichain.gen_sim.scene_engine.pipeline.utils.scene_exporter import SceneExporter
from embodichain.gen_sim.scene_engine.pipeline.utils.scene_importer import (
    SceneExportImporter,
)
from embodichain.gen_sim.scene_engine.pipeline.utils.scene_usd import (
    _apply_runtime_textures_to_usd,
    _externalize_glb_textures,
    load_scene_usd_into_sim,
)


def _scene_object(
    *,
    object_id: str,
    kind: str,
    glb_path: Path | None = None,
    physics: ObjectPhysics | None = None,
) -> SceneObject:
    return SceneObject(
        id=object_id,
        kind=kind,  # type: ignore[arg-type]
        category=kind,
        name=object_id,
        description=f"{kind} object",
        simready_glb_path=str(glb_path) if glb_path is not None else None,
        rot=[0.0, 0.0, 0.0],
        pos=[1.0, 2.0, 3.0],
        scale=[1.0, 2.0, 3.0],
        physics=physics,
    )


def _physics(body_type: str) -> ObjectPhysics:
    return ObjectPhysics(
        body_type=body_type,  # type: ignore[arg-type]
        attrs={"mass": 1.0, "static_friction": 0.8},
        max_convex_hull_num=16,
    )


def _scene_graph(scene: Scene) -> SceneGraph:
    if scene.table is None:
        raise ValueError("Test scene must contain a table.")
    return SceneGraph(
        nodes=[
            SceneGraphNode(object_id="table", parent_id=None),
            *[
                SceneGraphNode(
                    object_id=asset.id,
                    parent_id="table",
                    parent_relation="on",
                )
                for asset in scene.assets
            ],
        ]
    )


def test_scene_returns_one_table_and_ordered_assets() -> None:
    table = _scene_object(object_id="table", kind="table")
    asset = _scene_object(object_id="cup", kind="asset")
    scene = Scene(objects=[table, asset])

    assert scene.table is table
    assert scene.assets == [asset]
    assert scene.to_dict()["objects"][0]["id"] == "table"  # type: ignore[index]


def test_scene_rejects_multiple_tables() -> None:
    scene = Scene(
        objects=[
            _scene_object(object_id="table_001", kind="table"),
            _scene_object(object_id="table_002", kind="table"),
        ]
    )

    with pytest.raises(ValueError, match="only one table"):
        _ = scene.table


@pytest.mark.parametrize(
    ("body_type", "attrs", "hulls"),
    [
        ("static", {"mass": 1.0}, 1),
        ("dynamic", {}, 1),
        ("dynamic", {"mass": 1.0}, 0),
    ],
)
def test_object_physics_rejects_invalid_values(
    body_type: str,
    attrs: dict[str, float],
    hulls: int,
) -> None:
    with pytest.raises(ValueError):
        _physics = ObjectPhysics(  # noqa: F841
            body_type=body_type,  # type: ignore[arg-type]
            attrs=attrs,
            max_convex_hull_num=hulls,
        )


def test_scene_export_copies_meshes_and_converts_y_up_pose(tmp_path: Path) -> None:
    table_glb = tmp_path / "table.glb"
    asset_glb = tmp_path / "cup.glb"
    table_glb.write_bytes(b"glTF-table")
    asset_glb.write_bytes(b"glTF-cup")
    table = _scene_object(
        object_id="table",
        kind="table",
        glb_path=table_glb,
        physics=_physics("kinematic"),
    )
    asset = _scene_object(
        object_id="cup",
        kind="asset",
        glb_path=asset_glb,
        physics=_physics("dynamic"),
    )
    asset.center_xy = [0.25, -0.5]

    scene = Scene(objects=[table, asset])
    export_path = SceneExporter(
        scene=scene,
        scene_graph=_scene_graph(scene),
        output_root=tmp_path / "output",
    ).export()
    exported = json.loads(export_path.read_text(encoding="utf-8"))

    assert (
        export_path.parent / "mesh_assets/table/table.glb"
    ).read_bytes() == b"glTF-table"
    assert (export_path.parent / "mesh_assets/cup/cup.glb").read_bytes() == b"glTF-cup"
    entry = exported["rigid_object"][0]
    assert entry["uid"] == "cup"
    assert entry["category"] == "asset"
    assert entry["name"] == "cup"
    assert entry["is_articulated"] is False
    assert entry["body_type"] == "dynamic"
    assert entry["init_pos"] == [1.0, -3.0, 2.0]
    assert entry["body_scale"] == [1.0, 2.0, 3.0]
    assert entry["center_xy"] == [0.25, -0.5]
    assert np.allclose(entry["init_rot"], [0.0, 0.0, 0.0])
    assert json.loads((export_path.parent / "scene_graph.json").read_text()) == {
        "nodes": [
            {
                "object_id": "table",
                "parent_id": None,
                "parent_relation": None,
                "table_region": None,
                "pose_description": None,
            },
            {
                "object_id": "cup",
                "parent_id": "table",
                "parent_relation": "on",
                "table_region": None,
                "pose_description": None,
            },
        ],
        "relations": [],
    }

    imported_scene, imported_graph = SceneExportImporter(
        output_root=tmp_path / "output"
    ).import_scene_and_graph()
    assert [asset.id for asset in imported_scene.assets] == ["cup"]
    assert imported_scene.assets[0].category == "asset"
    assert imported_scene.assets[0].name == "cup"
    assert imported_scene.assets[0].is_articulated is False
    assert imported_graph.to_dict() == _scene_graph(scene).to_dict()


def test_scene_export_uses_usdc_for_articulated_runtime_and_glb_for_editing(
    tmp_path: Path,
) -> None:
    table_glb = tmp_path / "table.glb"
    drawer_glb = tmp_path / "drawer.glb"
    drawer_usdc = tmp_path / "drawer.usdc"
    table_glb.write_bytes(b"glTF-table")
    drawer_glb.write_bytes(b"glTF-drawer")
    drawer_usdc.write_bytes(b"USDC-drawer")
    table = _scene_object(
        object_id="table",
        kind="table",
        glb_path=table_glb,
        physics=_physics("kinematic"),
    )
    drawer = _scene_object(
        object_id="drawer",
        kind="asset",
        glb_path=drawer_glb,
        physics=_physics("dynamic"),
    )
    drawer.is_articulated = True
    drawer.articulated_usdc_path = str(drawer_usdc)
    drawer.articulated_usdc_scale = [1.25, 2.5, 3.75]
    scene = Scene(objects=[table, drawer])

    export_path = SceneExporter(
        scene=scene,
        scene_graph=_scene_graph(scene),
        output_root=tmp_path / "output",
    ).export()
    exported = json.loads(export_path.read_text(encoding="utf-8"))

    assert exported["rigid_object"] == []
    articulation = exported["articulation"][0]
    assert articulation["fpath"] == "articulated_assets/drawer/drawer.usdc"
    assert articulation["proxy_glb_fpath"] == "mesh_assets/drawer/drawer.glb"
    assert articulation["body_scale"] == [1.25, 2.5, 3.75]
    assert articulation["proxy_body_scale"] == [1.0, 2.0, 3.0]
    assert (export_path.parent / articulation["fpath"]).read_bytes() == b"USDC-drawer"
    assert (
        export_path.parent / articulation["proxy_glb_fpath"]
    ).read_bytes() == b"glTF-drawer"

    imported_scene = SceneExportImporter(output_root=tmp_path / "output").import_scene()
    imported_drawer = imported_scene.assets[0]
    assert imported_drawer.simready_glb_path == str(
        export_path.parent / "mesh_assets" / "drawer" / "drawer.glb"
    )
    assert imported_drawer.articulated_usdc_path == str(
        export_path.parent / "articulated_assets" / "drawer" / "drawer.usdc"
    )
    assert imported_drawer.articulated_usdc_scale == [1.25, 2.5, 3.75]


def test_scene_usd_manifest_restores_only_declared_scene_resources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root = tmp_path / "output"
    scene_usd_root = output_root / "scene_usd"
    scene_usd_root.mkdir(parents=True)
    (scene_usd_root / "scene.usda").write_text("#usda 1.0\n", encoding="utf-8")
    manifest_objects = [
        {
            "uid": "table",
            "kind": "rigid",
            "body_type": "kinematic",
            "runtime_name": "table_0",
            "source_asset": "mesh_assets/table/table.glb",
        },
        {
            "uid": "drawer",
            "kind": "articulation",
            "runtime_name": "drawer",
            "source_asset": "articulated_assets/drawer/drawer.usdc",
            "proxy_asset": "mesh_assets/drawer/drawer.glb",
            "fix_base": True,
        },
    ]
    (scene_usd_root / "scene_usd_manifest.json").write_text(
        json.dumps(
            {
                "format": "embodichain.scene-usd/v1",
                "scene_usd": "scene_usd/scene.usda",
                "source_scene_export": "scene_export/scene_config.json",
                "objects": manifest_objects,
            }
        ),
        encoding="utf-8",
    )
    scene_export_root = output_root / "scene_export"
    scene_export_root.mkdir()
    (scene_export_root / "scene_config.json").write_text(
        json.dumps(
            {
                "format": "embodichain.scene-export/v1",
                "background": [],
                "rigid_object": [],
                "articulation": [
                    {
                        "uid": "drawer",
                        "fpath": "articulated_assets/drawer/drawer.usdc",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    drawer = object()
    loaded: dict[str, object] = {}

    def _load_legacy_scene_export(**kwargs: object) -> list[object]:
        loaded.update(kwargs)
        return [drawer]

    monkeypatch.setattr(
        "embodichain.gen_sim.scene_engine.pipeline.utils.scene_usd.load_scene_export_into_sim",
        _load_legacy_scene_export,
    )

    sim = object()
    articulations = load_scene_usd_into_sim(  # type: ignore[arg-type]
        sim=sim,
        output_root=output_root,
    )

    assert loaded == {
        "sim": sim,
        "output_root": output_root,
        "force_static_rigids": True,
    }
    assert articulations == [drawer]


def test_scene_usd_preview_loads_packaged_runtime_assets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root = tmp_path / "output"
    scene_usd_root = output_root / "scene_usd"
    runtime_asset = scene_usd_root / "assets" / "table" / "model.gltf"
    runtime_asset.parent.mkdir(parents=True)
    runtime_asset.write_text("{}", encoding="utf-8")
    articulation_asset = scene_usd_root / "assets" / "drawer" / "model.usdc"
    articulation_asset.parent.mkdir(parents=True)
    articulation_asset.write_bytes(b"USDC-drawer")
    (scene_usd_root / "scene.usda").write_text("#usda 1.0\n", encoding="utf-8")
    (scene_usd_root / "scene_usd_manifest.json").write_text(
        json.dumps(
            {
                "format": "embodichain.scene-usd/v1",
                "scene_usd": "scene_usd/scene.usda",
                "source_scene_export": "scene_export/scene_config.json",
                "objects": [
                    {
                        "uid": "table",
                        "kind": "rigid",
                        "body_type": "kinematic",
                        "runtime_name": "table_0",
                        "runtime_asset": "scene_usd/assets/table/model.gltf",
                        "init_pos": [1.0, 2.0, 3.0],
                        "init_rot": [10.0, 20.0, 30.0],
                        "body_scale": [1.0, 2.0, 3.0],
                        "max_convex_hull_num": 16,
                    },
                    {
                        "uid": "drawer",
                        "kind": "articulation",
                        "runtime_name": "drawer",
                        "runtime_asset": "scene_usd/assets/drawer/model.usdc",
                        "init_pos": [4.0, 5.0, 6.0],
                        "init_rot": [40.0, 50.0, 60.0],
                        "body_scale": [1.5, 2.5, 3.5],
                        "fix_base": True,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    class _SceneUsdSim:
        def __init__(self) -> None:
            self.rigid_cfg: object | None = None
            self.articulation_cfg: object | None = None

        def add_rigid_object(self, cfg: object) -> None:
            self.rigid_cfg = cfg

        def add_articulation(self, cfg: object) -> object:
            self.articulation_cfg = cfg
            return "drawer-resource"

    monkeypatch.setattr(
        "embodichain.gen_sim.scene_engine.pipeline.utils.scene_usd._add_lights",
        lambda _: None,
    )

    sim = _SceneUsdSim()
    assert load_scene_usd_into_sim(sim=sim, output_root=output_root) == [  # type: ignore[arg-type]
        "drawer-resource"
    ]
    assert sim.rigid_cfg.uid == "table"  # type: ignore[union-attr]
    assert sim.rigid_cfg.shape.fpath == str(runtime_asset)  # type: ignore[union-attr]
    assert sim.rigid_cfg.init_pos == (1.0, 2.0, 3.0)  # type: ignore[union-attr]
    assert sim.articulation_cfg.uid == "drawer"  # type: ignore[union-attr]
    assert sim.articulation_cfg.fpath == str(articulation_asset)  # type: ignore[union-attr]
    assert sim.articulation_cfg.init_pos == (4.0, 5.0, 6.0)  # type: ignore[union-attr]


def test_scene_usd_externalizes_every_pbr_gltf_image(tmp_path: Path) -> None:
    """Keep base-colour and normal maps addressable after GLB conversion."""
    import trimesh
    from PIL import Image
    from trimesh.visual.material import PBRMaterial
    from trimesh.visual.texture import TextureVisuals

    BASE_COLOR = (255, 0, 0)
    NORMAL_COLOR = (128, 128, 255)
    EXPECTED_IMAGE_COUNT = 2
    mesh = trimesh.creation.box()
    mesh.visual = TextureVisuals(
        uv=np.zeros((len(mesh.vertices), 2)),
        material=PBRMaterial(
            baseColorTexture=Image.new("RGB", (2, 2), BASE_COLOR),
            normalTexture=Image.new("RGB", (2, 2), NORMAL_COLOR),
        ),
    )
    source_glb = tmp_path / "multi_texture.glb"
    source_glb.write_bytes(trimesh.Scene(mesh).export(file_type="glb"))

    packaged_gltf = _externalize_glb_textures(
        source_glb=source_glb,
        destination_root=tmp_path / "packaged",
    )

    tree = json.loads(packaged_gltf.read_text(encoding="utf-8"))
    images = tree["images"]
    assert len(images) == EXPECTED_IMAGE_COUNT
    assert all(
        "uri" in image and "bufferView" not in image and "mimeType" not in image
        for image in images
    )

    def _texture_color(texture_index: int) -> tuple[int, int, int]:
        image_index = tree["textures"][texture_index]["source"]
        texture_path = packaged_gltf.parent / images[image_index]["uri"]
        with Image.open(texture_path) as texture:
            return texture.convert("RGB").getpixel((0, 0))

    material = tree["materials"][0]
    base_color_index = material["pbrMetallicRoughness"]["baseColorTexture"]["index"]
    normal_index = material["normalTexture"]["index"]
    assert _texture_color(base_color_index) == BASE_COLOR
    assert _texture_color(normal_index) == NORMAL_COLOR


def test_scene_usd_binds_every_gltf_pbr_texture_channel(tmp_path: Path) -> None:
    """Author all GLTF PBR texture channels in the scene USD material graph."""
    from PIL import Image
    from pxr import Sdf, Usd, UsdGeom, UsdShade

    CHANNEL_COLORS = {
        "base": (255, 0, 0),
        "metallic_roughness": (0, 128, 255),
        "normal": (128, 128, 255),
        "occlusion": (64, 64, 64),
        "emissive": (0, 255, 0),
    }
    scene_usd_root = tmp_path / "scene_usd"
    runtime_root = tmp_path / "runtime"
    texture_root = runtime_root / "textures"
    texture_root.mkdir(parents=True)
    image_specs = []
    for channel, color in CHANNEL_COLORS.items():
        filename = f"{channel}.png"
        Image.new("RGB", (2, 2), color).save(texture_root / filename)
        image_specs.append({"uri": f"textures/{filename}"})
    runtime_gltf = runtime_root / "model.gltf"
    runtime_gltf.write_text(
        json.dumps(
            {
                "asset": {"version": "2.0"},
                "images": image_specs,
                "textures": [{"source": index} for index in range(len(image_specs))],
                "materials": [
                    {
                        "pbrMetallicRoughness": {
                            "baseColorTexture": {"index": 0},
                            "metallicRoughnessTexture": {"index": 1},
                        },
                        "normalTexture": {"index": 2},
                        "occlusionTexture": {"index": 3},
                        "emissiveTexture": {"index": 4},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    scene_usd_root.mkdir()
    scene_usd_path = scene_usd_root / "scene.usda"
    stage = Usd.Stage.CreateNew(str(scene_usd_path))
    object_prim = UsdGeom.Xform.Define(stage, "/World/object_0").GetPrim()
    mesh = UsdGeom.Mesh.Define(stage, "/World/object_0/mesh")
    material = UsdShade.Material.Define(
        stage,
        "/World/object_0/visuals/gltf_material_index_0",
    )
    surface = UsdShade.Shader.Define(
        stage,
        "/World/object_0/visuals/gltf_material_index_0/PBRShader",
    )
    surface.CreateIdAttr("UsdPreviewSurface")
    material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    stage.GetRootLayer().Save()

    class _Entity:
        @staticmethod
        def get_name() -> str:
            return "object_0"

    class _SceneObject:
        _entities = [_Entity()]

    class _SceneUsdSim:
        @staticmethod
        def get_rigid_object_uid_list() -> list[str]:
            return ["object"]

        @staticmethod
        def get_articulation_uid_list() -> list[str]:
            return []

        @staticmethod
        def get_rigid_object(uid: str) -> _SceneObject:
            assert uid == "object"
            return _SceneObject()

        @staticmethod
        def get_articulation(uid: str) -> None:
            raise AssertionError(f"Unexpected articulation lookup: {uid}")

    _apply_runtime_textures_to_usd(
        scene_usd_path=scene_usd_path,
        scene_usd_root=scene_usd_root,
        runtime_assets={"object": runtime_gltf},
        sim=_SceneUsdSim(),  # type: ignore[arg-type]
    )

    stage = Usd.Stage.Open(str(scene_usd_path))
    assert stage is not None
    shader = UsdShade.Shader.Get(stage, surface.GetPath())

    def _connected_texture_path(input_name: str) -> str:
        source = shader.GetInput(input_name).GetConnectedSource()
        assert source
        texture = UsdShade.Shader(source[0].GetPrim())
        asset_path = texture.GetInput("file").Get()
        assert isinstance(asset_path, Sdf.AssetPath)
        return asset_path.path

    assert _connected_texture_path("diffuseColor") == "textures/object_base.png"
    assert _connected_texture_path("metallic") == (
        "textures/object_metallic_roughness.png"
    )
    assert _connected_texture_path("roughness") == (
        "textures/object_metallic_roughness.png"
    )
    assert _connected_texture_path("normal") == "textures/object_normal.png"
    assert _connected_texture_path("occlusion") == "textures/object_occlusion.png"
    assert _connected_texture_path("emissiveColor") == "textures/object_emissive.png"


def test_preview_loads_exported_usdc_as_an_articulation(tmp_path: Path) -> None:
    class FakeSimulationManager:
        def __init__(self) -> None:
            self.articulation_cfgs: list[object] = []

        def add_articulation(self, cfg: object) -> None:
            self.articulation_cfgs.append(cfg)

    usdc_path = tmp_path / "articulated_assets" / "drawer" / "drawer.usdc"
    usdc_path.parent.mkdir(parents=True)
    usdc_path.write_bytes(b"USDC-drawer")
    sim = FakeSimulationManager()

    _add_articulations(
        sim=sim,  # type: ignore[arg-type]
        entries=[
            {
                "uid": "drawer",
                "fpath": "articulated_assets/drawer/drawer.usdc",
                "init_pos": [1.0, 2.0, 3.0],
                "init_rot": [10.0, 20.0, 30.0],
                "body_scale": [1.25, 2.5, 3.75],
                "fix_base": True,
            }
        ],
        config_dir=tmp_path,
    )

    articulation_cfg = sim.articulation_cfgs[0]
    assert articulation_cfg.uid == "drawer"  # type: ignore[attr-defined]
    assert articulation_cfg.fpath == str(usdc_path)  # type: ignore[attr-defined]
    assert articulation_cfg.body_scale == (1.25, 2.5, 3.75)  # type: ignore[attr-defined]


def test_preview_registers_exported_articulation_joint_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeRuntime:
        def __init__(self) -> None:
            self.provider: object | None = None

        def set_joint_control_provider(self, provider: object) -> None:
            self.provider = provider

    class FakeSimulationManager:
        def __init__(self) -> None:
            self.visualization_runtime = FakeRuntime()

    class FakeController:
        def __init__(self, articulations: list[object], runtime: FakeRuntime) -> None:
            self.articulations = articulations
            self.runtime = runtime
            self.has_controls = True
            self.update_count = 0

        def update(self) -> None:
            self.update_count += 1

    monkeypatch.setattr(
        "embodichain.lab.scripts.preview_joint_control.ArticulationPreviewController",
        FakeController,
    )
    articulation = object()
    sim = FakeSimulationManager()

    controller = _setup_viser_joint_control(
        sim=sim,  # type: ignore[arg-type]
        articulations=[articulation],  # type: ignore[list-item]
        enabled=True,
    )

    assert controller is sim.visualization_runtime.provider
    assert controller is not None
    assert controller.articulations == [articulation]  # type: ignore[attr-defined]
    assert controller.update_count == 1  # type: ignore[attr-defined]


def test_scene_graph_importer_restores_node_pose_description() -> None:
    imported_graph = SceneExportImporter._scene_graph_from_data(
        {
            "nodes": [
                {
                    "object_id": "table",
                    "parent_id": None,
                    "parent_relation": None,
                    "table_region": None,
                    "pose_description": None,
                },
                {
                    "object_id": "bottle_001",
                    "parent_id": "table",
                    "parent_relation": "on",
                    "table_region": None,
                    "pose_description": "Stand upright on its base.",
                },
            ],
            "relations": [],
        }
    )

    assert (
        imported_graph.node_by_id()["bottle_001"].pose_description
        == "Stand upright on its base."
    )


def test_scene_graph_importer_rejects_the_removed_orientation_state_schema() -> None:
    with pytest.raises(ValueError, match="serialized node schema"):
        SceneExportImporter._scene_graph_from_data(
            {
                "nodes": [
                    {
                        "object_id": "table",
                        "parent_id": None,
                        "parent_relation": None,
                        "table_region": None,
                        "orientation_state": None,
                    }
                ],
                "relations": [],
            }
        )


def test_scene_export_overwrites_an_existing_scene_export(tmp_path: Path) -> None:
    table_glb = tmp_path / "table.glb"
    cup_glb = tmp_path / "cup.glb"
    banana_glb = tmp_path / "banana.glb"
    table_glb.write_bytes(b"glTF-table")
    cup_glb.write_bytes(b"glTF-cup")
    banana_glb.write_bytes(b"glTF-banana")
    output_root = tmp_path / "output"

    initial_table = _scene_object(
        object_id="table",
        kind="table",
        glb_path=table_glb,
        physics=_physics("kinematic"),
    )
    initial_cup = _scene_object(
        object_id="cup",
        kind="asset",
        glb_path=cup_glb,
        physics=_physics("dynamic"),
    )
    initial_scene = Scene(objects=[initial_table, initial_cup])
    SceneExporter(
        scene=initial_scene,
        scene_graph=_scene_graph(initial_scene),
        output_root=output_root,
    ).export()

    # The imported table mesh already occupies its final export location.
    exported_table_glb = (
        output_root / "scene_export" / "mesh_assets" / "table" / "table.glb"
    )
    updated_table = _scene_object(
        object_id="table",
        kind="table",
        glb_path=exported_table_glb,
        physics=_physics("kinematic"),
    )
    banana = _scene_object(
        object_id="banana",
        kind="asset",
        glb_path=banana_glb,
        physics=_physics("dynamic"),
    )
    updated_scene = Scene(objects=[updated_table, banana])
    SceneExporter(
        scene=updated_scene,
        scene_graph=_scene_graph(updated_scene),
        output_root=output_root,
    ).export()

    scene_export_root = output_root / "scene_export"
    assert exported_table_glb.read_bytes() == b"glTF-table"
    assert (
        scene_export_root / "mesh_assets" / "banana" / "banana.glb"
    ).read_bytes() == b"glTF-banana"
    assert not (scene_export_root / "mesh_assets" / "cup").exists()
    assert (
        json.loads((scene_export_root / "scene.json").read_text(encoding="utf-8"))[
            "objects"
        ][1]["id"]
        == "banana"
    )


def test_scene_export_requires_final_physics(tmp_path: Path) -> None:
    glb_path = tmp_path / "table.glb"
    glb_path.write_bytes(b"glTF")
    table = _scene_object(object_id="table", kind="table", glb_path=glb_path)

    with pytest.raises(ValueError, match="no SimReady physics"):
        SceneExporter(
            scene=Scene(objects=[table]),
            scene_graph=SceneGraph(
                nodes=[SceneGraphNode(object_id="table", parent_id=None)]
            ),
            output_root=tmp_path,
        ).export()


def test_scene_export_rejects_backslash_in_object_id(tmp_path: Path) -> None:
    glb_path = tmp_path / "table.glb"
    glb_path.write_bytes(b"glTF")
    table = _scene_object(
        object_id="table",
        kind="table",
        glb_path=glb_path,
        physics=_physics("kinematic"),
    )
    unsafe_asset = _scene_object(
        object_id=r"..\evil",
        kind="asset",
        glb_path=glb_path,
        physics=_physics("dynamic"),
    )

    with pytest.raises(ValueError, match="not safe for a GLB filename"):
        scene = Scene(objects=[table, unsafe_asset])
        SceneExporter(
            scene=scene,
            scene_graph=_scene_graph(scene),
            output_root=tmp_path / "output",
        ).export()
