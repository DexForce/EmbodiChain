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

from contextlib import contextmanager
from collections.abc import Iterator
import json
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

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

from embodichain.gen_sim.scene_engine.pipeline.utils.scene_exporter import SceneExporter
from embodichain.gen_sim.scene_engine.pipeline.utils.scene_importer import (
    SceneExportImporter,
)
from embodichain.gen_sim.scene_engine.pipeline.utils.scene_usd import (
    _add_articulations,
    _add_objects,
    _apply_runtime_textures_to_usd,
    _copy_runtime_texture,
    _externalize_glb_textures,
    _validate_uid,
    build_scene_usdz,
)
from embodichain.gen_sim.scene_engine.pipeline.utils.usd_scene import (
    USD_SCENE_SCHEMA,
    UsdSceneIndex,
)
from embodichain.lab.visualization import VisualizationCfg


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
        attrs={
            "mass_props": {"mass": 1.0},
            "material_props": {"static_friction": 0.8},
        },
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


@pytest.mark.parametrize(
    "appearance,expected",
    [
        ("red", "red"),
        ("blue", "blue"),
        ("mixed", None),
        ("mesh_with_lines", None),
        ("material", "red"),
        ("material_masked_invisible", None),
        ("material_masked_visible", "red"),
        ("material_blended", None),
        ("material_opaque_alpha", "red"),
        ("material_mask_float_below_cutoff", None),
        ("material_mask_float_at_cutoff", "red"),
        ("material_mask_fully_opaque", "red"),
        ("material_blend_float", None),
        ("material_blend_fully_opaque", "red"),
        ("material_opaque_float", "red"),
        ("material_extension_mask", None),
        ("material_extension_blend", None),
        ("material_extension_opaque", None),
        ("material_with_vertex_color", None),
        ("textured", None),
        ("textured_external", None),
    ],
)
def test_scene_export_carries_measured_appearance(tmp_path, appearance, expected):
    import hashlib
    import trimesh
    from PIL import Image

    table_path = tmp_path / "table.glb"
    table_path.write_bytes(b"glTF-table")
    mesh = trimesh.creation.box()
    if appearance.startswith("material") or appearance.startswith("textured"):
        alpha, mode, cutoff = {
            "material_masked_invisible": (250, "MASK", 0.99),
            "material_masked_visible": (250, "MASK", 0.9),
            "material_blended": (250, "BLEND", 0.5),
            "material_opaque_alpha": (0, "OPAQUE", 0.5),
        }.get(appearance, (255, "OPAQUE", 0.5))
        material = trimesh.visual.material.PBRMaterial(
            baseColorFactor=(
                [255, 255, 255, alpha]
                if appearance == "textured_external"
                else [255, 0, 0, alpha]
            ),
            alphaMode=mode,
            alphaCutoff=cutoff,
            baseColorTexture=(
                Image.new("RGB", (2, 2), "blue")
                if appearance.startswith("textured")
                else None
            ),
        )
        mesh.visual = trimesh.visual.TextureVisuals(
            uv=np.zeros((len(mesh.vertices), 2)), material=material
        )
        if appearance == "material_with_vertex_color":
            # A red factor multiplied by blue COLOR_0 is black, not red.
            mesh.visual.vertex_attributes["color"] = np.tile(
                [0, 0, 255, 255], (len(mesh.vertices), 1)
            ).astype(np.uint8)
    else:
        mesh.visual.vertex_colors = (
            [0, 0, 255, 255] if appearance == "blue" else [255, 0, 0, 255]
        )
        if appearance == "mixed":
            mesh.visual.vertex_colors[0] = [0, 0, 255, 255]
    asset_path = tmp_path / "asset.glb"
    if appearance == "mesh_with_lines":
        mesh = trimesh.Scene(
            [mesh, trimesh.load_path(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))]
        )
    mesh.export(asset_path)
    if appearance in {
        "material_mask_float_below_cutoff",
        "material_mask_float_at_cutoff",
        "material_mask_fully_opaque",
        "material_blend_float",
        "material_blend_fully_opaque",
        "material_opaque_float",
    }:
        # Patch the authored GLB JSON, bypassing Trimesh's uint8 factor setter.
        # Preserve the geometry/binary chunks to exercise the real exporter.
        import struct

        data = asset_path.read_bytes()
        chunk_size = struct.unpack_from("<I", data, 12)[0]
        document = json.loads(data[20 : 20 + chunk_size])
        mode, alpha, cutoff = {
            "material_mask_float_below_cutoff": ("MASK", 0.999, 1.0),
            "material_mask_float_at_cutoff": ("MASK", 0.999, 0.999),
            "material_mask_fully_opaque": ("MASK", 1.0, 1.0),
            "material_blend_float": ("BLEND", 0.999, 0.5),
            "material_blend_fully_opaque": ("BLEND", 1.0, 0.5),
            "material_opaque_float": ("OPAQUE", 0.999, 0.5),
        }[appearance]
        material = document["materials"][0]
        material.update(alphaMode=mode, alphaCutoff=cutoff)
        material["pbrMetallicRoughness"]["baseColorFactor"][3] = alpha
        header = json.dumps(document).encode("utf-8")
        header += b" " * (-len(header) % 4)
        binary = data[20 + chunk_size :]
        asset_path.write_bytes(
            struct.pack(
                "<4sIIII",
                b"glTF",
                2,
                20 + len(header) + len(binary),
                len(header),
                0x4E4F534A,
            )
            + header
            + binary
        )
    if appearance == "textured_external":
        import struct

        data = asset_path.read_bytes()
        size = struct.unpack_from("<I", data, 12)[0]
        document = json.loads(data[20 : 20 + size])
        document["images"][0] = {"uri": "blue.png"}
        header = json.dumps(document).encode()
        header += b" " * (-len(header) % 4)
        binary = data[20 + size :]
        asset_path.write_bytes(
            struct.pack(
                "<4sIIII",
                b"glTF",
                2,
                20 + len(header) + len(binary),
                len(header),
                0x4E4F534A,
            )
            + header
            + binary
        )
        Image.new("RGB", (2, 2), "blue").save(tmp_path / "blue.png")
    if appearance.startswith("material_extension"):
        import struct

        data = asset_path.read_bytes()
        size = struct.unpack_from("<I", data, 12)[0]
        document = json.loads(data[20 : 20 + size])
        mode, alpha = {
            "material_extension_mask": ("MASK", 0.0),
            "material_extension_blend": ("BLEND", 0.999),
            "material_extension_opaque": ("OPAQUE", 1.0),
        }[appearance]
        document["materials"][0] = {
            "alphaMode": mode,
            "alphaCutoff": 0.5,
            "extensions": {
                "KHR_materials_pbrSpecularGlossiness": {
                    "diffuseFactor": [1.0, 1.0, 1.0, alpha]
                }
            },
        }
        document["extensionsUsed"] = ["KHR_materials_pbrSpecularGlossiness"]
        header = json.dumps(document).encode()
        header += b" " * (-len(header) % 4)
        binary = data[20 + size :]
        asset_path.write_bytes(
            struct.pack(
                "<4sIIII",
                b"glTF",
                2,
                20 + len(header) + len(binary),
                len(header),
                0x4E4F534A,
            )
            + header
            + binary
        )
    table = _scene_object(
        object_id="table",
        kind="table",
        glb_path=table_path,
        physics=_physics("kinematic"),
    )
    asset = _scene_object(
        object_id="asset",
        kind="asset",
        glb_path=asset_path,
        physics=_physics("dynamic"),
    )
    asset.description = "red waterproof cube"
    scene = Scene(objects=[table, asset])
    path = SceneExporter(
        scene=scene, scene_graph=_scene_graph(scene), output_root=tmp_path / "export"
    ).export()
    entry = json.loads(path.read_text())["rigid_object"][0]
    if expected is None:
        assert "properties" not in entry
    else:
        assert entry["properties"] == {"color": expected}
        proof = entry["property_evidence"]["color"]
        assert (
            proof["sha256"]
            == hashlib.sha256((path.parent / proof["source"]).read_bytes()).hexdigest()
        )
    if appearance == "textured_external":
        deployed = path.parent / "mesh_assets/asset/asset.glb"
        assert not deployed.with_name("blue.png").exists()
        original = hashlib.sha256(deployed.read_bytes()).hexdigest()
        Image.new("RGB", (2, 2), "blue").save(deployed.with_name("blue.png"))
        assert (
            trimesh.load_mesh(deployed, process=False).visual.material.baseColorTexture
            is not None
        )
        assert SceneExporter._appearance_metadata(deployed) == {}
        assert hashlib.sha256(deployed.read_bytes()).hexdigest() == original
    assert "waterproof" not in entry.get("properties", {})


def test_external_color_buffer_never_creates_glb_only_proof(tmp_path):
    import hashlib
    import struct
    import trimesh

    root = tmp_path / "export"
    folder = root / "scene_export/mesh_assets/asset"
    folder.mkdir(parents=True)
    path = folder / "asset.glb"
    mesh = trimesh.creation.box()
    mesh.visual.vertex_colors = [255, 0, 0, 255]
    data = mesh.export(file_type="glb")
    size = struct.unpack_from("<I", data, 12)[0]
    document = json.loads(data[20 : 20 + size])
    binary = bytearray(data[28 + size :])
    document["buffers"][0]["uri"] = "colors.bin"
    header = json.dumps(document).encode()
    header += b" " * (-len(header) % 4)
    path.write_bytes(
        struct.pack("<4sIIII", b"glTF", 2, 20 + len(header), len(header), 0x4E4F534A)
        + header
    )
    buffer = folder / "colors.bin"
    buffer.write_bytes(binary)
    original_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    asset = _scene_object(
        object_id="asset", kind="asset", glb_path=path, physics=_physics("dynamic")
    )
    table_path = tmp_path / "table.glb"
    table_path.write_bytes(b"glTF-table")
    table = _scene_object(
        object_id="table",
        kind="table",
        glb_path=table_path,
        physics=_physics("kinematic"),
    )
    scene = Scene(objects=[table, asset])
    exporter = SceneExporter(
        scene=scene, scene_graph=_scene_graph(scene), output_root=root
    )
    for color in ([255, 0, 0, 255], [0, 0, 255, 255]):
        index = document["meshes"][0]["primitives"][0]["attributes"]["COLOR_0"]
        accessor = document["accessors"][index]
        view = document["bufferViews"][accessor["bufferView"]]
        start = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
        for row in range(accessor["count"]):
            offset = start + row * view.get("byteStride", 4)
            binary[offset : offset + 4] = bytes(color)
        buffer.write_bytes(binary)
        loaded = trimesh.load_scene(path, process=False)
        assert (
            next(iter(loaded.geometry.values())).visual.vertex_colors[0].tolist()
            == color
        )
        exported = json.loads(exporter.export().read_text())["rigid_object"][0]
        assert "properties" not in exported
        assert "property_evidence" not in exported
        assert hashlib.sha256(path.read_bytes()).hexdigest() == original_hash


@pytest.mark.parametrize(
    ("table_state", "placement_support"),
    [(None, None), ("normal", None), ("degraded", None), ("degraded", "floor")],
)
def test_scene_export_copies_meshes_and_converts_y_up_pose(
    tmp_path: Path, table_state: str | None, placement_support: str | None
) -> None:
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
    asset.placement_support = placement_support  # type: ignore[assignment]
    table.table_state = table_state  # type: ignore[assignment]

    scene = Scene(objects=[table, asset])
    export_path = SceneExporter(
        scene=scene,
        scene_graph=_scene_graph(scene),
        output_root=tmp_path / "output",
    ).export()
    exported = json.loads(export_path.read_text(encoding="utf-8"))
    assert exported["background"][0]["table_state"] == table_state
    assert "placement_support" not in exported["background"][0]
    assert "table_state" not in exported["rigid_object"][0]
    assert table.to_dict()["table_state"] == table_state

    assert (
        export_path.parent / "mesh_assets/table/table.glb"
    ).read_bytes() == b"glTF-table"
    assert (export_path.parent / "mesh_assets/cup/cup.glb").read_bytes() == b"glTF-cup"
    entry = exported["rigid_object"][0]
    snapshot = json.loads(
        (export_path.parent / "scene.json").read_text(encoding="utf-8")
    )["objects"][1]
    for data in (entry, snapshot):
        if placement_support is None:
            assert "placement_support" not in data
        else:
            assert data["placement_support"] == placement_support
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
    assert imported_scene.assets[0].placement_support == placement_support
    assert imported_scene.table.table_state == table_state
    assert imported_graph.to_dict() == _scene_graph(scene).to_dict()


@pytest.fixture
def drawer_scene(tmp_path: Path) -> Scene:
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
    return Scene(objects=[table, drawer])


def _mock_drawer_fit(
    monkeypatch: pytest.MonkeyPatch,
    *,
    baked: bool,
    fitted_bytes: bytes = b"USDC-fitted",
) -> None:
    from embodichain.gen_sim.scene_engine.pipeline.utils import scene_exporter

    def fit_native_pose(
        self, *, scene_object, proxy_position, support_z, runtime_usdc_path, **_
    ):
        assert scene_object.id == "drawer"
        assert support_z == pytest.approx(0.73)
        original_path = None
        if baked:
            original_path = runtime_usdc_path.with_name("drawer.source.usdc")
            original_path.write_bytes(runtime_usdc_path.read_bytes())
            runtime_usdc_path.write_bytes(fitted_bytes)
        return (
            [proxy_position[0] + 0.1, proxy_position[1], 0.781],
            [0, 0, 25],
            [1 if baked else 2] * 3,
            original_path,
            270 if baked else None,
        )

    monkeypatch.setattr(SceneExporter, "_fit_articulation_pose", fit_native_pose)
    monkeypatch.setattr(
        scene_exporter,
        "measure_scene_object_z_up_world_aabb",
        lambda **kwargs: [[0, 0, 0], [1, 1, 0.73]],
    )


@pytest.mark.parametrize("baked", [False, True])
def test_scene_export_uses_usdc_for_articulated_runtime_and_glb_for_editing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drawer_scene: Scene,
    baked: bool,
) -> None:
    _mock_drawer_fit(monkeypatch, baked=baked)
    scene = drawer_scene
    drawer = scene.assets[0]

    export_path = SceneExporter(
        scene=scene,
        scene_graph=_scene_graph(scene),
        output_root=tmp_path / "output",
    ).export()
    exported = json.loads(export_path.read_text(encoding="utf-8"))

    assert exported["rigid_object"] == []
    articulation = exported["articulation"][0]
    assert "placement_support" not in articulation
    assert articulation["fpath"] == "articulated_assets/drawer/drawer.usdc"
    assert articulation["proxy_glb_fpath"] == "mesh_assets/drawer/drawer.glb"
    assert articulation["body_scale"] == [1.0 if baked else 2.0] * 3
    assert articulation["proxy_body_scale"] == [1.0, 2.0, 3.0]
    assert articulation["init_pos"][2] == pytest.approx(0.781)
    assert articulation["init_rot"] == [0.0, 0.0, 25.0]
    assert np.asarray(articulation["init_local_pose"])[:3, 3] == pytest.approx(
        articulation["init_pos"]
    )
    assert not np.allclose(articulation["proxy_init_rot"], articulation["init_rot"])
    assert (export_path.parent / articulation["fpath"]).read_bytes() == (
        b"USDC-fitted" if baked else b"USDC-drawer"
    )
    assert (
        export_path.parent / articulation["proxy_glb_fpath"]
    ).read_bytes() == b"glTF-drawer"

    imported_scene = SceneExportImporter(output_root=tmp_path / "output").import_scene()
    imported_drawer = imported_scene.assets[0]
    assert imported_drawer.placement_support is None
    assert imported_drawer.simready_glb_path == str(
        export_path.parent / "mesh_assets" / "drawer" / "drawer.glb"
    )
    expected_source = "drawer.source.usdc" if baked else "drawer.usdc"
    expected_source = (
        export_path.parent / "articulated_assets" / "drawer" / expected_source
    )
    assert imported_drawer.articulated_usdc_path == str(expected_source)
    assert expected_source.read_bytes() == b"USDC-drawer"
    assert imported_drawer.articulated_usdc_scale == [1.0 if baked else 2.0] * 3
    assert imported_drawer.articulation_yaw_offset_degrees == (270 if baked else None)
    assert np.allclose(imported_drawer.pos, drawer.pos)
    assert np.allclose(imported_drawer.rot, drawer.rot)


@pytest.mark.parametrize(
    "collision_enabled", [None, False, True], ids=["visual", "disabled", "enabled"]
)
def test_scene_export_checks_effective_collision_of_non_mesh_primitives(
    tmp_path: Path, drawer_scene: Scene, collision_enabled: bool | None
) -> None:
    from pxr import Usd, UsdGeom, UsdPhysics
    import trimesh

    table = drawer_scene.table
    assert table is not None
    (drawer,) = drawer_scene.assets
    table.pos = [0.0, 0.0, 0.0]
    table.scale = [1.0, 1.0, 1.0]
    table.support_surface_z = 0.5
    for item in (table, drawer):
        trimesh.creation.box().export(item.simready_glb_path)
    drawer.articulated_usdc_scale = [2.0, 3.0, 4.0]
    source = Path(drawer.articulated_usdc_path)
    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, "/Drawer").GetPrim()
    stage.SetDefaultPrim(root)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.ArticulationRootAPI.Apply(root)
    for uid in ("base", "slide"):
        body = UsdGeom.Xform.Define(stage, f"/Drawer/{uid}")
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        collider = UsdGeom.Mesh.Define(stage, f"/Drawer/{uid}/collision")
        collider.CreatePointsAttr([(0, 0, -0.25), (1, 0, 0), (0, 1, 0), (0, 0, 1)])
        collider.CreateFaceVertexCountsAttr([3, 3, 3, 3])
        collider.CreateFaceVertexIndicesAttr([0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3])
        UsdPhysics.CollisionAPI.Apply(collider.GetPrim())
    handle_path = "/Drawer/slide/visual_handle"
    handle = UsdGeom.Cube.Define(stage, handle_path)
    handle.CreateSizeAttr(0.1)
    if collision_enabled is not None:
        UsdPhysics.CollisionAPI.Apply(handle.GetPrim()).CreateCollisionEnabledAttr(
            collision_enabled
        )
    joint_path = "/Drawer/joint"
    joint = UsdPhysics.PrismaticJoint.Define(stage, joint_path)
    joint.CreateBody0Rel().SetTargets(["/Drawer/base"])
    joint.CreateBody1Rel().SetTargets(["/Drawer/slide"])
    joint.CreateUpperLimitAttr(0.2)
    joint.CreateLowerLimitAttr(0.0)
    stage.GetRootLayer().Export(str(source))

    exporter = SceneExporter(
        scene=drawer_scene, scene_graph=_scene_graph(drawer_scene), output_root=tmp_path
    )
    if collision_enabled:
        with pytest.raises(ValueError, match="requires mesh geometry"):
            exporter.export()
        return

    path = exporter.export()
    (config,) = json.loads(path.read_text())["articulation"]
    runtime = Usd.Stage.Open(str(path.parent / config["fpath"]))
    exported_handle = UsdGeom.Cube.Get(runtime, handle_path)
    assert exported_handle.GetSizeAttr().Get() == pytest.approx(0.1)
    if collision_enabled is False:
        assert (
            UsdPhysics.CollisionAPI(exported_handle.GetPrim())
            .GetCollisionEnabledAttr()
            .Get()
            is False
        )
    assert UsdPhysics.PrismaticJoint.Get(
        runtime, joint_path
    ).GetUpperLimitAttr().Get() == pytest.approx(0.2)
    assert config["body_scale"] == [2.0, 3.0, 4.0]
    # Half-metre table, one metre of scaled collision bottom, plus 1 mm clearance.
    assert config["init_pos"][2] == pytest.approx(1.501)


@pytest.mark.parametrize("failure_stage", ["fit", "usdz", "publish"])
def test_failed_in_place_scene_export_preserves_previous_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drawer_scene: Scene,
    failure_stage: str,
) -> None:
    from pxr import Usd

    from embodichain.gen_sim.scene_engine.pipeline.utils import scene_usd

    def drawer_layer(name: str) -> Path:
        path = tmp_path / f"{name}.usdc"
        stage = Usd.Stage.CreateNew(str(path))
        stage.DefinePrim(f"/{name}", "Xform")
        stage.GetRootLayer().Save()
        return path

    source = drawer_layer("source_drawer")
    fitted = drawer_layer("fitted_drawer")
    drawer_scene.assets[0].articulated_usdc_path = str(source)
    _mock_drawer_fit(monkeypatch, baked=True, fitted_bytes=fitted.read_bytes())
    output_root = tmp_path / "output"
    config_path = SceneExporter(
        scene=drawer_scene,
        scene_graph=_scene_graph(drawer_scene),
        output_root=output_root,
    ).export()
    export_root = config_path.parent
    drawer_root = export_root / "articulated_assets" / "drawer"
    original_model = drawer_root / "drawer.source.usdc"
    fitted_model = drawer_root / "drawer.usdc"
    assert original_model.read_bytes() == source.read_bytes()
    assert fitted_model.read_bytes() == fitted.read_bytes()
    assert original_model.read_bytes() != fitted_model.read_bytes()

    usd_path = output_root / "scene_usd" / "scene.usda"
    usd_path.parent.mkdir()
    usd_path.write_text('#usda 1.0\n\ndef Xform "drawer" {}\n', encoding="utf-8")
    usdz_path = build_scene_usdz(scene_usd_path=usd_path)
    package_path = output_root / "scene_package.zip"
    with ZipFile(package_path, "w") as package:
        package.write(usd_path, "asset.usda")
    previous_artifacts = {
        path: path.read_bytes()
        for path in (
            config_path,
            export_root / "scene_graph.json",
            export_root / "scene.json",
            original_model,
            fitted_model,
            usd_path,
            usdz_path,
            package_path,
        )
    }

    importer = SceneExportImporter(output_root=output_root)
    scene, graph = importer.read_scene_and_graph()
    assert Path(scene.assets[0].articulated_usdc_path) == original_model
    before_scene = scene.to_dict()
    scene.assets[0].pos = [4.0, 2.0, 3.0]

    def fail_fit(self, **kwargs) -> None:
        raise RuntimeError("injected fit failure")

    def stage_usd(*, output_root: Path) -> Path:
        nonlocal staged_usd_dir
        staged_usd_dir = output_root / "scene_usd"
        staged_usd_dir.mkdir()
        scene_path = staged_usd_dir / "scene.usda"
        scene_path.write_bytes(b"new USDA")
        (output_root / "scene_package.zip").write_bytes(b"new ZIP")
        return scene_path

    def stage_usdz(*, scene_usd_path: Path) -> Path:
        path = scene_usd_path.with_suffix(".usdz")
        path.write_bytes(b"new USDZ")
        if failure_stage == "usdz":
            raise RuntimeError("injected usdz failure")
        return path

    rename = Path.rename
    staged_usd_dir: Path | None = None

    def fail_publish(path: Path, destination: Path) -> Path:
        if path == staged_usd_dir and destination == usd_path.parent:
            assert config_path.read_bytes() != previous_artifacts[config_path]
            raise OSError("injected publish failure")
        return rename(path, destination)

    if failure_stage == "fit":
        monkeypatch.setattr(SceneExporter, "_fit_articulation_pose", fail_fit)
    else:
        monkeypatch.setattr(scene_usd, "build_scene_usd", stage_usd)
        monkeypatch.setattr(scene_usd, "build_scene_usdz", stage_usdz)
        if failure_stage == "publish":
            monkeypatch.setattr(Path, "rename", fail_publish)

    error = OSError if failure_stage == "publish" else RuntimeError
    with pytest.raises(error, match=f"injected {failure_stage} failure"):
        SceneExporter(scene=scene, scene_graph=graph, output_root=output_root).export(
            include_delivery=failure_stage != "fit"
        )

    for path, expected in previous_artifacts.items():
        assert path.read_bytes() == expected, path
    reloaded_scene, reloaded_graph = importer.read_scene_and_graph()
    assert reloaded_scene.to_dict() == before_scene
    assert reloaded_graph.to_dict() == graph.to_dict()
    previous_drawer = Usd.Stage.Open(str(fitted_model))
    assert previous_drawer.GetPrimAtPath("/fitted_drawer").IsValid()
    previous_stage = Usd.Stage.Open(str(usdz_path))
    assert previous_stage.GetPrimAtPath("/drawer").IsValid()
    with ZipFile(package_path) as package:
        assert package.read("asset.usda") == previous_artifacts[usd_path]


@pytest.mark.parametrize("max_hulls", [1, 8])
def test_preview_forwards_visacd_collision_policy(
    tmp_path: Path, max_hulls: int
) -> None:
    from embodichain.lab.sim.cfg import RigidObjectCfg
    from embodichain.lab.sim.spawn.descriptors import rigid_desc_from_cfg

    class FakeSimulationManager:
        def __init__(self) -> None:
            self.configs: list[RigidObjectCfg] = []

        def add_rigid_object(self, cfg: RigidObjectCfg) -> None:
            self.configs.append(cfg)

    (tmp_path / "mesh.glb").write_bytes(b"glTF")
    sim = FakeSimulationManager()
    _add_objects(
        sim=sim,  # type: ignore[arg-type]
        entries=[
            {
                "uid": "mesh",
                "placement_support": "floor",
                "shape": {"shape_type": "Mesh", "fpath": "mesh.glb"},
                "init_pos": [0.0, 0.0, 0.0],
                "init_rot": [0.0, 0.0, 0.0],
                "max_convex_hull_num": max_hulls,
                "attrs": {
                    "mass_props": {"mass": 0.25},
                    "material_props": {"static_friction": 0.7, "dynamic_friction": 0.4},
                },
            }
        ],
        config_dir=tmp_path,
        label="rigid_object",
        force_static=True,
    )

    cfg = sim.configs[0]
    assert cfg.body_type == "static"
    assert cfg.attrs.mass_props.mass == pytest.approx(0.25)
    assert cfg.attrs.material_props.static_friction == pytest.approx(0.7)
    assert cfg.attrs.material_props.dynamic_friction == pytest.approx(0.4)
    collision = cfg.shape.collision
    assert collision.acd_method == ("visacd" if max_hulls > 1 else None)
    descriptor, _ = rigid_desc_from_cfg(cfg)
    if max_hulls > 1:
        assert descriptor.collisions[0].decomp_algorithm == "visacd"
        assert descriptor.collisions[0].decomp_max_hulls == max_hulls


@pytest.fixture
def readable_scene_export(tmp_path: Path) -> Path:
    """Create a portable two-object export with all required final assets."""
    objects = []
    for object_id, kind, body_type in [
        ("table", "table", "kinematic"),
        ("cup", "asset", "dynamic"),
    ]:
        glb_path = tmp_path / f"{object_id}.glb"
        glb_path.write_bytes(f"glTF-{object_id}".encode())
        objects.append(
            _scene_object(
                object_id=object_id,
                kind=kind,
                glb_path=glb_path,
                physics=_physics(body_type),
            )
        )
    scene = Scene(objects=objects)
    output_root = tmp_path / "output"
    SceneExporter(
        scene=scene,
        scene_graph=_scene_graph(scene),
        output_root=output_root,
    ).export()
    return output_root


@pytest.mark.parametrize("has_debug_snapshot", [True, False])
def test_scene_export_read_preserves_source_files(
    readable_scene_export: Path, has_debug_snapshot: bool
) -> None:
    """Read-only consumers may load an export without rewriting its snapshot."""
    scene_export = readable_scene_export / "scene_export"
    scene_json = scene_export / "scene.json"
    config_path = scene_export / "scene_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    # Exports predating observed table completeness have no table_state field.
    config["background"][0].pop("table_state", None)
    config_path.write_text(json.dumps(config), encoding="utf-8")
    if has_debug_snapshot:
        # Preserve even a non-canonical snapshot instead of regenerating it.
        scene_json.write_text('{"debug": "keep verbatim"}\n', encoding="utf-8")
    else:
        scene_json.unlink()
    original_files = {
        path.relative_to(scene_export): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in scene_export.rglob("*")
        if path.is_file()
    }

    scene, graph = SceneExportImporter(
        output_root=readable_scene_export
    ).read_scene_and_graph()

    assert [scene_object.id for scene_object in scene.objects] == ["table", "cup"]
    assert scene.table.table_state is None
    assert np.allclose(scene.assets[0].pos, [1.0, 2.0, 3.0])
    assert graph.node_by_id()["cup"].parent_id == "table"
    assert {
        path.relative_to(scene_export): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in scene_export.rglob("*")
        if path.is_file()
    } == original_files


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("background", "table_state", "partial"),
        ("rigid_object", "table_state", "normal"),
        ("background", "placement_support", "floor"),
        ("rigid_object", "placement_support", "table"),
    ],
)
def test_scene_export_read_rejects_invalid_object_metadata(
    readable_scene_export: Path, section: str, field: str, value: str
) -> None:
    config_path = readable_scene_export / "scene_export" / "scene_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config[section][0][field] = value
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match=field):
        SceneExportImporter(output_root=readable_scene_export).read_scene_and_graph()


def test_scene_export_read_rejects_mismatched_graph(
    readable_scene_export: Path,
) -> None:
    scene_export = readable_scene_export / "scene_export"
    graph_path = scene_export / "scene_graph.json"
    graph_data = json.loads(graph_path.read_text(encoding="utf-8"))
    graph_data["nodes"] = graph_data["nodes"][:1]
    graph_path.write_text(json.dumps(graph_data), encoding="utf-8")
    scene_json = scene_export / "scene.json"
    original_snapshot = scene_json.read_bytes()

    with pytest.raises(ValueError, match="nodes must match"):
        SceneExportImporter(output_root=readable_scene_export).read_scene_and_graph()

    assert scene_json.read_bytes() == original_snapshot


def test_scene_export_read_requires_final_asset(readable_scene_export: Path) -> None:
    glb_path = readable_scene_export / "scene_export/mesh_assets/cup/cup.glb"
    glb_path.unlink()

    with pytest.raises(FileNotFoundError, match="GLB not found"):
        SceneExportImporter(output_root=readable_scene_export).read_scene_and_graph()


def test_scene_export_read_accepts_explicit_export_directory(
    readable_scene_export: Path,
) -> None:
    """Export identity comes from its files rather than its directory name."""
    export_root = readable_scene_export / "renamed_portable_scene"
    (readable_scene_export / "scene_export").rename(export_root)
    original_snapshot = (export_root / "scene.json").read_bytes()

    scene, graph = SceneExportImporter(
        output_root=export_root.parent,
        scene_export_root=export_root,
    ).read_scene_and_graph()

    assert [asset.id for asset in scene.assets] == ["cup"]
    assert graph.node_by_id()["cup"].parent_id == "table"
    assert scene.assets[0].simready_glb_path == str(
        export_root / "mesh_assets/cup/cup.glb"
    )
    assert (export_root / "scene.json").read_bytes() == original_snapshot


def test_scene_export_importer_rejects_explicit_directory_outside_output_root(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="must stay within output root"):
        SceneExportImporter(
            output_root=tmp_path / "output",
            scene_export_root=tmp_path / "outside",
        )


def test_scene_usd_overwrites_existing_runtime_texture(tmp_path: Path) -> None:
    scene_usd_root = tmp_path / "scene_usd"
    texture_root = scene_usd_root / "textures"
    texture_root.mkdir(parents=True)
    target_texture = texture_root / "table_image_0.png"
    target_texture.write_bytes(b"old")
    source_texture = tmp_path / "image_0.png"
    source_texture.write_bytes(b"new")

    relative_path = _copy_runtime_texture(
        source_texture=source_texture,
        texture_root=texture_root,
        scene_usd_root=scene_usd_root,
        uid="table",
    )

    assert relative_path == "textures/table_image_0.png"
    assert target_texture.read_bytes() == b"new"


def test_scene_usd_rejects_path_like_uids() -> None:
    with pytest.raises(ValueError, match="safe single path component"):
        _validate_uid("../outside", label="Scene object")


def test_usd_scene_index_reads_entity_metadata(tmp_path: Path) -> None:
    from pxr import Usd, UsdGeom, Vt

    scene_path = tmp_path / "scene.usda"
    stage = Usd.Stage.CreateNew(str(scene_path))
    world = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    world.SetCustomDataByKey("embodichain:scene_schema", USD_SCENE_SCHEMA)
    entity = UsdGeom.Xform.Define(stage, "/World/Scene/Entities/drawer").GetPrim()
    entity.SetCustomDataByKey("embodichain:uid", "drawer")
    entity.SetCustomDataByKey("embodichain:kind", "articulation")
    entity.SetCustomDataByKey("embodichain:runtime_name", "drawer_0")
    entity.SetCustomDataByKey("embodichain:fixed_base", True)
    entity.SetCustomDataByKey(
        "embodichain:joint_names", Vt.StringArray(["joint_a", "joint_b"])
    )
    entity.SetCustomDataByKey("embodichain:initial_qpos", Vt.DoubleArray([0.2, -0.4]))
    stage.GetRootLayer().Save()

    index = UsdSceneIndex.load(scene_path, require_schema=True)

    drawer = index.get("drawer")
    assert drawer.prim_path == "/World/Scene/Entities/drawer"
    assert drawer.kind == "articulation"
    assert drawer.fixed_base is True
    assert drawer.joint_names == ("joint_a", "joint_b")
    assert drawer.initial_qpos == (0.2, -0.4)


def test_build_scene_usdz_creates_relocatable_single_file(tmp_path: Path) -> None:
    from pxr import Usd, UsdGeom

    scene_path = tmp_path / "scene.usda"
    stage = Usd.Stage.CreateNew(str(scene_path))
    UsdGeom.Xform.Define(stage, "/World")
    stage.GetRootLayer().Save()

    package_path = build_scene_usdz(scene_usd_path=scene_path)

    assert package_path == tmp_path / "scene.usdz"
    assert package_path.is_file()
    assert Usd.Stage.Open(str(package_path)) is not None


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
    BOTTOM_ROW_COLOR = (17, 29, 43)  # A distinct second row exposes an inverted image.
    scene_usd_root = tmp_path / "scene_usd"
    runtime_root = tmp_path / "runtime"
    texture_root = runtime_root / "textures"
    texture_root.mkdir(parents=True)
    image_specs = []
    for channel, color in CHANNEL_COLORS.items():
        filename = f"{channel}.png"
        image = Image.new("RGB", (2, 2), color)
        image.putpixel((0, 1), BOTTOM_ROW_COLOR)
        image.putpixel((1, 1), BOTTOM_ROW_COLOR)
        image.save(texture_root / filename)
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
    for channel, color in CHANNEL_COLORS.items():
        with Image.open(
            scene_usd_root / "textures" / f"object_{channel}.png"
        ) as exported:
            assert exported.getpixel((0, 0)) == BOTTOM_ROW_COLOR
            assert exported.getpixel((0, 1)) == color
        with Image.open(texture_root / f"{channel}.png") as original:
            assert original.getpixel((0, 0)) == color
            assert original.getpixel((0, 1)) == BOTTOM_ROW_COLOR


def test_preview_and_gym_keep_exported_articulation_pose(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scipy.spatial.transform import Rotation

    from embodichain.gen_sim._run_binding import _physical_scene
    from embodichain.gen_sim.scene_engine.pipeline.utils import scene_usd
    from embodichain.lab.sim.cfg import ArticulationCfg

    monkeypatch.setattr(scene_usd, "_read_revolute_qpos_limits", lambda path: {})

    class FakeSimulationManager:
        def __init__(self) -> None:
            self.articulation_cfgs: list[object] = []

        def add_articulation(self, cfg: object) -> None:
            self.articulation_cfgs.append(cfg)

    usdc_path = tmp_path / "articulated_assets" / "drawer" / "drawer.usdc"
    usdc_path.parent.mkdir(parents=True)
    usdc_path.write_bytes(b"USDC-drawer")
    sim = FakeSimulationManager()
    position = [1.0, 2.0, 3.0]
    angles = [90.0, 0.0, 45.0]
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler("XYZ", angles, degrees=True).as_matrix()
    pose[:3, 3] = position
    entry = {
        "uid": "drawer",
        "fpath": "articulated_assets/drawer/drawer.usdc",
        "init_pos": position,
        "init_rot": angles,
        "init_local_pose": pose.tolist(),
        "body_scale": [1.25, 2.5, 3.75],
        "fix_base": True,
    }

    _add_articulations(
        sim=sim,  # type: ignore[arg-type]
        entries=[entry],
        config_dir=tmp_path,
    )
    deployment = _physical_scene(
        tmp_path,
        {"format": "embodichain.scene-export/v1", "articulation": [entry]},
        tmp_path / "assets",
    )
    gym_cfg = ArticulationCfg.from_dict(deployment["articulation"][0])

    (preview_cfg,) = sim.articulation_cfgs
    assert isinstance(preview_cfg, ArticulationCfg)
    assert preview_cfg.uid == gym_cfg.uid == "drawer"
    assert preview_cfg.fpath == gym_cfg.fpath == str(usdc_path)
    assert preview_cfg.body_scale == tuple(gym_cfg.body_scale) == (1.25, 2.5, 3.75)
    assert np.asarray(preview_cfg.init_local_pose) == pytest.approx(pose)
    assert gym_cfg.init_local_pose == pytest.approx(pose)


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

    stale_delivery = [
        output_root / "scene_package.zip",
        output_root / "scene_usd" / "scene.usda",
        output_root / "scene_usd" / "scene.usdz",
        output_root / "scene_usd" / "scene_usd_manifest.json",
    ]
    for path in stale_delivery:
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"previous scene")

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
    assert not any(path.exists() for path in stale_delivery)
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


def test_preview_converted_limits_reach_resolved_descriptor(tmp_path, monkeypatch):
    from dexsim.engine import JointType
    from dexsim.spawn import ArticulationDesc, LinkDesc, JointDesc
    from embodichain.gen_sim.scene_engine.pipeline.utils import scene_usd
    from embodichain.lab.sim.spawn.descriptors import configure_articulation_desc

    limits = {"hinge": [-0.25, 1.5]}
    monkeypatch.setattr(scene_usd, "_read_revolute_qpos_limits", lambda path: limits)
    descriptor = ArticulationDesc(
        name="door",
        links=[LinkDesc("base", "", np.eye(4)), LinkDesc("lid", "base", np.eye(4))],
        joints=[JointDesc("hinge", "base", "lid", JointType.REVOLUTE)],
        root_link_name="base",
    )

    class Simulation:
        def add_articulation(self, cfg):
            assert cfg.asset_physics_mode == "overlay"
            configure_articulation_desc(descriptor, cfg)
            return descriptor

    path = tmp_path / "door.usdc"
    path.write_bytes(b"asset")
    scene_usd._add_articulations(
        sim=Simulation(),
        entries=[
            {
                "uid": "door",
                "fpath": "door.usdc",
                "init_pos": [0, 0, 0],
                "init_rot": [0, 0, 0],
            }
        ],
        config_dir=tmp_path,
    )
    assert descriptor.joints[0].lower_limit == pytest.approx(-0.25)
    assert descriptor.joints[0].upper_limit == pytest.approx(1.5)
