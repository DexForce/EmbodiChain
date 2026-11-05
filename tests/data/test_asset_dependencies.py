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
"""Local input inventories preserve URI resolution and source bytes."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from embodichain.data.asset_dependencies import (
    asset_dependency_files,
    collect_declared_asset_files,
)


def test_xml_dependency_closure_retains_local_uri_and_package_resolution(tmp_path):
    package = tmp_path / "fixture_package"
    includes = package / "includes"
    includes.mkdir(parents=True)
    model = package / "model.urdf"
    mesh = package / "mesh input.stl"
    texture = package / "texture.png"
    child = includes / "child.xml"
    mesh.write_bytes(b"original opaque mesh input")
    texture.write_bytes(b"original texture input")
    child.write_text('<mujoco><include file="../model.urdf"/></mujoco>')
    model.write_text(
        '<robot xmlns:x="urn:fixture">'
        '<x:mesh filename="package://fixture_package/mesh%20input.stl"/>'
        '<mesh filename="mesh%20input.stl"/>'
        f'<texture file="{texture.as_uri()}"/>'
        '<include url="includes/child.xml"/>'
        "</robot>"
    )
    expected = {model, mesh, texture, child}
    original = {path: path.read_bytes() for path in expected}
    assert asset_dependency_files(model) == tuple(sorted(expected))
    assert {path: path.read_bytes() for path in expected} == original


def test_declaration_inventory_uses_owner_local_paths_before_registry(
    tmp_path, monkeypatch
):
    import embodichain.data.asset_dependencies as owner

    local = tmp_path / "local"
    local.mkdir()
    source = local / "model.urdf"
    source.write_text("<robot/>")
    registered = tmp_path / "registered.urdf"
    registered.write_text("<robot/>")
    requests = []

    def lookup(reference):
        requests.append(reference)
        assert reference == "PinnedAsset/model.urdf"
        return str(registered)

    monkeypatch.setattr(owner, "get_data_path", lookup)
    declarations = {
        "scene": [{"fpath": "local/model.urdf"}, {"urdf_path": str(source)}],
        "robot": (
            {"xml_path": "PinnedAsset/model.urdf"},
            {"mjcf_path": str(registered)},
        ),
        "metadata": {"description": "not a file reference", "fpath": None},
    }
    before = deepcopy(declarations)
    assert collect_declared_asset_files(declarations, base_dir=tmp_path) == {
        source,
        registered,
    }
    assert requests == ["PinnedAsset/model.urdf"]
    assert declarations == before


def test_usd_inventory_retains_used_layers_and_adjacent_source_annotation(tmp_path):
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom

    child = tmp_path / "child.usda"
    root = tmp_path / "model.usda"
    stage = Usd.Stage.CreateNew(str(child))
    UsdGeom.Xform.Define(stage, "/Child")
    stage.GetRootLayer().Save()
    stage = Usd.Stage.CreateNew(str(root))
    stage.GetRootLayer().subLayerPaths = [child.name]
    UsdGeom.Xform.Define(stage, "/Model")
    stage.GetRootLayer().Save()
    annotation = root.with_suffix(".twist.json")
    annotation.write_text('{"authored": "retained without semantic evaluation"}')
    before = {p: p.read_bytes() for p in (root, child, annotation)}
    assert asset_dependency_files(root) == tuple(sorted((root, child, annotation)))
    assert collect_declared_asset_files({"fpath": root.name}, base_dir=tmp_path) == {
        root,
        child,
        annotation,
    }
    assert {p: p.read_bytes() for p in before} == before


@pytest.mark.parametrize(
    "reference,error",
    [
        ("missing.stl", FileNotFoundError),
        ("https://example.invalid/model.stl", ValueError),
        ("ftp://example.invalid/model.stl", ValueError),
    ],
)
def test_declared_dependency_failures_remain_explicit(tmp_path, reference, error):
    model = tmp_path / "model.urdf"
    model.write_text(f'<robot><mesh filename="{reference}"/></robot>')
    with pytest.raises(error):
        asset_dependency_files(model)


def test_opaque_xml_looking_source_remains_a_direct_dependency(tmp_path):
    source = tmp_path / "opaque.urdf"
    source.write_bytes(b"opaque original source")
    assert asset_dependency_files(source) == (source,)


@pytest.mark.parametrize("binary", [False, True])
def test_gltf_inventory_covers_actual_reader_external_buffers_and_images(
    tmp_path, binary
):
    import json, struct
    import numpy as np
    import trimesh
    from PIL import Image

    data = (
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype="<f4").tobytes()
        + np.array([0, 1, 2], dtype="<u2").tobytes()
    )
    buffer = tmp_path / "geometry.bin"
    buffer.write_bytes(data)
    image = tmp_path / "albedo.png"
    Image.new("RGB", (1, 1), (17, 31, 53)).save(image)
    document = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": len(data), "uri": buffer.name}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": 36},
            {"buffer": 0, "byteOffset": 36, "byteLength": 6},
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": 3,
                "type": "VEC3",
                "min": [0, 0, 0],
                "max": [1, 1, 0],
            },
            {"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"},
        ],
        "images": [{"uri": image.name}],
        "textures": [{"source": 0}],
        "materials": [{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}],
        "meshes": [
            {
                "primitives": [
                    {"attributes": {"POSITION": 0}, "indices": 1, "material": 0}
                ]
            }
        ],
        "nodes": [{"mesh": 0}],
        "scenes": [{"nodes": [0]}],
        "scene": 0,
    }
    path = tmp_path / ("model.glb" if binary else "model.gltf")
    if binary:
        header = json.dumps(document).encode()
        header += b" " * ((-len(header)) % 4)
        path.write_bytes(
            struct.pack(
                "<4sIIII", b"glTF", 2, 20 + len(header), len(header), 0x4E4F534A
            )
            + header
        )
    else:
        path.write_text(json.dumps(document))
    expected = {path, buffer, image}
    before = {p: p.read_bytes() for p in expected}
    loaded = trimesh.load(str(path), force="scene")
    geometry = next(iter(loaded.geometry.values()))
    assert len(
        geometry.vertices
    ) == 3 and geometry.visual.material.baseColorTexture.getpixel((0, 0))[:3] == (
        17,
        31,
        53,
    )
    assert set(asset_dependency_files(path)) == expected
    assert (
        collect_declared_asset_files({"shape": {"fpath": str(path)}}, base_dir=tmp_path)
        == expected
    )
    assert {p: p.read_bytes() for p in expected} == before
    image.unlink()
    with pytest.raises(FileNotFoundError):
        asset_dependency_files(path)


def test_obj_inventory_covers_material_and_actual_texture_reader(tmp_path):
    import trimesh
    from PIL import Image

    image = tmp_path / "albedo.png"
    Image.new("RGB", (1, 1), (17, 31, 53)).save(image)
    material = tmp_path / "material.mtl"
    material.write_text("newmtl surface\nKd 1 1 1\nmap_Kd albedo.png\n")
    path = tmp_path / "model.obj"
    path.write_text(
        "mtllib material.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 0 1\nusemtl surface\nf 1/1 2/2 3/3\n"
    )
    loaded = trimesh.load(str(path), force="scene")
    assert next(iter(loaded.geometry.values())).visual.material.image.getpixel((0, 0))[
        :3
    ] == (17, 31, 53)
    expected = {path, material, image}
    assert set(asset_dependency_files(path)) == expected
    original = image.read_bytes()
    image.write_bytes(original + b"changed")
    import hashlib

    assert (
        hashlib.sha256(image.read_bytes()).hexdigest()
        != hashlib.sha256(original).hexdigest()
    )
    image.unlink()
    with pytest.raises(FileNotFoundError):
        asset_dependency_files(path)


def test_mesh_dependency_metadata_retains_embedded_data_and_uri_rules(tmp_path):
    import json

    mesh = tmp_path / "mesh data.bin"
    mesh.write_bytes(b"original")
    path = tmp_path / "model.gltf"
    path.write_text(
        json.dumps(
            {
                "asset": {"version": "2.0"},
                "buffers": [
                    {"uri": "mesh%20data.bin"},
                    {"uri": "data:application/octet-stream;base64,YQ=="},
                ],
                "images": [{"uri": "data:image/png;base64,YQ=="}],
            }
        )
    )
    assert asset_dependency_files(path) == tuple(sorted([path, mesh]))
    path.write_text(
        json.dumps({"images": [{"uri": "https://example.invalid/image.png"}]})
    )
    with pytest.raises(ValueError, match="Remote"):
        asset_dependency_files(path)


def test_obj_material_dependency_options_do_not_hide_texture_paths(tmp_path):
    material = tmp_path / "material.mtl"
    image = tmp_path / "normal map.png"
    image.write_bytes(b"original")
    material.write_text('newmtl surface\nmap_bump -bm .2 -s 1 1 1 "normal map.png"\n')
    path = tmp_path / "model.obj"
    path.write_text("mtllib material.mtl\n")
    assert set(asset_dependency_files(path)) == {path, material, image}
    material.write_text("newmtl surface\nmap_Kd -unknown x normal.png\n")
    with pytest.raises(ValueError, match="Unsupported MTL"):
        asset_dependency_files(path)


def test_model_owner_can_bind_exact_existing_archive_reference_without_guessing(
    tmp_path,
):
    original = tmp_path / "model.urdf"
    mesh = tmp_path / "relocated.obj"
    mesh.write_text("v 0 0 0\n")
    reference = "../legacy/Collision/mesh.obj"
    original.write_text(f'<robot><mesh filename="{reference}"/></robot>')
    with pytest.raises(FileNotFoundError):
        asset_dependency_files(original)
    assert set(
        asset_dependency_files(original, reference_overrides={reference: mesh})
    ) == {original, mesh}
    with pytest.raises(ValueError, match="absolute paths"):
        asset_dependency_files(
            original, reference_overrides={reference: "relative.obj"}
        )
    mesh.unlink()
    with pytest.raises(FileNotFoundError):
        asset_dependency_files(original, reference_overrides={reference: mesh})


def test_usd_inventory_includes_composed_layer_asset_properties_and_missing_textures(
    tmp_path,
):
    from pxr import Sdf, Usd, UsdGeom, UsdShade

    texture = tmp_path / "albedo.png"
    texture.write_bytes(b"original texture")
    child = tmp_path / "material.usda"
    stage = Usd.Stage.CreateNew(str(child))
    shader = UsdShade.Shader.Define(stage, "/Material/Texture")
    shader.CreateIdAttr("UsdUVTexture")
    shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath(texture.name)
    )
    stage.GetRootLayer().Save()
    root = tmp_path / "model.usda"
    stage = Usd.Stage.CreateNew(str(root))
    stage.GetRootLayer().subLayerPaths = [child.name]
    UsdGeom.Xform.Define(stage, "/Model")
    stage.GetRootLayer().Save()
    before = {p: p.read_bytes() for p in [root, child, texture]}
    expected = {root, child, texture}
    assert set(asset_dependency_files(root)) == expected
    assert (
        collect_declared_asset_files({"fpath": str(root)}, base_dir=tmp_path)
        == expected
    )
    assert {p: p.read_bytes() for p in expected} == before
    texture.unlink()
    with pytest.raises(FileNotFoundError, match="unresolved"):
        asset_dependency_files(root)


def test_usdz_inventory_hashes_the_outer_package_for_contained_assets(tmp_path):
    from pxr import Sdf, Usd, UsdGeom, UsdShade, UsdUtils
    from PIL import Image

    texture = tmp_path / "albedo.png"
    Image.new("RGB", (1, 1), (17, 31, 53)).save(texture)
    root = tmp_path / "model.usda"
    stage = Usd.Stage.CreateNew(str(root))
    prim = UsdGeom.Xform.Define(stage, "/Model")
    stage.SetDefaultPrim(prim.GetPrim())
    shader = UsdShade.Shader.Define(stage, "/Material/Texture")
    shader.CreateIdAttr("UsdUVTexture")
    shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath(texture.name)
    )
    stage.GetRootLayer().Save()
    package = tmp_path / "model.usdz"
    assert UsdUtils.CreateNewUsdzPackage(Sdf.AssetPath(str(root)), str(package))
    assert asset_dependency_files(package) == (package,)


def test_explicit_urdf_collision_consumer_does_not_read_visual_or_other_links(tmp_path):
    mesh = tmp_path / "selected.obj"
    mesh.write_text("v 0 0 0\n")
    source = tmp_path / "model.urdf"
    source.write_text(
        '<robot><link name="selected"><visual><geometry><mesh filename="missing_visual.dae"/></geometry></visual><collision><geometry><mesh filename="selected.obj"/></geometry></collision></link><link name="unused"><collision><geometry><mesh filename="missing_other.stl"/></geometry></collision></link></robot>'
    )
    assert set(asset_dependency_files(source, urdf_collision_links=["selected"])) == {
        source,
        mesh,
    }
    with pytest.raises(FileNotFoundError):
        asset_dependency_files(source)
    with pytest.raises(ValueError, match="exact existing links"):
        asset_dependency_files(source, urdf_collision_links=["missing"])
