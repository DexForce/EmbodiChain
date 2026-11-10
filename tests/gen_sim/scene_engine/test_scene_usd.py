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

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from pxr import Sdf, Usd, UsdGeom, UsdShade

from dexsim.kit.usd import parse_stage

from embodichain.gen_sim.scene_engine.pipeline.utils.scene_usd import (
    _bind_base_color_texture,
    _bind_gltf_material_textures,
    _copy_runtime_texture,
    _export_material_without_texture,
)


def _native_material() -> SimpleNamespace:
    return SimpleNamespace(
        get_name=lambda: "test_material",
        get_base_color=lambda: (1.0, 1.0, 1.0),
        get_emissive=lambda: (0.0, 0.0, 0.0),
        get_roughness=lambda: 0.6,
        get_ior=lambda: 1.5,
    )


@pytest.mark.parametrize("metallic", [0.0, 0.35, 1.0])
def test_scalar_material_export_preserves_metallic(metallic: float) -> None:
    stage = Usd.Stage.CreateInMemory()
    native = _native_material()
    native.get_metallic = lambda: metallic

    material = _export_material_without_texture(native, stage, Sdf.Path("/World"))

    shader = material.ComputeSurfaceSource()[0]
    assert shader.GetInput("metallic").Get() == pytest.approx(metallic)


@pytest.mark.parametrize("unavailable", ["missing", "none", "error"])
def test_scalar_material_export_tolerates_unavailable_metallic(
    unavailable: str,
) -> None:
    stage = Usd.Stage.CreateInMemory()
    native = _native_material()
    if unavailable == "none":
        native.get_metallic = lambda: None
    elif unavailable == "error":

        def _failed_getter() -> float:
            raise RuntimeError("The renderer cannot read this material parameter.")

        native.get_metallic = _failed_getter

    material = _export_material_without_texture(native, stage, Sdf.Path("/World"))

    shader = material.ComputeSurfaceSource()[0]
    assert shader.GetInput("roughness").Get() == pytest.approx(0.6)
    assert shader.GetInput("metallic").Get() is None


def _textured_triangle() -> tuple[Usd.Stage, UsdShade.Material]:
    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(root.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, "/World/Triangle")
    mesh.CreatePointsAttr().Set([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)])
    mesh.CreateFaceVertexCountsAttr().Set([3])
    mesh.CreateFaceVertexIndicesAttr().Set([0, 1, 2])
    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
    ).Set([(0.25, 0.25), (0.75, 0.25), (0.25, 0.75)])
    material = UsdShade.Material.Define(stage, "/World/Material")
    surface = UsdShade.Shader.Define(stage, "/World/Material/Surface")
    surface.CreateIdAttr("UsdPreviewSurface")
    material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    return stage, material


@pytest.mark.parametrize(
    ("channel", "surface_input", "output_name"),
    [
        ("base_color", "diffuseColor", "rgb"),
        ("metallic_roughness", "metallic", "b"),
        ("normal", "normal", "rgb"),
        ("occlusion", "occlusion", "r"),
        ("emissive", "emissiveColor", "rgb"),
    ],
)
def test_gltf_texture_orientation_survives_usd_import(
    tmp_path: Path, channel: str, surface_input: str, output_name: str
) -> None:
    # Four different corners expose a vertical inversion that a solid image hides.
    pixels = np.array(
        [[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 0]]],
        dtype=np.uint8,
    )
    texture_path = tmp_path / "corners.png"
    Image.fromarray(pixels).save(texture_path)
    stage, material = _textured_triangle()
    original_uvs = (
        stage.GetPrimAtPath("/World/Triangle").GetAttribute("primvars:st").Get()
    )
    # Re-exporting must not double-flip the image or modify shared mesh UVs.
    for _ in range(2):
        relative_texture = _copy_runtime_texture(
            source_texture=texture_path,
            texture_root=tmp_path / "textures",
            scene_usd_root=tmp_path,
            uid="triangle",
            flip_vertical=True,
        )
        _bind_gltf_material_textures(
            stage=stage,
            material=material,
            relative_textures={channel: str(tmp_path / relative_texture)},
        )

    # Check the portable USD graph, not unreleased DexSim material metadata.
    surface = material.ComputeSurfaceSource()[0]
    connected = surface.GetInput(surface_input).GetConnectedSource()
    assert connected[1] == output_name
    texture = UsdShade.Shader(connected[0].GetPrim())
    coordinate_source = texture.GetInput("st").GetConnectedSource()[0]
    reader = UsdShade.Shader(coordinate_source.GetPrim())
    assert reader.GetIdAttr().Get() == "UsdPrimvarReader_float2"
    assert reader.GetInput("varname").Get() == "st"
    if channel == "metallic_roughness":
        roughness = surface.GetInput("roughness").GetConnectedSource()
        assert roughness[0].GetPrim() == texture.GetPrim()
        assert roughness[1] == "g"
    parsed = parse_stage(stage)
    assert parsed.materials
    np.testing.assert_array_equal(
        parsed.mesh_objects[0].renders[0].uv_coords, np.asarray(original_uvs)
    )
    assert (
        stage.GetPrimAtPath("/World/Triangle").GetAttribute("primvars:st").Get()
        == original_uvs
    )
    # Sample all four glTF corners using USD's lower-left texture convention.
    gltf_uvs = np.array([(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)])
    x = np.floor(gltf_uvs[:, 0] * 2).astype(int)
    y = 1 - np.floor(gltf_uvs[:, 1] * 2).astype(int)
    with Image.open(texture.GetInput("file").Get().path) as exported:
        np.testing.assert_array_equal(np.asarray(exported)[y, x], pixels.reshape(-1, 3))
    with Image.open(texture_path) as original:
        np.testing.assert_array_equal(np.asarray(original), pixels)


def test_usd_source_texture_retains_native_orientation(tmp_path: Path) -> None:
    texture_path = tmp_path / "usd.png"
    Image.new("RGB", (2, 2), (255, 0, 0)).save(texture_path)
    stage, material = _textured_triangle()
    _bind_base_color_texture(
        stage=stage, material=material, relative_texture=str(texture_path)
    )

    parsed = parse_stage(stage)
    descriptor = next(iter(parsed.materials.values()))
    assert descriptor.base_color_map == str(texture_path)
    surface = material.ComputeSurfaceSource()[0]
    texture = UsdShade.Shader(
        surface.GetInput("diffuseColor").GetConnectedSource()[0].GetPrim()
    )
    reader = UsdShade.Shader(texture.GetInput("st").GetConnectedSource()[0].GetPrim())
    assert reader.GetIdAttr().Get() == "UsdPrimvarReader_float2"
    assert reader.GetInput("varname").Get() == "st"


def test_scalar_material_export_prefers_scene_linear_base_color():
    stage = Usd.Stage.CreateInMemory()
    native = _native_material()
    native.get_base_color = lambda: (0.5, 0.6, 0.7)
    linear_color = (0.21404114, 0.31854678, 0.44798842)
    native.get_base_color_linear = lambda: linear_color
    material = _export_material_without_texture(native, stage, Sdf.Path("/World"))
    shader = material.ComputeSurfaceSource()[0]
    np.testing.assert_allclose(shader.GetInput("diffuseColor").Get(), linear_color)


def test_scalar_material_export_keeps_legacy_base_color_getter():
    stage = Usd.Stage.CreateInMemory()
    native = _native_material()
    native.get_base_color = lambda: (0.1, 0.2, 0.3)
    material = _export_material_without_texture(native, stage, Sdf.Path("/World"))
    shader = material.ComputeSurfaceSource()[0]
    np.testing.assert_allclose(shader.GetInput("diffuseColor").Get(), (0.1, 0.2, 0.3))
