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
    ("channel", "map_name"),
    [
        ("base_color", "base_color_map"),
        ("metallic_roughness", "metallic_roughness_map"),
        ("normal", "normal_map"),
        ("occlusion", "ao_map"),
        ("emissive", "emissive_map"),
    ],
)
def test_gltf_texture_orientation_survives_usd_import(
    tmp_path: Path, channel: str, map_name: str
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
    # Rebinding must not apply a second conversion or modify shared mesh UVs.
    for _ in range(2):
        _bind_gltf_material_textures(
            stage=stage,
            material=material,
            relative_textures={channel: str(texture_path)},
        )

    parsed = parse_stage(stage)
    descriptor = next(iter(parsed.materials.values()))
    settings = descriptor.texture_settings[map_name]
    assert settings.uv_set == "st"
    assert descriptor.import_warnings == []
    assert (
        stage.GetPrimAtPath("/World/Triangle").GetAttribute("primvars:st").Get()
        == original_uvs
    )
    # Sample all four glTF corners using USD's lower-left texture convention.
    gltf_uvs = np.array([(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)])
    usd_uvs = gltf_uvs * settings.uv_scale + settings.uv_translation
    x = np.floor(usd_uvs[:, 0] * 2).astype(int)
    y = 1 - np.floor(usd_uvs[:, 1] * 2).astype(int)
    np.testing.assert_array_equal(pixels[y, x], pixels.reshape(-1, 3))


def test_usd_source_texture_retains_native_orientation(tmp_path: Path) -> None:
    texture_path = tmp_path / "usd.png"
    Image.new("RGB", (2, 2), (255, 0, 0)).save(texture_path)
    stage, material = _textured_triangle()
    _bind_base_color_texture(
        stage=stage, material=material, relative_texture=str(texture_path)
    )

    parsed = parse_stage(stage)
    descriptor = next(iter(parsed.materials.values()))
    settings = descriptor.texture_settings["base_color_map"]
    assert settings.uv_scale == (1.0, 1.0)
    assert settings.uv_translation == (0.0, 0.0)
