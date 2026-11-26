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

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from embodichain.gen_sim.scene_engine.pipeline.utils._articulation_orientation_review import (
    _write_preview_stage,
)
from embodichain.gen_sim.scene_engine.pipeline.utils._articulation_proxy_alignment import (
    align_articulation_proxy,
)
from embodichain.gen_sim.scene_engine.pipeline.utils._articulation_proxy_fit import (
    fit_articulation_proxy,
)
from embodichain.gen_sim.scene_engine.pipeline.utils._articulation_scale import (
    bake_articulation_scale,
)
from embodichain.lab.sim.scene.articulation_geometry import read_articulation_geometry
from embodichain.lab.sim.spawn.source_geometry import read_usd_source


def test_fit_inherits_proxy_yaw_and_centers_on_native_footprint() -> None:
    # Native model has a different up axis and an offset footprint. Its physical
    # bottom is also lower than the visible bottom.
    native_footprint = np.array([[1.0, 0.5, -2.0], [3.0, 2.0, -6.0]])
    native_collision = np.array([[1.0, -0.5, -2.0], [3.0, 2.0, -6.0]])
    fit = fit_articulation_proxy(
        proxy_standard_vertices=[[5.0, 3.0, 1.0], [6.0, 5.0, 2.0]],
        native_footprint_vertices=native_footprint,
        native_collision_vertices=native_collision,
        native_to_standard=[[1, 0, 0], [0, 0, -1], [0, 1, 0]],
        proxy_yaw_degrees=90.0,
        proxy_center_xy=[10.0, -4.0],
        support_z=0.75,
    )

    np.testing.assert_allclose(fit.standard_scale, [0.5, 0.5, 0.5])
    np.testing.assert_allclose(fit.fitted_footprint_size, [1.0, 2.0])
    np.testing.assert_allclose(
        fit.rotation_z_up, [[0, 0, 1], [1, 0, 0], [0, 1, 0]], atol=1e-8
    )
    np.testing.assert_allclose(fit.position_z_up, [12.0, -5.0, 1.001])
    collision_world = (
        native_collision @ fit.native_scale_transform.T
    ) @ fit.rotation_z_up.T + fit.position_z_up
    assert collision_world[:, 2].min() == pytest.approx(0.751)


def test_native_footprint_fits_each_standard_axis() -> None:
    native = [[0, 0, 0], [2, 4, 1]]
    fit = fit_articulation_proxy(
        proxy_standard_vertices=[[0, 0, 0], [2, 2, 1]],
        native_footprint_vertices=native,
        native_collision_vertices=native,
        native_to_standard=np.eye(3),
        proxy_yaw_degrees=0.0,
        proxy_center_xy=[0, 0],
        support_z=0.0,
    )
    np.testing.assert_allclose(fit.standard_scale, [1.0, 0.5, 0.5])
    np.testing.assert_allclose(fit.fitted_footprint_size, [2.0, 2.0])


def test_native_reflection_is_rejected() -> None:
    box = [[0, 0, 0], [1, 1, 1]]
    with pytest.raises(ValueError, match="proper 3×3 rotation"):
        fit_articulation_proxy(
            proxy_standard_vertices=box,
            native_footprint_vertices=box,
            native_collision_vertices=box,
            native_to_standard=np.diag([1, -1, 1]),
            proxy_yaw_degrees=0.0,
            proxy_center_xy=[0, 0],
            support_z=0.0,
        )


@pytest.mark.parametrize(
    ("reviewed_heading", "saved_heading", "heading", "orientation_source"),
    [(None, None, 0, "geometry"), (90, None, 90, "vision"), (None, 270, 270, "saved")],
)
def test_standard_frame_matching_inherits_world_yaw(
    reviewed_heading: int | None,
    saved_heading: int | None,
    heading: int,
    orientation_source: str,
) -> None:
    # Asymmetric footprint and off-center raised feature distinguish the two
    # front faces as well as the vertical axis.
    standard = np.array(
        [[x, y, z] for x in (0.0, 2.0) for y in (0.0, 1.0) for z in (0.0, 0.5)]
        + [[0.15, 0.3, 0.8], [0.6, 0.3, 0.8], [0.15, 0.7, 0.8]]
    )
    native_to_standard = Rotation.from_euler("y", 90, degrees=True).as_matrix()
    native = standard @ native_to_standard * 2 + [3, 0, 1]
    world_yaw = Rotation.from_euler("z", 37, degrees=True)
    position = np.array([0.5, 0.1, 0.8])
    world_proxy = world_yaw.apply(standard)
    result = align_articulation_proxy(
        proxy_oriented_vertices=world_proxy,
        proxy_world_rotation=world_yaw.as_matrix(),
        proxy_world_position=position,
        native_footprint_vertices=native,
        native_collision_vertices=native,
        native_root_up=native_to_standard.T @ [0.0, 0.0, 1.0],
        support_z=0.75,
        saved_heading_degrees=saved_heading,
        review_heading=lambda yaw, upright: (reviewed_heading, "reviewed features"),
    )
    assert result.near_equal_frames == (1 if orientation_source == "geometry" else None)
    assert result.proxy_yaw_degrees == pytest.approx(37)
    assert result.native_heading_degrees == heading
    assert result.orientation_source == orientation_source
    extra_heading = Rotation.from_euler("z", heading, degrees=True).as_matrix()
    np.testing.assert_allclose(
        result.fit.rotation_z_up,
        world_yaw.as_matrix() @ extra_heading @ native_to_standard,
        atol=1e-8,
    )
    scaled_native = native @ result.fit.native_scale_transform.T
    np.testing.assert_allclose(
        np.ptp(scaled_native @ result.native_to_standard.T, axis=0)[:2], [2.0, 1.0]
    )
    native_world = scaled_native @ result.fit.rotation_z_up.T
    native_world += result.fit.position_z_up
    assert native_world[:, 2].min() == pytest.approx(0.751)
    np.testing.assert_allclose(
        (native_world.min(axis=0) + native_world.max(axis=0))[:2] / 2,
        (world_proxy.min(axis=0) + world_proxy.max(axis=0))[:2] / 2 + position[:2],
    )


def test_geometry_matching_keeps_native_up_when_sideways_shape_scores_better() -> None:
    native = np.array(
        [[x, y, z] for x in (0.0, 3.0) for y in (0.0, 2.0) for z in (0.0, 1.0)]
    )
    sideways_proxy = Rotation.from_euler("x", 90, degrees=True).apply(native)
    result = align_articulation_proxy(
        proxy_oriented_vertices=sideways_proxy,
        proxy_world_rotation=np.eye(3),
        proxy_world_position=np.zeros(3),
        native_footprint_vertices=native,
        native_collision_vertices=native,
        native_root_up=np.array([0.0, 0.0, 1.0]),
        support_z=0.75,
    )
    np.testing.assert_allclose(result.fit.rotation_z_up[:, 2], [0.0, 0.0, 1.0])
    np.testing.assert_allclose(result.fit.fitted_footprint_size, [3.0, 1.0])


def test_baked_scale_preserves_off_origin_hinge_during_motion(tmp_path: Path) -> None:
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    source_path = tmp_path / "original.usdc"
    stage = Usd.Stage.CreateNew(str(source_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/object").GetPrim()
    stage.SetDefaultPrim(root)
    UsdPhysics.ArticulationRootAPI.Apply(root)
    for name, width in (("base", 2.0), ("door", 1.0)):
        link = UsdGeom.Xform.Define(stage, f"/object/{name}")
        UsdPhysics.RigidBodyAPI.Apply(link.GetPrim())
        mesh = UsdGeom.Mesh.Define(stage, f"/object/{name}/collision")
        mesh.CreatePointsAttr([(0, 0, 0), (width, 0, 0), (0, 1, 0), (0, 0, 0.25)])
        mesh.CreateFaceVertexCountsAttr([3, 3, 3, 3])
        mesh.CreateFaceVertexIndicesAttr([0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3])
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    door = UsdGeom.Xformable.Get(stage, "/object/door")
    door.AddTranslateOp().Set(Gf.Vec3d(2, 1, 0))
    door.AddRotateZOp().Set(90)
    visual = UsdGeom.Mesh.Define(stage, "/object/door/visual")
    visual.CreatePointsAttr([(1, 0, 0), (0, 0, 0), (0, 1, 0)])
    visual.CreateFaceVertexCountsAttr([3])
    visual.CreateFaceVertexIndicesAttr([0, 1, 2])
    hinge = UsdPhysics.RevoluteJoint.Define(stage, "/object/hinge")
    hinge.CreateBody0Rel().SetTargets(["/object/base"])
    hinge.CreateBody1Rel().SetTargets(["/object/door"])
    hinge.CreateAxisAttr("Z")
    hinge.CreateLocalPos0Attr(Gf.Vec3f(1, 1, 0))
    hinge.CreateLocalPos1Attr(Gf.Vec3f(0, 1, 0))
    hinge.CreateLocalRot1Attr(Gf.Quatf(2**-0.5, Gf.Vec3f(0, 0, -(2**-0.5))))
    hinge.CreateLowerLimitAttr(-90)
    hinge.CreateUpperLimitAttr(120)
    stage.GetRootLayer().Save()
    original_bytes = source_path.read_bytes()

    baked = bake_articulation_scale(
        source_path, tmp_path / "baked.usdc", native_scale_matrix=np.diag([2, 3, 1.5])
    )
    initial = read_articulation_geometry(baked, reset_to_default=True)
    np.testing.assert_allclose(np.ptp(initial.vertices, axis=0)[:2], [4, 6], atol=1e-6)
    np.testing.assert_allclose(initial.link_vertices["door"][1], [4, 6, 0], atol=1e-6)
    moved_stage = Usd.Stage.Open(str(baked))
    baked_visual = UsdGeom.Mesh.Get(moved_stage, "/object/door/visual")
    visual_world = UsdGeom.XformCache().GetLocalToWorldTransform(baked_visual.GetPrim())
    np.testing.assert_allclose(
        visual_world.Transform(Gf.Vec3d(baked_visual.GetPointsAttr().Get()[0])),
        [4, 6, 0],
        atol=1e-6,
    )
    moved_hinge = UsdPhysics.RevoluteJoint.Get(moved_stage, "/object/hinge")
    np.testing.assert_allclose(
        moved_hinge.GetLocalPos0Attr().Get(), [2, 3, 0], atol=1e-6
    )
    np.testing.assert_allclose(
        moved_hinge.GetLocalPos1Attr().Get(), [0, 2, 0], atol=1e-6
    )

    # A positive lower bound evaluates the same exported mechanism at q=90°.
    moved_hinge.GetLowerLimitAttr().Set(90)
    moved_stage.GetRootLayer().Save()
    opened = read_articulation_geometry(baked, reset_to_default=True)
    np.testing.assert_allclose(opened.link_vertices["door"][1], [-1, 5, 0], atol=1e-6)
    np.testing.assert_allclose(opened.link_vertices["door"][2], [2, 3, 0], atol=1e-6)
    assert source_path.read_bytes() == original_bytes


def test_preview_keeps_visible_collider_material_and_relative_texture(
    tmp_path: Path,
) -> None:
    from PIL import Image
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    texture_path = source_dir / "albedo.png"
    Image.new("RGB", (2, 2), "#df472a").save(texture_path)
    source_path = source_dir / "source.usdc"
    stage = Usd.Stage.CreateNew(str(source_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Part").GetPrim()
    stage.SetDefaultPrim(root)
    UsdPhysics.ArticulationRootAPI.Apply(root)
    body = UsdGeom.Xform.Define(stage, "/Part/Body")
    body.AddTranslateOp().Set(Gf.Vec3d(4, 2, 1))
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, "/Part/Body/Shell")
    mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 1)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
    )
    st.Set([(0, 0), (1, 0), (0, 1)])
    material = UsdShade.Material.Define(stage, "/Part/Paint")
    surface = UsdShade.Shader.Define(stage, "/Part/Paint/Preview")
    surface.CreateIdAttr("UsdPreviewSurface")
    material.CreateSurfaceOutput().ConnectToSource(
        surface.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    )
    texture = UsdShade.Shader.Define(stage, "/Part/Paint/Albedo")
    texture.CreateIdAttr("UsdUVTexture")
    texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("albedo.png")
    )
    surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    )
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    stage.GetRootLayer().Save()
    source = read_usd_source(source_path)

    preview_path = tmp_path / "preview.usda"
    _write_preview_stage(
        source, read_articulation_geometry(source, reset_to_default=True), preview_path
    )
    preview = Usd.Stage.Open(str(preview_path))
    rendered = UsdGeom.Mesh.Get(preview, "/Part/Body/Shell")
    assert rendered.ComputeVisibility() == UsdGeom.Tokens.inherited
    assert UsdPhysics.CollisionAPI(rendered.GetPrim())
    np.testing.assert_allclose(
        UsdGeom.PrimvarsAPI(rendered).GetPrimvar("st").Get(), [(0, 0), (1, 0), (0, 1)]
    )
    bound, _ = UsdShade.MaterialBindingAPI(rendered).ComputeBoundMaterial()
    assert bound.GetPath() == Sdf.Path("/Part/Paint")
    composed_texture = UsdShade.Shader.Get(preview, "/Part/Paint/Albedo")
    assert Path(composed_texture.GetInput("file").Get().resolvedPath) == texture_path
    np.testing.assert_allclose(
        UsdGeom.XformCache().GetLocalToWorldTransform(rendered.GetPrim()), np.eye(4)
    )
    assert source_path.read_bytes() == source.raw
