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

import numpy as np
import pytest
import json
from pathlib import Path
from scipy.spatial.transform import Rotation

from embodichain.lab.visualization.markers import (
    MarkerGroup,
    MarkerPrototypeCfg,
    RobotMarkers,
    RobotMarkersCfg,
    RobotMarkersAppearanceCfg,
)


class _Sim:
    def add_marker_group(self, cfg):
        self.group = MarkerGroup(cfg, num_envs=2, origins=[[0, 0, 0], [10, 0, 0]])
        return self.group


@pytest.fixture
def robot_path(tmp_path):
    path = tmp_path / "robot.urdf"
    path.write_text("""<robot name="fixture">
      <material name="red"><color rgba="1 0 0 0.8"/></material>
      <link name="base"><visual><origin xyz="1 0 0" rpy="0 0 1.5707963267948966"/>
        <geometry><box size="2 3 4"/></geometry><material name="red"/></visual>
        <visual><geometry><sphere radius="0.25"/></geometry></visual></link>
      <link name="tip"><visual><origin xyz="0 0 0.5"/>
        <geometry><cylinder radius="0.1" length="0.4"/></geometry></visual></link>
      <joint name="hinge" type="revolute"><parent link="base"/><child link="tip"/>
        <origin xyz="0 0 1"/><axis xyz="0 0 1"/><limit lower="-3" upper="3" effort="1" velocity="1"/></joint>
    </robot>""")
    return path


def _ghost(path, mode="asset", space="env"):
    sim = _Sim()
    ghost = RobotMarkers(
        RobotMarkersCfg(
            name="ghost",
            asset_path=str(path),
            coordinate_space=space,
            appearance=RobotMarkersAppearanceCfg(mode=mode, opacity=0.5),
        ),
        sim,
    )
    return ghost, sim.group


def test_visual_origin_scale_material_and_partial_environments(robot_path):
    ghost, group = _ghost(robot_path)
    assert ghost.link_names == ("base", "tip") and ghost.num_visuals == 3
    poses = np.tile(np.eye(4), (2, 2, 1, 1))
    poses[0, 0, :3, :3] = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    ghost.visualize_link_poses(poses)
    snapshots = group.snapshot()
    np.testing.assert_allclose(snapshots[0].position, [0, 1, 0], atol=1e-7)
    np.testing.assert_allclose(snapshots[0].scale, [2, 3, 4])
    np.testing.assert_allclose(snapshots[0].color, [1, 0, 0, 0.4])
    np.testing.assert_allclose(snapshots[3].position, [11, 0, 0])
    assert abs(snapshots[0].wxyz[3]) == pytest.approx(1)
    ghost.visualize_link_poses(np.tile(np.eye(4), (1, 2, 1, 1)), env_ids=[0])
    np.testing.assert_allclose(group.snapshot()[3].position, [11, 0, 0])


def test_world_poses_are_not_offset_twice(robot_path):
    ghost, group = _ghost(robot_path, space="world")
    ghost.visualize_link_poses(np.tile(np.eye(4), (2, 2, 1, 1)))
    np.testing.assert_allclose(group.snapshot()[3].position, [1, 0, 0])


def test_appearance_toggle_preserves_source_and_geometry(robot_path):
    ghost, group = _ghost(robot_path)
    ghost.visualize_qpos([[0], [0]])
    original = tuple(ghost._visuals[0].prototype.color)
    ghost.set_appearance(
        RobotMarkersAppearanceCfg(mode="override", color=(0, 1, 0), opacity=0)
    )
    np.testing.assert_allclose(group.snapshot()[0].color, [0, 1, 0, 0])
    ghost.set_appearance(RobotMarkersAppearanceCfg(mode="asset", opacity=1))
    np.testing.assert_allclose(group.snapshot()[0].color, original)
    assert ghost._visuals[0].prototype.color == original
    ghost.set_visibility(False, env_ids=[1])
    assert all(not item.visible for item in group.snapshot()[3:])
    assert all(item.visible for item in group.snapshot()[:3])
    ghost.remove()
    ghost.remove()
    assert group.snapshot() == ()


def test_ghost_lighting_survives_color_and_opacity_changes(robot_path):
    ghost, group = _ghost(robot_path)
    ghost.visualize_qpos([[0], [0]])
    assert all(not snapshot.unlit for snapshot in group.snapshot())
    ghost.set_appearance(RobotMarkersAppearanceCfg(mode="override", opacity=1))
    assert all(not snapshot.unlit for snapshot in group.snapshot())
    ghost.set_appearance(RobotMarkersAppearanceCfg(mode="asset", unlit=True))
    assert all(snapshot.unlit for snapshot in group.snapshot())
    ghost.set_appearance(RobotMarkersAppearanceCfg(mode="asset", opacity=0.2))
    assert all(not snapshot.unlit for snapshot in group.snapshot())


def test_reference_joint_fk_uses_named_link_order(robot_path):
    ghost, group = _ghost(robot_path)
    assert ghost.joint_names == ("hinge",)
    roots = np.tile([0, 0, 0, 0, 0, 0, 1], (2, 1)).astype(float)
    roots[:, 0] = 2
    ghost.visualize_qpos([[0], [np.pi / 2]], root_poses=roots)
    snapshots = group.snapshot()
    np.testing.assert_allclose(snapshots[2].position, [2, 0, 1.5])
    np.testing.assert_allclose(snapshots[5].position, [12, 0, 1.5])
    np.testing.assert_allclose(
        snapshots[5].wxyz, [np.sqrt(0.5), 0, 0, np.sqrt(0.5)], atol=1e-6
    )


@pytest.mark.parametrize(
    "appearance",
    [
        RobotMarkersAppearanceCfg(mode="bad"),
        RobotMarkersAppearanceCfg(opacity=-1),
        RobotMarkersAppearanceCfg(opacity=float("nan")),
        RobotMarkersAppearanceCfg(color=(1, 2, 3)),
    ],
)
def test_invalid_appearance_keeps_previous_state(robot_path, appearance):
    ghost, group = _ghost(robot_path)
    ghost.visualize_qpos([[0], [0]])
    before = group.snapshot()[0].color
    with pytest.raises(ValueError):
        ghost.set_appearance(appearance)
    assert group.snapshot()[0].color == before


def test_invalid_pose_keeps_previous_state(robot_path):
    ghost, group = _ghost(robot_path)
    ghost.visualize_qpos([[0], [0]])
    poses = np.tile(np.eye(4), (2, 2, 1, 1))
    poses[0, 0, 0, 0] = 2
    with pytest.raises(ValueError, match="rigid"):
        ghost.visualize_link_poses(poses)
    np.testing.assert_allclose(group.snapshot()[0].position, [1, 0, 0])


def test_missing_mesh_reports_source_error(tmp_path):
    path = tmp_path / "broken.urdf"
    path.write_text(
        '<robot name="broken"><link name="link"><visual><geometry><mesh filename="missing.stl"/></geometry></visual></link></robot>'
    )
    with pytest.raises(FileNotFoundError, match="missing.stl"):
        _ghost(path)


def test_empty_environment_pose_update_is_a_noop(robot_path):
    ghost, group = _ghost(robot_path)
    ghost.visualize_qpos([[0], [0]])
    before = group.snapshot()
    ghost.visualize_link_poses(np.empty((0, 2, 7)), env_ids=[])
    np.testing.assert_array_equal(group.snapshot()[0].position, before[0].position)


def test_multiple_references_share_prototypes_and_resize_selected_environments(
    robot_path,
):
    ghost, group = _ghost(robot_path)
    roots = np.tile(np.eye(4), (2, 3, 1, 1))
    roots[:, :, 1, 3] = [0, 2, 4]
    ghost.visualize_qpos(np.zeros((2, 3, 1)), root_poses=roots)
    assert group.counts == (9, 9)
    snapshots = group.snapshot()
    assert snapshots[0].vertices is snapshots[3].vertices
    np.testing.assert_allclose(snapshots[8].position, [0, 4, 1.5])
    ghost.visualize_qpos(np.zeros((1, 2, 1)), env_ids=[1])
    assert group.counts == (9, 6)
    ghost.visualize_link_poses(np.empty((1, 0, 2, 7)), env_ids=[0])
    assert group.counts == (0, 6)


def test_urdf_texture_and_mesh_scale_survive_appearance_switch(tmp_path):
    from PIL import Image
    import trimesh

    pixels = np.array(
        [
            [[255, 0, 0, 255], [0, 255, 0, 255]],
            [[0, 0, 255, 255], [255, 255, 255, 128]],
        ],
        dtype=np.uint8,
    )
    mesh = trimesh.Trimesh(
        vertices=[[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]],
        faces=[[0, 1, 2], [0, 2, 3]],
        process=False,
        visual=trimesh.visual.TextureVisuals(
            uv=[[0, 0], [1, 0], [1, 1], [0, 1]],
            material=trimesh.visual.material.PBRMaterial(
                baseColorTexture=Image.fromarray(pixels),
                baseColorFactor=[255, 255, 255, 255],
                metallicFactor=0.65,
                roughnessFactor=0.27,
                normalTexture=Image.fromarray(
                    np.full((2, 2, 3), [128, 128, 255], dtype=np.uint8)
                ),
            ),
        ),
    )
    mesh.export(tmp_path / "mesh.glb")
    path = tmp_path / "textured.urdf"
    path.write_text(
        '<robot name="texture"><link name="base"><visual>'
        '<geometry><mesh filename="mesh.glb" scale="2 3 4"/></geometry>'
        "</visual></link></robot>"
    )
    ghost, group = _ghost(path)
    ghost.visualize_link_poses(np.tile(np.eye(4), (2, 1, 1, 1)))
    snapshot = group.snapshot()[0]
    np.testing.assert_array_equal(snapshot.texture, pixels)
    np.testing.assert_allclose(snapshot.uv_coords, [[0, 0], [1, 0], [1, 1], [0, 1]])
    np.testing.assert_allclose(snapshot.scale, [2, 3, 4])
    before = json.loads(snapshot.material)
    assert before["metallic"] == pytest.approx(0.65)
    assert before["roughness"] == pytest.approx(0.27)
    assert Path(before["normal_map"]).is_file()
    ghost.set_appearance(RobotMarkersAppearanceCfg(mode="override", opacity=1))
    tinted = group.snapshot()[0]
    assert json.loads(tinted.material) == before
    np.testing.assert_allclose(tinted.normals, snapshot.normals)
    np.testing.assert_array_equal(tinted.texture, pixels)
    ghost.set_appearance(RobotMarkersAppearanceCfg(mode="asset"))
    np.testing.assert_array_equal(group.snapshot()[0].texture, pixels)


def test_vertex_color_asset_is_preserved_as_texture(tmp_path):
    import trimesh
    from embodichain.lab.visualization.markers._robot_marker_assets import _mesh_parts

    mesh = trimesh.creation.box()
    mesh.visual.vertex_colors = np.tile([255, 0, 0, 255], (len(mesh.vertices), 1))
    path = tmp_path / "colored.glb"
    mesh.export(path)
    (prototype,) = _mesh_parts(path)
    np.testing.assert_allclose(prototype.color, [1, 1, 1, 1])
    np.testing.assert_array_equal(prototype.texture[0, 0], [255, 0, 0, 255])
    assert prototype.uv_coords.shape == (len(prototype.vertices), 2)


def test_gltf_linear_color_factor_is_encoded_for_marker_protocol():
    import trimesh
    from embodichain.lab.visualization.markers._robot_marker_assets import (
        _mesh_prototype,
    )

    mesh = trimesh.creation.box()
    mesh.visual = trimesh.visual.TextureVisuals(
        material=trimesh.visual.material.PBRMaterial(baseColorFactor=[64, 64, 64, 128])
    )
    prototype = _mesh_prototype(mesh)
    np.testing.assert_allclose(
        prototype.color, [0.5380651] * 3 + [128 / 255], atol=1e-6
    )


@pytest.mark.parametrize("packaged", [False, True])
def test_usd_visual_origin_material_and_packaged_texture(tmp_path, packaged):
    from PIL import Image
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, UsdUtils

    pixels = np.full((2, 2, 4), [255, 0, 0, 255], dtype=np.uint8)
    Image.fromarray(pixels).save(tmp_path / "texture.png")
    path = tmp_path / "visual.usda"
    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/Visual")
    root.GetPrim().CreateAttribute("MeshObject", Sdf.ValueTypeNames.Token)
    stage.SetDefaultPrim(root.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, "/Visual/mesh")
    mesh.CreatePointsAttr([(-1, -1, 0), (1, -1, 0), (0, 1, 0)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    mesh.AddTranslateOp().Set(Gf.Vec3d(1, 2, 3))
    mesh.AddScaleOp().Set(Gf.Vec3f(2, 3, 4))
    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
    ).Set([(0, 0), (1, 0), (0.5, 1)])
    material = UsdShade.Material.Define(stage, "/Material")
    shader = UsdShade.Shader.Define(stage, "/Material/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.25))
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(0.8)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.27)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.65)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    stage.GetRootLayer().Save()
    ghost, group = _ghost(path)
    ghost.visualize_link_poses(np.tile(np.eye(4), (2, 1, 1, 1)))
    snapshot = group.snapshot()[0]
    np.testing.assert_allclose(snapshot.vertices, [[-1, -1, 3], [3, -1, 3], [1, 5, 3]])
    np.testing.assert_allclose(snapshot.color, [0.5370987] * 3 + [0.4], atol=1e-6)
    ghost.remove()
    texture = UsdShade.Shader.Define(stage, "/Material/Texture")
    texture.CreateIdAttr("UsdUVTexture")
    texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("texture.png")
    )
    texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    shader.GetInput("diffuseColor").ConnectToSource(texture.ConnectableAPI(), "rgb")
    stage.GetRootLayer().Save()
    if packaged:
        package = tmp_path / "visual.usdz"
        assert UsdUtils.CreateNewUsdzPackage(Sdf.AssetPath(str(path)), str(package))
        path = package
    ghost, group = _ghost(path)
    ghost.visualize_link_poses(np.tile(np.eye(4), (2, 1, 1, 1)))
    snapshot = group.snapshot()[0]
    # USD's default useMetadata resolves to a black border. It is baked into
    # padded pixels so the native and GLB samplers can both clamp correctly.
    np.testing.assert_array_equal(snapshot.texture[1:-1, 1:-1], pixels)
    np.testing.assert_array_equal(snapshot.texture[0, 0], [0, 0, 0, 255])
    np.testing.assert_allclose(
        snapshot.uv_coords, [[0.25, 0.25], [0.75, 0.25], [0.5, 0.75]]
    )
    assert snapshot.texture_wrap == ("clamp", "clamp")
    parameters = json.loads(snapshot.material)
    assert parameters["roughness"] == pytest.approx(0.27)
    assert parameters["metallic"] == pytest.approx(0.65)
    assert Path(parameters["base_color_map"]).is_file()
    np.testing.assert_allclose(snapshot.material_uv_coords, [[0, 0], [1, 0], [0.5, 1]])


def test_usd_texture_sampling_preserves_uv_transform_and_opacity(tmp_path):
    from PIL import Image
    from dexsim.spawn import MaterialDesc, MaterialTextureDesc
    from embodichain.lab.visualization.markers._robot_marker_assets import _usd_texture

    path = tmp_path / "sampling.png"
    Image.fromarray(np.full((2, 2, 4), [128, 128, 128, 64], np.uint8)).save(path)
    base = MaterialTextureDesc(
        color_space="raw",
        uv_scale=(2, 3),
        uv_translation=(0.1, 0.2),
        uv_rotation=90,
        wrap_s="mirror",
        wrap_t="clamp",
    )
    alpha = MaterialTextureDesc(
        channel="a",
        color_space="raw",
        uv_scale=(2, 3),
        uv_translation=(0.1, 0.2),
        uv_rotation=90,
        wrap_s="mirror",
        wrap_t="clamp",
    )
    material = MaterialDesc(
        name="source",
        base_color_map=str(path),
        opacity_map=str(path),
        texture_settings={"base_color_map": base, "opacity_map": alpha},
    )
    prototype = MarkerPrototypeCfg(
        shape="mesh", uv_coords=np.array([[1.0, 0.0], [0.0, 1.0]])
    )
    _usd_texture(prototype, material)
    np.testing.assert_allclose(
        prototype.uv_coords, [[0.1, 2.2], [-2.9, 0.2]], atol=1e-6
    )
    np.testing.assert_array_equal(prototype.texture[0, 0], [188, 188, 188, 64])
    assert prototype.texture_wrap == ("mirror", "clamp")
    alpha.uv_scale = (1, 1)
    with pytest.raises(ValueError, match="matching UV"):
        _usd_texture(prototype, material)


def test_shared_usd_source_selects_one_visual_and_rejects_ambiguous_address(tmp_path):
    from embodichain.lab.sim.cfg.source import UsdSourceCfg
    from pxr import Sdf, Usd, UsdGeom

    path = tmp_path / "objects.usda"
    stage = Usd.Stage.CreateNew(str(path))
    for name in ("A", "B"):
        root = UsdGeom.Xform.Define(stage, f"/{name}")
        root.GetPrim().CreateAttribute("MeshObject", Sdf.ValueTypeNames.Token)
        UsdGeom.Cube.Define(stage, f"/{name}/mesh")
    stage.GetRootLayer().Save()
    source = UsdSourceCfg(file_path=str(path), source_prim_path="/B")
    ghost = RobotMarkers(RobotMarkersCfg(name="selected", source=source), _Sim())
    assert ghost.link_names == ("B",)
    ghost.remove()
    with pytest.raises(ValueError, match="exactly one"):
        RobotMarkers(RobotMarkersCfg(name="ambiguous", asset_path=str(path)), _Sim())
    with pytest.raises(ValueError, match="not both"):
        RobotMarkers(
            RobotMarkersCfg(name="duplicate", source=source, asset_path=str(path)),
            _Sim(),
        )


def test_usd_mesh_without_authored_uv_retains_material(tmp_path):
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

    path = tmp_path / "untextured.usda"
    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/Visual")
    root.GetPrim().CreateAttribute("MeshObject", Sdf.ValueTypeNames.Token)
    mesh = UsdGeom.Mesh.Define(stage, "/Visual/mesh")
    mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    material = UsdShade.Material.Define(stage, "/Material")
    shader = UsdShade.Shader.Define(stage, "/Material/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.25))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.3)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    stage.GetRootLayer().Save()
    ghost, group = _ghost(path)
    ghost.visualize_link_poses(np.tile(np.eye(4), (2, 1, 1, 1)))
    snapshot = group.snapshot()[0]
    assert snapshot.material_uv_coords is None
    assert json.loads(snapshot.material)["roughness"] == pytest.approx(0.3)
    ghost.remove()
