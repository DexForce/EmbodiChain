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
"""Static source world geometry retains authored topology and conservative bounds."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest


def test_gltf_source_reader_preserves_node_transform_and_single_body_scale(tmp_path):
    import trimesh
    from embodichain.lab.sim.scene.source_geometry import read_gltf_z_up_mesh

    scene = trimesh.Scene()
    transform = np.eye(4)
    transform[:3, 3] = [1, 2, 3]
    scene.add_geometry(trimesh.creation.box(extents=[2, 4, 6]), transform=transform)
    path = tmp_path / "offset.glb"
    path.write_bytes(scene.export(file_type="glb"))
    original = path.read_bytes()
    mesh = read_gltf_z_up_mesh(path, body_scale=[1, 2, 3])
    np.testing.assert_array_equal(mesh.bounds, [[0, -12, 0], [2, 0, 12]])
    assert len(mesh.faces) == 12 and mesh.vertices.dtype == np.float64
    assert path.read_bytes() == original
    from embodichain.lab.sim.scene.source_geometry import source_longest_local_axis

    # Authored raw dimensions select the axis before glTF basis conversion.
    assert source_longest_local_axis(
        {"shape": {"fpath": str(path)}, "body_scale": [100, 1, 1]}
    ) == [0.0, -1.0, 0.0]
    assert source_longest_local_axis({"shape": {}}) == [0.0, 0.0, 1.0]
    with pytest.raises(ValueError, match="finite three-vector"):
        read_gltf_z_up_mesh(path, body_scale=[1, float("nan"), 1])


def test_static_source_envelope_keeps_single_scale_and_declared_rotation():
    from embodichain.lab.sim.scene.source_envelopes import (
        source_world_vertices,
        source_obstacle_bounds,
    )

    source = {
        "uid": "box",
        "shape": {"shape_type": "Cube", "size": [1, 2, 3]},
        "body_scale": [2, 1, 0.5],
        "init_pos": [0.4, 0.2, 0.8],
        "init_rot": [0, 0, 90],
    }
    vertices = source_world_vertices(source)
    expected = np.array([[-0.6, -0.8, 0.05], [1.4, 1.2, 1.55]])
    np.testing.assert_allclose(
        np.stack((vertices.min(0), vertices.max(0))), expected, rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        source_obstacle_bounds(source), expected, rtol=0, atol=1e-12
    )


def test_static_slider_envelope_includes_full_travel_and_rejects_rotation(tmp_path):
    from pxr import Usd, UsdGeom, UsdPhysics
    from embodichain.lab.sim.scene.source_envelopes import source_obstacle_bounds

    path = tmp_path / "slider.usda"
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/assembly").GetPrim()
    stage.SetDefaultPrim(root)
    UsdPhysics.ArticulationRootAPI.Apply(root)
    for name in ("base", "drawer"):
        body = UsdGeom.Xform.Define(stage, f"/assembly/{name}").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(body)
        mesh = UsdGeom.Mesh.Define(stage, f"/assembly/{name}/collider")
        mesh.CreatePointsAttr([(-0.125, -0.125, -0.125), (0.125, 0.125, 0.125)])
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr(True)
    joint = UsdPhysics.PrismaticJoint.Define(stage, "/assembly/slider")
    joint.CreateBody0Rel().SetTargets(["/assembly/base"])
    joint.CreateBody1Rel().SetTargets(["/assembly/drawer"])
    joint.CreateAxisAttr("X")
    joint.CreateLowerLimitAttr(-0.25)
    joint.CreateUpperLimitAttr(0.5)
    stage.GetRootLayer().Save()
    source = {
        "fpath": str(path),
        "fix_base": True,
        "init_pos": [1, 2, 3],
        "init_rot": [0, 0, 90],
        "body_scale": [2, 2, 2],
    }
    np.testing.assert_allclose(
        source_obstacle_bounds(source, articulation=True),
        [[0.75, 0.75, 2.75], [1.25, 3.25, 3.25]],
        atol=1e-8,
        rtol=0,
    )
    with pytest.raises(ValueError, match="fixed articulation"):
        source_obstacle_bounds({**source, "fix_base": False}, articulation=True)
    joint.GetPrim().SetTypeName("PhysicsRevoluteJoint")
    stage.GetRootLayer().Save()
    with pytest.raises(ValueError, match="non-prismatic"):
        source_obstacle_bounds(source, articulation=True)


def test_static_geometry_helpers_preserve_explicit_pose_raw_bounds_and_world_mesh():
    from embodichain.lab.sim.scene.source_geometry import (
        source_rotation,
        source_position,
        source_local_bounds,
        source_longest_local_axis,
        read_source_world_mesh,
    )

    source = {
        "uid": "box",
        "shape": {"shape_type": "Cube", "size": [1, 2, 3]},
        "body_scale": [2, 1, 0.5],
        "init_pos": [0.4, 0.2, 0.8],
        "init_rot": [0, 0, 0],
        "init_local_pose": [
            [0, -1, 0, 0.4],
            [1, 0, 0, 0.2],
            [0, 0, 1, 0.8],
            [0, 0, 0, 1],
        ],
    }
    assert source_position(source) == (0.4, 0.2, 0.8)
    assert np.array_equal(source_rotation(source), [[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    assert source_local_bounds(source) == ([-0.5, -1.0, -1.5], [0.5, 1.0, 1.5])
    assert source_longest_local_axis(source) == [0.0, 0.0, 1.0]
    mesh = read_source_world_mesh(source)
    assert len(mesh.faces) == 12
    assert np.allclose(
        mesh.bounds, [[-0.6, -0.8, 0.05], [1.4, 1.2, 1.55]], atol=1e-12, rtol=0
    )
    with pytest.raises(ValueError, match="three-value init_pos"):
        source_position({"init_local_pose": source["init_local_pose"]})
