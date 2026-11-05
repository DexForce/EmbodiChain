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

"""Source asset preparation preserves geometry, cache identity and invalid-input evidence."""

from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
import numpy as np
import trimesh
from embodichain.gen_sim.scene_engine.asset_normalization import (
    normalize_source_scene_assets,
)
from embodichain.compute.geometry.shading import has_inconsistent_shading_normals


def test_normalization_bakes_node_transform_and_axis_order_once(tmp_path):
    mesh = trimesh.creation.box(extents=[0.2, 0.3, 0.4])
    scene = trimesh.Scene()
    transform = np.eye(4)
    transform[:3, 3] = [0.2, 0.3, 0.4]
    scene.add_geometry(mesh, node_name="authored", geom_name="box", transform=transform)
    path = tmp_path / "source.glb"
    scene.export(path, file_type="glb")
    original = path.read_bytes()
    value = {
        "uid": "part",
        "shape": {"shape_type": "Mesh", "fpath": str(path)},
        "body_scale": [2.0, 3.0, 4.0],
    }
    before = deepcopy(value)
    first = normalize_source_scene_assets(
        background=(),
        rigid_objects=(value,),
        articulations=(),
        cache_root=tmp_path / "assets",
    )
    runtime = Path(first.rigid_objects[0]["shape"]["fpath"])
    baked = trimesh.load(runtime, force="scene").to_geometry()
    expected = trimesh.load(path, force="scene").to_geometry()
    expected.apply_transform(np.diag([2.0, 4.0, 3.0, 1.0]))
    np.testing.assert_allclose(baked.bounds, expected.bounds, atol=1e-7)
    assert len(baked.vertices) == len(expected.vertices)
    assert first.rigid_objects[0]["body_scale"] == [1.0, 1.0, 1.0]
    assert first.provenance[0]["status"] == "generated"
    assert (
        first.asset_hashes["part"] == hashlib.sha256(runtime.read_bytes()).hexdigest()
    )
    second = normalize_source_scene_assets(
        background=(),
        rigid_objects=(value,),
        articulations=(),
        cache_root=tmp_path / "assets",
    )
    assert second.provenance[0]["status"] == "reused"
    assert second.rigid_objects == first.rigid_objects
    assert value == before and path.read_bytes() == original


def test_only_inconsistent_corner_normals_require_vertex_splitting(tmp_path):
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    good = np.array([[0, 1, 2]])
    bad = np.array([[0, 1, 2], [0, 1, 2], [2, 1, 0]])
    assert not has_inconsistent_shading_normals(vertices, good)
    assert has_inconsistent_shading_normals(vertices, bad)
    path = tmp_path / "folded.glb"
    trimesh.Scene(trimesh.Trimesh(vertices, bad, process=False)).export(
        path, file_type="glb"
    )
    result = normalize_source_scene_assets(
        background=(
            {"uid": "folded", "shape": {"shape_type": "Mesh", "fpath": str(path)}},
        ),
        rigid_objects=(),
        articulations=(),
        cache_root=tmp_path / "assets",
    )
    baked = trimesh.load(
        result.background[0]["shape"]["fpath"], force="scene", process=False
    ).to_geometry()
    assert len(baked.vertices) == 9
    assert not has_inconsistent_shading_normals(baked.vertices, baked.faces)


def test_invalid_source_is_retained_with_its_original_digest(tmp_path):
    path = tmp_path / "invalid.glb"
    path.write_bytes(b"not a glb")
    original = {
        "uid": "invalid",
        "shape": {"shape_type": "Mesh", "fpath": str(path)},
        "body_scale": [2.0, 2.0, 2.0],
    }
    result = normalize_source_scene_assets(
        background=(),
        rigid_objects=(original,),
        articulations=({"uid": "native", "fpath": "model.usdc"},),
        cache_root=tmp_path / "assets",
    )
    assert result.rigid_objects[0] == original
    assert result.articulations == ({"uid": "native", "fpath": "model.usdc"},)
    record = result.provenance[0]
    assert record["status"] == "preserved_invalid_source"
    assert (
        record["source_sha256"]
        == record["runtime_sha256"]
        == hashlib.sha256(path.read_bytes()).hexdigest()
    )
    assert record["error"]
