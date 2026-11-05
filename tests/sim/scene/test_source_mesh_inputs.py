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
"""Authored source mesh inputs preserve frozen basis, topology and height."""

from __future__ import annotations
from pathlib import Path
import hashlib, json
import numpy as np
import pytest
from embodichain.lab.sim.scene.source_geometry import (
    read_gltf_z_up_mesh,
    estimate_source_mesh_top_z,
    read_source_vertices,
)

_FROZEN = json.loads(
    (Path(__file__).parent / "data/source_mesh_inputs_b0.json").read_text()
)
_FIXTURE = Path(__file__).parent / _FROZEN["fixture"]


def test_source_reader_preserves_frozen_node_transforms_basis_faces_and_input_bytes():
    original = _FIXTURE.read_bytes()
    assert hashlib.sha256(original).hexdigest() == _FROZEN["fixture_sha256"]
    mesh = read_gltf_z_up_mesh(_FIXTURE, body_scale=(2, 0.5, 3))
    assert np.array_equal(
        np.asarray(mesh.vertices),
        np.asarray(_FROZEN["source_vertices_z_up"]) * [2, 0.5, 3],
    )
    assert np.array_equal(mesh.faces, np.asarray(_FROZEN["source_faces"]))
    mesh.vertices[:] = 0
    assert _FIXTURE.read_bytes() == original


@pytest.mark.parametrize("case", _FROZEN["cases"])
def test_mesh_top_matches_frozen_source_world_pose_and_signed_scale(case):
    source = {key: value for key, value in case.items() if key != "expected_top_z"}
    source["shape"] = {"fpath": str(_FIXTURE)}
    assert estimate_source_mesh_top_z(source) == case["expected_top_z"]


def test_unavailable_authored_geometry_stays_unavailable(tmp_path):
    source = {"shape": {"fpath": str(tmp_path / "missing.glb")}}
    assert estimate_source_mesh_top_z(source) is None
    assert (
        estimate_source_mesh_top_z({"shape": {"shape_type": "Cube", "size": [1, 1, 1]}})
        is None
    )
    with pytest.raises(ValueError, match="Could not inspect source mesh"):
        read_gltf_z_up_mesh(source["shape"]["fpath"])


def test_local_source_vertices_keep_frozen_order_scale_dtype_and_ignore_world_pose():
    source = {
        "shape": {"shape_type": "Mesh", "fpath": str(_FIXTURE)},
        "body_scale": [2.0, 0.5, 3.0],
        "init_pos": [99, 88, 77],
        "init_rot": [10, 20, 30],
    }
    original = _FIXTURE.read_bytes()
    vertices = read_source_vertices(source)
    assert vertices.dtype == np.float64
    assert np.array_equal(
        vertices, np.asarray(_FROZEN["source_vertices_z_up"]) * [2.0, 0.5, 3.0]
    )
    vertices[:] = 0
    assert _FIXTURE.read_bytes() == original


def test_source_local_override_keeps_its_own_scale_and_dtype():
    vertices = np.arange(6, dtype=np.float32).reshape(2, 3)
    expected = vertices.copy()
    output = read_source_vertices({"body_scale": [4, 4, 4]}, local_vertices=vertices)
    assert output.dtype == np.float32 and np.array_equal(output, expected)
    with pytest.raises(ValueError, match="no mesh geometry"):
        read_source_vertices({"shape": {"shape_type": "Cube", "size": [1, 1, 1]}})


def test_articulation_source_vertices_match_original_source_reader(
    tmp_path, monkeypatch
):
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    from _control_source_fixture import knob_source

    config = knob_source(tmp_path)
    frozen = json.loads(
        (
            Path(__file__).parent / "data/source_local_vertices_reference.json"
        ).read_text()
    )
    assert (
        hashlib.sha256(Path(config["fpath"]).read_bytes()).hexdigest()
        == frozen["fixture_sha256"]
    )
    config["body_scale"] = [2.0, 2.0, 2.0]
    vertices = read_source_vertices(config, articulation=True)
    assert vertices.dtype == np.float64 and np.array_equal(
        vertices, np.asarray(frozen["vertices"])
    )
    with pytest.raises(ValueError, match="finite positive"):
        read_source_vertices(dict(config, body_scale=[1, -1, 1]), articulation=True)
