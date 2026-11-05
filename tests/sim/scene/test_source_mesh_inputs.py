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
