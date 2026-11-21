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

"""Reject altered raw/prepared geometry before conversion or model loading."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest


def test_debug_preparation_verifies_collision_meshes(tmp_path):
    pytest.importorskip("trimesh")
    from examples.learning.rigidformer._prepare import prepare

    raw = tmp_path / "raw"
    scene = raw / "data/movi_a/train-00000.parquet"
    mesh = raw / "assets/kubasic/cube/collision_geometry.obj"
    for path in (scene, mesh):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"original-payload")
    manifest = {
        "files": {
            str(path.relative_to(raw)): {
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()
            }
            for path in (scene, mesh)
        }
    }
    (raw / "manifest.json").write_text(json.dumps(manifest))
    mesh.write_bytes(b"changed-mesh")
    with pytest.raises(ValueError, match="raw data checksum differs: assets/"):
        prepare(raw, tmp_path / "prepared", "movi_a", limit_per_split=1)
    assert not (tmp_path / "prepared/MOVi-A").exists()


def test_evaluation_verifies_geometry_before_loading_weights(tmp_path):
    from examples.learning.rigidformer.__main__ import evaluate

    root = tmp_path / "prepared"
    root.mkdir()
    data = root / "frames.npy"
    data.write_bytes(b"committed-geometry")
    (root / "manifest.json").write_text(
        json.dumps(
            {"files": {data.name: hashlib.sha256(data.read_bytes()).hexdigest()}}
        )
    )
    data.write_bytes(b"altered-geometry")
    output = tmp_path / "evaluation"
    with pytest.raises(ValueError, match="geometry hash differs"):
        evaluate(
            SimpleNamespace(
                data_root=root, output=output, checkpoint=tmp_path / "absent.pt"
            )
        )
    assert json.loads((output / "failure.json").read_text())["type"] == "ValueError"
