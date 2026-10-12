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

"""Provider-free contracts for source indexing, evidence and result export."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

from embodichain.toolkits.mesh_affordance import (
    MeshAffordanceCfg,
    analyze_mesh_affordance,
    prepare_mesh_affordance,
    score_mesh_affordance,
)
from embodichain.toolkits.mesh_affordance import _pipeline
from embodichain.toolkits.mesh_affordance._mesh import load_mesh


@pytest.fixture
def obj_path(tmp_path):
    path = tmp_path / "mesh.obj"
    path.write_text(
        "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nv 9 9 9\nvt 0 0\nvn 0 0 1\nusemtl first\nf 1/1/1 2/1/1 3/1/1\nusemtl second\nf -5/1/1 -3/1/1 -2/1/1\n"
    )
    return path


def test_obj_preserves_source_vertices_across_materials_and_negative_indices(obj_path):
    vertices, faces, contract = load_mesh(obj_path)
    assert vertices.shape == (5, 3)
    assert faces.tolist() == [[0, 1, 2], [0, 2, 3]]
    assert contract == "obj_v_line_order"


def test_nontriangle_obj_is_rejected_without_silent_retriangulation(tmp_path):
    path = tmp_path / "quad.obj"
    path.write_text("v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4\n")
    with pytest.raises(ValueError, match="triangulate"):
        load_mesh(path)


def _fake_render(directory, cfg, mode):
    if mode == "evidence":
        path = directory / "evidence.json"
        manifest = json.loads(path.read_text())
        manifest["images"] = ["evidence_00.png"]
        path.write_text(json.dumps(manifest))
        (directory / "evidence_00.png").write_bytes(b"test renderer")
    else:
        (directory / "heatmap.png").write_bytes(b"test renderer")


def test_end_to_end_exports_source_indexed_values(obj_path, tmp_path, monkeypatch):
    monkeypatch.setattr(_pipeline, "_render", _fake_render)

    def fake_codex(directory, model, executable, timeout):
        assert model == "gpt-6-astra"
        return {
            "task_interpretation": "test",
            "assumptions": [],
            "regions": [
                {
                    "name": "grip",
                    "patch_ids": [0],
                    "score": 0.9,
                    "confidence": 0.8,
                    "reason": "test",
                },
                {
                    "name": "avoid",
                    "patch_ids": [1],
                    "score": 0.1,
                    "confidence": 0.8,
                    "reason": "test",
                },
            ],
        }

    monkeypatch.setattr(_pipeline, "run_codex", fake_codex)
    cfg = MeshAffordanceCfg(patch_count=2, codex_executable=sys.executable)
    result = analyze_mesh_affordance(obj_path, "object", "task", tmp_path / "out", cfg)
    assert result.scores.shape == (5,)
    assert result.scores[-1] == 0
    assert not result.graspable_mask[-1]
    output = np.load(result.output_dir / "affordance.npz", allow_pickle=False)
    np.testing.assert_array_equal(output["vertices"], load_mesh(obj_path)[0])
    np.testing.assert_array_equal(output["vertex_ids"], np.arange(5))
    assert output["graspable_mask"].sum() == 1
    import trimesh

    ply = trimesh.load(result.output_dir / "affordance.ply", process=False)
    np.testing.assert_allclose(ply.vertices, output["vertices"])
    with pytest.raises(FileExistsError):
        score_mesh_affordance(result.output_dir, cfg)


def test_tampered_geometry_is_rejected_before_inference(
    obj_path, tmp_path, monkeypatch
):
    monkeypatch.setattr(_pipeline, "_render", _fake_render)
    directory = prepare_mesh_affordance(
        obj_path, "object", "task", tmp_path / "out", MeshAffordanceCfg(patch_count=1)
    )
    (directory / "geometry.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="geometry changed"):
        score_mesh_affordance(directory)


def test_existing_results_and_invalid_config_are_preserved(obj_path, tmp_path):
    directory = tmp_path / "out"
    directory.mkdir()
    marker = directory / "keep.txt"
    marker.write_text("existing run")
    with pytest.raises(FileExistsError):
        prepare_mesh_affordance(obj_path, "object", "task", directory)
    assert marker.read_text() == "existing run"
    with pytest.raises(ValueError, match="threshold"):
        prepare_mesh_affordance(
            obj_path,
            "object",
            "task",
            tmp_path / "new",
            MeshAffordanceCfg(threshold=float("nan")),
        )


def test_part_segmentation_is_independent_of_grasp_suitability(
    obj_path, tmp_path, monkeypatch
):
    monkeypatch.setattr(_pipeline, "_render", _fake_render)

    def fake_codex(directory, model, executable, timeout):
        assert (
            json.loads((directory / "evidence.json").read_text())["target_part"]
            == "把手"
        )
        return {
            "task_interpretation": "pour",
            "assumptions": [],
            "regions": [
                {
                    "name": "handle root",
                    "patch_ids": [0],
                    "score": 0.1,
                    "confidence": 0.9,
                    "part_score": 1.0,
                    "part_confidence": 0.9,
                    "reason": "Part of handle but a poor grip",
                },
                {
                    "name": "body",
                    "patch_ids": [1],
                    "score": 0.9,
                    "confidence": 0.9,
                    "part_score": 0.0,
                    "part_confidence": 0.9,
                    "reason": "Graspable body is not a handle",
                },
            ],
        }

    monkeypatch.setattr(_pipeline, "run_codex", fake_codex)
    cfg = MeshAffordanceCfg(
        patch_count=2, target_part="把手", codex_executable=sys.executable
    )
    result = analyze_mesh_affordance(obj_path, "cup", "pour", tmp_path / "part", cfg)
    values = np.load(result.output_dir / "affordance.npz", allow_pickle=False)
    handle_faces = values["face_patch_ids"] == 0
    np.testing.assert_array_equal(values["part_face_mask"], handle_faces)
    assert not np.array_equal(result.graspable_mask, result.part_mask)
    submesh = np.load(result.output_dir / "target_part.npz", allow_pickle=False)
    np.testing.assert_array_equal(
        submesh["vertices"], values["vertices"][submesh["source_vertex_ids"]]
    )
    np.testing.assert_array_equal(
        submesh["source_vertex_ids"][submesh["faces"]],
        values["faces"][submesh["source_face_ids"]],
    )
    assert (result.output_dir / "target_part.ply").is_file()
