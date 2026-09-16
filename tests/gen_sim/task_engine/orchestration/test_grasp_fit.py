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

"""Opt-in E2 grasp-fit geometry and source preservation."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import trimesh

from embodichain.gen_sim.task_engine.orchestration.grasp_fit import fit_e2_grasp_asset
from embodichain.gen_sim.task_engine.orchestration.scene_assets import (
    normalize_scene_assets,
)
from embodichain.gen_sim.task_engine.orchestration.source_scene import PreparedScene


def _scene(tmp_path: Path, extents: tuple[float, float, float]) -> PreparedScene:
    path = tmp_path / "cup.glb"
    trimesh.creation.box(extents=extents).export(path)
    cup = {
        "uid": "cup",
        "shape": {"shape_type": "Mesh", "fpath": str(path)},
        "body_scale": [1.0, 1.0, 1.0],
        "init_pos": [0.2, -0.3, 0.8],
        "init_rot": [0.0, 0.0, 0.0],
    }
    table = {"uid": "table", "init_pos": [0.0, 0.0, 0.0]}
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return PreparedScene(
        source_config_path=tmp_path / "scene.json",
        scene_dir=tmp_path,
        planner_objects=({**cup, "runtime_uid": "cup", "role": "rigid_object"},),
        background=(table,),
        rigid_objects=(cup,),
        articulations=(),
        uid_map={"cup": "cup"},
        table_top_z=0.7,
        z_rotation_degrees=0.0,
        body_scale_policy="preserve",
        body_scale=(1.0, 1.0, 1.0),
        asset_hashes={"cup": digest},
    )


def _graph() -> dict:
    return {
        "task_groups": [{"task_type": "E2"}],
        "nodes": [
            {
                "task_type": "E2",
                "call": {
                    "kind": "pick",
                    "object": "cup",
                    "resources": {"primary": "left"},
                },
            }
        ],
    }


def test_grasp_fit_scales_only_target_and_preserves_support(tmp_path: Path) -> None:
    scene = _scene(tmp_path, (0.133, 0.19, 0.132))
    source = Path(scene.rigid_objects[0]["shape"]["fpath"])
    original_bytes = source.read_bytes()

    fitted, record = fit_e2_grasp_asset(scene, _graph(), opening=0.128)
    normalized = normalize_scene_assets(fitted, tmp_path / "bundle")

    assert record["scale_factor"] == pytest.approx(0.123 / 0.133)
    assert record["adapted_extents_m"][0] < 0.124
    assert record["support_bottom_z"] == pytest.approx(0.705)
    assert fitted.rigid_objects[0]["init_pos"][2] < 0.8
    assert fitted.planner_objects[0]["init_pos"] == fitted.rigid_objects[0]["init_pos"]
    assert normalized.rigid_objects[0]["body_scale"] == [1.0, 1.0, 1.0]
    assert normalized.asset_provenance[0]["source_sha256"] == record["source_sha256"]
    assert source.read_bytes() == original_bytes


def test_grasp_fit_skips_compatible_asset_and_rejects_excess_shrink(
    tmp_path: Path,
) -> None:
    compatible = _scene(tmp_path, (0.06, 0.19, 0.06))
    unchanged, record = fit_e2_grasp_asset(compatible, _graph(), opening=0.128)
    assert unchanged is compatible
    assert record["status"] == "unchanged"

    oversized = _scene(tmp_path, (0.51, 0.8, 0.51))
    with pytest.raises(ValueError, match="below minimum"):
        fit_e2_grasp_asset(oversized, _graph(), opening=0.128)


def test_grasp_fit_rejects_non_e2_graph(tmp_path: Path) -> None:
    graph = _graph()
    graph["task_groups"][0]["task_type"] = "E1"
    with pytest.raises(ValueError, match="one E2 task group"):
        fit_e2_grasp_asset(_scene(tmp_path, (0.133, 0.19, 0.132)), graph, opening=0.128)
