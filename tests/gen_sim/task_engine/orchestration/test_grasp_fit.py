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

"""Opt-in semantic grasp-fit geometry and source preservation."""

from __future__ import annotations

import hashlib
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
import trimesh

from embodichain.gen_sim.task_engine.orchestration.grasp_fit import fit_grasp_assets
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


def _graph(task_type: str = "E2", *, coordinated: bool = False) -> dict:
    call = (
        {
            "kind": "registered",
            "call_id": "simulation.coordinated_transport",
            "arguments": {"object": "cup"},
            "resources": {"left": "left", "right": "right"},
        }
        if coordinated
        else {
            "kind": "pick",
            "object": "cup",
            "resources": {"primary": "left"},
        }
    )
    return {
        "nodes": [
            {
                "task_type": task_type,
                "call": call,
            }
        ],
    }


def test_grasp_fit_scales_only_target_and_preserves_support(tmp_path: Path) -> None:
    scene = _scene(tmp_path, (0.133, 0.19, 0.132))
    source = Path(scene.rigid_objects[0]["shape"]["fpath"])
    original_bytes = source.read_bytes()

    fitted, report = fit_grasp_assets(
        scene, _graph(), openings={"left": 0.128, "right": 0.128}
    )
    record = report["records"][0]
    normalized = normalize_scene_assets(fitted, tmp_path / "bundle")

    assert record["scale_factor"] == pytest.approx(0.123 / 0.133)
    assert record["adapted_extents_m"][0] < 0.124
    assert record["support_bottom_z"] == pytest.approx(0.705)
    assert fitted.rigid_objects[0]["init_pos"][2] < 0.8
    assert fitted.planner_objects[0]["init_pos"] == fitted.rigid_objects[0]["init_pos"]
    assert normalized.rigid_objects[0]["body_scale"] == [1.0, 1.0, 1.0]
    assert normalized.asset_provenance[0]["source_sha256"] == record["source_sha256"]
    assert source.read_bytes() == original_bytes
    assert report["schema_version"] == "gen_sim.asset-adaptation/v3"
    assert record["task_types"] == ["E2"]
    assert record["fit_mode"] == "single_arm_transverse"


def test_grasp_fit_reserves_contact_envelopes(tmp_path: Path) -> None:
    scene = _scene(tmp_path, (0.133, 0.19, 0.132))
    original = deepcopy(scene.rigid_objects)
    fitted, report = fit_grasp_assets(
        scene, _graph(), openings={"left": 0.128}, contact_clearances={"cup": 0.009}
    )
    record = report["records"][0]
    assert record["scale_factor"] == pytest.approx(0.109 / 0.133)
    assert record["pad_margin_m"] == pytest.approx(0.009)
    assert record["contact_clearance_m"] == pytest.approx(0.009)
    assert record["usable_span_m"] == pytest.approx(0.110)
    assert fitted.rigid_objects[0]["body_scale"][0] >= 0.25
    assert scene.rigid_objects == original


@pytest.mark.parametrize(
    "clearances", [{}, {"cup": -0.01}, {"cup": float("nan")}, {"cup": True}]
)
def test_grasp_fit_rejects_invalid_contact_clearances(
    tmp_path: Path, clearances
) -> None:
    scene = _scene(tmp_path, (0.133, 0.19, 0.132))
    with pytest.raises(ValueError, match="contact"):
        fit_grasp_assets(
            scene, _graph(), openings={"left": 0.128}, contact_clearances=clearances
        )


def test_grasp_fit_skips_compatible_asset_and_rejects_excess_shrink(
    tmp_path: Path,
) -> None:
    compatible = _scene(tmp_path, (0.06, 0.12, 0.06))
    unchanged, report = fit_grasp_assets(
        compatible, _graph("E1"), openings={"left": 0.128}
    )
    assert unchanged is compatible
    assert report["status"] == "unchanged"

    oversized = _scene(tmp_path, (0.51, 0.8, 0.6))
    with pytest.raises(ValueError, match="below minimum"):
        fit_grasp_assets(oversized, _graph("E3"), openings={"left": 0.128})


@pytest.mark.parametrize("task_type", ["E1", "E3", "E4"])
def test_grasp_fit_supports_single_arm_pick_task_families(
    tmp_path: Path, task_type: str
) -> None:
    fitted, report = fit_grasp_assets(
        _scene(tmp_path, (0.14, 0.19, 0.13)),
        _graph(task_type),
        openings={"left": 0.128},
    )
    assert fitted is not None
    assert report["records"][0]["task_types"] == [task_type]
    assert report["records"][0]["fit_mode"] == "single_arm_transverse"


def test_grasp_fit_supports_e5_coordinated_acquisition(tmp_path: Path) -> None:
    _, report = fit_grasp_assets(
        _scene(tmp_path, (0.13, 0.30, 0.20)),
        _graph("E5", coordinated=True),
        openings={"left": 0.128, "right": 0.128},
    )
    record = report["records"][0]
    assert record["task_types"] == ["E5"]
    assert record["resources"] == ["left", "right"]
    assert record["scale_factor"] == pytest.approx(0.123 / 0.13)


def test_grasp_fit_safely_skips_graph_without_acquisition(tmp_path: Path) -> None:
    scene = _scene(tmp_path, (0.13, 0.30, 0.20))
    graph = {
        "nodes": [
            {
                "task_type": "E8",
                "call": {
                    "kind": "registered",
                    "call_id": "simulation.park",
                    "arguments": {},
                    "resources": {"primary": "left"},
                },
            }
        ]
    }
    fitted, report = fit_grasp_assets(scene, graph, openings={"left": 0.128})
    assert fitted is scene
    assert report["status"] == "not_applicable"
    assert report["records"] == []


def test_grasp_fit_adapts_multiple_acquired_objects_only(tmp_path: Path) -> None:
    scene = _scene(tmp_path, (0.14, 0.19, 0.13))
    second_path = tmp_path / "bottle.glb"
    trimesh.creation.box(extents=(0.15, 0.2, 0.14)).export(second_path)
    second = deepcopy(scene.rigid_objects[0])
    second["uid"] = "bottle"
    second["shape"]["fpath"] = str(second_path)
    second_planner = {**second, "runtime_uid": "bottle", "role": "rigid_object"}
    scene = replace(
        scene,
        rigid_objects=(*scene.rigid_objects, second),
        planner_objects=(*scene.planner_objects, second_planner),
        asset_hashes={
            **scene.asset_hashes,
            "bottle": hashlib.sha256(second_path.read_bytes()).hexdigest(),
        },
    )
    graph = _graph("E1")
    graph["nodes"].append(
        {
            "task_type": "E3",
            "call": {
                "kind": "pick",
                "object": "bottle",
                "resources": {"primary": "right"},
            },
        }
    )

    _, report = fit_grasp_assets(scene, graph, openings={"left": 0.128, "right": 0.128})

    assert [record["object_id"] for record in report["records"]] == ["bottle", "cup"]
    assert [record["task_types"] for record in report["records"]] == [["E3"], ["E1"]]
