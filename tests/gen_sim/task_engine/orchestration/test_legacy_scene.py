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

import json
from pathlib import Path

import trimesh

from embodichain.gen_sim.action_engine.generation.source_scene import prepare_scene
from embodichain.gen_sim.task_engine.orchestration.legacy_scene import (
    convert_legacy_gym_project,
    restore_locked_scene_entities,
)
from embodichain.gen_sim.task_engine.orchestration.scene_source import (
    fingerprint_scene_source,
)
from embodichain.gen_sim.task_engine.scene import build_conservative_scene_graph


def _legacy_project(tmp_path: Path) -> Path:
    project = tmp_path / "legacy"
    assets = project / "assets"
    assets.mkdir(parents=True)
    trimesh.creation.box(extents=[1.0, 1.0, 0.1]).export(
        assets / "table.glb", file_type="glb"
    )
    trimesh.creation.cylinder(radius=0.03, height=0.12).export(
        assets / "can.glb", file_type="glb"
    )
    (assets / "cabinet.urdf").write_text(
        '<robot name="cabinet"><link name="base"/></robot>\n',
        encoding="utf-8",
    )
    config = {
        "background": [
            {
                "uid": "table_0",
                "name": "table",
                "description": "A work table.",
                "category": "table",
                "shape": {"shape_type": "Mesh", "fpath": "assets/table.glb"},
                "init_pos": [0.0, 0.0, 0.0],
                "init_rot": [0.0, 0.0, 0.0],
                "body_scale": [1.0, 1.0, 1.0],
            }
        ],
        "rigid_object": [
            {
                "uid": "can_0",
                "name": "red can",
                "description": "A red can.",
                "category": "can",
                "shape": {"shape_type": "Mesh", "fpath": "assets/can.glb"},
                "init_pos": [0.0, 0.1, 0.2],
                "init_rot": [0.0, 0.0, 0.0],
                "body_scale": [1.0, 1.0, 1.0],
            }
        ],
        "articulation": [
            {
                "uid": "cabinet_0",
                "name": "cabinet",
                "description": "A fixed cabinet.",
                "category": "cabinet",
                "fpath": "assets/cabinet.urdf",
                "init_pos": [0.5, 0.0, 0.0],
                "init_rot": [0.0, 0.0, 0.0],
                "body_scale": [1.0, 1.0, 1.0],
            }
        ],
    }
    (project / "gym_config.json").write_text(json.dumps(config), encoding="utf-8")
    return project


def test_legacy_conversion_is_read_only_and_restores_locked_articulation(
    tmp_path: Path,
) -> None:
    project = _legacy_project(tmp_path)
    original = fingerprint_scene_source(project)

    revision = convert_legacy_gym_project(project, tmp_path / "revision")
    converted = json.loads(revision.scene_config_path.read_text(encoding="utf-8"))
    manifest = json.loads(revision.manifest_path.read_text(encoding="utf-8"))

    assert fingerprint_scene_source(project) == original
    assert converted["format"] == "embodichain.scene-export/v1"
    assert converted["background"][0]["uid"] == "table"
    assert converted["rigid_object"][0]["uid"] == "can"
    assert converted["articulation"][0]["uid"] == "cabinet"
    assert manifest["audit_hierarchy"] == "unknown"
    assert manifest["operational_hierarchy"] == "assumed_on_table"
    assert set(revision.locked_entity_uids) == {"table", "cabinet"}

    converted["articulation"] = []
    revision.scene_config_path.write_text(json.dumps(converted), encoding="utf-8")
    restore_locked_scene_entities(revision.output_root)
    restored = json.loads(revision.scene_config_path.read_text(encoding="utf-8"))

    assert restored["articulation"][0]["uid"] == "cabinet"
    assert Path(restored["articulation"][0]["fpath"]).is_file()
    assert fingerprint_scene_source(project) == original


def test_legacy_conversion_separates_audit_and_operational_hierarchy(
    tmp_path: Path,
) -> None:
    revision = convert_legacy_gym_project(
        _legacy_project(tmp_path),
        tmp_path / "revision",
    )

    operational = json.loads(revision.scene_graph_path.read_text(encoding="utf-8"))
    conservative = build_conservative_scene_graph(
        prepare_scene(revision.scene_config_path),
        scene_id="legacy-scene",
    )

    operational_can = next(
        node for node in operational["nodes"] if node["object_id"] == "can"
    )
    conservative_can = next(
        node for node in conservative["nodes"] if node["uid"] == "can"
    )
    assert operational_can["parent_id"] == "table"
    assert operational_can["parent_relation"] == "on"
    assert conservative_can["parent_uid"] == "unknown"
    assert conservative_can["parent_relation"] == "unknown"
    assert conservative_can["source"] == "conservative_import"
