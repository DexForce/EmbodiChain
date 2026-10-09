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
import pytest

from embodichain.gen_sim.task_engine.orchestration.source_scene import prepare_scene
from embodichain.gen_sim.task_engine.orchestration.legacy_scene import (
    convert_legacy_gym_project,
    restore_locked_scene_entities,
)
from embodichain.gen_sim.task_engine.orchestration.scene_source import (
    fingerprint_scene_source,
    scene_revision_id,
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
    trimesh.creation.box(extents=[0.3, 0.2, 0.4]).export(
        assets / "cabinet.glb", file_type="glb"
    )
    (assets / "cabinet.urdf").write_text(
        '<robot name="cabinet"><link name="base"><visual><geometry>'
        '<mesh filename="cabinet.glb"/></geometry></visual></link></robot>\n',
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
    assert converted["articulation"] == []
    assert manifest["locked_articulations"][0]["uid"] == "cabinet"
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


def test_scene_identity_covers_transitive_urdf_meshes(tmp_path: Path) -> None:
    project = _legacy_project(tmp_path)
    original_fingerprint = fingerprint_scene_source(project)
    original_revision = scene_revision_id(project)

    trimesh.creation.box(extents=[0.6, 0.2, 0.4]).export(
        project / "assets" / "cabinet.glb",
        file_type="glb",
    )

    changed_fingerprint = fingerprint_scene_source(project)
    assert changed_fingerprint.asset_sha256 != original_fingerprint.asset_sha256
    assert scene_revision_id(project) != original_revision


def test_converted_legacy_scene_round_trips_through_formal_edit_importer(
    tmp_path: Path,
) -> None:
    from embodichain.gen_sim.scene_engine.pipeline.utils.scene_importer import (
        SceneExportImporter,
    )
    from embodichain.gen_sim.scene_engine.pipeline.utils.scene_exporter import (
        SceneExporter,
    )
    from embodichain.gen_sim.task_engine.orchestration.source_scene import (
        resolve_source_scene,
    )

    project = _legacy_project(tmp_path)
    original = fingerprint_scene_source(project)
    before = prepare_scene(project)
    revision = convert_legacy_gym_project(project, tmp_path / "revision")
    imported, graph = SceneExportImporter(
        output_root=revision.output_root
    ).import_scene_and_graph()
    assert {item.id for item in imported.objects} == {"table", "can"}
    assert all(item.is_articulated is False for item in imported.objects)
    SceneExporter(
        scene=imported, scene_graph=graph, output_root=revision.output_root
    ).export()
    restored = restore_locked_scene_entities(revision.output_root)
    resolve_source_scene(restored)  # Checks companion/runtime ID and label agreement.
    after = prepare_scene(restored)
    assert after.rigid_objects[0]["init_pos"] == pytest.approx(
        before.rigid_objects[0]["init_pos"]
    )
    assert after.rigid_objects[0]["init_rot"] == pytest.approx(
        before.rigid_objects[0]["init_rot"]
    )
    assert {item["uid"] for item in after.articulations} == {"cabinet"}
    assert fingerprint_scene_source(project) == original


def test_legacy_primitive_geometry_keeps_native_z_up_height(tmp_path: Path) -> None:
    from embodichain.gen_sim.scene_engine.pipeline.utils.scene_importer import (
        SceneExportImporter,
    )

    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            {
                "background": [
                    {
                        "uid": "table",
                        "shape": {"shape_type": "Cube", "size": [1, 1, 0.1]},
                        "init_pos": [0, 0, 0.5],
                    }
                ],
                "rigid_object": [
                    {
                        "uid": "cube",
                        "shape": {"shape_type": "Cube", "size": [0.05] * 3},
                        "init_pos": [0.3, 0.4, 0.575],
                    }
                ],
            }
        )
    )
    revision = convert_legacy_gym_project(source, tmp_path / "revision")
    SceneExportImporter(output_root=revision.output_root).import_scene_and_graph()
    result = prepare_scene(revision.scene_config_path)
    assert result.table_top_z == pytest.approx(0.55, abs=1e-6)
    assert result.rigid_objects[0]["init_pos"] == pytest.approx([0.3, 0.4, 0.575])


@pytest.mark.parametrize("scale", [[1.0, 1.0, 1.0], [1.0, 2.0, 3.0]])
def test_legacy_edit_geometry_and_new_articulation_share_native_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scale: list[float],
) -> None:
    import numpy as np
    from embodichain.gen_sim.scene_engine.pipeline.utils.scene_importer import (
        SceneExportImporter,
    )
    from embodichain.gen_sim.scene_engine.pipeline.utils.scene_exporter import (
        SceneExporter,
    )
    from embodichain.gen_sim.scene_engine.pipeline.utils.scene_layout_utils import (
        measure_scene_object_z_up_world_aabb,
    )
    from embodichain.gen_sim.scene_engine.core.scene_object import SceneObject
    from embodichain.gen_sim.scene_engine.core.scene_graph import SceneGraphNode
    from embodichain.gen_sim.scene_engine.pipeline.utils import scene_exporter

    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            {
                "background": [
                    {
                        "uid": "table",
                        "shape": {"shape_type": "Cube", "size": [1.0, 1.0, 0.1]},
                        "init_pos": [0.0, 0.0, 0.5],
                        "body_scale": scale,
                    }
                ],
                "rigid_object": [
                    {
                        "uid": "cube",
                        "shape": {"shape_type": "Cube", "size": [0.05] * 3},
                        "init_pos": [0.3, 0.2, 0.7],
                    }
                ],
            }
        )
    )
    revision = convert_legacy_gym_project(source, tmp_path / "revision")
    scene, graph = SceneExportImporter(
        output_root=revision.output_root
    ).import_scene_and_graph()
    expected = np.array(
        [
            [-0.5 * scale[0], -0.5 * scale[1], 0.5 - 0.05 * scale[2]],
            [0.5 * scale[0], 0.5 * scale[1], 0.5 + 0.05 * scale[2]],
        ]
    )
    np.testing.assert_allclose(
        measure_scene_object_z_up_world_aabb(scene_object=scene.table),
        expected,
        atol=1e-6,
    )
    assert scene.table.scale == [1.0, 1.0, 1.0]
    addition = SceneObject(
        "new_drawer",
        "asset",
        "drawer",
        "drawer",
        "A drawer.",
        is_articulated=True,
        articulated_usdc_path=str(tmp_path / "new.usdc"),
        articulated_usdc_scale=[1.0] * 3,
        pos=[0.0, 0.55, 0.0],
        rot=[0.0] * 3,
        scale=[1.0] * 3,
    )
    scene.objects.append(addition)
    graph.nodes.append(SceneGraphNode("new_drawer", "table", "on"))
    # Asset-root height is a separate tested contract; isolate support placement.
    monkeypatch.setattr(scene_exporter, "_articulation_root_bottom_z", lambda *a: 0.0)
    entry = SceneExporter(
        scene=scene, scene_graph=graph, output_root=revision.output_root
    )._articulation_config(
        scene_object=addition,
        articulated_relative_path="new.usdc",
        proxy_glb_relative_path="proxy.glb",
    )
    assert entry["init_pos"][2] == pytest.approx(expected[1, 2] + 0.001, abs=1e-6)
