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
from types import SimpleNamespace

import pytest

from embodichain.gen_sim.scene_engine.core.scene import Scene
from embodichain.gen_sim.scene_engine.core.scene_graph import (
    SceneGraph,
    SceneGraphNode,
)
from embodichain.gen_sim.scene_engine.core.scene_object import SceneObject
from embodichain.gen_sim.scene_engine.pipeline.api import (
    SCENE_BLUEPRINT_SCHEMA,
    SceneBlueprintPackage,
    SceneMaterialization,
)
import embodichain.gen_sim.task_engine.scene_backend as scene_backend_module
from embodichain.gen_sim.task_engine.scene_backend import (
    SceneAnalysis,
    SceneEngineBackend,
    scene_blueprint_objects,
)
from embodichain.gen_sim.task_engine.workflow_contracts import TASK_RUN_REQUEST_SCHEMA


def _request(tmp_path: Path, project: Path, *, edit: str | None) -> dict:
    return {
        "schema_version": TASK_RUN_REQUEST_SCHEMA,
        "task_id": "task",
        "task_instruction": "Move the cup.",
        "image_path": None,
        "gym_project": project.as_posix(),
        "scene_edit_prompt": edit,
        "output_dir": (tmp_path / "run").as_posix(),
    }


def _scene_export(tmp_path: Path) -> Path:
    export = tmp_path / "project" / "scene_export"
    assets = export / "mesh_assets"
    assets.mkdir(parents=True)
    (assets / "table.glb").write_bytes(b"glTF-table")
    (assets / "cup.glb").write_bytes(b"glTF-cup")
    (export / "scene_config.json").write_text(
        json.dumps(
            {
                "format": "embodichain.scene-export/v1",
                "scene_id": "scene",
                "background": [
                    {
                        "uid": "table",
                        "shape": {
                            "shape_type": "Mesh",
                            "fpath": "mesh_assets/table.glb",
                        },
                    }
                ],
                "rigid_object": [
                    {
                        "uid": "cup",
                        "shape": {
                            "shape_type": "Mesh",
                            "fpath": "mesh_assets/cup.glb",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return export.parent


def test_blueprint_objects_keep_pose_description_orientation_unknown(
    tmp_path: Path,
) -> None:
    scene = Scene(
        objects=[
            SceneObject("table", "table", "table", "table", "A table."),
            SceneObject("cup", "asset", "cup", "red cup", "A red cup."),
        ]
    )
    graph = SceneGraph(
        nodes=[
            SceneGraphNode("table", None),
            SceneGraphNode(
                "cup",
                "table",
                "on",
                pose_description="Lie flat on the support surface.",
            ),
        ]
    )
    package = SceneBlueprintPackage(
        schema_version=SCENE_BLUEPRINT_SCHEMA,
        blueprint_id="blueprint",
        image_path=tmp_path / "input.png",
        output_root=tmp_path,
        manifest_path=tmp_path / "scene_blueprint.json",
        scene=scene,
        scene_graph=graph,
    )

    objects = scene_blueprint_objects(package)

    cup = next(item for item in objects if item["uid"] == "cup")
    assert cup["description"] == "A red cup."
    assert cup["initial_state"] == {}
    assert cup["affordances"] == []
    assert cup["init_pos"] == [0.0, 0.0, 0.0]


def test_backend_select_passes_v2_blueprint_contract(tmp_path: Path) -> None:
    scene = Scene(objects=[SceneObject("table", "table", "table", "table", "A table.")])
    graph = SceneGraph(nodes=[SceneGraphNode("table", None)])
    package = SceneBlueprintPackage(
        schema_version=SCENE_BLUEPRINT_SCHEMA,
        blueprint_id="blueprint",
        image_path=tmp_path / "input.png",
        output_root=tmp_path,
        manifest_path=tmp_path / "scene_blueprint.json",
        scene=scene,
        scene_graph=graph,
    )
    analysis = SceneAnalysis(
        input_kind="image",
        source=package.image_path,
        blueprint=package,
        source_fingerprint=None,
    )
    captured = {}
    marker = object()

    class CapturingAdapter:
        def select_objects(self, candidate_set, scene_objects, **kwargs):
            captured["candidate_set"] = candidate_set
            captured["scene_objects"] = scene_objects
            captured.update(kwargs)
            return marker

    result = SceneEngineBackend().select(
        analysis,
        {"candidate": "value"},
        CapturingAdapter(),
        force_most_likely=True,
    )

    assert result is marker
    assert captured["source_format"] == "embodichain.scene-blueprint/v2"
    assert captured["scene_objects"][0]["uid"] == "table"


def test_existing_scene_edit_creates_revision_and_never_writes_source(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = _scene_export(tmp_path)
    source_config = project / "scene_export" / "scene_config.json"
    source_value = json.loads(source_config.read_text(encoding="utf-8"))
    articulation_path = project / "scene_export" / "cabinet.urdf"
    articulation_path.write_text(
        '<robot name="cabinet"><link name="base"/></robot>\n',
        encoding="utf-8",
    )
    source_value["articulation"] = [
        {
            "uid": "cabinet",
            "name": "cabinet",
            "description": "A fixed cabinet.",
            "category": "cabinet",
            "fpath": "cabinet.urdf",
        }
    ]
    source_config.write_text(json.dumps(source_value), encoding="utf-8")
    original = source_config.read_bytes()
    prompts: list[str] = []

    def fake_analyze_edit(*, output_root, edit_prompt):
        prompts.append(edit_prompt)
        return SimpleNamespace(
            output_root=Path(output_root),
            scene_edit_plan=SimpleNamespace(
                to_dict=lambda: {"operations": [{"op": "move", "object_id": "cup"}]}
            ),
        )

    def fake_materialize_edit(blueprint):
        return SceneMaterialization(
            scene=Scene(),
            scene_graph=SceneGraph(nodes=[SceneGraphNode("table", None)]),
            output_root=blueprint.output_root,
            scene_config_path=blueprint.output_root
            / "scene_export"
            / "scene_config.json",
        )

    monkeypatch.setattr(scene_backend_module, "analyze_edit", fake_analyze_edit)
    monkeypatch.setattr(scene_backend_module, "materialize_edit", fake_materialize_edit)
    backend = SceneEngineBackend()
    request = _request(tmp_path, project, edit="Move the cup left.")
    analysis = backend.analyze(request, tmp_path / "analysis")

    revision = backend.materialize(
        analysis,
        request,
        tmp_path / "revision",
        seed=7,
    )

    assert prompts == ["Move the cup left."]
    assert revision.source != source_config
    assert revision.source.is_file()
    assert len(revision.revision_id) == 64
    assert revision.edit_plan == {"operations": [{"op": "move", "object_id": "cup"}]}
    assert source_config.read_bytes() == original
    revision_config = json.loads(revision.source.read_text(encoding="utf-8"))
    assert revision_config["articulation"][0]["uid"] == "cabinet"
    audit = json.loads(
        (tmp_path / "revision" / "scene_revision_attempt.json").read_text(
            encoding="utf-8"
        )
    )
    assert audit["seed"] == 7
    assert audit["revision_id"] == revision.revision_id
    assert audit["edit_plan"] == revision.edit_plan


def test_final_inspection_rejects_scene_changed_after_revision(tmp_path: Path) -> None:
    project = _scene_export(tmp_path)
    backend = SceneEngineBackend()
    request = _request(tmp_path, project, edit=None)
    revision = backend.materialize(
        backend.analyze(request, tmp_path / "analysis"),
        request,
        tmp_path / "unused",
        seed=0,
    )
    config_path = project / "scene_export" / "scene_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["rigid_object"][0]["init_pos"] = [0.25, 0.0, 0.0]
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(RuntimeError, match="changed before geometry inspection"):
        backend.inspect(revision, tmp_path / "inspection.json")
