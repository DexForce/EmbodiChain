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

from embodichain.gen_sim.scene_engine.core.scene import Scene
from embodichain.gen_sim.scene_engine.core.scene_edit_plan import SceneEditPlan
from embodichain.gen_sim.scene_engine.core.scene_graph import (
    SceneGraph,
    SceneGraphNode,
)
from embodichain.gen_sim.scene_engine.core.scene_object import SceneObject
from embodichain.gen_sim.scene_engine.pipeline import api


class _HealthyClient:
    def __init__(self) -> None:
        self.health_checks = 0

    def check_health(self) -> None:
        self.health_checks += 1


def _table_scene() -> tuple[Scene, SceneGraph]:
    scene = Scene(
        objects=[
            SceneObject(
                id="table",
                kind="table",
                category="table",
                name="table",
                description="A work table.",
            )
        ]
    )
    graph = SceneGraph(nodes=[SceneGraphNode(object_id="table", parent_id=None)])
    return scene, graph


def test_analyze_image_persists_blueprint_and_artifact_hashes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    image_path = tmp_path / "input.png"
    image_path.write_bytes(b"image")
    scene, graph = _table_scene()

    def fake_understand_scene(**kwargs):
        stage_root = Path(kwargs["output_root"]) / "scene_understanding"
        stage_root.mkdir(parents=True)
        (stage_root / "table-mask.png").write_bytes(b"mask")
        return scene, graph

    monkeypatch.setattr(api, "understand_scene", fake_understand_scene)
    segmentation = _HealthyClient()
    package = api.analyze_image(
        image_path,
        tmp_path / "output",
        vlm_client=object(),
        image_segmentation_client=segmentation,
    )
    document = json.loads(package.manifest_path.read_text(encoding="utf-8"))

    assert segmentation.health_checks == 1
    assert document["blueprint_id"] == package.blueprint_id
    assert document["scene_graph"] == graph.to_dict()
    assert document["artifacts"][0]["path"].endswith("table-mask.png")
    assert len(document["artifacts"][0]["sha256"]) == 64


def test_analyze_edit_persists_post_edit_blueprint(
    tmp_path: Path,
    monkeypatch,
) -> None:
    scene, graph = _table_scene()
    plan = SceneEditPlan(scene=scene, scene_graph=graph, operations=[])

    class FakeImporter:
        def __init__(self, *, output_root: Path) -> None:
            self.output_root = output_root

        def import_scene_and_graph(self):
            return scene, graph

    monkeypatch.setattr(api, "SceneExportImporter", FakeImporter)
    monkeypatch.setattr(
        api,
        "understand_scene_edit",
        lambda **_: (plan, graph),
    )
    package = api.analyze_edit(
        output_root=tmp_path,
        edit_prompt="Keep the scene unchanged.",
        vlm_client=object(),
    )
    document = json.loads(package.manifest_path.read_text(encoding="utf-8"))

    assert document["blueprint_id"] == package.blueprint_id
    assert document["scene_edit_plan"] == plan.to_dict()
    assert document["updated_scene_graph"] == graph.to_dict()
