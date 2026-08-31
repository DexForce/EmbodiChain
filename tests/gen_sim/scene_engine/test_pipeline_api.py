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

from copy import deepcopy
import json
from pathlib import Path

import pytest

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


class _OwnedClient(_HealthyClient):
    def __init__(self) -> None:
        super().__init__()
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


def _materialization(
    *,
    scene: Scene,
    scene_graph: SceneGraph,
    output_root: Path,
) -> api.SceneMaterialization:
    return api.SceneMaterialization(
        scene=scene,
        scene_graph=scene_graph,
        output_root=output_root,
        scene_config_path=output_root / "scene_export" / "scene_config.json",
    )


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
    assert package.schema_version == api.SCENE_BLUEPRINT_SCHEMA
    assert document["schema_version"] == "embodichain.scene-blueprint/v2"
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

    assert package.schema_version == api.SCENE_EDIT_BLUEPRINT_SCHEMA
    assert document["schema_version"] == "embodichain.scene-edit-blueprint/v2"
    assert document["blueprint_id"] == package.blueprint_id
    assert document["scene_edit_plan"] == plan.to_dict()
    assert document["updated_scene_graph"] == graph.to_dict()


def test_materialize_blueprint_does_not_mutate_audited_snapshot(
    tmp_path: Path,
    monkeypatch,
) -> None:
    scene, graph = _table_scene()
    manifest_path = tmp_path / "scene_blueprint.json"
    manifest_path.write_text("audited blueprint\n", encoding="utf-8")
    package = api.SceneBlueprintPackage(
        schema_version=api.SCENE_BLUEPRINT_SCHEMA,
        blueprint_id="blueprint",
        image_path=tmp_path / "input.png",
        output_root=tmp_path,
        manifest_path=manifest_path,
        scene=scene,
        scene_graph=graph,
    )
    original_scene = deepcopy(scene.to_dict())
    original_graph = deepcopy(graph.to_dict())

    def fake_generate_scene_and_refine(**kwargs):
        assert "seed" not in kwargs
        assert kwargs["articulated_generation_client"] is None
        assert kwargs["scene"] is not package.scene
        assert kwargs["scene_graph"] is not package.scene_graph
        kwargs["scene"].objects[0].name = "materialized table"
        return kwargs["scene"]

    monkeypatch.setattr(
        api,
        "generate_scene_and_refine",
        fake_generate_scene_and_refine,
    )
    monkeypatch.setattr(
        api,
        "_export_materialization",
        lambda *, scene, scene_graph, output_root: _materialization(
            scene=scene,
            scene_graph=scene_graph,
            output_root=output_root,
        ),
    )

    result = api.materialize_blueprint(
        package,
        vlm_client=object(),
        geometry_generation_client=_HealthyClient(),
    )

    assert result.scene.objects[0].name == "materialized table"
    assert package.scene.to_dict() == original_scene
    assert package.scene_graph.to_dict() == original_graph
    assert manifest_path.read_text(encoding="utf-8") == "audited blueprint\n"


def test_materialize_edit_does_not_mutate_audited_snapshot(
    tmp_path: Path,
    monkeypatch,
) -> None:
    scene, graph = _table_scene()
    plan = SceneEditPlan(scene=scene, scene_graph=graph, operations=[])
    manifest_path = tmp_path / "scene_edit_blueprint.json"
    manifest_path.write_text("audited edit blueprint\n", encoding="utf-8")
    package = api.SceneEditBlueprintPackage(
        schema_version=api.SCENE_EDIT_BLUEPRINT_SCHEMA,
        blueprint_id="edit-blueprint",
        edit_prompt="Keep the scene unchanged.",
        output_root=tmp_path,
        manifest_path=manifest_path,
        scene_edit_plan=plan,
        updated_scene_graph=graph,
    )
    original_plan = deepcopy(plan.to_dict())
    original_graph = deepcopy(graph.to_dict())

    def fake_prepare_scene_edit_assets(**kwargs):
        assert "seed" not in kwargs
        return []

    monkeypatch.setattr(
        api, "prepare_scene_edit_assets", fake_prepare_scene_edit_assets
    )

    def fake_edit_layout(**kwargs):
        assert kwargs["scene_edit_plan"] is not package.scene_edit_plan
        assert kwargs["updated_scene_graph"] is not package.updated_scene_graph
        kwargs["scene"].objects[0].name = "edited table"
        return kwargs["scene"]

    monkeypatch.setattr(api, "edit_layout", fake_edit_layout)
    monkeypatch.setattr(
        api,
        "_export_materialization",
        lambda *, scene, scene_graph, output_root: _materialization(
            scene=scene,
            scene_graph=scene_graph,
            output_root=output_root,
        ),
    )
    clients = [_HealthyClient(), _HealthyClient(), _HealthyClient()]

    result = api.materialize_edit(
        package,
        vlm_client=object(),
        image_generation_client=clients[0],
        geometry_generation_client=clients[1],
        image_segmentation_client=clients[2],
    )

    assert result.scene.objects[0].name == "edited table"
    assert package.scene_edit_plan.to_dict() == original_plan
    assert package.updated_scene_graph.to_dict() == original_graph
    assert manifest_path.read_text(encoding="utf-8") == "audited edit blueprint\n"


def test_scene_blueprint_package_rejects_v1_schema(tmp_path: Path) -> None:
    scene, graph = _table_scene()

    with pytest.raises(ValueError, match="scene-blueprint/v2"):
        api.SceneBlueprintPackage(
            schema_version="embodichain.scene-blueprint/v1",
            blueprint_id="legacy",
            image_path=tmp_path / "input.png",
            output_root=tmp_path,
            manifest_path=tmp_path / "scene_blueprint.json",
            scene=scene,
            scene_graph=graph,
        )


def test_scene_edit_blueprint_package_rejects_v1_schema(tmp_path: Path) -> None:
    scene, graph = _table_scene()
    plan = SceneEditPlan(scene=scene, scene_graph=graph, operations=[])

    with pytest.raises(ValueError, match="scene-edit-blueprint/v2"):
        api.SceneEditBlueprintPackage(
            schema_version="embodichain.scene-edit-blueprint/v1",
            blueprint_id="legacy-edit",
            edit_prompt="Keep the scene unchanged.",
            output_root=tmp_path,
            manifest_path=tmp_path / "scene_edit_blueprint.json",
            scene_edit_plan=plan,
            updated_scene_graph=graph,
        )


def test_materialize_blueprint_owns_articulated_client_lifecycle(
    tmp_path: Path,
    monkeypatch,
) -> None:
    scene, graph = _table_scene()
    scene.objects.append(
        SceneObject(
            id="microwave_001",
            kind="asset",
            category="microwave",
            name="microwave",
            description="An articulated microwave.",
            is_articulated=True,
        )
    )
    graph.nodes.append(
        SceneGraphNode(
            object_id="microwave_001",
            parent_id="table",
            parent_relation="on",
            pose_description="Stand upright on its base.",
        )
    )
    package = api.SceneBlueprintPackage(
        schema_version=api.SCENE_BLUEPRINT_SCHEMA,
        blueprint_id="articulated",
        image_path=tmp_path / "input.png",
        output_root=tmp_path,
        manifest_path=tmp_path / "scene_blueprint.json",
        scene=scene,
        scene_graph=graph,
    )
    articulated = _OwnedClient()
    monkeypatch.setattr(
        api.ArticulatedGenerationClient,
        "from_dotenv",
        lambda: articulated,
    )

    def fake_generate_scene_and_refine(**kwargs):
        assert kwargs["articulated_generation_client"] is articulated
        return kwargs["scene"]

    monkeypatch.setattr(
        api, "generate_scene_and_refine", fake_generate_scene_and_refine
    )
    monkeypatch.setattr(
        api,
        "_export_materialization",
        lambda *, scene, scene_graph, output_root: _materialization(
            scene=scene,
            scene_graph=scene_graph,
            output_root=output_root,
        ),
    )

    api.materialize_blueprint(
        package,
        vlm_client=object(),
        geometry_generation_client=_HealthyClient(),
    )

    assert articulated.health_checks == 1
    assert articulated.close_calls == 1
