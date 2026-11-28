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

from collections.abc import Callable
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import ModuleType

import pytest
from PIL import Image

from embodichain.gen_sim.scene_engine.core.scene import Scene
from embodichain.gen_sim.scene_engine.core.scene_edit_plan import (
    SceneEditPlan,
    SceneEditOperation,
)
from embodichain.gen_sim.scene_engine.core.scene_graph import (
    SceneGraph,
    SceneGraphNode,
)
from embodichain.gen_sim.scene_engine.core.scene_object import SceneObject
from embodichain.gen_sim.scene_engine.pipeline import api, generate


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


def _write_task_spec(root: Path) -> Path:
    path = root / "task_spec.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "task_spec/v1",
                "inputs": {"instruction": "Open the microwave door."},
                "interpretation": {
                    "references": [
                        {
                            "entities": [
                                {
                                    "id": "evidence_001",
                                    "description": "A microwave with a hinged door.",
                                    "is_articulated": True,
                                }
                            ]
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize("with_task_spec", [False, True])
def test_analyze_image_persists_blueprint_and_artifact_hashes(
    tmp_path: Path,
    monkeypatch,
    with_task_spec: bool,
) -> None:
    image_path = tmp_path / "input.png"
    image_path.write_bytes(b"image")
    scene, graph = _table_scene()
    task_spec = _write_task_spec(tmp_path) if with_task_spec else None
    if task_spec is not None:
        document = json.loads(task_spec.read_text(encoding="utf-8"))
        document["inputs"]["image"] = {
            "sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
        }
        task_spec.write_text(json.dumps(document), encoding="utf-8")
    scene.table.table_state = "degraded"

    def fake_understand_scene(**kwargs):
        task_input = kwargs["task_input"]
        if with_task_spec:
            assert task_input.instruction == "Open the microwave door."
        else:
            assert task_input is None
        assert [
            (entity.id, entity.description, entity.is_articulated)
            for entity in (task_input.entities if task_input is not None else ())
        ] == (
            [("evidence_001", "A microwave with a hinged door.", True)]
            if with_task_spec
            else []
        )
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
        task_spec=task_spec,
    )
    document = json.loads(package.manifest_path.read_text(encoding="utf-8"))

    assert segmentation.health_checks == 1
    assert package.schema_version == api.SCENE_BLUEPRINT_SCHEMA
    assert document["schema_version"] == "embodichain.scene-blueprint/v2"
    assert document["blueprint_id"] == package.blueprint_id
    assert document["scene_graph"] == graph.to_dict()
    assert document["scene"]["objects"][0]["table_state"] == "degraded"
    assert document["artifacts"][0]["path"].endswith("table-mask.png")
    assert len(document["artifacts"][0]["sha256"]) == 64
    image_record = next(
        artifact
        for artifact in document["artifacts"]
        if artifact["path"] == image_path.as_posix()
    )
    assert image_record["sha256"] == hashlib.sha256(image_path.read_bytes()).hexdigest()
    if task_spec is not None:
        data = task_spec.read_bytes()
        assert document["artifacts"][-1] == {
            "path": task_spec.as_posix(),
            "sha256": hashlib.sha256(data).hexdigest(),
            "size": len(data),
        }


def test_analyze_image_rejects_task_spec_replaced_during_scene_understanding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_path = tmp_path / "input.png"
    image_path.write_bytes(b"image")
    task_spec = _write_task_spec(tmp_path)
    replacement = json.loads(task_spec.read_text(encoding="utf-8"))
    replacement["inputs"]["instruction"] = "Close the microwave door."
    replacement_path = tmp_path / "replacement.json"
    replacement_path.write_text(json.dumps(replacement), encoding="utf-8")
    output_root = tmp_path / "output"

    def replace_task_spec(**kwargs) -> tuple[Scene, SceneGraph]:
        assert kwargs["task_input"].instruction == "Open the microwave door."
        replacement_path.replace(task_spec)
        return _table_scene()

    monkeypatch.setattr(api, "understand_scene", replace_task_spec)
    with pytest.raises(ValueError, match="TaskSpec changed during scene analysis"):
        api.analyze_image(
            image_path,
            output_root,
            vlm_client=object(),
            image_segmentation_client=_HealthyClient(),
            task_spec=task_spec,
        )

    assert not (output_root / "scene_blueprint.json").exists()


def test_analyze_image_rejects_replaced_source_image_without_mixing_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_path = tmp_path / "input.png"
    replacement = tmp_path / "replacement.png"
    Image.new("RGB", (4, 4), "red").save(image_path)
    Image.new("RGB", (4, 4), "blue").save(replacement)
    original_bytes = image_path.read_bytes()
    replacement_bytes = replacement.read_bytes()
    task_spec = _write_task_spec(tmp_path)
    document = json.loads(task_spec.read_text(encoding="utf-8"))
    document["inputs"]["image"] = {
        "sha256": hashlib.sha256(original_bytes).hexdigest(),
    }
    task_spec.write_text(json.dumps(document), encoding="utf-8")
    output_root = tmp_path / "output"
    analysis_reads = []

    def replace_source_image(*, image_path: Path, **_) -> tuple[Scene, SceneGraph]:
        analysis_reads.append(image_path.read_bytes())
        replacement.replace(tmp_path / "input.png")
        analysis_reads.append(image_path.read_bytes())
        return _table_scene()

    monkeypatch.setattr(api, "understand_scene", replace_source_image)
    with pytest.raises(ValueError, match="Source image changed during scene analysis"):
        api.analyze_image(
            image_path,
            output_root,
            task_spec=task_spec,
            vlm_client=object(),
            image_segmentation_client=_HealthyClient(),
        )

    assert analysis_reads == [original_bytes, original_bytes]
    assert image_path.read_bytes() == replacement_bytes
    assert not (output_root / "scene_blueprint.json").exists()


def test_analyze_image_requires_image_for_task_spec(tmp_path, monkeypatch) -> None:
    def unexpected(*args, **kwargs):
        raise AssertionError("invalid input must fail before client setup")

    monkeypatch.setattr(api, "VisionFacade", unexpected)
    with pytest.raises(ValueError, match="requires an image"):
        api.analyze_image(None, tmp_path, task_spec=tmp_path / "task_spec.json")


@pytest.mark.parametrize(
    ("module", "entry"),
    [(api, api.analyze_image), (generate, generate.generate_scene_from_image)],
)
def test_scene_entry_points_reject_task_spec_for_a_different_image(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
    entry: Callable[..., object],
) -> None:
    image_path = tmp_path / "different.png"
    image_path.write_bytes(b"different image")
    task_spec = _write_task_spec(tmp_path)
    document = json.loads(task_spec.read_text(encoding="utf-8"))
    document["inputs"]["image"] = {
        "ref": "source.png",
        "sha256": hashlib.sha256(b"source image").hexdigest(),
    }
    task_spec.write_text(json.dumps(document), encoding="utf-8")

    def unexpected(*args, **kwargs):
        raise AssertionError("mismatched input must fail before client setup")

    monkeypatch.setattr(module, "VisionFacade", unexpected)
    with pytest.raises(ValueError, match="SHA-256 does not match"):
        entry(image_path, tmp_path / "output", task_spec=task_spec)
    assert not (tmp_path / "output").exists()


def test_analyze_edit_persists_post_edit_blueprint(
    tmp_path: Path,
    monkeypatch,
) -> None:
    scene, graph = _table_scene()
    plan = SceneEditPlan(scene=scene, scene_graph=graph, operations=[])

    class FakeImporter:
        def __init__(self, *, output_root: Path) -> None:
            self.output_root = output_root

        def import_scene_and_graph(self, *, write_scene_json: bool = True):
            assert not write_scene_json
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
    scene.table.table_state = "normal"
    manifest_path = tmp_path / "scene_blueprint.json"
    manifest_path.write_text("audited blueprint\n", encoding="utf-8")
    image_path = tmp_path / "input.png"
    image_path.write_bytes(b"source image")
    monkeypatch.setattr(api, "understand_scene", lambda **kwargs: (scene, graph))
    package = api.analyze_image(
        image_path,
        tmp_path,
        vlm_client=object(),
        image_segmentation_client=_HealthyClient(),
    )
    manifest_path = package.manifest_path
    audited_manifest = manifest_path.read_text(encoding="utf-8")

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
        lambda *, scene, scene_graph, output_root, vision_client: _materialization(
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
    assert result.scene.table.table_state == "normal"
    assert package.scene.to_dict() == original_scene
    assert package.scene_graph.to_dict() == original_graph
    assert manifest_path.read_text(encoding="utf-8") == audited_manifest


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
        lambda *, scene, scene_graph, output_root, vision_client: _materialization(
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


def test_materialize_edit_rechecks_floor_targets_before_asset_preparation(
    tmp_path: Path,
) -> None:
    scene, graph = _table_scene()
    container = SceneObject(
        id="bin", kind="asset", category="bin", name="bin", description="A blue bin."
    )
    scene.objects.append(container)
    graph.nodes.append(
        SceneGraphNode(object_id=container.id, parent_id="table", parent_relation="on")
    )
    plan = SceneEditPlan(
        scene=scene,
        scene_graph=graph,
        operations=[
            SceneEditOperation(
                op="add",
                object_id="cup",
                target_id=container.id,
                relation="left_of",
                category="cup",
                name="cup",
                description="A ceramic cup.",
            )
        ],
    )
    package = api.SceneEditBlueprintPackage(
        schema_version=api.SCENE_EDIT_BLUEPRINT_SCHEMA,
        blueprint_id="edit-blueprint",
        edit_prompt="Add a cup left of the bin.",
        output_root=tmp_path,
        manifest_path=tmp_path / "scene_edit_blueprint.json",
        scene_edit_plan=plan,
        updated_scene_graph=graph,
    )
    container.placement_support = "floor"
    image_client = _HealthyClient()

    with pytest.raises(ValueError, match="floor-supported objects: 'bin'"):
        api.materialize_edit(package, image_generation_client=image_client)

    assert image_client.health_checks == 0


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


@pytest.mark.parametrize("caller_owned", [False, True])
def test_materialize_edit_preserves_client_ownership_when_setup_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caller_owned: bool,
) -> None:
    """A later client constructor cannot leak an earlier owned HTTP session."""
    scene, graph = _table_scene()
    package = api.SceneEditBlueprintPackage(
        schema_version=api.SCENE_EDIT_BLUEPRINT_SCHEMA,
        blueprint_id="edit-blueprint",
        edit_prompt="Keep the scene unchanged.",
        output_root=tmp_path,
        manifest_path=tmp_path / "scene_edit_blueprint.json",
        scene_edit_plan=SceneEditPlan(
            scene=scene,
            scene_graph=graph,
            operations=[
                SceneEditOperation(
                    op="add",
                    object_id="new_mug",
                    target_id="table",
                    relation="on",
                    category="mug",
                    name="mug",
                    description="A mug",
                )
            ],
        ),
        updated_scene_graph=graph,
    )
    image_client = _OwnedClient()
    monkeypatch.setattr(api, "ImageFacade", lambda _: image_client)

    def fail_geometry_setup(_):
        raise RuntimeError("geometry configuration unavailable")

    monkeypatch.setattr(api, "GeometryFacade", fail_geometry_setup)
    with pytest.raises(RuntimeError, match="geometry configuration unavailable"):
        api.materialize_edit(
            package,
            vlm_client=object(),
            image_generation_client=image_client if caller_owned else None,
        )

    assert image_client.close_calls == (0 if caller_owned else 1)


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
            id="evidence_001",
            kind="asset",
            category="microwave",
            name="microwave",
            description="An articulated microwave.",
            is_articulated=True,
        )
    )
    graph.nodes.append(
        SceneGraphNode(
            object_id="evidence_001",
            parent_id="table",
            parent_relation="on",
            pose_description="Stand upright on its base.",
        )
    )
    image_path = tmp_path / "input.png"
    image_path.write_bytes(b"source image")
    monkeypatch.setattr(api, "understand_scene", lambda **kwargs: (scene, graph))
    package = api.analyze_image(
        image_path,
        tmp_path,
        vlm_client=object(),
        image_segmentation_client=_HealthyClient(),
        task_spec=_write_task_spec(tmp_path),
    )
    manifest_path = package.manifest_path
    audited_manifest = manifest_path.read_text(encoding="utf-8")

    articulated = _OwnedClient()
    monkeypatch.setattr(
        api,
        "ArticulationFacade",
        lambda _: articulated,
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
        lambda *, scene, scene_graph, output_root, vision_client: _materialization(
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


@pytest.mark.parametrize(
    "change", ["modify", "delete", "reanalyze", "image", "task_spec"]
)
def test_materialize_rejects_stale_observations_before_any_clients(
    tmp_path, monkeypatch, change
):
    image = tmp_path / "input.png"
    image.write_bytes(b"source")
    scene, graph = _table_scene()
    mask = tmp_path / "scene_understanding/mask.png"

    def understand(**kwargs):
        mask.parent.mkdir(exist_ok=True)
        mask.write_bytes(b"original")
        return scene, graph

    monkeypatch.setattr(api, "understand_scene", understand)
    task_spec = _write_task_spec(tmp_path) if change == "task_spec" else None
    package = api.analyze_image(
        image,
        tmp_path,
        vlm_client=object(),
        image_segmentation_client=_HealthyClient(),
        task_spec=task_spec,
    )
    if change == "modify":
        mask.write_bytes(b"changed")
    elif change == "delete":
        mask.unlink()
    elif change == "image":
        image.write_bytes(b"changed source")
    elif change == "task_spec":
        task_spec.write_text("{}", encoding="utf-8")
    else:
        image.write_bytes(b"second source")
        api.analyze_image(
            image,
            tmp_path,
            vlm_client=object(),
            image_segmentation_client=_HealthyClient(),
        )

    def unexpected(*args, **kwargs):
        raise AssertionError("stale observations must fail before client setup")

    monkeypatch.setattr(api, "VisionFacade", unexpected)
    monkeypatch.setattr(api, "GeometryFacade", unexpected)
    with pytest.raises(ValueError, match="stale or invalid"):
        api.materialize_blueprint(package)


@pytest.mark.parametrize("operation", ["move", "delete"])
def test_layout_only_edit_requires_no_generation_clients(
    tmp_path, monkeypatch, operation
):
    scene, graph = _table_scene()
    scene.objects.append(
        SceneObject(
            id="mug", kind="asset", category="mug", name="mug", description="mug"
        )
    )
    graph.nodes.append(
        SceneGraphNode(object_id="mug", parent_id="table", parent_relation="on")
    )
    plan = SceneEditPlan(
        scene=scene,
        scene_graph=graph,
        operations=[
            (
                SceneEditOperation(op="delete", object_id="mug")
                if operation == "delete"
                else SceneEditOperation(
                    op="move",
                    object_id="mug",
                    target_id="table",
                    relation="on",
                    table_region="left_center",
                )
            )
        ],
    )
    package = api.SceneEditBlueprintPackage(
        api.SCENE_EDIT_BLUEPRINT_SCHEMA,
        "edit",
        "delete mug",
        tmp_path,
        tmp_path / "edit.json",
        plan,
        graph,
    )

    def unexpected(*args, **kwargs):
        raise AssertionError("layout-only edits cannot depend on generation services")

    for client in (
        "VisionFacade",
        "ImageFacade",
        "GeometryFacade",
        "SegmentationFacade",
    ):
        monkeypatch.setattr(api, client, unexpected)
    monkeypatch.setattr(api, "prepare_scene_edit_assets", unexpected)
    monkeypatch.setattr(api, "edit_layout", lambda **kwargs: kwargs["scene"])
    monkeypatch.setattr(
        api,
        "_export_materialization",
        lambda *, scene, scene_graph, output_root, vision_client: _materialization(
            scene=scene, scene_graph=scene_graph, output_root=output_root
        ),
    )
    result = api.materialize_edit(package)
    assert result.scene.objects[1].id == "mug"


def test_importer_does_not_load_generation_api_or_matplotlib():
    import subprocess
    import sys

    code = """
import sys
import importlib.abc
class BlockGeneration(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in ('embodichain.gen_sim.scene_engine.pipeline.api', 'embodichain.gen_sim.scene_engine.pipeline.generation.scene_generation', 'matplotlib'):
            raise ImportError('unneeded generation dependency: ' + fullname)
sys.meta_path.insert(0, BlockGeneration())
from embodichain.gen_sim.scene_engine.pipeline.utils.scene_importer import SceneExportImporter
assert 'embodichain.gen_sim.scene_engine.pipeline.api' not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True
    )
