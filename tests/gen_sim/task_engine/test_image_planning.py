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

"""Image planning preserves visible identities and stops before execution."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

from PIL import Image
import pytest

from embodichain.cli.main import main as dispatch
from embodichain.gen_sim.task_engine.image_planning import (
    generate_semantic_task_graph_from_image,
)


def _intent() -> dict:
    fixture = json.loads(
        (Path(__file__).parent / "data/semantic_graph_contract.json").read_text()
    )
    step = fixture["candidate"]["draft"]["steps"][0]
    step.update(task_type="E1", relation="inside", orientation_goal="none")
    step["object"]["reference"] = "red cup"
    step["target"] = dict(step["object"], reference="tray")
    return {"steps": [step]}


def _grounding() -> dict:
    return {
        "objects": [
            {"id": "red_cup", "description": "red cup", "bbox": [200, 400, 400, 800]},
            {"id": "tray", "description": "tray", "bbox": [500, 200, 900, 600]},
        ],
        "bindings": {"step_01.object": ["red_cup"], "step_01.target": ["tray"]},
    }


@pytest.fixture
def source_image(tmp_path: Path) -> Path:
    path = tmp_path / "scene.png"
    Image.new("RGB", (100, 80), "white").save(path)
    return path


def _plan(
    image: Path, *, grounding: dict | None = None, intent: dict | None = None
) -> dict:
    def image_caller(**kwargs):
        assert kwargs["image_path"] == image
        assert "red cup" in kwargs["user_prompt"]
        return json.dumps(_grounding() if grounding is None else grounding)

    return generate_semantic_task_graph_from_image(
        image,
        "把红色杯子放入托盘",
        task_id="cup_in_tray",
        instruction_caller=lambda **_: _intent() if intent is None else intent,
        image_caller=image_caller,
    )


def test_image_plan_preserves_inside_relation_and_reports_only_static_coverage(
    source_image,
):
    before = source_image.read_bytes()
    result = _plan(source_image)
    graph = result["semantic_task_graph"]
    pick, place = graph["nodes"]
    assert pick["call"] == {"kind": "pick", "object": "red_cup", "resources": {}}
    assert place["call"]["arguments"] == {
        "object": "red_cup",
        "reference": "tray",
        "relation": "inside",
    }
    assert place["depends_on"] == [pick["id"]]
    assert graph["targets"] == {}
    assert graph["integration_fingerprint"] == "0" * 64
    coverage = result["semantic_graph_validation"]["coverage"]
    assert coverage["status"] == "covered"
    assert coverage["runtime_evidence_status"] == "not_run"
    assert (
        result["image_grounding"]["image_sha256"] == hashlib.sha256(before).hexdigest()
    )
    assert source_image.read_bytes() == before
    assert result["image_grounding"]["objects"][0]["bbox"] == [0.2, 0.4, 0.4, 0.8]


@pytest.mark.parametrize("matches", [[], ["red_cup", "tray"], ["unknown"]])
def test_missing_ambiguous_or_foreign_image_matches_reject(source_image, matches):
    grounding = _grounding()
    grounding["bindings"]["step_01.object"] = matches
    with pytest.raises(ValueError, match="exactly one visible object"):
        _plan(source_image, grounding=grounding)


def test_same_source_and_destination_reject(source_image):
    grounding = _grounding()
    grounding["bindings"]["step_01.target"] = ["red_cup"]
    with pytest.raises(ValueError, match="cannot also be its destination"):
        _plan(source_image, grounding=grounding)


@pytest.mark.parametrize(
    "box", [[100, 200, 100, 600], [-1, 200, 400, 600], [0.2, 0.4, 0.6, 0.8]]
)
def test_image_boxes_reject_invalid_geometry_or_wrong_coordinate_units(
    source_image, box
):
    grounding = _grounding()
    grounding["objects"][0]["bbox"] = box
    with pytest.raises(ValueError, match="integer xyxy"):
        _plan(source_image, grounding=grounding)


def test_unsupported_intention_rejects_before_image_provider(source_image):
    intent = _intent()
    intent["steps"][0].update(
        task_type="E2", relation="none", orientation_goal="upright"
    )
    intent["steps"][0]["target"].update(kind="none", reference="")
    with pytest.raises(ValueError, match="currently supports only E1"):
        _plan(source_image, intent=intent)


def test_cli_writes_current_viewer_and_preserves_existing_output(
    source_image, tmp_path, monkeypatch
):
    from embodichain.gen_sim.task_engine import cli

    result = _plan(source_image)

    def generate(*args, **kwargs):
        assert "model" not in kwargs
        return deepcopy(result)

    monkeypatch.setattr(cli, "generate_semantic_task_graph_from_image", generate)
    output = tmp_path / "preview"
    args = [
        "task-engine",
        "--image",
        str(source_image),
        "--instruction",
        "把红色杯子放入托盘",
        "--task-id",
        "cup_in_tray",
        "--output",
        str(output),
    ]
    dispatch(args)
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["execution_ready"] is False
    assert manifest["runtime_evidence_status"] == "not_run"
    graph = json.loads((output / "semantic_task_graph.json").read_text())
    assert graph == result["semantic_task_graph"]
    html = (output / "semantic_task_graph.html").read_text()
    assert "red_cup" in html and "semantic_task_graph/v1" in html
    assert '"name":"red cup"' in html and '"cutout":"data:image/png;base64,' in html
    assert (output / "source_image.png").read_bytes() == source_image.read_bytes()
    with pytest.raises(FileExistsError):
        dispatch(args)
    assert (output / "semantic_task_graph.html").read_text() == html


def test_dependent_moves_preserve_group_and_call_order(source_image):
    intent = _intent()
    second = deepcopy(intent["steps"][0])
    second.update(id="step_02", relation="on", depends_on=["step_01"])
    second["target"]["reference"] = "table"
    intent["steps"].append(second)
    grounding = _grounding()
    grounding["objects"].append(
        {"id": "table", "description": "table", "bbox": [0, 0, 1000, 1000]}
    )
    grounding["bindings"].update(
        {"step_02.object": ["red_cup"], "step_02.target": ["table"]}
    )
    result = _plan(source_image, intent=intent, grounding=grounding)
    graph = result["semantic_task_graph"]
    assert graph["nodes"][2]["depends_on"] == [graph["nodes"][1]["id"]]
    assert graph["task_groups"][1]["depends_on"] == ["step_01"]
    assert graph["nodes"][3]["call"]["arguments"] == {
        "object": "red_cup",
        "reference": "table",
        "relation": "on",
    }
    assert result["semantic_graph_validation"]["coverage"]["status"] == "covered"
