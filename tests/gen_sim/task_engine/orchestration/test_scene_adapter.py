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
import pytest
from embodichain.gen_sim.task_engine.contracts import (
    SCENE_REQUEST_SCHEMA,
    SUCCESS_SPEC_SCHEMA,
    TASK_CANDIDATE_SET_SCHEMA,
    TASK_DRAFT_SCHEMA,
    canonical_hash,
)
from embodichain.gen_sim.task_engine.orchestration.scene_adapter import (
    SceneAdapter,
    SceneAdapterProtocolError,
)
from embodichain.gen_sim.task_engine.orchestration.scene_source import (
    SceneSourceRef,
    fingerprint_scene_source,
    verify_scene_source_fingerprint,
)

_UPRIGHT_CAN_INSTRUCTION = "test-instruction"


@pytest.fixture
def scene_export(tmp_path: Path) -> Path:
    export = tmp_path / "scene_export"
    assets = export / "meshes"
    assets.mkdir(parents=True)
    for name in ("table", "red_can", "blue_can"):
        (assets / f"{name}.glb").write_bytes(f"mesh:{name}".encode())
    config = {
        "format": "embodichain.scene-export/v1",
        "scene_id": "2026-03-18T10:20:30Z",
        "background": [
            {
                "uid": "table",
                "name": "table",
                "description": "A work table.",
                "category": "table",
                "affordances": ["support_surface"],
                "shape": {"shape_type": "Mesh", "fpath": "meshes/table.glb"},
                "init_pos": [0.0, 0.0, 0.0],
                "init_rot": [0.0, 0.0, 0.0],
                "body_scale": [1.0, 1.0, 1.0],
            }
        ],
        "rigid_object": [
            {
                "uid": f"{color}_can",
                "name": f"{color} can",
                "description": f"A {color} soda can.",
                "category": "can",
                "attributes": {
                    "color": color,
                    "geometry": {"position": [1.0, 2.0, 3.0]},
                },
                "affordances": ["graspable", "orientable", "placeable"],
                "initial_state": {"orientation": "fallen"},
                "shape": {
                    "shape_type": "Mesh",
                    "fpath": f"meshes/{color}_can.glb",
                },
                "init_pos": [0.0, offset, 0.7],
                "init_rot": [0.0, 0.0, 90.0],
                "body_scale": [1.0, 1.0, 1.0],
            }
            for color, offset in (("red", 0.2), ("blue", -0.2))
        ],
    }
    (export / "scene_config.json").write_text(json.dumps(config), encoding="utf-8")
    return export


def _selector(reference: str) -> dict:
    return {
        "kind": "scene_ref",
        "step_id": "",
        "reference": reference,
        "quantifier": "one",
        "count": 0,
    }


def _none_selector() -> dict:
    return {
        "kind": "none",
        "step_id": "",
        "reference": "",
        "quantifier": "one",
        "count": 0,
    }


def _candidate(candidate_id: str, reference: str, *, votes: int = 1) -> dict:
    step = {
        "id": "upright",
        "task_type": "E2",
        "object": _selector(reference),
        "target": _none_selector(),
        "relation": "none",
        "required_arm": "auto",
        "transfer_arm": "none",
        "receive_arm": "none",
        "orientation_goal": "upright",
        "target_state": "none",
        "target_setting": 0,
        "layout": "none",
        "axis": "none",
        "direction": "none",
        "terminal_behavior": "none",
        "depends_on": [],
    }
    draft = {
        "schema_version": TASK_DRAFT_SCHEMA,
        "task_id": "upright_can",
        "instruction": _UPRIGHT_CAN_INSTRUCTION,
        "steps": [step],
    }
    return {
        "candidate_id": candidate_id,
        "draft": draft,
        "scene_request": {
            "schema_version": SCENE_REQUEST_SCHEMA,
            "task_id": "upright_can",
            "references": [
                {
                    "reference_id": "upright.object",
                    "step_id": "upright",
                    "role": "object",
                    "reference": reference,
                    "quantifier": "one",
                    "count": 0,
                    "source_structure": "rigid_object",
                    "affordances": ["graspable", "orientable"],
                    "initial_state": {},
                    "attributes": {},
                }
            ],
        },
        "success_spec": {
            "schema_version": SUCCESS_SPEC_SCHEMA,
            "task_id": "upright_can",
            "op": "all",
            "terms": [{"step_id": "upright", "type": "object_upright"}],
        },
        "semantic_hash": canonical_hash([step]),
        "vote_count": votes,
        "attempts": 1,
        "normalizations": [],
    }


def _candidate_set(candidates: list[dict]) -> dict:
    return {
        "schema_version": TASK_CANDIDATE_SET_SCHEMA,
        "task_id": "upright_can",
        "instruction": _UPRIGHT_CAN_INSTRUCTION,
        "candidates": candidates,
        "requested_candidate_count": sum(item["vote_count"] for item in candidates),
        "valid_response_count": sum(item["vote_count"] for item in candidates),
        "errors": [],
    }


def _placement_candidate(candidate_id: str = "place") -> dict:
    candidate = _candidate(candidate_id, "red can")
    step = candidate["draft"]["steps"][0]
    step.update(
        {
            "task_type": "E1",
            "target": _selector("table"),
            "relation": "on",
            "orientation_goal": "preserve",
        }
    )
    candidate["scene_request"]["references"] = [
        {
            "reference_id": "upright.object",
            "step_id": "upright",
            "role": "object",
            "reference": "red can",
            "quantifier": "one",
            "count": 0,
            "source_structure": "rigid_object",
            "affordances": ["graspable", "placeable"],
            "initial_state": {},
            "attributes": {},
        },
        {
            "reference_id": "upright.target",
            "step_id": "upright",
            "role": "target",
            "reference": "table",
            "quantifier": "one",
            "count": 0,
            "source_structure": "physical_entity",
            "affordances": [],
            "initial_state": {},
            "attributes": {},
        },
    ]
    candidate["success_spec"]["terms"] = [
        {"step_id": "upright", "type": "semantic_goal"}
    ]
    candidate["semantic_hash"] = canonical_hash([step])
    return candidate


def _grounder(**kwargs) -> dict:
    prompt = kwargs["prompt"]
    uid = "blue_can" if '"reference": "blue can"' in prompt else "red_can"
    return {
        "bindings": [
            {
                "reference_id": "upright.object",
                "status": "resolved",
                "uids": [uid],
                "confidence": 0.95,
            }
        ]
    }


def test_scene_adapter_selects_bindable_majority_and_redacts_manifest(
    scene_export: Path,
) -> None:
    red = _candidate("red-majority", "red can", votes=2)
    blue = _candidate("blue-minority", "blue can")
    result = SceneAdapter(grounding_caller=_grounder).adapt(
        _candidate_set([red, blue]),
        scene_export,
    )

    hierarchy_by_uid = {
        node["uid"]: node for node in result.conservative_scene_graph["nodes"]
    }
    assert hierarchy_by_uid["red_can"]["parent_uid"] == "unknown"
    assert hierarchy_by_uid["red_can"]["parent_relation"] == "unknown"

    assert result.binding_report["status"] == "bound"
    assert result.binding_report["candidates"][0]["status"] == "resolved"
    assert result.selected_candidate_id == "red-majority"
    assert result.reference_bindings == {"upright.object": ["red_can"]}
    assert result.role_bindings["role_bindings"] == {}
    red_manifest = next(
        item for item in result.scene_manifest["objects"] if item["uid"] == "red_can"
    )
    assert "position" not in json.dumps(red_manifest)
    assert (
        result.prepared_scene.source_config_path == scene_export / "scene_config.json"
    )
    assert result.static_scene_manifest is not None
    static_by_uid = {
        item["uid"]: item for item in result.static_scene_manifest["objects"]
    }
    assert static_by_uid["red_can"]["physics"]["body_type"] == "dynamic"
    assert static_by_uid["red_can"]["geometry"]["asset_sha256"]


def test_scene_adapter_accepts_direct_source_and_rejects_bad_protocol(
    scene_export: Path,
) -> None:
    candidate = _candidate("red", "red can")
    direct = SceneAdapter(grounding_caller=_grounder).adapt(
        _candidate_set([candidate]),
        scene_export,
    )
    result = SceneAdapter(grounding_caller=_grounder).adapt(
        _candidate_set([candidate]),
        SceneSourceRef(scene_export),
    )
    assert result.scene_manifest == direct.scene_manifest
    assert result.role_bindings == direct.role_bindings

    with pytest.raises(SceneAdapterProtocolError, match="unsupported fields"):
        SceneAdapter(
            grounding_caller=lambda **_kwargs: {
                "bindings": [
                    {
                        "reference_id": "upright.object",
                        "status": "not_found",
                        "uids": [],
                        "confidence": 0.0,
                        "invented": True,
                    }
                ]
            }
        ).adapt(_candidate_set([candidate]), scene_export)


def test_scene_adapter_accepts_passive_support_target_and_rejects_self_reference(
    scene_export: Path,
) -> None:
    candidate = _placement_candidate()

    def place_on_table(**_kwargs):
        return {
            "bindings": [
                {
                    "reference_id": "upright.object",
                    "status": "resolved",
                    "uids": ["red_can"],
                    "confidence": 0.95,
                },
                {
                    "reference_id": "upright.target",
                    "status": "resolved",
                    "uids": ["table"],
                    "confidence": 0.95,
                },
            ]
        }

    bound = SceneAdapter(grounding_caller=place_on_table).adapt(
        _candidate_set([candidate]), scene_export
    )
    assert bound.binding_report["status"] == "bound"
    assert bound.reference_bindings["upright.target"] == ["table"]

    def self_reference(**_kwargs):
        response = place_on_table()
        response["bindings"][1]["uids"] = ["red_can"]
        return response

    incompatible = SceneAdapter(grounding_caller=self_reference).adapt(
        _candidate_set([candidate]), scene_export
    )
    assert incompatible.binding_report["status"] == "unsatisfied"
    assert incompatible.binding_report["candidates"][0]["status"] == "incompatible"


@pytest.mark.parametrize("kind", ["asset", "config"])
def test_scene_source_verification_rejects_later_mutation(
    scene_export: Path, kind
) -> None:
    expected = fingerprint_scene_source(scene_export).to_dict()
    if kind == "asset":
        (scene_export / "meshes" / "red_can.glb").write_bytes(b"changed later")
    else:
        path = scene_export / "scene_config.json"
        value = json.loads(path.read_text())
        value["rigid_object"][0]["init_pos"] = [0, 0, 1]
        path.write_text(json.dumps(value))

    with pytest.raises(RuntimeError, match="changed after Task Engine preparation"):
        verify_scene_source_fingerprint(expected)
