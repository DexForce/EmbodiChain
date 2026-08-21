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

import embodichain.gen_sim.task_engine.orchestration.scene_adapter as scene_adapter_module
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
from embodichain.gen_sim.task_engine.agent import (
    derive_scene_request,
    derive_success_spec,
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


def _legacy_gym_project(tmp_path: Path, filename: str) -> Path:
    project = tmp_path / filename.removesuffix(".json")
    assets = project / "assets"
    assets.mkdir(parents=True)
    for name in ("table.glb", "red_can.glb", "cabinet.urdf"):
        (assets / name).write_bytes(f"asset:{name}".encode())
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
                "uid": "red_can_0",
                "name": "red can",
                "description": "A red soda can.",
                "category": "can",
                "affordances": ["graspable", "orientable", "placeable"],
                "initial_state": {"orientation": "fallen"},
                "shape": {
                    "shape_type": "Mesh",
                    "fpath": "assets/red_can.glb",
                },
                "init_pos": [0.0, 0.2, 0.7],
                "init_rot": [0.0, 0.0, 90.0],
                "body_scale": [1.0, 1.0, 1.0],
            }
        ],
        "articulation": [
            {
                "uid": "cabinet_0",
                "name": "cabinet",
                "description": "A fixed articulated cabinet.",
                "category": "cabinet",
                "fpath": "assets/cabinet.urdf",
                "init_pos": [0.4, 0.0, 0.0],
                "init_rot": [0.0, 0.0, 0.0],
                "body_scale": [1.0, 1.0, 1.0],
            }
        ],
    }
    (project / filename).write_text(json.dumps(config), encoding="utf-8")
    return project


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
                    "initial_state": {"orientation": "fallen"},
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


def test_scene_source_fingerprint_reads_without_copying(scene_export: Path) -> None:
    before = sorted(path.relative_to(scene_export) for path in scene_export.rglob("*"))
    fingerprint = fingerprint_scene_source(SceneSourceRef(scene_export))
    after = sorted(path.relative_to(scene_export) for path in scene_export.rglob("*"))

    assert fingerprint.config_path == scene_export / "scene_config.json"
    assert len(fingerprint.config_sha256) == 64
    assert len(fingerprint.asset_sha256) == 3
    assert after == before


@pytest.mark.parametrize("filename", ["gym_config.json", "gym_config_merged.json"])
def test_scene_adapter_supports_legacy_gym_configs(
    tmp_path: Path,
    filename: str,
) -> None:
    project = _legacy_gym_project(tmp_path, filename)
    result = SceneAdapter(grounding_caller=_grounder).adapt(
        _candidate_set([_candidate("legacy", "red can")]),
        project,
    )

    assert result.selected_candidate_id == "legacy"
    assert result.static_scene_manifest["source_format"] == "legacy_gym_config"
    assert any(
        item["role"] == "articulation"
        for item in result.static_scene_manifest["objects"]
    )
    assert (
        result.prepared_scene.articulations[0]["fpath"]
        == (project / "assets" / "cabinet.urdf").resolve().as_posix()
    )


def test_scene_source_fingerprint_covers_articulation_fpath(tmp_path: Path) -> None:
    project = _legacy_gym_project(tmp_path, "gym_config.json")
    original = fingerprint_scene_source(project)
    articulation_path = project / "assets" / "cabinet.urdf"

    articulation_path.write_bytes(b"changed articulation")
    changed = fingerprint_scene_source(project)

    assert articulation_path.resolve().as_posix() in original.asset_sha256
    assert changed.asset_sha256 != original.asset_sha256
    with pytest.raises(RuntimeError, match="changed after Task Engine preparation"):
        verify_scene_source_fingerprint(original.to_dict())


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


def test_scene_adapter_returns_report_for_business_level_non_binding(
    scene_export: Path,
) -> None:
    candidate = _candidate("missing", "green can")

    def not_found(**_kwargs):
        return {
            "bindings": [
                {
                    "reference_id": "upright.object",
                    "status": "not_found",
                    "uids": [],
                    "confidence": 0.0,
                }
            ]
        }

    result = SceneAdapter(grounding_caller=not_found).adapt(
        _candidate_set([candidate]),
        scene_export,
    )

    assert result.selected_candidate is None
    assert result.role_bindings is None
    assert result.binding_report["status"] == "unsatisfied"
    assert (
        result.binding_report["candidates"][0]["references"][0]["status"] == "not_found"
    )


def test_semantic_blueprint_selection_forces_ranked_low_confidence_uid() -> None:
    candidate = _candidate("likely", "the can")
    scene_objects = [
        {
            "uid": "table",
            "role": "table",
            "name": "table",
            "description": "A work table.",
            "category": "table",
            "init_pos": [0.0, 0.0, 0.0],
            "affordances": ["support_surface"],
            "initial_state": {},
            "attributes": {},
        },
        *[
            {
                "uid": f"{color}_can",
                "role": "rigid_object",
                "name": f"{color} can",
                "description": f"A {color} can.",
                "category": "can",
                "init_pos": [0.0, offset, 0.1],
                "affordances": ["graspable", "orientable"],
                "initial_state": {"orientation": "fallen"},
                "attributes": {"color": color},
            }
            for color, offset in (("red", -0.1), ("blue", 0.1))
        ],
    ]

    def ambiguous(**_kwargs):
        return {
            "bindings": [
                {
                    "reference_id": "upright.object",
                    "status": "ambiguous",
                    "uids": ["red_can", "blue_can"],
                    "confidence": 0.2,
                }
            ]
        }

    result = SceneAdapter(grounding_caller=ambiguous).select_objects(
        _candidate_set([candidate]),
        scene_objects,
        force_most_likely=True,
    )

    assert result.selected_candidate_id == "likely"
    assert result.role_bindings["reference_bindings"] == {"upright.object": ["red_can"]}
    reference = result.binding_report["candidates"][0]["references"][0]
    assert reference["confidence"] == 0.2
    assert reference["candidate_uids"] == ["red_can", "blue_can"]
    assert reference["selected_uids"] == ["red_can"]
    assert reference["reasons"] == [
        "Forced the highest-ranked structurally compatible UID from an "
        "ambiguous low-confidence response."
    ]


def test_scene_adapter_uses_unique_bindable_then_injected_adjudication(
    scene_export: Path,
) -> None:
    red = _candidate("red", "red can")
    blue = _candidate("blue", "blue can")

    def one_missing(**kwargs):
        if '"reference": "red can"' in kwargs["prompt"]:
            return {
                "bindings": [
                    {
                        "reference_id": "upright.object",
                        "status": "not_found",
                        "uids": [],
                        "confidence": 0.0,
                    }
                ]
            }
        return _grounder(**kwargs)

    unique = SceneAdapter(grounding_caller=one_missing).adapt(
        _candidate_set([red, blue]),
        scene_export,
    )
    assert unique.selected_candidate_id == "blue"
    assert unique.binding_report["selection_reason"] == "unique_bindable"

    ambiguous = SceneAdapter(grounding_caller=_grounder).adapt(
        _candidate_set([red, blue]),
        scene_export,
    )
    assert ambiguous.binding_report["status"] == "ambiguous"

    adjudicated = SceneAdapter(
        grounding_caller=_grounder,
        adjudicator=lambda **_kwargs: {"candidate_id": "blue"},
    ).adapt(_candidate_set([red, blue]), scene_export)
    assert adjudicated.selected_candidate_id == "blue"
    assert adjudicated.binding_report["selection_reason"] == "adjudicated_bindable"


def test_scene_adapter_runs_one_default_structured_adjudication(
    scene_export: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adjudications = 0

    def caller(**kwargs):
        nonlocal adjudications
        if kwargs["schema"]["title"] == "ActionEngineTaskAdjudication":
            adjudications += 1
            return {"candidate_id": "blue"}
        return _grounder(**kwargs)

    monkeypatch.setattr(
        scene_adapter_module, "_default_grounding_caller", lambda: caller
    )
    result = SceneAdapter().adapt(
        _candidate_set([_candidate("red", "red can"), _candidate("blue", "blue can")]),
        scene_export,
    )

    assert result.selected_candidate_id == "blue"
    assert result.binding_report["selection_reason"] == "adjudicated_bindable"
    assert adjudications == 1


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


def test_explicit_scene_semantic_conflict_is_incompatible(
    scene_export: Path,
) -> None:
    config_path = scene_export / "scene_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["rigid_object"][0]["initial_state"]["orientation"] = "upright"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    result = SceneAdapter(grounding_caller=_grounder).adapt(
        _candidate_set([_candidate("red", "red can")]),
        scene_export,
    )
    reference = result.binding_report["candidates"][0]["references"][0]
    assert result.binding_report["status"] == "unsatisfied"
    assert reference["status"] == "incompatible"
    assert result.binding_report["candidates"][0]["status"] == "incompatible"
    assert "state 'orientation' conflicts" in reference["reasons"][0]


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


def test_scene_adapter_enforces_count_cardinality_in_audit(
    scene_export: Path,
) -> None:
    candidate = _candidate("two", "cans")
    selector = candidate["draft"]["steps"][0]["object"]
    selector.update(quantifier="count", count=2)
    request = candidate["scene_request"]["references"][0]
    request.update(reference="cans", quantifier="count", count=2)
    candidate["semantic_hash"] = canonical_hash(candidate["draft"]["steps"])

    def one_only(**_kwargs):
        return {
            "bindings": [
                {
                    "reference_id": "upright.object",
                    "status": "resolved",
                    "uids": ["red_can"],
                    "confidence": 0.95,
                }
            ]
        }

    result = SceneAdapter(grounding_caller=one_only).adapt(
        _candidate_set([candidate]), scene_export
    )

    assert result.binding_report["status"] == "unsatisfied"
    audit = result.binding_report["candidates"][0]
    assert audit["status"] == "incompatible"
    assert "requires exactly 2 UIDs" in audit["references"][0]["reasons"][0]


def test_scene_adapter_binds_all_matching_uids(
    scene_export: Path,
) -> None:
    candidate = _candidate("all", "all cans")
    candidate["draft"]["steps"][0]["object"].update(quantifier="all")
    candidate["scene_request"] = derive_scene_request(candidate["draft"])
    candidate["semantic_hash"] = canonical_hash(candidate["draft"]["steps"])

    def all_cans(**_kwargs):
        return {
            "bindings": [
                {
                    "reference_id": "upright.object",
                    "status": "resolved",
                    "uids": ["red_can", "blue_can"],
                    "confidence": 0.95,
                }
            ]
        }

    result = SceneAdapter(grounding_caller=all_cans).adapt(
        _candidate_set([candidate]), scene_export
    )

    assert result.binding_report["status"] == "bound"
    assert result.reference_bindings == {"upright.object": ["red_can", "blue_can"]}
    assert result.candidate_bindings[candidate["candidate_id"]][
        "reference_bindings"
    ] == {"upright.object": ["red_can", "blue_can"]}


def test_scene_adapter_rejects_step_result_object_matching_same_step_target(
    scene_export: Path,
) -> None:
    config_path = scene_export / "scene_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["rigid_object"][0]["affordances"].append("support_surface")
    config_path.write_text(json.dumps(config), encoding="utf-8")

    candidate = _candidate("self-reference", "red can")
    second = deepcopy(candidate["draft"]["steps"][0])
    second.update(
        {
            "id": "place_again",
            "task_type": "E1",
            "object": {
                "kind": "step_result",
                "step_id": "upright",
                "reference": "",
                "quantifier": "one",
                "count": 0,
            },
            "target": _selector("red can"),
            "relation": "on",
            "orientation_goal": "preserve",
            "depends_on": ["upright"],
        }
    )
    candidate["draft"]["steps"].append(second)
    candidate["scene_request"] = derive_scene_request(candidate["draft"])
    candidate["success_spec"] = derive_success_spec(candidate["draft"])
    candidate["semantic_hash"] = canonical_hash(candidate["draft"]["steps"])

    def same_uid(**_kwargs):
        return {
            "bindings": [
                {
                    "reference_id": "upright.object",
                    "status": "resolved",
                    "uids": ["red_can"],
                    "confidence": 0.95,
                },
                {
                    "reference_id": "place_again.target",
                    "status": "resolved",
                    "uids": ["red_can"],
                    "confidence": 0.95,
                },
            ]
        }

    result = SceneAdapter(grounding_caller=same_uid).adapt(
        _candidate_set([candidate]), scene_export
    )

    assert result.binding_report["status"] == "unsatisfied"
    target_audit = result.binding_report["candidates"][0]["references"][1]
    assert target_audit["status"] == "incompatible"
    assert "same UID as object and target" in target_audit["reasons"][0]


def test_scene_source_fingerprint_covers_assets_and_config(scene_export: Path) -> None:
    original = fingerprint_scene_source(scene_export)

    asset_path = scene_export / "meshes" / "red_can.glb"
    asset_path.write_bytes(b"changed asset")
    changed_asset = fingerprint_scene_source(scene_export)
    assert changed_asset.asset_sha256 != original.asset_sha256

    config_path = scene_export / "scene_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["rigid_object"][0]["body_scale"] = [1.1, 1.0, 1.0]
    config["rigid_object"][0]["physics"] = {"mass": 0.25}
    config_path.write_text(json.dumps(config), encoding="utf-8")
    changed_config = fingerprint_scene_source(scene_export)
    assert changed_config.config_sha256 != changed_asset.config_sha256


def test_scene_source_verification_rejects_later_mutation(scene_export: Path) -> None:
    expected = fingerprint_scene_source(scene_export).to_dict()
    (scene_export / "meshes" / "red_can.glb").write_bytes(b"changed later")

    with pytest.raises(RuntimeError, match="changed after Task Engine preparation"):
        verify_scene_source_fingerprint(expected)
