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
from pathlib import Path
from types import SimpleNamespace

import pytest

from embodichain.gen_sim.task_engine.scene import (
    FeasibilityBroker,
    SceneEngineV1Adapter,
)
from embodichain.gen_sim.task_engine.agent import derive_scene_request
from embodichain.gen_sim.task_engine.contracts import TASK_DRAFT_SCHEMA


def _prepared_scene(tmp_path: Path) -> SimpleNamespace:
    table = {
        "uid": "table",
        "source_uid": "table_0",
        "role": "background",
        "name": "table",
        "description": "A support table.",
        "category": "table",
        "color": "brown",
        "shape": {"shape_type": "Mesh", "fpath": "/assets/table.glb"},
        "init_pos": [0.0, 0.0, 0.0],
        "init_rot": [0.0, 0.0, 0.0],
        "body_scale": [1.0, 1.0, 1.0],
        "attributes": {},
        "initial_state": {},
        "affordances": [],
    }
    can = {
        "uid": "red_can",
        "source_uid": "red_can_0",
        "role": "rigid_object",
        "name": "red can",
        "description": "A fallen red can.",
        "category": "can",
        "color": "red",
        "shape": {"shape_type": "Mesh", "fpath": "/assets/can.glb"},
        "init_pos": [0.1, 0.0, 0.7],
        "init_rot": [90.0, 0.0, 0.0],
        "body_scale": [1.0, 1.0, 1.0],
        "attributes": {},
        "initial_state": {"orientation": "fallen"},
        "affordances": ["graspable", "orientable", "placeable"],
    }
    runtime_table = {
        "uid": "table",
        "shape": table["shape"],
        "attrs": {"mass": 10.0},
        "body_type": "kinematic",
    }
    runtime_can = {
        "uid": "red_can",
        "shape": can["shape"],
        "attrs": {"mass": 0.1},
        "body_type": "dynamic",
    }
    return SimpleNamespace(
        source_config_path=tmp_path / "scene_config.json",
        planner_objects=(table, can),
        background=(runtime_table,),
        rigid_objects=(runtime_can,),
        articulations=(),
        asset_hashes={"table": "a" * 64, "red_can": "b" * 64},
    )


def _candidate(task_type: str, affordances: list[str]) -> dict:
    return {
        "candidate_id": "candidate_01",
        "draft": {
            "task_id": "task",
            "steps": [{"id": "step_01", "task_type": task_type}],
        },
        "scene_request": {
            "references": [
                {
                    "reference_id": "step_01.object",
                    "role": "object",
                    "source_structure": "rigid_object",
                    "affordances": affordances,
                    "initial_state": (
                        {"orientation": "fallen"} if task_type == "E2" else {}
                    ),
                    "attributes": {},
                }
            ]
        },
    }


def _catalog(*, pour_available: bool = False) -> dict[str, dict]:
    return {
        name: {"runtime_available": True, "unavailable_reason": None}
        for name in ("PickUp", "MoveHeldObject", "Place", "TurnKnob")
    } | {
        "Pour": {
            "runtime_available": pour_available,
            "unavailable_reason": None if pour_available else "Pour is planning-only.",
        }
    }


def _selector(kind: str, *, reference: str = "") -> dict[str, object]:
    return {
        "kind": kind,
        "step_id": "",
        "reference": reference,
        "quantifier": "one",
        "count": 0,
    }


def _relation_candidate(relation: str) -> dict:
    draft = {
        "schema_version": TASK_DRAFT_SCHEMA,
        "task_id": "place_relative",
        "instruction": "place the can relative to the target",
        "steps": [
            {
                "id": "step_01",
                "task_type": "E1",
                "object": _selector("scene_ref", reference="red can"),
                "target": _selector("scene_ref", reference="target"),
                "relation": relation,
                "required_arm": "auto",
                "transfer_arm": "none",
                "receive_arm": "none",
                "orientation_goal": "preserve",
                "target_state": "none",
                "target_setting": 0,
                "layout": "none",
                "axis": "none",
                "direction": "none",
                "terminal_behavior": "none",
                "depends_on": [],
            }
        ],
    }
    return {
        "candidate_id": "candidate_01",
        "draft": draft,
        "scene_request": derive_scene_request(draft),
    }


def _manifest_with_target_kinds(tmp_path: Path) -> dict:
    manifest = SceneEngineV1Adapter().adapt_prepared_scene(
        _prepared_scene(tmp_path),
        source_format="test",
        robot_profile="dual_franka",
    )
    by_uid = {item["uid"]: item for item in manifest["objects"]}
    by_uid["red_can"]["affordances"].append(
        {
            "type": "container",
            "status": "declared",
            "confidence": None,
            "source": "test",
            "link_uid": "",
            "frame": {},
            "parameters": {},
        }
    )
    articulation = deepcopy(by_uid["red_can"])
    articulation.update(
        uid="cabinet",
        source_uid="cabinet_0",
        role="articulation",
        name="cabinet",
        category="cabinet",
        physics={},
        articulation={"runtime_uid": "cabinet"},
        affordances=[],
    )
    manifest["objects"].append(articulation)
    return manifest


def test_scene_engine_v1_adapter_preserves_static_execution_evidence(
    tmp_path: Path,
) -> None:
    manifest = SceneEngineV1Adapter().adapt_prepared_scene(
        _prepared_scene(tmp_path),
        source_format="embodichain.scene-export/v1",
        robot_profile="dual_franka",
    )

    by_uid = {item["uid"]: item for item in manifest["objects"]}
    assert manifest["adapter_capabilities"]["task_conditioned_generation"] is False
    assert by_uid["red_can"]["geometry"]["asset_sha256"] == "b" * 64
    assert by_uid["red_can"]["physics"]["body_type"] == "dynamic"
    assert {item["type"] for item in by_uid["table"]["affordances"]} == {
        "support_surface"
    }
    assert (
        next(
            item
            for item in by_uid["red_can"]["affordances"]
            if item["type"] == "graspable"
        )["status"]
        == "declared"
    )


def test_e2_feasibility_requires_runtime_probe_for_geometry(tmp_path: Path) -> None:
    manifest = SceneEngineV1Adapter().adapt_prepared_scene(
        _prepared_scene(tmp_path),
        source_format="test",
        robot_profile="dual_franka",
    )
    report = FeasibilityBroker().assess(
        _candidate("E2", ["graspable", "orientable"]),
        {"step_01.object": ["red_can"]},
        manifest,
        capability_catalog=_catalog(),
        task_actions={"E2": ("PickUp", "MoveHeldObject", "Place")},
    )

    assert report["status"] == "runtime_probe"
    assert report["remediation_class"] == "none"
    assert report["blockers"] == []
    assert report["summary"]["proven"] > 0
    assert report["summary"]["runtime_probe"] > 0


def test_planning_only_action_is_reported_as_contradicted(tmp_path: Path) -> None:
    manifest = SceneEngineV1Adapter().adapt_prepared_scene(
        _prepared_scene(tmp_path),
        source_format="test",
        robot_profile="dual_franka",
    )
    report = FeasibilityBroker().assess(
        _candidate("E3", ["graspable", "pourable"]),
        {"step_01.object": ["red_can"]},
        manifest,
        capability_catalog=_catalog(),
        task_actions={"E3": ("Pour",)},
    )

    assert report["status"] == "contradicted"
    assert report["remediation_class"] == "action_capability"
    assert any("planning-only" in blocker for blocker in report["blockers"])


def test_e3_does_not_require_runtime_content_observation_before_execution(
    tmp_path: Path,
) -> None:
    manifest = SceneEngineV1Adapter().adapt_prepared_scene(
        _prepared_scene(tmp_path),
        source_format="test",
        robot_profile="dual_franka",
    )
    report = FeasibilityBroker().assess(
        _candidate("E3", ["graspable", "pourable"]),
        {"step_01.object": ["red_can"]},
        manifest,
        capability_catalog=_catalog(pour_available=True),
        task_actions={"E3": ("PickUp", "MoveHeldObject", "Pour")},
    )

    assert report["status"] != "contradicted"
    assert all(check["kind"] != "content_observation" for check in report["checks"])


def test_e8_requires_explicit_setting_to_angle_mapping(tmp_path: Path) -> None:
    manifest = SceneEngineV1Adapter().adapt_prepared_scene(
        _prepared_scene(tmp_path),
        source_format="test",
        robot_profile="dual_franka",
    )
    report = FeasibilityBroker().assess(
        _candidate("E8", ["turnable"]),
        {"step_01.object": ["red_can"]},
        manifest,
        capability_catalog=_catalog(),
        task_actions={"E8": ("TurnKnob",)},
    )

    assert any(
        check["kind"] == "setting_mapping" and check["status"] == "contradicted"
        for check in report["checks"]
    )
    assert any("setting_values" in blocker for blocker in report["blockers"])


def test_final_orientation_conflict_is_scene_remediable(tmp_path: Path) -> None:
    manifest = SceneEngineV1Adapter().adapt_prepared_scene(
        _prepared_scene(tmp_path),
        source_format="test",
        robot_profile="dual_franka",
    )
    can = next(item for item in manifest["objects"] if item["uid"] == "red_can")
    can["initial_state"]["orientation"] = "upright"

    report = FeasibilityBroker().assess(
        _candidate("E2", ["graspable", "orientable"]),
        {"step_01.object": ["red_can"]},
        manifest,
        capability_catalog=_catalog(),
        task_actions={"E2": ("PickUp", "MoveHeldObject", "Place")},
    )

    assert report["status"] == "contradicted"
    assert report["remediation_class"] == "scene_remediable"


def test_missing_affordance_remains_unknown_instead_of_becoming_supported(
    tmp_path: Path,
) -> None:
    manifest = SceneEngineV1Adapter().adapt_prepared_scene(
        _prepared_scene(tmp_path),
        source_format="test",
        robot_profile="dual_franka",
    )
    report = FeasibilityBroker().assess(
        _candidate("E1", ["graspable", "liquid_safe"]),
        {"step_01.object": ["red_can"]},
        manifest,
        capability_catalog=_catalog(),
        task_actions={"E1": ("PickUp", "MoveHeldObject", "Place")},
    )

    assert report["status"] == "unknown"
    assert any(
        check["status"] == "unknown" and "liquid_safe" in check["reason"]
        for check in report["checks"]
    )


def test_physical_object_can_be_a_runtime_support_without_support_affordance(
    tmp_path: Path,
) -> None:
    manifest = SceneEngineV1Adapter().adapt_prepared_scene(
        _prepared_scene(tmp_path),
        source_format="test",
        robot_profile="dual_franka",
    )
    candidate = _candidate("E1", ["graspable", "placeable"])
    candidate["draft"]["steps"][0].update(
        target={"kind": "scene_ref"},
        relation="on",
    )
    candidate["scene_request"]["references"].append(
        {
            "reference_id": "step_01.target",
            "role": "target",
            "source_structure": "physical_entity",
            "affordances": [],
            "initial_state": {},
            "attributes": {},
        }
    )

    report = FeasibilityBroker().assess(
        candidate,
        {
            "step_01.object": ["red_can"],
            "step_01.target": ["red_can"],
        },
        manifest,
        capability_catalog=_catalog(),
        task_actions={"E1": ("PickUp", "MoveHeldObject", "Place")},
    )

    structure = next(
        check
        for check in report["checks"]
        if check["kind"] == "structure" and check["subject"] == "step_01.target:red_can"
    )
    assert structure["status"] == "proven"
    support_probe = next(
        check for check in report["checks"] if check["kind"] == "placement_support"
    )
    assert support_probe["status"] == "runtime_probe"
    assert support_probe["evidence"]["runtime_obligations"] == [
        "placement_candidates",
        "object_supported_by",
        "stable_for",
        "final_support_revalidation",
    ]
    assert report["blockers"] == []


@pytest.mark.parametrize(
    ("relation", "target_uid", "expected_structure", "expected_status"),
    [
        ("on", "red_can", "physical_entity", "proven"),
        ("on", "table", "physical_entity", "proven"),
        ("on", "cabinet", "physical_entity", "contradicted"),
        ("inside", "red_can", "rigid_object", "proven"),
        ("inside", "table", "rigid_object", "contradicted"),
        ("inside", "cabinet", "rigid_object", "contradicted"),
        ("behind", "red_can", "spatial_reference", "proven"),
        ("behind", "table", "spatial_reference", "proven"),
        ("behind", "cabinet", "spatial_reference", "runtime_probe"),
        ("front_of", "red_can", "spatial_reference", "proven"),
        ("front_of", "table", "spatial_reference", "proven"),
        ("front_of", "cabinet", "spatial_reference", "runtime_probe"),
        ("left_of", "red_can", "spatial_reference", "proven"),
        ("left_of", "table", "spatial_reference", "proven"),
        ("left_of", "cabinet", "spatial_reference", "runtime_probe"),
        ("right_of", "red_can", "spatial_reference", "proven"),
        ("right_of", "table", "spatial_reference", "proven"),
        ("right_of", "cabinet", "spatial_reference", "runtime_probe"),
    ],
)
def test_relation_target_structure_matrix_uses_capability_semantics(
    tmp_path: Path,
    relation: str,
    target_uid: str,
    expected_structure: str,
    expected_status: str,
) -> None:
    candidate = _relation_candidate(relation)
    target_request = next(
        item
        for item in candidate["scene_request"]["references"]
        if item["role"] == "target"
    )
    report = FeasibilityBroker().assess(
        candidate,
        {
            "step_01.object": ["red_can"],
            "step_01.target": [target_uid],
        },
        _manifest_with_target_kinds(tmp_path),
        capability_catalog=_catalog(),
        task_actions={"E1": ("PickUp", "MoveHeldObject", "Place")},
    )

    structure = next(
        check
        for check in report["checks"]
        if check["kind"] == "structure"
        and check["subject"] == f"step_01.target:{target_uid}"
    )
    assert target_request["source_structure"] == expected_structure
    assert structure["status"] == expected_status


def test_legacy_scene_entity_target_is_treated_as_an_abstract_structure(
    tmp_path: Path,
) -> None:
    candidate = _relation_candidate("behind")
    target_request = next(
        item
        for item in candidate["scene_request"]["references"]
        if item["role"] == "target"
    )
    target_request["source_structure"] = "scene_entity"

    report = FeasibilityBroker().assess(
        candidate,
        {
            "step_01.object": ["red_can"],
            "step_01.target": ["red_can"],
        },
        _manifest_with_target_kinds(tmp_path),
        capability_catalog=_catalog(),
        task_actions={"E1": ("PickUp", "MoveHeldObject", "Place")},
    )

    structure = next(
        check
        for check in report["checks"]
        if check["kind"] == "structure" and check["subject"] == "step_01.target:red_can"
    )
    assert structure["status"] == "proven"
    assert report["blockers"] == []


def test_unknown_structure_contract_is_not_a_scene_contradiction(
    tmp_path: Path,
) -> None:
    candidate = _relation_candidate("behind")
    target_request = next(
        item
        for item in candidate["scene_request"]["references"]
        if item["role"] == "target"
    )
    target_request["source_structure"] = "future_spatial_capability"

    report = FeasibilityBroker().assess(
        candidate,
        {
            "step_01.object": ["red_can"],
            "step_01.target": ["red_can"],
        },
        _manifest_with_target_kinds(tmp_path),
        capability_catalog=_catalog(),
        task_actions={"E1": ("PickUp", "MoveHeldObject", "Place")},
    )

    structure = next(
        check
        for check in report["checks"]
        if check["kind"] == "structure" and check["subject"] == "step_01.target:red_can"
    )
    assert structure["status"] == "unknown"
    assert not any("future_spatial_capability" in item for item in report["blockers"])


def test_required_arm_side_requires_the_live_robot_frame(
    tmp_path: Path,
) -> None:
    manifest = SceneEngineV1Adapter().adapt_prepared_scene(
        _prepared_scene(tmp_path),
        source_format="test",
        robot_profile="dual_franka",
    )
    red_can = next(item for item in manifest["objects"] if item["uid"] == "red_can")
    red_can["initial_pose"]["position"][1] = -0.20
    candidate = _candidate("E2", ["graspable", "orientable"])
    candidate["draft"]["steps"][0]["required_arm"] = "right_arm"

    report = FeasibilityBroker().assess(
        candidate,
        {"step_01.object": ["red_can"]},
        manifest,
        capability_catalog=_catalog(),
        task_actions={"E2": ("PickUp", "MoveHeldObject", "Place")},
    )

    probe = next(
        check for check in report["checks"] if check["kind"] == "arm_layout_risk"
    )
    assert probe["status"] == "runtime_probe"
    assert probe["evidence"]["arm_side_frame"] == "live_robot"
    assert probe["evidence"]["mismatch_risk"] is None
    assert "expected_arm" not in probe["evidence"]
    assert probe["evidence"]["geometry_certificate"] is False
    assert report["blockers"] == []


def test_workspace_report_covers_complete_task_phases(tmp_path: Path) -> None:
    manifest = SceneEngineV1Adapter().adapt_prepared_scene(
        _prepared_scene(tmp_path),
        source_format="test",
        robot_profile="dual_franka",
    )
    report = FeasibilityBroker().assess(
        _candidate("E2", ["graspable", "orientable"]),
        {"step_01.object": ["red_can"]},
        manifest,
        capability_catalog=_catalog(),
        task_actions={"E2": ("PickUp", "MoveHeldObject", "Place")},
    )

    workflow = next(
        check for check in report["checks"] if check["kind"] == "task_workspace"
    )
    phases = {item["phase"] for item in workflow["evidence"]["phases"]}
    assert phases == {"pickup", "safety_clearance"}
    assert workflow["status"] == "runtime_probe"
