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
import threading
from time import sleep

import pytest

from embodichain.gen_sim.task_engine.contracts import (
    SUCCESS_SPEC_SCHEMA,
    TASK_DRAFT_SCHEMA,
    validate_success_spec,
    validate_task_candidate,
    validate_task_draft,
)
from embodichain.gen_sim.task_engine.agent import (
    TaskAgent,
    TaskGenerationError,
    derive_scene_request,
    derive_success_spec,
)
from embodichain.gen_sim.task_engine.interpretation import (
    InstructionDraftResult,
    validate_instruction_intent,
)
from embodichain.gen_sim.task_engine.orchestration.contracts import (
    ROLE_BINDINGS_SCHEMA,
)
from embodichain.gen_sim.task_engine.semantic_planner import SemanticTaskPlanner

_TEST_INSTRUCTION = "test-instruction"


def _selector(kind="none", *, step_id="", reference="", quantifier="one", count=0):
    return {
        "kind": kind,
        "step_id": step_id,
        "reference": reference,
        "quantifier": quantifier,
        "count": count,
    }


def _step(step_id="orient", reference="purple can"):
    return {
        "id": step_id,
        "task_type": "E2",
        "object": _selector("scene_ref", reference=reference),
        "target": _selector(),
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


def _result(step):
    return InstructionDraftResult(
        intent={"steps": [deepcopy(step)]},
        model="injected_caller",
        attempts=1,
        latency_seconds=0.01,
        normalizations=(),
    )


def test_task_agent_generates_concurrently_deduplicates_and_counts_votes():
    barrier = threading.Barrier(3)
    lock = threading.Lock()
    assigned = 0

    def interpreter(_instruction, **_kwargs):
        nonlocal assigned
        with lock:
            index = assigned
            assigned += 1
        barrier.wait(timeout=2)
        sleep(0.01)
        if index < 2:
            return _result(_step(step_id=f"arbitrary_{index}"))
        return _result(_step(step_id="different", reference="orange can"))

    result = TaskAgent(interpreter=interpreter).generate("task", _TEST_INSTRUCTION)

    assert result["requested_candidate_count"] == 3
    assert result["valid_response_count"] == 3
    assert len(result["candidates"]) == 2
    assert sorted(item["vote_count"] for item in result["candidates"]) == [1, 2]
    assert {item["draft"]["steps"][0]["id"] for item in result["candidates"]} == {
        "step_01"
    }


def test_scene_request_and_success_are_deterministic_contract_derivations():
    draft = {
        "schema_version": TASK_DRAFT_SCHEMA,
        "task_id": "upright",
        "instruction": _TEST_INSTRUCTION,
        "steps": [_step(reference="all cans")],
    }
    draft["steps"][0]["object"].update(quantifier="all")

    request = derive_scene_request(draft)
    success = derive_success_spec(draft)

    assert request["references"] == [
        {
            "reference_id": "orient.object",
            "step_id": "orient",
            "role": "object",
            "reference": "all cans",
            "quantifier": "all",
            "count": 0,
            "source_structure": "rigid_object",
            "affordances": ["graspable", "orientable"],
            "initial_state": {"orientation": "fallen"},
            "attributes": {},
        }
    ]
    assert success["terms"] == [{"step_id": "orient", "type": "object_upright"}]


@pytest.mark.parametrize(
    ("relation", "expected_structure", "expected_affordances"),
    [
        ("on", "physical_entity", []),
        ("inside", "rigid_object", ["container"]),
        ("behind", "spatial_reference", []),
        ("front_of", "spatial_reference", []),
        ("left_of", "spatial_reference", []),
        ("right_of", "spatial_reference", []),
    ],
)
def test_target_requirements_describe_capabilities_not_concrete_roles(
    relation: str,
    expected_structure: str,
    expected_affordances: list[str],
) -> None:
    step = _step(step_id="place", reference="green can")
    step.update(
        task_type="E1",
        target=_selector("scene_ref", reference="red can"),
        relation=relation,
        orientation_goal="preserve",
    )
    draft = {
        "schema_version": TASK_DRAFT_SCHEMA,
        "task_id": "stack",
        "instruction": _TEST_INSTRUCTION,
        "steps": [step],
    }

    request = derive_scene_request(draft)

    target = next(
        reference
        for reference in request["references"]
        if reference["role"] == "target"
    )
    assert target["source_structure"] == expected_structure
    assert target["affordances"] == expected_affordances


def test_semantic_planner_rejects_implicit_multi_object_expansion():
    def interpreter(_instruction, **_kwargs):
        step = _step(reference="all cans")
        step["object"].update(quantifier="all")
        return _result(step)

    candidate = TaskAgent(interpreter=interpreter).generate(
        "upright", _TEST_INSTRUCTION, candidate_count=1
    )["candidates"][0]
    with pytest.raises(ValueError, match="must resolve to exactly one scene entity"):
        SemanticTaskPlanner().plan(
            candidate,
            {
                "schema_version": ROLE_BINDINGS_SCHEMA,
                "task_id": "upright",
                "candidate_id": candidate["candidate_id"],
                "reference_bindings": {"step_01.object": ["can_a", "can_b"]},
                "role_bindings": {},
            },
            [
                {"uid": "can_a", "init_pos": [0.0, -0.2, 0.7]},
                {"uid": "can_b", "init_pos": [0.0, 0.2, 0.7]},
            ],
        )


@pytest.mark.parametrize("task_type", ("E1", "E2"))
def test_manipulation_intent_requires_an_explicit_or_auto_arm(
    task_type: str,
) -> None:
    """An executable single-arm task cannot retain the inapplicable sentinel."""
    step = _step()
    step["task_type"] = task_type
    step["required_arm"] = "none"
    if task_type == "E1":
        step["target"] = _selector("scene_ref", reference="tray")
        step["relation"] = "on"
        step["orientation_goal"] = "preserve"

    with pytest.raises(ValueError, match="requires required_arm"):
        validate_instruction_intent({"steps": [step]})


def test_semantic_planner_adds_profile_bound_cleanup_without_joint_data() -> None:
    """Task-group cleanup stays a semantic call and never embeds robot qpos."""

    def interpreter(_instruction, **_kwargs):
        step = _step(step_id="place", reference="cube")
        step.update(
            task_type="E1",
            target=_selector("scene_ref", reference="tray"),
            relation="inside",
            required_arm="left_arm",
            orientation_goal="preserve",
        )
        return _result(step)

    candidate = TaskAgent(interpreter=interpreter).generate(
        "place_cube", _TEST_INSTRUCTION, candidate_count=1
    )["candidates"][0]
    graph = SemanticTaskPlanner().plan(
        candidate,
        {
            "schema_version": ROLE_BINDINGS_SCHEMA,
            "task_id": "place_cube",
            "candidate_id": candidate["candidate_id"],
            "reference_bindings": {
                "step_01.object": ["cube"],
                "step_01.target": ["tray"],
            },
            "role_bindings": {},
        },
        [
            {"runtime_uid": "cube", "init_pos": [0.0, -0.2, 0.7]},
            {"runtime_uid": "tray", "init_pos": [0.0, 0.0, 0.7]},
        ],
    )

    assert [node["call"]["kind"] for node in graph["nodes"]] == [
        "pick",
        "place",
        "registered",
    ]
    cleanup = graph["nodes"][-1]
    assert cleanup["role"] == "cleanup"
    assert cleanup["call"] == {
        "kind": "registered",
        "call_id": "simulation.park",
        "arguments": {},
        "resources": {"primary": "left"},
    }
    assert "qpos" not in repr(graph).lower()


def test_semantic_planner_routes_upright_through_axis_align_and_release() -> None:
    """E2 remains semantic while the integration owns axis and motion details."""
    candidate = TaskAgent(
        interpreter=lambda *_args, **_kwargs: _result(
            {**_step(), "required_arm": "right_arm"}
        )
    ).generate("upright", _TEST_INSTRUCTION, candidate_count=1)["candidates"][0]
    graph = SemanticTaskPlanner().plan(
        candidate,
        {
            "schema_version": ROLE_BINDINGS_SCHEMA,
            "task_id": "upright",
            "candidate_id": candidate["candidate_id"],
            "reference_bindings": {"step_01.object": ["purple_can"]},
            "role_bindings": {},
        },
        [
            {"runtime_uid": "purple_can", "init_pos": [0.0, 0.2, 0.7]},
            {"runtime_uid": "table", "init_pos": [0.0, 0.0, 0.0]},
        ],
    )

    assert [node["call"] for node in graph["nodes"]] == [
        {
            "kind": "registered",
            "call_id": "simulation.axis_align",
            "arguments": {"object": "purple_can"},
            "resources": {"primary": "right"},
        },
        {
            "kind": "registered",
            "call_id": "simulation.place_relative",
            "arguments": {
                "object": "purple_can",
                "reference": "table",
                "relation": "on",
            },
            "resources": {"primary": "right"},
        },
        {
            "kind": "registered",
            "call_id": "simulation.park",
            "arguments": {},
            "resources": {"primary": "right"},
        },
    ]
    assert graph["targets"] == {}


def test_semantic_planner_routes_handover_through_verified_pick_state() -> None:
    """A transfer starts from a verified source attachment boundary."""
    step = _step(step_id="handover", reference="can")
    step.update(
        task_type="E4",
        transfer_arm="left_arm",
        receive_arm="right_arm",
        orientation_goal="preserve",
        terminal_behavior="hold",
    )
    candidate = TaskAgent(interpreter=lambda *_args, **_kwargs: _result(step)).generate(
        "handover", _TEST_INSTRUCTION, candidate_count=1
    )["candidates"][0]
    graph = SemanticTaskPlanner().plan(
        candidate,
        {
            "schema_version": ROLE_BINDINGS_SCHEMA,
            "task_id": "handover",
            "candidate_id": candidate["candidate_id"],
            "reference_bindings": {"step_01.object": ["can"]},
            "role_bindings": {},
        },
        [{"runtime_uid": "can", "init_pos": [0.0, -0.2, 0.7]}],
    )

    assert [node["call"] for node in graph["nodes"]] == [
        {
            "kind": "pick",
            "object": "can",
            "resources": {"primary": "left"},
        },
        {
            "kind": "hand_over",
            "object": "can",
            "resources": {"source": "left", "destination": "right"},
        },
    ]


@pytest.mark.parametrize(
    "relation",
    ("above", "behind", "front_of", "left_of", "on", "right_of"),
)
def test_semantic_planner_keeps_spatial_relations_late_bound(relation: str) -> None:
    """Every supported E1 relation names entities instead of an initial pose."""
    step = _step(step_id="place", reference="can")
    step.update(
        task_type="E1",
        target=_selector("scene_ref", reference="notebook"),
        relation=relation,
        required_arm="left_arm",
        orientation_goal="preserve",
    )
    candidate = TaskAgent(interpreter=lambda *_args, **_kwargs: _result(step)).generate(
        "place", _TEST_INSTRUCTION, candidate_count=1
    )["candidates"][0]
    graph = SemanticTaskPlanner().plan(
        candidate,
        {
            "schema_version": ROLE_BINDINGS_SCHEMA,
            "task_id": "place",
            "candidate_id": candidate["candidate_id"],
            "reference_bindings": {
                "step_01.object": ["can"],
                "step_01.target": ["notebook"],
            },
            "role_bindings": {},
        },
        [
            {"runtime_uid": "can", "init_pos": [0.0, -0.2, 0.7]},
            {"runtime_uid": "notebook", "init_pos": [0.0, 0.0, 0.7]},
        ],
    )

    assert graph["targets"] == {}
    assert graph["nodes"][1]["call"] == {
        "kind": "registered",
        "call_id": "simulation.place_relative",
        "arguments": {
            "object": "can",
            "reference": "notebook",
            "relation": relation,
        },
        "resources": {"primary": "left"},
    }


def test_draft_rejects_grounded_fields_and_task_agent_fails_closed():
    draft = {
        "schema_version": TASK_DRAFT_SCHEMA,
        "task_id": "bad",
        "instruction": "bad",
        "steps": [_step()],
    }
    draft["steps"][0]["object"]["uid"] = "scene_uid"
    with pytest.raises(ValueError, match="forbidden|exactly fields"):
        validate_task_draft(draft)

    def invalid(_instruction, **_kwargs):
        raise ValueError("invalid draft after repair")

    with pytest.raises(TaskGenerationError, match="All Task Agent candidates"):
        TaskAgent(interpreter=invalid).generate("bad", "bad")


def test_task_candidate_rejects_scene_constraints_not_derived_from_draft():
    candidate = TaskAgent(
        interpreter=lambda *_args, **_kwargs: _result(_step())
    ).generate("upright", _TEST_INSTRUCTION, candidate_count=1)["candidates"][0]
    candidate["scene_request"]["references"][0]["affordances"] = []

    with pytest.raises(ValueError, match="derived exactly"):
        validate_task_candidate(candidate)


def test_success_spec_rejects_types_outside_task_ontology():
    with pytest.raises(ValueError, match="must be one of"):
        validate_success_spec(
            {
                "schema_version": SUCCESS_SPEC_SCHEMA,
                "task_id": "bad_success",
                "op": "all",
                "terms": [{"step_id": "step_01", "type": "looks_good"}],
            }
        )


def test_task_agent_isolates_invalid_interpreter_results():
    lock = threading.Lock()
    calls = 0

    def interpreter(_instruction, **_kwargs):
        nonlocal calls
        with lock:
            index = calls
            calls += 1
        if index == 0:
            invalid = _step()
            invalid["object"]["uid"] = "red_can"
            return _result(invalid)
        return _result(_step())

    result = TaskAgent(interpreter=interpreter).generate(
        "upright", _TEST_INSTRUCTION, candidate_count=2
    )

    assert result["valid_response_count"] == 1
    assert len(result["errors"]) == 1
    assert len(result["candidates"]) == 1
