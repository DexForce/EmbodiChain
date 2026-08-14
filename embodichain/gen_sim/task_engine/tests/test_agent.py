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
from embodichain.gen_sim.task_engine.interpretation import InstructionDraftResult
from embodichain.gen_sim.collaboration.coordinator import lower_task_candidate


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

    result = TaskAgent(interpreter=interpreter).generate("task", "扶正易拉罐")

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
        "instruction": "扶正所有易拉罐",
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


def test_on_target_requires_a_physical_entity_not_a_unary_support_label():
    step = _step(step_id="place", reference="green can")
    step.update(
        task_type="E1",
        target=_selector("scene_ref", reference="red can"),
        relation="on",
        orientation_goal="preserve",
    )
    draft = {
        "schema_version": TASK_DRAFT_SCHEMA,
        "task_id": "stack",
        "instruction": "把绿罐放到红罐上",
        "steps": [step],
    }

    request = derive_scene_request(draft)

    target = next(
        reference
        for reference in request["references"]
        if reference["role"] == "target"
    )
    assert target["source_structure"] == "physical_entity"
    assert target["affordances"] == []


def test_lower_task_candidate_expands_success_for_all_binding():
    def interpreter(_instruction, **_kwargs):
        step = _step(reference="all cans")
        step["object"].update(quantifier="all")
        return _result(step)

    candidate = TaskAgent(interpreter=interpreter).generate(
        "upright", "扶正所有易拉罐", candidate_count=1
    )["candidates"][0]
    grounded = lower_task_candidate(
        candidate,
        {"step_01.object": ["can_a", "can_b"]},
        [
            {"uid": "can_a", "role": "rigid_object", "description": "A can."},
            {"uid": "can_b", "role": "rigid_object", "description": "A can."},
        ],
        "dual_franka",
    )

    assert grounded.task_spec["level"] == "L2"
    assert [term["type"] for term in grounded.task_spec["success"]["terms"]] == [
        "object_upright",
        "object_upright",
    ]


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
    ).generate("upright", "扶正易拉罐", candidate_count=1)["candidates"][0]
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
        "upright", "扶正易拉罐", candidate_count=2
    )

    assert result["valid_response_count"] == 1
    assert len(result["errors"]) == 1
    assert len(result["candidates"]) == 1
