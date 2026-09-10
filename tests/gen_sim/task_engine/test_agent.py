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
from embodichain.gen_sim.task_engine.semantic_planner import (
    SemanticTaskPlanner,
    UnsupportedSemanticCapabilityError,
)

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


def test_semantic_planner_keeps_explicit_e2_arm_through_release() -> None:
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

    assert [node["call"].get("call_id") for node in graph["nodes"]] == [
        "simulation.pick",
        "gen_sim.align_held",
        "gen_sim.align_held",
        "simulation.place_relative",
        "gen_sim.clear_released",
        "simulation.park",
    ]
    assert all(
        node["call"]["resources"] == {"primary": "right"} for node in graph["nodes"]
    )
    release = next(
        node["call"]
        for node in graph["nodes"]
        if node["call"].get("call_id") == "simulation.place_relative"
    )
    assert release["arguments"] == {
        "object": "purple_can",
        "reference": "table",
        "relation": "on",
    }
    assert set(graph["targets"]) == {
        "step_01_upright_target",
        "step_01_upright_staging_target",
    }


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


def test_semantic_planner_composes_e2_from_pick_move_and_move_joints() -> None:
    """E2 separates verified pickup, upright transport, opening, and retreat."""

    def interpreter(_instruction, **_kwargs):
        return _result(_step(reference="purple can"))

    candidate = TaskAgent(interpreter=interpreter).generate(
        "upright_can", _TEST_INSTRUCTION, candidate_count=1
    )["candidates"][0]
    graph = SemanticTaskPlanner().plan(
        candidate,
        {
            "schema_version": ROLE_BINDINGS_SCHEMA,
            "task_id": "upright_can",
            "candidate_id": candidate["candidate_id"],
            "reference_bindings": {"step_01.object": ["can"]},
            "role_bindings": {},
        },
        [
            {
                "runtime_uid": "can",
                "role": "rigid_object",
                "init_pos": [0.1, -0.2, 0.76],
                "attributes": {
                    "final_world_aabb": {
                        "min": [0.04, -0.26, 0.73],
                        "max": [0.16, -0.14, 0.79],
                    }
                },
            },
            {
                "runtime_uid": "table",
                "role": "background",
                "init_pos": [0.0, 0.0, 0.0],
                "attributes": {
                    "final_world_aabb": {
                        "min": [-0.5, -0.5, 0.0],
                        "max": [0.5, 0.5, 0.72],
                    }
                },
            },
        ],
    )

    assert [node["call"] for node in graph["nodes"]] == [
        {
            "kind": "registered",
            "call_id": "simulation.pick",
            "arguments": {"object": "can", "target": "step_01_upright_target"},
            "resources": {"primary": "left"},
        },
        {
            "kind": "registered",
            "call_id": "gen_sim.align_held",
            "arguments": {
                "object": "can",
                "target": "step_01_upright_staging_target",
                "preserve_yaw": False,
            },
            "resources": {"primary": "left"},
        },
        {
            "kind": "registered",
            "call_id": "gen_sim.align_held",
            "arguments": {
                "object": "can",
                "target": "step_01_upright_staging_target",
                "preserve_yaw": True,
            },
            "resources": {"primary": "left"},
        },
        {
            "kind": "registered",
            "call_id": "simulation.place_relative",
            "arguments": {
                "object": "can",
                "reference": "table",
                "relation": "on",
            },
            "resources": {"primary": "left"},
        },
        {
            "kind": "registered",
            "call_id": "gen_sim.clear_released",
            "arguments": {
                "object": "can",
                "target": "step_01_upright_staging_target",
            },
            "resources": {"primary": "left"},
        },
        {
            "kind": "registered",
            "call_id": "simulation.park",
            "arguments": {},
            "resources": {"primary": "left"},
        },
    ]
    assert graph["targets"]["step_01_upright_target"]["values"][0][
        "position"
    ] == pytest.approx([0.1, -0.2, 0.79])
    assert graph["task_groups"][0]["success"]["type"] == "object_upright"


@pytest.mark.parametrize(
    "relation",
    [
        "left_of",
        "right_of",
        "front_of",
        "behind",
        "front_left_of",
        "front_right_of",
        "back_left_of",
        "back_right_of",
    ],
)
def test_semantic_planner_keeps_directional_e1_targets_live(
    relation: str,
) -> None:
    """Directional E1 calls retain their reference identity until execution."""

    def interpreter(_instruction, **_kwargs):
        step = _step(step_id="place", reference="cube")
        step.update(
            task_type="E1",
            target=_selector("scene_ref", reference="bottle"),
            relation=relation,
            required_arm="left_arm",
            orientation_goal="preserve",
        )
        return _result(step)

    candidate = TaskAgent(interpreter=interpreter).generate(
        "relative_place", _TEST_INSTRUCTION, candidate_count=1
    )["candidates"][0]
    graph = SemanticTaskPlanner().plan(
        candidate,
        {
            "schema_version": ROLE_BINDINGS_SCHEMA,
            "task_id": "relative_place",
            "candidate_id": candidate["candidate_id"],
            "reference_bindings": {
                "step_01.object": ["cube"],
                "step_01.target": ["bottle"],
            },
            "role_bindings": {},
        },
        [
            {"runtime_uid": "cube", "init_pos": [0.0, 0.0, 0.8]},
            {"runtime_uid": "bottle", "init_pos": [0.2, 0.3, 0.7]},
        ],
    )

    call = graph["nodes"][1]["call"]
    assert call["call_id"] == "simulation.place_relative"
    assert call["arguments"] == {
        "object": "cube",
        "reference": "bottle",
        "relation": relation,
    }
    assert graph["targets"] == {}


def test_semantic_planner_keeps_e1_support_relation_late_bound() -> None:
    """Place-on selects trusted scene geometry rather than a frozen pose."""

    def interpreter(_instruction, **_kwargs):
        step = _step(step_id="stack", reference="apple")
        step.update(
            task_type="E1",
            target=_selector("scene_ref", reference="can"),
            relation="on",
            required_arm="left_arm",
            orientation_goal="preserve",
        )
        return _result(step)

    candidate = TaskAgent(interpreter=interpreter).generate(
        "stack_apple", _TEST_INSTRUCTION, candidate_count=1
    )["candidates"][0]
    graph = SemanticTaskPlanner().plan(
        candidate,
        {
            "schema_version": ROLE_BINDINGS_SCHEMA,
            "task_id": "stack_apple",
            "candidate_id": candidate["candidate_id"],
            "reference_bindings": {
                "step_01.object": ["apple"],
                "step_01.target": ["can"],
            },
            "role_bindings": {},
        },
        [
            {"runtime_uid": "apple", "init_pos": [0.0, 0.0, 0.8]},
            {"runtime_uid": "can", "init_pos": [0.2, 0.3, 0.8]},
        ],
    )

    assert graph["nodes"][1]["call"] == {
        "kind": "registered",
        "call_id": "simulation.place_relative",
        "arguments": {"object": "apple", "reference": "can", "relation": "on"},
        "resources": {"primary": "left"},
    }


def test_semantic_planner_composes_e3_from_existing_pour_skills() -> None:
    """E3 emits only canonical calls while retaining the live target container."""

    def interpreter(_instruction, **_kwargs):
        step = _step(step_id="pour", reference="bottle")
        step.update(
            task_type="E3",
            target=_selector("scene_ref", reference="cup"),
            relation="above",
            required_arm="right_arm",
            orientation_goal="none",
        )
        return _result(step)

    candidate = TaskAgent(interpreter=interpreter).generate(
        "pour_water", _TEST_INSTRUCTION, candidate_count=1
    )["candidates"][0]
    graph = SemanticTaskPlanner().plan(
        candidate,
        {
            "schema_version": ROLE_BINDINGS_SCHEMA,
            "task_id": "pour_water",
            "candidate_id": candidate["candidate_id"],
            "reference_bindings": {
                "step_01.object": ["bottle"],
                "step_01.target": ["cup"],
            },
            "role_bindings": {},
        },
        [
            {"runtime_uid": "bottle", "init_pos": [0.1, 0.2, 0.75]},
            {"runtime_uid": "cup", "init_pos": [0.0, 0.0, 0.75]},
        ],
    )

    assert [node["call"]["kind"] for node in graph["nodes"]] == [
        "pick",
        "registered",
        "registered",
        "place",
        "registered",
    ]
    assert graph["nodes"][1]["call"] == {
        "kind": "registered",
        "call_id": "simulation.move_held_object",
        "arguments": {
            "object": "bottle",
            "target": "step_01_pour_target",
            "reference": "cup",
        },
        "resources": {"primary": "right"},
    }
    assert graph["nodes"][2]["call"] == {
        "kind": "registered",
        "call_id": "simulation.pour",
        "arguments": {"object": "bottle"},
        "resources": {"primary": "right"},
    }
    assert graph["nodes"][3]["call"]["at"] == {
        "kind": "target_ref",
        "target": "step_01_return_target",
    }
    assert graph["task_groups"][0]["success"]["type"] == "poured"


@pytest.mark.parametrize(
    ("task_type", "terminal"), [("E1", "place"), ("E4", "hold"), ("E4", "place")]
)
def test_explicit_upright_adds_alignment_on_the_final_holding_arm(
    task_type: str, terminal: str
) -> None:
    step = _step(reference="can")
    step.update(
        task_type=task_type, required_arm="right_arm" if task_type == "E1" else "none"
    )
    if task_type == "E4":
        step.update(
            transfer_arm="left_arm", receive_arm="right_arm", terminal_behavior=terminal
        )
    if terminal == "place":
        step.update(target=_selector("scene_ref", reference="table"), relation="on")
    candidate = TaskAgent(interpreter=lambda *_a, **_kw: _result(step)).generate(
        "upright_goal", _TEST_INSTRUCTION, candidate_count=1
    )["candidates"][0]
    graph = SemanticTaskPlanner().plan(
        candidate,
        {
            "schema_version": ROLE_BINDINGS_SCHEMA,
            "task_id": "upright_goal",
            "candidate_id": candidate["candidate_id"],
            "reference_bindings": {
                "step_01.object": ["can"],
                **({"step_01.target": ["table"]} if terminal == "place" else {}),
            },
            "role_bindings": {},
        },
        [
            {"runtime_uid": "can", "init_pos": [0.0, -0.2, 0.8]},
            {"runtime_uid": "table", "init_pos": [0.0, 0.0, 0.72]},
        ],
    )
    calls = [node["call"] for node in graph["nodes"]]
    alignment = next(
        call
        for call in calls
        if call.get("call_id") == "gen_sim.align_held"
        and call["resources"] == {"primary": "right"}
        and call["arguments"]["preserve_yaw"] is True
    )
    assert alignment["resources"] == {"primary": "right"}
    assert alignment["arguments"] == {
        "object": "can",
        "target": "current_object_pose",
        "preserve_yaw": True,
    }
    if task_type == "E4":
        source_alignment = next(
            call
            for call in calls
            if call.get("call_id") == "gen_sim.align_held"
            and call["resources"] == {"primary": "left"}
        )
        assert calls.index(source_alignment) < next(
            i for i, call in enumerate(calls) if call["kind"] == "hand_over"
        )
        assert calls.index(alignment) > next(
            i for i, call in enumerate(calls) if call["kind"] == "hand_over"
        )
    if terminal == "place":
        assert calls.index(alignment) < next(
            i
            for i, call in enumerate(calls)
            if call.get("call_id") == "simulation.place_relative"
        )
        assert (
            calls[-2]["call_id"] == "simulation.park"
            if task_type == "E4"
            else calls[-2]["call_id"] == "gen_sim.clear_released"
        )
    else:
        assert calls[-1] is alignment


def test_handover_continuation_does_not_pick_an_already_held_object() -> None:
    first = _step(step_id="one", reference="can")
    first.update(
        task_type="E4",
        orientation_goal="none",
        required_arm="none",
        transfer_arm="left_arm",
        receive_arm="right_arm",
        terminal_behavior="hold",
    )
    second = deepcopy(first)
    second.update(
        id="two",
        object=_selector("step_result", step_id="one"),
        transfer_arm="right_arm",
        receive_arm="left_arm",
        depends_on=["one"],
    )
    interpreted = _result(first)
    interpreted.intent["steps"].append(second)
    candidate = TaskAgent(interpreter=lambda *_a, **_kw: interpreted).generate(
        "return_handover", _TEST_INSTRUCTION, candidate_count=1
    )["candidates"][0]
    graph = SemanticTaskPlanner().plan(
        candidate,
        {
            "schema_version": ROLE_BINDINGS_SCHEMA,
            "task_id": "return_handover",
            "candidate_id": candidate["candidate_id"],
            "reference_bindings": {"step_01.object": ["can"]},
            "role_bindings": {},
        },
        [{"runtime_uid": "can", "init_pos": [0.0, -0.2, 0.8]}],
    )
    assert [node["call"]["kind"] for node in graph["nodes"]] == [
        "pick",
        "hand_over",
        "hand_over",
    ]


def test_semantic_planner_preserves_e4_terminal_place() -> None:
    """A handover-place task transfers and then releases at its requested relation."""

    def interpreter(_instruction, **_kwargs):
        step = _step(step_id="handover", reference="can")
        step.update(
            task_type="E4",
            target=_selector("scene_ref", reference="tray"),
            relation="inside",
            required_arm="none",
            transfer_arm="left_arm",
            receive_arm="right_arm",
            orientation_goal="none",
            terminal_behavior="place",
        )
        return _result(step)

    candidate = TaskAgent(interpreter=interpreter).generate(
        "handover_place", _TEST_INSTRUCTION, candidate_count=1
    )["candidates"][0]
    graph = SemanticTaskPlanner().plan(
        candidate,
        {
            "schema_version": ROLE_BINDINGS_SCHEMA,
            "task_id": "handover_place",
            "candidate_id": candidate["candidate_id"],
            "reference_bindings": {
                "step_01.object": ["can"],
                "step_01.target": ["tray"],
            },
            "role_bindings": {},
        },
        [
            {"runtime_uid": "can", "init_pos": [0.0, -0.2, 0.75]},
            {"runtime_uid": "tray", "init_pos": [0.0, 0.2, 0.75]},
        ],
    )

    assert [node["call"]["kind"] for node in graph["nodes"]] == [
        "pick",
        "hand_over",
        "place",
        "registered",
        "registered",
    ]
    assert graph["nodes"][2]["call"] == {
        "kind": "place",
        "object": "can",
        "inside": "inside__tray__can",
        "resources": {"primary": "right"},
    }
    assert [node["call"]["resources"]["primary"] for node in graph["nodes"][3:]] == [
        "left",
        "right",
    ]
    assert graph["task_groups"][0]["success"]["type"] == "semantic_goal"


@pytest.mark.parametrize(
    ("terminal_behavior", "call_id", "expected_displacement", "cleanup_count"),
    [
        ("hold", "simulation.coordinated_hold", [0.0, 0.0, 0.14], 0),
        (
            "place",
            "simulation.coordinated_transport",
            [-0.14 / 2**0.5, 0.14 / 2**0.5, 0.0],
            2,
        ),
    ],
)
def test_semantic_planner_preserves_e5_direction_and_terminal_behavior(
    terminal_behavior: str,
    call_id: str,
    expected_displacement: list[float],
    cleanup_count: int,
) -> None:
    def interpreter(_instruction, **_kwargs):
        step = _step(step_id="transport", reference="tray")
        step.update(
            task_type="E5",
            required_arm="none",
            orientation_goal="none",
            direction=("up" if terminal_behavior == "hold" else "front_right"),
            terminal_behavior=terminal_behavior,
        )
        return _result(step)

    candidate = TaskAgent(interpreter=interpreter).generate(
        "coordinated", _TEST_INSTRUCTION, candidate_count=1
    )["candidates"][0]
    graph = SemanticTaskPlanner().plan(
        candidate,
        {
            "schema_version": ROLE_BINDINGS_SCHEMA,
            "task_id": "coordinated",
            "candidate_id": candidate["candidate_id"],
            "reference_bindings": {"step_01.object": ["tray"]},
            "role_bindings": {},
        },
        [{"runtime_uid": "tray", "init_pos": [0.0, 0.0, 0.75]}],
    )

    call = graph["nodes"][0]["call"]
    assert call["call_id"] == call_id
    assert call["arguments"]["world_displacement"] == pytest.approx(
        expected_displacement
    )
    assert len(graph["nodes"]) == 1 + cleanup_count
    assert graph["task_groups"][0]["success"]["type"] == (
        "held_by_both_grippers" if terminal_behavior == "hold" else "semantic_goal"
    )


@pytest.mark.parametrize(
    ("task_type", "target_state"), [("E6", "open"), ("E7", "closed")]
)
def test_semantic_planner_rejects_out_of_scope_articulation_slide(
    task_type: str,
    target_state: str,
) -> None:
    def interpreter(_instruction, **_kwargs):
        step = _step(step_id="drawer", reference="drawer")
        step.update(
            task_type=task_type,
            required_arm="left_arm",
            orientation_goal="none",
            target_state=target_state,
        )
        return _result(step)

    candidate = TaskAgent(interpreter=interpreter).generate(
        "drawer_task", _TEST_INSTRUCTION, candidate_count=1
    )["candidates"][0]
    with pytest.raises(UnsupportedSemanticCapabilityError, match="only E1-E5"):
        SemanticTaskPlanner().plan(
            candidate,
            {
                "schema_version": ROLE_BINDINGS_SCHEMA,
                "task_id": "drawer_task",
                "candidate_id": candidate["candidate_id"],
                "reference_bindings": {"step_01.object": ["drawer"]},
                "role_bindings": {},
            },
            [{"runtime_uid": "drawer", "role": "articulation", "init_pos": [0, 0, 0]}],
        )


@pytest.mark.parametrize(
    ("task_type", "call_id", "argument_key", "argument_value"),
    [
        ("E8", "simulation.articulation_link_twist", "target_setting", 2),
        ("E9", "simulation.articulation_link_press", "target_state", "activated"),
    ],
)
def test_semantic_planner_rejects_out_of_scope_calibrated_articulation_call(
    task_type: str,
    call_id: str,
    argument_key: str,
    argument_value: object,
) -> None:
    def interpreter(_instruction, **_kwargs):
        step = _step(step_id="control", reference="control")
        step.update(
            task_type=task_type,
            required_arm="right_arm",
            orientation_goal="none",
        )
        step[argument_key] = argument_value
        return _result(step)

    candidate = TaskAgent(interpreter=interpreter).generate(
        "control_task", _TEST_INSTRUCTION, candidate_count=1
    )["candidates"][0]
    with pytest.raises(UnsupportedSemanticCapabilityError, match="only E1-E5"):
        SemanticTaskPlanner().plan(
            candidate,
            {
                "schema_version": ROLE_BINDINGS_SCHEMA,
                "task_id": "control_task",
                "candidate_id": candidate["candidate_id"],
                "reference_bindings": {"step_01.object": ["control"]},
                "role_bindings": {},
            },
            [{"runtime_uid": "control", "role": "articulation", "init_pos": [0, 0, 0]}],
        )
