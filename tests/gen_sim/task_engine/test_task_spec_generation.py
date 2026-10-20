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
from typing import Any

import pytest

from embodichain.gen_sim.task_engine.agent import TaskAgent
from embodichain.gen_sim.task_engine.interpretation import InstructionDraftResult
from embodichain.gen_sim.task_engine.task_spec import task_spec_from_candidate


def selector(
    reference: str = "", *, result: str = "", count: int = 0
) -> dict[str, Any]:
    return {
        "kind": "step_result" if result else "scene_ref" if reference else "none",
        "step_id": result,
        "reference": reference,
        "quantifier": "count" if count else "one",
        "count": count,
    }


def step(kind: str, reference: str, **changes: Any) -> dict[str, Any]:
    value = {
        "id": "",
        "task_type": kind,
        "object": selector(reference),
        "target": selector(),
        "relation": "none",
        "required_arm": "auto" if kind not in {"E4", "E5"} else "none",
        "transfer_arm": "auto" if kind == "E4" else "none",
        "receive_arm": "auto" if kind == "E4" else "none",
        "orientation_goal": "upright" if kind == "E2" else "none",
        "target_state": "none",
        "target_setting": 0,
        "layout": "none",
        "axis": "none",
        "direction": "none",
        "terminal_behavior": "hold" if kind in {"E4", "E5"} else "none",
        "depends_on": [],
    }
    value.update(changes)
    return value


def candidate(*steps: dict[str, Any]) -> dict[str, Any]:
    values = deepcopy(steps)
    for index, value in enumerate(values, 1):
        value["id"] = f"s{index}"
        value["depends_on"] = [f"s{index-1}"] if index > 1 else []
    result = InstructionDraftResult(
        intent={"steps": list(values)},
        model="test",
        attempts=1,
        latency_seconds=0.0,
        normalizations=(),
    )
    return TaskAgent(interpreter=lambda *_a, **_k: result).generate(
        "demo", "Validated multi-stage task", candidate_count=1
    )["candidates"][0]


def test_explicit_pour_motion_is_distinct_and_legacy_fluid_identity_is_preserved():
    legacy = candidate(step("E3", "vessel", target=selector("cup"), relation="above"))
    fluid = candidate(
        step(
            "E3",
            "vessel",
            target=selector("cup"),
            relation="above",
            pour_goal_mode="fluid",
        )
    )
    motion = candidate(
        step(
            "E3",
            "vessel",
            target=selector("cup"),
            relation="above",
            pour_goal_mode="motion",
        )
    )
    old_spec, fluid_spec, motion_spec = [
        task_spec_from_candidate(item) for item in (legacy, fluid, motion)
    ]
    assert "pour_goal_mode" not in legacy["draft"]["steps"][0]
    assert old_spec.goal[0].name == fluid_spec.goal[0].name == "poured"
    assert old_spec.semantic_hash == fluid_spec.semantic_hash
    assert motion_spec.goal[0].name == "pour_motion"
    assert motion_spec.milestones[0].achieve[0].name == "pour_motion"
    assert motion_spec.semantic_hash != old_spec.semantic_hash
    assert motion["scene_request"] == legacy["scene_request"]


def test_pour_motion_remains_a_historical_achievement_after_later_placement():
    spec = task_spec_from_candidate(
        candidate(
            step(
                "E3",
                "vessel",
                target=selector("cup"),
                relation="above",
                pour_goal_mode="motion",
            ),
            step("E1", "vessel", target=selector("table"), relation="on"),
        )
    )
    assert {item.name for item in spec.goal} == {"pour_motion", "semantic_goal"}
    assert spec.milestones[1].after == (spec.milestones[0].id,)


@pytest.mark.parametrize(
    "kind,mode", [("E2", "motion"), ("E3", "contact_only"), ("E3", None), ("E3", True)]
)
def test_pour_goal_mode_rejects_wrong_task_or_unknown_meaning(kind, mode):
    from embodichain.gen_sim.task_engine.interpretation import (
        validate_instruction_intent,
    )

    value = step(kind, "vessel", id="s1", pour_goal_mode=mode)
    if kind == "E3":
        value.update(target=selector("cup"), relation="above")
    with pytest.raises(ValueError):
        validate_instruction_intent({"steps": [value]})


def test_independent_upright_subjects_are_both_terminal_goals() -> None:
    spec = task_spec_from_candidate(
        candidate(step("E2", "red can"), step("E2", "blue can"))
    )
    assert len(spec.goal) == 2
    assert len({item.arguments["subject"] for item in spec.goal}) == 2
    assert len(spec.milestones) == 2


def test_same_subject_last_position_supersedes_earlier_position() -> None:
    spec = task_spec_from_candidate(
        candidate(
            step("E1", "can", target=selector("table"), relation="on"),
            step("E1", "can", target=selector("box"), relation="inside"),
        )
    )
    assert len(spec.goal) == 1
    assert spec.goal[0].arguments["relation"] == "inside"
    assert len(spec.milestones) == 2


def test_upright_preserves_supported_position_but_not_obsolete_orientation() -> None:
    spec = task_spec_from_candidate(
        candidate(
            step(
                "E1",
                "can",
                target=selector("box"),
                relation="on",
                orientation_goal="preserve",
            ),
            step("E2", "can"),
        )
    )
    assert {item.name for item in spec.goal} == {"semantic_goal", "object_upright"}
    placement = next(item for item in spec.goal if item.name == "semantic_goal")
    assert placement.arguments["relation"] == "on"
    assert "orientation_goal" not in placement.arguments
    assert len(spec.milestones[0].achieve) == 2


def test_open_then_close_keeps_both_milestones_but_only_closed_goal() -> None:
    spec = task_spec_from_candidate(
        candidate(
            step("E6", "drawer", target_state="open"),
            step("E6", "drawer", target_state="closed"),
        )
    )
    assert len(spec.goal) == 1
    assert spec.goal[0].arguments["target_state"] == "closed"
    assert len(spec.milestones) == 2


def test_handover_then_place_does_not_require_terminal_hold() -> None:
    spec = task_spec_from_candidate(
        candidate(
            step("E4", "can", transfer_arm="left_arm", receive_arm="right_arm"),
            step("E1", "can", target=selector("table"), relation="on"),
        )
    )
    assert {item.name for item in spec.goal} == {"semantic_goal"}
    assert spec.milestones[0].achieve[0].name == "handover_complete"


def test_explicit_receiver_changes_spec_identity_but_auto_remains_a_witness_choice() -> (
    None
):
    first = task_spec_from_candidate(
        candidate(step("E4", "can", transfer_arm="left_arm", receive_arm="right_arm"))
    )
    second = task_spec_from_candidate(
        candidate(step("E4", "can", transfer_arm="right_arm", receive_arm="left_arm"))
    )
    assert first.semantic_hash != second.semantic_hash
    assert first.goal[0].name == "held"
    assert first.goal[0].arguments["receive_arm"] == "right_arm"
    assert any(
        r.consumer == "embodiment" and r.value == ["left_arm", "right_arm"]
        for r in first.requirements
    )


def test_quantified_and_single_roles_are_not_silently_merged() -> None:
    spec = task_spec_from_candidate(
        candidate(
            step("E2", "cans"),
            step("E2", "cans", object=selector("cans", count=2)),
        )
    )
    assert len(spec.roles) == 2
    assert {(role.quantifier, role.count) for role in spec.roles} == {
        ("one", 0),
        ("count", 2),
    }


def test_set_step_result_cannot_silently_select_one_subject() -> None:
    with pytest.raises(ValueError, match="step_result.*multiple|multiple.*step_result"):
        task_spec_from_candidate(
            candidate(
                step("E2", "cans", object=selector("cans", count=2)),
                step(
                    "E1",
                    "",
                    object=selector(result="s1"),
                    target=selector("table"),
                    relation="on",
                ),
            )
        )


def test_pour_achievement_survives_later_vessel_position_and_return_is_not_hold() -> (
    None
):
    spec = task_spec_from_candidate(
        candidate(
            step("E3", "vessel", target=selector("cup"), relation="above"),
            step("E1", "vessel", target=selector("table"), relation="on"),
        )
    )
    assert {item.name for item in spec.goal} == {"poured", "semantic_goal"}
    returned = task_spec_from_candidate(
        candidate(step("E5", "tray", terminal_behavior="place"))
    )
    assert {item.name for item in returned.goal} == {"semantic_goal"}
    assert len(returned.milestones) == 2
    acquire = next(m for m in returned.milestones if m.id.endswith(".acquired"))
    released = next(m for m in returned.milestones if not m.id.endswith(".acquired"))
    assert acquire.achieve[0].name == "held_by_both_grippers"
    assert released.after == (acquire.id,)
    from embodichain.gen_sim.task_spec import evaluate_task_spec

    evidence = {p.id: True for p in returned.goal}
    trajectory = [
        {acquire.achieve[0].id: True, released.achieve[0].id: False},
        {acquire.achieve[0].id: False, released.achieve[0].id: True},
    ]
    assert evaluate_task_spec(
        returned, {"goal": evidence}, trajectory=trajectory
    ).certificate_ready
