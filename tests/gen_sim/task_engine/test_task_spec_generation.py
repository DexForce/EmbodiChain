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
