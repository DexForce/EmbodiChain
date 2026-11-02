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

"""An explicit coarse turn must not invent a printed-label goal."""

from __future__ import annotations

from copy import deepcopy
import math
from dataclasses import dataclass

import pytest

from embodichain.gen_sim.task_engine.orchestration.contracts import ROLE_BINDINGS_SCHEMA
from embodichain.gen_sim.task_engine.semantic_planner import (
    SemanticTaskPlanner,
    UnsupportedSemanticCapabilityError,
)
from embodichain.gen_sim.task_engine.semantic_graph_validation import (
    validate_graph_coverage,
)
from .test_task_spec_generation import candidate, step


def _plan(*, coarse=False, labels=None):
    value = candidate(
        step("E8", "knob", **({"twist_goal_mode": "coarse"} if coarse else {}))
    )
    sid = value["draft"]["steps"][0]["id"]
    binding = {
        "schema_version": ROLE_BINDINGS_SCHEMA,
        "task_id": value["draft"]["task_id"],
        "candidate_id": value["candidate_id"],
        "reference_bindings": {f"{sid}.object": ["knob"]},
        "role_bindings": {},
    }
    objects = [
        {
            "runtime_uid": "knob",
            "init_pos": [0.1, -0.2, 0.8],
            "attributes": {
                "twist_control_joint": "knob_joint",
                "twist_setting_labels": labels or {"1": math.pi / 4},
            },
        }
    ]
    return SemanticTaskPlanner().plan_spec(value, binding, objects)


def test_explicit_coarse_uses_calibration_only_as_a_planning_reference():
    plan = _plan(coarse=True)
    original = plan.candidate["draft"]["steps"][0]
    assert original["target_setting"] == 0
    assert original["twist_goal_mode"] == "coarse"
    control_calls = [
        n["call"]
        for n in plan.graph["nodes"]
        if n["call"].get("call_id", "").startswith("gen_sim.twist")
    ]
    assert control_calls and all(c["arguments"]["setting"] == 1 for c in control_calls)
    goal = next(g for g in plan.task_spec.goal if g.arguments.get("task_type") == "E8")
    assert goal.arguments["twist_goal_mode"] == "coarse"
    assert "target_setting" not in goal.arguments
    report = validate_graph_coverage(
        graph=plan.graph,
        task_spec=plan.task_spec,
        candidate=plan.candidate,
        role_bindings=plan.role_bindings,
        lineage=plan.lineage,
    )
    assert report["status"] == "covered"


def test_legacy_unsatisfied_label_is_still_rejected():
    with pytest.raises(UnsupportedSemanticCapabilityError, match="source-calibrated"):
        _plan()


def test_legacy_calibrated_label_keeps_its_original_recipe():
    plan = _plan(labels={"0": math.pi / 4})
    assert "twist_goal_mode" not in plan.candidate["draft"]["steps"][0]
    controls = [
        n["call"]
        for n in plan.graph["nodes"]
        if n["call"].get("call_id", "").startswith("gen_sim.twist")
    ]
    assert all(c["arguments"]["setting"] == 0 for c in controls)


def test_coarse_reference_has_motion_headroom_and_respects_the_call_budget():
    # A near-zero printed mark cannot cover contact travel plus initial drift.
    labels = {"0": math.radians(15.24), "3": math.radians(75), "5": math.radians(285)}
    plan = _plan(coarse=True, labels=labels)
    controls = [
        n["call"]
        for n in plan.graph["nodes"]
        if n["call"].get("call_id", "").startswith("gen_sim.twist")
    ]
    assert all(c["arguments"]["setting"] == 3 for c in controls)
    assert (
        plan.task_spec.semantic_hash
        == _plan(coarse=True, labels={"3": labels["3"]}).task_spec.semantic_hash
    )


def test_coarse_rejects_calibration_references_beyond_the_bounded_call_budget():
    with pytest.raises(UnsupportedSemanticCapabilityError, match="bounded"):
        _plan(coarse=True, labels={"5": math.radians(285)})


def test_legacy_small_reference_is_not_replaced_by_coarse_headroom_selection():
    plan = _plan(labels={"0": math.radians(15.24), "3": math.radians(75)})
    controls = [
        n["call"]
        for n in plan.graph["nodes"]
        if n["call"].get("call_id", "").startswith("gen_sim.twist")
    ]
    assert all(c["arguments"]["setting"] == 0 for c in controls)


@pytest.mark.parametrize(
    "kind,mode,setting",
    [("E1", "coarse", 0), ("E8", "unknown", 0), ("E8", "coarse", 1)],
)
def test_coarse_mode_is_explicit_typed_and_cannot_hide_a_numeric_goal(
    kind, mode, setting
):
    with pytest.raises(ValueError, match="twist_goal_mode|coarse"):
        candidate(step(kind, "knob", twist_goal_mode=mode, target_setting=setting))


def test_coarse_graph_cannot_change_reference_between_chunks():
    plan = _plan(coarse=True)
    graph = deepcopy(plan.graph)
    controls = [
        n
        for n in graph["nodes"]
        if n["call"].get("call_id", "").startswith("gen_sim.twist.")
    ]
    controls[-1]["call"]["arguments"]["setting"] = 2
    report = validate_graph_coverage(
        graph=graph,
        task_spec=plan.task_spec,
        candidate=plan.candidate,
        role_bindings=plan.role_bindings,
        lineage=plan.lineage,
    )
    assert report["status"] != "covered"


@dataclass
class _Route:
    operation_id: str
    chunk_angle: float = math.radians(5)


def test_coarse_motion_span_uses_explicit_intent_lineage_not_task_or_object_ids():
    from embodichain.gen_sim.task_engine.task_program_bundle import _coarse_twist_routes

    routes = (_Route("expanded_a"), _Route("legacy_b"))
    value = {
        "draft": {
            "steps": [
                {"id": "s1", "task_type": "E8", "twist_goal_mode": "coarse"},
                {"id": "s2", "task_type": "E8"},
            ]
        }
    }
    lineage = {"s1": [{"group_id": "expanded_a"}], "s2": [{"group_id": "legacy_b"}]}
    selected = _coarse_twist_routes(routes, value, lineage)
    assert selected[0].chunk_angle == math.radians(20)
    assert selected[1] is routes[1]
    assert routes[0].chunk_angle == math.radians(5)


def test_legacy_motion_span_is_unchanged_and_coarse_requires_lineage():
    from embodichain.gen_sim.task_engine.task_program_bundle import _coarse_twist_routes

    routes = (_Route("s1"),)
    assert _coarse_twist_routes(routes, None, None) is routes
    value = {
        "draft": {
            "steps": [{"id": "s1", "task_type": "E8", "twist_goal_mode": "coarse"}]
        }
    }
    with pytest.raises(ValueError, match="lineage"):
        _coarse_twist_routes(routes, value, None)
