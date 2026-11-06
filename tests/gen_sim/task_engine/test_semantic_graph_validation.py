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
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from embodichain.gen_sim.task_spec import Predicate, TaskSpec
from embodichain.gen_sim.task_engine.semantic_planner import SemanticTaskPlanner
from embodichain.gen_sim.task_engine.semantic_graph_validation import (
    validate_graph_coverage,
)
from .test_task_spec_generation import candidate, selector, step


def _plan(*, count: int = 0):
    source = candidate(step("E2", "cans", object=selector("cans", count=count)))
    bindings = {
        "schema_version": "action_engine_role_bindings_v1",
        "task_id": source["draft"]["task_id"],
        "candidate_id": source["candidate_id"],
        "reference_bindings": {
            "step_01.object": ["a", "b"][:count] if count else ["a"]
        },
        "role_bindings": {},
    }
    return SemanticTaskPlanner().plan_spec(
        source,
        bindings,
        [
            {
                "runtime_uid": uid,
                "init_pos": [0.1, -0.2, 0.85],
                "init_rot": [0.0, 0.0, 0.0],
            }
            for uid in ("a", "b")
        ],
    )


def _coverage(plan, **changes):
    arguments = {
        "graph": plan.graph,
        "task_spec": plan.task_spec,
        "candidate": plan.candidate,
        "role_bindings": plan.role_bindings,
        "lineage": plan.lineage,
    }
    arguments.update(changes)
    return validate_graph_coverage(**arguments)


def test_expansion_preserves_source_spec_and_covers_every_set_member() -> None:
    plan = _plan(count=2)
    report = _coverage(plan)
    assert report["status"] == "covered"
    assert len(plan.task_spec.milestones) == 1
    assert len(plan.graph["task_groups"]) == 2
    assert {m["object_uid"] for m in plan.lineage["step_01"]} == {"a", "b"}
    assert len(report["goals"][0]["groups"]) == 2
    assert report["runtime_evidence_status"] == "not_run"


@pytest.mark.parametrize(
    "damage",
    [
        "missing_member",
        "duplicate_member",
        "wrong_uid",
        "wrong_goal",
        "wrong_call",
        "wrong_arm",
    ],
)
def test_declaration_claims_cannot_hide_missing_or_wrong_coverage(damage: str) -> None:
    plan = _plan(count=2)
    changes = {}
    lineage = deepcopy(plan.lineage)
    if damage == "missing_member":
        lineage["step_01"].pop()
    elif damage == "duplicate_member":
        lineage["step_01"][1] = deepcopy(lineage["step_01"][0])
    elif damage == "wrong_uid":
        lineage["step_01"][0]["object_uid"] = "other"
    elif damage == "wrong_goal":
        changes["task_spec"] = replace(
            plan.task_spec,
            goal=(
                *plan.task_spec.goal,
                Predicate("pressed", {"subject": "role_001"}),
            ),
        )
    else:
        graph = deepcopy(plan.graph)
        if damage == "wrong_call":
            for node in graph["nodes"]:
                if node["call"].get("call_id") == "gen_sim.align_held":
                    node["call"]["call_id"] = "simulation.park"
                    node["call"]["arguments"] = {}
        else:
            from embodichain.gen_sim.task_engine.agent import (
                derive_scene_request,
                derive_success_spec,
            )
            from embodichain.gen_sim.task_engine.contracts import canonical_hash
            from embodichain.gen_sim.task_engine.task_spec import (
                task_spec_from_candidate,
            )

            source = deepcopy(plan.candidate)
            source["draft"]["steps"][0]["required_arm"] = "right_arm"
            source["scene_request"] = derive_scene_request(source["draft"])
            source["success_spec"] = derive_success_spec(source["draft"])
            source["semantic_hash"] = canonical_hash(source["draft"]["steps"])
            changes.update(candidate=source, task_spec=task_spec_from_candidate(source))
            for node in graph["nodes"]:
                node["call"].setdefault("resources", {})["primary"] = "left"
        changes["graph"] = graph
    changes["lineage"] = lineage
    assert _coverage(plan, **changes)["status"] == "uncovered"


def _frozen_plan():
    value = json.loads(
        (Path(__file__).parent / "data/semantic_graph_contract.json").read_text()
    )
    value["task_spec"] = TaskSpec.from_dict(value["task_spec"])
    return SimpleNamespace(**value)


def test_frozen_graph_retains_original_declared_coverage_without_execution():
    plan = _frozen_plan()
    before = deepcopy((plan.graph, plan.role_bindings, plan.lineage))
    assert _coverage(plan) == plan.coverage
    assert plan.coverage["status"] == "covered"
    assert plan.coverage["runtime_evidence_status"] == "not_run"
    assert (plan.graph, plan.role_bindings, plan.lineage) == before


@pytest.mark.parametrize(
    "damage",
    ["missing_member", "foreign_member", "wrong_task", "wrong_normative_goal"],
)
def test_stale_graph_or_grounding_cannot_claim_complete_coverage(damage):
    plan = _frozen_plan()
    if damage == "missing_member":
        plan.lineage[next(iter(plan.lineage))] = []
    elif damage == "foreign_member":
        plan.lineage[next(iter(plan.lineage))][0]["object_uid"] = "other"
    elif damage == "wrong_task":
        plan.graph["task_id"] = "another-task"
    else:
        from dataclasses import replace

        plan.task_spec = replace(
            plan.task_spec,
            goal=(
                replace(plan.task_spec.goal[0], expected=False),
                *plan.task_spec.goal[1:],
            ),
        )
    assert _coverage(plan)["status"] != "covered"
