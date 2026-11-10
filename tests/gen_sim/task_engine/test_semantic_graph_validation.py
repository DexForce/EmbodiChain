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

"""Coverage is checked against a fixed source graph, without a live planner."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from embodichain.gen_sim.task_spec import TaskSpec
from embodichain.gen_sim.task_engine.semantic_graph_validation import (
    validate_graph_coverage,
)


def _plan():
    value = json.loads(
        (Path(__file__).parent / "data/semantic_graph_contract.json").read_text()
    )
    value["task_spec"] = TaskSpec.from_dict(value["task_spec"])
    return SimpleNamespace(**value)


def _coverage(plan, **changes):
    values = dict(
        graph=plan.graph,
        task_spec=plan.task_spec,
        candidate=plan.candidate,
        role_bindings=plan.role_bindings,
        lineage=plan.lineage,
    )
    values.update(changes)
    return validate_graph_coverage(**values)


def test_frozen_graph_retains_original_declared_coverage_without_execution():
    plan = _plan()
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
    plan = _plan()
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


@pytest.mark.parametrize(
    "field",
    [
        "action_options",
        "skill_options",
        "action_invocation",
        "skill_invocation",
        "atomic_action",
        "atomic_skill",
        "AtomicAction",
        "AtomicSkill",
    ],
)
def test_semantic_graph_rejects_execution_aliases_inside_calls(field: str) -> None:
    from embodichain.gen_sim.task_engine.semantic_graph import (
        validate_semantic_task_graph,
    )

    graph = deepcopy(_plan().graph)
    graph["nodes"][0]["call"]["arguments"][field] = {}
    with pytest.raises(ValueError):
        validate_semantic_task_graph(graph)
