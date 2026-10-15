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
from types import SimpleNamespace

import pytest

from embodichain.gen_sim.task_engine.contracts import canonical_hash
from embodichain.gen_sim.task_engine.semantic_planner import SemanticTaskPlanner
from embodichain.gen_sim.task_engine._task_program.task_spec_runtime import (
    capture_final_goals,
    evaluate_runtime_evidence,
)
from embodichain.utils.utility import save_config
from .test_semantic_graph_validation import _coverage
from .test_task_spec_generation import candidate, selector, step


def _evidence(*, repeated=False):
    steps = [step("E5", "object", direction="up")]
    if repeated:
        steps.append(step("E5", "", object=selector(result="s1"), direction="up"))
    source = candidate(*steps)
    bindings = {
        "schema_version": "action_engine_role_bindings_v1",
        "task_id": source["draft"]["task_id"],
        "candidate_id": source["candidate_id"],
        "reference_bindings": {
            source["draft"]["steps"][0]["id"] + ".object": ["object"]
        },
        "role_bindings": {},
    }
    plan = SemanticTaskPlanner().plan_spec(
        source,
        bindings,
        [{"runtime_uid": "object", "init_pos": [0.0, 0.0, 0.6], "init_rot": [0.0] * 3}],
    )
    program = {"program": {"items": [{"name": n["id"]} for n in plan.graph["nodes"]]}}
    constraints = {"presets": {}}
    contract = {
        "task_spec_digest": canonical_hash(plan.task_spec.to_dict()),
        "graph_digest": canonical_hash(plan.graph),
        "program_digest": canonical_hash(program),
        "constraints_digest": canonical_hash(constraints),
        "coverage": _coverage(plan),
    }
    runtime = {
        "segments": [
            {
                "name": n["id"],
                "active": [True],
                "successes": [True],
                "metadata": {"runtime": {"env_ids": [0]}},
            }
            for n in plan.graph["nodes"]
        ]
    }
    observations = {
        group["id"]: {
            "group_id": group["id"],
            "uid": "object",
            "part_id": None,
            "accepted": [True],
            "measurements": {"scope": "coordinated_wiring_fixture"},
        }
        for group in plan.graph["task_groups"]
    }
    final = {
        predicate.id: {
            "members": [deepcopy(observations[g]) for g in coverage["groups"]]
        }
        for predicate in plan.task_spec.goal
        for coverage in contract["coverage"]["goals"]
        if predicate.id == coverage["id"]
    }
    return plan, contract, runtime, final, program, constraints, observations


def _evaluate(data, *, observed=True):
    plan, contract, runtime, final, program, constraints, observations = data
    extra = {"coordinated_milestones": observations} if observed else {}
    return evaluate_runtime_evidence(
        plan.task_spec,
        contract,
        plan.graph,
        runtime,
        final,
        num_envs=1,
        run_id="coordinated-observer-wiring",
        program=program,
        constraints=constraints,
        **extra,
    )["environments"][0]


def _wire(monkeypatch):
    from embodichain.gen_sim.task_engine._task_program import coordinated_evidence

    # Physical receipt authenticity is exercised in the observer's own tests.
    monkeypatch.setattr(
        coordinated_evidence,
        "coordinated_record_matches",
        lambda record, *_args, **_kwargs: record.get("measurements", {}).get("scope")
        == "coordinated_wiring_fixture",
    )


def test_completed_dual_hold_commands_cannot_certify_without_observation():
    assert _evaluate(_evidence(), observed=False)["certificate_ready"] is False


def test_collected_dual_hold_predicate_reaches_task_spec(monkeypatch):
    _wire(monkeypatch)
    result = _evaluate(_evidence())
    assert result["status"] == "satisfied"
    assert result["certificate_ready"] is True


def test_missing_final_dual_hold_cannot_borrow_milestone(monkeypatch):
    _wire(monkeypatch)
    data = _evidence()
    data[3].clear()
    result = _evaluate(data)
    assert result["milestones"][0]["status"] == "satisfied"
    assert result["certificate_ready"] is False


@pytest.mark.parametrize("damage", ["uid", "part", "group", "mask", "measurement"])
def test_wrong_coordinated_member_cannot_certify_milestone(monkeypatch, damage):
    _wire(monkeypatch)
    data = _evidence()
    record = next(iter(data[6].values()))
    if damage == "uid":
        record["uid"] = "other"
    elif damage == "part":
        record["part_id"] = "other"
    elif damage == "group":
        record["group_id"] = "other"
    elif damage == "mask":
        record["accepted"] = [1]
    else:
        record["measurements"] = {}
    assert _evaluate(data)["certificate_ready"] is False


def test_repeated_dual_hold_cannot_borrow_another_occurrence(monkeypatch):
    _wire(monkeypatch)
    data = _evidence(repeated=True)
    assert _evaluate(data)["certificate_ready"] is True
    first, second = data[0].graph["task_groups"]
    data[6][first["id"]] = deepcopy(data[6][second["id"]])
    assert _evaluate(data)["certificate_ready"] is False


@pytest.mark.parametrize("missing", [False, True])
def test_final_collector_requests_fresh_coordinated_observation(
    monkeypatch, tmp_path, missing
):
    from embodichain.gen_sim.task_engine._task_program import coordinated_evidence

    plan, contract, runtime, _, program, constraints, observations = _evidence()
    if missing:
        observations.clear()
    requested = []

    def observe(*_args, final=False):
        requested.append(final)
        return observations

    monkeypatch.setattr(coordinated_evidence, "capture_coordinated_holds", observe)
    save_config(tmp_path / "task_spec.json", plan.task_spec.to_dict())
    (tmp_path / "task_program").mkdir()
    save_config(tmp_path / "task_program/program.yaml", program)
    save_config(tmp_path / "task_program/constraints.json", constraints)
    env = SimpleNamespace(sim=object())
    deployment = SimpleNamespace(
        integration=SimpleNamespace(
            registration=SimpleNamespace(
                scene_binding=SimpleNamespace(rigid_objects=[])
            )
        )
    )
    result = capture_final_goals(
        env, deployment, tmp_path, plan.graph, contract, 1, runtime=runtime
    )
    assert requested == [True]
    assert ("unavailable" in result[plan.task_spec.goal[0].id]) is missing
