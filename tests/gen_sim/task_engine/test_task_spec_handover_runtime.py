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


@pytest.fixture(autouse=True)
def _receipt_boundary(monkeypatch):
    # Receipt authenticity is tested separately; this file tests evaluator wiring.
    from embodichain.gen_sim.task_engine._task_program import handover_evidence

    monkeypatch.setattr(
        handover_evidence,
        "handover_record_matches",
        lambda record, *_args, **_kwargs: record.get("measurements", {}).get("scope")
        == "observer_wiring_fixture",
    )


def _evidence(*, repeated: bool = False):
    steps = [step("E4", "object", transfer_arm="left_arm", receive_arm="right_arm")]
    if repeated:
        steps.append(
            step(
                "E4",
                "",
                object=selector(result="s1"),
                transfer_arm="right_arm",
                receive_arm="left_arm",
            )
        )
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
        [
            {
                "runtime_uid": "object",
                "init_pos": [0.2, -0.2, 0.85],
                "init_rot": [0.0] * 3,
            }
        ],
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
            name: {
                "group_id": group["id"],
                "uid": "object",
                "part_id": None,
                "accepted": [True],
                "measurements": {"scope": "observer_wiring_fixture"},
            }
            for name in ("handover_complete", "held")
        }
        for group in plan.graph["task_groups"]
    }
    final = {
        predicate.id: {
            "members": [
                deepcopy(observations[id][predicate.name]) for id in coverage["groups"]
            ]
        }
        for predicate in plan.task_spec.goal
        for coverage in contract["coverage"]["goals"]
        if predicate.id == coverage["id"]
    }
    return plan, contract, runtime, final, program, constraints, observations


def _evaluate(data, *, observed=True):
    plan, contract, runtime, final, program, constraints, observations = data
    return evaluate_runtime_evidence(
        plan.task_spec,
        contract,
        plan.graph,
        runtime,
        final,
        num_envs=1,
        run_id="handover-observer-wiring",
        program=program,
        constraints=constraints,
        handover_milestones=observations if observed else None,
    )["environments"][0]


def test_completed_commands_without_handover_observer_remain_unavailable():
    assert _evaluate(_evidence(), observed=False)["status"] == "unavailable"


def test_handover_event_and_terminal_held_observations_are_both_required():
    data = _evidence()
    data[3].clear()
    result = _evaluate(data)
    assert result["milestones"][0]["status"] == "satisfied"
    assert result["certificate_ready"] is False


def test_observer_results_reach_existing_task_spec_evaluator():
    result = _evaluate(_evidence())
    assert result["status"] == "satisfied"
    assert result["certificate_ready"] is True


@pytest.mark.parametrize("damage", ["uid", "part", "group", "mask", "measurement"])
def test_wrong_handover_member_cannot_certify_milestone(damage):
    data = _evidence()
    record = next(iter(data[6].values()))["handover_complete"]
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


def test_early_receipt_cannot_hide_failed_handover_segment():
    data = _evidence()
    data[2]["segments"][-1]["successes"] = [False]
    assert _evaluate(data)["certificate_ready"] is False


def test_repeated_handover_cannot_borrow_second_occurrence():
    data = _evidence(repeated=True)
    assert _evaluate(data)["certificate_ready"] is True
    first, second = data[0].graph["task_groups"]
    data[6][first["id"]] = deepcopy(data[6][second["id"]])
    assert _evaluate(data)["certificate_ready"] is False


@pytest.mark.parametrize("missing_held", [False, True])
def test_final_collector_requests_fresh_holding_not_milestone_flags(
    monkeypatch, tmp_path, missing_held
):
    from embodichain.gen_sim.task_engine._task_program import handover_evidence

    plan, contract, runtime, _, program, constraints, observations = _evidence()
    if missing_held:
        for record in observations.values():
            record.pop("held")
    requested = []

    def observe(*_args, final=False):
        requested.append(final)
        return observations

    monkeypatch.setattr(handover_evidence, "capture_handover_operations", observe)
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
    goal = next(p for p in plan.task_spec.goal if p.name == "held")
    assert ("unavailable" in result[goal.id]) is missing_held
