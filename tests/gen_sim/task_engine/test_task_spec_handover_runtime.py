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
from .test_task_spec_runtime import _placement_evidence, _table_region_evidence


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


def _placed_evidence(
    tmp_path,
    monkeypatch,
    *,
    relation="inside",
    transfer=True,
    damage=None,
    strong=False,
):
    from embodichain.gen_sim.task_engine._task_program import handover_evidence
    from embodichain.gen_sim.task_engine._task_program.task_spec_runtime import (
        capture_placement_milestones,
    )

    data = (
        _placement_evidence(tmp_path, task_type="E4")
        if relation == "inside"
        else _table_region_evidence(tmp_path, monkeypatch, task_type="E4")
    )
    if strong:
        from .test_support_runtime import _strong_case

        case = _strong_case(
            tmp_path, task_type="E4", relation=relation, reference="region"
        )
        data = case.data
        if damage == "support_trace":
            case.terminal["metadata"]["post_policies"] = []
    plan, contract, runtime, program, constraints, env, deployment = data
    group = plan.graph["task_groups"][0]
    transfer_records = (
        {
            group["id"]: {
                "handover_complete": {
                    "group_id": group["id"],
                    "uid": "cube",
                    "part_id": None,
                    "accepted": [True],
                    "measurements": {"scope": "observer_wiring_fixture"},
                }
            }
        }
        if transfer
        else {}
    )
    monkeypatch.setattr(
        handover_evidence,
        "capture_handover_operations",
        lambda *a, **k: transfer_records,
    )
    if damage == "release":
        for segment in runtime["segments"]:
            for call in segment["metadata"]["runtime"]["calls"]:
                call["effects"] = []
    elif damage in {"wrong_receiver", "receiver_and_receipt"}:
        for node in plan.graph["nodes"]:
            if node["call"].get("kind") == "place":
                node["call"]["resources"]["primary"] = "right"
        if damage == "receiver_and_receipt":
            for segment in runtime["segments"]:
                for call in segment["metadata"]["runtime"]["calls"]:
                    for receipt in call["effects"]:
                        receipt["effect_spec"]["state_expectations"][0][
                            "resource_id"
                        ] = "right"
        contract["graph_digest"] = canonical_hash(plan.graph)
    elif damage == "footprint":
        env.sim.get_rigid_object("cube").get_local_pose()[0, 0, 3] = 1.0
    final = capture_final_goals(
        env, deployment, tmp_path, plan.graph, contract, 1, runtime=runtime
    )
    placements = capture_placement_milestones(
        env, deployment, plan.graph, contract, runtime, 1, spec=plan.task_spec
    )
    if damage == "boundary" and group["id"] in placements:
        placements[group["id"]]["measurements"]["boundary_frames"] = [999]
    if damage == "foreign_placement" and group["id"] in placements:
        placements[group["id"]]["group_id"] = "other_occurrence"
    if damage == "foreign_transfer":
        transfer_records[group["id"]]["handover_complete"][
            "group_id"
        ] = "other_occurrence"
    return evaluate_runtime_evidence(
        plan.task_spec,
        contract,
        plan.graph,
        runtime,
        final,
        num_envs=1,
        run_id="handover-placement-wiring",
        program=program,
        constraints=constraints,
        placement_milestones=placements,
        handover_milestones=transfer_records,
    )["environments"][0]


@pytest.mark.parametrize("relation", ["inside", "on"])
def test_handover_place_uses_measured_relation_and_transfer(
    tmp_path, monkeypatch, relation
):
    result = _placed_evidence(tmp_path, monkeypatch, relation=relation)
    assert result["status"] == "satisfied"
    assert result["certificate_ready"] is True


def test_handover_final_placement_cannot_replace_transfer(tmp_path, monkeypatch):
    result = _placed_evidence(tmp_path, monkeypatch, transfer=False)
    assert result["goal"][0]["status"] == "satisfied"
    assert result["certificate_ready"] is False


@pytest.mark.parametrize(
    "damage",
    [
        "release",
        "wrong_receiver",
        "receiver_and_receipt",
        "footprint",
        "boundary",
        "foreign_transfer",
        "foreign_placement",
    ],
)
def test_handover_placement_rejects_incomplete_or_foreign_evidence(
    tmp_path, monkeypatch, damage
):
    result = _placed_evidence(tmp_path, monkeypatch, damage=damage)
    assert result["certificate_ready"] is False


@pytest.mark.parametrize("relation", ["inside", "on"])
def test_handover_strong_placement_retains_support_contract(
    tmp_path, monkeypatch, relation
):
    result = _placed_evidence(tmp_path, monkeypatch, relation=relation, strong=True)
    assert result["certificate_ready"] is True
    assert (
        result["final_measurements"]["goal_001"]["members"][0]["measurements"]["scope"]
        == "native_supported_placement"
    )


def test_handover_strong_support_never_downgrades_missing_trace(tmp_path, monkeypatch):
    result = _placed_evidence(
        tmp_path, monkeypatch, strong=True, damage="support_trace"
    )
    assert result["certificate_ready"] is False


@pytest.mark.parametrize("borrow_first", [False, True])
def test_repeated_handover_places_keep_independent_support_windows(
    tmp_path, monkeypatch, borrow_first
):
    from .test_support_runtime import _repeated_case, _records
    from embodichain.gen_sim.task_engine._task_program import handover_evidence

    case = _repeated_case(tmp_path, task_type="E4")
    if borrow_first:
        case.source["rows"][0]["sample"] = deepcopy(case.first_sample)
    plan, contract, runtime, program, constraints, env, deployment = case.data
    transfers = {
        group["id"]: {
            "handover_complete": {
                "group_id": group["id"],
                "uid": "cube",
                "part_id": None,
                "accepted": [True],
                "measurements": {"scope": "observer_wiring_fixture"},
            }
        }
        for group in plan.graph["task_groups"]
    }
    monkeypatch.setattr(
        handover_evidence, "capture_handover_operations", lambda *a, **k: transfers
    )
    final, history = _records(tmp_path, case)
    row = evaluate_runtime_evidence(
        plan.task_spec,
        contract,
        plan.graph,
        runtime,
        final,
        num_envs=1,
        run_id="repeated-handover-place",
        program=program,
        constraints=constraints,
        placement_milestones=history,
        handover_milestones=transfers,
    )["environments"][0]
    milestones = {m["predicate_id"]: m for m in row["milestones"]}
    assert milestones["milestone:step_01"]["status"] == "satisfied"
    assert row["certificate_ready"] is (not borrow_first)
    if not borrow_first:
        assert (
            milestones["milestone:step_01.acquired"]["observed"]["step"]
            < milestones["milestone:step_01"]["observed"]["step"]
        )
        assert (
            milestones["milestone:step_02.acquired"]["observed"]["step"]
            < milestones["milestone:step_02"]["observed"]["step"]
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
