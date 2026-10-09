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
import pytest
from types import SimpleNamespace
import torch

from embodichain.gen_sim.task_engine._task_program.task_spec_runtime import (
    evaluate_runtime_evidence,
)
from embodichain.gen_sim.task_engine._bundle_runner import _apply_task_spec_verdict
from embodichain.gen_sim.task_engine.contracts import canonical_hash
from .test_semantic_graph_validation import _plan, _coverage
from embodichain.utils.utility import save_config


def _evidence():
    plan = _plan()
    coverage = _coverage(plan)
    nodes = plan.graph["nodes"]
    place = next(
        n for n in nodes if n["call"].get("call_id") == "simulation.place_relative"
    )
    preset = "upright.final"
    program = {
        "program": {
            "items": [
                {
                    "name": n["id"],
                    "post": (
                        [{"kind": "wait_stable", "preset": preset}]
                        if n == place
                        else []
                    ),
                }
                for n in nodes
            ]
        }
    }
    constraints = {"presets": {preset: {"entity": "a", "kind": "upright"}}}
    contract = {
        "task_spec_digest": canonical_hash(plan.task_spec.to_dict()),
        "graph_digest": canonical_hash(plan.graph),
        "program_digest": canonical_hash(program),
        "constraints_digest": canonical_hash(constraints),
        "coverage": coverage,
    }
    runtime = {
        "segments": [
            {
                "name": n["id"],
                "active": [True],
                "successes": [True],
                "metadata": {
                    "runtime": {"env_ids": [0]},
                    "post_policies": (
                        [
                            {
                                "result": {
                                    "kind": "upright",
                                    "preset": preset,
                                    "accepted_mask": [True],
                                    "measurements": {"alignment": [1.0]},
                                }
                            }
                        ]
                        if n == place
                        else []
                    ),
                },
            }
            for n in nodes
        ]
    }
    final = {
        p.id: {
            "members": [
                {
                    "group_id": coverage["goals"][0]["groups"][0],
                    "uid": "a",
                    "part_id": None,
                    "accepted": [True],
                    "measurements": {"alignment": [1.0]},
                }
            ]
        }
        for p in plan.task_spec.goal
    }
    return plan, contract, runtime, final, program, constraints


def _evaluate(data):
    plan, contract, runtime, final, program, constraints = data
    return evaluate_runtime_evidence(
        plan.task_spec,
        contract,
        plan.graph,
        runtime,
        final,
        num_envs=1,
        run_id="local-test",
        program=program,
        constraints=constraints,
    )


def test_real_producer_keys_measured_milestone_and_final_snapshot_certify() -> None:
    evaluation = _evaluate(_evidence())
    assert evaluation["environments"][0]["certificate_ready"] is True
    assert evaluation["environments"][0]["milestones"][0]["status"] == "satisfied"


def test_early_upright_success_does_not_hide_failed_terminal_observation() -> None:
    data = _evidence()
    for record in data[3].values():
        record["members"][0]["accepted"] = [False]
        record["members"][0]["measurements"]["alignment"] = [0.0]
    result = _evaluate(data)["environments"][0]
    assert result["status"] == "failed"
    assert result["milestones"][0]["status"] == "satisfied"


def test_initial_observation_cannot_read_final_goal_through_fallback() -> None:
    from embodichain.gen_sim.task_spec import Predicate

    data = _evidence()
    plan = data[0]
    initial = Predicate(
        "object_upright",
        {"subject": plan.task_spec.goal[0].arguments["subject"]},
        predicate_id="initial",
        observation=f"goal.{plan.task_spec.goal[0].id}",
    )
    spec = replace(plan.task_spec, init=(initial,))
    data[1]["task_spec_digest"] = canonical_hash(spec.to_dict())
    result = evaluate_runtime_evidence(
        spec,
        data[1],
        plan.graph,
        data[2],
        data[3],
        num_envs=1,
        run_id="local-test",
        program=data[4],
        constraints=data[5],
    )
    assert result["environments"][0]["init"][0]["status"] == "unavailable"
    assert result["environments"][0]["certificate_ready"] is False


@pytest.mark.parametrize("alignment,accepted", [(1.0, True), (0.6, False)])
def test_final_native_pose_uses_normalized_preset_axis(
    tmp_path, alignment, accepted
) -> None:
    from embodichain.gen_sim.task_engine._task_program.task_spec_runtime import (
        capture_final_goals,
    )

    data = _evidence()
    plan, contract = data[:2]
    pose = torch.eye(4).unsqueeze(0)
    sine = (1 - alignment**2) ** 0.5
    pose[0, :3, :3] = torch.tensor(
        [[alignment, 0.0, -sine], [0.0, 1.0, 0.0], [sine, 0.0, alignment]]
    )
    native = SimpleNamespace(get_local_pose=lambda **_kwargs: pose)
    env = SimpleNamespace(sim=SimpleNamespace(get_rigid_object=lambda _uid: native))
    registration = SimpleNamespace(
        scene_binding=SimpleNamespace(
            rigid_objects=(SimpleNamespace(entity_id="a", simulation_uid="native-a"),)
        )
    )
    deployment = SimpleNamespace(integration=SimpleNamespace(registration=registration))
    save_config(tmp_path / "task_spec.json", plan.task_spec.to_dict())
    path = tmp_path / "task_program/constraints.json"
    path.parent.mkdir()
    save_config(
        path,
        {
            "presets": {
                f"gen_sim.{plan.graph['nodes'][-1]['id']}.stable": {
                    "entity": "a",
                    "kind": "upright",
                    "local_axis": [0.0, 0.0, 2.0],
                    "minimum_alignment": 0.9,
                }
            }
        },
    )
    goals = capture_final_goals(env, deployment, tmp_path, plan.graph, contract, 1)
    assert next(iter(goals.values()))["members"][0]["accepted"] == [accepted]


def test_workflow_consumes_full_verdict_not_only_flat_program_success(tmp_path) -> None:
    from .test_semantic_graph_validation import _contract
    from embodichain.gen_sim.task_engine.workflow import _environment_successes

    plan = _contract(tmp_path)
    save_config(tmp_path / "semantic_task_graph.json", plan.graph)
    report = {"environments": [{"env_id": 0, "success": True}]}
    assert _environment_successes(report, bundle=tmp_path) == [False]
    data = _evidence()
    save_config(tmp_path / "task_program/program.yaml", data[4])
    save_config(tmp_path / "task_program/constraints.json", data[5])
    from embodichain.gen_sim.task_engine.semantic_graph_validation import (
        write_graph_contract,
    )

    write_graph_contract(
        tmp_path,
        plan.graph,
        plan.task_spec,
        plan.candidate,
        plan.role_bindings,
        plan.lineage,
    )
    evaluation = _evaluate(data)
    report["runtime_result"] = {"task_spec_evaluation": evaluation}
    assert _environment_successes(report, bundle=tmp_path) == [True]
    evaluation["environments"][0]["goal"][0]["status"] = "failed"
    with pytest.raises(ValueError, match="summary"):
        _environment_successes(report, bundle=tmp_path)


@pytest.mark.parametrize(
    "damage",
    [
        "active",
        "env_ids",
        "preset",
        "wrong_kind",
        "missing_measurements",
        "wrong_uid",
        "missing_member",
        "summary_only",
        "short_mask",
    ],
)
def test_missing_or_misattributed_evidence_never_certifies(damage: str) -> None:
    data = _evidence()
    segment = next(s for s in data[2]["segments"] if s["metadata"]["post_policies"])
    record = next(iter(data[3].values()))
    if damage == "active":
        segment.pop("active")
    elif damage == "env_ids":
        segment["metadata"]["runtime"]["env_ids"] = [7]
    elif damage in {"preset", "wrong_kind", "missing_measurements"}:
        policy = segment["metadata"]["post_policies"][0]["result"]
        if damage == "missing_measurements":
            policy.pop("measurements")
        else:
            policy["preset" if damage == "preset" else "kind"] = "other"
    elif damage == "summary_only":
        record.clear()
        record.update(values=[True], status="satisfied")
    elif damage == "missing_member":
        record["members"] = []
    elif damage == "wrong_uid":
        record["members"][0]["uid"] = "other"
    else:
        record["members"][0]["accepted"] = []
    assert _evaluate(data)["environments"][0]["certificate_ready"] is False


@pytest.mark.parametrize(
    "identity",
    ["task_spec_digest", "graph_digest", "program_digest", "constraints_digest"],
)
def test_runtime_identity_drift_is_rejected(identity: str) -> None:
    data = _evidence()
    data[1][identity] = "foreign"
    with pytest.raises(ValueError, match="identity"):
        _evaluate(data)


@pytest.mark.parametrize(
    "status,ready,reason",
    [
        ("satisfied", True, "completed"),
        ("failed", False, "task_spec_failed"),
        ("unavailable", False, "task_spec_unverified"),
        ("unavailable", True, "task_spec_unverified"),
    ],
)
def test_new_normative_verdict_gates_success_but_legacy_stays_unchanged(
    status, ready, reason
) -> None:
    assert _apply_task_spec_verdict({}, [True], ["completed"]) == (
        [True],
        ["completed"],
    )
    runtime = {
        "task_spec_evaluation": {
            "environments": [
                {"env_id": 0, "status": status, "certificate_ready": ready}
            ]
        }
    }
    success, reasons = _apply_task_spec_verdict(runtime, [True], ["completed"])
    assert success == [status == "satisfied" and ready]
    assert reasons == [reason]
    assert _apply_task_spec_verdict(runtime, [False], ["execution_failed"]) == (
        [False],
        ["execution_failed"],
    )
