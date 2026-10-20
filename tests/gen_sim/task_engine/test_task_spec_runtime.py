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
    assert evaluation["observer_revision"] == 4
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


def test_unmodeled_poured_goal_reports_explicit_boundary_without_stability_fallback(
    tmp_path,
) -> None:
    from embodichain.gen_sim.task_spec import Predicate
    from embodichain.gen_sim.task_engine._task_program.task_spec_runtime import (
        capture_final_goals,
    )

    plan, contract = _evidence()[:2]
    original = plan.task_spec.goal[0]
    predicate = Predicate(
        "poured",
        {**original.arguments, "task_type": "E3"},
        predicate_id=original.id,
    )
    spec = replace(plan.task_spec, goal=(predicate,))
    save_config(tmp_path / "task_spec.json", spec.to_dict())
    path = tmp_path / "task_program/constraints.json"
    path.parent.mkdir()
    save_config(path, {"presets": {}})
    registration = SimpleNamespace(
        scene_binding=SimpleNamespace(
            rigid_objects=(SimpleNamespace(entity_id="a", simulation_uid="native-a"),)
        )
    )
    goals = capture_final_goals(
        SimpleNamespace(sim=SimpleNamespace()),
        SimpleNamespace(integration=SimpleNamespace(registration=registration)),
        tmp_path,
        plan.graph,
        contract,
        1,
    )
    assert goals[predicate.id] == {
        "unavailable": [
            "Fluid transfer is unmodeled; tilt/restore pose evidence "
            "does not certify the poured predicate."
        ]
    }


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


def _articulation_evidence(states=("open",), *, num_envs=1):
    from embodichain.gen_sim.task_engine.semantic_planner import SemanticTaskPlanner
    from .test_task_spec_generation import candidate, step

    source = candidate(*(step("E6", "drawer", target_state=state) for state in states))
    references = [f"{item['id']}.object" for item in source["draft"]["steps"]]
    bindings = {
        "schema_version": "action_engine_role_bindings_v1",
        "task_id": source["draft"]["task_id"],
        "candidate_id": source["candidate_id"],
        "reference_bindings": {key: ["drawer"] for key in references},
        "role_bindings": {key: "part_test" for key in references},
    }
    plan = SemanticTaskPlanner().plan_spec(
        source,
        bindings,
        [{"runtime_uid": "drawer", "init_pos": [0.1, -0.2, 0.85]}],
    )
    coverage = _coverage(plan)
    groups = {group["id"]: group for group in plan.graph["task_groups"]}
    program = {"program": {"items": []}}
    segments = []
    env_ids, accepted = list(range(num_envs)), [True] * num_envs
    targets = {"open": -0.08, "closed": 0.0}
    for index, node in enumerate(plan.graph["nodes"]):
        group = groups[node["task_instance_id"]]
        state = next(
            item["target_state"]
            for item in source["draft"]["steps"]
            if item["id"] == group["id"]
        )
        target, tolerance = targets[state], 0.03
        declaration = {
            "kind": "articulation_joint_position",
            "articulation": "drawer",
            "joint": "slide",
            "minimum_position": target - tolerance,
            "maximum_position": target + tolerance,
        }
        program["program"]["items"].append(
            {
                "name": node["id"],
                "post": [
                    {
                        "kind": "wait_stable",
                        "entity": "drawer",
                        "preset": f"gen_sim.articulation.drawer.part_test.{state}",
                    }
                ],
                "validators": [declaration],
            }
        )
        path = ["program", "items", index, "validators", 0]
        segments.append(
            {
                "name": node["id"],
                "active": accepted.copy(),
                "successes": accepted.copy(),
                "end_steps": [index + 1] * num_envs,
                "metadata": {
                    "runtime": {"env_ids": env_ids.copy()},
                    "validation": {
                        "env_ids": env_ids.copy(),
                        "runtime_success_mask": accepted.copy(),
                        "eligible_mask_before_validation": accepted.copy(),
                        "post_policy_success_mask": accepted.copy(),
                        "accepted_mask": accepted.copy(),
                        "validators": [
                            {
                                "validator_index": 0,
                                "kind": "articulation_joint_position",
                                "source_path": path,
                                "result_mask": accepted.copy(),
                                "result": {
                                    "kind": "articulation_joint_position",
                                    "articulation_id": "drawer",
                                    "joint": "slide",
                                    "source_path": path.copy(),
                                    "minimum_position": target - tolerance,
                                    "maximum_position": target + tolerance,
                                    "env_ids": env_ids.copy(),
                                    "joint_position": [target] * num_envs,
                                    "accepted_mask": accepted.copy(),
                                },
                            }
                        ],
                    },
                    "post_policies": [
                        {
                            "policy_index": 0,
                            "kind": "wait_stable",
                            "source_path": ["program", "items", index, "post", 0],
                            "result_mask": accepted.copy(),
                            "result": {
                                "joint": "slide",
                                "target": target,
                                "observed": [target] * num_envs,
                                "tolerance": tolerance,
                                "stable_seconds": 1.0,
                                "accepted": accepted.copy(),
                                "evidence": "joint_position_and_stability",
                                "contact_qualification": "not_provided",
                            },
                        }
                    ],
                },
            }
        )
    constraints = {"presets": {}}
    contract = {
        "task_spec_digest": canonical_hash(plan.task_spec.to_dict()),
        "graph_digest": canonical_hash(plan.graph),
        "program_digest": canonical_hash(program),
        "constraints_digest": canonical_hash(constraints),
        "coverage": coverage,
    }
    final = {
        predicate.id: {
            "members": [
                {
                    "group_id": coverage["goals"][0]["groups"][0],
                    "uid": "drawer",
                    "part_id": "part_test",
                    "accepted": accepted.copy(),
                    "measurements": {
                        "joint_position": [targets[states[-1]]] * num_envs,
                        "joint_target": targets[states[-1]],
                    },
                }
            ]
        }
        for predicate in plan.task_spec.goal
    }
    return plan, contract, {"segments": segments}, final, program, constraints


def _evaluate_articulation(data, *, num_envs=1):
    plan, contract, runtime, final, program, constraints = data
    return evaluate_runtime_evidence(
        plan.task_spec,
        contract,
        plan.graph,
        runtime,
        final,
        num_envs=num_envs,
        run_id="articulation-test",
        program=program,
        constraints=constraints,
    )["environments"]


@pytest.mark.parametrize("states", [("open",), ("open", "closed")])
def test_e6_measured_boundaries_certify_ordered_achievements(states) -> None:
    data = _articulation_evidence(states)
    result = _evaluate_articulation(data)[0]
    assert result["certificate_ready"] is True
    assert [item["status"] for item in result["milestones"]] == ["satisfied"] * len(
        states
    )
    assert [item.arguments["target_state"] for item in data[0].task_spec.goal] == [
        states[-1]
    ]


@pytest.mark.parametrize(
    "damage",
    [
        "missing_validator",
        "missing_post",
        "failed_validator",
        "failed_post",
        "wrong_uid",
        "wrong_joint",
        "wrong_validator_path",
        "wrong_post_path",
        "wrong_state",
        "wrong_part",
        "wrong_call_state",
        "wrong_call_uid",
        "wrong_env_ids",
        "short_mask",
        "nonfinite_position",
        "out_of_range_position",
    ],
)
def test_e6_final_success_never_fills_missing_or_foreign_history(damage) -> None:
    data = _articulation_evidence(("open", "closed"))
    first_group = data[0].graph["task_groups"][0]
    boundary = next(
        segment
        for segment in data[2]["segments"]
        if segment["name"] == first_group["node_ids"][-1]
    )
    metadata = boundary["metadata"]
    validation = metadata["validation"]
    validator = validation["validators"][0]
    post = metadata["post_policies"][0]
    if damage == "missing_validator":
        validation["validators"] = []
    elif damage == "missing_post":
        metadata["post_policies"] = []
    elif damage == "failed_validator":
        validator["result_mask"] = validator["result"]["accepted_mask"] = [False]
    elif damage == "failed_post":
        post["result_mask"] = post["result"]["accepted"] = [False]
    elif damage == "wrong_uid":
        validator["result"]["articulation_id"] = "another_drawer"
    elif damage == "wrong_joint":
        validator["result"]["joint"] = "another_joint"
    elif damage == "wrong_validator_path":
        validator["result"]["source_path"] = ["program", "items", 99, "validators", 0]
    elif damage == "wrong_post_path":
        post["source_path"] = ["program", "items", 99, "post", 0]
    elif damage == "wrong_state":
        post["result"]["target"] = 0.0
    elif damage in {"wrong_part", "wrong_call_state", "wrong_call_uid"}:
        node = data[0].graph["nodes"][0]
        key, value = {
            "wrong_part": ("part", "part_other"),
            "wrong_call_state": ("state", "closed"),
            "wrong_call_uid": ("object", "another_drawer"),
        }[damage]
        node["call"]["arguments"][key] = value
        data[1]["graph_digest"] = canonical_hash(data[0].graph)
    elif damage == "wrong_env_ids":
        validator["result"]["env_ids"] = [7]
    elif damage == "short_mask":
        validator["result"]["accepted_mask"] = []
    elif damage == "nonfinite_position":
        validator["result"]["joint_position"] = [float("nan")]
    else:
        validator["result"]["joint_position"] = [0.0]
    result = _evaluate_articulation(data)[0]
    assert result["goal"][0]["status"] == "satisfied"
    assert result["certificate_ready"] is False
    assert result["milestones"][0]["status"] != "satisfied"


@pytest.mark.parametrize("identities", [[0, 1], [1, 0]])
def test_e6_receipt_masks_are_evaluated_per_environment(identities) -> None:
    data = _articulation_evidence(num_envs=2)
    metadata = data[2]["segments"][-1]["metadata"]
    validator = metadata["validation"]["validators"][0]
    metadata["runtime"]["env_ids"] = identities
    metadata["validation"]["env_ids"] = validator["result"]["env_ids"] = identities
    mask = [identity == 0 for identity in identities]
    metadata["validation"]["accepted_mask"] = mask
    validator["result_mask"] = validator["result"]["accepted_mask"] = mask
    post = metadata["post_policies"][0]
    post["result_mask"] = post["result"]["accepted"] = mask
    rows = _evaluate_articulation(data, num_envs=2)
    assert rows[0]["certificate_ready"] is True
    assert rows[1]["certificate_ready"] is False


def _placement_evidence(
    tmp_path,
    *,
    x=0.0,
    z=0.045,
    hole=False,
    relation="inside",
    reference_uid="region",
    task_type="E1",
):
    import numpy as np
    import trimesh
    from embodichain.gen_sim.task_engine.semantic_planner import SemanticTaskPlanner
    from .test_task_spec_generation import candidate, selector, step

    source = candidate(
        step(
            task_type,
            "cube",
            target=selector(reference_uid),
            relation=relation,
            **(
                {"required_arm": "left_arm"}
                if task_type == "E1"
                else {
                    "transfer_arm": "right_arm",
                    "receive_arm": "left_arm",
                    "terminal_behavior": "place",
                }
            ),
        )
    )
    bindings = {
        "schema_version": "action_engine_role_bindings_v1",
        "task_id": source["draft"]["task_id"],
        "candidate_id": source["candidate_id"],
        "reference_bindings": {
            "step_01.object": ["cube"],
            "step_01.target": [reference_uid],
        },
        "role_bindings": {},
    }
    plan = SemanticTaskPlanner().plan_spec(
        source,
        bindings,
        [
            {"runtime_uid": uid, "init_pos": [0.0, 0.0, 0.7], "init_rot": [0.0] * 3}
            for uid in ("cube", reference_uid)
        ],
    )
    coverage = _coverage(plan)
    nodes = plan.graph["nodes"]
    placement = next(
        n
        for n in nodes
        if n["call"]["kind"] == "place"
        or n["call"].get("call_id") == "simulation.place_relative"
    )
    program = {"program": {"items": [{"name": n["id"], "post": []} for n in nodes]}}
    constraints = {"presets": {}}
    contract = {
        "task_spec_digest": canonical_hash(plan.task_spec.to_dict()),
        "graph_digest": canonical_hash(plan.graph),
        "program_digest": canonical_hash(program),
        "constraints_digest": canonical_hash(constraints),
        "coverage": coverage,
    }
    invocation = "program/place:0"
    proof = {
        "boundary": {"kind": "terminal"},
        "timestamp": 0.08,
        "observation_revision": 2,
        "effect_spec": {
            "effect_kind": "release",
            "invocation_id": invocation,
            "invocation_revision": 0,
            "env_ids": [0],
            "state_expectations": [
                {
                    "expectation_id": "source",
                    "relation": "detached",
                    "object_id": "cube",
                    "resource_id": "left",
                }
            ],
            "clauses": [
                {
                    "clause_id": "source.constraint",
                    "expectation_id": "source",
                    "expected": False,
                    "evidence_kind": "constraint",
                },
                {
                    "clause_id": "source.pose",
                    "expectation_id": "source",
                    "expectation": "separated",
                },
            ],
        },
        "decision": {"success_mask": [True]},
        "evidence": {
            "source.constraint": {
                "values": [False],
                "valid_mask": [True],
                "env_ids": [0],
                "timestamp": 0.08,
                "observation_revision": 2,
            },
            "source.pose": {
                "object_to_endpoint": torch.eye(4)[None].tolist(),
                "valid_mask": [True],
                "env_ids": [0],
                "timestamp": 0.08,
                "observation_revision": 2,
            },
        },
    }
    runtime = {
        "segments": [
            {
                "name": n["id"],
                "active": [True],
                "successes": [True],
                "start_steps": [i],
                "end_steps": [i + 1],
                "start_step": i,
                "end_step": i + 1,
                "metadata": {
                    "runtime": {
                        "env_ids": [0],
                        "task_state": {
                            "batch_size": 1,
                            "held_objects": [],
                            "coordinated_held_objects": [],
                        },
                        "calls": (
                            [
                                {
                                    "invocation_id": invocation,
                                    "invocation_revision": 0,
                                    "status": "completed",
                                    "effects": [proof],
                                }
                            ]
                            if n == placement
                            else []
                        ),
                    }
                },
            }
            for i, n in enumerate(nodes)
        ]
    }
    child = trimesh.creation.box(extents=[0.08] * 3)
    reference = (
        trimesh.creation.annulus(r_min=0.07, r_max=0.25, height=0.01)
        if hole
        else trimesh.creation.box(extents=[0.5, 0.3, 0.01])
    )
    pose = torch.eye(4)[None]
    pose[:, :3, 3] = torch.tensor([x, 0.0, z])
    reference_pose = torch.eye(4)[None]

    def native(mesh, value):
        return SimpleNamespace(
            get_local_pose=lambda **_kw: value,
            get_vertices=lambda **_kw: torch.tensor(
                np.asarray(mesh.vertices), dtype=torch.float32
            )[None],
            get_triangles=lambda **_kw: torch.tensor(
                np.asarray(mesh.faces), dtype=torch.int64
            )[None],
        )

    objects = {
        "cube": native(child, pose),
        reference_uid: native(reference, reference_pose),
    }
    records = {uid: {"pose": torch.zeros(1, len(nodes) + 1, 7)} for uid in objects}
    for uid, value in (("cube", pose), (reference_uid, reference_pose)):
        records[uid]["pose"][..., :3] = value[:, None, :3, 3]
        records[uid]["pose"][..., 6] = 1.0
    env = SimpleNamespace(
        sim=SimpleNamespace(get_rigid_object=objects.get),
        _traj_buffer={"states": {"rigid_objects": records}},
        _traj_steps=torch.tensor([len(nodes)]),
    )
    registration = SimpleNamespace(
        scene_binding=SimpleNamespace(
            rigid_objects=tuple(
                SimpleNamespace(entity_id=uid, simulation_uid=uid) for uid in objects
            ),
            containers=(
                (
                    SimpleNamespace(
                        entity_id=placement["call"]["inside"], parent_id=reference_uid
                    ),
                )
                if relation == "inside"
                else ()
            ),
            support_surfaces=(),
        )
    )
    deployment = SimpleNamespace(integration=SimpleNamespace(registration=registration))
    save_config(tmp_path / "task_spec.json", plan.task_spec.to_dict())
    path = tmp_path / "task_program/constraints.json"
    path.parent.mkdir(exist_ok=True)
    save_config(path, constraints)
    save_config(tmp_path / "task_program/program.yaml", program)
    return plan, contract, runtime, program, constraints, env, deployment


def _table_region_evidence(tmp_path, monkeypatch, *, contact=True, **kwargs):
    from embodichain.gen_sim.task_engine._task_program import support_evidence
    from embodichain.gen_sim.task_engine._task_program.placement_evidence import (
        measure_planar_placement,
    )
    import numpy as np

    data = _placement_evidence(tmp_path, relation="on", reference_uid="table", **kwargs)
    plan, contract, runtime, program, constraints, env, deployment = data
    placement = next(
        node
        for node in plan.graph["nodes"]
        if node["call"].get("call_id") == "simulation.place_relative"
    )
    preset = f"gen_sim.{placement['id']}.stable"
    cfg = {
        "kind": "supported_region",
        "entity": "cube",
        "reference": "table",
        "duration": 0.08,
        "support_tolerance": 0.01,
    }
    constraints["presets"][preset] = cfg
    deployment.integration.adapter_factory = SimpleNamespace(
        constraints=((preset, SimpleNamespace(**cfg)),)
    )
    terminal = runtime["segments"][-1]
    terminal_index = len(program["program"]["items"]) - 1
    program["program"]["items"][-1]["post"] = [
        {"kind": "wait_stable", "entity": "cube", "preset": preset}
    ]
    prefix = ["program", "items", terminal_index]
    terminal["metadata"]["program_segment_source_path"] = prefix

    def observe(object_pose, reference_pose):
        positions = np.array(
            [[-0.02, -0.02, 0.005], [0.02, -0.02, 0.005], [0.0, 0.02, 0.005]]
        )
        positions[:, :2] += object_pose[0, :2, 3].numpy()
        accepted, row = support_evidence.support_from_contacts(
            np.tile([11, 22], (len(positions), 1)),
            positions,
            np.tile([0.0, 0.0, 1.0], (len(positions), 1)),
            np.full(len(positions), 0.01 if contact else 0.0),
            object_pose[0, :3, 3].numpy(),
            object_actor=22,
            reference_actor=11,
            tolerance=0.003,
        )
        return torch.tensor([accepted]), {
            "scope": "native_contact_support",
            "entity": "cube",
            "reference": "table",
            "env_ids": [0],
            "supported": [accepted],
            "rows": [row],
            "object_pose": object_pose.tolist(),
            "reference_pose": reference_pose.tolist(),
        }

    monkeypatch.setattr(
        support_evidence,
        "NativeSupportObservation",
        lambda *args, **kwargs: SimpleNamespace(observe=observe),
    )
    object_pose = env.sim.get_rigid_object("cube").get_local_pose()
    reference_pose = env.sim.get_rigid_object("table").get_local_pose()
    _, native = observe(object_pose, reference_pose)
    _, region = measure_planar_placement(
        env.sim,
        SimpleNamespace(simulation_uid="cube"),
        SimpleNamespace(simulation_uid="table"),
        object_pose,
        reference_pose,
    )
    terminal["metadata"]["post_policies"] = [
        {
            "kind": "wait_stable",
            "policy_index": 0,
            "source_path": [*prefix, "post", 0],
            "result_mask": [True],
            "result": {
                "kind": "supported_region",
                "preset": preset,
                "accepted_mask": [True],
                "required_seconds": cfg["duration"],
                "stable_seconds": [cfg["duration"]],
                "measurements": {
                    "native_support": native,
                    "landing_region": region,
                    "support_gap": [float(object_pose[0, 2, 3]) - 0.045],
                },
            },
        }
    ]
    for segment in runtime["segments"]:
        for call in segment["metadata"]["runtime"]["calls"]:
            call["plan_attempts"] = [{"planned_at": 0.04}]
            call["events"] = [
                {
                    "kind": "action_completed",
                    "invocation_id": call["invocation_id"],
                    "invocation_revision": call["invocation_revision"],
                    "timestamp": 0.08,
                }
            ]
    contract["program_digest"] = canonical_hash(program)
    contract["constraints_digest"] = canonical_hash(constraints)
    save_config(tmp_path / "task_program/constraints.json", constraints)
    return data


@pytest.mark.parametrize("changes", [{"x": 0.24}, {"hole": True}])
def test_table_contact_and_origin_bounds_cannot_replace_full_region(
    tmp_path, monkeypatch, changes
):
    data = _table_region_evidence(tmp_path, monkeypatch, **changes)
    result = _observe_placement(tmp_path, data)
    assert result["certificate_ready"] is False
    assert result["goal"][0]["status"] == "failed"
    assert result["milestones"][0]["status"] == "failed"


@pytest.mark.parametrize(
    "damage", ["no_contact", "no_release", "future", "before_plan"]
)
def test_table_region_requires_exact_contact_release_and_time_window(
    tmp_path, monkeypatch, damage
):
    data = _table_region_evidence(tmp_path, monkeypatch, contact=damage != "no_contact")
    call = next(
        segment["metadata"]["runtime"]["calls"][0]
        for segment in data[2]["segments"]
        if segment["metadata"]["runtime"]["calls"]
    )
    if damage == "no_release":
        call["effects"] = []
    elif damage in {"future", "before_plan"}:
        timestamp = 0.12 if damage == "future" else 0.0
        proof = call["effects"][0]
        proof["timestamp"] = timestamp
        for evidence in proof["evidence"].values():
            evidence["timestamp"] = timestamp
    assert _observe_placement(tmp_path, data)["certificate_ready"] is False


def test_table_region_measurements_retain_native_and_complete_geometry(
    tmp_path, monkeypatch
):
    data = _table_region_evidence(tmp_path, monkeypatch)
    result = _observe_placement(tmp_path, data)
    assert result["certificate_ready"] is True
    record = next(iter(result["final_measurements"].values()))["members"][0]
    metrics = record["measurements"]
    assert metrics["scope"] == "native_supported_landing_region"
    assert metrics["landing_region"]["full_footprint_inside"] == [True]
    assert metrics["native_support"]["supported"] == [True]


@pytest.mark.parametrize("target", ["final", "milestone"])
@pytest.mark.parametrize(
    "damage",
    [
        "missing_region",
        "legacy_scope",
        "footprint",
        "gap",
        "no_contact",
        "no_load",
        "node",
        "reference",
        "poses",
        "release",
        "window",
    ],
)
def test_table_region_matcher_rejects_incomplete_or_foreign_physical_records(
    tmp_path, monkeypatch, target, damage
):
    from embodichain.gen_sim.task_engine._task_program.task_spec_runtime import (
        capture_final_goals,
        capture_placement_milestones,
    )

    data = _table_region_evidence(tmp_path, monkeypatch)
    plan, contract, runtime, program, constraints, env, deployment = data
    final = capture_final_goals(
        env, deployment, tmp_path, plan.graph, contract, 1, runtime=runtime
    )
    milestones = capture_placement_milestones(
        env, deployment, plan.graph, contract, runtime, 1, spec=plan.task_spec
    )
    record = (
        next(iter(final.values()))["members"][0]
        if target == "final"
        else next(iter(milestones.values()))
    )
    metrics = record["measurements"]
    if damage == "missing_region":
        metrics.pop("landing_region")
    elif damage == "legacy_scope":
        record["measurements"] = {
            **metrics["native_support"],
            "reference_uid": "table",
            "relation": "on",
            "released": [True],
        }
    elif damage == "footprint":
        metrics["landing_region"]["full_footprint_inside"] = [False]
    elif damage == "gap":
        metrics["landing_region"]["support_gap_m"] = [0.12]
    elif damage == "no_contact":
        metrics["native_support"]["supported"] = [False]
    elif damage == "no_load":
        metrics["native_support"]["rows"][0]["upward_impulse"] = 0.0
    elif damage == "node":
        metrics["placement_node_id"] = "other"
    elif damage == "reference":
        metrics["native_support"]["reference"] = "other"
    elif damage == "poses":
        metrics["native_support"]["object_pose"][0][0][3] += 0.02
    elif damage == "release":
        metrics["released"] = [False]
    else:
        metrics["release_window"]["timestamp"] = 0.04
    result = evaluate_runtime_evidence(
        plan.task_spec,
        contract,
        plan.graph,
        runtime,
        final,
        num_envs=1,
        run_id="invalid-region",
        program=program,
        constraints=constraints,
        placement_milestones=milestones,
    )["environments"][0]
    assert result["certificate_ready"] is False


@pytest.mark.parametrize("damage", ["frames", "coherent_pose", "policy", "duration"])
def test_table_region_milestone_stays_at_its_exact_recorded_boundary(
    tmp_path, monkeypatch, damage
):
    data = _table_region_evidence(tmp_path, monkeypatch)
    terminal = data[2]["segments"][-1]
    if damage == "frames":
        terminal["end_steps"] = [0]
    elif damage == "policy":
        terminal["metadata"]["post_policies"][0]["source_path"] = [
            "program",
            "items",
            0,
            "post",
            0,
        ]
    elif damage == "duration":
        terminal["metadata"]["post_policies"][0]["result"]["stable_seconds"] = [0.0]
    else:
        trace = terminal["metadata"]["post_policies"][0]["result"]["measurements"]
        for key in ("native_support", "landing_region"):
            trace[key]["object_pose"][0][0][3] += 0.02
    result = _observe_placement(tmp_path, data)
    assert result["certificate_ready"] is False
    assert result["goal"][0]["status"] == "satisfied"
    assert result["milestones"][0]["status"] == "unavailable"


@pytest.mark.parametrize("legacy_kind", ["upright", "supported_placement"])
def test_inherited_legacy_table_presets_do_not_acquire_region_semantics(
    tmp_path, monkeypatch, legacy_kind
):
    data = _table_region_evidence(tmp_path, monkeypatch)
    constraints = data[4]
    preset, cfg = next(iter(constraints["presets"].items()))
    cfg["kind"] = legacy_kind
    data[6].integration.adapter_factory.constraints = (
        (preset, SimpleNamespace(**cfg)),
    )
    data[2]["segments"][-1]["metadata"]["post_policies"][0]["result"][
        "kind"
    ] = legacy_kind
    data[1]["constraints_digest"] = canonical_hash(constraints)
    save_config(tmp_path / "task_program/constraints.json", constraints)
    result = _observe_placement(tmp_path, data)
    assert result["certificate_ready"] is True
    metrics = next(iter(result["final_measurements"].values()))["members"][0][
        "measurements"
    ]
    assert metrics["scope"] == "native_contact_support"


def _observe_placement(tmp_path, data):
    from embodichain.gen_sim.task_engine._task_program.task_spec_runtime import (
        capture_final_goals,
        capture_placement_milestones,
    )

    plan, contract, runtime, program, constraints, env, deployment = data
    final = capture_final_goals(
        env, deployment, tmp_path, plan.graph, contract, 1, runtime=runtime
    )
    milestones = capture_placement_milestones(
        env, deployment, plan.graph, contract, runtime, 1, spec=plan.task_spec
    )
    return evaluate_runtime_evidence(
        plan.task_spec,
        contract,
        plan.graph,
        runtime,
        final,
        num_envs=1,
        run_id="placement-test",
        program=program,
        constraints=constraints,
        placement_milestones=milestones,
    )["environments"][0]


def test_empty_legacy_presets_do_not_block_measured_e1_final_and_milestone(
    tmp_path,
) -> None:
    result = _observe_placement(tmp_path, _placement_evidence(tmp_path))
    assert result["certificate_ready"] is True
    assert result["milestones"][0]["status"] == "satisfied"


@pytest.mark.parametrize("gap", [0.010443, 0.04])
def test_supported_placement_accepts_clearance_below_fifty_mm(tmp_path, gap) -> None:
    result = _observe_placement(tmp_path, _placement_evidence(tmp_path, z=0.045 + gap))
    assert result["certificate_ready"] is True


@pytest.mark.parametrize("gap", [0.051, -0.02])
def test_support_clearance_relaxation_does_not_allow_high_or_penetrated_objects(
    tmp_path, gap
) -> None:
    result = _observe_placement(tmp_path, _placement_evidence(tmp_path, z=0.045 + gap))
    assert result["goal"][0]["status"] == "failed"


@pytest.mark.parametrize(
    "changes", [{"x": 0.24}, {"z": 0.2}, {"z": 0.0}, {"hole": True}]
)
def test_full_footprint_height_and_actual_floor_holes_reject_bad_placement(
    tmp_path, changes
) -> None:
    result = _observe_placement(tmp_path, _placement_evidence(tmp_path, **changes))
    assert result["certificate_ready"] is False
    assert result["goal"][0]["status"] == "failed"
    assert result["milestones"][0]["status"] == "failed"


@pytest.mark.parametrize(
    "damage",
    [
        "missing",
        "wrong_object",
        "wrong_env",
        "stale",
        "guard_only",
        "constraint_present",
    ],
)
def test_release_receipts_must_be_measured_current_and_exactly_bound(
    tmp_path, damage
) -> None:
    data = _placement_evidence(tmp_path)
    call = next(
        s["metadata"]["runtime"]["calls"][0]
        for s in data[2]["segments"]
        if s["metadata"]["runtime"]["calls"]
    )
    proof = call["effects"][0]
    if damage == "missing":
        call["effects"] = []
    elif damage == "wrong_object":
        proof["effect_spec"]["state_expectations"][0]["object_id"] = "other"
    elif damage == "wrong_env":
        proof["effect_spec"]["env_ids"] = [7]
    elif damage == "stale":
        proof["evidence"]["source.constraint"]["timestamp"] = 0.04
    elif damage == "guard_only":
        proof["boundary"]["kind"] = "phase_gate"
    else:
        proof["evidence"]["source.constraint"]["values"] = [True]
    assert _observe_placement(tmp_path, data)["certificate_ready"] is False


def test_past_placement_milestone_does_not_certify_changed_final_native_pose(
    tmp_path,
) -> None:
    data = _placement_evidence(tmp_path)
    native = data[5].sim.get_rigid_object("cube")
    moved = torch.eye(4)[None]
    moved[:, :3, 3] = torch.tensor([0.5, 0.0, 0.045])
    native.get_local_pose = lambda **_kwargs: moved
    result = _observe_placement(tmp_path, data)
    assert result["goal"][0]["status"] == "failed"
    assert result["milestones"][0]["status"] == "satisfied"


def test_missing_boundary_record_does_not_borrow_final_pose_for_milestone(
    tmp_path,
) -> None:
    data = _placement_evidence(tmp_path)
    data[5]._traj_steps = torch.tensor([0])
    result = _observe_placement(tmp_path, data)
    assert result["goal"][0]["status"] == "satisfied"
    assert result["milestones"][0]["status"] == "unavailable"


def test_scaled_native_geometry_is_batch_local_and_world_yaw_invariant(
    tmp_path,
) -> None:
    from embodichain.gen_sim.task_engine._task_program.placement_evidence import (
        measure_planar_placement,
    )

    data = _placement_evidence(tmp_path)
    sim = data[5].sim
    for uid in ("cube", "region"):
        native = sim.get_rigid_object(uid)
        vertices, faces = native.get_vertices(), native.get_triangles()
        native.get_vertices = lambda vertices=vertices, **_kwargs: vertices.expand(
            2, -1, -1
        )
        native.get_triangles = lambda faces=faces, **_kwargs: faces.expand(2, -1, -1)
    reference = torch.eye(4, dtype=torch.float64)[None].repeat(2, 1, 1)
    reference[:, :3, :3] = reference.new_tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    reference[:, :3, 3] = reference.new_tensor([1.0, 2.0, 0.5])
    relative = torch.eye(4, dtype=torch.float64)[None].repeat(2, 1, 1)
    relative[:, 2, 3] = 0.045
    relative[1, 0, 3] = 0.24
    accepted, _ = measure_planar_placement(
        sim,
        SimpleNamespace(simulation_uid="cube"),
        SimpleNamespace(simulation_uid="region"),
        reference @ relative,
        reference,
    )
    assert accepted == [True, False]


@pytest.mark.parametrize("damage", ["uid", "reference", "frame", "released", "summary"])
def test_milestone_measurement_cannot_be_misattributed_or_summary_only(
    tmp_path, damage
) -> None:
    from embodichain.gen_sim.task_engine._task_program.task_spec_runtime import (
        capture_final_goals,
        capture_placement_milestones,
    )

    data = _placement_evidence(tmp_path)
    plan, contract, runtime, program, constraints, env, deployment = data
    final = capture_final_goals(
        env, deployment, tmp_path, plan.graph, contract, 1, runtime=runtime
    )
    milestones = capture_placement_milestones(
        env, deployment, plan.graph, contract, runtime, 1, spec=plan.task_spec
    )
    record = next(iter(milestones.values()))
    if damage == "uid":
        record["uid"] = "other"
    elif damage == "summary":
        record["measurements"] = {"success": True}
    else:
        key = {
            "reference": "reference_uid",
            "frame": "boundary_frames",
            "released": "released",
        }[damage]
        record["measurements"][key] = {
            "reference": "other",
            "frame": [0],
            "released": [False],
        }[damage]
    result = evaluate_runtime_evidence(
        plan.task_spec,
        contract,
        plan.graph,
        runtime,
        final,
        num_envs=1,
        run_id="invalid-proof",
        program=program,
        constraints=constraints,
        placement_milestones=milestones,
    )["environments"][0]
    assert result["certificate_ready"] is False


def test_registered_e1_on_without_native_contacts_cannot_certify(
    tmp_path,
) -> None:
    assert (
        _observe_placement(tmp_path, _placement_evidence(tmp_path, relation="on"))[
            "certificate_ready"
        ]
        is False
    )


def test_later_holding_same_object_contradicts_final_release(tmp_path) -> None:
    data = _placement_evidence(tmp_path)
    data[2]["segments"][-1]["metadata"]["runtime"]["task_state"]["held_objects"] = [
        {"object_id": "cube", "active_mask": [True]}
    ]
    result = _observe_placement(tmp_path, data)
    assert result["goal"][0]["status"] == "failed"


def test_superseded_placement_is_measured_at_its_own_milestone_reference(
    tmp_path,
) -> None:
    from embodichain.gen_sim.task_engine.semantic_planner import SemanticTaskPlanner
    from .test_task_spec_generation import candidate, selector, step

    data = list(_placement_evidence(tmp_path))
    source = candidate(
        step(
            "E1",
            "cube",
            target=selector("first"),
            relation="inside",
            required_arm="left_arm",
        ),
        step(
            "E1",
            "cube",
            target=selector("second"),
            relation="inside",
            required_arm="left_arm",
        ),
    )
    bindings = {
        "schema_version": "action_engine_role_bindings_v1",
        "task_id": source["draft"]["task_id"],
        "candidate_id": source["candidate_id"],
        "role_bindings": {},
        "reference_bindings": {
            "step_01.object": ["cube"],
            "step_01.target": ["region"],
            "step_02.object": ["cube"],
            "step_02.target": ["second"],
        },
    }
    plan = SemanticTaskPlanner().plan_spec(
        source,
        bindings,
        [
            {"runtime_uid": uid, "init_pos": [0.0, 0.0, 0.7], "init_rot": [0.0] * 3}
            for uid in ("cube", "region", "second")
        ],
    )
    original = deepcopy(data[2]["segments"])
    nodes = plan.graph["nodes"]
    segments = []
    for i, node in enumerate(nodes):
        segment = deepcopy(original[i % 3])
        segment.update(
            name=node["id"],
            start_step=i,
            end_step=i + 1,
            start_steps=[i],
            end_steps=[i + 1],
        )
        segments.append(segment)
    program = {"program": {"items": [{"name": n["id"], "post": []} for n in nodes]}}
    data[0], data[2], data[3] = plan, {"segments": segments}, program
    data[1] = {
        "coverage": _coverage(plan),
        "task_spec_digest": canonical_hash(plan.task_spec.to_dict()),
        "graph_digest": canonical_hash(plan.graph),
        "program_digest": canonical_hash(program),
        "constraints_digest": canonical_hash(data[4]),
    }
    env, registration = data[5], data[6].integration.registration
    first = env.sim.get_rigid_object("region")
    second_pose = torch.eye(4)[None]
    second_pose[:, 0, 3] = 1.0
    second = SimpleNamespace(
        get_local_pose=lambda **_kw: second_pose,
        get_vertices=first.get_vertices,
        get_triangles=first.get_triangles,
    )
    objects = {uid: env.sim.get_rigid_object(uid) for uid in ("cube", "region")}
    objects["second"] = second
    env.sim.get_rigid_object = objects.get
    final_pose = torch.eye(4)[None]
    final_pose[:, :3, 3] = torch.tensor([1.0, 0.0, 0.045])
    objects["cube"].get_local_pose = lambda **_kw: final_pose
    records = {uid: {"pose": torch.zeros(1, 7, 7)} for uid in objects}
    for value in records.values():
        value["pose"][..., 6] = 1.0
    records["cube"]["pose"][..., 2] = 0.045
    records["cube"]["pose"][:, 4:, 0] = 1.0
    records["second"]["pose"][..., 0] = 1.0
    env._traj_buffer = {"states": {"rigid_objects": records}}
    env._traj_steps = torch.tensor([6])
    registration.scene_binding.rigid_objects += (
        SimpleNamespace(entity_id="second", simulation_uid="second"),
    )
    registration.scene_binding.containers = tuple(
        SimpleNamespace(
            entity_id=n["call"]["inside"], parent_id="region" if i == 1 else "second"
        )
        for i, n in enumerate(nodes)
        if n["call"]["kind"] == "place"
    )
    save_config(tmp_path / "task_spec.json", plan.task_spec.to_dict())
    result = _observe_placement(tmp_path, data)
    assert result["certificate_ready"] is True
    assert [v["status"] for v in result["milestones"]] == ["satisfied", "satisfied"]
