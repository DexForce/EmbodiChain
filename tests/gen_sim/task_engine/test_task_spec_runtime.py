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

from dataclasses import replace
import pytest
from types import SimpleNamespace
import torch
from embodichain.gen_sim.task_engine._task_program.task_spec_runtime import (
    evaluate_runtime_evidence,
)
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


def test_registered_e1_on_without_native_contacts_cannot_certify(
    tmp_path,
) -> None:
    assert (
        _observe_placement(tmp_path, _placement_evidence(tmp_path, relation="on"))[
            "certificate_ready"
        ]
        is False
    )
