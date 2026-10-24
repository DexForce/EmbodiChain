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
import json
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
    num_envs=1,
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
            "env_ids": list(range(num_envs)),
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
        "decision": {"success_mask": [True] * num_envs},
        "evidence": {
            "source.constraint": {
                "values": [False] * num_envs,
                "valid_mask": [True] * num_envs,
                "env_ids": list(range(num_envs)),
                "timestamp": 0.08,
                "observation_revision": 2,
            },
            "source.pose": {
                "object_to_endpoint": torch.eye(4).repeat(num_envs, 1, 1).tolist(),
                "valid_mask": [True] * num_envs,
                "env_ids": list(range(num_envs)),
                "timestamp": 0.08,
                "observation_revision": 2,
            },
        },
    }
    runtime = {
        "segments": [
            {
                "name": n["id"],
                "active": [True] * num_envs,
                "successes": [True] * num_envs,
                "start_steps": [i] * num_envs,
                "end_steps": [i + 1] * num_envs,
                "start_step": i,
                "end_step": i + 1,
                "metadata": {
                    "runtime": {
                        "env_ids": list(range(num_envs)),
                        "task_state": {
                            "batch_size": num_envs,
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
    pose = torch.eye(4).repeat(num_envs, 1, 1)
    pose[:, :3, 3] = torch.tensor([x, 0.0, z])
    reference_pose = torch.eye(4).repeat(num_envs, 1, 1)

    def native(mesh, value):
        return SimpleNamespace(
            get_local_pose=lambda **_kw: value,
            get_vertices=lambda **_kw: torch.tensor(
                np.asarray(mesh.vertices), dtype=torch.float32
            )[None].repeat(num_envs, 1, 1),
            get_triangles=lambda **_kw: torch.tensor(
                np.asarray(mesh.faces), dtype=torch.int64
            )[None].repeat(num_envs, 1, 1),
        )

    objects = {
        "cube": native(child, pose),
        reference_uid: native(reference, reference_pose),
    }
    records = {
        uid: {"pose": torch.zeros(num_envs, len(nodes) + 1, 7)} for uid in objects
    }
    for uid, value in (("cube", pose), (reference_uid, reference_pose)):
        records[uid]["pose"][..., :3] = value[:, None, :3, 3]
        records[uid]["pose"][..., 6] = 1.0
    env = SimpleNamespace(
        num_envs=num_envs,
        step_dt=0.04,
        sim=SimpleNamespace(get_rigid_object=objects.get),
        _traj_buffer={"states": {"rigid_objects": records}},
        _traj_steps=torch.tensor([len(nodes)] * num_envs),
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


def test_triangle_silhouette_does_not_invent_material_across_a_concavity():
    import numpy as np
    import trimesh
    from embodichain.gen_sim.task_engine._task_program.placement_evidence import (
        measure_planar_placement,
    )

    def u_mesh(x, y):
        vertices = np.array([(a, b, 0.0) for b in y for a in x])
        faces = []
        for row, column in ((0, 0), (0, 1), (0, 2), (1, 0), (1, 2)):
            first = row * 4 + column
            faces.extend(((first, first + 1, first + 5), (first, first + 5, first + 4)))
        return trimesh.Trimesh(vertices, faces, process=False)

    # Both are U-shaped; the smaller U fits, but its convex hull crosses the notch.
    meshes = {
        "floor": u_mesh((-0.3, -0.1, 0.1, 0.3), (-0.3, -0.1, 0.3)),
        "object": u_mesh((-0.25, -0.13, 0.13, 0.25), (-0.25, -0.13, 0.25)),
    }
    natives = {
        name: SimpleNamespace(
            get_vertices=lambda mesh=mesh, **_kw: torch.tensor(
                mesh.vertices.copy()
            ).float()[None],
            get_triangles=lambda mesh=mesh, **_kw: torch.tensor(
                mesh.faces.copy()
            ).long()[None],
        )
        for name, mesh in meshes.items()
    }
    sim = SimpleNamespace(get_rigid_object=natives.get)
    obj, ref = (SimpleNamespace(simulation_uid=name) for name in ("object", "floor"))
    pose = torch.eye(4)[None]
    old, _ = measure_planar_placement(sim, obj, ref, pose, pose)
    current, measured = measure_planar_placement(
        sim, obj, ref, pose, pose, exact_silhouette=True
    )
    assert old == [False]
    assert current == [True]
    assert measured["footprint_kind"] == "native_triangle_projection/v1"


def test_empty_legacy_presets_do_not_block_measured_e1_final_and_milestone(
    tmp_path,
) -> None:
    result = _observe_placement(tmp_path, _placement_evidence(tmp_path))
    assert result["certificate_ready"] is True
    assert result["milestones"][0]["status"] == "satisfied"


@pytest.mark.parametrize(
    "x,z,expected",
    [
        (0.24, 0.045, "unavailable"),
        (0.0, -0.1, "failed"),
        (0.0, 0.5, "failed"),
        (1.0, 0.045, "failed"),
    ],
)
def test_unqualified_cavity_does_not_hide_penetration_hovering_or_clear_miss(
    tmp_path, x, z, expected
):
    import numpy as np
    import trimesh

    data = _placement_evidence(tmp_path, x=x, z=z)
    native = data[-2].sim.get_rigid_object("region")
    floor = trimesh.creation.box(extents=[0.5, 0.3, 0.01])
    wall = trimesh.creation.box(extents=[0.01, 0.3, 0.06])
    wall.apply_translation([0.245, 0.0, 0.035])
    mesh = trimesh.util.concatenate([floor, wall])
    native.get_vertices = lambda **_kw: torch.tensor(np.asarray(mesh.vertices)).float()[
        None
    ]
    native.get_triangles = lambda **_kw: torch.tensor(np.asarray(mesh.faces)).long()[
        None
    ]
    assert _observe_placement(tmp_path, data)["status"] == expected


@pytest.mark.parametrize("x, expected", [(0.24, None), (1.0, False), (0.0, True)])
def test_cavity_uncertainty_is_row_local_for_final_and_milestone(tmp_path, x, expected):
    import trimesh
    from embodichain.gen_sim.task_engine._task_program.task_spec_runtime import (
        capture_final_goals,
        capture_placement_milestones,
    )

    plan, contract, runtime, program, constraints, env, deployment = (
        _placement_evidence(tmp_path, num_envs=2)
    )
    floor = trimesh.creation.box(extents=[0.5, 0.3, 0.01])
    wall = trimesh.creation.box(extents=[0.01, 0.3, 0.06])
    wall.apply_translation([0.245, 0.0, 0.035])
    mesh = trimesh.util.concatenate([floor, wall])
    reference = env.sim.get_rigid_object("region")
    reference.get_vertices = (
        lambda **_kw: torch.tensor(mesh.vertices.copy()).float()[None].repeat(2, 1, 1)
    )
    reference.get_triangles = (
        lambda **_kw: torch.tensor(mesh.faces.copy()).long()[None].repeat(2, 1, 1)
    )
    env.sim.get_rigid_object("cube").get_local_pose()[1, 0, 3] = x
    env._traj_buffer["states"]["rigid_objects"]["cube"]["pose"][1, :, 0] = x
    final = capture_final_goals(
        env, deployment, tmp_path, plan.graph, contract, 2, runtime=runtime
    )
    milestone = capture_placement_milestones(
        env, deployment, plan.graph, contract, runtime, 2, spec=plan.task_spec
    )
    assert final[plan.task_spec.goal[0].id]["values"] == [True, expected]
    rows = evaluate_runtime_evidence(
        plan.task_spec,
        contract,
        plan.graph,
        runtime,
        final,
        num_envs=2,
        run_id="batch-placement-test",
        program=program,
        constraints=constraints,
        placement_milestones=milestone,
    )["environments"]
    statuses = [
        "satisfied",
        "unavailable" if expected is None else "satisfied" if expected else "failed",
    ]
    assert [row["goal"][0]["status"] for row in rows] == statuses
    assert [row["milestones"][0]["status"] for row in rows] == statuses
    assert [row["certificate_ready"] for row in rows] == [True, expected is True]
    if expected is None:
        from copy import deepcopy
        from embodichain.gen_sim.task_engine._task_program.task_spec_runtime import (
            _placement_measurement_matches,
        )

        for available in ([True, True], [True]):
            tampered = deepcopy(final[plan.task_spec.goal[0].id]["members"][0])
            tampered["available"] = available
            assert not _placement_measurement_matches(
                tampered,
                plan.task_spec.goal[0],
                contract["coverage"]["roles"],
                2,
                graph=plan.graph,
                constraints=constraints,
                segments={s["name"]: s for s in runtime["segments"]},
                program=program,
            )


def _rigid_side_evidence(tmp_path, *, x=0.35):
    from embodichain.gen_sim.task_engine._task_program.services import (
        _RelativePlaceRoute,
    )
    from embodichain.gen_sim.task_engine._task_program.stability import (
        StabilityConstraint,
        TaskStabilityPort,
    )
    from embodichain.lab.task_program.compiler.program import CompiledPostPolicy
    from embodichain.lab.task_program.semantics.scene import SceneObjectRef
    from embodichain.lab.task_program.language.schema import WaitStablePostCfg

    data = _placement_evidence(
        tmp_path, relation="behind", reference_uid="reference", x=x
    )
    plan, contract, runtime, program, constraints, env, deployment = data
    node = next(
        n
        for n in plan.graph["nodes"]
        if n["call"].get("call_id") == "simulation.place_relative"
    )
    registration = deployment.integration.registration
    scene = registration.scene_binding
    scene.articulations, scene.links = (), ()
    scene.rigid_objects += (SimpleNamespace(entity_id="table", simulation_uid="table"),)
    original = env.sim.get_rigid_object
    reference = original("reference")
    # The tabletop must contain the complete object, unlike the side reference.
    import trimesh

    table_mesh = trimesh.creation.box(extents=[2.0, 2.0, 0.01])
    table = SimpleNamespace(
        get_local_pose=reference.get_local_pose,
        get_vertices=lambda **_kw: torch.tensor(table_mesh.vertices.copy()).float()[
            None
        ],
        get_triangles=lambda **_kw: torch.tensor(table_mesh.faces.copy()).long()[None],
    )
    env.sim.get_rigid_object = lambda uid: table if uid == "table" else original(uid)
    env._traj_buffer["states"]["rigid_objects"]["table"] = env._traj_buffer["states"][
        "rigid_objects"
    ]["reference"]
    from embodichain.gen_sim.task_engine.task_program_bundle import (
        _LATERAL_RELEASE_CLEARANCE,
        _PLACEMENT_CLEARANCE,
    )

    settled = (x, 0.0, 0.02)
    route = _RelativePlaceRoute(
        "cube",
        "reference",
        "behind",
        (
            *settled[:2],
            settled[2] + (_LATERAL_RELEASE_CLEARANCE - _PLACEMENT_CLEARANCE),
        ),
    )
    registration.registered_semantic_lowerer_factories = (
        SimpleNamespace(call_id="simulation.place_relative", routes=(route,)),
    )
    preset = f"gen_sim.{node['id']}.stable"
    cfg = StabilityConstraint(
        kind="placement",
        entity="cube",
        reference="reference",
        displacement=settled,
        duration=0.1,
    )
    deployment.integration.adapter_factory = SimpleNamespace(
        constraints=((preset, cfg),)
    )
    terminal = runtime["segments"][-1]
    prefix = ["program", "items", len(program["program"]["items"]) - 1]
    policy = CompiledPostPolicy(
        WaitStablePostCfg(entity="cube", preset=preset),
        SceneObjectRef("cube"),
        (*prefix, "post", 0),
    )
    segment = SimpleNamespace(post_policies=(policy,))
    port = TaskStabilityPort(
        SimpleNamespace(),
        env.sim,
        SimpleNamespace(get_qpos=lambda **_kw: torch.zeros(1, 2)),
        scene,
        {preset: cfg},
        step_dt=0.1,
    )
    list(port.actions(policy, segment=segment, active_mask=torch.tensor([True])))
    terminal["metadata"].update(
        program_segment_source_path=prefix,
        post_policies=[
            {
                "kind": "wait_stable",
                "policy_index": 0,
                "source_path": [*prefix, "post", 0],
                "result_mask": port.post_policy_result(
                    policy, segment=segment
                ).tolist(),
                "result": port.post_policy_metadata(policy, segment=segment),
            }
        ],
    )
    return data


def test_rigid_side_relation_certifies_measured_release_and_table_landing(tmp_path):
    result = _observe_placement(tmp_path, _rigid_side_evidence(tmp_path))
    assert result["certificate_ready"] is True
    assert result["milestones"][0]["status"] == "satisfied"


@pytest.mark.parametrize("damage", [None, "x", "z", "unauthorized_task"])
def test_rigid_side_release_route_matches_its_distinct_json_settled_target(
    tmp_path, damage
):
    from dataclasses import asdict, replace
    from embodichain.gen_sim.task_engine._task_program.stability import (
        StabilityConstraint,
    )

    data = _rigid_side_evidence(tmp_path)
    factory = data[-1].integration.adapter_factory
    preset, cfg = factory.constraints[0]
    cfg = StabilityConstraint.decode(json.loads(json.dumps(asdict(cfg))))
    if damage in {"x", "z"}:
        displacement = list(cfg.displacement)
        displacement[0 if damage == "x" else 2] += 0.001
        cfg = replace(cfg, displacement=tuple(displacement))
    elif damage == "unauthorized_task":
        for node in data[0].graph["nodes"]:
            node["task_type"] = "E2"
    factory.constraints = ((preset, cfg),)
    if damage == "unauthorized_task":
        with pytest.raises(ValueError, match="identity mismatch"):
            _observe_placement(tmp_path, data)
        return
    result = _observe_placement(tmp_path, data)
    assert result["certificate_ready"] is (damage is None)


@pytest.mark.parametrize("damage", ["overlap", "missing_release", "stale_reference"])
def test_rigid_side_relation_never_uses_center_or_unmeasured_success(tmp_path, damage):
    data = _rigid_side_evidence(tmp_path, x=0.28 if damage == "overlap" else 0.35)
    if damage == "missing_release":
        for segment in data[2]["segments"]:
            for call in segment["metadata"]["runtime"]["calls"]:
                call["effects"] = []
    elif damage == "stale_reference":
        pose = data[2]["segments"][-1]["metadata"]["post_policies"][0]["result"][
            "measurements"
        ]["reference_pose"]
        pose[0][0][3] += 0.1
    result = _observe_placement(tmp_path, data)
    assert result["certificate_ready"] is False
    assert result["milestones"][0]["status"] != "satisfied"


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


@pytest.mark.parametrize("x, expected", [(0.0, True), (0.23, False)])
def test_supported_region_keeps_footprint_with_an_e5_contact_timeline(
    tmp_path, monkeypatch, x, expected
):
    from dataclasses import asdict
    from embodichain.gen_sim.task_engine._task_program import support_evidence
    from embodichain.gen_sim.task_engine._task_program.stability import (
        StabilityConstraint,
    )
    from embodichain.gen_sim.task_engine._task_program.stable_support_evidence import (
        capture_stable_support,
    )
    from embodichain.gen_sim.task_engine._task_program.task_spec_runtime import (
        _placement_measurement_matches,
        _placement_record,
    )

    plan, contract, runtime, program, _, env, deployment = _placement_evidence(
        tmp_path, x=x, relation="on", reference_uid="table"
    )
    node = next(
        n
        for n in plan.graph["nodes"]
        if n["call"].get("call_id") == "simulation.place_relative"
    )
    cfg = StabilityConstraint(
        entity="cube", reference="table", kind="supported_region", duration=0.04
    )
    preset = f"gen_sim.{node['id']}.stable"
    deployment.integration.adapter_factory = SimpleNamespace(
        constraints=((preset, cfg),)
    )
    end = int(env._traj_steps[0])
    raw = {"pairs": [[2, 1]], "normals": [[0, 0, 1]], "impulses": [0.01]}
    timeline = {
        "errors": [],
        "step_dt": 0.04,
        "object_actors": {"cube": [[1]], "table": [[2]]},
        "object_body_types": {"cube": "dynamic", "table": "static"},
        "samples": [
            {
                "step": end,
                "time": end * 0.04,
                "dropped_contacts": 0,
                "rows": [raw],
                "substeps": [
                    {"dt": 0.04, "offset": 0.04, "dropped_contacts": 0, "rows": [raw]}
                ],
            }
        ],
    }
    env._gen_sim_acceptance_contacts = SimpleNamespace(snapshot=lambda: timeline)
    bindings = {
        v.entity_id: v
        for v in deployment.integration.registration.scene_binding.rigid_objects
    }
    obj = env.sim.get_rigid_object("cube").get_local_pose()
    ref = env.sim.get_rigid_object("table").get_local_pose()
    native = {
        "scope": "native_contact_support",
        "entity": "cube",
        "reference": "table",
        "env_ids": [0],
        "supported": [True],
        "object_pose": obj.tolist(),
        "reference_pose": ref.tolist(),
        "rows": [
            {
                "com_patch_distance_m": 0.0,
                "com_tolerance_m": 0.001,
                "upward_impulse": 0.01,
                "contacts_world": [[x, 0.0, 0.005]],
                "center_of_mass_world": [x, 0.0, 0.045],
            }
        ],
    }
    monkeypatch.setattr(
        support_evidence,
        "NativeSupportObservation",
        lambda *args, **kwargs: SimpleNamespace(
            observe=lambda *poses: (torch.tensor([True]), native)
        ),
    )
    predicate = plan.task_spec.goal[0]
    member = next(iter(contract["coverage"]["lineage"].values()))[0]
    segments = {s["name"]: s for s in runtime["segments"]}
    boundary = runtime["segments"][-1]
    call = segments[node["id"]]["metadata"]["runtime"]["calls"][0]
    call["plan_attempts"] = [{"planned_at": 0.04}]
    call["events"] = [
        {
            "kind": "action_completed",
            "timestamp": 0.08,
            "invocation_id": call["invocation_id"],
            "invocation_revision": call["invocation_revision"],
        }
    ]
    record = _placement_record(
        env, deployment, plan.graph, contract, predicate, member, segments, boundary, 1
    )
    assert record["accepted"] == [expected]
    assert record["measurements"]["scope"] == "native_supported_landing_region"
    kwargs = dict(
        graph=plan.graph,
        constraints={"presets": {preset: asdict(cfg)}},
        segments=segments,
        program=program,
        contact_timeline=timeline,
    )
    assert _placement_measurement_matches(
        record, predicate, contract["coverage"]["roles"], 1, **kwargs
    )

    # A genuine native load is still insufficient for the stricter region contract.
    supported, metrics = capture_stable_support(
        env, bindings, "cube", "table", end, cfg, 0.08
    )
    metrics.update(
        reference_uid="table",
        relation="on",
        released=[True],
        placement_node_id=node["id"],
        boundary_frames=[end],
        observation_kind="final",
    )
    support_only = {**record, "accepted": supported, "measurements": metrics}
    assert not _placement_measurement_matches(
        support_only, predicate, contract["coverage"]["roles"], 1, **kwargs
    )
