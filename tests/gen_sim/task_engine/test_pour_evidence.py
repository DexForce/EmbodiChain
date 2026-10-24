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

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import torch

from embodichain.gen_sim.task_engine._task_program.pour_evidence import (
    capture_pour_motions,
)
from embodichain.gen_sim.task_engine._task_program.task_spec_runtime import (
    capture_final_goals,
    evaluate_runtime_evidence,
)
from embodichain.gen_sim.task_engine.contracts import canonical_hash
from embodichain.gen_sim.task_engine.semantic_planner import SemanticTaskPlanner
from embodichain.utils.utility import save_config
from .test_semantic_graph_validation import _coverage
from .test_task_spec_generation import candidate, selector, step
from .test_task_spec_runtime import _placement_evidence


def _case(
    tmp_path, *, tilt=60.0, restore=0.0, offset=0.0, rotation_axis="y", mode="motion"
):
    old = _placement_evidence(tmp_path)
    env, deployment = old[-2:]
    proof = deepcopy(
        next(
            c["effects"][0]
            for s in old[2]["segments"]
            for c in s["metadata"]["runtime"]["calls"]
        )
    )
    source = candidate(
        step(
            "E3",
            "cube",
            target=selector("region"),
            relation="above",
            required_arm="left_arm",
            pour_goal_mode=mode,
        )
    )
    bindings = {
        "schema_version": "action_engine_role_bindings_v1",
        "task_id": source["draft"]["task_id"],
        "candidate_id": source["candidate_id"],
        "reference_bindings": {
            "step_01.object": ["cube"],
            "step_01.target": ["region"],
        },
        "role_bindings": {},
    }
    plan = SemanticTaskPlanner().plan_spec(
        source,
        bindings,
        [
            {"runtime_uid": uid, "init_pos": [0.0, 0.0, 0.7], "init_rot": [0.0] * 3}
            for uid in ("cube", "region")
        ],
    )
    nodes = plan.graph["nodes"]
    program = {
        "program_id": "pour-test",
        "program": {
            "items": [
                {"name": n["id"], "steps": {"call": n["call"]}, "post": []}
                for n in nodes
            ]
        },
    }
    constraints = {"presets": {}}
    contract = {
        "task_spec_digest": canonical_hash(plan.task_spec.to_dict()),
        "graph_digest": canonical_hash(plan.graph),
        "program_digest": canonical_hash(program),
        "constraints_digest": canonical_hash(constraints),
        "coverage": _coverage(plan),
    }
    segments = []
    for index, node in enumerate(nodes):
        segment_id = f"pour-test:segment:$.program.items[{index}]"
        invocation = f"pour-test/{segment_id}:0"
        release = node["call"]["kind"] == "place"
        receipt = deepcopy(proof)
        receipt["effect_spec"]["invocation_id"] = invocation
        segments.append(
            {
                "name": node["id"],
                "active": [True],
                "successes": [True],
                "start_steps": [index * 5],
                "end_steps": [(index + 1) * 5],
                "metadata": {
                    "task_program_id": "pour-test",
                    "program_segment_id": segment_id,
                    "program_segment_source_path": ["program", "items", index],
                    "validation": {"accepted_mask": [True]},
                    "post_policies": [],
                    "runtime": {
                        "env_ids": [0],
                        "task_state": {
                            "batch_size": 1,
                            "held_objects": [],
                            "coordinated_held_objects": [],
                        },
                        "calls": [
                            {
                                "semantic_id": node["call"].get(
                                    "call_id", node["call"]["kind"]
                                ),
                                "skill_id": "pour",
                                "invocation_id": invocation,
                                "invocation_revision": 0,
                                "status": "completed",
                                "command_count": 5,
                                "effects": [receipt] if release else [],
                            }
                        ],
                    },
                },
            }
        )
    count = len(nodes) * 5
    state = {}
    for uid in ("cube", "region"):
        pose = torch.zeros(1, count + 1, 7)
        pose[..., 6] = 1.0
        state[uid] = {"pose": pose}
    pour_index = next(
        i for i, n in enumerate(nodes) if n["call"].get("call_id") == "simulation.pour"
    )
    start = pour_index * 5
    angles = [0.0, tilt / 2, tilt, tilt, restore, restore]
    rotations = Rotation.from_euler(rotation_axis, angles, degrees=True)
    poses = state["cube"]["pose"][0, start : start + 6]
    poses[:, 3:] = torch.tensor(rotations.as_quat()).float()
    pivot = np.array([0.0, 0.0, 0.04])
    poses[:, :3] = torch.tensor(
        np.array([offset, 0.0, 0.4]) - rotations.apply(pivot)
    ).float()
    env._traj_buffer = {"states": {"rigid_objects": state}}
    env._traj_steps = torch.tensor([count])
    save_config(tmp_path / "task_spec.json", plan.task_spec.to_dict())
    save_config(tmp_path / "task_program/constraints.json", constraints)
    return plan, contract, {"segments": segments}, program, constraints, env, deployment


def _observe(tmp_path, data, *, damage=None):
    plan, contract, runtime, program, constraints, env, deployment = data
    records = capture_pour_motions(env, deployment, plan.graph, runtime, program, 1)
    if damage is not None:
        record = records["step_01"]
        if damage == "foreign_receiver":
            record["measurements"]["reference_uid"] = "another-bowl"
        elif damage == "foreign_invocation":
            record["measurements"]["invocation_id"] = "another-occurrence"
        elif damage == "cached_success":
            record["accepted"] = [True]
            record["measurements"]["rows"][0]["maximum_tilt_rad"] = 0.0
    final = capture_final_goals(
        env,
        deployment,
        tmp_path,
        plan.graph,
        contract,
        1,
        runtime=runtime,
        pour_motions=records,
    )
    result = evaluate_runtime_evidence(
        plan.task_spec,
        contract,
        plan.graph,
        runtime,
        final,
        num_envs=1,
        run_id="pour-test",
        program=program,
        constraints=constraints,
        pour_milestones=records,
    )
    return result["environments"][0], records


def test_native_tilt_restore_above_bound_receiver_certifies_action_not_liquid(tmp_path):
    result, records = _observe(tmp_path, _case(tmp_path))
    assert result["certificate_ready"] is True
    assert result["milestones"][0]["status"] == "satisfied"
    assert records["step_01"]["measurements"]["fluid_transfer_verified"] is False


def test_motion_receipts_never_certify_existing_fluid_transfer_spec(tmp_path):
    result, _ = _observe(tmp_path, _case(tmp_path, mode="fluid"))
    assert result["certificate_ready"] is False
    assert result["goal"][0]["predicate"] == "poured"
    assert result["goal"][0]["status"] == "unavailable"


def test_pour_occurrence_retains_successful_rows_in_a_partial_batch(tmp_path):
    from embodichain.gen_sim.task_engine._task_program.pour_evidence import _occurrence

    plan, _, runtime, program, _, _, _ = _case(tmp_path)
    for segment in runtime["segments"]:
        segment["active"] = [True, True]
        segment["successes"] = [True, False]
        segment["metadata"]["validation"]["accepted_mask"] = [True, False]
        segment["metadata"]["runtime"]["env_ids"] = [0, 1]
    *_, eligible = _occurrence(
        plan.graph["task_groups"][0], plan.graph, runtime, program, 2
    )
    assert eligible == [True, False]


def test_full_pour_observer_keeps_a_valid_row_when_another_environment_failed(tmp_path):
    plan, contract, runtime, program, constraints, env, deployment = _case(tmp_path)
    for name in ("cube", "region"):
        native = env.sim.get_rigid_object(name)
        vertices, faces = native.get_vertices(), native.get_triangles()
        native.get_vertices = lambda vertices=vertices, **_kw: vertices.repeat(2, 1, 1)
        native.get_triangles = lambda faces=faces, **_kw: faces.repeat(2, 1, 1)
        values = env._traj_buffer["states"]["rigid_objects"][name]["pose"]
        env._traj_buffer["states"]["rigid_objects"][name]["pose"] = values.repeat(
            2, 1, 1
        )
    env._traj_steps = env._traj_steps.repeat(2)
    for segment in runtime["segments"]:
        segment.update(active=[True, True], successes=[True, False])
        for key in ("start_steps", "end_steps"):
            segment[key] *= 2
        metadata = segment["metadata"]
        metadata["validation"]["accepted_mask"] = [True, False]
        metadata["runtime"]["env_ids"] = [0, 1]
        metadata["runtime"]["task_state"]["batch_size"] = 2
        for call in metadata["runtime"]["calls"]:
            for receipt in call["effects"]:
                receipt["effect_spec"]["env_ids"] = [0, 1]
                receipt["decision"]["success_mask"] = [True, False]
                for sample in receipt["evidence"].values():
                    sample["env_ids"] = [0, 1]
                    sample["valid_mask"] = [True, False]
                    if "values" in sample:
                        sample["values"] *= 2
                    if "object_to_endpoint" in sample:
                        sample["object_to_endpoint"] *= 2
    records = capture_pour_motions(env, deployment, plan.graph, runtime, program, 2)
    assert records["step_01"]["accepted"] == [True, False]
    final = capture_final_goals(
        env,
        deployment,
        tmp_path,
        plan.graph,
        contract,
        2,
        runtime=runtime,
        pour_motions=records,
    )
    result = evaluate_runtime_evidence(
        plan.task_spec,
        contract,
        plan.graph,
        runtime,
        final,
        num_envs=2,
        run_id="partial-batch",
        program=program,
        constraints=constraints,
        pour_milestones=records,
    )
    assert result["environments"][0]["certificate_ready"] is True
    assert result["environments"][1]["certificate_ready"] is False


@pytest.mark.parametrize(
    "change",
    [
        {"tilt": 0.0},
        {"tilt": 20.0},
        {"restore": 40.0},
        {"offset": 1.0},
        {"rotation_axis": "z"},
    ],
)
def test_program_completion_never_substitutes_for_measured_pour_motion(
    tmp_path, change
):
    result, records = _observe(tmp_path, _case(tmp_path, **change))
    assert records["step_01"]["accepted"] == [False]
    assert result["certificate_ready"] is False


@pytest.mark.parametrize(
    "damage", ["foreign_receiver", "foreign_invocation", "cached_success"]
)
def test_motion_verdict_rejects_borrowed_or_inconsistent_receipts(tmp_path, damage):
    result, _ = _observe(tmp_path, _case(tmp_path), damage=damage)
    assert result["certificate_ready"] is False


@pytest.mark.parametrize("damage", ["missing_slot", "missing_release", "zero_commands"])
def test_missing_motion_or_release_evidence_is_unavailable(tmp_path, damage):
    data = _case(tmp_path)
    if damage == "missing_slot":
        data[-2]._traj_steps[:] = 1
    else:
        for segment in data[2]["segments"]:
            for call in segment["metadata"]["runtime"]["calls"]:
                if damage == "missing_release":
                    call["effects"] = []
                else:
                    call["command_count"] = 0
    result, records = _observe(tmp_path, data)
    assert "unavailable" in records["step_01"]
    assert result["certificate_ready"] is False
