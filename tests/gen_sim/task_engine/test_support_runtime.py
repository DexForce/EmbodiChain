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

"""Provider-free strong support collection and TaskSpec receipt regressions."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from embodichain.gen_sim.task_engine._task_program.placement_evidence import (
    measure_planar_placement,
)
from embodichain.gen_sim.task_engine._task_program.stability import (
    StabilityConstraint,
    _constraint_pose_flags,
)
from embodichain.gen_sim.task_engine._task_program.support_evidence import (
    support_from_contacts,
)
from embodichain.gen_sim.task_engine._task_program.task_spec_runtime import (
    capture_final_goals,
    capture_placement_milestones,
    evaluate_runtime_evidence,
)
from embodichain.gen_sim.task_engine.contracts import canonical_hash
from embodichain.utils.math import trans_matrix_to_xyz_quat
from embodichain.utils.utility import save_config

from .test_task_spec_runtime import _coverage, _observe_placement, _placement_evidence

CONTROL_DT = 0.04
SEGMENT_STEPS = 8
STABLE_STEPS = 2


def _strong_case(tmp_path, *, relation="inside", reference="tray", **kwargs):
    data = _placement_evidence(
        tmp_path, relation=relation, reference_uid=reference, **kwargs
    )
    plan, contract, runtime, program, constraints, env, deployment = data
    nodes = plan.graph["nodes"]
    placement = next(
        n
        for n in nodes
        if n["call"]["kind"] == "place"
        or n["call"].get("call_id") == "simulation.place_relative"
    )
    preset = f"gen_sim.{placement['id']}.stable"
    cfg = StabilityConstraint(
        entity="cube",
        reference=reference,
        kind="supported_region",
        duration=STABLE_STEPS * CONTROL_DT,
        support_contract_revision=1,
    )
    constraints.update(support_contract_revision=1, presets={preset: asdict(cfg)})
    deployment.integration.adapter_factory = SimpleNamespace(
        constraints=((preset, cfg),),
        support_contract_revision=1,
    )
    for index, segment in enumerate(runtime["segments"]):
        segment.update(
            start_steps=[index * SEGMENT_STEPS], end_steps=[(index + 1) * SEGMENT_STEPS]
        )
    terminal = runtime["segments"][-1]
    end_step = terminal["end_steps"][0]
    env._traj_steps = torch.tensor([end_step])
    for states in env._traj_buffer["states"]["rigid_objects"].values():
        states["pose"] = states["pose"][:, :1].repeat(1, end_step + 1, 1)
    place_index = nodes.index(placement)
    start_step = place_index * SEGMENT_STEPS
    released_at = (start_step + 2) * CONTROL_DT
    call = runtime["segments"][place_index]["metadata"]["runtime"]["calls"][0]
    call.update(call_index=place_index, active_plan_attempt_generation=0)
    receipt = call["effects"][0]
    receipt.update(timestamp=released_at, verification_id=3)
    for sample in receipt["evidence"].values():
        sample["timestamp"] = released_at
    identity = {key: call[key] for key in ("invocation_id", "invocation_revision")}
    call["plan_attempts"] = [
        {
            **identity,
            "attempt_generation": 0,
            "invocation_index": 0,
            "planned_at": start_step * CONTROL_DT,
            "trajectory_segments": [
                {"name": "release", "start": 1, "stop": 3, "waypoint_count": 2}
            ],
        }
    ]
    call["events"] = [
        {
            **identity,
            "invocation_index": 0,
            "env_mask": [True],
            "kind": kind,
            "timestamp": (start_step + offset) * CONTROL_DT,
            **({"segment_name": "release"} if offset == 1 else {}),
        }
        for kind, offset in (("trajectory_segment_entered", 1), ("action_completed", 3))
    ]
    obj, ref = env.sim.get_rigid_object("cube"), env.sim.get_rigid_object(reference)
    pose, reference_pose = obj.get_local_pose(), ref.get_local_pose()
    points = np.array([[-0.02, -0.02, 0.005], [0.02, -0.02, 0.005], [0.0, 0.02, 0.005]])
    points[:, :2] += pose[0, :2, 3].numpy()
    supported, row = support_from_contacts(
        np.tile([11, 22], (3, 1)),
        points,
        np.tile([0.0, 0.0, 1.0], (3, 1)),
        np.full(3, 0.01),
        pose[0, :3, 3].numpy(),
        object_actor=22,
        reference_actor=11,
        tolerance=0.003,
    )
    row.update(
        retained=True,
        sample={
            "timestamp": released_at,
            "control_step": start_step + 2,
            "substep": 1,
            "serial": 1,
            "native_revision": 1,
            "topology_revision": 0,
            "episode_generation": 0,
        },
    )
    row["raw_contacts"] = {
        "user_ids": np.tile([11, 22], (3, 1)).tolist(),
        "position": points.tolist(),
        "normal": np.tile([0.0, 0.0, 1.0], (3, 1)).tolist(),
        "impulse": np.full(3, 0.01).tolist(),
    }

    def actor_info(actor, uid):
        return {
            "actor_id": actor,
            "env_id": 0,
            "link_name": None,
            "path": f"/arena/{uid}",
            "arena_name": "support-test",
        }

    native = {
        "scope": "native_contact_support",
        "support_contract_revision": 1,
        "entity": "cube",
        "reference": reference,
        "env_ids": [0],
        "object_pose": pose.tolist(),
        "reference_pose": reference_pose.tolist(),
        "supported": [supported],
        "rows": [row],
        "episode_generations": [0],
        "actor_pairs": [
            {
                "object_uid": "cube",
                "reference_uid": reference,
                "object_actor": 22,
                "reference_actor": 11,
                "object_info": actor_info(22, "cube"),
                "reference_info": actor_info(11, reference),
            }
        ],
        "clock": "environment_step",
        "step_dt": CONTROL_DT,
        "observed_step": end_step,
        "observed_at": end_step * CONTROL_DT,
    }
    _, landing = measure_planar_placement(
        env.sim,
        SimpleNamespace(simulation_uid="cube"),
        SimpleNamespace(simulation_uid=reference),
        pose,
        reference_pose,
    )
    prefix = ["program", "items", len(nodes) - 1]
    terminal["metadata"]["program_segment_source_path"] = prefix
    policy = {
        "kind": "wait_stable",
        "policy_index": 0,
        "source_path": [*prefix, "post", 0],
        "result_mask": [True],
        "result": {
            "kind": cfg.kind,
            "preset": preset,
            "entity": "cube",
            "reference": reference,
            "support_contract_revision": 1,
            "accepted_mask": [True],
            "required_seconds": cfg.duration,
            "stable_seconds": [cfg.duration],
            "start_step": end_step - STABLE_STEPS,
            "end_step": end_step,
            "clock_started_at": (end_step - STABLE_STEPS) * CONTROL_DT,
            "clock_ended_at": end_step * CONTROL_DT,
            "measurements": {
                "native_support": deepcopy(native),
                "landing_region": landing,
                "support_gap": [float(pose[0, 2, 3]) - 0.045],
            },
        },
    }
    terminal["metadata"]["post_policies"] = [policy]
    program["program"]["items"][-1]["post"] = [
        {"kind": "wait_stable", "entity": "cube", "preset": preset}
    ]
    live = deepcopy(native)

    def observe(object_pose, ref_pose, **_kwargs):
        snapshot = deepcopy(live)
        snapshot.update(
            object_pose=object_pose.tolist(), reference_pose=ref_pose.tolist()
        )
        return torch.tensor(snapshot["supported"]), snapshot

    reader = SimpleNamespace(
        entity_id="cube",
        reference_id=reference,
        object=obj,
        reference=ref,
        observe=observe,
    )
    sensor = SimpleNamespace(support_reader=reader)
    env.sim.get_sensor = lambda uid: (
        sensor if uid == f"gen_sim_support__cube__{reference}" else None
    )
    contract["program_digest"] = canonical_hash(program)
    contract["constraints_digest"] = canonical_hash(constraints)
    save_config(tmp_path / "task_program/constraints.json", constraints)
    return SimpleNamespace(
        data=data,
        source=policy["result"]["measurements"]["native_support"],
        live=live,
        policy=policy,
        call=call,
        placement=placement,
        terminal=terminal,
        reader=reader,
        end_step=end_step,
    )


def _records(tmp_path, case):
    plan, contract, runtime, _program, _constraints, env, deployment = case.data
    final = capture_final_goals(
        env, deployment, tmp_path, plan.graph, contract, 1, runtime=runtime
    )
    history = capture_placement_milestones(
        env, deployment, plan.graph, contract, runtime, 1, spec=plan.task_spec
    )
    return final, history


def _evaluate(case, final, history):
    plan, contract, runtime, program, constraints, _env, _deployment = case.data
    return evaluate_runtime_evidence(
        plan.task_spec,
        contract,
        plan.graph,
        runtime,
        final,
        num_envs=1,
        run_id="strong-support-test",
        program=program,
        constraints=constraints,
        placement_milestones=history,
    )["environments"][0]


@pytest.mark.parametrize(
    "relation,reference", [("inside", "tray"), ("on", "table"), ("on", "stack_base")]
)
def test_strong_rigid_placement_collects_and_certifies(tmp_path, relation, reference):
    case = _strong_case(tmp_path, relation=relation, reference=reference)
    final, history = _records(tmp_path, case)
    result = _evaluate(case, final, history)
    assert result["certificate_ready"] is True
    record = next(iter(final.values()))["members"][0]
    assert record["measurements"]["scope"] == "native_supported_placement"
    assert record["measurements"]["reference_uid"] == reference


@pytest.mark.parametrize(
    "scope", ["measured_planar_landing_region", "native_contact_support"]
)
@pytest.mark.parametrize("target", ["final", "history"])
def test_strong_root_never_downgrades_to_legacy_receipt(tmp_path, scope, target):
    case = _strong_case(tmp_path)
    final, history = _records(tmp_path, case)
    record = (
        next(iter(final.values()))["members"][0]
        if target == "final"
        else next(iter(history.values()))
    )
    record["measurements"]["scope"] = scope
    assert _evaluate(case, final, history)["certificate_ready"] is False


@pytest.mark.parametrize("target", ["final", "history"])
def test_release_phase_load_with_verified_release_and_stability_is_accepted(
    tmp_path, target
):
    case = _strong_case(tmp_path)
    source = case.live if target == "final" else case.source
    entered_at = case.call["events"][0]["timestamp"]
    source["rows"][0]["sample"].update(
        timestamp=entered_at, control_step=round(entered_at / CONTROL_DT)
    )
    assert entered_at < case.call["effects"][0]["timestamp"]
    final, history = _records(tmp_path, case)
    result = _evaluate(case, final, history)
    assert result["certificate_ready"] is True
    assert result["goal"][0]["status"] == "satisfied"
    assert result["milestones"][0]["status"] == "satisfied"
    assert source["rows"][0]["sample"]["timestamp"] == entered_at


@pytest.mark.parametrize(
    "damage",
    [
        "before_release",
        "future",
        "restamped",
        "epoch",
        "foreign_uid",
        "foreign_actor",
        "link",
        "env",
        "zero_load",
        "outside_com",
        "zero_raw_impulse",
        "downward_raw_normal",
        "wrong_raw_pair",
    ],
)
def test_strong_receipts_reject_foreign_or_stale_native_evidence(tmp_path, damage):
    case = _strong_case(tmp_path)
    sample = case.source["rows"][0]["sample"]
    row = case.source["rows"][0]
    pair = case.source["actor_pairs"][0]
    if damage == "before_release":
        sample.update(
            timestamp=case.call["plan_attempts"][-1]["planned_at"],
            control_step=sample["control_step"] - 2,
        )
    elif damage == "future":
        sample.update(
            timestamp=(case.end_step + 1) * CONTROL_DT, control_step=case.end_step + 1
        )
    elif damage == "restamped":
        sample["timestamp"] = case.source["observed_at"]
    elif damage == "epoch":
        sample["episode_generation"] += 1
    elif damage == "foreign_uid":
        pair["reference_uid"] = "other_tray"
    elif damage == "foreign_actor":
        pair["reference_info"]["actor_id"] = 33
    elif damage == "link":
        pair["reference_info"]["link_name"] = "drawer_2"
    elif damage == "env":
        pair["reference_info"]["env_id"] = 1
    elif damage == "zero_load":
        row["upward_impulse"] = 0.0
    elif damage == "outside_com":
        row["com_patch_distance_m"] = row["com_tolerance_m"] + 0.01
    elif damage == "zero_raw_impulse":
        row["raw_contacts"]["impulse"] = [0.0] * 3
    elif damage == "downward_raw_normal":
        row["raw_contacts"]["normal"] = [[0.0, 0.0, -1.0]] * 3
    else:
        row["raw_contacts"]["user_ids"] = [[11, 33]] * 3
    final, history = _records(tmp_path, case)
    assert _evaluate(case, final, history)["certificate_ready"] is False


@pytest.mark.parametrize(
    "damage",
    [
        "no_seeded_slot",
        "wrong_frame",
        "pose_at_previous_frame",
        "source_path",
        "policy_index",
        "duplicate",
        "short_window",
    ],
)
def test_strong_history_requires_exact_seeded_terminal_boundary(tmp_path, damage):
    case = _strong_case(tmp_path)
    env = case.data[5]
    if damage == "no_seeded_slot":
        for states in env._traj_buffer["states"]["rigid_objects"].values():
            states["pose"] = states["pose"][:, : case.end_step]
    elif damage == "wrong_frame":
        case.terminal["end_steps"] = [case.end_step - 1]
    elif damage == "pose_at_previous_frame":
        env._traj_buffer["states"]["rigid_objects"]["cube"]["pose"][
            0, case.end_step, 0
        ] += 0.1
    elif damage == "source_path":
        case.policy["source_path"] = ["program", "items", 0, "post", 0]
    elif damage == "policy_index":
        case.policy["policy_index"] = 1
    elif damage == "duplicate":
        case.terminal["metadata"]["post_policies"].append(deepcopy(case.policy))
    else:
        case.policy["result"]["clock_started_at"] = case.policy["result"][
            "clock_ended_at"
        ]
    _final, history = _records(tmp_path, case)
    assert "unavailable" in next(iter(history.values()))


@pytest.mark.parametrize(
    "damage",
    [
        "revision",
        "generation",
        "release_event",
        "completion_mask",
        "receipt_object",
        "receipt_resource",
        "missing_terminal_receipt",
        "unverified_release",
    ],
)
def test_strong_release_joins_one_actual_plan_and_physical_receipt(tmp_path, damage):
    case = _strong_case(tmp_path)
    if damage == "revision":
        case.call["plan_attempts"][-1]["invocation_revision"] += 1
    elif damage == "generation":
        case.call["active_plan_attempt_generation"] += 1
    elif damage == "release_event":
        case.call["events"][0]["invocation_id"] = "foreign/release"
    elif damage == "completion_mask":
        case.call["events"][-1]["env_mask"] = [False]
    elif damage == "missing_terminal_receipt":
        case.call["effects"] = []
    elif damage == "unverified_release":
        case.call["effects"][0]["decision"]["success_mask"] = [False]
    else:
        expectation = case.call["effects"][0]["effect_spec"]["state_expectations"][0]
        expectation["object_id" if damage == "receipt_object" else "resource_id"] = (
            "foreign"
        )
    final, history = _records(tmp_path, case)
    assert _evaluate(case, final, history)["certificate_ready"] is False


def test_later_fresh_same_pair_contact_does_not_invalidate_qualified_history(tmp_path):
    case = _strong_case(tmp_path)
    later_step = case.end_step + SEGMENT_STEPS
    case.live.update(observed_step=later_step, observed_at=later_step * CONTROL_DT)
    case.live["rows"][0].update(
        retained=False,
        sample={
            **case.live["rows"][0]["sample"],
            "control_step": later_step,
            "timestamp": later_step * CONTROL_DT,
            "serial": 2,
        },
    )
    final, history = _records(tmp_path, case)
    assert _evaluate(case, final, history)["certificate_ready"] is True


def test_final_contact_cannot_fill_a_missing_historical_patch(tmp_path):
    case = _strong_case(tmp_path)
    case.source["rows"][0]["sample"] = None
    final, history = _records(tmp_path, case)
    assert _evaluate(case, final, history)["certificate_ready"] is False


@pytest.mark.parametrize("damage", ["epoch", "actor_pair", "env_ids", "no_reader"])
def test_current_final_support_cannot_borrow_another_episode_or_reader(
    tmp_path, damage
):
    case = _strong_case(tmp_path)
    if damage == "epoch":
        case.live["episode_generations"] = [1]
        case.live["rows"][0]["sample"]["episode_generation"] = 1
    elif damage == "actor_pair":
        pair = case.live["actor_pairs"][0]
        pair["reference_actor"] = pair["reference_info"]["actor_id"] = 33
    elif damage == "env_ids":
        case.live["env_ids"] = [1]
    else:
        case.data[5].sim.get_sensor = lambda _uid: None
    final, history = _records(tmp_path, case)
    result = _evaluate(case, final, history)
    assert result["milestones"][0]["status"] == "satisfied"
    assert result["goal"][0]["status"] != "satisfied"


def test_historical_support_survives_later_object_motion_but_final_fails(tmp_path):
    case = _strong_case(tmp_path)
    pose = case.data[5].sim.get_rigid_object("cube").get_local_pose()
    pose[0, 0, 3] += 0.4
    case.live["supported"] = [False]
    case.live["rows"][0]["sample"] = None
    final, history = _records(tmp_path, case)
    result = _evaluate(case, final, history)
    assert result["milestones"][0]["status"] == "satisfied"
    assert result["goal"][0]["status"] != "satisfied"


def test_second_same_pair_invocation_cannot_reuse_the_first_release_patch(tmp_path):
    case = _strong_case(tmp_path)
    old_sample = deepcopy(case.source["rows"][0]["sample"])
    shift = CONTROL_DT * 2
    case.call["invocation_id"] = "program/place:second"
    for attempt in case.call["plan_attempts"]:
        attempt.update(
            invocation_id=case.call["invocation_id"],
            planned_at=attempt["planned_at"] + shift,
        )
    for event in case.call["events"]:
        event.update(
            invocation_id=case.call["invocation_id"],
            timestamp=event["timestamp"] + shift,
        )
    receipt = case.call["effects"][0]
    receipt["effect_spec"]["invocation_id"] = case.call["invocation_id"]
    receipt["timestamp"] += shift
    for sample in receipt["evidence"].values():
        sample["timestamp"] = receipt["timestamp"]
    assert case.source["rows"][0]["sample"] == old_sample
    final, history = _records(tmp_path, case)
    assert _evaluate(case, final, history)["certificate_ready"] is False


@pytest.mark.parametrize("changes", [{"x": 0.24}, {"hole": True}, {"z": 0.2}])
def test_real_native_load_does_not_replace_full_planar_relation(tmp_path, changes):
    case = _strong_case(tmp_path, **changes)
    final, history = _records(tmp_path, case)
    assert _evaluate(case, final, history)["certificate_ready"] is False


def _repeated_case(tmp_path):
    from embodichain.gen_sim.task_engine.semantic_planner import SemanticTaskPlanner
    from .test_task_spec_generation import candidate, selector, step

    case = _strong_case(tmp_path)
    source = candidate(
        *[
            step(
                "E1",
                "cube",
                target=selector("tray"),
                relation="inside",
                required_arm="left_arm",
            )
            for _ in range(2)
        ]
    )
    bindings = {
        "schema_version": "action_engine_role_bindings_v1",
        "task_id": source["draft"]["task_id"],
        "candidate_id": source["candidate_id"],
        "role_bindings": {},
        "reference_bindings": {
            f"step_0{index}.{role}": [uid]
            for index in (1, 2)
            for role, uid in (("object", "cube"), ("target", "tray"))
        },
    }
    plan = SemanticTaskPlanner().plan_spec(
        source,
        bindings,
        [
            {"runtime_uid": uid, "init_pos": [0.0, 0.0, 0.7], "init_rot": [0.0] * 3}
            for uid in ("cube", "tray")
        ],
    )
    old_plan, _contract, original, _program, constraints, env, deployment = case.data
    width = len(old_plan.graph["nodes"])
    cfg = next(iter(deployment.integration.adapter_factory.constraints))[1]
    presets, adapter_constraints, segments = {}, [], []
    program = {"program": {"items": []}}
    first_sample = deepcopy(case.source["rows"][0]["sample"])
    for index, node in enumerate(plan.graph["nodes"]):
        segment = deepcopy(original["segments"][index % width])
        group_offset = index // width * width * SEGMENT_STEPS
        segment.update(
            name=node["id"],
            start_steps=[index * SEGMENT_STEPS],
            end_steps=[(index + 1) * SEGMENT_STEPS],
        )
        item = {"name": node["id"], "post": []}
        for call in segment["metadata"]["runtime"]["calls"]:
            invocation = f"program/place:{index // width}"
            call["invocation_id"] = invocation
            for attempt in call["plan_attempts"]:
                attempt.update(
                    invocation_id=invocation,
                    planned_at=attempt["planned_at"] + group_offset * CONTROL_DT,
                )
            for event in call["events"]:
                event.update(
                    invocation_id=invocation,
                    timestamp=event["timestamp"] + group_offset * CONTROL_DT,
                )
            for receipt in call["effects"]:
                receipt["effect_spec"]["invocation_id"] = invocation
                receipt["timestamp"] += group_offset * CONTROL_DT
                for sample in receipt["evidence"].values():
                    sample["timestamp"] = receipt["timestamp"]
        if (index + 1) % width == 0:
            place = plan.graph["nodes"][index - 1]
            preset = f"gen_sim.{place['id']}.stable"
            presets[preset] = asdict(cfg)
            adapter_constraints.append((preset, cfg))
            prefix = ["program", "items", index]
            segment["metadata"]["program_segment_source_path"] = prefix
            policy = segment["metadata"]["post_policies"][0]
            policy["source_path"] = [*prefix, "post", 0]
            result = policy["result"]
            result.update(
                preset=preset,
                start_step=result["start_step"] + group_offset,
                end_step=result["end_step"] + group_offset,
                clock_started_at=result["clock_started_at"] + group_offset * CONTROL_DT,
                clock_ended_at=result["clock_ended_at"] + group_offset * CONTROL_DT,
            )
            native = result["measurements"]["native_support"]
            native.update(
                observed_step=result["end_step"], observed_at=result["clock_ended_at"]
            )
            sample = native["rows"][0]["sample"]
            sample.update(
                control_step=sample["control_step"] + group_offset,
                timestamp=sample["timestamp"] + group_offset * CONTROL_DT,
                serial=index // width + 1,
            )
            item["post"] = [{"kind": "wait_stable", "entity": "cube", "preset": preset}]
        segments.append(segment)
        program["program"]["items"].append(item)
    constraints["presets"] = presets
    deployment.integration.adapter_factory.constraints = tuple(adapter_constraints)
    end_step = len(segments) * SEGMENT_STEPS
    env._traj_steps = torch.tensor([end_step])
    for states in env._traj_buffer["states"]["rigid_objects"].values():
        states["pose"] = states["pose"][:, :1].repeat(1, end_step + 1, 1)
    contract = {
        "coverage": _coverage(plan),
        "task_spec_digest": canonical_hash(plan.task_spec.to_dict()),
        "graph_digest": canonical_hash(plan.graph),
        "program_digest": canonical_hash(program),
        "constraints_digest": canonical_hash(constraints),
    }
    case.source = segments[-1]["metadata"]["post_policies"][0]["result"][
        "measurements"
    ]["native_support"]
    case.live.clear()
    case.live.update(deepcopy(case.source))
    case.data = (
        plan,
        contract,
        {"segments": segments},
        program,
        constraints,
        env,
        deployment,
    )
    case.first_sample = first_sample
    save_config(tmp_path / "task_spec.json", plan.task_spec.to_dict())
    save_config(tmp_path / "task_program/constraints.json", constraints)
    return case


@pytest.mark.parametrize("reuse_first_sample", [False, True])
def test_repeated_same_pair_groups_keep_independent_historical_windows(
    tmp_path, reuse_first_sample
):
    case = _repeated_case(tmp_path)
    if reuse_first_sample:
        case.source["rows"][0]["sample"] = deepcopy(case.first_sample)
    final, history = _records(tmp_path, case)
    result = _evaluate(case, final, history)
    assert result["certificate_ready"] is (not reuse_first_sample)
    assert result["milestones"][0]["status"] == "satisfied"


@pytest.mark.parametrize("unsupported", ["missing_preset", "reference_part"])
def test_unqualified_strong_reference_is_unavailable_without_legacy_downgrade(
    tmp_path, unsupported
):
    case = _strong_case(tmp_path)
    if unsupported == "missing_preset":
        case.data[6].integration.adapter_factory.constraints = ()
    else:
        references = next(
            v
            for v in case.data[1]["coverage"]["roles"].values()
            if v[0]["uid"] == "tray"
        )
        references[0]["part_id"] = "drawer_2"
    final, history = _records(tmp_path, case)
    assert "unavailable" in next(iter(final.values()))
    assert "unavailable" in next(iter(history.values()))


def test_legacy_geometry_fixture_keeps_its_original_certificate(tmp_path):
    assert (
        _observe_placement(tmp_path, _placement_evidence(tmp_path))["certificate_ready"]
        is True
    )


def _replace_constraint(tmp_path, case, **changes):
    _plan, contract, _runtime, _program, constraints, _env, deployment = case.data
    preset, original = deployment.integration.adapter_factory.constraints[0]
    cfg = replace(original, **changes)
    deployment.integration.adapter_factory.constraints = ((preset, cfg),)
    constraints["presets"][preset] = asdict(cfg)
    contract["constraints_digest"] = canonical_hash(constraints)
    save_config(tmp_path / "task_program/constraints.json", constraints)


def _set_supported_pose(case, pose, *, recorded):
    native = case.source if recorded else case.live
    delta = pose[0, :3, 3] - torch.tensor(native["object_pose"])[0, :3, 3]
    native["object_pose"] = pose.tolist()
    row = native["rows"][0]
    row["center_of_mass_world"] = pose[0, :3, 3].tolist()
    for owner, key in ((row, "contacts_world"), (row["raw_contacts"], "position")):
        owner[key] = (torch.tensor(owner[key]) + delta).tolist()
    env = case.data[5]
    if recorded:
        env._traj_buffer["states"]["rigid_objects"]["cube"]["pose"][
            0, case.end_step
        ] = trans_matrix_to_xyz_quat(pose)[0]
        _, landing = measure_planar_placement(
            env.sim,
            SimpleNamespace(simulation_uid="cube"),
            SimpleNamespace(simulation_uid="tray"),
            pose,
            torch.tensor(native["reference_pose"]),
        )
        case.policy["result"]["measurements"]["landing_region"] = landing
    else:
        env.sim.get_rigid_object("cube").get_local_pose().copy_(pose)


@pytest.mark.parametrize(
    "condition", ["absolute_target", "relative_target", "local_axis"]
)
@pytest.mark.parametrize("invalid_at", ["history", "final"])
def test_native_supported_region_preserves_original_pose_obligations(
    tmp_path, condition, invalid_at
):
    case = _strong_case(tmp_path)
    if condition == "absolute_target":
        changes = {"target_position": (0.0, 0.0, 0.045)}
    elif condition == "relative_target":
        changes = {"displacement": (0.0, 0.0, 0.045)}
    else:
        changes = {"local_axis": (0.0, 0.0, 1.0)}
    _replace_constraint(tmp_path, case, **changes)
    assert _evaluate(case, *_records(tmp_path, case))["certificate_ready"] is True
    pose = torch.tensor(case.source["object_pose"])
    if condition == "local_axis":
        # A quarter turn preserves this cube's footprint and support height.
        pose[0, :3, :3] = torch.tensor(
            [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]
        )
    else:
        # Eight centimetres remains inside the tray but exceeds the 4 cm target gate.
        pose[0, 0, 3] = 0.08
    _set_supported_pose(case, pose, recorded=invalid_at == "history")
    final, history = _records(tmp_path, case)
    result = _evaluate(case, final, history)
    assert result["certificate_ready"] is False
    if invalid_at == "final":
        assert result["milestones"][0]["status"] == "satisfied"
        assert result["goal"][0]["status"] != "satisfied"
    else:
        assert result["milestones"][0]["status"] != "satisfied"


@pytest.mark.parametrize("damage", ["excessive_com_tolerance", "scalar_com"])
@pytest.mark.parametrize("target", ["history", "final"])
def test_strong_load_rejects_unqualified_com_shape_and_tolerance(
    tmp_path, damage, target
):
    case = _strong_case(tmp_path)
    row = (case.source if target == "history" else case.live)["rows"][0]
    if damage == "excessive_com_tolerance":
        row["com_tolerance_m"] = 2.0
    else:
        row["center_of_mass_world"] = 0.0
    final, history = _records(tmp_path, case)
    result = _evaluate(case, final, history)
    assert result["certificate_ready"] is False
    if target == "final":
        assert result["milestones"][0]["status"] == "satisfied"


@pytest.mark.parametrize(
    "damage", [None, "outside_width", "object_axis", "reference_axis", "height_gap"]
)
def test_original_stack_pose_constraints_remain_required(damage):
    cfg = StabilityConstraint(
        entity="cube",
        reference="base",
        kind="stack",
        support_contract_revision=1,
        local_axis=(0.0, 0.0, 1.0),
        reference_axis=(0.0, 0.0, 1.0),
        object_bottom=-0.04,
        reference_top=0.005,
        reference_half_extents=(0.1, 0.1),
    )
    pose, reference = torch.eye(4)[None], torch.eye(4)[None]
    pose[0, 2, 3] = 0.045
    if damage == "outside_width":
        pose[0, 0, 3] = 0.11
    elif damage in {"object_axis", "reference_axis"}:
        tilted = pose if damage == "object_axis" else reference
        tilted[0, :3, :3] = torch.tensor(
            [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]
        )
    elif damage == "height_gap":
        pose[0, 2, 3] += 0.02
    assert _constraint_pose_flags(cfg, pose, reference).tolist() == [damage is None]


@pytest.mark.parametrize("call_id", ["gen_sim.place_upright", "gen_sim.stack_place"])
def test_explicit_strong_on_calls_reach_collector_and_matcher(tmp_path, call_id):
    case = _strong_case(tmp_path, relation="on", reference="tray")
    case.placement["call"]["call_id"] = call_id
    plan, contract = case.data[:2]
    contract["graph_digest"] = canonical_hash(plan.graph)
    final, history = _records(tmp_path, case)
    assert _evaluate(case, final, history)["certificate_ready"] is True
