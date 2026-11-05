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
import json
from pathlib import Path

import pytest

from embodichain.gen_sim.task_spec import TaskSpec, canonical_hash, EvaluationStatus
from embodichain.gen_sim.task_spec.condition_group_evidence import (
    ConditionEvidenceGroupBinding,
    evaluate_condition_group_evidence,
)
from embodichain.gen_sim.task_spec.evidence_report import (
    evaluate_physical_evidence_report,
)
from embodichain.gen_sim.task_spec.condition_evidence import ConditionEvidenceBinding

REFERENCE = json.loads(
    (Path(__file__).parent / "data/e2_count_source_reference.json").read_text()
)


def _declaration(*, recorded, data=REFERENCE, predicate_name=None):
    spec, graph, validation, program = (
        data["spec"],
        data["graph"],
        data["validation"],
        data["program"],
    )
    occurrences = {
        node["id"]: dict(segment_name=node["id"], source_path=["program", "items", i])
        for i, node in enumerate(graph["nodes"])
    }
    lineage = {
        v["group_id"]: v
        for values in validation["coverage"]["lineage"].values()
        for v in values
    }
    groups = graph["task_groups"]
    members = []
    for group in groups:
        uid, terminal = lineage[group["id"]]["object_uid"], group["node_ids"][-1]
        terminal_source = program
        for key in occurrences[terminal]["source_path"]:
            terminal_source = terminal_source[key]
        preset = terminal_source["post"][-1]["preset"]
        members.append(
            dict(
                group_id=group["id"],
                state=dict(
                    entity_id=uid,
                    native_uid=data.get("native_uids", {}).get(
                        uid, data.get("native_uid", uid)
                    ),
                    part_id=lineage[group["id"]].get("object_part_id"),
                    condition=data.get(
                        "resolved_conditions", data["constraints"]["presets"]
                    )[preset],
                ),
                boundary_node_id=terminal if recorded else groups[-1]["node_ids"][-1],
                policy_node_id=terminal,
                stability_preset=preset,
            )
        )
    predicates = spec["milestones"][0]["achieve"] if recorded else spec["goal"]
    predicate = next(
        value
        for value in predicates
        if predicate_name is None or value["predicate"] == predicate_name
    )
    selected = []
    for member, group in zip(members, groups, strict=True):
        state = member["state"]
        policy = occurrences[member["policy_node_id"]]
        boundary = occurrences[member["boundary_node_id"]]
        selected.append(
            ConditionEvidenceBinding(
                canonical_hash(spec),
                canonical_hash(program),
                predicate["id"],
                state["entity_id"],
                state["native_uid"],
                (
                    "joint_endpoint"
                    if spec["goal"][0]["arguments"]["task_type"] == "E6"
                    else "orientation"
                ),
                "recorded" if recorded else "final",
                state["condition"],
                boundary_segment_id=boundary["segment_name"],
                boundary_source_path=tuple(boundary["source_path"]),
                policy_segment_id=policy["segment_name"],
                policy_source_path=tuple(policy["source_path"]),
                stability_preset=member["stability_preset"],
                required_execution=(
                    tuple(
                        dict(
                            segment_id=name,
                            source_path=occurrences[name]["source_path"],
                        )
                        for name in group["node_ids"]
                    )
                    if recorded
                    else ()
                ),
                episode_ids=(4,),
            )
        )
    role_id = predicate["subject"]
    return ConditionEvidenceGroupBinding(
        canonical_hash(spec),
        canonical_hash(program),
        predicate["id"],
        role_id,
        tuple(group["id"] for group in groups),
        tuple(selected),
    )


def _packet(binding, case):
    if binding.members[0].evidence_kind == "joint_endpoint":
        return _joint_packet(binding, case)
    identity = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    members = {}
    segments = {v["name"]: v for v in case["runtime"]["segments"]}
    for value in binding.members:
        if (
            binding.observation_kind == "recorded"
            and value.boundary_segment_id not in segments
        ):
            continue
        available = (
            binding.observation_kind == "recorded" or value.entity_id in case["poses"]
        )
        poses = (
            [identity]
            if binding.observation_kind == "recorded"
            else case["poses"].get(value.entity_id, [None])
        )
        packet = dict(
            schema_version="simulation.state_observation/v1",
            entity_id=value.entity_id,
            native_uid=value.native_uid,
            observation_kind=binding.observation_kind,
            episode_ids=[4],
            state_acquisition=dict(
                evidence_id="rigid.pose",
                value_dtype="torch.float32",
                values=poses,
                valid=[available],
                acquisition_errors=[
                    (
                        None
                        if available
                        else "Required native rigid object is not present."
                    )
                ],
                timestamp=1.0,
                env_ids=[0],
                observation_revision=1,
                clock_domain="simulation_time",
                revision_kind="collector_acquisition",
            ),
        )
        if binding.observation_kind == "recorded":
            packet.update(
                segments=case["runtime"]["segments"],
                boundary_frames=segments[value.boundary_segment_id]["end_steps"],
            )
        members[value.binding_hash] = packet
    packet = dict(
        schema_version="task_spec.condition_group_observation/v1",
        binding_hash=binding.binding_hash,
        episode_ids=[4],
        members=members,
    )
    if binding.observation_kind == "recorded" and len(members) == len(binding.members):
        packet["boundary_frames"] = [
            max(v["boundary_frames"][0] for v in members.values())
        ]
    return packet


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["name"])
def test_finite_count_matches_actual_b0_final_and_milestone_classification(case):
    before = deepcopy(REFERENCE["spec"])
    final, recorded = _declaration(recorded=False), _declaration(recorded=True)
    packets = {value.predicate_id: _packet(value, case) for value in (final, recorded)}
    report = evaluate_physical_evidence_report(
        REFERENCE["spec"],
        (recorded, final),
        packets,
        program=REFERENCE["program"],
        num_envs=1,
    )[0].to_dict()
    observed = dict(
        goal=[v["status"] for v in report["goal"]],
        milestones=[v["status"] for v in report["milestones"]],
        status=report["status"],
        certificate_ready=report["certificate_ready"],
    )
    assert observed == case["expected"]
    assert REFERENCE["spec"] == before
    assert TaskSpec.from_dict(before).to_dict() == before


def test_complete_count_cannot_be_replaced_by_a_subset_or_duplicate():
    group = _declaration(recorded=False)
    incomplete = replace(
        group, members=group.members[:1], group_ids=group.group_ids[:1], binding_hash=""
    )
    with pytest.raises(ValueError, match="exact declared finite count"):
        evaluate_condition_group_evidence(
            REFERENCE["spec"], incomplete, {}, program=REFERENCE["program"]
        )
    with pytest.raises(ValueError, match="duplicate"):
        replace(group, members=(group.members[0], group.members[0]), binding_hash="")
    case = deepcopy(REFERENCE["cases"][1])
    packet = _packet(group, case)
    del packet["members"][group.members[1].binding_hash]
    assert (
        evaluate_condition_group_evidence(
            REFERENCE["spec"], group, packet, program=REFERENCE["program"]
        )[0].status
        is EvaluationStatus.UNAVAILABLE
    )


def test_group_and_member_identity_drift_are_not_resigned():
    group = _declaration(recorded=False)
    original = group.to_dict()
    assert ConditionEvidenceGroupBinding.from_dict(group).to_dict() == original
    group.members[0].condition["minimum_alignment"] = 0.0
    with pytest.raises(ValueError, match="binding_hash"):
        ConditionEvidenceGroupBinding.from_dict(group)
    changed = deepcopy(original)
    changed["group_ids"].reverse()
    with pytest.raises(ValueError, match="binding_hash"):
        ConditionEvidenceGroupBinding.from_dict(changed)


def test_raw_member_and_event_boundaries_cannot_borrow_another_episode_or_time():
    group = _declaration(recorded=True)
    packet = _packet(group, REFERENCE["cases"][0])
    packet["members"][group.members[1].binding_hash]["episode_ids"] = [5]
    assert (
        evaluate_condition_group_evidence(
            REFERENCE["spec"], group, packet, program=REFERENCE["program"]
        )[0].status
        is EvaluationStatus.UNAVAILABLE
    )
    packet = _packet(group, REFERENCE["cases"][0])
    packet["boundary_frames"] = packet["members"][group.members[0].binding_hash][
        "boundary_frames"
    ]
    assert (
        evaluate_condition_group_evidence(
            REFERENCE["spec"], group, packet, program=REFERENCE["program"]
        )[0].status
        is EvaluationStatus.UNAVAILABLE
    )


def test_all_universe_remains_outside_the_source_count_contract():
    source = REFERENCE["all_universe_source"]
    assert (
        source["expected"]["goal"]
        == source["expected"]["milestones"]
        == ["unavailable"]
    )
    group = _declaration(recorded=False)
    # This is an intentionally new declaration for the actual frozen all-spec,
    # not re-signing or reducing that spec to a synthetic single-member child.
    digest = canonical_hash(source["spec"])
    members = tuple(
        replace(value, task_spec_digest=digest, binding_hash="")
        for value in group.members
    )
    declared = replace(group, task_spec_digest=digest, members=members, binding_hash="")
    with pytest.raises(ValueError, match="finite count"):
        evaluate_condition_group_evidence(
            source["spec"], declared, {}, program=REFERENCE["program"]
        )


def test_current_count_cannot_borrow_an_earlier_member_sample_or_another_run():
    group = _declaration(recorded=False)
    packet = _packet(group, REFERENCE["cases"][0])
    packet["members"][group.members[0].binding_hash]["state_acquisition"][
        "timestamp"
    ] = 0.9
    outcome = evaluate_condition_group_evidence(
        REFERENCE["spec"], group, packet, program=REFERENCE["program"]
    )[0]
    assert outcome.status is EvaluationStatus.UNAVAILABLE
    packet = _packet(group, REFERENCE["cases"][0])
    packet["run_id"] = "new-run"
    for value in packet["members"].values():
        value["run_id"] = "new-run"
    packet["members"][group.members[0].binding_hash]["run_id"] = "old-run"
    assert (
        evaluate_condition_group_evidence(
            REFERENCE["spec"], group, packet, program=REFERENCE["program"]
        )[0].status
        is EvaluationStatus.UNAVAILABLE
    )


def test_existing_single_binding_keeps_its_original_legal_payload_and_hash():
    original = REFERENCE["existing_single_binding"]["binding"]
    restored = ConditionEvidenceBinding.from_dict(original)
    assert restored.to_dict() == original
    assert (
        restored.binding_hash
        == "864734ff8d262ad54ea2c2195b7c849416f9fa06e102f948abf270a3eb8ed6b8"
    )


def test_group_snapshot_decode_has_no_simulation_or_numeric_import_dependency():
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[3]
    data = _declaration(recorded=False).to_dict()
    script = f"import sys;sys.path.insert(0,{str(root)!r});from embodichain.gen_sim.task_spec.condition_group_evidence import ConditionEvidenceGroupBinding;ConditionEvidenceGroupBinding.from_dict({data!r});assert not any(name=='torch' or name=='numpy' or name.startswith('embodichain.lab') or name.startswith('embodichain.gen_sim.task_engine') for name in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-I", "-c", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _joint_packet(binding, case):
    packets = {}
    segments = {value["name"]: value for value in case["runtime"]["segments"]}
    for member in binding.members:
        if (
            binding.observation_kind == "recorded"
            and member.boundary_segment_id not in segments
        ):
            continue
        packet = dict(
            schema_version="simulation.state_observation/v1",
            entity_id=member.entity_id,
            native_uid=member.native_uid,
            observation_kind=binding.observation_kind,
            episode_ids=[4],
            joint=member.condition["joint"],
        )
        if binding.observation_kind == "recorded":
            terminal = segments[member.boundary_segment_id]
            packet.update(
                segments=case["runtime"]["segments"],
                boundary_frames=terminal["end_steps"],
                policy_result=terminal["metadata"]["post_policies"][0]["result"],
            )
        else:
            available = member.condition["joint"] in case["joint_names"]
            value = (
                case["joint_values"][
                    case["joint_names"].index(member.condition["joint"])
                ]
                if available
                else None
            )
            packet["state_acquisition"] = dict(
                evidence_id="articulation.joint_position",
                value_dtype="torch.float32",
                values=[value],
                valid=[available],
                acquisition_errors=[
                    None if available else "Required native joint is missing."
                ],
                timestamp=1.0,
                env_ids=[0],
                observation_revision=1,
                clock_domain="simulation_time",
                revision_kind="collector_acquisition",
            )
        packets[member.binding_hash] = packet
    packet = dict(
        schema_version="task_spec.condition_group_observation/v1",
        binding_hash=binding.binding_hash,
        episode_ids=[4],
        members=packets,
    )
    if binding.observation_kind == "recorded" and len(packets) == len(binding.members):
        packet["boundary_frames"] = [
            max(value["boundary_frames"][0] for value in packets.values())
        ]
    return packet


PARTS = json.loads(
    (Path(__file__).parent / "data/e6_count_parts_source_reference.json").read_text()
)


@pytest.mark.parametrize("case", PARTS["cases"], ids=lambda value: value["name"])
def test_shared_articulation_parts_retain_actual_source_final_and_unavailable_milestone(
    case,
):
    assert PARTS["terminal_validator_counts"] == [1, 2]
    assert case["expected"]["milestones"] == ["unavailable"]
    final, past = _declaration(recorded=False, data=PARTS), _declaration(
        recorded=True, data=PARTS
    )
    assert len({value.native_uid for value in final.members}) == 1
    assert len({value.condition["joint"] for value in final.members}) == 2
    packets = {value.predicate_id: _packet(value, case) for value in (final, past)}
    report = evaluate_physical_evidence_report(
        PARTS["spec"], (past, final), packets, program=PARTS["program"], num_envs=1
    )[0].to_dict()
    observed = dict(
        goal=[value["status"] for value in report["goal"]],
        milestones=[value["status"] for value in report["milestones"]],
        status=report["status"],
        certificate_ready=report["certificate_ready"],
    )
    assert observed == case["expected"]


def test_articulation_count_cannot_duplicate_a_joint_under_different_part_names():
    group = _declaration(recorded=False, data=PARTS)
    first, second = group.members
    aliased = replace(
        second,
        condition={**second.condition, "joint": first.condition["joint"]},
        binding_hash="",
    )
    with pytest.raises(ValueError, match="duplicate"):
        replace(group, members=(first, aliased), binding_hash="")


E1_REFERENCE = json.loads(
    (Path(__file__).parent / "data/e1_orientation_source_reference.json").read_text()
)


@pytest.mark.parametrize(
    "data,case",
    [
        pytest.param(
            data,
            case,
            id=f"{data['orientation']}-{data['count']}-{case['name']}",
        )
        for data in E1_REFERENCE["declarations"]
        if data["orientation"] in {"upright", "vertical"}
        for case in data["cases"]
    ],
)
def test_e1_original_orientation_facts_keep_the_full_normal_specification(data, case):
    from embodichain.gen_sim.task_spec.condition_evidence import (
        evaluate_condition_evidence,
    )

    name = "object_upright" if data["orientation"] == "upright" else "orientation"
    group = _declaration(recorded=False, data=data, predicate_name=name)
    final = group.members[0] if data["count"] == 1 else group
    packet = _packet(group, case)
    if data["count"] == 1:
        packet = packet["members"][final.binding_hash]
    bindings = [final]
    packets = {final.predicate_id: packet}
    if name == "object_upright":
        past = _declaration(recorded=True, data=data, predicate_name=name)
        past_packet = _packet(past, case)
        binding = past.members[0] if data["count"] == 1 else past
        if data["count"] == 1:
            past_packet = past_packet["members"].get(binding.binding_hash, {})
        bindings.append(binding)
        packets[binding.predicate_id] = past_packet
        verifier = (
            evaluate_condition_evidence
            if data["count"] == 1
            else evaluate_condition_group_evidence
        )
        source_observed = any(
            snapshot.get(binding.predicate_id) is True
            for snapshot in case["normative_trajectory"]
        )
        assert verifier(data["spec"], binding, past_packet, program=data["program"])[
            0
        ].status is (
            EvaluationStatus.SATISFIED
            if source_observed
            else EvaluationStatus.UNAVAILABLE
        )
    else:
        # The source emits vertical final geometry but no historical
        # orientation milestone. A richer-looking aligned receipt cannot
        # silently change that existing qualification boundary.
        past = _declaration(recorded=True, data=data, predicate_name=name)
        with pytest.raises(ValueError, match="recorded.*upright"):
            evaluate_condition_group_evidence(
                data["spec"], past, _packet(past, case), program=data["program"]
            )
    report = evaluate_physical_evidence_report(
        data["spec"], bindings, packets, program=data["program"], num_envs=1
    )[0].to_dict()
    observed = dict(
        goal=[v["status"] for v in report["goal"]],
        milestones=[v["status"] for v in report["milestones"]],
        status=report["status"],
        certificate_ready=report["certificate_ready"],
    )
    # Placement observations were unavailable in this original acquisition-port
    # oracle. Restoring independent orientation facts cannot certify them.
    assert observed == case["expected"]
    assert TaskSpec.from_dict(data["spec"]).to_dict() == data["spec"]


def test_e1_preserve_orientation_remains_outside_the_source_state_observer():
    source = next(
        value
        for value in E1_REFERENCE["declarations"]
        if value["orientation"] == "preserve"
    )
    assert all(
        set(case["expected"]["goal"]) == {"unavailable"}
        and not case["expected"]["certificate_ready"]
        for case in source["cases"]
    )
    with pytest.raises(ValueError, match="explicit axis"):
        _declaration(recorded=False, data=source, predicate_name="orientation")
