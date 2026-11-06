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
from pathlib import Path
import json
import pytest
from embodichain.gen_sim.task_spec import (
    TaskSpec,
    RoleSpec,
    Predicate,
    MilestoneSpec,
    canonical_hash,
)
from embodichain.lab.task_program.integrations.evidence.handover_evidence import (
    HandOverEvidenceBinding,
)
from ._evaluation import evaluate_handover_evidence

_SOURCE = json.loads(
    (Path(__file__).parent / "data/handover_source_reference.json").read_text()
)


def _case(index, name="held", *, raw=False):
    case = deepcopy(_SOURCE["cases"][index])
    segment = case["runtime"]["segments"][0]
    meta = segment["metadata"]
    call = meta["runtime"]["calls"][0]
    descriptor = dict(
        segment_id=meta["program_segment_id"],
        segment_name=segment["name"],
        source_path=meta["program_segment_source_path"],
        invocation_id=call["invocation_id"],
    )
    operation = dict(
        operation_id="e4",
        object_id="object",
        source_resource="left",
        destination_resource="right",
        transfer=descriptor,
        boundary=descriptor,
        hold_condition=case["conditions"].get("hold"),
        hold_preset="hold" if case["with_hold"] else None,
    )
    predicate = Predicate(
        name, predicate_id="proof", arguments={"subject": "object", "task_type": "E4"}
    )
    spec = TaskSpec(
        roles=(RoleSpec("object"),),
        goal=(
            (predicate,)
            if case["final"]
            else (Predicate(name, predicate_id="final", arguments=predicate.arguments),)
        ),
        milestones=() if case["final"] else (MilestoneSpec("event", (predicate,)),),
    )
    binding = HandOverEvidenceBinding(
        canonical_hash(spec.to_dict()),
        canonical_hash(case["program"]),
        "proof",
        operation,
        "final" if case["final"] else "recorded",
        0.04,
        query_kind=name,
        evidence_format="raw_snapshot" if raw else "historical_record",
        episode_ids=(7,),
        program_fingerprint="a" * 64,
        object_native_uid="native_object",
        robot_uid="native_robot",
    )
    record = case["records"][name]
    evidence = {**case["runtime"], "episode_ids": [7], "record": record}
    if raw:
        evidence.pop("record")
        evidence["program_fingerprint"] = "a" * 64
        metrics = case["records"]["held"]["measurements"]
        boundary = metrics["boundary_frames"]
        evidence["boundary_frames"] = boundary
        part = metrics["motion_part"]
        channels = {
            "rigid.pose": metrics["object_pose"],
            "robot.tcp_pose:" + part: metrics["receiver_tcp_pose"],
        }
        frame = dict(
            timestamp=12.0,
            observation_revision=1,
            env_ids=[0],
            clock_domain="simulation_time",
            revision_kind="held_pose_acquisition",
            batches={
                name: dict(
                    evidence_id=name,
                    values=values,
                    value_dtype="torch.float64",
                    valid=[True],
                    acquisition_errors=[None],
                    timestamp=12.0,
                    observation_revision=1,
                    env_ids=[0],
                )
                for name, values in channels.items()
            },
        )
        evidence["held_poses"] = dict(
            schema_version="simulation.held_pose_observation/v1",
            program_fingerprint="a" * 64,
            object_id="object",
            object_native_uid="native_object",
            robot_uid="native_robot",
            episode_ids=[7],
            observation_kind=binding.observation_kind,
            boundary_frames=boundary,
            motion_parts=[part],
            step_dt=0.04,
            object_pose=metrics["object_pose"],
            tcp_poses={part: metrics["receiver_tcp_pose"]},
            pose_frame=frame,
        )
        if case["final"]:
            sample = deepcopy(metrics["fresh_grip"])
            query = sample["query"]
            sample.update(
                episode_ids=[7],
                boundary_frames=boundary,
                grip_acquisition=dict(
                    evidence_id="grip.bilateral",
                    values=[row["value"] for row in query["rows"]],
                    valid=[True],
                    acquisition_errors=[None],
                    env_ids=[0],
                    timestamp=query["timestamp"],
                    observation_revision=query["observation_revision"],
                    clock_domain="environment_step",
                    revision_kind="semantic_query_revision",
                ),
            )
            evidence["fresh_grip"] = sample
    return spec, binding, evidence, case["program"]


@pytest.mark.parametrize("index", range(4))
@pytest.mark.parametrize("raw", [False, True])
@pytest.mark.parametrize("name", ["handover_complete", "held", "orientation"])
def test_handover_source_receipts_retain_outcomes(index, raw, name):
    if name == "orientation" and not _SOURCE["cases"][index]["with_hold"]:
        pytest.skip("Source declares no orientation predicate without hold")
    spec, binding, evidence, program = _case(index, name, raw=raw)
    outcome = evaluate_handover_evidence(spec, binding, evidence, program=program)[0]
    assert outcome.status.value == "satisfied", outcome.reason


@pytest.mark.parametrize(
    "mutation",
    [
        "episode",
        "program",
        "robot",
        "boundary",
        "channel",
        "revision",
        "grip",
        "mix",
        "release",
    ],
)
def test_handover_rejects_cross_scope_or_missing_raw_facts(mutation):
    spec, binding, evidence, program = _case(3, raw=True)
    pose = evidence["held_poses"]
    if mutation == "episode":
        pose["episode_ids"] = [8]
    elif mutation == "program":
        pose["program_fingerprint"] = "b" * 64
    elif mutation == "robot":
        pose["robot_uid"] = "other"
    elif mutation == "boundary":
        pose["boundary_frames"] = [1]
    elif mutation == "channel":
        pose["pose_frame"]["batches"]["rigid.pose"]["valid"] = [False]
    elif mutation == "revision":
        pose["pose_frame"]["batches"]["rigid.pose"]["observation_revision"] = 2
    elif mutation == "grip":
        evidence["fresh_grip"]["grip_acquisition"]["valid"] = [False]
    elif mutation == "mix":
        evidence["record"] = {}
    else:
        call = evidence["segments"][0]["metadata"]["runtime"]["calls"][0]
        call["events"] = [
            event
            for event in call["events"]
            if event["kind"] != "phase_effect_gate_satisfied"
        ]
    assert (
        evaluate_handover_evidence(spec, binding, evidence, program=program)[
            0
        ].status.value
        == "unavailable"
    )


def test_handover_fresh_valid_empty_contacts_are_a_negative():
    spec, binding, evidence, program = _case(3, raw=True)
    sample = evidence["fresh_grip"]
    row = sample["query"]["rows"][0]
    row["raw_contacts"] = {"pairs": [], "normals": [], "impulses": []}
    row["value"] = False
    sample["grip_acquisition"]["values"] = [False]
    assert (
        evaluate_handover_evidence(spec, binding, evidence, program=program)[
            0
        ].status.value
        == "failed"
    )


def test_handover_common_report_preserves_exact_declarations():
    from ._evaluation import evaluate_physical_evidence_report

    spec, binding, evidence, program = _case(3, raw=True)
    normalized = HandOverEvidenceBinding.from_dict(binding.to_dict())
    assert normalized.to_dict() == binding.to_dict()
    reports = evaluate_physical_evidence_report(
        spec, [normalized], {"proof": evidence}, program=program, num_envs=1
    )
    assert reports[0].certificate_ready
    evidence["boundary_frames"] = [1]
    assert not evaluate_physical_evidence_report(
        spec, [normalized], {"proof": evidence}, program=program, num_envs=1
    )[0].certificate_ready


@pytest.mark.parametrize("index", range(4))
def test_held_observation_request_uses_exact_receipt_endpoints_and_revision(index):
    from embodichain.lab.task_program.integrations.evidence.observation_requests import (
        held_observation_request,
    )

    spec, binding, evidence, program = _case(index, "held", raw=False)
    request = held_observation_request(binding, evidence, program=program)
    metrics = evidence["record"]["measurements"]
    assert request["motion_parts"] == [metrics["motion_part"]]
    assert request["boundary_frames"] == metrics["boundary_frames"]
    if binding.observation_kind == "final":
        assert (
            request["grips"][0]["control_part"]
            == metrics["fresh_grip"]["query"]["control_part"]
        )
        assert (
            request["grips"][0]["observation_revision"]
            == metrics["fresh_grip"]["query"]["observation_revision"]
        )
    else:
        assert request["grips"] == []


@pytest.mark.parametrize(
    "quantifier,count,expected",
    [
        ("one", 0, "satisfied"),
        ("count", 1, "satisfied"),
        ("count", 2, "unavailable"),
        ("all", 0, "unavailable"),
    ],
)
def test_handover_retains_original_single_member_qualification(
    quantifier, count, expected
):
    from dataclasses import replace

    spec, binding, evidence, program = _case(3, raw=True)
    selected = TaskSpec(
        roles=(RoleSpec("object", quantifier=quantifier, count=count),),
        goal=spec.goal,
        milestones=spec.milestones,
    )
    declaration = replace(
        binding, task_spec_digest=canonical_hash(selected.to_dict()), binding_hash=""
    )
    assert (
        evaluate_handover_evidence(selected, declaration, evidence, program=program)[
            0
        ].status.value
        == expected
    )


@pytest.mark.parametrize(
    "shift,receipt_error,expected",
    [
        (0.0, 0.0, "satisfied"),
        (1.0, 0.0, "unavailable"),
        (1.0, 1.0, "failed"),
        (1.0, None, "unavailable"),
    ],
)
def test_world_target_is_checked_when_object_and_receiver_translate_together(
    shift, receipt_error, expected
):
    from dataclasses import replace

    spec, binding, evidence, program = _case(3, raw=True)
    operation = deepcopy(binding.operation)
    operation["hold_condition"].update(
        target_position=[0.0, 0.0, 0.0], local_axis=None, world_axis=None
    )
    binding = replace(binding, operation=operation, binding_hash="")
    held = evidence["held_poses"]
    for pose in [held["object_pose"], *held["tcp_poses"].values()]:
        pose[0][0][3] += shift
    for name, batch in held["pose_frame"]["batches"].items():
        batch["values"] = (
            held["object_pose"]
            if name == "rigid.pose"
            else held["tcp_poses"][name.split(":", 1)[1]]
        )
    measurements = evidence["segments"][0]["metadata"]["post_policies"][0]["result"][
        "measurements"
    ]
    if receipt_error is not None:
        measurements["position_error"] = [receipt_error]
    outcome = evaluate_handover_evidence(spec, binding, evidence, program=program)[0]
    assert outcome.status.value == expected, outcome.reason
