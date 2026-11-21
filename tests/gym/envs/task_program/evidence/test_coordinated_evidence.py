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
from embodichain.lab.task_program.integrations.evidence.coordinated_evidence import (
    CoordinatedHoldEvidenceBinding,
)
from ._evaluation import evaluate_coordinated_hold_evidence
from embodichain.lab.task_program.integrations.evidence._coordinated_evidence import (
    record,
)

_SOURCE = json.loads(
    (Path(__file__).parent / "data/coordinated_source_reference.json").read_text()
)


def _case(final=True, raw=False):
    case = deepcopy(_SOURCE["case"])
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
        operation_id="e5",
        object_id="object",
        resources=dict(left="left", right="right"),
        boundary=descriptor,
        hold_condition=case["conditions"]["hold"],
        hold_preset="hold",
    )
    predicate = Predicate(
        "held_by_both_grippers",
        predicate_id="proof",
        arguments=dict(subject="object", task_type="E5"),
    )
    spec = TaskSpec(
        roles=(RoleSpec("object"),),
        goal=(
            (predicate,)
            if final
            else (
                Predicate(
                    "held_by_both_grippers",
                    predicate_id="final",
                    arguments=predicate.arguments,
                ),
            )
        ),
        milestones=() if final else (MilestoneSpec("event", (predicate,)),),
    )
    binding = CoordinatedHoldEvidenceBinding(
        canonical_hash(spec.to_dict()),
        canonical_hash(case["program"]),
        "proof",
        operation,
        "final" if final else "recorded",
        0.04,
        evidence_format="raw_snapshot" if raw else "historical_record",
        episode_ids=(7,),
        program_fingerprint="a" * 64,
        object_native_uid="native_object",
        robot_uid="native_robot",
    )
    original = case["records"][str(final)]
    metrics = original["measurements"]
    evidence = {**case["runtime"], "record": original, "episode_ids": [7]}
    if raw:
        evidence.pop("record")
        evidence.update(
            program_fingerprint="a" * 64, boundary_frames=metrics["boundary_frames"]
        )
        channels = {
            "rigid.pose": metrics["object_pose"],
            **{
                "robot.tcp_pose:" + part: pose
                for part, pose in metrics["tcp_poses"].items()
            },
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
            observation_kind="final",
            boundary_frames=metrics["boundary_frames"],
            motion_parts=["left_arm", "right_arm"],
            step_dt=0.04,
            object_pose=metrics["object_pose"],
            tcp_poses=metrics["tcp_poses"],
            recorded_acquisition=dict(
                field_ids={"pose": "frozen.rigid.pose", "qpos": "frozen.robot.qpos"},
                env_ids=[0],
                episode_ids=[7],
                frame_indices=list(metrics["boundary_frames"]),
                observation_revisions=[1],
                timestamps=[12.0],
                clock_domain="simulation_time",
            ),
            pose_frame=frame,
        )
        evidence["fresh_grips"] = deepcopy(metrics["fresh_grips"])
        for sample in evidence["fresh_grips"].values():
            query = sample["query"]
            sample.update(
                episode_ids=[7],
                boundary_frames=metrics["boundary_frames"],
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
    return spec, binding, evidence, case["program"]


@pytest.mark.parametrize("final", [False, True])
@pytest.mark.parametrize("raw", [False, True])
def test_source_dual_attachment_and_hold_outcomes(final, raw):
    spec, binding, evidence, program = _case(final, raw)
    outcome = evaluate_coordinated_hold_evidence(
        spec, binding, evidence, program=program
    )[0]
    assert outcome.status.value == "satisfied", outcome.reason
    original = _SOURCE["case"]["records"][str(final)]
    metrics = original["measurements"]
    computed = record(
        binding.operation,
        _SOURCE["case"]["runtime"],
        program,
        1,
        0.04,
        final,
        metrics["object_pose"],
        metrics["tcp_poses"],
        metrics["fresh_grips"],
    )
    computed.pop("_normalized_facts")
    assert canonical_hash(computed) == canonical_hash(
        {key: value for key, value in original.items() if key != "accepted"}
    )
    assert outcome.observed == original["accepted"][0]


@pytest.mark.parametrize(
    "damage",
    [
        "episode",
        "shared_pads",
        "missing_hand",
        "recorded_pose",
        "boundary",
        "world_target",
        "slot",
    ],
)
def test_missing_or_cross_participant_evidence_is_unavailable(damage):
    spec, binding, evidence, program = _case(raw=True)
    if damage == "episode":
        evidence["fresh_grips"]["left"]["episode_ids"] = [8]
    elif damage == "shared_pads":
        evidence["fresh_grips"]["left"]["pad_actor_ids"] = deepcopy(
            evidence["fresh_grips"]["right"]["pad_actor_ids"]
        )
    elif damage == "missing_hand":
        del evidence["fresh_grips"]["left"]
    elif damage == "recorded_pose":
        evidence["held_poses"]["observation_kind"] = "recorded"
    elif damage == "boundary":
        evidence["boundary_frames"] = [1]
    elif damage == "world_target":
        evidence["segments"][0]["metadata"]["post_policies"][0]["result"][
            "measurements"
        ]["position_error"] = [0.01]
    else:
        evidence["segments"][0]["metadata"]["runtime"]["calls"][0]["effects"][0][
            "effect_spec"
        ]["state_expectations"][0]["slot_id"] = "other"
    assert (
        evaluate_coordinated_hold_evidence(spec, binding, evidence, program=program)[
            0
        ].status.value
        == "unavailable"
    )


def test_valid_loss_of_one_grip_is_negative():
    spec, binding, evidence, program = _case(raw=True)
    sample = evidence["fresh_grips"]["left"]
    row = sample["query"]["rows"][0]
    row["raw_contacts"] = {"pairs": [], "normals": [], "impulses": []}
    row["value"] = False
    sample["grip_acquisition"]["values"] = [False]
    outcome = evaluate_coordinated_hold_evidence(
        spec, binding, evidence, program=program
    )[0]
    assert outcome.status.value == "failed", outcome.reason


def test_coordinated_common_report_preserves_exact_declarations():
    from ._evaluation import evaluate_physical_evidence_report

    spec, binding, evidence, program = _case(raw=True)
    normalized = CoordinatedHoldEvidenceBinding.from_dict(binding.to_dict())
    assert normalized.to_dict() == binding.to_dict()
    assert evaluate_physical_evidence_report(
        spec, [normalized], {"proof": evidence}, program=program, num_envs=1
    )[0].certificate_ready


def test_coordinated_request_keeps_fresh_current_grips_for_recorded_boundary():
    from embodichain.lab.task_program.integrations.evidence.observation_requests import (
        held_observation_request,
    )

    spec, binding, evidence, program = _case(final=False, raw=False)
    request = held_observation_request(binding, evidence, program=program)
    assert request["pose_observation_kind"] == "final"
    assert request["motion_parts"] == ["left_arm", "right_arm"]
    assert [item["control_part"] for item in request["grips"]] == [
        "left_eef",
        "right_eef",
    ]


@pytest.mark.parametrize(
    "quantifier,count,expected",
    [
        ("one", 0, "satisfied"),
        ("count", 1, "satisfied"),
        ("count", 2, "unavailable"),
        ("all", 0, "unavailable"),
    ],
)
def test_coordinated_hold_retains_original_single_member_qualification(
    quantifier, count, expected
):
    from dataclasses import replace

    spec, binding, evidence, program = _case(raw=True)
    selected = TaskSpec(
        roles=(RoleSpec("object", quantifier=quantifier, count=count),),
        goal=spec.goal,
        milestones=spec.milestones,
    )
    declaration = replace(
        binding, task_spec_digest=canonical_hash(selected.to_dict()), binding_hash=""
    )
    assert (
        evaluate_coordinated_hold_evidence(
            selected, declaration, evidence, program=program
        )[0].status.value
        == expected
    )


@pytest.mark.parametrize(
    "reported_alignment,expected",
    [(1.0, "unavailable"), (0.0, "failed"), (None, "unavailable")],
)
def test_rotating_both_hands_with_object_cannot_bypass_declared_axis(
    reported_alignment, expected
):
    import torch

    spec, binding, evidence, program = _case(raw=True)
    held = evidence["held_poses"]
    # A 90-degree world X rotation keeps both relative attachments and the target.
    rotation = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    held["object_pose"] = (
        rotation @ torch.tensor(held["object_pose"], dtype=torch.float64)
    ).tolist()
    held["tcp_poses"] = {
        part: (rotation @ torch.tensor(pose, dtype=torch.float64)).tolist()
        for part, pose in held["tcp_poses"].items()
    }
    for name, batch in held["pose_frame"]["batches"].items():
        batch["values"] = (
            held["object_pose"]
            if name == "rigid.pose"
            else held["tcp_poses"][name.split(":", 1)[1]]
        )
    measurements = evidence["segments"][0]["metadata"]["post_policies"][0]["result"][
        "measurements"
    ]
    if reported_alignment is None:
        measurements.pop("alignment")
    else:
        measurements["alignment"] = [reported_alignment]
    outcome = evaluate_coordinated_hold_evidence(
        spec, binding, evidence, program=program
    )[0]
    assert outcome.status.value == expected, outcome.reason
