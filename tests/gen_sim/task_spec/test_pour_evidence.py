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
import subprocess
import sys

import pytest

from embodichain.gen_sim.task_spec import TaskSpec, canonical_hash, EvaluationStatus
from embodichain.gen_sim.task_spec.pour_evidence import (
    PourEvidenceBinding,
    evaluate_pour_evidence,
)

_SOURCE = json.loads(
    (
        Path(__file__).resolve().parents[2]
        / "gym/envs/task_program/data/pour_observation_source_reference.json"
    ).read_text()
)


def _case(name="nominal", *, past=False, raw=False):
    case = deepcopy(next(item for item in _SOURCE["cases"] if item["name"] == name))
    common = deepcopy(_SOURCE["common"])
    spec = TaskSpec.from_dict(common["spec"])
    program = common["program"]
    occurrences = tuple(
        dict(
            segment_id=segment["metadata"]["program_segment_id"],
            segment_name=segment["name"],
            source_path=segment["metadata"]["program_segment_source_path"],
            invocation_id=segment["metadata"]["runtime"]["calls"][0]["invocation_id"],
        )
        for segment in common["runtime"]["segments"]
    )
    ids = [item["segment_id"] for item in occurrences]
    binding = PourEvidenceBinding(
        task_spec_digest=canonical_hash(spec.to_dict()),
        program_digest=canonical_hash(program),
        predicate_id=spec.milestones[0].achieve[0].id if past else spec.goal[0].id,
        operation_id="step_01",
        object_id="cube",
        reference_id="region",
        resource_id="left",
        object_native_uid="cube",
        reference_native_uid="region",
        occurrences=occurrences,
        stage_segment_id=ids[1],
        pour_segment_id=ids[2],
        release_segment_id=ids[4],
        boundary_segment_id=ids[5],
        observation_kind="recorded" if past else "final",
        evidence_format="raw_snapshot" if raw else "historical_record",
        program_fingerprint="a" * 64 if raw else None,
        step_dt=0.04 if raw else None,
        episode_ids=(0,) if raw else None,
    )
    evidence = dict(record=case["record"], segments=common["runtime"]["segments"])
    if raw:
        metrics = case["record"]["measurements"]
        row = metrics["rows"][0]
        values = [
            row["maximum_tilt_rad"],
            row["restore_error_rad"],
            *row["anchor_world_m"],
            row["receiver_projection_distance_m"],
            row["receiver_top_clearance_m"],
        ]
        assert row["receiver_top_clearance_m"] >= 0
        channels = {}
        for channel, values, dtype in (
            ("pour.motion_geometry", [values], "torch.float64"),
            ("pour.anchor_projected_inside", [row["above_receiver"]], "torch.bool"),
        ):
            channels[channel] = dict(
                evidence_id=channel,
                values=values,
                value_dtype=dtype,
                valid=[True],
                acquisition_errors=[None],
                timestamp=1.3,
                observation_revision=1,
                env_ids=[0],
            )
        evidence = dict(
            segments=evidence["segments"],
            schema_version="simulation.pour_geometry_observation/v1",
            program_id=program["program_id"],
            program_fingerprint="a" * 64,
            object_id="cube",
            reference_id="region",
            object_native_uid="cube",
            reference_native_uid="region",
            episode_ids=[0],
            start_frames=metrics["start_frames"],
            end_frames=metrics["end_frames"],
            peak_frames=metrics["peak_frames"],
            boundary_frames=metrics["boundary_frames"],
            sample_counts=[6],
            recording_clock="environment_step",
            recording_step_dt=0.04,
            geometry_columns=[
                "maximum_tilt_rad",
                "restore_error_rad",
                "anchor_world_x_m",
                "anchor_world_y_m",
                "anchor_world_z_m",
                "receiver_projection_distance_m",
                "receiver_top_clearance_m",
            ],
            geometry_frame=dict(
                timestamp=1.3,
                observation_revision=1,
                env_ids=[0],
                clock_domain="simulation_time",
                revision_kind="recorded_geometry_acquisition",
                batches=channels,
            ),
        )
    return spec, binding, evidence, program


@pytest.mark.parametrize("name", [item["name"] for item in _SOURCE["cases"]])
@pytest.mark.parametrize("raw", [False, True])
@pytest.mark.parametrize("past", [False, True])
def test_pour_motion_retains_frozen_source_outcomes_for_current_goals_and_milestones(
    name, raw, past
):
    spec, binding, evidence, program = _case(name, past=past, raw=raw)
    expected = next(item for item in _SOURCE["cases"] if item["name"] == name)[
        "record"
    ]["accepted"][0]
    outcome = evaluate_pour_evidence(spec, binding, evidence, program=program)[0]
    assert outcome.status is (
        EvaluationStatus.SATISFIED if expected else EvaluationStatus.FAILED
    )
    assert outcome.observed is expected
    assert "fluid transfer is not established" in outcome.reason


@pytest.mark.parametrize("raw", [False, True])
@pytest.mark.parametrize("past", [False, True])
def test_explicit_single_count_preserves_pour_proof_without_rewriting_selectors(
    raw, past
):
    spec, binding, evidence, program = _case(raw=raw, past=past)
    original_spec, original_binding = spec.to_dict(), binding.to_dict()
    counted = replace(
        spec,
        roles=tuple(replace(role, quantifier="count", count=1) for role in spec.roles),
    )
    declared = replace(
        binding, task_spec_digest=canonical_hash(counted.to_dict()), binding_hash=""
    )
    expected = evaluate_pour_evidence(spec, binding, evidence, program=program)[0]
    outcome = evaluate_pour_evidence(counted, declared, evidence, program=program)[0]
    assert outcome.status is expected.status is EvaluationStatus.SATISFIED
    assert outcome.observed is expected.observed is True
    assert all(role.quantifier == "count" and role.count == 1 for role in counted.roles)
    assert counted.semantic_hash != spec.semantic_hash
    assert spec.to_dict() == original_spec and binding.to_dict() == original_binding


@pytest.mark.parametrize("role_index", [0, 1])
@pytest.mark.parametrize("quantifier,count", [("count", 2), ("all", 0)])
def test_pour_single_member_proof_cannot_certify_a_larger_selector(
    role_index, quantifier, count
):
    spec, binding, evidence, program = _case(raw=True)
    roles = list(spec.roles)
    roles[role_index] = replace(roles[role_index], quantifier=quantifier, count=count)
    selected = replace(spec, roles=tuple(roles))
    declared = replace(
        binding, task_spec_digest=canonical_hash(selected.to_dict()), binding_hash=""
    )
    outcome = evaluate_pour_evidence(selected, declared, evidence, program=program)[0]
    assert outcome.status is EvaluationStatus.UNAVAILABLE and outcome.observed is None


@pytest.mark.parametrize(
    "damage",
    [
        "missing_channel",
        "foreign_episode",
        "foreign_clock",
        "foreign_program",
        "unwritten_end",
        "foreign_invocation",
        "zero_commands",
        "missing_release",
        "historical_overlay",
    ],
)
def test_raw_pour_missing_or_foreign_acquisition_cannot_become_a_measured_negative(
    damage,
):
    spec, binding, evidence, program = _case(raw=True)
    if damage == "missing_channel":
        evidence["geometry_frame"]["batches"].pop("pour.motion_geometry")
    elif damage == "foreign_episode":
        evidence["episode_ids"] = [1]
    elif damage == "foreign_program":
        evidence["program_fingerprint"] = "b" * 64
    elif damage == "foreign_clock":
        evidence["recording_step_dt"] = 0.08
    elif damage == "unwritten_end":
        evidence["end_frames"][0] -= 1
    elif damage == "foreign_invocation":
        evidence["segments"][2]["metadata"]["runtime"]["calls"][0][
            "invocation_id"
        ] = "different"
    elif damage == "zero_commands":
        evidence["segments"][2]["metadata"]["runtime"]["calls"][0]["command_count"] = 0
    elif damage == "missing_release":
        evidence["segments"][4]["metadata"]["runtime"]["calls"][0]["effects"] = []
    else:
        evidence["record"] = {}
    assert (
        evaluate_pour_evidence(spec, binding, evidence, program=program)[0].status
        is EvaluationStatus.UNAVAILABLE
    )


def test_raw_pour_acquisition_invalidity_and_incomplete_milestone_remain_unavailable():
    spec, binding, evidence, program = _case(raw=True)
    evidence["geometry_frame"]["batches"]["pour.motion_geometry"].update(
        valid=[False], values=[None], acquisition_errors=["missing recorded window"]
    )
    outcome = evaluate_pour_evidence(spec, binding, evidence, program=program)[0]
    assert (
        outcome.status is EvaluationStatus.UNAVAILABLE
        and outcome.reason == "missing recorded window"
    )
    spec, binding, evidence, program = _case(raw=True, past=True)
    evidence["segments"][0]["successes"] = [False]
    assert (
        evaluate_pour_evidence(spec, binding, evidence, program=program)[0].status
        is EvaluationStatus.UNAVAILABLE
    )


def test_pour_motion_proof_does_not_certify_a_fluid_predicate_or_resign_drift():
    spec, binding, evidence, program = _case()
    changed = replace(spec, goal=(replace(spec.goal[0], name="poured"),))
    wrong = replace(
        binding, task_spec_digest=canonical_hash(changed.to_dict()), binding_hash=""
    )
    with pytest.raises(ValueError, match="not fluid transfer"):
        evaluate_pour_evidence(changed, wrong, evidence, program=program)
    binding.occurrences[0]["source_path"][-1] = 10
    with pytest.raises(ValueError, match="binding_hash"):
        evaluate_pour_evidence(spec, binding, evidence, program=program)


def test_pour_binding_cold_import_and_decode_do_not_load_numeric_or_simulation_owners():
    root = Path(__file__).resolve().parents[3]
    data = _case()[1].to_dict()
    script = f"import sys;sys.path.insert(0,{str(root)!r});from embodichain.gen_sim.task_spec.pour_evidence import PourEvidenceBinding;PourEvidenceBinding.from_dict({data!r});assert not any(name=='torch' or name=='numpy' or name.startswith('embodichain.lab') for name in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-I", "-c", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("raw", [False, True])
def test_pour_report_requires_independent_goal_and_milestone_proofs(raw):
    from embodichain.gen_sim.task_spec.evidence_report import (
        evaluate_physical_evidence_report,
    )

    spec, goal, current, program = _case(raw=raw)
    _, milestone, past, _ = _case(raw=raw, past=True)
    report = evaluate_physical_evidence_report(
        spec,
        [goal, milestone],
        {goal.predicate_id: current, milestone.predicate_id: past},
        program=program,
        num_envs=1,
    )[0]
    assert report.certificate_ready
    assert report.goal[0].status is EvaluationStatus.SATISFIED
    assert report.milestones[0].status is EvaluationStatus.SATISFIED
    report = evaluate_physical_evidence_report(
        spec, [goal], {goal.predicate_id: current}, program=program, num_envs=1
    )[0]
    assert report.milestones[0].status is EvaluationStatus.UNAVAILABLE
    assert not report.certificate_ready


def test_raw_pour_retains_valid_rows_when_another_geometry_channel_is_unavailable():
    spec, binding, evidence, program = _case(raw=True)
    binding = replace(binding, env_ids=(0, 1), episode_ids=(0, 0), binding_hash="")
    for key in (
        "episode_ids",
        "start_frames",
        "end_frames",
        "peak_frames",
        "boundary_frames",
        "sample_counts",
    ):
        evidence[key] *= 2
    frame = evidence["geometry_frame"]
    frame["env_ids"] = [0, 1]
    for batch in frame["batches"].values():
        batch["env_ids"] = [0, 1]
        for key in ("values", "valid", "acquisition_errors"):
            batch[key] = deepcopy(batch[key]) * 2
    for segment in evidence["segments"]:
        for key in ("active", "successes", "start_steps", "end_steps"):
            segment[key] *= 2
        metadata = segment["metadata"]
        metadata["validation"]["accepted_mask"] *= 2
        runtime = metadata["runtime"]
        runtime["env_ids"] = [0, 1]
        runtime["task_state"]["batch_size"] = 2
        for call in runtime["calls"]:
            for receipt in call["effects"]:
                receipt["effect_spec"]["env_ids"] = [0, 1]
                receipt["decision"]["success_mask"] *= 2
                for sample in receipt["evidence"].values():
                    sample["env_ids"] = [0, 1]
                    sample["valid_mask"] *= 2
                    for key in ("values", "object_to_endpoint"):
                        if key in sample:
                            sample[key] = deepcopy(sample[key]) * 2
    channel = frame["batches"]["pour.motion_geometry"]
    channel["values"][1] = None
    channel["valid"][1] = False
    channel["acquisition_errors"][1] = "recording row missing"
    outcomes = evaluate_pour_evidence(spec, binding, evidence, program=program)
    assert [outcome.status for outcome in outcomes] == [
        EvaluationStatus.SATISFIED,
        EvaluationStatus.UNAVAILABLE,
    ]
