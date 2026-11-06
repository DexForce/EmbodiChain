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
from pathlib import Path
import subprocess
import sys

import pytest

from embodichain.gen_sim.task_spec import (
    TaskSpec,
    Predicate,
    RoleSpec,
    canonical_hash,
    EvaluationStatus,
)
from embodichain.gen_sim.task_spec.condition_evidence import (
    ConditionEvidenceBinding,
    evaluate_condition_evidence,
)


def _case(kind="orientation"):
    predicate = (
        Predicate(
            "object_upright",
            predicate_id="state",
            arguments={"subject": "thing", "task_type": "E2"},
        )
        if kind == "orientation"
        else Predicate(
            "articulation_joint_near",
            predicate_id="state",
            arguments={"subject": "thing", "task_type": "E6", "target_state": "open"},
        )
    )
    spec = TaskSpec(roles=(RoleSpec("thing"),), goal=(predicate,))
    program = {"program": {"kind": "segment", "name": "observe"}}
    condition = (
        dict(
            entity="object",
            kind="upright",
            local_axis=[0.0, 0.0, 1.0],
            minimum_alignment=0.8660254037844386,
            axis_direction_free=False,
        )
        if kind == "orientation"
        else dict(
            entity="object",
            joint="slide_joint",
            target_state="open",
            target=0.2,
            tolerance=0.03,
        )
    )
    binding = ConditionEvidenceBinding(
        canonical_hash(spec.to_dict()),
        canonical_hash(program),
        predicate.id,
        "object",
        "native_object",
        kind,
        "final",
        condition,
        env_ids=(0, 1),
        episode_ids=(2, 3),
    )
    pose = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    rotated = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    values = [pose, rotated] if kind == "orientation" else [0.18, 0.1]
    evidence = dict(
        schema_version="simulation.state_observation/v1",
        entity_id="object",
        native_uid="native_object",
        observation_kind="final",
        episode_ids=[2, 3],
        state_acquisition=dict(
            evidence_id=(
                "rigid.pose" if kind == "orientation" else "articulation.joint_position"
            ),
            value_dtype="torch.float32",
            values=values,
            valid=[True, True],
            acquisition_errors=[None, None],
            timestamp=1.1,
            env_ids=[0, 1],
            observation_revision=1,
            clock_domain="simulation_time",
            revision_kind="collector_acquisition",
        ),
    )
    if kind == "joint_endpoint":
        evidence["joint"] = "slide_joint"
    return spec, binding, evidence, program


@pytest.mark.parametrize("kind", ["orientation", "joint_endpoint"])
def test_current_state_truth_is_independent_per_row_and_needs_no_program_success(kind):
    spec, binding, evidence, program = _case(kind)
    outcomes = evaluate_condition_evidence(spec, binding, evidence, program=program)
    assert [item.status for item in outcomes] == [
        EvaluationStatus.SATISFIED,
        EvaluationStatus.FAILED,
    ]
    evidence["state_acquisition"]["valid"][1] = False
    evidence["state_acquisition"]["values"][1] = None
    evidence["state_acquisition"]["acquisition_errors"][1] = "native state missing"
    outcomes = evaluate_condition_evidence(spec, binding, evidence, program=program)
    assert [item.status for item in outcomes] == [
        EvaluationStatus.SATISFIED,
        EvaluationStatus.UNAVAILABLE,
    ]
    assert outcomes[1].reason == "native state missing"


@pytest.mark.parametrize(
    "damage",
    [
        "missing_channel",
        "wrong_entity",
        "wrong_episode",
        "wrong_clock",
        "wrong_joint",
        "historical_overlay",
    ],
)
def test_state_scope_drift_and_missing_acquisition_never_certify(damage):
    spec, binding, evidence, program = _case("joint_endpoint")
    if damage == "missing_channel":
        evidence.pop("state_acquisition")
    elif damage == "wrong_entity":
        evidence["native_uid"] = "other"
    elif damage == "wrong_episode":
        evidence["episode_ids"] = [3, 3]
    elif damage == "wrong_clock":
        evidence["state_acquisition"]["clock_domain"] = "environment_step"
    elif damage == "wrong_joint":
        evidence["joint"] = "other"
    else:
        evidence["record"] = {}
    outcomes = evaluate_condition_evidence(spec, binding, evidence, program=program)
    assert all(item.status is EvaluationStatus.UNAVAILABLE for item in outcomes)


@pytest.mark.parametrize("kind", ["orientation", "joint_endpoint"])
def test_historical_state_claims_are_recomputed_without_adopting_raw_format(kind):
    spec, binding, raw, program = _case(kind)
    binding = replace(
        binding, evidence_format="historical_record", episode_ids=None, binding_hash=""
    )
    measurements = (
        dict(pose=raw["state_acquisition"]["values"], alignment=[1.0, 0.0])
        if kind == "orientation"
        else dict(joint_position=[0.18, 0.1], joint_target=0.2)
    )
    evidence = dict(
        record=dict(
            uid="object",
            part_id=None,
            accepted=[True, False],
            measurements=measurements,
        )
    )
    outcomes = evaluate_condition_evidence(spec, binding, evidence, program=program)
    assert [item.status for item in outcomes] == [
        EvaluationStatus.SATISFIED,
        EvaluationStatus.FAILED,
    ]
    evidence["record"]["accepted"][1] = True
    assert all(
        item.status is EvaluationStatus.UNAVAILABLE
        for item in evaluate_condition_evidence(
            spec, binding, evidence, program=program
        )
    )


def test_binding_instance_mutation_invalidates_original_identity():
    spec, binding, evidence, program = _case()
    binding.condition["minimum_alignment"] = 0.0
    with pytest.raises(ValueError, match="binding_hash"):
        evaluate_condition_evidence(spec, binding, evidence, program=program)


def test_state_binding_cold_import_and_normalization_do_not_load_simulation_or_numeric_backends():
    import embodichain

    root = Path(embodichain.__file__).resolve().parent.parent
    data = _case()[1].to_dict()
    script = f"import sys;sys.path.insert(0,{str(root)!r});from embodichain.gen_sim.task_spec.condition_evidence import ConditionEvidenceBinding;ConditionEvidenceBinding.from_dict({data!r});assert not any(name=='torch' or name=='numpy' or name.startswith('embodichain.lab') for name in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-I", "-c", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _recorded_case(kind="orientation"):
    from embodichain.gen_sim.task_spec import MilestoneSpec

    original, binding, evidence, _ = _case(kind)
    spec = TaskSpec(
        roles=original.roles,
        goal=(replace(original.goal[0], predicate_id="final_state"),),
        milestones=(MilestoneSpec("done", achieve=original.goal),),
    )
    path = ["program", "items", 0]
    item = dict(
        name="operation",
        post=[dict(kind="wait_stable", entity="object", preset="settled")],
    )
    policy = dict(
        policy_index=0,
        kind="wait_stable",
        source_path=path + ["post", 0],
        result_mask=[True, True],
    )
    metadata = dict(
        program_segment_source_path=path,
        runtime=dict(env_ids=[0, 1]),
        post_policies=[policy],
    )
    segment = dict(
        name="operation",
        active=[True, True],
        successes=[True, True],
        end_steps=[11, 11],
        metadata=metadata,
    )
    if kind == "orientation":
        # Both rows reached upright at the selected historical boundary.
        evidence["state_acquisition"]["values"][1] = deepcopy(
            evidence["state_acquisition"]["values"][0]
        )
        policy["result"] = dict(
            kind="upright",
            preset="settled",
            accepted_mask=[True, True],
            measurements=dict(alignment=[1.0, 1.0]),
        )
    else:
        target, tolerance = binding.condition["target"], binding.condition["tolerance"]
        minimum, maximum = target - tolerance, target + tolerance
        item["steps"] = dict(
            kind="invoke",
            call=dict(
                call_id="gen_sim.articulation_slide",
                arguments=dict(object="object", part=None, state="open"),
            ),
        )
        item["validators"] = [
            dict(
                kind="articulation_joint_position",
                articulation="object",
                joint="slide_joint",
                minimum_position=minimum,
                maximum_position=maximum,
            )
        ]
        result = dict(
            kind="articulation_joint_position",
            articulation_id="object",
            joint="slide_joint",
            source_path=path + ["validators", 0],
            env_ids=[0, 1],
            accepted_mask=[True, True],
            joint_position=[0.2, 0.19],
            minimum_position=minimum,
            maximum_position=maximum,
        )
        metadata["validation"] = dict(
            env_ids=[0, 1],
            accepted_mask=[True, True],
            validators=[
                dict(
                    validator_index=0,
                    kind=result["kind"],
                    source_path=result["source_path"],
                    result_mask=[True, True],
                    result=result,
                )
            ],
        )
        policy["result"] = dict(
            joint="slide_joint",
            evidence="joint_position_and_stability",
            stable_seconds=1.0,
            target=target,
            tolerance=tolerance,
            observed=[0.2, 0.19],
            accepted=[True, True],
            joint_observation=dict(
                evidence_id="articulation.joint_position",
                values=[0.2, 0.19],
                valid=[True, True],
                acquisition_errors=[None, None],
                timestamp=1.1,
                env_ids=[0, 1],
                observation_revision=11,
                revision_scope="policy_window_qpos_acquisition_sequence",
                clock="environment_step",
                start_step=1,
                end_step=11,
                started_at=0.1,
                step_dt=0.1,
            ),
        )
        evidence.pop("state_acquisition")
    if kind == "joint_endpoint":
        evidence["policy_result"] = policy["result"]
    program = dict(program=dict(items=[item]))
    binding = replace(
        binding,
        task_spec_digest=canonical_hash(spec.to_dict()),
        program_digest=canonical_hash(program),
        observation_kind="recorded",
        boundary_segment_id="operation",
        boundary_source_path=tuple(path),
        policy_segment_id="operation",
        policy_source_path=tuple(path),
        stability_preset="settled",
        required_execution=({"segment_id": "operation", "source_path": path},),
        binding_hash="",
    )
    evidence.update(
        observation_kind="recorded", boundary_frames=[11, 11], segments=[segment]
    )
    return spec, binding, evidence, program


@pytest.mark.parametrize("kind", ["orientation", "joint_endpoint"])
@pytest.mark.parametrize("historical", [False, True])
def test_recorded_state_requires_its_own_complete_execution_prefix(kind, historical):
    spec, binding, evidence, program = _recorded_case(kind)
    if historical:
        binding = replace(
            binding,
            evidence_format="historical_record",
            episode_ids=None,
            binding_hash="",
        )
        evidence.pop("state_acquisition", None)
        evidence["record"] = {"uid": "object"}
        if kind == "joint_endpoint":
            evidence["segments"][0]["metadata"]["post_policies"][0]["result"].pop(
                "joint_observation"
            )
    assert all(
        item.status is EvaluationStatus.SATISFIED
        for item in evaluate_condition_evidence(
            spec, binding, evidence, program=program
        )
    )
    evidence["segments"][0]["successes"][1] = False
    outcomes = evaluate_condition_evidence(spec, binding, evidence, program=program)
    assert [item.status for item in outcomes] == [
        EvaluationStatus.SATISFIED,
        EvaluationStatus.UNAVAILABLE,
    ]


@pytest.mark.parametrize(
    "damage",
    [
        "missing_channel",
        "wrong_revision",
        "wrong_time",
        "wrong_joint",
        "wrong_validator_path",
        "missing_mask",
        "wrong_boundary",
        "invalid_row",
    ],
)
def test_recorded_joint_rejects_incomplete_or_foreign_receipts(damage):
    spec, binding, evidence, program = _recorded_case("joint_endpoint")
    metadata = evidence["segments"][0]["metadata"]
    stable = metadata["post_policies"][0]["result"]
    if damage == "missing_channel":
        stable.pop("joint_observation")
    elif damage == "wrong_revision":
        stable["joint_observation"]["observation_revision"] = 1
    elif damage == "wrong_time":
        stable["joint_observation"]["timestamp"] = 0.1
    elif damage == "wrong_joint":
        stable["joint"] = "another_joint"
    elif damage == "wrong_validator_path":
        metadata["validation"]["validators"][0]["source_path"][2] = 1
    elif damage == "missing_mask":
        stable.pop("accepted")
    elif damage == "wrong_boundary":
        evidence["boundary_frames"] = [10, 10]
    else:
        stable["joint_observation"]["valid"][1] = False
        stable["joint_observation"]["acquisition_errors"][1] = "joint read missing"
    outcomes = evaluate_condition_evidence(spec, binding, evidence, program=program)
    assert outcomes[-1].status is EvaluationStatus.UNAVAILABLE
    assert outcomes[0].status is (
        EvaluationStatus.SATISFIED
        if damage == "invalid_row"
        else EvaluationStatus.UNAVAILABLE
    )


@pytest.mark.parametrize("kind", ["orientation", "joint_endpoint"])
def test_standard_reports_preserve_current_and_recorded_state_qualification(kind):
    from embodichain.gen_sim.task_spec.evidence_report import (
        evaluate_physical_evidence_report,
    )

    spec, binding, evidence, program = _recorded_case(kind)
    evidence["segments"][0]["successes"][1] = False
    reports = evaluate_physical_evidence_report(
        spec, [binding], {binding.predicate_id: evidence}, program=program, num_envs=2
    )
    assert reports[0].milestones[0].status is EvaluationStatus.SATISFIED
    assert reports[1].milestones[0].status is EvaluationStatus.UNAVAILABLE
    spec, binding, evidence, program = _case(kind)
    reports = evaluate_physical_evidence_report(
        spec,
        [binding.to_dict()],
        {binding.predicate_id: evidence},
        program=program,
        num_envs=2,
    )
    assert reports[0].goal[0].status is EvaluationStatus.SATISFIED
    assert reports[1].goal[0].status is EvaluationStatus.FAILED


def test_final_joint_replays_native_tensor_rounding_at_tolerance_boundary():
    import torch

    spec, binding, evidence, program = _case("joint_endpoint")
    binding = replace(
        binding, condition={**binding.condition, "tolerance": 0.02}, binding_hash=""
    )
    evidence["state_acquisition"]["values"] = [0.18, 0.22]
    expected = ((torch.tensor([0.18, 0.22]) - 0.2).abs() <= 0.02).tolist()
    actual = evaluate_condition_evidence(spec, binding, evidence, program=program)
    assert [item.observed for item in actual] == expected
    assert abs(0.18 - 0.2) > 0.02  # Python scalar arithmetic would regress this row.
