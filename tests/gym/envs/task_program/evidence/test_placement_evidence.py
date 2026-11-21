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
import subprocess
import sys

import pytest

from embodichain.gen_sim.task_spec import (
    EvaluationStatus,
    Predicate,
    RoleSpec,
    TaskSpec,
    canonical_hash,
    evaluate_task_spec,
)
from embodichain.gen_sim.task_spec.evidence import (
    SupportedPlacementEvidenceBinding,
)


def _legacy_evidence():
    """Use the frozen source release receipt shape, without importing its frontend."""
    spec = TaskSpec(
        roles=(RoleSpec("item"), RoleSpec("surface", kind="surface")),
        goal=(
            Predicate(
                "semantic_goal",
                predicate_id="placed",
                arguments={
                    "subject": "item",
                    "reference": "surface",
                    "relation": "on",
                },
            ),
        ),
    )
    program = {"program": {"items": [{"name": "place"}, {"name": "park"}]}}
    pose = [
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ]
    receipt = {
        "boundary": {"kind": "terminal"},
        "timestamp": 0.08,
        "observation_revision": 2,
        "effect_spec": {
            "effect_kind": "release",
            "invocation_id": "place:0",
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
                    "clause_id": "constraint",
                    "expectation_id": "source",
                    "expected": False,
                    "evidence_kind": "constraint",
                },
                {
                    "clause_id": "pose",
                    "expectation_id": "source",
                    "expectation": "separated",
                },
            ],
        },
        "decision": {"success_mask": [True]},
        "evidence": {
            "constraint": {
                "values": [False],
                "valid_mask": [True],
                "env_ids": [0],
                "timestamp": 0.08,
                "observation_revision": 2,
            },
            "pose": {
                "object_to_endpoint": pose,
                "valid_mask": [True],
                "env_ids": [0],
                "timestamp": 0.08,
                "observation_revision": 2,
            },
        },
    }
    state = {"batch_size": 1, "held_objects": [], "coordinated_held_objects": []}
    native = {
        "scope": "native_contact_support",
        "entity": "cube",
        "reference": "pad",
        "env_ids": [0],
        "supported": [True],
        "object_pose": pose,
        "reference_pose": pose,
        "rows": [],
    }
    segments = [
        {
            "name": name,
            "end_steps": [index + 1],
            "metadata": {
                "program_segment_source_path": ["program", "items", index],
                "runtime": {
                    "env_ids": [0],
                    "task_state": deepcopy(state),
                    "calls": (
                        [
                            {
                                "invocation_id": "place:0",
                                "invocation_revision": 0,
                                "status": "completed",
                                "effects": [receipt],
                            }
                        ]
                        if index == 0
                        else []
                    ),
                },
                "post_policies": (
                    [
                        {
                            "kind": "wait_stable",
                            "policy_index": 0,
                            "source_path": ["program", "items", 1, "post", 0],
                            "result": {
                                "preset": "declared_support",
                                "measurements": {"native_support": deepcopy(native)},
                            },
                        }
                    ]
                    if index == 1
                    else []
                ),
            },
        }
        for index, name in enumerate(("place", "park"))
    ]
    binding = SupportedPlacementEvidenceBinding(
        task_spec_digest=canonical_hash(spec.to_dict()),
        program_digest=canonical_hash(program),
        predicate_id="placed",
        object_id="cube",
        reference_id="pad",
        resource_id="left",
        release_segment_id="place",
        release_invocation_id="place:0",
        release_source_path=("program", "items", 0),
        support_segment_id="park",
        stability_preset="declared_support",
        support_source_path=("program", "items", 1),
        boundary_segment_id="park",
        boundary_source_path=("program", "items", 1),
        evidence_scope="native_contact_support",
        observation_kind="final",
        evidence_format="historical_record",
        condition={
            "entity": "cube",
            "reference": "pad",
            "displacement": [0.0, 0.0, 0.0],
        },
    )
    evidence = {
        "segments": segments,
        "record": {
            "uid": "cube",
            "part_id": None,
            "accepted": [True],
            "measurements": {
                **deepcopy(native),
                "reference_uid": "pad",
                "relation": "on",
                "released": [True],
            },
        },
        "boundary_poses": {"cube": deepcopy(pose), "pad": deepcopy(pose)},
    }
    return spec, binding, evidence, program


def test_legacy_scope_keeps_measured_outcome_without_claiming_a_strong_window():
    spec, binding, evidence, program = _legacy_evidence()
    result = evaluate_supported_placement_evidence(
        spec, binding, evidence, program=program
    )[0]
    assert result.status is EvaluationStatus.SATISFIED
    assert "no strong window" in result.reason
    report = evaluate_task_spec(
        spec, {"placed": {"status": result.status.value, "value": result.observed}}
    )
    assert report.certificate_ready
    evidence["record"]["accepted"] = [False]
    evidence["record"]["measurements"]["supported"] = [False]
    result = evaluate_supported_placement_evidence(
        spec, binding, evidence, program=program
    )[0]
    assert result.status is EvaluationStatus.FAILED


@pytest.mark.parametrize(
    "damage",
    [
        "missing_release",
        "foreign_invocation",
        "foreign_source",
        "stale_pose",
        "unavailable_acquisition",
        "strong_in_legacy",
        "invalid_receipt_revision",
        "invalid_release_sample",
    ],
)
def test_plans_or_foreign_missing_physical_proofs_never_certify(damage):
    spec, binding, evidence, program = _legacy_evidence()
    call = evidence["segments"][0]["metadata"]["runtime"]["calls"][0]
    if damage == "missing_release":
        call["effects"] = []
    elif damage == "foreign_invocation":
        call["invocation_id"] = "another:0"
    elif damage == "foreign_source":
        evidence["segments"][0]["metadata"]["program_segment_source_path"][-1] = 4
    elif damage == "stale_pose":
        evidence["boundary_poses"]["cube"][0][0][3] = 0.1
    elif damage == "unavailable_acquisition":
        evidence["support_acquisition"] = {
            "env_ids": [0],
            "valid": [False],
            "acquisition_errors": ["query overflow"],
        }
    elif damage == "invalid_release_sample":
        call["effects"][0]["evidence"]["constraint"]["valid_mask"] = [False]
        evidence["record"]["accepted"] = [False]
        evidence["record"]["measurements"]["released"] = [False]
    elif damage == "invalid_receipt_revision":
        receipt = call["effects"][0]
        receipt["observation_revision"] = -1
        for sample in receipt["evidence"].values():
            sample["observation_revision"] = -1
    elif damage == "strong_in_legacy":
        evidence["record"]["measurements"]["support_contract_revision"] = 1
    result = evaluate_supported_placement_evidence(
        spec, binding, evidence, program=program
    )[0]
    assert result.status is EvaluationStatus.UNAVAILABLE
    assert result.observed is None
    assert not evaluate_task_spec(
        spec, {"placed": {"status": result.status.value, "reason": result.reason}}
    ).certificate_ready


def test_binding_instance_owns_data_and_rejects_stale_nested_identity():
    spec, binding, evidence, program = _legacy_evidence()
    owned = SupportedPlacementEvidenceBinding.from_dict(binding)
    binding.condition["displacement"][0] = 0.1
    assert owned.condition["displacement"] == [0.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="binding_hash"):
        evaluate_supported_placement_evidence(spec, binding, evidence, program=program)


@pytest.mark.parametrize("strong", [False, True])
def test_evidence_cold_import_does_not_load_numeric_or_simulation_backends(strong):
    import embodichain

    root = Path(embodichain.__file__).resolve().parent.parent
    binding = (_strong_evidence() if strong else _legacy_evidence())[1].to_dict()
    source = f"import sys; sys.path.insert(0, {str(root)!r}); from embodichain.gen_sim.task_spec.evidence import SupportedPlacementEvidenceBinding; SupportedPlacementEvidenceBinding.from_dict({binding!r}); assert not any(name == 'torch' or name == 'numpy' or name.startswith('embodichain.lab') for name in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-I", "-c", source], text=True, capture_output=True, timeout=30
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _sampled_window_evidence():
    spec, original, evidence, program = _legacy_evidence()
    # Normalized frozen B0 StabilityConstraint payload; no Lab config import.
    condition = {
        "entity": "cube",
        "kind": "supported_placement",
        "duration": 0.08,
        "timeout": 12.0,
        "reference": "pad",
        "local_axis": None,
        "reference_axis": None,
        "displacement": [0.0, 0.0, 0.0],
        "target_position": None,
        "reference_half_extents": None,
        "object_bottom": 0.0,
        "reference_top": 0.0,
        "position_tolerance": 0.04,
        "minimum_alignment": 0.8660254037844386,
        "axis_direction_free": False,
        "support_tolerance": 0.01,
        "translation_drift": 0.02,
        "rotation_drift": 0.1,
        "motion_parts": [],
        "world_axis": None,
        "support_contract_revision": None,
        "com_tolerance": None,
    }
    raw_binding = original.to_dict()
    raw_binding.update(
        evidence_scope="native_stable_support", condition=condition, binding_hash=""
    )
    binding = SupportedPlacementEvidenceBinding.from_dict(raw_binding)
    receipt = evidence["segments"][0]["metadata"]["runtime"]["calls"][0]["effects"][0]
    receipt["timestamp"] = 0.0
    for value in receipt["evidence"].values():
        value["timestamp"] = 0.0
    raw = {"pairs": [[2, 1]], "normals": [[0.0, 0.0, 1.0]], "impulses": [1.0]}
    timeline = {
        "errors": [],
        "step_dt": 0.04,
        "object_actors": {"cube": [[1]], "pad": [[2]]},
        "object_body_types": {"cube": "dynamic", "pad": "static"},
        "samples": [
            {
                "step": step,
                "time": step * 0.04,
                "dropped_contacts": 0,
                "rows": [deepcopy(raw)],
                "substeps": [
                    {
                        "dt": 0.01,
                        "offset": index * 0.01,
                        "dropped_contacts": 0,
                        "rows": [deepcopy(raw)],
                    }
                    for index in range(1, 5)
                ],
            }
            for step in (1, 2)
        ],
    }
    pose = evidence["boundary_poses"]["cube"]
    evidence["contact_timeline"] = timeline
    evidence["record"]["measurements"] = {
        "scope": "native_stable_support",
        "criterion": "native_stable_reference_support/v2",
        "entity": "cube",
        "reference": "pad",
        "reference_uid": "pad",
        "relation": "on",
        "released": [True],
        "observation_kind": "final",
        "placement_node_id": "place",
        "boundary_frames": [2],
        "start_step": 0,
        "end_step": 2,
        "step_dt": 0.04,
        "duration": 0.08,
        "release_timestamp": 0.0,
        "constraint": deepcopy(condition),
        "relative_poses": [deepcopy(pose[0]) for _ in range(3)],
        "timeline_digest": canonical_hash(timeline),
        "object_pose": deepcopy(pose),
        "reference_pose": deepcopy(pose),
    }
    return spec, binding, evidence, program


@pytest.mark.parametrize(
    "damage,status",
    [
        (None, EvaluationStatus.SATISFIED),
        ("empty", EvaluationStatus.UNAVAILABLE),
        ("partial", EvaluationStatus.UNAVAILABLE),
        ("external", EvaluationStatus.FAILED),
        ("drift", EvaluationStatus.FAILED),
        ("foreign_hash", EvaluationStatus.UNAVAILABLE),
    ],
)
def test_selected_v2_window_rechecks_coverage_load_and_relative_motion(damage, status):
    spec, binding, evidence, program = _sampled_window_evidence()
    timeline = evidence["contact_timeline"]
    metrics = evidence["record"]["measurements"]
    if damage == "empty":
        for substep in timeline["samples"][0]["substeps"]:
            substep["rows"] = [{"pairs": [], "normals": [], "impulses": []}]
    elif damage == "partial":
        timeline["samples"][0]["substeps"].pop()
    elif damage == "external":
        raw = timeline["samples"][0]["substeps"][0]["rows"][0]
        raw["pairs"].append([1, 3])
        raw["normals"].append([1.0, 0.0, 0.0])
        raw["impulses"].append(1.0)
        evidence["record"]["accepted"] = [False]
    elif damage == "drift":
        metrics["relative_poses"][-1][0][3] = 0.021
        evidence["record"]["accepted"] = [False]
    elif damage == "foreign_hash":
        metrics["timeline_digest"] = "0" * 64
    if damage != "foreign_hash":
        metrics["timeline_digest"] = canonical_hash(timeline)
    result = evaluate_supported_placement_evidence(
        spec, binding, evidence, program=program
    )[0]
    assert result.status is status, result.reason


def test_milestone_requires_the_projected_execution_prefix_in_addition_to_geometry():
    from dataclasses import replace
    from embodichain.gen_sim.task_spec import MilestoneSpec

    spec, binding, evidence, program = _legacy_evidence()
    milestone = replace(spec.goal[0], predicate_id="placed_milestone")
    spec = replace(spec, milestones=(MilestoneSpec("place_once", (milestone,)),))
    raw = binding.to_dict()
    raw.update(
        task_spec_digest=canonical_hash(spec.to_dict()),
        predicate_id="placed_milestone",
        observation_kind="recorded",
        binding_hash="",
    )
    binding = SupportedPlacementEvidenceBinding.from_dict(raw)
    with pytest.raises(ValueError, match="execution occurrence prefix"):
        evaluate_supported_placement_evidence(spec, binding, evidence, program=program)
    raw["required_execution"] = [
        {"segment_id": name, "source_path": ["program", "items", index]}
        for index, name in enumerate(("place", "park"))
    ]
    binding = SupportedPlacementEvidenceBinding.from_dict(raw)
    evidence["record"]["measurements"]["boundary_frames"] = [2]
    for segment in evidence["segments"]:
        segment.update(active=[True], successes=[True])
    result = evaluate_supported_placement_evidence(
        spec, binding, evidence, program=program
    )[0]
    assert result.status is EvaluationStatus.SATISFIED, result.reason
    evidence["segments"][0]["successes"] = [False]
    result = evaluate_supported_placement_evidence(
        spec, binding, evidence, program=program
    )[0]
    assert result.status is EvaluationStatus.UNAVAILABLE


def test_raw_lab_snapshot_leaves_release_and_acceptance_composition_to_taskspec():
    spec, binding, evidence, program = _legacy_evidence()
    raw_binding = binding.to_dict()
    raw_binding.update(episode_ids=[3], evidence_format="raw_snapshot", binding_hash="")
    binding = SupportedPlacementEvidenceBinding.from_dict(raw_binding)
    evidence["episode_ids"] = [3]
    native = evidence.pop("record")["measurements"]
    for name in ("released", "reference_uid", "relation"):
        native.pop(name)
    evidence["native_support"] = native
    evidence["support_acquisition"] = dict(
        evidence_id="support.com_patch",
        values=[True],
        valid=[True],
        acquisition_errors=[None],
        timestamp=1.2,
        env_ids=[0],
        observation_revision=7,
        clock_domain="simulation_time",
    )
    result = evaluate_supported_placement_evidence(
        spec, binding, evidence, program=program
    )[0]
    assert result.status is EvaluationStatus.SATISFIED
    assert "released" not in evidence["native_support"]
    assert "record" not in evidence
    evidence["segments"][0]["metadata"]["runtime"]["calls"][0]["effects"] = []
    assert (
        evaluate_supported_placement_evidence(spec, binding, evidence, program=program)[
            0
        ].status
        is EvaluationStatus.UNAVAILABLE
    )


def test_report_requires_recorded_milestones_and_keeps_final_failure_independent():
    from dataclasses import replace
    from embodichain.gen_sim.task_spec import MilestoneSpec, TemporalCondition
    from ._evaluation import (
        evaluate_physical_evidence_report as evaluate_placement_evidence_report,
    )

    spec, original, final, program = _legacy_evidence()
    achievement = replace(spec.goal[0], predicate_id="milestone_place")
    spec = replace(spec, milestones=(MilestoneSpec("place_once", (achievement,)),))
    raw = original.to_dict()
    raw.update(task_spec_digest=canonical_hash(spec.to_dict()), binding_hash="")
    goal_binding = SupportedPlacementEvidenceBinding.from_dict(raw)
    recorded = deepcopy(final)
    recorded["record"]["measurements"]["boundary_frames"] = [2]
    for segment in recorded["segments"]:
        segment.update(active=[True], successes=[True])
    raw.update(
        predicate_id="milestone_place",
        observation_kind="recorded",
        required_execution=[
            {"segment_id": name, "source_path": ["program", "items", index]}
            for index, name in enumerate(("place", "park"))
        ],
    )
    milestone_binding = SupportedPlacementEvidenceBinding.from_dict(raw)
    report = evaluate_placement_evidence_report(
        spec, [goal_binding], {"placed": final}, program=program, num_envs=1
    )[0]
    assert not report.certificate_ready
    reports = {"placed": final, "milestone_place": recorded}
    bindings = [goal_binding, milestone_binding]
    assert evaluate_placement_evidence_report(
        spec, bindings, reports, program=program, num_envs=1
    )[0].certificate_ready
    final["record"]["accepted"] = [False]
    final["record"]["measurements"]["supported"] = [False]
    report = evaluate_placement_evidence_report(
        spec, bindings, reports, program=program, num_envs=1
    )[0]
    assert report.goal[0].status is EvaluationStatus.FAILED
    assert report.milestones[0].status is EvaluationStatus.SATISFIED
    assert not report.certificate_ready
    final["record"]["accepted"] = [True]
    final["record"]["measurements"]["supported"] = [True]
    temporal = replace(
        spec,
        temporal=(
            TemporalCondition(
                Predicate("contact", observation="contact"), within_steps=1
            ),
        ),
    )
    rebound = []
    for binding in bindings:
        values = binding.to_dict()
        values.update(
            task_spec_digest=canonical_hash(temporal.to_dict()), binding_hash=""
        )
        rebound.append(SupportedPlacementEvidenceBinding.from_dict(values))
    report = evaluate_placement_evidence_report(
        temporal, rebound, reports, program=program, num_envs=1
    )[0]
    assert report.temporal[0].status is EvaluationStatus.UNAVAILABLE
    assert not report.certificate_ready


def test_acquisition_exception_marker_keeps_the_unavailable_reason():
    spec, binding, _, program = _legacy_evidence()
    result = evaluate_supported_placement_evidence(
        spec,
        binding,
        {"unavailable": "Rigid trajectory recording is unavailable."},
        program=program,
    )[0]
    assert result.status is EvaluationStatus.UNAVAILABLE
    assert result.reason == "Rigid trajectory recording is unavailable."


@pytest.mark.parametrize("family", ["E4", "E5"])
def test_single_placement_proof_cannot_replace_transfer_or_coordinated_transport(
    family,
):
    from dataclasses import replace

    spec, binding, evidence, program = _legacy_evidence()
    predicate = spec.goal[0]
    spec = replace(
        spec,
        goal=(
            replace(
                predicate,
                arguments={
                    **predicate.arguments,
                    "task_type": family,
                    "terminal_behavior": "place",
                },
            ),
        ),
    )
    binding = replace(
        binding, task_spec_digest=canonical_hash(spec.to_dict()), binding_hash=""
    )
    with pytest.raises(ValueError, match="task-family evidence"):
        evaluate_supported_placement_evidence(spec, binding, evidence, program=program)


@pytest.mark.parametrize(
    "damage",
    [
        "missing_channel",
        "missing_values",
        "foreign_channel",
        "negative_timestamp",
        "foreign_clock",
        "contradictory_values",
        "add_record",
        "change_format",
    ],
)
def test_raw_support_contract_cannot_omit_acquisition_or_downgrade_to_history(damage):
    from dataclasses import replace

    spec, old, evidence, program = _legacy_evidence()
    binding = replace(
        old, evidence_format="raw_snapshot", episode_ids=(0,), binding_hash=""
    )
    historical = evidence.pop("record")
    native = deepcopy(historical["measurements"])
    for name in ("released", "reference_uid", "relation"):
        native.pop(name)
    evidence.update(
        native_support=native,
        episode_ids=[0],
        support_acquisition=dict(
            evidence_id="support.com_patch",
            values=[True],
            valid=[True],
            acquisition_errors=[None],
            timestamp=1.2,
            env_ids=[0],
            observation_revision=7,
            clock_domain="simulation_time",
        ),
    )
    if damage == "missing_channel":
        evidence.pop("support_acquisition")
    elif damage == "missing_values":
        evidence["support_acquisition"].pop("values")
    elif damage == "foreign_channel":
        evidence["support_acquisition"]["evidence_id"] = "other"
    elif damage == "negative_timestamp":
        evidence["support_acquisition"]["timestamp"] = -1.0
    elif damage == "foreign_clock":
        evidence["support_acquisition"]["clock_domain"] = "environment_step"
    elif damage == "contradictory_values":
        evidence["support_acquisition"]["values"] = [False]
    elif damage == "add_record":
        evidence["record"] = historical
    else:
        raw = binding.to_dict()
        raw["evidence_format"] = "historical_record"
        with pytest.raises(ValueError, match="binding_hash"):
            SupportedPlacementEvidenceBinding.from_dict(raw)
        return
    result = evaluate_supported_placement_evidence(
        spec, binding, evidence, program=program
    )[0]
    assert result.status is EvaluationStatus.UNAVAILABLE


def test_old_binding_version_requires_explicit_reconstruction():
    _, binding, _, _ = _legacy_evidence()
    raw = binding.to_dict()
    raw.pop("evidence_format")
    raw["schema_version"] = "task_spec.placement_evidence_binding/v1"
    raw["binding_hash"] = canonical_hash(
        {key: value for key, value in raw.items() if key != "binding_hash"}
    )
    with pytest.raises(ValueError, match="rebuild bindings"):
        SupportedPlacementEvidenceBinding.from_dict(raw)


def test_report_cannot_mix_new_runtime_and_historical_input_contracts():
    from dataclasses import replace
    from embodichain.gen_sim.task_spec import MilestoneSpec
    from ._evaluation import (
        evaluate_physical_evidence_report as evaluate_placement_evidence_report,
    )

    spec, original, _, program = _legacy_evidence()
    milestone = replace(spec.goal[0], predicate_id="past_place")
    spec = replace(spec, milestones=(MilestoneSpec("past", (milestone,)),))
    common = dict(
        task_spec_digest=canonical_hash(spec.to_dict()),
        episode_ids=(0,),
        binding_hash="",
    )
    goal = replace(original, evidence_format="raw_snapshot", **common)
    past = replace(
        original,
        predicate_id="past_place",
        observation_kind="recorded",
        required_execution=(
            {"segment_id": "park", "source_path": ["program", "items", 1]},
        ),
        **common,
    )
    with pytest.raises(ValueError, match="cannot mix raw and historical"):
        evaluate_placement_evidence_report(
            spec, (goal, past), {}, program=program, num_envs=1
        )


def test_raw_timeline_cannot_borrow_the_native_com_patch_contract():
    from dataclasses import replace

    _, binding, _, _ = _sampled_window_evidence()
    with pytest.raises(ValueError, match="Raw sampled-timeline"):
        replace(
            binding, evidence_format="raw_snapshot", episode_ids=(0,), binding_hash=""
        )


def _strong_evidence(*, region=False, historical=False):
    from dataclasses import replace
    import math

    spec, old, evidence, program = _legacy_evidence()
    relation = "inside" if region else "on"
    if region:
        predicate = spec.goal[0]
        spec = replace(
            spec,
            goal=(
                replace(
                    predicate, arguments={**predicate.arguments, "relation": relation}
                ),
            ),
        )
    condition = dict(
        entity="cube",
        reference="pad",
        kind="supported_region" if region else "supported_placement",
        duration=0.12,
        timeout=1.0,
        translation_drift=0.02,
        rotation_drift=math.pi / 18,
        support_contract_revision=1,
    )
    if not region:
        condition["displacement"] = [0.0, 0.0, 0.0]
    binding = replace(
        old,
        task_spec_digest=canonical_hash(spec.to_dict()),
        evidence_scope="native_supported_placement",
        evidence_format="historical_record" if historical else "raw_snapshot",
        condition=condition,
        episode_ids=(0,),
        binding_hash="",
    )
    call = evidence["segments"][0]["metadata"]["runtime"]["calls"][0]
    receipt = call["effects"][0]
    receipt["verification_id"] = 3
    call["active_plan_attempt_generation"] = 0
    call["plan_attempts"] = [
        dict(
            invocation_id="place:0",
            invocation_revision=0,
            invocation_index=0,
            attempt_generation=0,
            planned_at=0.0,
        )
    ]
    call["events"] = [
        dict(
            invocation_id="place:0",
            invocation_revision=0,
            invocation_index=0,
            timestamp=time,
            kind=kind,
            env_mask=[True],
            **extra,
        )
        for kind, time, extra in [
            ("trajectory_segment_entered", 0.04, {"segment_name": "release"}),
            ("skill_completed", 0.08, {}),
        ]
    ]
    points = [
        [-0.01, -0.01, 0.0],
        [0.01, -0.01, 0.0],
        [0.01, 0.01, 0.0],
        [-0.01, 0.01, 0.0],
    ]
    native = evidence["record"]["measurements"]
    native.pop("reference_uid")
    native.pop("relation")
    native.pop("released")
    native.update(
        support_contract_revision=1,
        clock="environment_step",
        observed_at=0.2,
        observed_step=5,
        step_dt=0.04,
        episode_generations=[0],
        actor_pairs=[
            dict(
                object_uid="cube",
                reference_uid="pad",
                object_actor=1,
                reference_actor=2,
                **{
                    kind
                    + "_info": dict(
                        actor_id=index,
                        env_id=0,
                        link_name=None,
                        path="/Arena/" + uid,
                        arena_name="Arena",
                    )
                    for kind, index, uid in [
                        ("object", 1, "cube"),
                        ("reference", 2, "pad"),
                    ]
                },
            )
        ],
    )
    native["rows"] = [
        dict(
            contacts_world=points,
            center_of_mass_world=[0.0, 0.0, 0.02],
            com_patch_distance_m=0.0,
            com_tolerance_m=0.003,
            upward_impulse=1.0,
            raw_contacts=dict(
                user_ids=[[2, 1]] * 4,
                position=points,
                normal=[[0.0, 0.0, 1.0]] * 4,
                impulse=[0.25] * 4,
            ),
            sample=dict(
                timestamp=0.2,
                control_step=5,
                substep=4,
                serial=8,
                native_revision=8,
                topology_revision=0,
                episode_generation=0,
            ),
            retained=False,
        )
    ]
    segment = evidence["segments"][1]
    segment["end_steps"] = [5]
    trace = segment["metadata"]["post_policies"][0]
    trace["result_mask"] = [True]
    trace["result"].update(
        kind=condition["kind"],
        required_seconds=0.12,
        accepted_mask=[True],
        stable_seconds=[0.12],
        start_step=2,
        end_step=5,
        clock_started_at=0.08,
        clock_ended_at=0.2,
        support_contract_revision=1,
        entity="cube",
        reference="pad",
    )
    trace["result"]["measurements"]["native_support"] = deepcopy(native)
    raw = dict(
        segments=evidence["segments"],
        native_support=deepcopy(native),
        boundary_poses=evidence["boundary_poses"],
        episode_ids=[0],
    )
    if not historical:
        raw["support_acquisition"] = dict(
            evidence_id="support.com_patch",
            values=[True],
            valid=[True],
            acquisition_errors=[None],
            timestamp=0.2,
            env_ids=[0],
            observation_revision=10,
            clock_domain="environment_step",
        )
    if region:
        landing = dict(
            scope="measured_planar_landing_region",
            object_pose=deepcopy(native["object_pose"]),
            reference_pose=deepcopy(native["reference_pose"]),
            support_tolerance_m=0.05,
            penetration_tolerance_m=0.01,
            full_footprint_inside=[True],
            support_gap_m=[0.0],
            footprint_area_m2=[0.0004],
        )
        trace["result"]["measurements"].update(
            landing_region=deepcopy(landing), support_gap=[0.0]
        )
        raw.update(landing_region=landing, support_gap_m=[0.0])
    if historical:
        from embodichain.lab.task_program.integrations.evidence._release_evidence import (
            release_window,
        )

        metrics = dict(
            scope="native_supported_placement",
            support_contract_revision=1,
            placement_node_id="place",
            support_boundary_node_id="park",
            boundary_node_id="park",
            stability_preset="declared_support",
            observation_kind="final",
            reference_uid="pad",
            relation=relation,
            released=[True],
            release_window=release_window(call, receipt, 1),
            native_support=raw.pop("native_support"),
        )
        for key in ("landing_region", "support_gap_m"):
            if key in raw:
                metrics[key] = raw.pop(key)
        raw["record"] = dict(
            uid="cube", part_id=None, accepted=[True], measurements=metrics
        )
    return spec, binding, raw, program


@pytest.mark.parametrize("region", [False, True])
@pytest.mark.parametrize("historical", [False, True])
def test_strong_support_rechecks_raw_load_window_and_declared_footprint(
    region, historical
):
    spec, binding, evidence, program = _strong_evidence(
        region=region, historical=historical
    )
    result = evaluate_supported_placement_evidence(
        spec, binding, evidence, program=program
    )
    assert result[0].status is EvaluationStatus.SATISFIED, result[0].reason


@pytest.mark.parametrize(
    "damage",
    [
        "foreign_actor",
        "superseded_plan",
        "missing_phase",
        "changed_patch",
        "wide_native_tolerance",
        "stale_boundary",
        "missing_raw",
        "wrong_acquisition_clock",
        "wrong_acquisition_time",
    ],
)
def test_strong_support_never_substitutes_derived_claims_for_qualified_sources(damage):
    spec, binding, evidence, program = _strong_evidence()
    if damage == "foreign_actor":
        evidence["native_support"]["actor_pairs"][0]["object_uid"] = "other"
    elif damage == "superseded_plan":
        evidence["segments"][0]["metadata"]["runtime"]["calls"][0][
            "active_plan_attempt_generation"
        ] = 1
    elif damage == "missing_phase":
        evidence["segments"][0]["metadata"]["runtime"]["calls"][0]["events"].pop(0)
    elif damage == "changed_patch":
        evidence["native_support"]["rows"][0]["upward_impulse"] = 2.0
    elif damage == "wide_native_tolerance":
        evidence["native_support"]["rows"][0]["com_tolerance_m"] = 0.1
    elif damage == "missing_raw":
        evidence["native_support"]["rows"][0].pop("raw_contacts")
    elif damage == "wrong_acquisition_clock":
        evidence["support_acquisition"]["clock_domain"] = "simulation_time"
    elif damage == "wrong_acquisition_time":
        evidence["support_acquisition"]["timestamp"] = 0.24
    else:
        evidence["boundary_poses"]["cube"][0][0][3] = 0.02
    result = evaluate_supported_placement_evidence(
        spec, binding, evidence, program=program
    )
    assert result[0].status is EvaluationStatus.UNAVAILABLE
    assert result[0].reason


def test_current_full_footprint_negative_is_not_replaced_by_historical_success():
    spec, binding, evidence, program = _strong_evidence(region=True)
    evidence["landing_region"]["full_footprint_inside"] = [False]
    result = evaluate_supported_placement_evidence(
        spec, binding, evidence, program=program
    )
    assert result[0].status is EvaluationStatus.FAILED, result[0].reason


def test_strong_stack_preserves_its_explicit_reference_axis_requirement():
    from dataclasses import replace

    spec, binding, evidence, program = _strong_evidence()
    condition = {
        **binding.condition,
        "kind": "stack",
        "displacement": None,
        "local_axis": [0.0, 0.0, 1.0],
        "reference_axis": [0.0, 0.0, 1.0],
        "reference_half_extents": [0.1, 0.1],
    }
    binding = replace(binding, condition=condition, binding_hash="")
    trace = evidence["segments"][1]["metadata"]["post_policies"][0]["result"]
    trace["kind"] = "stack"
    result = evaluate_supported_placement_evidence(
        spec, binding, evidence, program=program
    )[0]
    assert result.status is EvaluationStatus.SATISFIED, result.reason
    evidence["native_support"]["reference_pose"] = deepcopy(
        evidence["native_support"]["reference_pose"]
    )
    for poses in (
        evidence["boundary_poses"]["pad"],
        evidence["native_support"]["reference_pose"],
    ):
        poses[0][0][0] = -1.0
        poses[0][2][2] = -1.0
    result = evaluate_supported_placement_evidence(
        spec, binding, evidence, program=program
    )[0]
    assert result.status is EvaluationStatus.FAILED, result.reason


def test_recorded_full_footprint_uses_recomputed_seeded_pose_without_borrowing_history():
    from dataclasses import replace
    from embodichain.gen_sim.task_spec import MilestoneSpec

    spec, binding, evidence, program = _strong_evidence(region=True)
    predicate = replace(spec.goal[0], predicate_id="inside_recorded")
    spec = replace(spec, milestones=(MilestoneSpec("placed_once", (predicate,)),))
    binding = replace(
        binding,
        task_spec_digest=canonical_hash(spec.to_dict()),
        predicate_id=predicate.id,
        observation_kind="recorded",
        required_execution=tuple(
            {"segment_id": name, "source_path": ["program", "items", i]}
            for i, name in enumerate(("place", "park"))
        ),
        binding_hash="",
    )
    for segment in evidence["segments"]:
        segment.update(active=[True], successes=[True])
    evidence["boundary_frames"] = [5]
    result = evaluate_supported_placement_evidence(
        spec, binding, evidence, program=program
    )[0]
    assert result.status is EvaluationStatus.SATISFIED, result.reason
    # Below the existing pose-match tolerance, but the exact geometry record
    # must still agree with the original policy snapshot after recomputation.
    evidence["boundary_poses"]["cube"][0][0][3] = 0.00008
    evidence["landing_region"]["object_pose"][0][0][3] = 0.00008
    result = evaluate_supported_placement_evidence(
        spec, binding, evidence, program=program
    )[0]
    assert result.status is EvaluationStatus.UNAVAILABLE
    assert "own policy snapshot" in result.reason


@pytest.mark.parametrize("input_format", ["historical_record", "raw_snapshot"])
@pytest.mark.parametrize("supported", [False, True])
def test_final_legacy_support_survives_unexecuted_park_but_milestone_does_not(
    input_format, supported
):
    from dataclasses import replace
    from embodichain.gen_sim.task_spec import MilestoneSpec

    original_spec, original_binding, evidence, program = _legacy_evidence()
    milestone = replace(original_spec.goal[0], predicate_id="past_place")
    spec = replace(
        original_spec, milestones=(MilestoneSpec("placed_once", (milestone,)),)
    )
    binding = replace(
        original_binding,
        task_spec_digest=canonical_hash(spec.to_dict()),
        evidence_format=input_format,
        episode_ids=(0,) if input_format == "raw_snapshot" else None,
        boundary_segment_id="place",
        boundary_source_path=("program", "items", 0),
        binding_hash="",
    )
    # The release effect completed, then Place's post-policy failed and Park
    # never ran. B0 still evaluates current final support, independently of
    # program success; a full-group milestone remains unestablished.
    evidence["segments"] = evidence["segments"][:1]
    evidence["segments"][0].update(active=[True], successes=[False])
    evidence["record"]["accepted"] = [supported]
    evidence["record"]["measurements"]["supported"] = [supported]
    if input_format == "raw_snapshot":
        native = evidence.pop("record")["measurements"]
        for key in ("released", "reference_uid", "relation"):
            native.pop(key)
        evidence.update(
            native_support=native,
            episode_ids=[0],
            support_acquisition=dict(
                evidence_id="support.com_patch",
                values=[supported],
                valid=[True],
                acquisition_errors=[None],
                timestamp=0.2,
                env_ids=[0],
                observation_revision=3,
                clock_domain="simulation_time",
            ),
        )
    outcome = evaluate_supported_placement_evidence(
        spec, binding, evidence, program=program
    )[0]
    assert outcome.status is (
        EvaluationStatus.SATISFIED if supported else EvaluationStatus.FAILED
    )
    past = replace(
        binding,
        predicate_id="past_place",
        observation_kind="recorded",
        boundary_segment_id="park",
        boundary_source_path=("program", "items", 1),
        required_execution=tuple(
            {"segment_id": name, "source_path": ["program", "items", index]}
            for index, name in enumerate(("place", "park"))
        ),
        binding_hash="",
    )
    assert (
        evaluate_supported_placement_evidence(spec, past, evidence, program=program)[
            0
        ].status
        is EvaluationStatus.UNAVAILABLE
    )
    evidence["segments"][0]["metadata"]["runtime"]["calls"][0]["effects"] = []
    assert (
        evaluate_supported_placement_evidence(spec, binding, evidence, program=program)[
            0
        ].status
        is EvaluationStatus.UNAVAILABLE
    )


def test_terminal_placement_requires_the_bound_handover_before_receiver_release():
    import json
    from dataclasses import replace

    spec, binding, evidence, program = _legacy_evidence()
    predicate = Predicate(
        "semantic_goal",
        predicate_id="placed",
        arguments={**spec.goal[0].arguments, "task_type": "E4"},
    )
    spec = TaskSpec(roles=spec.roles, goal=(predicate,))
    source = json.loads(
        (Path(__file__).parent / "data/handover_source_reference.json").read_text()
    )["cases"][0]

    def swap(value):
        if isinstance(value, dict):
            return {key: swap(item) for key, item in value.items()}
        if isinstance(value, list):
            return [swap(item) for item in value]
        if isinstance(value, str):
            return {
                "object": "cube",
                "left": "right",
                "right": "left",
                "left_arm": "right_arm",
                "right_arm": "left_arm",
                "left_eef": "right_eef",
                "right_eef": "left_eef",
            }.get(value, value)
        return value

    source = swap(source)
    transfer = source["runtime"]["segments"][0]
    program["program"]["items"].insert(0, source["program"]["program"]["items"][0])
    program["program_id"] = source["program"]["program_id"]
    for index, segment in enumerate(evidence["segments"], 1):
        segment["metadata"]["program_segment_source_path"] = ["program", "items", index]
        segment["metadata"][
            "program_segment_id"
        ] = f"program:segment:$.program.items[{index}]"
    release = evidence["segments"][0]["metadata"]["runtime"]["calls"][0]["effects"][0]
    release["timestamp"] = 1.28
    for sample in release["evidence"].values():
        sample["timestamp"] = 1.28
    evidence["segments"].insert(0, transfer)
    evidence["record"]["group_id"] = "e4"
    meta = transfer["metadata"]
    call = meta["runtime"]["calls"][0]
    descriptor = dict(
        segment_id=meta["program_segment_id"],
        segment_name=transfer["name"],
        source_path=meta["program_segment_source_path"],
        invocation_id=call["invocation_id"],
    )
    operation = dict(
        operation_id="e4",
        object_id="cube",
        source_resource="right",
        destination_resource="left",
        transfer=descriptor,
        boundary=descriptor,
        hold_condition=None,
        hold_preset=None,
    )
    linked = replace(
        binding,
        task_spec_digest=canonical_hash(spec.to_dict()),
        program_digest=canonical_hash(program),
        release_source_path=("program", "items", 1),
        support_source_path=("program", "items", 2),
        boundary_source_path=("program", "items", 2),
        operation_evidence=dict(family="E4", operation=operation, step_dt=0.04),
        binding_hash="",
    )
    assert (
        linked.to_dict()["schema_version"] == "task_spec.placement_evidence_binding/v3"
    )
    outcome = evaluate_supported_placement_evidence(
        spec, linked, evidence, program=program
    )[0]
    assert outcome.status is EvaluationStatus.SATISFIED, outcome.reason
    release["timestamp"] = 0.08
    for sample in release["evidence"].values():
        sample["timestamp"] = 0.08
    assert (
        evaluate_supported_placement_evidence(spec, linked, evidence, program=program)[
            0
        ].status
        is EvaluationStatus.UNAVAILABLE
    )
    raw = linked.to_dict()
    raw["schema_version"] = "task_spec.placement_evidence_binding/v2"
    with pytest.raises(ValueError, match="rebuild bindings"):
        SupportedPlacementEvidenceBinding.from_dict(raw)


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("supported", EvaluationStatus.SATISFIED),
        ("overhang", EvaluationStatus.SATISFIED),
        ("table_support", EvaluationStatus.FAILED),
    ],
)
def test_coordinated_terminal_placement_preserves_source_load_path(mode, expected):
    import json
    from embodichain.gen_sim.task_spec import MilestoneSpec

    source = json.loads(
        (
            Path(__file__).parents[4]
            / "gen_sim/task_spec/data/transport_placement_source_reference.json"
        ).read_text()
    )
    case = next(value for value in source["cases"] if value["mode"] == mode)
    predicate = Predicate(
        "semantic_goal",
        predicate_id="placed",
        arguments=dict(
            subject="item", reference="surface", relation="on", task_type="E5"
        ),
    )
    spec = TaskSpec(
        roles=(RoleSpec("item"), RoleSpec("surface", kind="surface")),
        goal=(
            Predicate(
                "semantic_goal", predicate_id="final", arguments=predicate.arguments
            ),
        ),
        milestones=(MilestoneSpec("landing", (predicate,)),),
    )
    segment = case["runtime"]["segments"][0]
    segment["active"] = [True]
    meta = segment["metadata"]
    call = meta["runtime"]["calls"][0]
    descriptor = dict(
        segment_id=meta["program_segment_id"],
        segment_name=segment["name"],
        source_path=meta["program_segment_source_path"],
        invocation_id=call["invocation_id"],
    )
    operation = dict(
        operation_id="g",
        object_id="pan",
        reference_id="pad",
        target_id="target",
        resources=dict(left="left", right="right"),
        transport=descriptor,
    )
    binding = SupportedPlacementEvidenceBinding(
        task_spec_digest=canonical_hash(spec.to_dict()),
        program_digest=canonical_hash(case["program"]),
        predicate_id="placed",
        object_id="pan",
        reference_id="pad",
        resource_id="left",
        release_segment_id="n",
        release_invocation_id=call["invocation_id"],
        release_source_path=("program", "items", 0),
        support_segment_id="n",
        stability_preset="gen_sim.n.stable",
        support_source_path=("program", "items", 0),
        boundary_segment_id="n",
        boundary_source_path=("program", "items", 0),
        evidence_scope="native_stable_support",
        observation_kind="recorded",
        evidence_format="historical_record",
        condition=case["condition"],
        required_execution=(
            {"segment_id": "n", "source_path": ["program", "items", 0]},
        ),
        operation_evidence=dict(family="E5", operation=operation, step_dt=0.04),
    )
    evidence = {
        **case["runtime"],
        "record": case["record"],
        "contact_timeline": case["timeline"],
        "boundary_poses": case["boundary_poses"],
    }
    outcome = evaluate_supported_placement_evidence(
        spec, binding, evidence, program=case["program"]
    )[0]
    assert outcome.status is expected, outcome.reason
    # A later release and valid support do not replace the missing dual acquisition.
    evidence["contact_timeline"]["samples"][0]["rows"][0]["pairs"] = []
    assert (
        evaluate_supported_placement_evidence(
            spec, binding, evidence, program=case["program"]
        )[0].status
        is EvaluationStatus.UNAVAILABLE
    )


def _raw_sampled_window_evidence():
    from dataclasses import replace

    spec, binding, evidence, program = _sampled_window_evidence()
    timeline = evidence.pop("contact_timeline")
    metrics = evidence.pop("record")["measurements"]
    identity = dict(
        program_fingerprint="a" * 64,
        sensor_uid="contacts",
        step_dt=0.04,
        object_native_uids={"cube": "native_cube", "pad": "native_pad"},
        object_actors=timeline["object_actors"],
        object_body_types=timeline["object_body_types"],
        pad_actors={},
        topology_revision=3,
        timeline_episode_generations=[0],
    )
    samples = []
    for old in timeline["samples"]:
        subs = []
        for ordinal, sub in enumerate(old["substeps"], 1):
            subs.append(
                dict(
                    offset=sub["offset"],
                    dt=sub["dt"],
                    timestamp=(old["step"] - 1) * 0.04 + sub["offset"],
                    observation_revision=(old["step"] - 1) * 4 + ordinal,
                    native_revision=(old["step"] - 1) * 4 + ordinal,
                    topology_revision=3,
                    dropped_contacts=0,
                    available=True,
                    contacts=sub["rows"][0],
                )
            )
        samples.append(
            dict(
                step=old["step"],
                time=old["time"],
                episode_generation=0,
                native_revision=subs[-1]["native_revision"],
                topology_revision=3,
                dropped_contacts=0,
                available=True,
                contacts=old["rows"][0],
                substeps=subs,
                hand_opening={},
            )
        )
    poses = {
        name: [deepcopy(values[0]) for _ in range(3)]
        for name, values in evidence["boundary_poses"].items()
    }
    frame = dict(
        timestamp=10.0,
        observation_revision=1,
        env_ids=[0],
        clock_domain="simulation_time",
        revision_kind="recorded_contact_pose_acquisition",
        batches={
            "rigid.pose_window:"
            + name: dict(
                evidence_id="rigid.pose_window:" + name,
                value_dtype="torch.float64",
                values=[values],
                valid=[True],
                acquisition_errors=[None],
                timestamp=10.0,
                observation_revision=1,
                env_ids=[0],
            )
            for name, values in poses.items()
        },
    )
    raw = dict(
        schema_version="simulation.recorded_contacts/v1",
        **identity,
        episode_ids=[7],
        env_ids=[0],
        window_start_frames=[0],
        boundary_frames=[2],
        object_id="cube",
        reference_id="pad",
        timeline=dict(
            scope="native_substep_samples_not_continuous_contact_proof",
            step_dt=0.04,
            retained_contact_rows=8,
            environments=[
                dict(
                    env_id=0,
                    episode_generation=0,
                    errors=[],
                    retained_contact_rows=8,
                    samples=samples,
                    incomplete_control_step=None,
                )
            ],
        ),
        pose_frame=frame,
    )
    binding = replace(
        binding,
        evidence_format="raw_snapshot",
        episode_ids=(7,),
        sampled_contact_identity=identity,
        binding_hash="",
    )
    return (
        spec,
        binding,
        {**evidence, "episode_ids": [7], "recorded_contacts": raw},
        program,
    )


@pytest.mark.parametrize(
    "damage",
    [
        None,
        "missing_substep",
        "episode",
        "topology",
        "native_revision",
        "pose_window",
        "stale_program",
        "reset",
        "wrong_start",
    ],
)
def test_raw_sampled_support_requires_complete_native_window(damage):
    spec, binding, evidence, program = _raw_sampled_window_evidence()
    raw = evidence["recorded_contacts"]
    row = raw["timeline"]["environments"][0]
    if damage == "missing_substep":
        row["samples"][0]["substeps"].pop()
    elif damage == "episode":
        raw["episode_ids"] = [8]
    elif damage == "topology":
        row["samples"][0]["substeps"][0]["topology_revision"] = 4
    elif damage == "native_revision":
        row["samples"][0]["substeps"][0]["native_revision"] = None
    elif damage == "pose_window":
        raw["pose_frame"]["batches"]["rigid.pose_window:cube"]["values"][0].pop()
    elif damage == "stale_program":
        raw["program_fingerprint"] = "b" * 64
    elif damage == "reset":
        row["episode_generation"] = 1
    elif damage == "wrong_start":
        raw["window_start_frames"] = [1]
    outcome = evaluate_supported_placement_evidence(
        spec, binding, evidence, program=program
    )[0]
    assert outcome.status is (
        EvaluationStatus.SATISFIED if damage is None else EvaluationStatus.UNAVAILABLE
    ), outcome.reason
    assert (
        binding.to_dict()["schema_version"] == "task_spec.placement_evidence_binding/v4"
    )


def test_raw_sampled_support_retains_valid_external_load_as_negative():
    spec, binding, evidence, program = _raw_sampled_window_evidence()
    for sample in evidence["recorded_contacts"]["timeline"]["environments"][0][
        "samples"
    ]:
        sample["contacts"]["pairs"] = [[99, 1]]
        for sub in sample["substeps"]:
            sub["contacts"]["pairs"] = [[99, 1]]
    outcome = evaluate_supported_placement_evidence(
        spec, binding, evidence, program=program
    )[0]
    assert outcome.status is EvaluationStatus.FAILED, outcome.reason


def _planar_evidence(case_index=0, row=0, *, raw=False, relation="on"):
    import json
    from dataclasses import replace

    source = json.loads(
        (
            Path(__file__).parents[4] / "sim/scene/fixtures/planar_placement_b0.json"
        ).read_text()
    )["cases"][case_index]
    spec, binding, evidence, program = _legacy_evidence()
    predicate = Predicate(
        "semantic_goal",
        predicate_id="placed",
        arguments={**spec.goal[0].arguments, "relation": relation},
    )
    spec = TaskSpec(roles=spec.roles, goal=(predicate,))
    metrics = {
        key: [value[row]] if isinstance(value, list) else value
        for key, value in source["measurements"].items()
    }
    identity = dict(
        exact_silhouette=source["exact_silhouette"],
        program_fingerprint="a" * 64,
        object_native_uid="native_cube",
        reference_native_uid="native_pad",
    )
    binding = replace(
        binding,
        task_spec_digest=canonical_hash(spec.to_dict()),
        evidence_scope="measured_planar_landing_region",
        evidence_format="raw_snapshot" if raw else "historical_record",
        condition={},
        episode_ids=(7,),
        planar_geometry=identity,
        binding_hash="",
    )
    evidence["episode_ids"] = [7]
    evidence["record"]["measurements"] = {
        **metrics,
        "reference_uid": "pad",
        "relation": relation,
        "released": [True],
    }
    evidence["record"]["accepted"] = [source["accepted"][row]]
    if raw:
        evidence.pop("record")
        metrics.pop("support_tolerance_m")
        metrics.pop("penetration_tolerance_m")
        columns = ["support_gap_m", "footprint_area_m2"] + (
            ["outside_floor_area_m2"] if source["exact_silhouette"] else []
        )
        channels = {
            "planar.geometry": [[metrics[key][0] for key in columns]],
            "planar.full_footprint_inside": metrics["full_footprint_inside"],
            "rigid.object_pose": metrics["object_pose"],
            "rigid.reference_pose": metrics["reference_pose"],
        }
        if source["exact_silhouette"]:
            channels.update(
                {
                    "planar." + key: metrics[key]
                    for key in (
                        "reference_has_raised_geometry",
                        "reference_aabb_overlap",
                    )
                }
            )
        frame = dict(
            timestamp=1.0,
            observation_revision=1,
            env_ids=[0],
            clock_domain="simulation_time",
            revision_kind="planar_geometry_acquisition",
            batches={
                key: dict(
                    evidence_id=key,
                    value_dtype=(
                        "torch.float64"
                        if key == "planar.geometry"
                        else (
                            "torch.float32"
                            if key.startswith("rigid.")
                            else "torch.bool"
                        )
                    ),
                    values=value,
                    valid=[True],
                    acquisition_errors=[None],
                    timestamp=1.0,
                    observation_revision=1,
                    env_ids=[0],
                )
                for key, value in channels.items()
            },
        )
        evidence.update(
            schema_version="simulation.planar_observation/v1",
            object_id="cube",
            reference_id="pad",
            **identity,
            observation_kind="final",
            boundary_frames=[2],
            native_planar=metrics,
            geometry_columns=columns,
            geometry_frame=frame,
        )
    return spec, binding, evidence, program, source["accepted"][row]


@pytest.mark.parametrize("case_index", range(4))
@pytest.mark.parametrize("row", range(2))
@pytest.mark.parametrize("raw", [False, True])
def test_planar_geometry_keeps_frozen_b0_hull_and_triangle_outcomes(
    case_index, row, raw
):
    spec, binding, evidence, program, expected = _planar_evidence(
        case_index, row, raw=raw
    )
    outcome = evaluate_supported_placement_evidence(
        spec, binding, evidence, program=program
    )[0]
    assert outcome.status is (
        EvaluationStatus.SATISFIED if expected else EvaluationStatus.FAILED
    ), outcome.reason


def test_planar_triangle_mode_cannot_borrow_hull_and_final_cannot_borrow_earlier_boundary():
    spec, binding, evidence, program, _ = _planar_evidence(1, raw=True)
    evidence["exact_silhouette"] = False
    assert (
        evaluate_supported_placement_evidence(spec, binding, evidence, program=program)[
            0
        ].status
        is EvaluationStatus.UNAVAILABLE
    )
    spec, binding, evidence, program, _ = _planar_evidence(raw=True)
    later = deepcopy(evidence["segments"][-1])
    later["name"] = "later"
    evidence["segments"].append(later)
    assert (
        evaluate_supported_placement_evidence(spec, binding, evidence, program=program)[
            0
        ].status
        is EvaluationStatus.UNAVAILABLE
    )


def test_triangle_projection_does_not_invent_3d_cavity_containment():
    spec, binding, evidence, program, _ = _planar_evidence(
        1, raw=False, relation="inside"
    )
    metrics = evidence["record"]["measurements"]
    metrics.update(
        full_footprint_inside=[False],
        reference_has_raised_geometry=[True],
        reference_aabb_overlap=[True],
        support_gap_m=[0.0],
    )
    evidence["record"].update(available=[False], accepted=[False])
    outcome = evaluate_supported_placement_evidence(
        spec, binding, evidence, program=program
    )[0]
    assert outcome.status is EvaluationStatus.UNAVAILABLE, outcome.reason


def _directional_evidence_case(index=0, *, raw=False):
    import json
    from embodichain.gen_sim.task_spec import MilestoneSpec
    from embodichain.lab.task_program.integrations.evidence.observation_requests import (
        directional_observation_axes,
    )

    source = json.loads(
        (
            Path(__file__).parents[4]
            / "gen_sim/task_spec/data/side_placement_source_reference.json"
        ).read_text()
    )["cases"][index]
    predicate = Predicate(
        "semantic_goal",
        predicate_id="side",
        arguments=dict(
            subject="item",
            reference="surface",
            relation=source["relation"],
            task_type="E1",
        ),
    )
    spec = TaskSpec(
        roles=(RoleSpec("item"), RoleSpec("surface")),
        goal=(
            (
                Predicate(
                    "semantic_goal", predicate_id="final", arguments=predicate.arguments
                ),
            )
            if source["recorded"]
            else (predicate,)
        ),
        milestones=(
            (MilestoneSpec("placed", (predicate,)),) if source["recorded"] else ()
        ),
    )
    geometry = dict(
        relation=source["relation"],
        program_fingerprint="a" * 64,
        object_native_uid="native_cube",
        reference_native_uid="native_block",
        table_id="table",
        table_native_uid="native_table",
        reference_part=None,
        reference_link_id=None,
        native_link_name=None,
        world_displacement=source["world_displacement"],
    )
    binding = SupportedPlacementEvidenceBinding(
        task_spec_digest=canonical_hash(spec.to_dict()),
        program_digest=canonical_hash(source["program"]),
        predicate_id="side",
        object_id="cube",
        reference_id="block",
        resource_id="left",
        release_segment_id="place",
        release_invocation_id="place:0",
        release_source_path=("program", "items", 0),
        support_segment_id="place",
        stability_preset="gen_sim.place.stable",
        support_source_path=("program", "items", 0),
        boundary_segment_id="place",
        boundary_source_path=("program", "items", 0),
        evidence_scope="bound_rigid_world_axes",
        observation_kind="recorded" if source["recorded"] else "final",
        evidence_format="raw_snapshot" if raw else "historical_record",
        condition=source["condition"],
        required_execution=(
            ({"segment_id": "place", "source_path": ["program", "items", 0]},)
            if source["recorded"]
            else ()
        ),
        episode_ids=(7,),
        directional_geometry=geometry,
    )
    evidence = dict(
        segments=[source["segment"]], record=source["record"], episode_ids=[7]
    )
    if raw:
        evidence.pop("record")
        old = source["record"]["measurements"]
        table = {
            key: value
            for key, value in old["tabletop"].items()
            if key not in {"support_tolerance_m", "penetration_tolerance_m"}
        }
        native = dict(
            object_pose=old["object_pose"],
            reference_pose=old["reference_pose"],
            envelope_gaps_m=old["envelope_gaps_m"],
            tabletop=table,
        )
        values = {
            "directional.envelope_gaps": [
                list(row) for row in zip(*old["envelope_gaps_m"], strict=True)
            ],
            "rigid.object_pose": old["object_pose"],
            "reference.pose": old["reference_pose"],
            "table.pose": table["reference_pose"],
            "tabletop.geometry": [
                [table["support_gap_m"][0], table["footprint_area_m2"][0]]
            ],
            "tabletop.full_footprint_inside": table["full_footprint_inside"],
        }
        frame = dict(
            timestamp=4.0,
            observation_revision=1,
            env_ids=[0],
            clock_domain="simulation_time",
            revision_kind="directional_geometry_acquisition",
            batches={
                key: dict(
                    evidence_id=key,
                    value_dtype=(
                        "torch.bool"
                        if key == "tabletop.full_footprint_inside"
                        else (
                            "torch.float64"
                            if key == "tabletop.geometry"
                            else "torch.float32"
                        )
                    ),
                    values=value,
                    valid=[True],
                    acquisition_errors=[None],
                    timestamp=4.0,
                    observation_revision=1,
                    env_ids=[0],
                )
                for key, value in values.items()
            },
        )
        evidence.update(
            schema_version="simulation.directional_observation/v1",
            object_id="cube",
            reference_id="block",
            **{
                key: value
                for key, value in geometry.items()
                if key not in {"relation", "world_displacement"}
            },
            observation_kind=binding.observation_kind,
            boundary_frames=[100],
            axes=[
                list(pair) for pair in directional_observation_axes(binding.to_dict())
            ],
            policy_segment_name="place" if source["recorded"] else None,
            policy_source_path=["program", "items", 0] if source["recorded"] else None,
            native_directional=native,
            geometry_frame=frame,
        )
    return spec, binding, evidence, source["program"], source["record"]["accepted"][0]


@pytest.mark.parametrize("index", range(16))
@pytest.mark.parametrize("raw", [False, True])
def test_directional_placement_retains_frozen_b0_world_axes_and_hold(index, raw):
    spec, binding, evidence, program, accepted = _directional_evidence_case(
        index, raw=raw
    )
    outcome = evaluate_supported_placement_evidence(
        spec, binding, evidence, program=program
    )[0]
    assert outcome.status is (
        EvaluationStatus.SATISFIED if accepted else EvaluationStatus.FAILED
    ), outcome.reason


@pytest.mark.parametrize("damage", ["axis", "native", "hold", "clock", "pose"])
def test_directional_channels_cannot_borrow_another_route_or_recorded_policy(damage):
    spec, binding, evidence, program, _ = _directional_evidence_case(1, raw=True)
    if damage == "axis":
        evidence["axes"] = [[2, 1]]
    elif damage == "native":
        evidence["reference_native_uid"] = "other"
    elif damage == "hold":
        evidence["segments"][0]["metadata"]["post_policies"][0]["result"][
            "stable_seconds"
        ] = [0.0]
    elif damage == "clock":
        evidence["geometry_frame"]["batches"]["reference.pose"][
            "observation_revision"
        ] = 2
    else:
        evidence["segments"][0]["metadata"]["post_policies"][0]["result"][
            "measurements"
        ]["reference_pose"][0][0][3] += 0.01
    assert (
        evaluate_supported_placement_evidence(spec, binding, evidence, program=program)[
            0
        ].status
        is EvaluationStatus.UNAVAILABLE
    )


@pytest.mark.parametrize("raw", [False, True])
def test_directional_link_identity_is_bound_to_its_recorded_policy(raw):
    from dataclasses import replace

    spec, binding, evidence, program, _ = _directional_evidence_case(13, raw=raw)
    geometry = {
        **binding.directional_geometry,
        "reference_part": "drawer",
        "reference_link_id": "block__drawer",
        "native_link_name": "drawer",
        "world_displacement": binding.condition["displacement"],
    }
    condition = {**binding.condition, "reference": "block__drawer"}
    program["program"]["items"][0]["steps"]["call"]["arguments"][
        "reference_part"
    ] = "drawer"
    evidence["segments"][0]["metadata"]["post_policies"][0]["result"]["measurements"][
        "reference"
    ] = "block__drawer"
    if raw:
        evidence.update(
            reference_part="drawer",
            reference_link_id="block__drawer",
            native_link_name="drawer",
        )
    else:
        evidence["record"]["measurements"].update(
            scope="bound_link_world_axes",
            reference_part="drawer",
            reference_link_id="block__drawer",
        )
    binding = replace(
        binding,
        program_digest=canonical_hash(program),
        condition=condition,
        evidence_scope="bound_link_world_axes",
        directional_geometry=geometry,
        binding_hash="",
    )
    outcome = evaluate_supported_placement_evidence(
        spec, binding, evidence, program=program
    )[0]
    assert outcome.status is EvaluationStatus.SATISFIED, outcome.reason


def _legacy_region_evidence(*, raw=False, recorded=False):
    from dataclasses import replace
    from embodichain.gen_sim.task_spec import MilestoneSpec

    spec, binding, evidence, program = _strong_evidence(region=True, historical=not raw)

    def rename(value):
        if isinstance(value, dict):
            return {
                ("table" if key == "pad" else key): rename(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [rename(item) for item in value]
        return "table" if value == "pad" else value

    evidence = rename(evidence)
    condition = {
        **binding.condition,
        "reference": "table",
        "support_contract_revision": None,
    }
    predicate = Predicate(
        "semantic_goal",
        predicate_id="placed",
        arguments=dict(
            subject="item", reference="surface", relation="on", task_type="E1"
        ),
    )
    spec = TaskSpec(
        roles=spec.roles,
        goal=(
            (
                Predicate(
                    "semantic_goal", predicate_id="final", arguments=predicate.arguments
                ),
            )
            if recorded
            else (predicate,)
        ),
        milestones=(MilestoneSpec("landing", (predicate,)),) if recorded else (),
    )
    program["program"]["items"][0].update(
        kind="segment",
        steps=dict(
            kind="invoke",
            call=dict(
                kind="registered",
                call_id="simulation.place_relative",
                arguments=dict(object="cube", reference="table", relation="on"),
                resources=dict(primary="left"),
            ),
        ),
    )
    native = (
        evidence["native_support"]
        if raw
        else evidence["record"]["measurements"]["native_support"]
    )
    native.pop("support_contract_revision", None)
    trace = evidence["segments"][1]["metadata"]["post_policies"][0]
    trace["result"].pop("support_contract_revision", None)
    trace["result"]["measurements"]["native_support"] = deepcopy(native)
    for segment in evidence["segments"]:
        segment.update(active=[True], successes=[True])
    if raw:
        evidence["support_acquisition"]["clock_domain"] = "simulation_time"
    else:
        metrics = evidence["record"]["measurements"]
        metrics.update(
            scope="native_supported_landing_region",
            reference_uid="table",
            relation="on",
            contact_tolerance_m=0.01,
            observation_kind="recorded" if recorded else "final",
            boundary_frames=[5],
        )
        metrics.pop("support_contract_revision", None)
    binding = replace(
        binding,
        task_spec_digest=canonical_hash(spec.to_dict()),
        program_digest=canonical_hash(program),
        reference_id="table",
        condition=condition,
        evidence_scope="native_supported_landing_region",
        observation_kind="recorded" if recorded else "final",
        required_execution=(
            tuple(
                dict(
                    segment_id=segment["name"],
                    source_path=segment["metadata"]["program_segment_source_path"],
                )
                for segment in evidence["segments"]
            )
            if recorded
            else ()
        ),
        binding_hash="",
    )
    return spec, binding, evidence, program


@pytest.mark.parametrize("raw", [False, True])
@pytest.mark.parametrize("recorded", [False, True])
def test_legacy_region_keeps_native_load_footprint_and_own_release_scope(raw, recorded):
    spec, binding, evidence, program = _legacy_region_evidence(
        raw=raw, recorded=recorded
    )
    outcome = evaluate_supported_placement_evidence(
        spec, binding, evidence, program=program
    )[0]
    assert outcome.status is EvaluationStatus.SATISFIED, outcome.reason
    assert (
        binding.to_dict()["schema_version"] == "task_spec.placement_evidence_binding/v7"
    )
    landing = (
        evidence["landing_region"]
        if raw
        else evidence["record"]["measurements"]["landing_region"]
    )
    landing["full_footprint_inside"] = [False]
    if not raw:
        evidence["record"]["accepted"] = [False]
    outcome = evaluate_supported_placement_evidence(
        spec, binding, evidence, program=program
    )[0]
    assert outcome.status is EvaluationStatus.FAILED, outcome.reason


@pytest.mark.parametrize(
    "damage",
    [
        "strong_revision",
        "triangle_substitute",
        "missing_hold",
        "reference_pose",
        "release_window",
    ],
)
def test_legacy_region_cannot_borrow_other_contracts_or_policy_boundaries(damage):
    spec, binding, evidence, program = _legacy_region_evidence(recorded=True)
    metrics = evidence["record"]["measurements"]
    if damage == "strong_revision":
        metrics["native_support"]["support_contract_revision"] = 1
    elif damage == "triangle_substitute":
        metrics["landing_region"]["footprint_kind"] = "native_triangle_projection/v1"
    elif damage == "missing_hold":
        evidence["segments"][1]["metadata"]["post_policies"][0]["result"][
            "stable_seconds"
        ] = [0.0]
    elif damage == "reference_pose":
        metrics["landing_region"]["reference_pose"][0][0][3] = 0.1
    else:
        metrics["release_window"]["timestamp"] += 0.04
    assert (
        evaluate_supported_placement_evidence(spec, binding, evidence, program=program)[
            0
        ].status
        is EvaluationStatus.UNAVAILABLE
    )


@pytest.mark.parametrize("scope", ["region", "planar", "directional", "sampled"])
def test_final_physical_scope_does_not_require_an_unexecuted_future_post_policy(scope):
    from dataclasses import replace

    if scope == "region":
        spec, binding, evidence, program = _legacy_region_evidence(raw=True)
    elif scope == "planar":
        spec, binding, evidence, program, _ = _planar_evidence(raw=True)
    elif scope == "directional":
        spec, binding, evidence, program, _ = _directional_evidence_case(12, raw=True)
    else:
        spec, binding, evidence, program = _raw_sampled_window_evidence()
    index = len(program["program"]["items"])
    program["program"]["items"].append(dict(name="future_policy"))
    binding = replace(
        binding,
        program_digest=canonical_hash(program),
        support_segment_id="future_policy",
        support_source_path=("program", "items", index),
        binding_hash="",
    )
    outcome = evaluate_supported_placement_evidence(
        spec, binding, evidence, program=program
    )[0]
    assert outcome.status is EvaluationStatus.SATISFIED, outcome.reason


from ._evaluation import evaluate_supported_placement_evidence
