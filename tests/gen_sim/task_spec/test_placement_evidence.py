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
    evaluate_supported_placement_evidence,
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


def test_evidence_cold_import_does_not_load_numeric_or_simulation_backends():
    root = Path(__file__).resolve().parents[3]
    source = f"import sys; sys.path.insert(0, {str(root)!r}); import embodichain.gen_sim.task_spec.evidence; assert not any(name == 'torch' or name == 'numpy' or name.startswith('embodichain.lab') for name in sys.modules)"
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
    raw_binding.update(episode_ids=[3], binding_hash="")
    binding = SupportedPlacementEvidenceBinding.from_dict(raw_binding)
    evidence["episode_ids"] = [3]
    native = evidence.pop("record")["measurements"]
    for name in ("released", "reference_uid", "relation"):
        native.pop(name)
    evidence["native_support"] = native
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
    from embodichain.gen_sim.task_spec.evidence import (
        evaluate_placement_evidence_report,
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
