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

"""Raw control v2 rechecks actual public producer packets with no live owner."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from embodichain.gen_sim.task_spec import (
    TaskSpec,
    RoleSpec,
    Predicate,
    MilestoneSpec,
    EvaluationStatus,
    canonical_hash,
)
from embodichain.gen_sim.task_spec.control_evidence import (
    ControlEvidenceBinding,
    evaluate_control_evidence,
)
from embodichain.gen_sim.task_spec.evidence_report import (
    evaluate_physical_evidence_report,
)
from .test_control_evidence import _case

DATA = Path(__file__).parent / "data"
ENCODED = gzip.decompress((DATA / "raw_control_public1057.json.gz").read_bytes())
PROVENANCE = json.loads((DATA / "raw_control_public1057_provenance.json").read_text())
assert hashlib.sha256(ENCODED).hexdigest() == PROVENANCE["sha256_uncompressed"]
CASES = {x["case"]: x for x in json.loads(ENCODED)["cases"]}


def raw_case(name="press", *, past=False):
    case = deepcopy(CASES[name])
    predicate = Predicate(
        "pressed" if case["kind"] == "press" else "semantic_goal",
        predicate_id="control",
        arguments={
            "subject": "device",
            "task_type": "E9" if case["kind"] == "press" else "E8",
        },
    )
    spec = TaskSpec(
        roles=(RoleSpec("device"),),
        goal=(
            replace(predicate, predicate_id="unproved_final") if past else predicate,
        ),
        milestones=(MilestoneSpec("operated", (predicate,)),) if past else (),
    )
    binding = ControlEvidenceBinding(
        canonical_hash(spec.to_dict()),
        canonical_hash(case["program"]),
        predicate.id,
        case["entity_id"],
        case["kind"],
        "recorded" if past else "final",
        case["operation"],
        "raw_snapshot",
        episode_ids=tuple(case["episode_ids"]),
        control_identity=case["control_identity"],
        policy_occurrences=tuple(case["policy_occurrences"]),
    )
    return spec, binding, case["evidence"], case["program"]


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("past", [False, True])
def test_public_raw_control_packets_preserve_full_proof_and_report_dispatch(name, past):
    spec, binding, evidence, program = raw_case(name, past=past)
    serialized = binding.to_dict()
    assert serialized["schema_version"] == "task_spec.control_evidence_binding/v2"
    assert ControlEvidenceBinding.from_dict(serialized).to_dict() == serialized
    result = evaluate_control_evidence(spec, serialized, evidence, program=program)
    assert len(result) == 1 and result[0].status is EvaluationStatus.SATISFIED, result[
        0
    ].reason
    report = evaluate_physical_evidence_report(
        spec, (serialized,), {"control": evidence}, program=program, num_envs=1
    )[0]
    if past:
        assert (
            report.milestones[0].status is EvaluationStatus.SATISFIED
            and not report.certificate_ready
        )
    else:
        assert report.certificate_ready


@pytest.mark.parametrize(
    "damage",
    [
        "clearance",
        "limits",
        "instance",
        "episode",
        "invocation",
        "window_prefix",
        "missing_channel",
        "invalid_finite",
        "boundary",
    ],
)
def test_raw_control_cannot_borrow_foreign_or_missing_physical_facts(damage):
    spec, binding, evidence, program = raw_case()
    packet = evidence["control_acquisition"]
    facts = packet["policies"][-1]["measurements"]
    if damage == "clearance":
        facts["clearance"]["batches"]["control.terminal_clearance"]["values"][0] = 0.0
    elif damage == "limits":
        packet["policies"][0]["measurements"]["joint_limits"]["batches"][
            "articulation.joint_limits"
        ]["values"] = [[0.0, 1.0]]
    elif damage == "instance":
        packet["control_identity"]["observer_id"] = "foreign"
    elif damage == "episode":
        packet["episode_ids"] = [1]
    elif damage == "invocation":
        evidence["segments"][-1]["metadata"]["program_segment_id"] = "foreign"
    elif damage == "window_prefix":
        packet["window"]["records"].pop(0)
    elif damage == "missing_channel":
        facts["clearance"]["batches"].clear()
    elif damage == "invalid_finite":
        facts["clearance"]["batches"]["control.terminal_clearance"]["valid"] = [False]
    else:
        packet["boundary_frames"] = [facts["hold_end_step"] + 1]
    outcome = evaluate_control_evidence(spec, binding, evidence, program=program)[0]
    assert outcome.status is EvaluationStatus.UNAVAILABLE and outcome.observed is None
    assert outcome.reason


def test_twist_raw_control_rechecks_non_target_joint_baseline():
    spec, binding, evidence, program = raw_case("twist")
    evidence["control_acquisition"]["policies"][-1]["measurements"]["current_qpos"][
        "batches"
    ]["articulation.joint_positions"]["values"][0][1] += 0.1
    outcome = evaluate_control_evidence(spec, binding, evidence, program=program)[0]
    assert outcome.status is EvaluationStatus.UNAVAILABLE


@pytest.mark.parametrize("damage", ["missing_shape", "wrong_shape"])
def test_empty_contact_tail_requires_its_original_shape_metadata(damage):
    spec, binding, evidence, program = raw_case("press_empty_initial")
    frames = [x["frame"] for x in evidence["control_acquisition"]["window"]["records"]]
    frame = next(
        x for x in frames if x["batches"]["contact.user_ids"]["value_shape"][1] == 0
    )
    channel = frame["batches"]["contact.user_ids"]
    if damage == "missing_shape":
        channel.pop("value_shape")
    else:
        channel["value_shape"] = [1, 1, 2]
    assert (
        evaluate_control_evidence(spec, binding, evidence, program=program)[0].status
        is EvaluationStatus.UNAVAILABLE
    )


@pytest.mark.parametrize("version", ["v1_with_raw_fields", "v2_with_historical_format"])
def test_binding_versions_never_upgrade_old_records_or_change_verifier(version):
    _, binding, _, _ = raw_case()
    raw = binding.to_dict()
    if version == "v1_with_raw_fields":
        raw["schema_version"] = "task_spec.control_evidence_binding/v1"
    else:
        raw["evidence_format"] = "historical_record"
    with pytest.raises(ValueError):
        ControlEvidenceBinding.from_dict(raw)


def test_historical_control_hash_serialization_outcomes_and_reports_match_actual_main():
    for expected in json.loads((DATA / "v1_main_oracles.json").read_text()):
        spec, binding, evidence, program = _case(
            expected["kind"], past=expected["past"]
        )
        assert binding.to_dict() == expected["binding"]
        assert [
            x.to_dict()
            for x in evaluate_control_evidence(spec, binding, evidence, program=program)
        ] == expected["outcomes"]
        assert [
            x.to_dict()
            for x in evaluate_physical_evidence_report(
                spec, (binding,), {"control": evidence}, program=program, num_envs=1
            )
        ] == expected["reports"]
        assert (
            "control_identity" not in binding.to_dict()
            and "policy_occurrences" not in binding.to_dict()
        )


def test_raw_binding_decode_stays_standard_library_only():
    _, binding, _, _ = raw_case()
    root = Path(__file__).resolve().parents[3]
    script = f"import sys;sys.path.insert(0,{str(root)!r});from embodichain.gen_sim.task_spec.control_evidence import ControlEvidenceBinding;ControlEvidenceBinding.from_dict({binding.to_dict()!r});assert not any(n=='torch' or n=='numpy' or n.startswith('embodichain.lab') or n.startswith('embodichain.gen_sim.task_engine') for n in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-I", "-c", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_actual_producer_bindings_and_complete_reports_match_unmodified_source():
    for expected in json.loads((DATA / "raw_control_source_oracles.json").read_text()):
        spec, binding, evidence, program = raw_case(
            expected["case"], past=expected["past"]
        )
        assert binding.to_dict() == expected["binding"]
        assert [
            x.to_dict()
            for x in evaluate_control_evidence(spec, binding, evidence, program=program)
        ] == expected["outcomes"]
        assert [
            x.to_dict()
            for x in evaluate_physical_evidence_report(
                spec,
                (binding.to_dict(),),
                {"control": evidence},
                program=program,
                num_envs=1,
            )
        ] == expected["reports"]


@pytest.mark.parametrize("format", ["raw_snapshot", "historical_record"])
def test_explicit_format_cannot_borrow_another_contracts_snapshot(format):
    if format == "raw_snapshot":
        spec, binding, evidence, program = raw_case()
        evidence["record"] = {"accepted": True}
    else:
        spec, binding, evidence, program = _case()
        evidence["control_acquisition"] = CASES["press"]["evidence"][
            "control_acquisition"
        ]
    assert (
        evaluate_control_evidence(spec, binding, evidence, program=program)[0].status
        is EvaluationStatus.UNAVAILABLE
    )


def test_raw_scope_rejects_undeclared_batch_and_binding_drift_before_evaluation():
    _, binding, _, _ = raw_case()
    with pytest.raises(ValueError, match="B=1"):
        replace(binding, env_ids=(0, 1), episode_ids=(0, 0), binding_hash="")
    binding.control_identity["observer_id"] = "foreign"
    with pytest.raises(ValueError, match="binding_hash"):
        ControlEvidenceBinding.from_dict(binding)
