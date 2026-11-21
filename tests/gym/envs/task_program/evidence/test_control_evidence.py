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

from embodichain.gen_sim.task_spec import (
    TaskSpec,
    RoleSpec,
    Predicate,
    MilestoneSpec,
    canonical_hash,
    EvaluationStatus,
)
from embodichain.gen_sim.task_spec.control_evidence import (
    ControlEvidenceBinding,
)


def _case(kind="press", *, past=False):
    source = json.loads(
        (
            Path(__file__).parent / "data/control_receipt_source_reference.json"
        ).read_text()
    )
    case = next(value for value in source["cases"] if value["kind"] == kind)
    predicate = Predicate(
        "pressed" if kind == "press" else "semantic_goal",
        predicate_id="control",
        arguments={
            "subject": "appliance",
            "task_type": "E9" if kind == "press" else "E8",
        },
    )
    spec = TaskSpec(
        roles=(RoleSpec("appliance", kind="articulation"),),
        goal=(replace(predicate, predicate_id="final_state") if past else predicate,),
        milestones=(MilestoneSpec("operated", (predicate,)),) if past else (),
    )
    operation = {
        key: value
        for key, value in case["declaration"].items()
        if key != "control_kind"
    }
    binding = ControlEvidenceBinding(
        canonical_hash(spec.to_dict()),
        canonical_hash(case["program"]),
        predicate.id,
        "appliance",
        kind,
        "recorded" if past else "final",
        operation,
        "historical_record",
    )
    return spec, binding, case["evidence"], case["program"]


@pytest.mark.parametrize("kind", ["press", "twist"])
@pytest.mark.parametrize("past", [False, True])
def test_control_events_replay_exact_b0_receipts_without_live_owners(kind, past):
    spec, binding, evidence, program = _case(kind, past=past)
    before = deepcopy(evidence)
    outcome = evaluate_control_evidence(spec, binding, evidence, program=program)[0]
    assert outcome.status is EvaluationStatus.SATISFIED
    assert evidence == before


@pytest.mark.parametrize("kind", ["press", "twist"])
@pytest.mark.parametrize(
    "damage",
    [
        "prefix_failed",
        "wrong_occurrence",
        "wrong_route",
        "missing_window",
        "cleanup_failed",
        "effect_failed",
        "missing_record",
        "raw_overlay",
    ],
)
def test_control_missing_identity_or_operation_proof_remains_unavailable(kind, damage):
    spec, binding, evidence, program = _case(kind)
    terminal = evidence["segments"][-1]
    result = terminal["metadata"]["post_policies"][0]["result"]
    if damage == "prefix_failed":
        evidence["segments"][0]["successes"] = [False]
    elif damage == "wrong_occurrence":
        terminal["metadata"]["post_policies"][0]["source_path"][-1] = 1
    elif damage == "wrong_route":
        result["control"]["route_digest"] = "b" * 64
    elif damage == "missing_window":
        result["control"]["window"] = [0.1, 0.1]
    elif damage == "cleanup_failed":
        result["contact_released"] = False
    elif damage == "effect_failed":
        result["event_accepted" if kind == "press" else "stable"] = False
    elif damage == "missing_record":
        evidence.pop("record")
    else:
        evidence["control_acquisition"] = {}
    outcome = evaluate_control_evidence(spec, binding, evidence, program=program)[0]
    assert outcome.status is EvaluationStatus.UNAVAILABLE
    assert outcome.observed is None


def test_control_binding_rejects_drift_and_raw_scope_before_observation():
    spec, binding, evidence, program = _case()
    with pytest.raises(ValueError, match="operation-window acquisition"):
        replace(binding, evidence_format="raw_snapshot", binding_hash="")
    binding.operation["route_payload"]["binding"]["joint"] = "another_joint"
    with pytest.raises(ValueError, match="binding_hash"):
        evaluate_control_evidence(spec, binding, evidence, program=program)


def test_coarse_turn_is_not_implicitly_upgraded_to_calibrated_setting():
    spec, binding, evidence, program = _case("twist")
    result = evidence["segments"][-1]["metadata"]["post_policies"][0]["result"]
    result["target_error"] = 0.1
    evidence["record"]["measurements"]["result"]["target_error"] = 0.1
    assert (
        evaluate_control_evidence(spec, binding, evidence, program=program)[0].status
        is EvaluationStatus.SATISFIED
    )
    operation = deepcopy(binding.operation)
    operation["route_payload"]["success_criterion"] = "calibrated_setting"
    binding = replace(binding, operation=operation, binding_hash="")
    identity = result["control"]
    identity["criterion"] = "calibrated_setting"
    identity["route_digest"] = canonical_hash(operation["route_payload"])
    evidence["record"]["measurements"]["control"] = deepcopy(identity)
    evidence["record"]["measurements"]["result"] = deepcopy(result)
    assert (
        evaluate_control_evidence(spec, binding, evidence, program=program)[0].status
        is EvaluationStatus.UNAVAILABLE
    )


def test_control_import_and_binding_decode_are_standard_library_only():
    root = Path(sys.modules["embodichain"].__file__).resolve().parents[1]
    data = _case()[1].to_dict()
    script = f"import sys;sys.path.insert(0,{str(root)!r});from embodichain.gen_sim.task_spec.control_evidence import ControlEvidenceBinding;ControlEvidenceBinding.from_dict({data!r});assert not any(name=='torch' or name=='numpy' or name.startswith('embodichain.lab') for name in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-I", "-c", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("kind", ["press", "twist"])
def test_common_report_preserves_control_milestone_without_filling_final_goal(kind):
    from ._evaluation import evaluate_physical_evidence_report

    spec, binding, evidence, program = _case(kind, past=True)
    report = evaluate_physical_evidence_report(
        spec, [binding], {binding.predicate_id: evidence}, program=program, num_envs=1
    )[0]
    assert report.milestones[0].status is EvaluationStatus.SATISFIED
    assert report.goal[0].status is EvaluationStatus.UNAVAILABLE
    assert not report.certificate_ready


from ._evaluation import evaluate_control_evidence
