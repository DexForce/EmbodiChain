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

"""Normative control facts stay independent of control execution and observers."""

from __future__ import annotations

from copy import deepcopy
import pytest

from embodichain.gen_sim.task_spec import (
    TaskSpec,
    RoleSpec,
    Predicate,
    canonical_hash,
    EvaluationStatus,
)
from embodichain.gen_sim.task_spec.control_evidence import (
    ControlEvidenceBinding,
    evaluate_control_evidence,
)


def _case():
    program = {"program_id": "control-test", "program": {"items": []}}
    predicate = Predicate(
        "pressed",
        predicate_id="pressed",
        arguments={"subject": "device", "task_type": "E9"},
    )
    spec = TaskSpec(roles=(RoleSpec("device"),), goal=(predicate,))
    path = ["program", "items", 0]
    operation = dict(
        route_payload=dict(
            operation_id="press",
            binding=dict(
                object_id="button", joint="joint", link="link", source_sha256="a" * 64
            ),
        ),
        terminal_segment_id="event",
        terminal_source_path=path,
        preset="press",
        required_execution=[dict(segment_id="event", source_path=path)],
    )
    binding = ControlEvidenceBinding(
        canonical_hash(spec.to_dict()),
        canonical_hash(program),
        predicate.id,
        "button",
        "press",
        "final",
        operation,
        "historical_record",
        episode_ids=(7,),
    )
    batch = dict(
        evidence_id="control_complete",
        value_dtype="torch.bool",
        value_shape=[1],
        values=[True],
        valid=[True],
        acquisition_errors=[None],
        timestamp=1.0,
        observation_revision=3,
        env_ids=[0],
    )
    evidence = dict(
        schema_version="physical_evidence.normalized/v1",
        kind="control_press",
        env_ids=[0],
        episode_ids=[7],
        source_binding_hash=binding.binding_hash,
        evidence_format="historical_record",
        criterion="contact_backed_press",
        evidence_refs=[binding.binding_hash, "b" * 64],
        frame=dict(
            timestamp=1.0,
            observation_revision=3,
            env_ids=[0],
            batches={"control_complete": batch},
        ),
    )
    evidence["measurement_hash"] = canonical_hash(evidence)
    return spec, binding, evidence, program


@pytest.mark.parametrize("observed", [True, False])
def test_control_compares_valid_physical_effect_facts(observed):
    spec, binding, evidence, program = _case()
    evidence["frame"]["batches"]["control_complete"]["values"] = [observed]
    evidence["measurement_hash"] = canonical_hash(
        {k: v for k, v in evidence.items() if k != "measurement_hash"}
    )
    before = deepcopy(evidence)
    result = evaluate_control_evidence(spec, binding, evidence, program=program)[0]
    assert result.status is (
        EvaluationStatus.SATISFIED if observed else EvaluationStatus.FAILED
    )
    assert result.observed is observed and evidence == before


@pytest.mark.parametrize(
    "damage",
    [
        "missing",
        "invalid",
        "episode",
        "source",
        "row",
        "shape",
        "hash",
        "raw",
        "env_float",
        "episode_float",
        "boundary",
    ],
)
def test_unavailable_or_unassociated_control_facts_cannot_satisfy_a_goal(damage):
    spec, binding, evidence, program = _case()
    batch = evidence["frame"]["batches"]["control_complete"]
    if damage == "missing":
        evidence["frame"]["batches"].clear()
    elif damage == "invalid":
        batch.update(valid=[False], acquisition_errors=["contact channel missing"])
    elif damage == "episode":
        evidence["episode_ids"] = [8]
    elif damage == "source":
        evidence["source_binding_hash"] = "c" * 64
    elif damage == "row":
        batch["env_ids"] = [1]
    elif damage == "shape":
        batch["value_shape"] = [1, 1]
    elif damage == "env_float":
        evidence["env_ids"] = [0.0]
    elif damage == "episode_float":
        evidence["episode_ids"] = [7.0]
    elif damage == "boundary":
        evidence["boundary_frames"] = [4]
    elif damage == "hash":
        batch["values"] = [False]
    else:
        evidence = {"segments": [], "record": {"accepted": True}}
    if damage not in {"hash", "raw", "env_float", "episode_float", "boundary"}:
        evidence["measurement_hash"] = canonical_hash(
            {k: v for k, v in evidence.items() if k != "measurement_hash"}
        )
    result = evaluate_control_evidence(spec, binding, evidence, program=program)[0]
    assert result.status is EvaluationStatus.UNAVAILABLE and result.observed is None


def test_control_declaration_drift_is_not_resigned_before_evaluation():
    spec, binding, evidence, program = _case()
    assert ControlEvidenceBinding.from_dict(binding).to_dict() == binding.to_dict()
    binding.operation["route_payload"]["binding"]["joint"] = "foreign"
    with pytest.raises(ValueError, match="binding_hash"):
        evaluate_control_evidence(spec, binding, evidence, program=program)
