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

"""Normative acceptance is independent of execution and native observation APIs."""

from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys

import pytest

from embodichain.gen_sim.task_spec import Predicate, RoleSpec, TaskSpec, canonical_hash
from embodichain.gen_sim.task_spec.manipulation_evidence import (
    HoldAcceptanceCriteria,
    ManipulationEvidenceBinding,
    evaluate_manipulation_evidence,
)


def _inputs():
    spec = TaskSpec(
        roles=(RoleSpec("payload"),),
        goal=(
            Predicate(
                "held", predicate_id="held_goal", arguments={"subject": "payload"}
            ),
        ),
    )
    program = {"program_id": "opaque_program"}
    criteria = HoldAcceptanceCriteria(0.02, 0.20, hold_seconds=3.0)
    binding = ManipulationEvidenceBinding(
        canonical_hash(spec.to_dict()),
        canonical_hash(program),
        "held_goal",
        "receiver_hold",
        "final",
        (0,),
        (2,),
        criteria,
        "b" * 64,
    )
    facts = dict(
        kind="receiver_hold",
        env_ids=[0],
        episode_ids=[2],
        source_binding_hash="b" * 64,
        valid=[True],
        acquisition_errors=[None],
        facts=dict(
            attachment_translation=[[0.01]],
            attachment_rotation=[[0.05]],
            observed_seconds=[3.0],
            stable_seconds=[3.0],
            grips=[[True]],
        ),
        evidence_refs=["c" * 64],
    )
    return spec, binding, facts, program


def _serialized(packet):
    """Build the documented shared frame JSON contract, without importing Lab."""
    result = deepcopy(packet)
    values = result.pop("facts")
    valid, errors = result.pop("valid"), result.pop("acquisition_errors")
    result["schema_version"] = "physical_evidence.normalized/v1"
    result["frame"] = dict(
        timestamp=3.0,
        observation_revision=75,
        env_ids=[0],
        batches={
            name: dict(
                evidence_id=name,
                values=data,
                valid=valid,
                acquisition_errors=errors,
                timestamp=3.0,
                observation_revision=75,
                env_ids=[0],
                value_dtype="torch.bool" if name == "grips" else "torch.float64",
            )
            for name, data in values.items()
        },
    )
    result["measurement_hash"] = canonical_hash(result)
    return result


@pytest.mark.parametrize(
    "change,status",
    [
        (None, "satisfied"),
        ("grip", "failed"),
        ("translation", "failed"),
        ("unstable", "failed"),
        ("incomplete", "unavailable"),
        ("missing", "unavailable"),
        ("episode", "unavailable"),
    ],
)
def test_normalized_measurement_outcomes(change, status):
    spec, binding, packet, program = _inputs()
    if change == "grip":
        packet["facts"]["grips"] = [[False]]
    elif change == "translation":
        packet["facts"]["attachment_translation"] = [[0.03]]
    elif change == "unstable":
        packet["facts"]["stable_seconds"] = [2.0]
    elif change == "incomplete":
        packet["facts"]["observed_seconds"] = [2.0]
    elif change == "missing":
        packet["valid"] = [False]
        packet["acquisition_errors"] = ["channel was not acquired"]
    elif change == "episode":
        packet["episode_ids"] = [3]
    outcome = evaluate_manipulation_evidence(
        spec, binding, _serialized(packet), program=program
    )[0]
    assert outcome.status.value == status
    assert outcome.observed is (
        None if status == "unavailable" else status == "satisfied"
    )


def test_execution_metadata_cannot_loosen_the_sealed_criteria():
    spec, binding, packet, program = _inputs()
    packet["facts"]["attachment_translation"] = [[0.03]]
    changed = deepcopy(packet)
    changed["executor_monitor"] = {"attached_translation_threshold": 1.0}
    before = evaluate_manipulation_evidence(
        spec, binding, _serialized(packet), program=program
    )
    after = evaluate_manipulation_evidence(
        spec, binding, _serialized(changed), program=program
    )
    assert before[0].observed is after[0].observed is False
    assert before[0].status.value == after[0].status.value == "failed"
    serialized = binding.to_dict()
    serialized["criteria"]["translation_limit"] = 1.0
    with pytest.raises(ValueError, match="sealed identity"):
        ManipulationEvidenceBinding.from_dict(serialized)


def test_normalized_measurement_drift_does_not_keep_an_old_identity():
    spec, binding, packet, program = _inputs()
    sealed = _serialized(packet)
    sealed["frame"]["batches"]["attachment_translation"]["values"] = [[0.03]]
    result = evaluate_manipulation_evidence(spec, binding, sealed, program=program)[0]
    assert result.status.value == "unavailable" and result.observed is None


def test_normative_import_requires_no_runtime_or_numerical_dependency(tmp_path):
    repository = Path(__file__).parents[3]
    env = {**os.environ, "PYTHONPATH": str(repository), "PYTHONDONTWRITEBYTECODE": "1"}
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            "import sys; import embodichain.gen_sim.task_spec.manipulation_evidence; "
            "assert not any(n.startswith(('torch', 'numpy', 'embodichain.lab', 'dexsim')) for n in sys.modules)",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_non_axis_aligned_criteria_keep_identity_on_repeated_decoding():
    criteria = HoldAcceptanceCriteria(
        0.02,
        0.2,
        local_axis=(0.8444218515250481, 0.7579544029403025, 0.420571580830845),
        world_axis=(0, 0, 1),
        minimum_alignment=0.8,
    )
    original = criteria.to_dict()
    for _ in range(5):
        criteria = HoldAcceptanceCriteria.from_dict(criteria)
        assert criteria.to_dict() == original


def test_declared_hold_alignment_is_compared_from_the_measured_rotation():
    from dataclasses import replace

    spec, binding, packet, program = _inputs()
    criteria = HoldAcceptanceCriteria(
        0.02,
        0.2,
        hold_seconds=3,
        local_axis=(1, 0, 0),
        world_axis=(0, 0, 1),
        minimum_alignment=0.8,
    )
    binding = replace(binding, criteria=criteria, binding_hash="")
    packet["facts"]["object_rotation"] = [
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    ]
    outcome = evaluate_manipulation_evidence(
        spec, binding, _serialized(packet), program=program
    )[0]
    assert outcome.status.value == "failed"
