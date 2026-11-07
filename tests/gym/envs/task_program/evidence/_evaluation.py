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

"""Compose Lab facts and explicit frozen criteria for the owning regressions."""

from __future__ import annotations

import math
from embodichain.lab.task_program.integrations.evidence import (
    HandOverEvidenceBinding,
    CoordinatedHoldEvidenceBinding,
    normalize_handover_evidence,
    normalize_coordinated_hold_evidence,
)
from embodichain.gen_sim.task_spec.manipulation_evidence import (
    HoldAcceptanceCriteria,
    ManipulationEvidenceBinding,
    evaluate_manipulation_evidence,
)
from embodichain.gen_sim.task_spec.evidence_report import (
    evaluate_physical_evidence_report as normative_report,
)


def normalized(binding, packet, program, spec=None):
    if type(binding) is HandOverEvidenceBinding:
        kind = {
            "handover_complete": "handover_complete",
            "held": "receiver_hold",
            "orientation": "receiver_orientation",
        }[binding.query_kind]
        sample = normalize_handover_evidence(binding, packet, program=program)
    elif type(binding) is CoordinatedHoldEvidenceBinding:
        kind = "coordinated_hold"
        sample = normalize_coordinated_hold_evidence(binding, packet, program=program)
    criteria = None
    if kind in {"receiver_hold", "receiver_orientation", "coordinated_hold"}:
        condition = binding.operation["hold_condition"]
        if condition is None:
            # Explicit frozen Source fixture limits; never read mutable receipts.
            criteria = HoldAcceptanceCriteria(0.06, 3.0)
        else:
            target = condition.get("target_position")
            axis = condition.get("local_axis")
            criteria = HoldAcceptanceCriteria(
                condition.get("translation_drift", 0.02),
                condition.get("rotation_drift", math.pi / 18),
                hold_seconds=condition.get("duration", 3.0),
                target_position=target,
                position_tolerance=(
                    condition.get("position_tolerance", 0.04)
                    if target is not None
                    else None
                ),
                local_axis=axis,
                world_axis=(
                    condition.get("world_axis") or (0, 0, 1)
                    if axis is not None
                    else None
                ),
                minimum_alignment=(
                    condition.get("minimum_alignment", 0.8660254037844386)
                    if axis is not None
                    else None
                ),
            )
    norm = ManipulationEvidenceBinding(
        binding.task_spec_digest,
        binding.program_digest,
        binding.predicate_id,
        kind,
        binding.observation_kind,
        binding.env_ids,
        binding.episode_ids,
        criteria,
        binding.binding_hash,
        evidence_format=binding.evidence_format,
    )
    return norm, sample


def evaluate(spec, binding, packet, *, program):
    norm, sample = normalized(binding, packet, program, spec)
    return evaluate_manipulation_evidence(spec, norm, sample, program=program)


evaluate_handover_evidence = evaluate
evaluate_coordinated_hold_evidence = evaluate


def evaluate_physical_evidence_report(spec, bindings, evidence, *, program, num_envs):
    pairs = [
        normalized(binding, evidence.get(binding.predicate_id, {}), program, spec)
        for binding in bindings
    ]
    return normative_report(
        spec,
        [p[0] for p in pairs],
        {p[0].predicate_id: p[1] for p in pairs},
        program=program,
        num_envs=num_envs,
    )
