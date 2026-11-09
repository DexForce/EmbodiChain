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

"""Test composition of independent runtime decoding and normative evaluation."""

from __future__ import annotations
import math
from embodichain.lab.task_program.integrations.evidence import (
    HandOverEvidenceBinding,
    CoordinatedHoldEvidenceBinding,
    CoordinatedTransportEvidenceBinding,
    normalize_handover_evidence,
    normalize_coordinated_hold_evidence,
    normalize_coordinated_transport_evidence,
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
    from embodichain.gen_sim.task_spec import TaskSpec

    if spec is not None:
        spec = TaskSpec.from_dict(spec)
    from copy import deepcopy
    from embodichain.gen_sim.task_spec.evidence import SupportedPlacementEvidenceBinding
    from embodichain.gen_sim.task_spec.placement_group_evidence import (
        PlacementEvidenceGroupBinding,
    )
    from embodichain.gen_sim.task_spec.pour_evidence import PourEvidenceBinding
    from embodichain.lab.task_program.integrations.evidence.placement_evidence import (
        normalize_placement_evidence,
        normalize_pour_release_evidence,
    )

    from embodichain.gen_sim.task_spec.control_evidence import ControlEvidenceBinding

    if isinstance(binding, dict) and binding.get("schema_version") in {
        "task_spec.control_evidence_binding/v1",
        "task_spec.control_evidence_binding/v2",
    }:
        binding = ControlEvidenceBinding.from_dict(binding)
    if isinstance(binding, ControlEvidenceBinding):
        from embodichain.lab.task_program.integrations.evidence import (
            normalize_control_evidence,
        )

        binding = ControlEvidenceBinding.from_dict(binding)
        return binding, normalize_control_evidence(
            binding.to_dict(), packet, program=program
        )
    if isinstance(binding, SupportedPlacementEvidenceBinding):
        predicate = (
            next(
                (
                    p
                    for p in (
                        *spec.goal,
                        *(p for m in spec.milestones for p in m.achieve),
                    )
                    if p.id == binding.predicate_id
                ),
                None,
            )
            if spec is not None
            else None
        )
        return binding, normalize_placement_evidence(
            binding.to_dict(),
            packet,
            program=program,
            source_family=(
                None if predicate is None else predicate.arguments.get("task_type")
            ),
        )
    if isinstance(binding, PlacementEvidenceGroupBinding):
        data = deepcopy(packet)
        if isinstance(data.get("members"), dict):
            for member in binding.members:
                if member.binding_hash in data["members"]:
                    data["members"][member.binding_hash] = normalized(
                        member, data["members"][member.binding_hash], program, spec
                    )[1]
        return binding, data
    if isinstance(binding, PourEvidenceBinding):
        return binding, normalize_pour_release_evidence(
            binding.to_dict(), packet, program=program
        )

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
    else:
        kind = "coordinated_acquisition"
        sample = normalize_coordinated_transport_evidence(
            binding, packet, program=program
        )
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
evaluate_coordinated_transport_evidence = evaluate


def evaluate_physical_evidence_report(spec, bindings, evidence, *, program, num_envs):
    pairs = [
        normalized(
            binding,
            evidence.get(
                (
                    binding["predicate_id"]
                    if isinstance(binding, dict)
                    else binding.predicate_id
                ),
                {},
            ),
            program,
            spec,
        )
        for binding in bindings
    ]
    return normative_report(
        spec,
        [p[0] for p in pairs],
        {p[0].predicate_id: p[1] for p in pairs},
        program=program,
        num_envs=num_envs,
    )


def evaluate_supported_placement_evidence(spec, binding, packet, *, program):
    from embodichain.gen_sim.task_spec.evidence import (
        evaluate_supported_placement_evidence as compare,
    )

    bound, normalized_packet = normalized(binding, packet, program, spec)
    return compare(spec, bound, normalized_packet, program=program)


def evaluate_placement_group_evidence(spec, binding, packet, *, program):
    from embodichain.gen_sim.task_spec.placement_group_evidence import (
        evaluate_placement_group_evidence as compare,
    )

    bound, normalized_packet = normalized(binding, packet, program, spec)
    return compare(spec, bound, normalized_packet, program=program)


def evaluate_pour_evidence(spec, binding, packet, *, program):
    from embodichain.gen_sim.task_spec.pour_evidence import (
        evaluate_pour_evidence as compare,
    )

    bound, normalized_packet = normalized(binding, packet, program, spec)
    return compare(spec, bound, normalized_packet, program=program)


def evaluate_control_evidence(spec, binding, packet, *, program):
    from embodichain.gen_sim.task_spec.control_evidence import (
        evaluate_control_evidence as compare,
    )

    norm, sample = normalized(binding, packet, program, spec)
    return compare(spec, norm, sample, program=program)
