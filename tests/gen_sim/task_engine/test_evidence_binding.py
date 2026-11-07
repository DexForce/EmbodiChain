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

"""Verify the production runtime-to-normative declaration and report boundary."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import pytest

from embodichain.gen_sim.task_engine.evidence_binding import (
    TaskEngineManipulationBinding,
    evaluate_task_engine_evidence,
)
from embodichain.gen_sim.task_spec.manipulation_evidence import HoldAcceptanceCriteria
from gym.envs.task_program.evidence.test_handover_evidence import _case
from gym.envs.task_program.evidence._evaluation import normalized


@pytest.mark.parametrize("index", range(4))
@pytest.mark.parametrize("raw", [False, True])
def test_production_composition_replays_frozen_held_outcomes(index, raw):
    spec, runtime, evidence, program = _case(index, raw=raw)
    acceptance, _ = normalized(runtime, evidence, program, spec)
    declaration = TaskEngineManipulationBinding(runtime, acceptance)
    sealed = declaration.to_dict()
    decoded = TaskEngineManipulationBinding.from_dict(sealed)
    assert decoded.to_dict() == sealed
    result = evaluate_task_engine_evidence(
        spec, [sealed], {"proof": evidence}, program=program, num_envs=1
    )
    # Historical truth is pinned by the owning frozen source fixture.
    pred = result[0].to_dict()[
        "goal" if runtime.observation_kind == "final" else "milestones"
    ]
    # A milestone retains its distinct, past-only evidence scope.
    if runtime.observation_kind == "recorded":
        pred = pred[0]["observed"]["predicates"]
    assert pred[0]["status"] == "satisfied"
    assert declaration.to_dict() == sealed


@pytest.mark.parametrize("damage", ["criteria", "runtime", "association", "old_schema"])
def test_declaration_rejects_drift_or_cross_owner_identity(damage):
    spec, runtime, evidence, program = _case(0, raw=True)
    acceptance, _ = normalized(runtime, evidence, program, spec)
    value = TaskEngineManipulationBinding(runtime, acceptance).to_dict()
    if damage == "criteria":
        value["acceptance"]["criteria"]["translation_limit"] *= 2
    elif damage == "runtime":
        value["runtime"]["operation"]["object_id"] = "another"
    elif damage == "association":
        changed = replace(acceptance, source_binding_hash="b" * 64, binding_hash="")
        with pytest.raises(ValueError, match="another runtime"):
            TaskEngineManipulationBinding(runtime, changed)
        return
    else:
        value["schema_version"] = "task_spec.handover_evidence_binding/v1"
    with pytest.raises(ValueError):
        TaskEngineManipulationBinding.from_dict(value)


def test_public_projection_requires_explicit_criteria_without_a_hold_declaration():
    from embodichain.gen_sim.task_engine.evidence import (
        project_handover_evidence_binding,
    )
    from embodichain.gen_sim.task_spec import canonical_hash

    spec, runtime, evidence, program = _case(0, raw=True)
    acceptance, _ = normalized(runtime, evidence, program, spec)
    occurrence = runtime.operation["transfer"]
    node_id = occurrence["segment_name"]
    declared = program
    for key in occurrence["source_path"]:
        declared = declared[key]
    graph = dict(
        nodes=[
            dict(
                id=node_id,
                task_type="E4",
                task_instance_id="e4",
                call=declared["steps"]["call"],
            )
        ],
        task_groups=[dict(id="e4", task_type="E4", node_ids=[node_id])],
    )
    coverage = dict(
        status="covered",
        issues=[],
        goals=[dict(id=p.id, groups=["e4"]) for p in spec.goal],
        milestones=[dict(id=m.id, groups=["e4"]) for m in spec.milestones],
        roles={"object": [dict(uid="object", part_id=None)]},
        lineage={
            "object": [dict(group_id="e4", object_uid="object", object_part_id=None)]
        },
    )
    validation = dict(
        task_spec_digest=canonical_hash(spec.to_dict()),
        program_digest=canonical_hash(program),
        graph_digest=canonical_hash(graph),
        coverage=coverage,
    )

    def project(**extra):
        return project_handover_evidence_binding(
            spec,
            graph,
            validation,
            program,
            runtime.operation,
            predicate_id="proof",
            transfer_node_id=node_id,
            boundary_node_id=node_id,
            step_dt=0.04,
            episode_ids=(7,),
            program_fingerprint="a" * 64,
            object_native_uid="native_object",
            robot_uid="native_robot",
            **extra,
        )

    with pytest.raises(ValueError, match="explicit criteria"):
        project()
    paired = project(acceptance_criteria=acceptance.criteria)
    assert paired.acceptance.criteria == acceptance.criteria
    assert paired.acceptance.source_binding_hash == paired.runtime.binding_hash


@pytest.mark.parametrize("kind", ["press", "twist"])
@pytest.mark.parametrize("past", [False, True])
def test_control_bridge_uses_lab_normalization(kind, past):
    from gym.envs.task_program.evidence.test_control_evidence import (
        _case as control_case,
    )

    spec, binding, evidence, program = control_case(kind, past=past)
    before = deepcopy(evidence)
    report = evaluate_task_engine_evidence(
        spec,
        [binding.to_dict()],
        {binding.predicate_id: evidence},
        program=program,
        num_envs=1,
    )[0].to_dict()
    predicates = (
        report["milestones"][0]["observed"]["predicates"] if past else report["goal"]
    )
    assert predicates[0]["status"] == "satisfied"
    assert evidence == before
    missing = evaluate_task_engine_evidence(
        spec,
        [binding.to_dict()],
        {},
        program=program,
        num_envs=1,
    )[0].to_dict()
    assert missing["milestones" if past else "goal"][0]["status"] == "unavailable"
