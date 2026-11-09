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
import json
from pathlib import Path

import pytest

from embodichain.gen_sim.task_engine.evidence import project_placement_evidence_binding


def _declared():
    payload = json.loads(
        (Path(__file__).parent / "data/e1_frozen_declaration.json").read_text()
    )
    nodes = payload["graph"]["nodes"]
    occurrences = {
        node["id"]: {
            "segment_name": node["id"],
            "source_path": ["program", "items", index],
        }
        for index, node in enumerate(nodes)
    }
    # Opaque descriptor supplied by the compile host; the projection must not
    # manufacture or rewrite this into a task-name-based invocation ID.
    occurrences["step_01__call_02"].update(
        invocation_id="provided/compiled-occurrence:17",
        object_id="cylinder",
        reference_id="pad",
        resource_id="left",
    )
    return payload, occurrences


def _project(payload, occurrences, *, milestone=False):
    return project_placement_evidence_binding(
        payload["spec"],
        payload["graph"],
        payload["validation"],
        payload["program"],
        payload["constraints"],
        occurrences,
        predicate_id="milestone:step_01:0" if milestone else "goal_001",
        release_node_id="step_01__call_02",
        boundary_node_id="step_01__call_03",
        evidence_scope="native_contact_support",
        stability_preset="gen_sim.step_01__call_02.stable",
        observation_kind="recorded" if milestone else "final",
        episode_ids=[2],
    )


def test_frozen_e1_projection_preserves_exact_declarations_and_opaque_occurrence():
    payload, occurrences = _declared()
    before = deepcopy((payload, occurrences))
    goal = _project(payload, occurrences)
    milestone = _project(payload, occurrences, milestone=True)
    assert goal.release_invocation_id == "provided/compiled-occurrence:17"
    assert goal.condition["com_tolerance"] == 0.1
    assert goal.evidence_scope == "native_contact_support"
    assert goal.required_execution == ()
    assert [item["segment_id"] for item in milestone.required_execution] == [
        node["id"] for node in payload["graph"]["nodes"]
    ]
    assert (payload, occurrences) == before
    payload["constraints"]["presets"]["gen_sim.step_01__call_02.stable"][
        "com_tolerance"
    ] = 0.2
    assert goal.condition["com_tolerance"] == 0.1


@pytest.mark.parametrize(
    "damage", ["program", "missing_occurrence", "foreign_role", "foreign_source"]
)
def test_projection_rejects_unmatched_or_uncompiled_inputs_instead_of_filling_gaps(
    damage,
):
    payload, occurrences = _declared()
    if damage == "program":
        payload["program"]["program"]["items"][1].pop("validators")
    elif damage == "missing_occurrence":
        occurrences.pop("step_01__call_02")
    elif damage == "foreign_role":
        occurrences["step_01__call_02"]["reference_id"] = "another_surface"
    else:
        occurrences["step_01__call_02"]["source_path"][-1] = 0
    with pytest.raises((KeyError, ValueError)):
        _project(payload, occurrences)


def _state_declaration(kind):
    from embodichain.gen_sim.task_spec import (
        TaskSpec,
        Predicate,
        RoleSpec,
        MilestoneSpec,
        canonical_hash,
    )

    joint = kind == "joint_endpoint"
    predicate = Predicate(
        "articulation_joint_near" if joint else "object_upright",
        predicate_id="final",
        arguments={
            "subject": "thing",
            "task_type": "E6" if joint else "E2",
            **({"target_state": "open"} if joint else {}),
        },
    )
    from dataclasses import replace

    spec = TaskSpec(
        roles=(RoleSpec("thing"),),
        goal=(predicate,),
        milestones=(
            MilestoneSpec("operation", (replace(predicate, predicate_id="past"),)),
        ),
    )
    call = dict(
        kind="registered",
        call_id="gen_sim.articulation_slide" if joint else "custom.orientation",
        arguments=dict(object="payload", part="panel" if joint else None, state="open"),
    )
    node_id = "arbitrary-node"
    graph = dict(
        nodes=[dict(id=node_id, call=call)],
        task_groups=[dict(id="arbitrary-group", node_ids=[node_id])],
    )
    program = dict(
        program=dict(
            kind="segment",
            name="opaque/compiled:7",
            steps=dict(kind="invoke", call=call),
            post=[dict(kind="wait_stable", entity="payload", preset="named")],
        )
    )
    condition = (
        dict(entity="payload", joint="authored_joint", target=0.2, tolerance=0.01)
        if joint
        else dict(entity="payload", kind="upright", local_axis=[0, 0, 1])
    )
    resolved = dict(
        entity_id="payload",
        native_uid="native_payload",
        part_id="panel" if joint else None,
        condition=condition,
    )
    coverage = dict(
        status="covered",
        issues=[],
        goals=[dict(id="final", groups=["arbitrary-group"])],
        milestones=[dict(id="operation", groups=["arbitrary-group"])],
        roles={"thing": [dict(uid="payload", part_id=resolved["part_id"])]},
        lineage={
            "thing": [
                dict(
                    group_id="arbitrary-group",
                    object_uid="payload",
                    object_part_id=resolved["part_id"],
                )
            ]
        },
    )
    validation = dict(
        task_spec_digest=canonical_hash(spec.to_dict()),
        graph_digest=canonical_hash(graph),
        program_digest=canonical_hash(program),
        coverage=coverage,
    )
    occurrences = {
        node_id: dict(segment_name="opaque/compiled:7", source_path=["program"])
    }
    return spec, graph, validation, program, occurrences, resolved


@pytest.mark.parametrize("kind", ["orientation", "joint_endpoint"])
def test_state_projection_keeps_owner_criteria_and_compiled_occurrence_identities(kind):
    from embodichain.gen_sim.task_engine.evidence import (
        project_condition_evidence_binding,
    )

    args = _state_declaration(kind)
    before = deepcopy(args)
    final = project_condition_evidence_binding(
        *args,
        predicate_id="final",
        boundary_node_id="arbitrary-node",
        policy_node_id="arbitrary-node",
        stability_preset="named",
        evidence_kind=kind,
        episode_ids=[4],
    )
    recorded = project_condition_evidence_binding(
        *args,
        predicate_id="past",
        boundary_node_id="arbitrary-node",
        policy_node_id="arbitrary-node",
        stability_preset="named",
        evidence_kind=kind,
        episode_ids=[4],
    )
    assert final.required_execution == ()
    assert recorded.required_execution == (
        {"segment_id": "opaque/compiled:7", "source_path": ["program"]},
    )
    assert recorded.policy_segment_id == "opaque/compiled:7"
    assert recorded.condition.items() >= args[-1]["condition"].items()
    assert args == before
    if kind == "joint_endpoint":
        assert recorded.condition["target_state"] == "open"
        assert recorded.condition["part_id"] == "panel"


@pytest.mark.parametrize(
    "damage", ["foreign_role", "foreign_path", "foreign_endpoint", "unknown_preset"]
)
def test_state_projection_rejects_inconsistent_grounding_without_selecting_a_fallback(
    damage,
):
    from embodichain.gen_sim.task_engine.evidence import (
        project_condition_evidence_binding,
    )

    args = _state_declaration("joint_endpoint")
    preset = "named"
    if damage == "foreign_role":
        args[-1]["entity_id"] = "other"
    elif damage == "foreign_path":
        args[-2]["arbitrary-node"]["source_path"] = ["program", "steps"]
    elif damage == "foreign_endpoint":
        args[-1]["condition"]["target_state"] = "closed"
    else:
        preset = "other"
    with pytest.raises((ValueError, KeyError)):
        project_condition_evidence_binding(
            *args,
            predicate_id="past",
            boundary_node_id="arbitrary-node",
            policy_node_id="arbitrary-node",
            stability_preset=preset,
            evidence_kind="joint_endpoint",
            episode_ids=[4],
        )


def test_configured_owner_projection_needs_no_legacy_constraints_payload_or_digest():
    from embodichain.gen_sim.task_engine.evidence import (
        project_configured_placement_evidence_binding,
    )

    payload, occurrences = _declared()
    expected = _project(payload, occurrences)
    preset = "gen_sim.step_01__call_02.stable"
    condition = deepcopy(payload.pop("constraints")["presets"][preset])
    payload["validation"].pop("constraints_digest")
    before = deepcopy(condition)
    result = project_configured_placement_evidence_binding(
        payload["spec"],
        payload["graph"],
        payload["validation"],
        payload["program"],
        occurrences,
        condition=condition,
        predicate_id="goal_001",
        release_node_id="step_01__call_02",
        boundary_node_id="step_01__call_03",
        evidence_scope="native_contact_support",
        stability_preset=preset,
        observation_kind="final",
        episode_ids=[2],
    )
    assert result.to_dict() == expected.to_dict()
    assert condition == before
    condition["com_tolerance"] = 0.2
    assert result.condition["com_tolerance"] == 0.1
    condition["entity"] = "other"
    with pytest.raises(ValueError, match="grounded endpoints"):
        project_configured_placement_evidence_binding(
            payload["spec"],
            payload["graph"],
            payload["validation"],
            payload["program"],
            occurrences,
            condition=condition,
            predicate_id="goal_001",
            release_node_id="step_01__call_02",
            boundary_node_id="step_01__call_03",
            evidence_scope="native_contact_support",
            stability_preset=preset,
            observation_kind="final",
            episode_ids=[2],
        )
