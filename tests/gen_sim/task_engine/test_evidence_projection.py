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
