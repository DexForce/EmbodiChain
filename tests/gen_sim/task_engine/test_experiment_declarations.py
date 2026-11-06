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

"""Experiment selections retain source graph identity and explicit permissions."""

from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import json, hashlib
import pytest
from embodichain.gen_sim.task_engine.experiment_declarations import (
    horizontal_handover_permissions,
    add_reacquisition_clearance,
    merge_contact_permissions,
    validate_post_handover_place_declarations,
    select_end_grip_post_handover_places,
)

REFERENCE = json.loads(
    (Path(__file__).parent / "data/experiment_declarations_b0.json").read_text()
)
GRAPHS = {
    row["source_case"]: row["graph"]
    for row in json.loads(
        (Path(__file__).parent / "data/action_profiles_b0.json").read_text()
    )["cases"]
}


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda c: c["source_case"])
def test_source_experiment_permission_and_idle_graph_match_frozen_b0(case):
    graph = deepcopy(case.get("graph", GRAPHS.get(case["source_case"])))
    before = deepcopy(graph)
    places, revisions = horizontal_handover_permissions(graph)
    adjusted, idle = add_reacquisition_clearance(graph, REFERENCE["embodiment"])
    assert places == case["post_handover_places"]
    assert revisions == case["native_revisions"]
    assert idle == case["idle_routes"]
    assert (
        hashlib.sha256(
            json.dumps(adjusted, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        == case["adjusted_graph_sha256"]
    )
    assert graph == before


def test_explicit_contact_permissions_cannot_override_profile_ownership():
    assert merge_contact_permissions({"source": 8}, {"source": 8, "other": 4}) == {
        "other": 4,
        "source": 8,
    }
    with pytest.raises(ValueError, match="Conflicting"):
        merge_contact_permissions({"source": 4}, {"source": 8})
    with pytest.raises(ValueError, match="integer revisions"):
        merge_contact_permissions(None, {"source": True})


@pytest.mark.parametrize("one_physical_arm", [False, True])
def test_idle_clearance_counts_declared_motion_members_once_without_grasp(
    one_physical_arm,
):
    case = next(
        row
        for row in REFERENCE["cases"]
        if row["source_case"] == "same_object_reacquisition"
    )
    graph = deepcopy(case["graph"])
    embodiment = deepcopy(REFERENCE["embodiment"])
    resources = embodiment["skill_profile"]["resources"]
    if one_physical_arm:
        resources[1]["members"] = ["left"]
        resources[1]["endpoints"] = deepcopy(resources[0]["endpoints"])
    resources.extend(
        [
            dict(resource_id="working", members=["left"], endpoints=[]),
            dict(resource_id="resting", members=["right"], endpoints=[]),
            dict(resource_id="paired", members=["left", "right"], endpoints=[]),
        ]
    )
    for node in graph["nodes"]:
        node["call"]["resources"] = {
            role: {"left": "working", "right": "resting"}[resource]
            for role, resource in node["call"]["resources"].items()
        }
    original = deepcopy((graph, embodiment))
    adjusted, idle = add_reacquisition_clearance(graph, embodiment)
    assert idle == ([] if one_physical_arm else case["idle_routes"])
    if one_physical_arm:
        assert adjusted == graph
    else:
        clear = next(
            row for row in adjusted["nodes"] if row.get("role") == "idle_clearance"
        )
        assert clear["call"]["resources"] == {"primary": "right"}
    assert (graph, embodiment) == original


def test_post_handover_declaration_retains_receiver_and_support_scope():
    graph = {
        "nodes": [
            {
                "id": "transfer",
                "call": {
                    "kind": "hand_over",
                    "object": "body",
                    "resources": {"source": "a", "destination": "b"},
                },
            },
            {
                "id": "release",
                "call": {
                    "kind": "registered",
                    "call_id": "simulation.place_relative",
                    "arguments": {
                        "object": "body",
                        "reference": "support",
                        "relation": "on",
                    },
                    "resources": {"primary": "b"},
                },
            },
        ]
    }
    assert validate_post_handover_place_declarations(
        graph, ["release"], frozenset({"body", "support"})
    ) == frozenset({"release"})
    wrong = deepcopy(graph)
    wrong["nodes"][1]["call"]["resources"]["primary"] = "a"
    with pytest.raises(ValueError, match="receiver"):
        validate_post_handover_place_declarations(
            wrong, ["release"], frozenset({"body", "support"})
        )
    with pytest.raises(ValueError, match="missing"):
        validate_post_handover_place_declarations(
            graph, ["absent"], frozenset({"body", "support"})
        )


@pytest.mark.parametrize(
    "purpose,expected",
    [("handover_source", True), ("ordinary", False), ("none", False)],
)
def test_post_handover_permission_retains_declared_end_grip_provenance(
    purpose, expected
):
    graph = {
        "nodes": [
            {
                "id": "pick",
                "call": {
                    "kind": "registered",
                    "call_id": "semantic.pick",
                    "arguments": {"object": "body"},
                    "resources": {"primary": "a"},
                },
            },
            {
                "id": "transfer",
                "call": {
                    "kind": "hand_over",
                    "object": "body",
                    "resources": {"source": "a", "destination": "b"},
                },
            },
            {
                "id": "place",
                "call": {
                    "kind": "registered",
                    "call_id": "simulation.place_relative",
                    "arguments": {"object": "body"},
                    "resources": {"primary": "b"},
                },
            },
        ]
    }
    result = select_end_grip_post_handover_places(
        graph,
        {"semantic.pick": {"kind": "pick_up", "adaptive_grasp": purpose}},
        frozenset({"place"}),
    )
    assert bool(result) is expected


@pytest.mark.parametrize(
    "case",
    json.loads(
        (Path(__file__).parent / "data/released_hand_bindings_b0.json").read_text()
    )["cases"],
    ids=lambda row: row["source_case"],
)
def test_released_hand_routes_match_original_occurrence_binder(case):
    """Freeze adjacency, Park order and distinct resource rejection from B0."""
    from embodichain.gen_sim.task_engine.continuation_declarations import (
        resolve_released_hand_declarations,
    )
    import re

    graph = deepcopy(case.get("graph", GRAPHS.get(case["source_case"])))
    before = deepcopy(graph)
    occurrences = {
        node["id"]: f"source_program/{node['id']}:0" for node in graph["nodes"]
    }
    if "error" in case:
        with pytest.raises(ValueError, match=re.escape(case["error"])):
            resolve_released_hand_declarations(
                graph,
                program_fingerprint="source-fingerprint",
                occurrences=occurrences,
            )
    else:
        assert (
            resolve_released_hand_declarations(
                graph,
                program_fingerprint="source-fingerprint",
                occurrences=occurrences,
            )
            == case["expected"]
        )
    assert graph == before
