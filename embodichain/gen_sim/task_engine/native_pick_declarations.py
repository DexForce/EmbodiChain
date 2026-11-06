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

"""Explicit native Pick permissions and final-release reference declarations."""

from __future__ import annotations
from collections.abc import Mapping, Sequence
from typing import Any

__all__ = ["resolve_native_pick_revisions", "resolve_native_pick_bindings"]


def resolve_native_pick_revisions(
    graph: Mapping[str, Any],
    default_revision: int | None,
    overrides: Mapping[str, int] | None,
) -> dict[str, int]:
    """Validate source global/per-occurrence permissions without reading runtime IDs.

    Args:
        graph: Canonical source nodes and stable public semantic aliases.
        default_revision: Original optional global closing revision 1 through 9.
        overrides: Explicit source-node selections; registered source aliases
            retain their original revision-eight qualification boundary.

    Returns:
        Per-node revisions for declared Pick calls only.
    """
    if default_revision is not None and (
        type(default_revision) is not int or default_revision not in range(1, 10)
    ):
        raise ValueError("Pick contact revision must be an integer from 1 through 9.")
    if overrides is not None and not isinstance(overrides, Mapping):
        raise ValueError("Pick contact revisions require a source-node mapping.")
    remaining = dict(overrides or {})
    if any(
        type(name) is not str
        or not name
        or type(value) is not int
        or value not in range(1, 10)
        for name, value in remaining.items()
    ):
        raise ValueError(
            "Pick contact revisions require exact node identities and integer revisions 1-9."
        )
    result = {}
    for node in graph["nodes"]:
        call = node["call"]
        identity = call.get("call_id", "")
        alias = identity == "simulation.pick" or identity.startswith("gen_sim.pick.")
        if node["id"] in remaining:
            value = remaining.pop(node["id"])
            end_source = (
                identity == "gen_sim.pick.handover_source"
                or identity.startswith("gen_sim.pick.handover_horizontal_source.")
            )
            if call["kind"] != "pick" and not (end_source and value == 8):
                raise ValueError(
                    "Per-call native revisions require plain Pick or revision-eight handover source."
                )
            result[node["id"]] = value
        elif default_revision is not None and (call["kind"] == "pick" or alias):
            result[node["id"]] = default_revision
    if remaining:
        raise ValueError("Pick contact revision names a missing source node.")
    return result


def resolve_native_pick_bindings(
    graph: Mapping[str, Any],
    *,
    revisions: Mapping[str, int],
    occurrences: Mapping[str, str],
    options: Mapping[str, Mapping[str, Any]],
    continuations: Sequence[Mapping[str, Any]],
    scene_binding: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Bind native inputs and original ordinary-on opening targets explicitly.

    Args:
        graph: Canonical source calls.
        revisions: Already validated source node revision permissions.
        occurrences: Public compiled call identities.
        options: Effective source action option declarations by node.
        continuations: Exact source/Place continuation declarations.
        scene_binding: Canonical rigid and support-affordance owners.

    Returns:
        Closed native Pick service declarations; final placed poses remain in
        the request's grounded downstream tuple, never copied from static analysis.
    """
    nodes = {node["id"]: node for node in graph["nodes"]}
    by_occurrence = {value: key for key, value in occurrences.items()}
    follows = {
        row["source_invocation_id"]: row
        for row in continuations
        if row.get("kind", "pick") == "pick"
    }
    rigid = {item["entity_id"]: item for item in scene_binding["rigid_objects"]}
    result = []
    for name, revision in revisions.items():
        call = nodes[name]["call"]
        subject = call.get("object", call.get("arguments", {}).get("object"))
        row = dict(invocation_id=occurrences[name], object_id=subject)
        continuation = follows.get(occurrences[name])
        if (
            revision >= 3
            and options[name].get("adaptive_grasp") == "ordinary"
            and continuation is not None
        ):
            place = nodes[by_occurrence[continuation["place_invocation_id"]]]["call"]
            parent = None
            if place["kind"] == "place" and place.get("on") is not None:
                target = place["on"]
                if target in rigid:
                    parent = target
                else:
                    for identity, owner in rigid.items():
                        if target == f"on__{identity}__{subject}" and any(
                            aff.get("entity_id") == target
                            for aff in owner.get("affordances", ())
                        ):
                            parent = identity
            elif place.get("call_id") == "simulation.place_relative":
                args = place["arguments"]
                if args.get("relation") == "on" and args.get("reference_part") is None:
                    parent = args.get("reference")
            if parent in rigid:
                row.update(
                    opening_reference_id=parent,
                    opening_place_invocation_id=continuation["place_invocation_id"],
                )
        result.append(row)
    return result
