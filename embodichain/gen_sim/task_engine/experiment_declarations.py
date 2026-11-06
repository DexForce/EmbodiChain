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

"""Pure source experiment selections over semantic and physical declarations."""

from __future__ import annotations
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any
from embodichain.lab.sim.scene.declaration_placement import (
    source_horizontal_slender_object,
)

__all__ = [
    "horizontal_handover_permissions",
    "add_reacquisition_clearance",
    "horizontal_on_pick_permissions",
    "merge_contact_permissions",
    "validate_post_handover_place_declarations",
    "select_end_grip_post_handover_places",
]


def horizontal_handover_permissions(
    graph: Mapping[str, Any],
) -> tuple[list[str], dict[str, int]]:
    """Derive permissions from the end-grasp recipe, never node or object names."""
    objects = {
        n["call"]["arguments"]["object"]
        for n in graph["nodes"]
        if n["call"]
        .get("call_id", "")
        .startswith("gen_sim.pick.handover_horizontal_source.")
    }
    revisions = {
        n["id"]: 8
        for n in graph["nodes"]
        if n["call"]
        .get("call_id", "")
        .startswith("gen_sim.pick.handover_horizontal_source.")
    }
    places = []
    for previous, node in zip(graph["nodes"], graph["nodes"][1:]):
        before, call = previous["call"], node["call"]
        args = call.get("arguments", {})
        if (
            call.get("call_id") == "simulation.place_relative"
            and args.get("object") in objects
            and args.get("relation") == "on"
            and before["kind"] == "hand_over"
            and before.get("object") == args["object"]
            and before.get("resources", {}).get("destination")
            == call.get("resources", {}).get("primary")
        ):
            places.append(node["id"])
    return places, revisions


def add_reacquisition_clearance(
    graph: dict[str, Any], embodiment: Mapping[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Clear an otherwise unused arm before a terminal same-object reacquisition."""
    from copy import deepcopy

    from embodichain.lab.task_program.integrations.simulation.idle_clearance import (
        CALL_ID,
    )

    result = deepcopy(graph)
    nodes = result["nodes"]
    group = result["task_groups"][-1]
    members = [node for node in nodes if node["id"] in group["node_ids"]]
    first, last = members[0], members[-1]
    call = first["call"]
    if group["task_type"] != "E1" or call["kind"] != "pick":
        return graph, []
    prefix = nodes[: nodes.index(first)]
    if not any(
        n["call"].get("object") == call["object"] and n["call"]["kind"] == "pick"
        for n in prefix
    ):
        return graph, []
    declared = {
        row["resource_id"]: row for row in embodiment["skill_profile"]["resources"]
    }
    resources = {
        identity: row
        for identity, row in declared.items()
        if not row.get("members")
        and any(e["endpoint_id"] == "motion" for e in row.get("endpoints", ()))
    }

    def motion_leaves(
        identity: str, seen: frozenset[str] = frozenset()
    ) -> frozenset[str]:
        if identity in seen:
            raise ValueError("Motion resource members must not be cyclic.")
        members = declared[identity].get("members", ())
        if members:
            return frozenset(
                leaf
                for member in members
                for leaf in motion_leaves(member, seen | {identity})
            )
        return frozenset({identity}) if identity in resources else frozenset()

    active_leaves = motion_leaves(call["resources"]["primary"])
    if len(active_leaves) != 1:
        return graph, []
    active = next(iter(active_leaves))
    unused = set(resources) - {active}
    if len(unused) != 1:
        return graph, []
    idle = unused.pop()
    last_use = next(
        (
            node
            for node in reversed(prefix)
            if any(
                idle in motion_leaves(resource)
                for resource in node["call"].get("resources", {}).values()
            )
        ),
        None,
    )
    if last_use is not None and last_use["call"].get("call_id") != "simulation.park":
        return graph, []

    def motion(resource):
        return next(
            e["control_part"]
            for e in resources[resource]["endpoints"]
            if e["endpoint_id"] == "motion"
        )

    clear_id, restore_id = (
        first["id"] + "__idle_clearance",
        last["id"] + "__idle_restore",
    )
    route_id = "clear_" + group["id"] + "_" + idle
    clear, restore = deepcopy(first), deepcopy(last)
    clear.update(
        id=clear_id,
        role="idle_clearance",
        call={
            "kind": "registered",
            "call_id": CALL_ID,
            "arguments": {"route": route_id},
            "resources": {"primary": idle},
        },
    )
    restore.update(
        id=restore_id,
        role="idle_restore",
        depends_on=[last["id"]],
        call={
            "kind": "registered",
            "call_id": "simulation.park",
            "arguments": {},
            "resources": {"primary": idle},
        },
    )
    first["depends_on"] = [clear_id]
    start, end = nodes.index(first), nodes.index(last)
    result["nodes"] = [
        *nodes[:start],
        clear,
        *nodes[start : end + 1],
        restore,
        *nodes[end + 1 :],
    ]
    group["node_ids"] = [clear_id, *group["node_ids"], restore_id]
    route = {
        "route_id": route_id,
        "clear_segment": clear_id,
        "restore_segment": restore_id,
        "resource": idle,
        "joint_name": embodiment["simulation"]["control_parts"][motion(idle)][0],
        "offset_rad": 0.6,
        "search": True,
        "protected_control_part": motion(active),
    }
    return result, [route]


def horizontal_on_pick_permissions(
    graph: Mapping[str, Any],
    scene_objects: Mapping[str, Mapping[str, Any]],
) -> dict[str, int]:
    """Declare native revision eight only for the source horizontal rigid on-pair.

    Args:
        graph: Canonical ordered semantic calls.
        scene_objects: Owner-resolved physical declarations by logical identity.

    Returns:
        Per-source-node native revision selections. Geometry qualification remains
        in shared Scene/Compute; task or object names do not infer shape/capability.
    """
    result = {}
    for node, following in zip(graph["nodes"], graph["nodes"][1:]):
        call, place = node["call"], following["call"]
        args = place.get("arguments", {})
        if (
            node.get("task_type") != "E1"
            or call["kind"] != "pick"
            or call.get("grasp") is not None
            or place.get("call_id") != "simulation.place_relative"
            or args.get("relation") != "on"
            or args.get("object") != call["object"]
            or place.get("resources", {}).get("primary")
            != call.get("resources", {}).get("primary")
        ):
            continue
        if source_horizontal_slender_object(scene_objects[call["object"]]):
            result[node["id"]] = 8
    return result


def merge_contact_permissions(
    selected: Mapping[str, int] | None,
    generated: Mapping[str, int],
) -> dict[str, int]:
    """Merge independently selected revisions without silently overriding either.

    Args:
        selected: Explicit caller permissions, if any.
        generated: Source experiment's semantic/geometry-derived permissions.

    Returns:
        Sorted unique exact permissions. Conflicting revisions reject.
    """
    if selected is not None and not isinstance(selected, Mapping):
        raise ValueError("Selected Pick contact revisions must be a mapping.")
    result = {}
    for declaration in (selected or {}, generated):
        if not isinstance(declaration, Mapping) or any(
            type(name) is not str
            or not name
            or type(revision) is not int
            or revision not in range(1, 10)
            for name, revision in declaration.items()
        ):
            raise ValueError(
                "Pick contact revisions require exact source nodes and integer revisions 1-9."
            )
        for name, revision in declaration.items():
            if name in result and result[name] != revision:
                raise ValueError(f"Conflicting Pick contact revisions for {name}.")
            result[name] = revision
    return dict(sorted(result.items()))


def validate_post_handover_place_declarations(
    graph: Mapping[str, Any],
    selections: Sequence[str] | None,
    rigid_ids: frozenset[str],
) -> frozenset[str]:
    """Validate the exact original adjacent receiver/rigid-support on-release.

    Args:
        graph: Canonical source call sequence.
        selections: Explicit source Place node IDs, including profile-generated IDs.
        rigid_ids: Exact physical rigid identities from the selected scene owner.

    Returns:
        Validated immutable Place node identities. Resource/object/support drift
        rejects before publication; no runtime strategy is inferred by a name.
    """
    if selections is None:
        return frozenset()
    if (
        type(selections) not in (tuple, list)
        or any(
            type(name) is not str or not name or name != name.strip()
            for name in selections
        )
        or len(set(selections)) != len(selections)
    ):
        raise ValueError("Post-handover Place requires unique exact source nodes.")
    remaining = set(selections)
    for previous, node in zip(graph["nodes"], graph["nodes"][1:]):
        if node["id"] not in remaining:
            continue
        transfer, place = previous["call"], node["call"]
        args = place.get("arguments", {})
        resources = transfer.get("resources", {})
        if (
            transfer["kind"] != "hand_over"
            or transfer.get("final_target") is not None
            or set(resources) != {"source", "destination"}
            or resources["source"] == resources["destination"]
            or place.get("call_id") != "simulation.place_relative"
            or set(args) != {"object", "reference", "relation"}
            or args["relation"] != "on"
            or args["object"] != transfer["object"]
            or place.get("resources", {}) != {"primary": resources["destination"]}
            or args["object"] not in rigid_ids
            or args["reference"] not in rigid_ids
            or args["object"] == args["reference"]
        ):
            raise ValueError(
                "Post-handover Place must preserve the exact receiver, object and rigid support."
            )
        remaining.remove(node["id"])
    if remaining:
        raise ValueError(
            "Post-handover Place references a missing or non-adjacent source node."
        )
    return frozenset(selections)


def select_end_grip_post_handover_places(
    graph: Mapping[str, Any],
    action_options: Mapping[str, Mapping[str, Any]],
    declarations: frozenset[str],
) -> frozenset[str]:
    """Keep explicit release permission inactive for a different source grip mode.

    Args:
        graph: Canonical serial call sequence from an empty source task state.
        action_options: Existing alias option declarations owning acquisition mode.
        declarations: Structurally validated adjacent post-handover Place nodes.

    Returns:
        Only permissions reached through the original end-source acquisition.
        Ordinary/constrained source grips keep their existing Place path. This
        is declaration provenance, not a replacement for runtime held-state checks.
    """
    held = {}
    result = set()
    for index, node in enumerate(graph["nodes"]):
        call = node["call"]
        identity = call.get("call_id", call["kind"])
        subject = call.get("object", call.get("arguments", {}).get("object"))
        resources = call.get("resources", {})
        options = action_options.get(identity, {})
        if options.get("kind") == "pick_up":
            held[(subject, resources.get("primary", "left"))] = (
                options.get("adaptive_grasp") == "handover_source"
            )
        elif call["kind"] == "hand_over":
            mode = held.pop((subject, resources.get("source", "left")), False)
            held[(subject, resources.get("destination", "right"))] = mode
            if (
                mode
                and index + 1 < len(graph["nodes"])
                and graph["nodes"][index + 1]["id"] in declarations
            ):
                result.add(graph["nodes"][index + 1]["id"])
        elif call["kind"] == "place" or identity in {
            "simulation.place_relative",
            "gen_sim.place_upright",
            "gen_sim.stack_place",
        }:
            held.pop((subject, resources.get("primary", "left")), None)
    return frozenset(result)
