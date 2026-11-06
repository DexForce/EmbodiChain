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

"""Select source continuation capabilities using exact semantic occurrences."""

from __future__ import annotations
from collections.abc import Mapping, Sequence
from typing import Any

__all__ = [
    "resolve_source_continuation_declarations",
    "source_pick_release_pairs",
    "resolve_inside_yaw_declarations",
    "resolve_measured_rigid_declarations",
    "resolve_released_hand_declarations",
]

_PLACES = frozenset(
    {"simulation.place_relative", "gen_sim.place_upright", "gen_sim.stack_place"}
)


def resolve_source_continuation_declarations(
    graph: Mapping[str, Any],
    *,
    program_fingerprint: str,
    occurrences: Mapping[str, str],
    options: Mapping[str, Mapping[str, Any]],
    motions: Mapping[str, Mapping[str, Any]],
    recoveries: Mapping[str, Mapping[str, Any]],
    staging_bindings: Sequence[Mapping[str, Any]],
    deferred_picks: frozenset[str] = frozenset(),
) -> list[dict[str, Any]]:
    """Declare only complete original Pick/align-to-Place capability selections.

    Args:
        graph: Source serial calls with exact object/resource declarations.
        program_fingerprint: Identity from public compile-only composition.
        occurrences: Compiled identity for each unique node.
        options: Effective typed-option-shaped declaration per node.
        motions: Effective motion declarations per node.
        recoveries: Selected planning recovery declaration per node.
        staging_bindings: Explicit same-resource retained predecessor declarations.
        deferred_picks: Exact drawer acquisition nodes deferring all lookahead.

    Returns:
        Closed public continuation bindings. Runtime resolves goals and validates
        actual references; this layer neither plans nor grounds future actions.
    """
    result = []
    nodes = graph["nodes"]
    retained = {
        row["invocation_id"]
        for row in staging_bindings
        if row.get("predecessor_invocation_id") is not None
    }
    for index, node in enumerate(nodes):
        call = node["call"]
        identity = call.get("call_id", call["kind"])
        name = node["id"]
        option = options[name]
        motion = motions[name]
        rule = recoveries[name]
        subject = call.get("object", call.get("arguments", {}).get("object"))
        resources = call.get("resources", {})
        if set(resources) != {"primary"} or name in deferred_picks:
            continue
        kind = None
        if (
            call["kind"] == "pick"
            and option.get("adaptive_grasp") == "ordinary"
            and not any(
                option.get(field)
                for field in (
                    "linear_segments",
                    "constrained_grasp",
                    "qualify_fixed_grasp",
                )
            )
            and all(
                option.get(field) is None
                for field in (
                    "fixed_object_to_eef",
                    "rotate_upright",
                    "obj_upright_direction",
                )
            )
            and option.get("pick_object_part", "center") == "center"
            and not call.get("grasp")
            and rule.get("kind") == "pick_bounded"
            and motion.get("strategy") == "ik_interp"
            and motion.get("dynamic_collision_mode", "off") == "off"
        ):
            kind = "pick"
        elif (
            identity == "gen_sim.align_held"
            and call["arguments"].get("preserve_yaw") is False
            and rule.get("kind") == "held_alignment_bounded"
            and rule.get("staging") is True
            and occurrences[name] in retained
            and option.get("report_retained_attachment") is True
            and motion.get("sample_count") == 260
        ):
            kind = "held_alignment"
        if kind is None or index + 1 >= len(nodes):
            continue
        next_index = index + 1
        transport = None
        following = nodes[next_index]["call"]
        if kind == "pick" and following.get("call_id") == "simulation.move_held_object":
            if (
                following.get("arguments", {}).get("object") != subject
                or following.get("resources", {}) != resources
                or next_index + 1 >= len(nodes)
            ):
                continue
            transport = nodes[next_index]["id"]
            next_index += 1
        place_node = nodes[next_index]
        place = place_node["call"]
        registered = place.get("call_id") in _PLACES
        if (
            kind == "pick"
            and registered
            and (
                place.get("call_id") != "simulation.place_relative"
                or place.get("arguments", {}).get("reference_part") is not None
            )
        ):
            continue
        if not (registered or (kind == "pick" and place["kind"] == "place")):
            continue
        if (
            place.get("object", place.get("arguments", {}).get("object")) != subject
            or place.get("resources", {}) != resources
        ):
            continue
        if (
            kind == "held_alignment"
            and motions[place_node["id"]].get("sample_count") != 260
        ):
            continue
        declaration = dict(
            program_fingerprint=program_fingerprint,
            source_invocation_id=occurrences[name],
            place_invocation_id=occurrences[place_node["id"]],
        )
        if kind != "pick":
            declaration["kind"] = kind
        if transport is not None:
            declaration["transport_invocation_id"] = occurrences[transport]
        result.append(declaration)
    return result


def source_pick_release_pairs(
    graph: Mapping[str, Any], effect_monitors: Mapping[str, Any]
) -> tuple[tuple[str, str], ...]:
    """Find source structural preview pairs independently of planner eligibility.

    Args:
        graph: Canonical serial calls with exact object/resource declarations.
        effect_monitors: Source preset's explicitly declared monitor identities.

    Returns:
        Pick/Place node pairs owning the original whole-request wall budget.
        A paired budget does not imply permission to run prospective recovery.
    """
    result = []
    nodes = graph["nodes"]
    for index, node in enumerate(nodes[:-1]):
        pick = node["call"]
        if pick["kind"] != "pick" or set(pick.get("resources", {})) != {"primary"}:
            continue
        following = index + 1
        transport = None
        middle = nodes[following]["call"]
        if middle.get("call_id") == "simulation.move_held_object":
            args = middle["arguments"]
            if (
                following + 1 >= len(nodes)
                or set(args) - {"object", "target", "reference"}
                or not {"object", "target"} <= set(args)
                or args["object"] != pick["object"]
                or middle.get("resources", {}) != pick["resources"]
            ):
                continue
            transport = middle["call_id"]
            following += 1
        place = nodes[following]["call"]
        identity = place.get("call_id", place["kind"])
        if place["kind"] == "place":
            subject = place["object"]
        elif identity == "simulation.place_relative":
            args = place["arguments"]
            if (
                not {"object", "reference", "relation"} <= set(args)
                or set(args)
                - {"object", "reference", "relation", "stage_id", "reference_part"}
                or args.get("reference_part") is not None
            ):
                continue
            subject = args["object"]
        else:
            continue
        if (
            subject == pick["object"]
            and place.get("resources", {}) == pick["resources"]
            and all(name in effect_monitors for name in ("pick", identity))
            and (transport is None or transport in effect_monitors)
        ):
            result.append((node["id"], nodes[following]["id"]))
    return tuple(result)


def resolve_inside_yaw_declarations(
    graph: Mapping[str, Any],
    selections: Sequence[str] | None,
    *,
    scene_binding: Mapping[str, Any],
    effect_monitors: Mapping[str, Any],
    verified: bool,
) -> dict[str, tuple[str, str]]:
    """Bind source inside-yaw permissions to ordinary structural rigid pairs.

    Args:
        graph: Canonical source semantic calls.
        selections: Explicit unique inside Place node IDs; None selects nothing.
        scene_binding: Exact canonical rigid/container declarations.
        effect_monitors: Declared source monitor IDs for structural pairing.
        verified: Whether the selected execution policy owns verified effects.

    Returns:
        Place node mapped to its source subject/container IDs. Explicit permissions
        require the original ordinary Pick/release pair, rigid parent and 12 cm
        release contract. This does not select a yaw or inspect runtime geometry.
    """
    if selections is None:
        return {}
    if (
        type(selections) not in (list, tuple)
        or any(
            type(name) is not str or not name or name != name.strip()
            for name in selections
        )
        or len(set(selections)) != len(selections)
    ):
        raise ValueError("World yaw freedom requires unique exact inside Place nodes.")
    if selections and not verified:
        raise ValueError("World yaw freedom requires a verified ordinary rigid pair.")
    nodes = {node["id"]: node for node in graph["nodes"]}
    pairs = {
        placed: picked
        for picked, placed in source_pick_release_pairs(graph, effect_monitors)
    }
    rigid = {value["entity_id"]: value for value in scene_binding["rigid_objects"]}
    containers = {
        affordance["entity_id"]: (identifier, affordance)
        for identifier, value in rigid.items()
        for affordance in value.get("affordances", ())
        if affordance.get("kind") == "container"
    }
    result = {}
    for name in selections:
        node = nodes.get(name)
        call = {} if node is None else node["call"]
        picked = pairs.get(name)
        pick = {} if picked is None else nodes[picked]["call"]
        container = containers.get(call.get("inside"))
        if (
            call.get("kind") != "place"
            or container is None
            or pick.get("kind") != "pick"
            or pick.get("grasp") is not None
            or nodes[picked]["task_type"] == "E3"
            or call.get("object") not in rigid
            or container[1].get("release_clearance") != 0.12
        ):
            raise ValueError(
                "World yaw freedom requires its bound ordinary rigid inside route."
            )
        result[name] = (call["object"], call["inside"])
    return result


def resolve_measured_rigid_declarations(
    graph: Mapping[str, Any],
    selections: Sequence[str] | None,
    *,
    scene_binding: Mapping[str, Any],
    native_revisions: Mapping[str, int],
) -> dict[str, dict[str, str]]:
    """Validate original explicit ordinary Pick/rigid-inside measurement scope.

    Args:
        graph: Canonical source call declarations.
        selections: Unique explicit source node identities, or None.
        scene_binding: Exact physical rigid/container ownership declarations.
        native_revisions: Already resolved source native-closing permissions.

    Returns:
        Per-node subject and optional inside target identity, without compiling,
        planning, sampling or reading future state. Native preshape uses only its
        qualified original revision-four closing contract when explicitly selected.
    """
    if selections is None:
        return {}
    if (
        type(selections) not in (list, tuple)
        or any(
            type(name) is not str or not name or name != name.strip()
            for name in selections
        )
        or len(set(selections)) != len(selections)
    ):
        raise ValueError("Measured rigid calls require unique exact source nodes.")
    rigid = {value["entity_id"]: value for value in scene_binding["rigid_objects"]}
    containers = {
        affordance["entity_id"]: owner
        for owner, value in rigid.items()
        for affordance in value.get("affordances", ())
        if affordance.get("kind") == "container"
    }
    nodes = {node["id"]: node for node in graph["nodes"]}
    result = {}
    for name in selections:
        node = nodes.get(name)
        call = {} if node is None else node["call"]
        if call.get("object") not in rigid:
            raise ValueError(
                "Measured manipulation requires exact rigid scene bindings."
            )
        if (
            call.get("kind") == "pick"
            and node["task_type"] != "E3"
            and call.get("grasp") is None
        ):
            if native_revisions.get(name) not in (None, 4):
                raise ValueError(
                    "Measured preshape requires the qualified revision4 closing contract."
                )
            result[name] = dict(object_id=call["object"])
        elif call.get("kind") == "place" and call.get("inside") in containers:
            result[name] = dict(object_id=call["object"], container_id=call["inside"])
        else:
            raise ValueError(
                "Measured calls require ordinary Pick or rigid inside Place."
            )
    return result


def resolve_released_hand_declarations(
    graph: Mapping[str, Any],
    *,
    program_fingerprint: str,
    occurrences: Mapping[str, str],
) -> list[dict[str, Any]]:
    """Declare the original adjacent release/clearance/one-or-two-Park sequence.

    Args:
        graph: Canonical serial source calls with explicit resource identities.
        program_fingerprint: Complete provider-free compiled program identity.
        occurrences: Exact current occurrence identity for every source node.

    Returns:
        Closed Lab bindings preserving source Park order. A selected cleanup
        with a missing/mismatched release or Park suffix rejects before runtime;
        no task number, object name or cached analysis position selects strategy.
    """
    result = []
    nodes = graph["nodes"]
    releases = {
        "simulation.place_relative",
        "gen_sim.place_upright",
        "gen_sim.stack_place",
    }
    for index, node in enumerate(nodes):
        call = node["call"]
        if call.get("call_id") != "gen_sim.clear_released":
            continue
        parks = []
        for following in nodes[index + 1 :]:
            park = following["call"]
            if park.get("call_id") != "simulation.park":
                break
            if (
                park["kind"] != "registered"
                or park.get("arguments", {})
                or set(park.get("resources", {})) != {"primary"}
            ):
                raise ValueError(
                    "Released-hand clearance requires explicit pure parks."
                )
            parks.append(following)
        resources = [park["call"]["resources"] for park in parks]
        if (
            call["kind"] != "registered"
            or set(call.get("resources", {})) != {"primary"}
            or not 1 <= len(parks) <= 2
            or len({row["primary"] for row in resources}) != len(parks)
            or call["resources"] not in resources
        ):
            raise ValueError(
                "Released-hand clearance requires at most two distinct sequential parks including its explicit resource."
            )
        previous = nodes[index - 1] if index else None
        release = {} if previous is None else previous["call"]
        if (
            release.get("kind") not in {"registered", "place"}
            or release.get("kind") == "registered"
            and release.get("call_id") not in releases
            or release.get("object", release.get("arguments", {}).get("object"))
            != call.get("arguments", {}).get("object")
            or release.get("resources", {}) != call["resources"]
        ):
            raise ValueError(
                "Released-hand clearance requires its preceding same-object release."
            )
        result.append(
            dict(
                program_fingerprint=program_fingerprint,
                release_invocation_id=occurrences[previous["id"]],
                clearance_invocation_id=occurrences[node["id"]],
                park_invocation_ids=[occurrences[park["id"]] for park in parks],
            )
        )
    return result
