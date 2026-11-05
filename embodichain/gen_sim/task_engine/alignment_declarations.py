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

"""Source Pick, alignment and released-hand semantic declarations."""

from __future__ import annotations
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any
from embodichain.lab.task_program.integrations.hand_calibration import (
    TaskHandCalibration,
)

__all__ = [
    "resolve_alignment_pick_declarations",
    "resolve_alignment_lowerer_declarations",
    "resolve_handover_profile_declarations",
]


def _target(graph: Mapping[str, Any], identity: str) -> dict[str, Any]:
    target = graph["targets"][identity]
    if target["kind"] != "cyclic_pose" or len(target["values"]) != 1:
        raise ValueError("Source alignment requires one exact declared pose.")
    return deepcopy(target["values"][0])


def resolve_alignment_pick_declarations(
    graph: Mapping[str, Any],
    *,
    table_top_z: float,
    hands: Sequence[TaskHandCalibration],
    default_pick_options: Mapping[str, Any],
    source_objects: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Declare exact source alias policies and upper-half release constraints.

    Args:
        graph: Canonical source graph with explicit per-operation aliases.
        table_top_z: Frozen source scene height, not a guessed global constant.
        hands: Explicit model-owned nominal hand envelopes.
        default_pick_options: Neutral source Pick option declaration.
        source_objects: Explicit authored source facts for constrained aliases.
            Existing ordinary E2 calls may omit these geometry declarations.

    Returns:
        Alias option templates and closed public Pick lowerer declarations.
    """
    from embodichain.lab.sim.scene.source_target_geometry import (
        source_constrained_pick_geometry,
    )

    options, routes = {}, {}
    upright_objects = set()
    supported_groups = {
        node["task_instance_id"]
        for node in graph["nodes"]
        if node["call"].get("call_id") == "gen_sim.place_upright"
    }
    for node in graph["nodes"]:
        call = node["call"]
        identity = call.get("call_id", "")
        if identity in {"simulation.axis_align", "gen_sim.align_held"} or (
            identity == "simulation.move_held_object" and node["task_type"] == "E2"
        ):
            upright_objects.add(call["arguments"]["object"])
        if identity != "simulation.pick" and not identity.startswith("gen_sim.pick."):
            continue
        args = call["arguments"]
        route = dict(object_id=args["object"], target_id=args["target"])
        if identity == "gen_sim.pick.handover_source" or identity.startswith(
            "gen_sim.pick.handover_horizontal_source."
        ):
            selected = {
                **deepcopy(dict(default_pick_options)),
                "pick_object_part": "center",
                "adaptive_grasp": "handover_source",
            }
        else:
            pose = _target(graph, args["target"])
            supported = node["task_instance_id"] in supported_groups
            if source_objects is not None:
                approach, plane = source_constrained_pick_geometry(
                    source_objects[args["object"]],
                    table_top_z=table_top_z,
                    release_origin_z=pose["position"][2],
                    previously_aligned=args["object"] in upright_objects,
                    force_vertical=node["task_type"] == "E2" or supported,
                    supported_upright=supported,
                    ordinary_release_clearance=0.01,
                    slender_release_clearance=0.01,
                )
            elif node["task_type"] == "E2" and not supported:
                approach, plane = [0.0, 0.0, -1.0], float(table_top_z)
            else:
                raise ValueError(
                    "Non-upright constrained aliases require their explicit geometry declaration."
                )
            selected = dict(
                kind="pick_up",
                hand_interp_steps=default_pick_options["hand_interp_steps"],
                pre_grasp_distance=0.08,
                grasp_settle_steps=16,
                pick_object_part="center",
                approach_direction=approach,
            )
            route.update(
                release_clearance_object_pose={
                    "kind": "pose",
                    **pose,
                },
                release_clearance_plane_z=plane,
                release_clearance_safety_margin=0.02,
                grasp_region="upper_half",
            )
        if identity in options and options[identity] != selected:
            raise ValueError("Distinct source Pick options require distinct aliases.")
        options[identity] = selected
        routes.setdefault(identity, []).append(route)
    return options, [
        dict(
            kind="pick",
            call_id=identity,
            routes=values,
            verify_phases=False,
            hands=[hand.nominal_envelope_declaration() for hand in hands],
        )
        for identity, values in routes.items()
    ]


def resolve_alignment_lowerer_declarations(
    graph: Mapping[str, Any],
    *,
    axes: Mapping[str, Sequence[float]],
    hands: Sequence[TaskHandCalibration],
) -> list[dict[str, Any]]:
    """Declare source held alignment and released-hand routes from semantic targets.

    Args:
        graph: Canonical semantic graph and source pose targets.
        axes: Axis declarations from the existing stability contract owner.
        hands: Explicit model-owned nominal released-hand envelopes.

    Returns:
        Shared alias factories retaining source insertion order and identities.
    """
    align, clear = {}, {}
    for node in graph["nodes"]:
        call = node["call"]
        identity = call.get("call_id")
        if identity not in {"gen_sim.align_held", "gen_sim.clear_released"}:
            continue
        args = call["arguments"]
        obj, target = args["object"], args["target"]
        if identity == "gen_sim.clear_released":
            clear[(obj, target)] = dict(
                object_id=obj,
                target_id=target,
                minimum_height=float(_target(graph, target)["position"][2]),
            )
            continue
        preserve = args.get("preserve_yaw")
        direction = args.get("axis_direction_free", False)
        if type(preserve) is not bool or type(direction) is not bool or obj not in axes:
            raise ValueError(
                "Held alignment requires explicit yaw/direction policy and axis."
            )
        route = dict(
            object_id=obj,
            target_id=target,
            preserve_yaw=preserve,
            axis_direction_free=direction,
            local_axis=list(axes[obj]),
        )
        if target != "current_object_pose":
            route["position"] = _target(graph, target)["position"]
        align[(obj, target, preserve, direction)] = route
    result = []
    if align:
        result.append(
            dict(kind="align_held", verify_retention=True, routes=list(align.values()))
        )
    if clear:
        result.append(
            dict(
                kind="clear_released",
                routes=list(clear.values()),
                hands=[hand.nominal_envelope_declaration() for hand in hands],
                retreat_distance=0.15,
            )
        )
    return result


def resolve_handover_profile_declarations(
    graph: Mapping[str, Any], pick_lowerers: Sequence[Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Bind existing-hold HandOver behavior from declared source grasp provenance.

    Args:
        graph: Ordered source calls, including release and resource transitions.
        pick_lowerers: Explicit source alias routes and their grasp constraints.

    Returns:
        Per-node options for the source adaptive receiver. Entity and resource
        identities are semantic keys, never heuristics for physical capability.
    """
    alias_modes = {}
    for lowerer in pick_lowerers:
        if lowerer["kind"] != "pick":
            continue
        identity = lowerer.get("call_id", "simulation.pick")
        alias_modes[identity] = (
            "constrained"
            if any(
                route.get("grasp_region") is not None
                or route.get("release_clearance_object_pose") is not None
                for route in lowerer["routes"]
            )
            else "handover_source"
        )
    held, result = {}, {}
    for node in graph["nodes"]:
        call = node["call"]
        obj = call.get("object", call.get("arguments", {}).get("object"))
        resources = call.get("resources", {})
        identity = call.get("call_id", call["kind"])
        if call["kind"] == "pick":
            held[(obj, resources.get("primary", "left"))] = (
                "none" if call.get("grasp") else "ordinary"
            )
        elif identity in alias_modes:
            held[(obj, resources.get("primary", "left"))] = alias_modes[identity]
        elif call["kind"] == "place" or identity in {
            "simulation.place_relative",
            "gen_sim.place_upright",
            "gen_sim.stack_place",
        }:
            held.pop((obj, resources.get("primary", "left")), None)
        elif call["kind"] == "hand_over":
            source, destination = resources.get("source", "left"), resources.get(
                "destination", "right"
            )
            mode = held.pop((obj, source), "none")
            adaptive = mode in {"ordinary", "pour", "handover_source"}
            result[node["id"]] = dict(
                adaptive_receive=adaptive,
                require_receiver_scene=adaptive,
                existing_hold_hand_source="grasp_command",
                receive_region="supported_end" if adaptive else "mesh_extent",
            )
            held[(obj, destination)] = mode
    return result
