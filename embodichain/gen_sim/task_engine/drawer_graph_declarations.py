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

"""Pure composite drawer call declarations over shared part discovery."""

from __future__ import annotations
from copy import deepcopy
from typing import Any
from embodichain.lab.sim.scene.articulation_inputs import (
    discover_prismatic_parts,
    inspect_prismatic_part,
)
from .prismatic_declarations import resolve_prismatic_program_declarations
from .placement_declarations import _inside_parts as inside_parts
from .semantic_graph import SemanticTaskGraph

__all__ = ["prepare_source_drawer_graph", "rewrite_source_drawer_close_resources"]

SLIDE_CALL = "gen_sim.articulation_slide"
WITHDRAW_CALL = "gen_sim.articulation_withdraw"
DRAWER_TRANSPORT_CALL = "gen_sim.drawer_transport"


def prepare_source_drawer_graph(
    graph: SemanticTaskGraph, scene: Any
) -> SemanticTaskGraph:
    """Canonicalize composite drawer parts and insert their declared transport.

    Args:
        graph: Serial semantic graph with explicit E6 recipes and inside calls.
        scene: Prepared physical articulation declarations and table height.

    Returns:
        Detached graph when a composite drawer is selected; unchanged graph
        otherwise. Parts resolve through shared physical discovery and the
        existing prismatic semantic owner. Ambiguous parts and conflicting
        transport declarations reject before asset preparation or simulation.
    """
    configs = {str(c["uid"]): c for c in getattr(scene, "articulations", ())}
    if not configs:
        return graph
    places = [
        n
        for n in graph["nodes"]
        if n["call"].get("kind") == "place" and "inside" in n["call"]
    ]
    if not any(inside_parts(n["call"]["inside"])[0] in configs for n in places):
        return graph
    result = deepcopy(graph)
    bindings = resolve_prismatic_program_declarations(graph, scene).bindings
    canonical = {}
    for selected in bindings:
        binding = selected.source
        if not any(
            inside_parts(n["call"]["inside"])[0] == binding.object_id for n in places
        ):
            continue
        part = next(
            p
            for p in discover_prismatic_parts(configs[binding.object_id])
            if p.joint == binding.joint
        )
        canonical[(binding.object_id, binding.part_id)] = inspect_prismatic_part(
            configs[binding.object_id], part.part_id
        )
    for node in result["nodes"]:
        call = node["call"]
        if call.get("call_id") in {SLIDE_CALL, WITHDRAW_CALL}:
            args = call["arguments"]
            binding = canonical.get((args["object"], args.get("part", "")))
            if binding is not None:
                args["part"] = binding.part_id
        elif call.get("kind") == "place" and "inside" in call:
            container, obj, part = inside_parts(call["inside"])
            if container not in configs:
                continue
            matches = [
                b
                for b in dict.fromkeys(canonical.values())
                if b.object_id == container and (part is None or part == b.part_id)
            ]
            if len(matches) != 1:
                raise ValueError(
                    "Drawer placement requires one explicit E6 part; use its step_result or target part binding."
                )
            binding = matches[0]
            call["inside"] = f"inside__{container}__{obj}__{binding.part_id}"
    by_id = {node["id"]: node for node in result["nodes"]}
    expanded = []
    for node in result["nodes"]:
        call = node["call"]
        if (
            call.get("kind") == "place"
            and "inside" in call
            and inside_parts(call["inside"])[0] in configs
        ):
            transport_id = f"{node['id']}__transport"
            transport_call = {
                "kind": "registered",
                "call_id": DRAWER_TRANSPORT_CALL,
                "arguments": {"object": call["object"], "target": call["inside"]},
                "resources": deepcopy(call["resources"]),
            }
            if transport_id in by_id:
                existing = by_id[transport_id]["call"]
                if existing == {
                    **transport_call,
                    "call_id": "simulation.move_held_object",
                }:
                    existing["call_id"] = DRAWER_TRANSPORT_CALL
                if by_id[transport_id]["call"] != transport_call or node[
                    "depends_on"
                ] != [transport_id]:
                    raise ValueError(
                        "Drawer transport conflicts with the placement route."
                    )
            else:
                expanded.append(
                    {**deepcopy(node), "id": transport_id, "call": transport_call}
                )
                node["depends_on"] = [transport_id]
                group = next(
                    g for g in result["task_groups"] if node["id"] in g["node_ids"]
                )
                group["node_ids"].insert(
                    group["node_ids"].index(node["id"]), transport_id
                )
        expanded.append(node)
    result["nodes"] = expanded
    return result


def rewrite_source_drawer_close_resources(
    graph: SemanticTaskGraph,
    *,
    drawer_bindings: frozenset[tuple[str, str]] = frozenset(),
) -> SemanticTaskGraph:
    """Declare the source alternate resource for closing a bound composite drawer.

    Args:
        graph: Serial semantic graph with explicit prismatic task groups.
        drawer_bindings: Exact selected physical object/part pairs.

    Returns:
        Detached graph retaining the source left-to-right close ownership and
        idempotent already-right calls. This changes declarations before program
        compilation; no runtime request or resource cache is rewritten.
    """
    result = deepcopy(graph)
    nodes = {node["id"]: node for node in result["nodes"]}
    if not drawer_bindings:
        return result
    for group in result.get("task_groups", ()):
        ids = group.get("node_ids", ())
        if group.get("task_type") != "E6" or not ids:
            continue
        first = nodes[ids[0]]["call"]
        if first.get("arguments", {}).get("state") != "closed":
            continue
        arguments = first.get("arguments", {})
        object_id = str(arguments.get("object", ""))
        part_id = str(arguments.get("part", ""))
        if not any(
            bound_object == object_id and (not part_id or bound_part == part_id)
            for bound_object, bound_part in drawer_bindings
        ):
            continue
        primary = first.get("resources", {}).get("primary")
        # The right arm is the alternate in the supported dual-arm profile.
        # Keep an already-right close route idempotent across bundle passes.
        alternate = "right" if primary == "left" else primary
        if alternate is None:
            continue
        for node_id in ids:
            resources = nodes[node_id]["call"].get("resources")
            if resources is not None:
                resources["primary"] = alternate
    return result
