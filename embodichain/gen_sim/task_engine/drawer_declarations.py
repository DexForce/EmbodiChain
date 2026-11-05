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

"""Drawer semantic declarations over shared source geometry and runtime owners."""

from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from embodichain.lab.sim.scene.articulation_inputs import PrismaticBinding
from embodichain.lab.sim.scene.drawer import (
    read_drawer_region,
    source_drawer_container_pose,
)
from .prismatic_declarations import PrismaticProgramDeclarations
from .placement_declarations import _inside_parts

__all__ = [
    "DrawerDeclaration",
    "resolve_drawer_declarations",
    "bind_drawer_occurrences",
]


@dataclass(frozen=True, slots=True)
class DrawerDeclaration:
    """Exact payload/container/link route with declared vertical-preservation calls."""

    target_id: str
    object_id: str
    link_entity_id: str
    source: PrismaticBinding
    lower: tuple[float, float, float]
    upper: tuple[float, float, float]
    object_target_pose: tuple[float, ...]
    vertical_nodes: tuple[str, ...]

    def container_declaration(self) -> dict[str, Any]:
        """Return the public link-owned container affordance declaration."""
        return dict(
            parent_id=self.link_entity_id,
            entity_id=self.target_id,
            kind="container",
            native_name=self.target_id,
            object_target_pose=list(self.object_target_pose),
            release_clearance=0.0,
        )

    def invocation_declaration(
        self, invocation_id: str, operation: str
    ) -> dict[str, Any]:
        """Bind an exact compiled occurrence to the shared drawer service.

        Args:
            invocation_id: Public compiled occurrence identity.
            operation: Explicit Place, transport or closing Slide selection.

        Returns:
            Closed shared service declaration, including the physical part owner.
        """
        if operation not in {"place", "transport", "close"}:
            raise ValueError("Unknown declared drawer operation.")
        return dict(
            invocation_id=invocation_id,
            operation=operation,
            target_id=self.link_entity_id if operation == "close" else self.target_id,
            link_entity_id=self.link_entity_id,
            object_id=self.object_id,
            source=self.source.payload(),
            lower=list(self.lower),
            upper=list(self.upper),
        )


def resolve_drawer_declarations(
    graph: Mapping[str, Any], scene: Any, prismatic: PrismaticProgramDeclarations
) -> tuple[DrawerDeclaration, ...]:
    """Resolve declared inside targets through the same inspected E6 part.

    Args:
        graph: Canonical source graph with explicit part-qualified inside IDs.
        scene: Frozen physical source scene.
        prismatic: Already validated shared E6 physical part bindings.

    Returns:
        Source-order independent container routes. No robot, simulator, collision
        query or candidate search is constructed while producing declarations.
    """
    configs = {str(item["uid"]): item for item in scene.articulations}
    objects = {str(item["runtime_uid"]): item for item in scene.planner_objects}
    bindings = {
        (item.source.object_id, item.source.part_id): item
        for item in prismatic.bindings
    }
    upright_groups = {
        (node["task_instance_id"], node["call"]["arguments"]["object"])
        for node in graph["nodes"]
        if node["call"].get("call_id") == "gen_sim.align_held"
        and node["call"].get("arguments", {}).get("target") == "current_object_pose"
    }
    vertical = {}
    for node in graph["nodes"]:
        call = node["call"]
        args = call.get("arguments", {})
        obj = call.get("object", args.get("object"))
        if (node.get("task_instance_id"), obj) not in upright_groups:
            continue
        target = (
            call.get("inside")
            if call["kind"] == "place"
            else (
                args.get("target")
                if call.get("call_id") == "gen_sim.drawer_transport"
                else None
            )
        )
        if target is not None:
            vertical.setdefault(target, set()).add(node["id"])
    result = {}
    for node in graph["nodes"]:
        call = node["call"]
        if call["kind"] != "place" or "inside" not in call:
            continue
        target = call["inside"]
        container, obj, part = _inside_parts(target)
        if container not in configs or target in result:
            continue
        if part is None or (container, part) not in bindings:
            raise ValueError("Drawer placement requires its exact canonical E6 part.")
        binding = bindings[(container, part)]
        low, high = read_drawer_region(configs[container], binding.source)
        pose = source_drawer_container_pose(
            configs[container], binding.source, objects[obj], low, high
        )
        result[target] = DrawerDeclaration(
            target,
            obj,
            binding.entity_id,
            binding.source,
            tuple(map(float, low)),
            tuple(map(float, high)),
            tuple(pose.reshape(-1).tolist()),
            tuple(sorted(vertical.get(target, ()))),
        )
    return tuple(result.values())


def bind_drawer_occurrences(
    graph: Mapping[str, Any],
    routes: tuple[DrawerDeclaration, ...],
    occurrences: Mapping[str, str],
) -> tuple[dict[str, list[dict[str, Any]]], frozenset[str]]:
    """Bind declared drawer operations and exact preceding acquisition deferrals.

    Args:
        graph: Canonical ordered graph with explicit same-resource dependencies.
        routes: Resolved physical/container declarations.
        occurrences: Public compiled identity for each unique source node.

    Returns:
        Per-node shared service rows and the source Pick nodes whose complete
        downstream target tuple is explicitly deferred to drawer planning.
    """
    by_target = {route.target_id: route for route in routes}
    nodes = {node["id"]: node for node in graph["nodes"]}
    rows, deferred = {}, set()
    for node in graph["nodes"]:
        call = node["call"]
        args = call.get("arguments", {})
        target = (
            call.get("inside")
            if call["kind"] == "place"
            else (
                args.get("target")
                if call.get("call_id") == "gen_sim.drawer_transport"
                else None
            )
        )
        if target in by_target:
            route = by_target[target]
            operation = "place" if call["kind"] == "place" else "transport"
            if call.get("object", args.get("object")) != route.object_id:
                raise ValueError(
                    "Drawer call subject differs from its explicit target."
                )
            rows[node["id"]] = [
                route.invocation_declaration(occurrences[node["id"]], operation)
            ]
            if operation != "place":
                continue
            resource = call["resources"]["primary"]
            previous = node
            visited = set()
            while len(previous.get("depends_on", ())) == 1:
                identity = previous["depends_on"][0]
                if identity in visited:
                    raise ValueError("Drawer acquisition dependencies contain a cycle.")
                visited.add(identity)
                previous = nodes[identity]
                before = previous["call"]
                if (
                    before.get("object", before.get("arguments", {}).get("object"))
                    != route.object_id
                    or before.get("resources", {}).get("primary") != resource
                ):
                    break
                semantic = before.get("call_id", before["kind"])
                if (
                    before["kind"] == "pick"
                    or semantic == "simulation.pick"
                    or semantic.startswith("gen_sim.pick.")
                ):
                    deferred.add(previous["id"])
                    break
                if semantic not in {
                    "gen_sim.align_held",
                    "gen_sim.drawer_transport",
                    "simulation.move_held_object",
                }:
                    break
        elif (
            call.get("call_id") == "gen_sim.articulation_slide"
            and args.get("state") == "closed"
        ):
            selected = [
                route
                for route in routes
                if route.source.object_id == args.get("object")
                and route.source.part_id == args.get("part", "")
            ]
            if selected:
                rows[node["id"]] = [
                    route.invocation_declaration(occurrences[node["id"]], "close")
                    for route in selected
                ]
    return rows, frozenset(deferred)
