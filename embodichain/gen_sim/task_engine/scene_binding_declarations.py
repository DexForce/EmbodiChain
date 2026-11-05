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

"""Project declared semantic affordances onto shared static source geometry."""

from __future__ import annotations
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any
import numpy as np
from scipy.spatial.transform import Rotation
from embodichain.lab.sim.scene.source_geometry import (
    source_rotation,
    source_position,
    source_longest_local_axis,
)
from embodichain.lab.sim.scene.source_robot_geometry import source_arm_base_position
from embodichain.lab.sim.scene.declaration_placement import (
    source_container_target_pose,
    source_support_target_pose,
)
from .semantic_graph import SemanticTaskGraph
from embodichain.lab.sim.scene.source_collisions import read_source_collision_meshes
from .placement_declarations import _inside_parts, _graph_references

__all__ = ["resolve_scene_binding_declaration"]
_ALIGN_HELD_CALL_ID = "gen_sim.align_held"
_MOVE_HELD_OBJECT_CALL_ID = "simulation.move_held_object"


def _planned_object_rotations(
    graph: SemanticTaskGraph, *, before_node_id: str | None = None
) -> dict[str, np.ndarray]:
    """Return declared orientations, optionally stopping before one call.

    Bundle geometry must follow explicit GenSim alignment targets rather than
    silently reverting to the source ``init_rot`` after an earlier E2/E4 call.
    This is a planning snapshot only; runtime still observes the released pose.
    """
    from scipy.spatial.transform import Rotation

    result: dict[str, np.ndarray] = {}
    for node in graph["nodes"]:
        if before_node_id is not None and node.get("id") == before_node_id:
            break
        call = node["call"]
        if call.get("kind") != "registered" or call.get("call_id") not in {
            _ALIGN_HELD_CALL_ID,
            _MOVE_HELD_OBJECT_CALL_ID,
        }:
            continue
        arguments = call.get("arguments", {})
        object_id = arguments.get("object")
        target_id = arguments.get("target")
        if not isinstance(object_id, str) or not isinstance(target_id, str):
            continue
        if target_id == "current_object_pose":
            continue
        target = graph["targets"].get(target_id)
        values = target.get("values") if isinstance(target, Mapping) else None
        pose_value = (
            values[0] if isinstance(values, list) and len(values) == 1 else None
        )
        quaternion = (
            pose_value.get("quaternion_xyzw")
            if isinstance(pose_value, Mapping)
            else None
        )
        if isinstance(quaternion, list) and len(quaternion) == 4:
            result[object_id] = Rotation.from_quat(
                np.asarray(quaternion, dtype=float)
            ).as_matrix()
    return result


def resolve_scene_binding_declaration(
    graph: SemanticTaskGraph,
    scene: Any,
    embodiment: Mapping[str, Any],
    *,
    program_id: str,
    collision_world: bool = False,
    link_declarations: Sequence[Mapping[str, Any]] = (),
    container_declarations: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Declare source entities/affordances without native construction or planning.

    Args:
        graph: Ordered graph supplying exact relation/target identifiers.
        scene: Prepared physical source scene with planner entities/articulations.
        embodiment: Source robot component with explicit semantic resource bindings.
        program_id: Generated program identity used for scene contracts.
        collision_world: Preserve explicit source collision role metadata.
        link_declarations: Qualified shared-owner articulation/link declarations.
        container_declarations: Optional already-resolved drawer/link affordances
            keyed by canonical inside ID. Ordinary rigid containers are measured
            through shared Scene/Compute owners using declared release order.

    Returns:
        Existing closed scene_binding format. Every source rigid obstacle stays
        declared independently of the selected motion backend. Geometry/placement
        results are proposals, not native physical-effect certificates.
    """
    objects = {str(item["runtime_uid"]): item for item in scene.planner_objects}
    containers = {} if container_declarations is None else dict(container_declarations)
    inside_routes, on_routes = [], []
    directed_axes = set()
    support_axes = set()
    for node in graph["nodes"]:
        call = node["call"]
        args = call.get("arguments", {})
        identity = call.get("call_id", "")
        if identity in {"simulation.axis_align", "gen_sim.align_held"} or (
            identity == "simulation.move_held_object" and node.get("task_type") == "E2"
        ):
            support_axes.add(str(args["object"]))
            directed_axes.add(str(args["object"]))
        if identity in {
            "simulation.pour",
            "gen_sim.pick.handover_source",
        } or identity.startswith("gen_sim.pick.handover_horizontal_source."):
            directed_axes.add(str(args["object"]))
        if call["kind"] == "place" and "inside" in call:
            affordance = str(call["inside"])
            container, child, _ = _inside_parts(affordance)
            if affordance not in containers:
                inside_routes.append(
                    (
                        affordance,
                        container,
                        child,
                        str(call.get("resources", {}).get("primary", "left")),
                        _planned_object_rotations(graph, before_node_id=node["id"]),
                    )
                )
        if call["kind"] == "place" and "on" in call:
            parts = str(call["on"]).split("__")
            if len(parts) != 3 or parts[0] != "on":
                raise ValueError(f"Unsupported generated on affordance {call['on']!r}.")
            on_routes.append((str(call["on"]), parts[1], parts[2]))
    simulation = embodiment["simulation"]
    resources = {
        item["resource_id"]: item for item in embodiment["skill_profile"]["resources"]
    }
    rigid = []
    for entity, source in sorted(objects.items()):
        if source["role"] == "articulation":
            if any(container == entity for _, container, *_ in inside_routes) or any(
                support == entity for _, support, _ in on_routes
            ):
                raise ValueError(
                    "Articulated supports/containers require qualified link bindings."
                )
            continue
        affordances = []
        if source["role"] == "rigid_object":
            grasp = {"entity_id": f"{entity}_grasp", "kind": "antipodal_grasp"}
            if entity in directed_axes:
                grasp["internal_axis"] = source_longest_local_axis(source)
            affordances.append(grasp)
        occupied = []
        occupied_rotations = {}
        for identity, container, child, resource, rotations in inside_routes:
            if container != entity:
                continue
            if child not in objects:
                raise ValueError(
                    f"Inside relation references missing source entity {child!r}."
                )
            resource_binding = resources.get(resource)
            if resource_binding is None:
                raise ValueError(
                    f"Inside landing resource {resource!r} is not declared."
                )
            motion = next(
                endpoint
                for endpoint in resource_binding["endpoints"]
                if endpoint["endpoint_id"] == "motion"
            )
            base = source_arm_base_position(simulation, motion["control_part"])
            world = source_rotation(simulation) @ base + np.asarray(
                source_position(simulation)
            )
            target = source_container_target_pose(
                source,
                objects[child],
                world,
                occupied=occupied,
                child_rotation=rotations.get(child),
                occupied_rotations=tuple(
                    occupied_rotations.get(str(previous.get("runtime_uid")))
                    for _, previous in occupied
                ),
            )
            if target is None:
                raise ValueError(
                    f"Placement requires a measured container floor: {entity!r}."
                )
            occupied.append((target, objects[child]))
            occupied_rotations.setdefault(child, rotations.get(child))
            affordances.append(
                dict(
                    entity_id=identity,
                    kind="container",
                    native_name=identity,
                    object_target_pose=target,
                    release_clearance=0.12,
                )
            )
        for identity, support, child in on_routes:
            if support != entity:
                continue
            if child not in objects:
                raise ValueError(
                    f"On relation references missing scene entity {child!r}."
                )
            attributes = source.get("attributes", {})
            bounds = (
                attributes.get("final_world_aabb")
                if isinstance(attributes, dict)
                else None
            )
            top = (
                float(bounds["max"][2])
                if isinstance(bounds, dict) and isinstance(bounds.get("max"), list)
                else None
            )
            target = source_support_target_pose(
                source,
                objects[child],
                axis_aligned=entity in support_axes,
                clearance=0.01,
                support_fallback_extents=(0.1, 0.1, 0.1),
                child_fallback_extents=(0.05, 0.05, 0.05),
                world_support_top=top,
            )
            affordances.append(
                dict(
                    entity_id=identity,
                    kind="support_surface",
                    native_name=identity,
                    object_target_pose=target,
                )
            )
        rigid.append(
            dict(
                entity_id=entity,
                simulation_uid=entity,
                dynamics="dynamic" if source["role"] == "rigid_object" else "static",
                semantic_type=str(source.get("category") or entity),
                **(
                    {
                        "collision_role": (
                            "dynamic" if source["role"] == "rigid_object" else "static"
                        )
                    }
                    if collision_world
                    else {}
                ),
                affordances=affordances,
            )
        )
    if "table" not in objects:
        raise ValueError("Generated scene requires its declared table reference.")
    links = deepcopy(list(link_declarations))
    declared_ids = {item["entity_id"] for item in links}
    spatial = _graph_references(graph, scene)
    links = [
        dict(
            entity_id=reference.link_id,
            articulation_id=reference.object_id,
            native_link_name=reference.geometry.link,
        )
        for reference in spatial.values()
        if reference.link_id not in declared_ids
    ] + links
    native_ids = {}
    used_ids = {str(link["entity_id"]) for link in links}
    for link in links:
        key = (str(link["articulation_id"]), str(link["native_link_name"]))
        identity = str(link["entity_id"])
        if key in native_ids and native_ids[key] != identity:
            raise ValueError(
                "One physical collision link has ambiguous semantic identities."
            )
        native_ids[key] = identity
    physical_links = []
    for config in sorted(scene.articulations, key=lambda value: str(value["uid"])):
        uid = str(config["uid"])
        for native in sorted(
            {owner for owner, _ in read_source_collision_meshes(dict(config))}
        ):
            if (uid, native) in native_ids:
                continue
            identity = f"gen_sim.collision_link.{uid}.{native}"
            if identity in used_ids:
                raise ValueError(
                    "Collision-link identity conflicts with an existing link."
                )
            used_ids.add(identity)
            native_ids[(uid, native)] = identity
            physical_links.append(
                dict(entity_id=identity, articulation_id=uid, native_link_name=native)
            )
    links = physical_links + links
    by_link = {item["entity_id"]: item for item in links}
    if len(by_link) != len(links):
        raise ValueError("Link declarations must have unique canonical identities.")
    for identity, affordance in containers.items():
        owner = affordance.get("parent_id")
        if owner not in by_link:
            raise ValueError(
                "Qualified container requires its explicit link declaration."
            )
        payload = deepcopy(dict(affordance))
        payload.pop("parent_id")
        if payload.get("entity_id") != identity:
            raise ValueError(
                "Qualified container key must match its canonical identity."
            )
        by_link[owner].setdefault("affordances", []).append(payload)
    return dict(
        contract_id=f"{program_id}_scene_v1",
        registry_id=f"{program_id}_scene_registry",
        rigid_objects=rigid,
        articulations=[
            dict(entity_id=str(item["uid"]), simulation_uid=str(item["uid"]))
            for item in sorted(scene.articulations, key=lambda item: str(item["uid"]))
        ],
        links=links,
    )
