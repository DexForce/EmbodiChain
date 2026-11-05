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

"""Bind source semantic targets through public authored-geometry services."""

from __future__ import annotations
from copy import deepcopy
import math
from typing import Any
from embodichain.lab.sim.scene.source_geometry import source_position as _position
from embodichain.lab.sim.scene.source_target_geometry import (
    source_axis_vertical_component,
    source_vertical_bounds as _vertical_mesh_bounds,
    source_upright_target,
    source_handover_staging_pose,
)
from embodichain.compute.geometry.source_targets import bounded_carry_lift
from .semantic_graph import SemanticTaskGraph, validate_semantic_task_graph

__all__ = [
    "refine_source_manipulation_declarations",
    "refine_source_return_declarations",
]

_ALIGN_HELD_CALL_ID = "gen_sim.align_held"
_AXIS_ALIGN_CALL_ID = "simulation.axis_align"
_UPRIGHT_PLACE_CALL_ID = "gen_sim.place_upright"
_COORDINATED_HOLD_CALL_ID = "simulation.coordinated_hold"
_MOVE_HELD_OBJECT_CALL_ID = "simulation.move_held_object"
_PICK_CALL_ID = "simulation.pick"
_CLEAR_RELEASED_CALL_ID = "gen_sim.clear_released"
_HANDOVER_SOURCE_PICK = "gen_sim.pick.handover_source"
_HORIZONTAL_HANDOVER_SOURCE_PICK = "gen_sim.pick.handover_horizontal_source"


def refine_source_manipulation_declarations(
    graph: SemanticTaskGraph, scene: Any
) -> SemanticTaskGraph:
    """Retain source upright, carrier and fresh handover declaration ordering.

    Args:
        graph: Validated serial semantic declarations before runtime assembly.
        scene: Explicit prepared source facts with physical geometry ownership.

    Returns:
        Owned validated graph with exact existing target/call identities and
        source mesh-dependent poses. Shared Scene/Compute services read geometry
        and calculate poses; no skill, simulator, IK or runtime policy is created.
    """
    selected = _refine_upright_targets(graph, scene)
    selected = _refine_coordinated_targets(selected, scene)
    objects = {item["runtime_uid"]: item for item in scene.planner_objects}
    selected = _preserve_already_upright_pick(selected, objects)
    for node in selected["nodes"]:
        call = node["call"]
        if node["task_type"] == "E4" and call["kind"] == "pick":
            node["call"] = {
                "kind": "registered",
                "call_id": (
                    f"{_HORIZONTAL_HANDOVER_SOURCE_PICK}.{call['object']}"
                    if abs(source_axis_vertical_component(objects[call["object"]]))
                    < 0.5
                    else _HANDOVER_SOURCE_PICK
                ),
                "arguments": {
                    "object": call["object"],
                    "target": f"{node['task_instance_id']}_source_grasp",
                },
                "resources": deepcopy(call["resources"]),
            }
        if node["call"].get("call_id") == _PICK_CALL_ID:
            node["call"]["call_id"] = f"gen_sim.pick.{node['task_instance_id']}"
    return _add_handover_staging(selected, scene)


def _refine_upright_targets(
    graph: SemanticTaskGraph,
    scene: Any,
) -> SemanticTaskGraph:
    """Use normalized mesh origins and the stage's declared supporting surface."""
    selected = deepcopy(graph)
    objects = {str(item["runtime_uid"]): item for item in scene.planner_objects}
    table = objects.get("table")
    if table is None:
        return selected
    if scene.table_top_z is None:
        return selected
    table_top = float(scene.table_top_z)
    support_heights: dict[str, float] = {}
    upright_objects: set[str] = set()
    for node in selected["nodes"]:
        call = node["call"]
        if call.get("call_id") in {_ALIGN_HELD_CALL_ID, _AXIS_ALIGN_CALL_ID}:
            upright_objects.add(call["arguments"]["object"])
        if call.get("call_id") == _UPRIGHT_PLACE_CALL_ID:
            support_id = call["arguments"]["reference"]
            support = objects[support_id]
            _, top = _vertical_mesh_bounds(
                support, axis_aligned=support_id in upright_objects
            )
            support_heights[node["task_instance_id"]] = _position(support)[2] + top
    target_routes: dict[str, tuple[str, str]] = {}
    target_heights: dict[str, float] = {}
    upright_staging_targets: set[str] = set()
    transfer_sources = {
        (
            node["task_instance_id"],
            node["depends_on"][0],
            node["call"]["object"],
            node["call"]["resources"]["source"],
        )
        for node in selected["nodes"]
        if node["task_type"] == "E4"
        and node["call"]["kind"] == "hand_over"
        and len(node["depends_on"]) == 1
    }
    for node in selected["nodes"]:
        call = node["call"]
        # A transfer pose is not an upright landing target, including when an
        # already expanded E4 graph is regenerated with a new Pick policy.
        if (
            node.get("task_type") == "E4"
            and call.get("call_id") == _MOVE_HELD_OBJECT_CALL_ID
            and (
                node["task_instance_id"],
                node["id"],
                call.get("arguments", {}).get("object"),
                call.get("resources", {}).get("primary"),
            )
            in transfer_sources
        ):
            continue
        if (
            node.get("task_type") not in {"E1", "E2", "E4"}
            or call.get("kind") != "registered"
            or call.get("call_id")
            not in {
                _MOVE_HELD_OBJECT_CALL_ID,
                _PICK_CALL_ID,
                _ALIGN_HELD_CALL_ID,
                _CLEAR_RELEASED_CALL_ID,
            }
        ):
            continue
        arguments = call.get("arguments", {})
        object_id = str(arguments.get("object", ""))
        target_id = str(arguments.get("target", ""))
        resource = str(call.get("resources", {}).get("primary", ""))
        target_routes[target_id] = (object_id, resource)
        if node.get("task_type") == "E2":
            upright_staging_targets.add(target_id)
        support_height = support_heights.get(node["task_instance_id"], table_top)
        target_heights[target_id] = support_height
        # Place owns the final descent; its release target still needs the
        # same mesh-origin refinement when no separate final Move is emitted.
        target_routes[f"{node['task_instance_id']}_upright_target"] = (
            object_id,
            resource,
        )
        target_heights[f"{node['task_instance_id']}_upright_target"] = support_height
    for target_id, (object_id, resource) in target_routes.items():
        source = objects.get(object_id)
        target = selected["targets"].get(target_id) if target_id is not None else None
        if source is None or not isinstance(target, dict):
            continue
        values = target.get("values")
        if not isinstance(values, list) or len(values) != 1:
            continue
        position = values[0].get("position")
        if not isinstance(position, list) or len(position) != 3:
            continue
        staging = target_id.endswith("_upright_staging_target")
        full_rotation = staging and target_id in upright_staging_targets
        proposal = source_upright_target(
            source,
            support_height=target_heights[target_id],
            world_yaw=math.pi if resource == "right" else 0.0,
            ordinary_release_clearance=0.01,
            slender_release_clearance=0.01,
            full_rotation_staging=full_rotation,
            staging_clearance=0.20 if staging and not full_rotation else 0.0,
            rotation_margin=0.02,
        )
        if proposal is not None:
            position[2], values[0]["quaternion_xyzw"] = proposal
    return validate_semantic_task_graph(selected)


def _refine_coordinated_targets(
    graph: SemanticTaskGraph, scene: Any
) -> SemanticTaskGraph:
    """Keep a carried object above its support at the declared terminal pose."""
    result = deepcopy(graph)
    objects = {str(item["runtime_uid"]): item for item in scene.planner_objects}
    for node in result["nodes"]:
        call = node["call"]
        if call["kind"] != "registered" or call["call_id"] != _COORDINATED_HOLD_CALL_ID:
            continue
        displacement = list(call["arguments"]["world_displacement"])
        if abs(float(displacement[2])) > 1e-8:
            continue
        source = objects[str(call["arguments"]["object"])]
        bottom, top = _vertical_mesh_bounds(source, axis_aligned=False)
        displacement[2] = bounded_carry_lift(
            bottom, top, minimum=0.05, maximum=0.10, reserve=0.03
        )
        call["arguments"]["world_displacement"] = displacement
    return validate_semantic_task_graph(result)


def _preserve_already_upright_pick(
    graph: SemanticTaskGraph, objects: dict[str, Any]
) -> SemanticTaskGraph:
    """Do not impose an inversion grip on an untouched, already-upright E1 source."""
    selected = deepcopy(graph)
    aligned = {
        (node["task_instance_id"], node["call"]["arguments"]["object"])
        for node in graph["nodes"]
        if node["call"].get("call_id") == _ALIGN_HELD_CALL_ID
        and node["call"]["arguments"].get("target") == "current_object_pose"
    }
    moved = set()
    for node in selected["nodes"]:
        call = node["call"]
        obj = call.get("object", call.get("arguments", {}).get("object"))
        source = objects.get(obj)
        if (
            node["task_type"] == "E1"
            and call.get("call_id") == _PICK_CALL_ID
            and (node["task_instance_id"], obj) in aligned
            and obj not in moved
            and source is not None
            and source_axis_vertical_component(source) >= math.cos(math.radians(10.0))
        ):
            node["call"] = {
                "kind": "pick",
                "object": obj,
                "resources": deepcopy(call["resources"]),
            }
        if obj is not None:
            moved.add(obj)
    return validate_semantic_task_graph(selected)


def _add_handover_staging(graph: SemanticTaskGraph, scene: Any) -> SemanticTaskGraph:
    """Propose a table-inward transfer using existing skills after a fresh E4 pick.

    This is a geometric candidate, not a reachability certificate. Normal
    planning and effect gates remain responsible for rejecting invalid motion.
    Previously manipulated objects cannot use source-scene pose predictions.
    """
    result = deepcopy(graph)
    prior = []
    for node in graph["nodes"]:
        call = node["call"]
        if call["kind"] != "hand_over" or not prior:
            prior.append(node)
            continue
        pick = prior[-1]
        pick_call = pick["call"]
        object_id = call["object"]
        eligible = (
            node["task_type"] == "E4"
            and pick["task_instance_id"] == node["task_instance_id"]
            and (
                pick_call.get("call_id") == _HANDOVER_SOURCE_PICK
                or pick_call.get("call_id", "").startswith(
                    _HORIZONTAL_HANDOVER_SOURCE_PICK + "."
                )
            )
            and pick_call.get("arguments", {}).get("object") == object_id
            and pick_call.get("resources", {}).get("primary")
            == call["resources"]["source"]
            and node["depends_on"] == [pick["id"]]
            and not any(
                earlier["call"].get(
                    "object", earlier["call"].get("arguments", {}).get("object")
                )
                == object_id
                for earlier in prior[:-1]
            )
        )
        prior.append(node)
        if not eligible:
            continue
        source = next(
            item for item in scene.planner_objects if item["runtime_uid"] == object_id
        )
        table = next(item for item in scene.background if item["uid"] == "table")
        pose = source_handover_staging_pose(source, table, lift_height=0.16)
        staging_id = f"{node['id']}__staging"
        target_id = f"{staging_id}_target"
        if target_id in result["targets"] or any(
            item["id"] == staging_id for item in result["nodes"]
        ):
            raise ValueError(
                "Generated handover staging identity collides with existing content."
            )
        result["targets"][target_id] = {
            "kind": "cyclic_pose",
            "values": [pose],
        }
        stage = deepcopy(node)
        stage["id"] = staging_id
        stage["call"] = {
            "kind": "registered",
            "call_id": _MOVE_HELD_OBJECT_CALL_ID,
            "arguments": {"object": object_id, "target": target_id},
            "resources": {"primary": call["resources"]["source"]},
        }
        index = next(
            i for i, item in enumerate(result["nodes"]) if item["id"] == node["id"]
        )
        result["nodes"][index]["depends_on"] = [staging_id]
        result["nodes"].insert(index, stage)
        group = next(
            item
            for item in result["task_groups"]
            if item["id"] == node["task_instance_id"]
        )
        group["node_ids"].insert(group["node_ids"].index(node["id"]), staging_id)
    return validate_semantic_task_graph(result)


def refine_source_return_declarations(
    graph: SemanticTaskGraph, scene: Any
) -> SemanticTaskGraph:
    """Bind original Pour returns to the declared tabletop and source origin.

    Args:
        graph: Canonical serial source graph after semantic profile expansion.
        scene: Prepared source geometry and optional explicit table height.

    Returns:
        Owned graph retaining target XY/orientation and the original source
        bottom offset. Without a table height the original graph is returned.
        This declaration is not an observation of a settled physical return.
    """
    if scene.table_top_z is None:
        return graph
    objects = {str(item["runtime_uid"]): item for item in scene.planner_objects}
    result = deepcopy(graph)
    for node in result["nodes"]:
        if node.get("task_type") != "E3" or node["call"].get("kind") != "place":
            continue
        target = node["call"].get("at")
        if not isinstance(target, dict) or target.get("kind") != "target_ref":
            continue
        target_id = str(target["target"])
        values = result["targets"].get(target_id, {}).get("values", ())
        object_id = str(node["call"]["object"])
        source = objects.get(object_id)
        if source is None or len(values) != 1:
            continue
        bottom, _ = _vertical_mesh_bounds(source, axis_aligned=False)
        values[0]["position"][2] = float(scene.table_top_z) - bottom
    return validate_semantic_task_graph(result)
