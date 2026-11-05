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
"""Bind semantic stability goals to the existing public Task Program conditions."""

from __future__ import annotations
from collections.abc import Mapping, Callable
from typing import Any
import math
import numpy as np
from embodichain.compute.geometry.placement_dimensions import (
    horizontal_placement_extent,
)
from embodichain.lab.sim.scene.source_geometry import (
    source_longest_local_axis,
    source_position,
    source_rotation,
)
from embodichain.lab.task_program.integrations.simulation.stability import (
    SimulationStabilityConditionCfg,
    SupportSensorDeclarationCfg,
)
from .semantic_graph import SemanticTaskGraph
from .placement_declarations import (
    resolve_relative_placement_routes,
    _mesh_vertices,
    _vertical_mesh_bounds,
    _route_key_for_node,
    _inside_parts,
    _ALIGN_HELD_CALL_ID,
    _AXIS_ALIGN_CALL_ID,
    _COORDINATED_TRANSPORT_CALL_ID,
    _MOVE_HELD_OBJECT_CALL_ID,
    _PLACE_RELATIVE_CALL_ID,
    _STACK_PLACE_CALL_ID,
    _UPRIGHT_PLACE_CALL_ID,
)

__all__ = ["resolve_stability_conditions"]
_RELATIVE_POSITION_TOLERANCE = 0.05
_HORIZONTAL_HANDOVER_SOURCE_PICK = "gen_sim.pick.handover_horizontal_source"
_COORDINATED_HOLD_CALL_ID = "simulation.coordinated_hold"


def _route_key(route: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(route["stage_id"]),
        str(route["object_id"]),
        str(route["reference_entity_id"]),
        str(route["relation"]),
    )


def _lookup_route(
    routes: Mapping[tuple[str, ...], Mapping[str, Any]],
    node: Mapping[str, Any],
    object_id: str,
    reference_id: str,
    relation: str,
) -> Mapping[str, Any]:
    return routes[_route_key_for_node(node, object_id, reference_id, relation)]


def _horizontal_half_extent(
    source: dict[str, Any], *, world_axis: int, axis_aligned: bool
) -> float:
    return horizontal_placement_extent(
        _mesh_vertices(source),
        source_rotation(source),
        world_axis=world_axis,
        axis_aligned=axis_aligned,
    )


def resolve_stability_conditions(
    graph: SemanticTaskGraph,
    scene: Any,
    motion_parts: Mapping[str, str],
    *,
    support_contract_revision: int | None = None,
    support_sensors: Mapping[str, SupportSensorDeclarationCfg] | None = None,
    support_sensor_factory: Callable[[str], SupportSensorDeclarationCfg] | None = None,
) -> dict[str, SimulationStabilityConditionCfg]:
    """Resolve immutable goal operands into owned, validated named conditions.

    Args:
        graph: Ordered semantic graph and task groups.
        scene: Frozen prepared scene with source geometry and table height.
        motion_parts: Explicit semantic resource to native control-part mapping.
        support_contract_revision: Explicit strong support revision, or the
            original default acceptance contract.
        support_sensor_factory: Optional pure declaration factory for the exact
            required condition names; mutually exclusive with support_sensors.
        support_sensors: Exact condition-name to standard sensor declaration
            mapping. Missing or unused declarations are rejected.

    Returns:
        Owned standard Task Program conditions for ``settle_presets``. The
        function reads no runtime state, installs no sensors and issues no
        acceptance result. Pick/motion/recovery options are separate owners.
    """
    if not isinstance(motion_parts, Mapping) or any(
        type(k) is not str
        or not k
        or k != k.strip()
        or type(v) is not str
        or not v
        or v != v.strip()
        for k, v in motion_parts.items()
    ):
        raise ValueError(
            "Motion parts require explicit exact resource/control identifiers."
        )
    objects = {str(item["runtime_uid"]): item for item in scene.planner_objects}
    if support_contract_revision is not None and (
        type(support_contract_revision) is not int or support_contract_revision != 1
    ):
        raise ValueError("Unsupported GenSim support contract revision.")
    upright: set[str] = set()
    oriented_groups = {
        node["task_instance_id"]
        for node in graph["nodes"]
        if node["task_type"] in {"E1", "E4"}
        and node["call"].get("call_id") == _ALIGN_HELD_CALL_ID
        and node["call"]["arguments"].get("target") == "current_object_pose"
    }
    terminal_nodes = {group["node_ids"][-1] for group in graph["task_groups"]}
    horizontal_handover_groups = {
        node["task_instance_id"]
        for node in graph["nodes"]
        if node["call"]
        .get("call_id", "")
        .startswith(_HORIZONTAL_HANDOVER_SOURCE_PICK + ".")
    }
    routes = {
        _route_key(r): r
        for r in resolve_relative_placement_routes(graph, scene, settled=True)
    }
    upright_routes = {
        _route_key(r): r
        for r in resolve_relative_placement_routes(
            graph, scene, settled=True, upright_only=True
        )
    }
    coordinated_on_routes = {
        _route_key(r): r
        for r in resolve_relative_placement_routes(
            graph, scene, settled=True, coordinated_only=True
        )
    }
    presets: dict[str, Any] = {}
    coordinated_positions: dict[str, list[float]] = {}
    for node in graph["nodes"]:
        call = node["call"]
        entity = call.get("object", call.get("arguments", {}).get("object"))
        if call.get("call_id") not in {
            _COORDINATED_HOLD_CALL_ID,
            _COORDINATED_TRANSPORT_CALL_ID,
        }:
            coordinated_positions.pop(entity, None)
        if (
            call["kind"] == "hand_over"
            and node["id"] in terminal_nodes
            and node["task_instance_id"] in horizontal_handover_groups
        ):
            object_id = str(call["object"])
            axis = source_longest_local_axis(objects[object_id])
            world_axis = (
                np.asarray([0.0, 0.0, 1.0])
                if object_id in upright
                else source_rotation(objects[object_id]) @ np.asarray(axis)
            )
            presets[f"gen_sim.{node['id']}.stable"] = {
                "kind": "hold",
                "entity": object_id,
                "local_axis": axis,
                "world_axis": world_axis.tolist(),
                "minimum_alignment": math.cos(math.pi / 18.0),
                "motion_parts": [motion_parts[call["resources"]["destination"]]],
            }
        if call["kind"] == "place" and node.get("task_instance_id") in oriented_groups:
            object_id = str(call["object"])
            presets[f"gen_sim.{node['id']}.stable"] = {
                "kind": "upright",
                "entity": object_id,
                "local_axis": source_longest_local_axis(objects[object_id]),
            }
        if call["kind"] != "registered":
            continue
        arguments = call["arguments"]
        if call["call_id"] in {_AXIS_ALIGN_CALL_ID, _ALIGN_HELD_CALL_ID} or (
            node.get("task_type") == "E2"
            and call["call_id"] == _MOVE_HELD_OBJECT_CALL_ID
        ):
            upright.add(str(arguments["object"]))
        if call["call_id"] in {
            _PLACE_RELATIVE_CALL_ID,
            _STACK_PLACE_CALL_ID,
            _UPRIGHT_PLACE_CALL_ID,
        }:
            object_id = str(arguments["object"])
            reference_id = str(arguments["reference"])
            if (
                node["task_type"] == "E2"
                or node.get("task_instance_id") in oriented_groups
                or (
                    object_id in upright
                    and reference_id == "table"
                    and arguments["relation"] in {"on", "above"}
                )
            ):
                selected_routes = (
                    upright_routes
                    if call["call_id"] == _UPRIGHT_PLACE_CALL_ID
                    else routes
                )
                route = _lookup_route(
                    selected_routes,
                    node,
                    object_id,
                    reference_id,
                    arguments["relation"],
                )
                presets[f"gen_sim.{node['id']}.stable"] = {
                    "kind": "upright",
                    "entity": object_id,
                    "local_axis": source_longest_local_axis(objects[object_id]),
                    "reference": reference_id,
                    "displacement": route["world_displacement"],
                    "position_tolerance": _RELATIVE_POSITION_TOLERANCE,
                }
            elif (
                arguments["relation"] in {"on", "above"}
                and object_id in upright
                and reference_id in upright
            ):
                bottom, _ = _vertical_mesh_bounds(objects[object_id], axis_aligned=True)
                _, top = _vertical_mesh_bounds(objects[reference_id], axis_aligned=True)
                presets[f"gen_sim.{node['id']}.stable"] = {
                    "kind": "stack",
                    "entity": object_id,
                    "reference": reference_id,
                    "local_axis": source_longest_local_axis(objects[object_id]),
                    "reference_axis": source_longest_local_axis(objects[reference_id]),
                    "object_bottom": bottom,
                    "reference_top": top,
                    "reference_half_extents": [
                        max(
                            0.001,
                            _horizontal_half_extent(
                                objects[reference_id],
                                world_axis=axis,
                                axis_aligned=True,
                            )
                            - 0.002,
                        )
                        for axis in (0, 1)
                    ],
                    "minimum_alignment": math.cos(math.pi / 18.0),
                }
            else:
                route = _lookup_route(
                    routes,
                    node,
                    object_id,
                    reference_id,
                    arguments["relation"],
                )
                presets[f"gen_sim.{node['id']}.stable"] = {
                    "kind": "placement",
                    "entity": object_id,
                    "reference": route.get("reference_link_id", reference_id),
                    "displacement": route["world_displacement"],
                    "position_tolerance": _RELATIVE_POSITION_TOLERANCE,
                }
                if node["task_type"] == "E1" and arguments["relation"] == "on":
                    presets[f"gen_sim.{node['id']}.stable"].update(
                        kind="supported_placement",
                    )
                    if (
                        reference_id == "table"
                        and call["call_id"] == _PLACE_RELATIVE_CALL_ID
                        and set(arguments).issubset(
                            {"object", "reference", "relation", "stage_id"}
                        )
                    ):
                        presets[f"gen_sim.{node['id']}.stable"] = {
                            "kind": "supported_region",
                            "entity": object_id,
                            "reference": reference_id,
                        }
        elif (
            call["call_id"] == _ALIGN_HELD_CALL_ID
            and node["task_type"] == "E4"
            and node["id"] in terminal_nodes
        ):
            object_id = str(arguments["object"])
            presets[f"gen_sim.{node['id']}.stable"] = {
                "kind": "hold",
                "entity": object_id,
                "local_axis": source_longest_local_axis(objects[object_id]),
                "motion_parts": [motion_parts[call["resources"]["primary"]]],
            }
        elif call["call_id"] in {
            _COORDINATED_HOLD_CALL_ID,
            _COORDINATED_TRANSPORT_CALL_ID,
        }:
            object_id = str(arguments["object"])
            if "reference" in arguments:
                reference_id = str(arguments["reference"])
                route = _lookup_route(
                    coordinated_on_routes,
                    node,
                    object_id,
                    reference_id,
                    arguments["relation"],
                )
                presets[f"gen_sim.{node['id']}.stable"] = {
                    "kind": "placement",
                    "entity": object_id,
                    "reference": reference_id,
                    "displacement": route["world_displacement"],
                    "position_tolerance": _RELATIVE_POSITION_TOLERANCE,
                }
                coordinated_positions[object_id] = [
                    value + delta
                    for value, delta in zip(
                        source_position(objects[reference_id]),
                        route["world_displacement"],
                        strict=True,
                    )
                ]
                continue
            position = coordinated_positions.get(
                object_id, list(source_position(objects[object_id]))
            )
            support = objects[object_id].get("attributes", {}).get("final_support", {})
            if (
                object_id not in coordinated_positions
                and support.get("parent_uid") == "table"
                and scene.table_top_z is not None
            ):
                # The skill offsets the live resting pose, not the import hover gap.
                bottom, _ = _vertical_mesh_bounds(
                    objects[object_id], axis_aligned=False
                )
                position[2] = float(scene.table_top_z) - bottom
            displacement = arguments["world_displacement"]
            position = [position[i] + float(displacement[i]) for i in range(3)]
            coordinated_positions[object_id] = position
            holding = call["call_id"] == _COORDINATED_HOLD_CALL_ID
            presets[f"gen_sim.{node['id']}.stable"] = {
                "kind": "hold" if holding else "placement",
                "entity": object_id,
                "target_position": position,
                "motion_parts": (
                    [
                        motion_parts[call["resources"][slot]]
                        for slot in ("left", "right")
                    ]
                    if holding
                    else []
                ),
                "position_tolerance": 0.05 if holding else _RELATIVE_POSITION_TOLERANCE,
            }
    if support_contract_revision == 1:

        for node in graph["nodes"]:
            if node["task_type"] not in {"E1", "E4"}:
                continue
            call = node["call"]
            args = call.get("arguments", {})
            obj = call.get("object", args.get("object"))
            ref = None
            if call["kind"] == "place" and "inside" in call:
                ref, parsed_obj, part = _inside_parts(call["inside"])
                if parsed_obj != obj or part:
                    continue
            elif call.get("call_id") in {
                _PLACE_RELATIVE_CALL_ID,
                _UPRIGHT_PLACE_CALL_ID,
                _STACK_PLACE_CALL_ID,
            } and args.get("relation") in {"on", "inside"}:
                ref = args.get("reference")
            if ref not in objects or objects[ref].get("role") == "articulation":
                continue
            key = f"gen_sim.{node['id']}.stable"
            cfg = dict(presets.get(key, {"kind": "supported_region", "entity": obj}))
            if cfg["kind"] in {"placement", "upright"}:
                cfg["kind"] = "supported_region"
            cfg.update(reference=ref, support_contract_revision=1)
            presets[key] = cfg
    vertical = {
        (n["task_instance_id"], n["call"]["arguments"]["object"])
        for n in graph["nodes"]
        if n["call"].get("call_id") == _ALIGN_HELD_CALL_ID
        and n["call"]["arguments"].get("axis_direction_free") is True
    }
    for node in graph["nodes"]:
        cfg = presets.get(f"gen_sim.{node['id']}.stable")
        if (
            cfg is not None
            and cfg.get("local_axis") is not None
            and (node["task_instance_id"], cfg["entity"]) in vertical
        ):
            cfg["axis_direction_free"] = True

    sensors = {} if support_sensors is None else dict(support_sensors)
    required = {
        name
        for name, cfg in presets.items()
        if cfg["kind"] in {"stack", "supported_placement", "supported_region"}
    }
    if support_sensor_factory is not None:
        if support_sensors is not None:
            raise ValueError(
                "Support sensor declarations and factory are mutually exclusive."
            )
        sensors = {name: support_sensor_factory(name) for name in sorted(required)}
    if set(sensors) != required:
        raise ValueError(
            f"Support sensor declarations must match conditions; missing={sorted(required-set(sensors))}, unused={sorted(set(sensors)-required)}."
        )
    return {
        name: SimulationStabilityConditionCfg(
            **cfg, support_sensor=sensors.get(name)
        ).snapshot()
        for name, cfg in presets.items()
    }
