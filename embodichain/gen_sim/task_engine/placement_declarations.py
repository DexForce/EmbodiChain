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

"""Resolve semantic placement declarations through shared source geometry owners."""

from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
import numpy as np
from embodichain.compute.geometry.landing import (
    planar_floor_region,
    select_table_landing,
)
from embodichain.compute.geometry.placement_dimensions import (
    dominant_local_axis,
    vertical_placement_bounds,
    horizontal_placement_distance,
)
from embodichain.compute.geometry.placement_proposals import (
    reserved_placement_footprint,
    first_unoccupied_landing,
)
from embodichain.lab.sim.scene.source_geometry import (
    read_source_vertices,
    source_rotation,
    source_position,
    read_gltf_z_up_mesh,
    read_source_world_mesh,
    source_local_bounds,
)
from embodichain.lab.sim.scene.source_envelopes import source_obstacle_bounds
from embodichain.lab.sim.scene.spatial_reference import (
    read_spatial_reference,
    SpatialReferenceGeometry,
)
from .semantic_graph import SemanticTaskGraph

__all__ = ["resolve_relative_placement_routes"]

_LATERAL_RELATION_DISTANCE = 0.1
_FRONT_RELATION_DISTANCE = 0.18
_RELATION_CLEARANCE = 0.02
_AXIS_ALIGNED_RELATION_CLEARANCE = 0.04
_PLACEMENT_CLEARANCE = 0.01
_LATERAL_RELEASE_CLEARANCE = 0.03
_AXIS_ALIGN_CALL_ID = "simulation.axis_align"
_COORDINATED_TRANSPORT_CALL_ID = "simulation.coordinated_transport"
_PLACE_RELATIVE_CALL_ID = "simulation.place_relative"
_UPRIGHT_PLACE_CALL_ID = "gen_sim.place_upright"
_STACK_PLACE_CALL_ID = "gen_sim.stack_place"
_MOVE_HELD_OBJECT_CALL_ID = "simulation.move_held_object"
_ALIGN_HELD_CALL_ID = "gen_sim.align_held"
_UPRIGHT_RELEASE_CLEARANCE = 0.01
_SLENDER_UPRIGHT_RELEASE_CLEARANCE = 0.01
SLIDE_CALL = "gen_sim.articulation_slide"


def _inside_parts(affordance: str) -> tuple[str, str, str | None]:
    parts = affordance.split("__")
    if len(parts) not in (3, 4) or parts[0] != "inside" or not all(parts):
        raise ValueError(f"Unsupported generated inside affordance {affordance!r}.")
    if len(parts) == 4 and not parts[3].startswith("part_"):
        raise ValueError("Drawer inside target requires a stable part ID.")
    return parts[1], parts[2], parts[3] if len(parts) == 4 else None


def _is_handover_source_call_id(value: object) -> bool:
    """Recognize only the registered end-grasp source aliases owned by GenSim."""
    return isinstance(value, str) and (
        value == "gen_sim.pick.handover_source"
        or value.startswith("gen_sim.pick.handover_horizontal_source.")
        and bool(value.removeprefix("gen_sim.pick.handover_horizontal_source."))
    )


@dataclass(frozen=True, slots=True)
class _SpatialReference:
    object_id: str
    part_id: str | None
    geometry: SpatialReferenceGeometry

    @property
    def link_id(self) -> str:
        return f"{self.object_id}__{self.geometry.link}"

    @property
    def initial_pose(self) -> np.ndarray:
        return self.geometry.initial_pose

    def planner_object(self) -> dict[str, Any]:
        return {
            "runtime_uid": self.link_id,
            "role": "articulation",
            "init_pos": self.initial_pose[:3, 3].tolist(),
            "init_local_pose": self.initial_pose.tolist(),
            "_spatial_vertices": self.geometry.vertices,
        }


def _graph_references(
    graph: Mapping[str, Any], scene: Any
) -> dict[tuple[str, str | None], _SpatialReference]:
    configs = {str(c["uid"]): c for c in getattr(scene, "articulations", ())}
    references = {}
    relations = {
        "left_of",
        "right_of",
        "front_of",
        "behind",
        "front_left_of",
        "front_right_of",
        "back_left_of",
        "back_right_of",
    }
    for node in graph["nodes"]:
        call = node["call"]
        args = call.get("arguments", {})
        uid = args.get("reference")
        if call.get("call_id") != "simulation.place_relative" or uid not in configs:
            continue
        if args.get("relation") not in relations:
            continue
        key = (str(uid), args.get("reference_part"))
        if key not in references:
            references[key] = _SpatialReference(
                str(uid), key[1], read_spatial_reference(configs[str(uid)], key[1])
            )
    return references


def _mesh_vertices(source: dict[str, Any]) -> np.ndarray:
    return read_source_vertices(
        source,
        articulation=source.get("role") == "articulation",
        local_vertices=source.get("_spatial_vertices"),
    )


def _rigid_mesh_zup(source: dict[str, Any]) -> Any:
    return read_gltf_z_up_mesh(
        source["shape"]["fpath"], body_scale=source.get("body_scale", [1.0] * 3)
    )


def _vertical_mesh_bounds(
    source: dict[str, Any], *, axis_aligned: bool
) -> tuple[float, float]:
    return vertical_placement_bounds(
        _mesh_vertices(source), source_rotation(source), axis_aligned=axis_aligned
    )


def _horizontal_relation_distance(
    obj: dict[str, Any],
    reference: dict[str, Any],
    *,
    world_axis: int,
    object_axis_aligned: bool,
    reference_axis_aligned: bool,
    minimum: float,
    reference_direction: int = 1,
) -> float:
    return horizontal_placement_distance(
        _mesh_vertices(obj),
        source_rotation(obj),
        _mesh_vertices(reference),
        source_rotation(reference),
        reference_is_articulation=reference.get("role") == "articulation",
        relation_clearance=_RELATION_CLEARANCE,
        axis_aligned_clearance=_AXIS_ALIGNED_RELATION_CLEARANCE,
        world_axis=world_axis,
        object_axis_aligned=object_axis_aligned,
        reference_axis_aligned=reference_axis_aligned,
        minimum=minimum,
        reference_direction=reference_direction,
    )


def _table_landing_position(
    object_id: str,
    scene_objects: dict[str, Any],
    *,
    axis_aligned: bool,
    table_top_z: float,
) -> tuple[float, float, float]:
    obj, table = scene_objects[object_id], scene_objects["table"]
    vertices = _mesh_vertices(obj)
    major = int(np.argmax(dominant_local_axis(vertices))) if axis_aligned else None
    bottom, top = _vertical_mesh_bounds(obj, axis_aligned=axis_aligned)
    obstacles = (
        source_obstacle_bounds(
            obstacle,
            articulation=obstacle.get("role") == "articulation",
            local_vertices=obstacle.get("_spatial_vertices"),
        )
        for uid, obstacle in sorted(scene_objects.items())
        if uid not in {object_id, "table"}
    )
    return select_table_landing(
        read_source_world_mesh(table),
        vertices if axis_aligned else vertices @ source_rotation(obj).T,
        source_position(obj),
        obstacles,
        table_top_z=table_top_z,
        object_bottom=bottom,
        object_top=top,
        axis_aligned_axis=major,
        placement_clearance=_PLACEMENT_CLEARANCE,
        relation_clearance=_RELATION_CLEARANCE,
    )


def _local_extents(source: dict[str, Any]) -> list[float] | None:
    bounds = source_local_bounds(source)
    return (
        None
        if bounds is None
        else [maximum - minimum for minimum, maximum in zip(*bounds, strict=True)]
    )


def _unoccupied_support_landing(
    obj: dict[str, Any],
    reference: dict[str, Any],
    displacement: list[float],
    occupied: tuple[Any, ...],
) -> tuple[list[float], Any]:
    selected = first_unoccupied_landing(
        _rigid_mesh_zup(obj),
        read_source_world_mesh(reference),
        source_rotation(obj),
        source_position(obj)[:2],
        occupied,
        clearance_m=_PLACEMENT_CLEARANCE,
        separation_m=_RELATION_CLEARANCE,
    )
    if selected is None:
        raise ValueError(
            f"Support {reference['runtime_uid']!r} has no unoccupied full-footprint landing for {obj['runtime_uid']!r}."
        )
    position, footprint = selected
    result = list(displacement)
    result[:2] = (position[:2] - np.asarray(source_position(reference))[:2]).tolist()
    return result, footprint


def _route_key_for_node(
    node: Mapping[str, Any],
    object_id: str,
    reference_id: str,
    relation: str,
) -> tuple[str, str, str, str]:
    """Return the route key for one graph node occurrence."""
    arguments = node.get("call", {}).get("arguments", {})
    if "stage_id" in arguments:
        stage_id = arguments["stage_id"]
        if type(stage_id) is not str or not stage_id or stage_id != stage_id.strip():
            raise ValueError("Placement stage_id must be a non-empty exact identifier.")
    else:
        stage_id = str(node.get("task_instance_id", node.get("id", "")))
    return (stage_id, object_id, reference_id, relation)


def _inside_reference_height(
    call: Mapping[str, Any], objects: dict[str, Any]
) -> float | None:
    """Project a prior rigid-inside landing, not an executed-placement receipt."""

    container_id, child_id, part = _inside_parts(str(call["inside"]))
    container, child = objects.get(container_id, {}), objects.get(child_id, {})
    if (
        part is not None
        or any(item.get("role") != "rigid_object" for item in (container, child))
        or not container.get("shape", {}).get("fpath")
        or not np.allclose(source_rotation(container)[2], [0, 0, 1], atol=1e-6, rtol=0)
    ):
        return None
    measured = planar_floor_region(_rigid_mesh_zup(container))
    if measured is None:
        return None
    _, floor = measured
    bottom, _ = _vertical_mesh_bounds(child, axis_aligned=False)
    return source_position(container)[2] + floor - bottom + 0.001


def _relative_world_displacement(
    relation: str,
    *,
    object_id: str,
    reference_id: str,
    scene_objects: dict[str, Any],
    axis_align_objects: set[str],
    table_top_z: float | None,
) -> list[float]:
    """Derive one world-frame semantic relation from trusted scene geometry."""
    try:
        obj = scene_objects[object_id]
        reference = scene_objects[reference_id]
    except KeyError as exc:
        raise ValueError(
            f"Relative placement references missing scene entity {exc.args[0]!r}."
        ) from exc
    if relation in {"on", "above"}:
        object_bottom, _ = _vertical_mesh_bounds(
            obj,
            axis_aligned=object_id in axis_align_objects,
        )
        if reference_id == "table":
            if table_top_z is None:
                raise ValueError(
                    "Relative placement on the table requires a derived tabletop height."
                )
            reference_position = source_position(reference)
            reference_top = float(table_top_z) - reference_position[2]
            object_position = source_position(obj)
            if relation == "on":
                object_position = _table_landing_position(
                    object_id,
                    scene_objects,
                    axis_aligned=object_id in axis_align_objects,
                    table_top_z=float(table_top_z),
                )
            x_offset = object_position[0] - reference_position[0]
            y_offset = object_position[1] - reference_position[1]
        else:
            _, reference_top = _vertical_mesh_bounds(
                reference,
                axis_aligned=reference_id in axis_align_objects,
            )
            x_offset = 0.0
            y_offset = 0.0
        return [
            x_offset,
            y_offset,
            reference_top - object_bottom + _PLACEMENT_CLEARANCE,
        ]

    diagonal_relations = {
        "front_left_of": ("front_of", "left_of"),
        "front_right_of": ("front_of", "right_of"),
        "back_left_of": ("behind", "left_of"),
        "back_right_of": ("behind", "right_of"),
    }
    if relation in diagonal_relations:
        components = [
            _relative_world_displacement(
                component,
                object_id=object_id,
                reference_id=reference_id,
                scene_objects=scene_objects,
                axis_align_objects=axis_align_objects,
                table_top_z=table_top_z,
            )
            for component in diagonal_relations[relation]
        ]
        return [components[0][0], components[1][1], components[0][2]]

    if table_top_z is None:
        raise ValueError(
            "Lateral placement requires a measured tabletop landing height."
        )
    object_bottom, _ = _vertical_mesh_bounds(
        obj, axis_aligned=object_id in axis_align_objects
    )
    object_support_z = float(table_top_z) - object_bottom
    reference_support_z = _support_origin_z(
        reference,
        axis_aligned=reference_id in axis_align_objects,
        table_top_z=table_top_z,
    )
    displacement = [
        0.0,
        0.0,
        object_support_z - reference_support_z + _PLACEMENT_CLEARANCE,
    ]
    if relation == "right_of":
        displacement[1] = _horizontal_relation_distance(
            obj,
            reference,
            world_axis=1,
            object_axis_aligned=object_id in axis_align_objects,
            reference_axis_aligned=reference_id in axis_align_objects,
            minimum=_LATERAL_RELATION_DISTANCE,
        )
    elif relation == "left_of":
        displacement[1] = -_horizontal_relation_distance(
            obj,
            reference,
            world_axis=1,
            object_axis_aligned=object_id in axis_align_objects,
            reference_axis_aligned=reference_id in axis_align_objects,
            minimum=_LATERAL_RELATION_DISTANCE,
            reference_direction=-1,
        )
    elif relation == "front_of":
        displacement[0] = -_horizontal_relation_distance(
            obj,
            reference,
            world_axis=0,
            object_axis_aligned=object_id in axis_align_objects,
            reference_axis_aligned=reference_id in axis_align_objects,
            minimum=_FRONT_RELATION_DISTANCE,
            reference_direction=-1,
        )
    elif relation == "behind":
        displacement[0] = _horizontal_relation_distance(
            obj,
            reference,
            world_axis=0,
            object_axis_aligned=object_id in axis_align_objects,
            reference_axis_aligned=reference_id in axis_align_objects,
            minimum=_FRONT_RELATION_DISTANCE,
        )
    else:
        raise ValueError(f"Unsupported relative placement relation {relation!r}.")
    return displacement


def _support_origin_z(
    source: dict[str, Any],
    *,
    axis_aligned: bool,
    table_top_z: float | None,
) -> float:
    """Return the object-origin height when resting on the scene table."""
    if not axis_aligned:
        return source_position(source)[2]
    if table_top_z is None:
        raise ValueError("Axis-aligned placement requires a derived tabletop height.")
    bottom, _ = _vertical_mesh_bounds(source, axis_aligned=True)
    return float(table_top_z) - bottom + _PLACEMENT_CLEARANCE


def _single_target_pose(
    graph: SemanticTaskGraph,
    target_id: str,
) -> dict[str, Any]:
    """Return one exact generated target pose."""
    target = graph["targets"].get(target_id)
    if not isinstance(target, dict):
        raise ValueError(f"Missing generated target {target_id!r}.")
    values = target.get("values")
    if not isinstance(values, list) or len(values) != 1:
        raise ValueError(f"Generated target {target_id!r} must contain one pose.")
    pose = values[0]
    if (
        not isinstance(pose, dict)
        or not isinstance(pose.get("position"), list)
        or len(pose["position"]) != 3
        or not isinstance(pose.get("quaternion_xyzw"), list)
        or len(pose["quaternion_xyzw"]) != 4
    ):
        raise ValueError(f"Generated target {target_id!r} has an invalid pose.")
    return pose


def _upright_release_clearance(source: dict[str, Any]) -> float:
    """Return the E2 recipe's drop distance above the final support surface."""
    extents = _local_extents(source)
    if extents is None:
        raise ValueError("Upright release requires finite object extents.")
    ordered_extents = sorted(extents)
    slenderness = ordered_extents[-1] / max(ordered_extents[-2], 1.0e-6)
    return (
        _SLENDER_UPRIGHT_RELEASE_CLEARANCE
        if slenderness >= 2.5
        else _UPRIGHT_RELEASE_CLEARANCE
    )


def resolve_relative_placement_routes(
    graph: SemanticTaskGraph,
    scene: Any,
    *,
    settled: bool = False,
    upright_only: bool = False,
    coordinated_only: bool = False,
) -> list[dict[str, Any]]:
    """Resolve stage-qualified placement routes from preceding semantic calls.

    Args:
        graph: Semantic graph with ordered nodes and declared targets.
        scene: Prepared physical scene exposing planner objects, articulations
            and measured table height. No simulator is constructed.
        settled: Resolve the corresponding acceptance target after settling.
        upright_only: Select declared upright-placement calls.
        coordinated_only: Select coordinated relative-transport calls.

    Returns:
        Deterministically sorted route declarations with independent stage IDs.
        Earlier placements reserve stationary support footprints until a repick.
        Geometry proposals are not evidence of physical execution or success.
    """
    axis_align_objects: set[str] = set()
    scene_objects = {str(item["runtime_uid"]): item for item in scene.planner_objects}

    spatial_references = _graph_references(graph, scene)
    support_subjects: dict[str, set[str]] = {}
    manipulated = {
        node["call"].get("object", node["call"].get("arguments", {}).get("object"))
        for node in graph["nodes"]
    }
    for node in graph["nodes"]:
        call = node["call"]
        arguments = call.get("arguments", {})
        reference_id = str(arguments.get("reference", ""))
        if (
            node.get("task_type") == "E1"
            and call.get("call_id") == _PLACE_RELATIVE_CALL_ID
            and arguments.get("relation") == "on"
            and reference_id != "table"
            and reference_id not in manipulated
            and scene_objects.get(reference_id, {}).get("role") == "rigid_object"
            and "reference_part" not in arguments
        ):
            support_subjects.setdefault(reference_id, set()).add(
                str(arguments["object"])
            )
    shared_supports = {
        reference
        for reference, subjects in support_subjects.items()
        if len(subjects) > 1
    }
    landings: dict[str, dict[str, Any]] = {}
    actuated_references = {
        node["call"].get("arguments", {}).get("object")
        for node in graph["nodes"]
        if node["call"].get("call_id") == SLIDE_CALL
    }
    routes: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    reference_heights: dict[str, float] = {}
    lateral_references = {
        node["call"]["arguments"]["reference"]
        for node in graph["nodes"]
        if node.get("task_type") == "E1"
        and node["call"].get("call_id") == _PLACE_RELATIVE_CALL_ID
        and node["call"]["arguments"].get("relation") not in {"on", "above"}
    }

    for node in graph["nodes"]:
        call = node["call"]
        if call["kind"] == "hand_over" or _is_handover_source_call_id(
            call.get("call_id")
        ):
            reference_heights.pop(
                str(call.get("object", call.get("arguments", {}).get("object"))), None
            )
        if call["kind"] == "pick":
            reference_heights.pop(str(call["object"]), None)
            for reservations in landings.values():
                reservations.pop(str(call["object"]), None)
        if (
            node.get("task_type") == "E1"
            and call["kind"] == "place"
            and "inside" in call
            and str(call["object"]) in lateral_references - axis_align_objects
        ):
            container_id = _inside_parts(str(call["inside"]))[0]
            if container_id not in manipulated | axis_align_objects:
                height = _inside_reference_height(call, scene_objects)
                if height is not None:
                    reference_heights[str(call["object"])] = height
        if call["kind"] != "registered":
            continue
        if call["call_id"] in {_AXIS_ALIGN_CALL_ID, _ALIGN_HELD_CALL_ID} or (
            node.get("task_type") == "E2"
            and call["call_id"] == _MOVE_HELD_OBJECT_CALL_ID
        ):
            axis_align_objects.add(str(call["arguments"]["object"]))
        selected_calls = (
            {_COORDINATED_TRANSPORT_CALL_ID}
            if coordinated_only
            else (
                {_UPRIGHT_PLACE_CALL_ID}
                if upright_only
                else {_PLACE_RELATIVE_CALL_ID, _STACK_PLACE_CALL_ID}
            )
        )
        if call["call_id"] not in selected_calls:
            continue
        arguments = call["arguments"]
        if coordinated_only and "reference" not in arguments:
            continue
        if coordinated_only and (
            arguments.get("relation") != "on"
            or arguments["reference"] == arguments["object"]
        ):
            raise ValueError(
                "Coordinated relative placement requires on a distinct rigid support."
            )
        stage_id = _route_key_for_node(
            node,
            str(arguments["object"]),
            str(arguments["reference"]),
            str(arguments["relation"]),
        )[0]
        selector = (
            stage_id,
            str(arguments["object"]),
            str(arguments["reference"]),
            str(arguments["relation"]),
        )
        _, object_id, reference_id, relation = selector
        spatial_reference = spatial_references.get(
            (reference_id, arguments.get("reference_part"))
        )
        if reference_id in actuated_references and spatial_reference is None:
            raise ValueError(
                f"Spatial reference {reference_id!r} changes joints in this program; "
                "use qualified link geometry instead of an authored root envelope."
            )
        if scene_objects.get(reference_id, {}).get(
            "role"
        ) == "articulation" and relation in {
            "on",
            "above",
        }:
            raise ValueError(
                "Articulated supports require qualified link bindings; root references support side relations only."
            )
        if reference_id == "table" and relation not in {"on", "above"}:
            raise ValueError(
                "A lateral table-relative placement has no supported landing surface. "
                "Bind the named relation anchor, not the surrounding tabletop region."
            )
        if node.get("task_type") == "E2":
            target_id = f"{node['task_instance_id']}_upright_target"
            target = _single_target_pose(graph, target_id)
            reference_position = source_position(scene_objects[reference_id])
            displacement = [
                float(target["position"][index]) - reference_position[index]
                for index in range(3)
            ]
            if settled:
                # Release clearance is a command target, not a resting height.
                displacement[2] -= _upright_release_clearance(scene_objects[object_id])
        else:
            placement_objects = scene_objects
            if (
                reference_id in reference_heights
                and reference_id not in axis_align_objects
                and relation not in {"on", "above"}
            ):
                reference_position = list(source_position(scene_objects[reference_id]))
                reference_position[2] = reference_heights[reference_id]
                placement_objects = {
                    **scene_objects,
                    reference_id: {
                        **scene_objects[reference_id],
                        "init_pos": reference_position,
                    },
                }
            if reference_id == "table" and relation == "on":
                # The planner view omits fixed-base and joint configuration.
                # Endpoint clearance needs the actual authored mechanisms.
                placement_objects = {
                    **scene_objects,
                    **{
                        str(config["uid"]): {
                            **scene_objects.get(str(config["uid"]), {}),
                            **config,
                            "runtime_uid": str(config["uid"]),
                            "role": "articulation",
                        }
                        for config in getattr(scene, "articulations", ())
                    },
                }
            displacement = _relative_world_displacement(
                relation,
                object_id=object_id,
                reference_id=reference_id,
                scene_objects=(
                    {
                        **placement_objects,
                        reference_id: spatial_reference.planner_object(),
                    }
                    if spatial_reference is not None
                    else placement_objects
                ),
                axis_align_objects=axis_align_objects,
                table_top_z=scene.table_top_z,
            )
            if spatial_reference is not None:
                if scene.table_top_z is None:
                    raise ValueError(
                        "Articulated side placement requires a measured tabletop."
                    )
                bottom, _ = _vertical_mesh_bounds(
                    scene_objects[object_id],
                    axis_aligned=object_id in axis_align_objects,
                )
                displacement[2] = (
                    float(scene.table_top_z)
                    - bottom
                    + _PLACEMENT_CLEARANCE
                    - float(spatial_reference.initial_pose[2, 3])
                )
            if (
                not settled
                and node.get("task_type") == "E1"
                and call["call_id"] == _PLACE_RELATIVE_CALL_ID
                and relation in {"left_of", "right_of", "front_of", "behind"}
                and spatial_reference is None
                and scene_objects[reference_id].get("role") == "rigid_object"
            ):
                # Release above the resting target; settled evidence stays unchanged.
                displacement[2] += _LATERAL_RELEASE_CLEARANCE - _PLACEMENT_CLEARANCE
            if settled and (
                call["call_id"] == _STACK_PLACE_CALL_ID or coordinated_only
            ):
                displacement[2] -= _PLACEMENT_CLEARANCE
        if (
            reference_id in shared_supports
            and node.get("task_type") == "E1"
            and call["call_id"] == _PLACE_RELATIVE_CALL_ID
            and relation == "on"
            and object_id not in axis_align_objects
            and spatial_reference is None
        ):
            reservations = landings.setdefault(reference_id, {})
            reservations.pop(object_id, None)
            obj, reference = scene_objects[object_id], scene_objects[reference_id]
            position = np.asarray(source_position(reference)) + displacement
            placed, conflicts = reserved_placement_footprint(
                _mesh_vertices(obj),
                source_rotation(obj),
                position,
                reservations.values(),
                separation_m=_RELATION_CLEARANCE,
            )
            if conflicts:
                displacement, placed = _unoccupied_support_landing(
                    obj, reference, displacement, tuple(reservations.values())
                )
            reservations[object_id] = placed
        route = {
            "stage_id": stage_id,
            "object_id": object_id,
            "reference_entity_id": reference_id,
            "relation": relation,
            "world_displacement": displacement,
            **(
                {"reference_link_id": spatial_reference.link_id}
                if spatial_reference is not None
                else {}
            ),
            **(
                {"reference_part": spatial_reference.part_id}
                if spatial_reference is not None
                and spatial_reference.part_id is not None
                else {}
            ),
        }
        # Long-horizon programs may place one object/reference pair more than
        # once.  The graph task instance is the occurrence selector; do not
        # collapse routes from different stages.
        if selector in routes and routes[selector] != route:
            raise ValueError(
                f"Repeated placement {selector!r} requires stage-specific routes; "
                f"existing={routes[selector]!r}, new={route!r}."
            )
        routes[selector] = route
    return [routes[selector] for selector in sorted(routes)]
