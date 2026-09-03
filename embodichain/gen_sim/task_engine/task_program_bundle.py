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

"""Materialize one SemanticTaskGraph as a configured Task Program bundle."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
import shutil
from typing import Any, Final

import numpy as np

from embodichain.lab.task_program.integrations._configured_composition import (
    _load_configured_task_program_deployment,
)
from embodichain.lab.task_program.language import load_task_program
from embodichain.utils.utility import load_config, save_config

from .semantic_graph import SemanticTaskGraph, validate_semantic_task_graph

__all__ = ["TaskProgramBundlePaths", "generate_task_program_bundle"]

_EMBODIMENT_COMPONENTS: Final = {
    "dual_franka": "dual_franka_robotiq_arg2f_140.yaml",
    "dual_franka_robotiq_arg2f_140": "dual_franka_robotiq_arg2f_140.yaml",
}

# The generated ``move forward`` task intent has no metric distance.  Phase one
# supports only the dual-Franka embodiment, whose two-arm top-down tray grasp
# retains a reachable continuation over 0.14 m.  Keep this semantic route
# target in the integration builder: the Atomic Action must execute the exact
# grounded goal and must not silently clamp an unreachable caller request.
_DUAL_FRANKA_COORDINATED_TRANSPORT_DISTANCE: Final = 0.14
_DUAL_FRANKA_HANDOVER_CLEARANCE: Final = 0.193
_DUAL_FRANKA_TABLE_MOUNT_OFFSET: Final = 0.35
_LATERAL_RELATION_DISTANCE: Final = 0.10
_FRONT_RELATION_DISTANCE: Final = 0.18
# Leave enough free space around a placed object's support reference for the
# configured parallel-jaw fingers to close during a later semantic Pick.  A
# tall object aligned by E2 needs the larger margin because its side grasp
# sweeps the finger length through the support plane.  Both routes still come
# from scene geometry rather than task-owned robot poses.
_RELATION_CLEARANCE: Final = 0.02
_AXIS_ALIGNED_RELATION_CLEARANCE: Final = 0.04
_PLACEMENT_CLEARANCE: Final = 0.01
_RELATIVE_POSITION_TOLERANCE: Final = 0.04
_AXIS_ALIGN_CALL_ID: Final = "simulation.axis_align"
_COORDINATED_TRANSPORT_CALL_ID: Final = "simulation.coordinated_transport"
_PARK_CALL_ID: Final = "simulation.park"
_PLACE_RELATIVE_CALL_ID: Final = "simulation.place_relative"


@dataclass(frozen=True, slots=True)
class TaskProgramBundlePaths:
    """Files composing one portable configured Task Program deployment.

    Attributes:
        root: Bundle root directory.
        deployment: Runnable Gym deployment configuration.
        program: Embodiment-independent Task Program source.
        integration: Scene and runtime-service integration configuration.
        scene: Physical scene component.
        embodiment: Robot, sensor, and skill-profile component.
        execution_policy: Canonical runtime execution-policy component.
        semantic_task_graph: Immutable source semantic graph.
        integration_fingerprint: Composed integration identity artifact.
    """

    root: Path
    deployment: Path
    program: Path
    integration: Path
    scene: Path
    embodiment: Path
    execution_policy: Path
    semantic_task_graph: Path
    integration_fingerprint: Path


def generate_task_program_bundle(
    graph: SemanticTaskGraph,
    prepared_scene: Any,
    output_dir: str | Path,
    *,
    robot_profile: str,
    max_episodes: int | None = None,
    max_episode_steps: int | None = None,
) -> tuple[SemanticTaskGraph, TaskProgramBundlePaths]:
    """Write, compose, and provider-free preflight one semantic deployment.

    Args:
        graph: Provider-free semantic task graph. Its provisional fingerprint
            is replaced with the exact composed integration fingerprint.
        prepared_scene: Scene Adapter output with physical and planner views.
        output_dir: Fresh bundle staging directory.
        robot_profile: Task Engine robot-profile selector. Phase one supports
            only the canonical dual-Franka embodiment.
        max_episodes: Optional Gym episode limit.
        max_episode_steps: Optional Gym step limit.

    Returns:
        Final fingerprint-bound graph and all generated paths.

    Raises:
        ValueError: If the robot profile is unsupported or graph/scene
            integration cannot be composed and preflighted.
    """
    selected_graph = validate_semantic_task_graph(graph)
    normalized_profile = str(robot_profile).strip()
    try:
        embodiment_filename = _EMBODIMENT_COMPONENTS[normalized_profile]
    except KeyError as exc:
        raise ValueError(
            "Phase-one semantic bundle generation supports only dual_franka; "
            f"received robot_profile={normalized_profile!r}."
        ) from exc
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    component_root = root / "components"
    task_program_root = root / "task_program"
    component_root.mkdir(parents=True, exist_ok=True)
    task_program_root.mkdir(parents=True, exist_ok=True)

    from embodichain.gen_sim.task_engine.orchestration.scene_assets import (
        normalize_scene_assets,
    )

    scene = normalize_scene_assets(prepared_scene, root)
    paths = TaskProgramBundlePaths(
        root=root,
        deployment=root / "task_program_deployment.yaml",
        program=task_program_root / "program.yaml",
        integration=task_program_root / "integration.yaml",
        scene=component_root / "scene.yaml",
        embodiment=component_root / "embodiment.yaml",
        execution_policy=component_root / "execution_policy.yaml",
        semantic_task_graph=root / "semantic_task_graph.json",
        integration_fingerprint=root / "integration_fingerprint.json",
    )
    project_root = Path(__file__).resolve().parents[3]
    embodiment_source = (
        project_root
        / "embodichain_tasks/configs/components/embodiments"
        / embodiment_filename
    )
    policy_source = (
        project_root
        / "embodichain_tasks/configs/components/execution_policies"
        / "dual_arm_trajectory_verified.yaml"
    )
    embodiment_payload = load_config(embodiment_source)
    _bind_embodiment_to_scene(embodiment_payload, table_top_z=scene.table_top_z)
    save_config(paths.embodiment, embodiment_payload)
    shutil.copy2(policy_source, paths.execution_policy)

    program_id = _program_identifier(selected_graph["task_id"])
    scene_contract = f"{program_id}_scene_v1"
    save_config(paths.program, _program_payload(selected_graph, program_id, scene))
    save_config(
        paths.integration,
        _integration_payload(
            selected_graph,
            scene,
            program_id=program_id,
            scene_contract=scene_contract,
        ),
    )
    save_config(paths.scene, _scene_payload(scene, program_id=program_id))
    save_config(
        paths.deployment,
        {
            "id": f"GenSimTaskProgram-{program_id}-v1",
            "max_episodes": int(max_episodes or 1),
            "max_episode_steps": int(max_episode_steps or 6000),
            "num_envs": 1,
            "arena_space": 2.5,
            "physics_config": {"enable_ccd": True},
            "env": {
                "sim_steps_per_control": 4,
                "events": {
                    "settle_objects_on_reset": {
                        "func": "wait_for_dynamic_objects_to_settle",
                        "mode": "reset",
                        "params": {
                            "entity_cfgs": [
                                {"uid": str(item["uid"])}
                                for item in scene.rigid_objects
                            ],
                            "min_steps": 10,
                            # Tall generated objects may need several seconds
                            # to finish a final low-energy roll after import.
                            "max_steps": 600,
                            "check_interval_steps": 2,
                            "required_stable_checks": 3,
                            "timeout_behavior": "raise",
                        },
                    }
                },
                "extensions": {},
            },
            "scene": {"component": "components/scene.yaml"},
            "embodiment": {"component": "components/embodiment.yaml"},
            "task_program": {
                "program": "task_program/program.yaml",
                "integration": "task_program/integration.yaml",
                "execution_policy": "components/execution_policy.yaml",
            },
        },
    )

    embodiment = load_config(paths.embodiment)
    deployment = _load_configured_task_program_deployment(
        task_program=load_config(paths.deployment)["task_program"],
        skill_profile=embodiment["skill_profile"],
        base_dir=root,
    )
    fingerprint = deployment.integration.integration_fingerprint
    selected_graph["integration_fingerprint"] = fingerprint
    selected_graph = validate_semantic_task_graph(selected_graph)
    _write_json(paths.semantic_task_graph, selected_graph)
    _write_json(
        paths.integration_fingerprint,
        {
            "schema_version": "semantic_integration_fingerprint/v1",
            "integration_id": deployment.integration_id,
            "integration_fingerprint": fingerprint,
            "registration_fingerprint": deployment.integration.registration.fingerprint,
        },
    )

    program = load_task_program(
        paths.program,
        integration=deployment.selection,
        validation_context=deployment.integration.registration.catalog,
    )
    deployment.integration.registration.catalog.preflight(program)
    return selected_graph, paths


def _program_payload(
    graph: SemanticTaskGraph,
    program_id: str,
    scene: Any,
) -> dict[str, Any]:
    relative_routes = {
        (
            route["object_id"],
            route["reference_entity_id"],
            route["relation"],
        ): route
        for route in _relative_place_route_payloads(graph, scene)
    }
    return {
        "program_id": program_id,
        "targets": deepcopy(graph["targets"]),
        "program": {
            "kind": "sequence",
            "items": [
                _program_node(node, relative_routes=relative_routes)
                for node in graph["nodes"]
            ],
        },
    }


def _program_node(
    node: dict[str, Any],
    *,
    relative_routes: dict[tuple[str, str, str], dict[str, Any]],
) -> dict[str, Any]:
    """Materialize one task node as one canonical runtime segment."""
    call = deepcopy(node["call"])
    segment: dict[str, Any] = {
        "kind": "segment",
        "name": str(node["id"]),
        "steps": {"kind": "invoke", "call": call},
    }
    settle_entities: list[str] = []
    settle_preset = "rigid_object"
    if call["kind"] == "place":
        inside = call.get("inside")
        if inside is not None:
            parts = str(inside).split("__")
            if len(parts) != 3 or parts[0] != "inside":
                raise ValueError(f"Unsupported generated inside affordance {inside!r}.")
            settle_preset = "contained_rigid_object"
        settle_entities.append(str(call["object"]))
    elif call["kind"] == "registered" and call["call_id"] in {
        _COORDINATED_TRANSPORT_CALL_ID,
        _PLACE_RELATIVE_CALL_ID,
    }:
        if call["call_id"] == _COORDINATED_TRANSPORT_CALL_ID:
            settle_preset = "transported_rigid_object"
        settle_entities.append(str(call["arguments"]["object"]))
    if settle_entities:
        segment["post"] = [
            {
                "kind": "wait_stable",
                "entity": settle_entity,
                "preset": settle_preset,
            }
            for settle_entity in dict.fromkeys(settle_entities)
        ]
    if call["kind"] == "registered" and call["call_id"] == _PLACE_RELATIVE_CALL_ID:
        arguments = call["arguments"]
        selector = (
            str(arguments["object"]),
            str(arguments["reference"]),
            str(arguments["relation"]),
        )
        try:
            route = relative_routes[selector]
        except KeyError as exc:
            raise ValueError(
                f"Relative placement has no generated route for {selector!r}."
            ) from exc
        segment["validators"] = [
            {
                "kind": "object_near_relative_target",
                "object": route["object_id"],
                "reference": route["reference_entity_id"],
                "displacement": deepcopy(route["world_displacement"]),
                "position_tolerance": _RELATIVE_POSITION_TOLERANCE,
            }
        ]
    return segment


def _integration_payload(
    graph: SemanticTaskGraph,
    scene: Any,
    *,
    program_id: str,
    scene_contract: str,
) -> dict[str, Any]:
    scene_objects = {str(item["runtime_uid"]): item for item in scene.planner_objects}
    referenced_objects: set[str] = set()
    inside_routes: list[tuple[str, str, str]] = []
    coordinated_routes: list[tuple[str, str]] = []
    axis_align_objects: set[str] = set()
    relative_place_routes: set[tuple[str, str, str]] = set()
    has_park_call = False
    for node in graph["nodes"]:
        call = node["call"]
        if call["kind"] in {"pick", "place", "hand_over"}:
            referenced_objects.add(str(call["object"]))
        if call["kind"] == "place" and "inside" in call:
            affordance = str(call["inside"])
            parts = affordance.split("__")
            if len(parts) != 3 or parts[0] != "inside":
                raise ValueError(
                    f"Unsupported generated inside affordance {affordance!r}."
                )
            container_id, object_id = parts[1], parts[2]
            referenced_objects.add(container_id)
            inside_routes.append((affordance, container_id, object_id))
        if (
            call["kind"] == "registered"
            and call["call_id"] == _COORDINATED_TRANSPORT_CALL_ID
        ):
            arguments = call["arguments"]
            object_id = str(arguments["object"])
            referenced_objects.add(object_id)
            coordinated_routes.append((object_id, str(arguments["target"])))
        elif call["kind"] == "registered" and call["call_id"] == _AXIS_ALIGN_CALL_ID:
            object_id = str(call["arguments"]["object"])
            referenced_objects.add(object_id)
            axis_align_objects.add(object_id)
        elif (
            call["kind"] == "registered" and call["call_id"] == _PLACE_RELATIVE_CALL_ID
        ):
            arguments = call["arguments"]
            object_id = str(arguments["object"])
            reference_id = str(arguments["reference"])
            relation = str(arguments["relation"])
            referenced_objects.update((object_id, reference_id))
            relative_place_routes.add((object_id, reference_id, relation))
        elif call["kind"] == "registered" and call["call_id"] == _PARK_CALL_ID:
            has_park_call = True
        elif call["kind"] == "registered":
            raise ValueError(
                f"Unsupported generated registered call {call['call_id']!r}."
            )

    rigid_bindings: list[dict[str, Any]] = []
    for entity_id in sorted(referenced_objects | {"table"}):
        source = scene_objects.get(entity_id)
        if source is None:
            raise ValueError(
                f"Semantic graph references missing scene entity {entity_id!r}."
            )
        affordances: list[dict[str, Any]] = []
        if str(source["role"]) == "rigid_object":
            grasp: dict[str, Any] = {
                "entity_id": f"{entity_id}_grasp",
                "kind": "antipodal_grasp",
            }
            if entity_id in axis_align_objects:
                grasp["internal_axis"] = _dominant_local_axis(source)
            affordances.append(grasp)
        for affordance_id, container_id, object_id in inside_routes:
            if container_id != entity_id:
                continue
            lateral = 0.06 if "apple" in object_id else -0.06
            affordances.append(
                {
                    "entity_id": affordance_id,
                    "kind": "container",
                    "native_name": affordance_id,
                    "object_target_pose": _translation_pose(lateral, 0.0, 0.008),
                    "release_clearance": 0.12,
                }
            )
        rigid_bindings.append(
            {
                "entity_id": entity_id,
                "simulation_uid": entity_id,
                "dynamics": (
                    "dynamic" if str(source["role"]) == "rigid_object" else "static"
                ),
                "semantic_type": str(source.get("category") or entity_id),
                "affordances": affordances,
            }
        )

    coordinated_lowerer_routes = []
    for object_id, target_id in coordinated_routes:
        coordinated_lowerer_routes.append(
            {
                "object_id": object_id,
                "target_id": target_id,
                "world_displacement": [
                    -_DUAL_FRANKA_COORDINATED_TRANSPORT_DISTANCE,
                    0.0,
                    0.0,
                ],
            }
        )
    relative_lowerer_routes = _relative_place_route_payloads(graph, scene)
    handover_position_z = _handover_position_z(scene.table_top_z)
    return {
        "integration_id": f"{program_id}_integration_v1",
        "program_id": program_id,
        "requires": {
            "scene_contract": scene_contract,
            "embodiment_contract": "dual_arm_parallel_gripper",
        },
        "scene_binding": {
            "contract_id": scene_contract,
            "registry_id": f"{program_id}_scene_registry",
            "rigid_objects": rigid_bindings,
        },
        "profile": {
            "defaults": {
                "pick_up": {"primary": "left"},
                "place": {"primary": "left"},
                "hand_over": {"source": "left", "destination": "right"},
                "axis_align": {"primary": "left"},
                "coordinated_pickment": {"left": "left", "right": "right"},
            },
            "action_options": {
                "pick": {
                    "kind": "pick_up",
                    "pre_grasp_distance": 0.15,
                    "lift_height": 0.16,
                    "hand_interp_steps": 5,
                    "grasp_settle_steps": 0,
                    "grasp_commit_fraction": 1.0,
                },
                "place": {
                    "kind": "place",
                    "hand_interp_steps": 12,
                    "release_settle_steps": 60,
                    "cartesian_waypoint_count": 2,
                    "preserve_current_object_orientation": True,
                },
                "hand_over": {
                    "kind": "hand_over",
                    "pre_grasp_distance": 0.08,
                    "lift_height": 0.04,
                    "hand_interp_steps": 10,
                    "hold_steps": 4,
                    "retreat_steps": 28,
                    "retreat_distance": 0.12,
                    "receive_pick_object_part": "bottom",
                    "release_at_target": False,
                    "arm_selection": "bound",
                },
                **(
                    {
                        _AXIS_ALIGN_CALL_ID: {
                            "kind": "axis_align",
                            "pre_grasp_distance": 0.15,
                            "lift_height": 0.16,
                            "hand_interp_steps": 5,
                            "grasp_settle_steps": 0,
                            "grasp_commit_fraction": 1.0,
                            "target_axis": [0.0, 0.0, 1.0],
                        }
                    }
                    if axis_align_objects
                    else {}
                ),
                **(
                    {
                        _PLACE_RELATIVE_CALL_ID: {
                            "kind": "place",
                            "hand_interp_steps": 12,
                            "release_settle_steps": 60,
                            "cartesian_waypoint_count": 2,
                            "preserve_current_object_orientation": True,
                        }
                    }
                    if relative_lowerer_routes
                    else {}
                ),
                **({_PARK_CALL_ID: {"kind": "move_joints"}} if has_park_call else {}),
                **(
                    {
                        _COORDINATED_TRANSPORT_CALL_ID: {
                            "kind": "coordinated_pickment",
                            "object_motion_keyframes": 8,
                            "pre_grasp_distance": 0.10,
                            "lift_height": 0.08,
                            "hand_interp_steps": 10,
                            "hold_steps": 4,
                            "release": True,
                            "release_steps": 10,
                            "retreat_distance": 0.08,
                            "retreat_steps": 12,
                            "approach_direction": [0.0, 0.0, -1.0],
                            "left_to_right_arm_direction": [0.0, 1.0, 0.0],
                            "middle_empty_ratio": 0.4,
                        }
                    }
                    if coordinated_lowerer_routes
                    else {}
                ),
            },
            # Effect truth stays in Semantic Skill.  The generated task
            # planner only selects the canonical built-in monitor for the
            # curated effectful calls; it never computes a held relation or
            # synthesizes a successful postcondition itself.
            "effect_monitors": {
                semantic_id: {
                    "monitor_id": "builtin.composite_effect",
                    "revision": "1",
                    "params": {
                        "consecutive_samples": 3,
                        # Generated cube/apple meshes admit equivalent grasp
                        # orientations.  Translation and the gripper/contact
                        # clause remain strict physical checks; orientation
                        # is intentionally relaxed for this calibrated scene.
                        "attached_translation_threshold": 0.06,
                        "attached_rotation_threshold": 3.0,
                        "detached_translation_threshold": 0.08,
                        "detached_rotation_threshold": 3.141592653589793,
                    },
                }
                for semantic_id in (
                    "pick",
                    "place",
                    "hand_over",
                    _AXIS_ALIGN_CALL_ID,
                    _COORDINATED_TRANSPORT_CALL_ID,
                    _PLACE_RELATIVE_CALL_ID,
                )
                if any(
                    node["call"].get("kind") == semantic_id
                    or node["call"].get("call_id") == semantic_id
                    for node in graph["nodes"]
                )
            },
            "grounding_providers": {"hand_over": "simulation.configured_handover_pose"},
        },
        "runtime_services": {
            "handover_pose_providers": [
                {
                    "kind": "configured_pose",
                    # Keep the exchange above the tray rim.  This is a
                    # semantic object-space staging target; the Atomic Action
                    # derives all arm/EEF poses from it at runtime.
                    "final_position": [0.0, -0.08, handover_position_z],
                    "final_quaternion_wxyz": [
                        0.7071067812,
                        0.7071067812,
                        0.0,
                        0.0,
                    ],
                }
            ],
            "registered_semantic_lowerers": [
                *(
                    [
                        {
                            "kind": "axis_align",
                            "object_ids": sorted(axis_align_objects),
                        }
                    ]
                    if axis_align_objects
                    else []
                ),
                *([{"kind": "park"}] if has_park_call else []),
                *(
                    [{"kind": "place_relative", "routes": relative_lowerer_routes}]
                    if relative_lowerer_routes
                    else []
                ),
                *(
                    [
                        {
                            "kind": "coordinated_transport",
                            "routes": coordinated_lowerer_routes,
                        }
                    ]
                    if coordinated_lowerer_routes
                    else []
                ),
            ],
        },
    }


def _scene_payload(scene: Any, *, program_id: str) -> dict[str, Any]:
    return {
        "scene_id": f"{program_id}_generated_scene",
        "simulation": {
            "light": {
                "direct": [
                    {
                        "uid": "main_light",
                        "color": [0.6, 0.6, 0.6],
                        "intensity": 30.0,
                        "init_pos": [0.5, 0.0, 3.0],
                    }
                ]
            },
            "background": [deepcopy(value) for value in scene.background],
            "rigid_object": [deepcopy(value) for value in scene.rigid_objects],
            "rigid_object_group": [],
            "articulation": [deepcopy(value) for value in scene.articulations],
        },
    }


def _bind_embodiment_to_scene(
    embodiment: dict[str, Any],
    *,
    table_top_z: float | None,
) -> None:
    """Bind the generated deployment's robot mount to the current tabletop."""
    if table_top_z is None or not math.isfinite(float(table_top_z)):
        raise ValueError("Dual-Franka deployment requires a derived tabletop height.")
    simulation = embodiment.get("simulation")
    if not isinstance(simulation, dict):
        raise ValueError("Embodiment component has no simulation mapping.")
    init_pos = simulation.get("init_pos")
    if not isinstance(init_pos, list) or len(init_pos) != 3:
        raise ValueError("Embodiment simulation.init_pos must contain three values.")
    simulation["init_pos"] = [
        float(init_pos[0]),
        float(init_pos[1]),
        float(table_top_z) - _DUAL_FRANKA_TABLE_MOUNT_OFFSET,
    ]


def _relative_place_route_payloads(
    graph: SemanticTaskGraph,
    scene: Any,
) -> list[dict[str, Any]]:
    """Build the single route projection shared by lowering and validation."""
    axis_align_objects: set[str] = set()
    selectors: set[tuple[str, str, str]] = set()
    for node in graph["nodes"]:
        call = node["call"]
        if call["kind"] != "registered":
            continue
        if call["call_id"] == _AXIS_ALIGN_CALL_ID:
            axis_align_objects.add(str(call["arguments"]["object"]))
        elif call["call_id"] == _PLACE_RELATIVE_CALL_ID:
            arguments = call["arguments"]
            selectors.add(
                (
                    str(arguments["object"]),
                    str(arguments["reference"]),
                    str(arguments["relation"]),
                )
            )
    scene_objects = {str(item["runtime_uid"]): item for item in scene.planner_objects}
    return [
        {
            "object_id": object_id,
            "reference_entity_id": reference_id,
            "relation": relation,
            "world_displacement": _relative_world_displacement(
                relation,
                object_id=object_id,
                reference_id=reference_id,
                scene_objects=scene_objects,
                axis_align_objects=axis_align_objects,
                table_top_z=scene.table_top_z,
            ),
        }
        for object_id, reference_id, relation in sorted(selectors)
    ]


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
            reference_position = _position(reference)
            reference_top = float(table_top_z) - reference_position[2]
            object_position = _position(obj)
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

    object_support_z = _support_origin_z(
        obj,
        axis_aligned=object_id in axis_align_objects,
        table_top_z=table_top_z,
    )
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
        )
    elif relation == "front_of":
        displacement[0] = -_horizontal_relation_distance(
            obj,
            reference,
            world_axis=0,
            object_axis_aligned=object_id in axis_align_objects,
            reference_axis_aligned=reference_id in axis_align_objects,
            minimum=_FRONT_RELATION_DISTANCE,
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


def _horizontal_relation_distance(
    obj: dict[str, Any],
    reference: dict[str, Any],
    *,
    world_axis: int,
    object_axis_aligned: bool,
    reference_axis_aligned: bool,
    minimum: float,
) -> float:
    """Return geometry-aware center separation for one planar relation."""
    return max(
        minimum,
        _horizontal_half_extent(
            obj,
            world_axis=world_axis,
            axis_aligned=object_axis_aligned,
        )
        + _horizontal_half_extent(
            reference,
            world_axis=world_axis,
            axis_aligned=reference_axis_aligned,
        )
        + (
            _AXIS_ALIGNED_RELATION_CLEARANCE
            if object_axis_aligned
            else _RELATION_CLEARANCE
        ),
    )


def _horizontal_half_extent(
    source: dict[str, Any],
    *,
    world_axis: int,
    axis_aligned: bool,
) -> float:
    """Return a conservative horizontal half extent in the intended pose."""
    vertices = _mesh_vertices(source)
    if axis_aligned:
        dominant_axis = int(np.argmax(np.ptp(vertices, axis=0)))
        horizontal_extents = np.delete(np.ptp(vertices, axis=0), dominant_axis)
        return 0.5 * float(horizontal_extents.max())
    from scipy.spatial.transform import Rotation

    world_vertices = Rotation.from_euler(
        "XYZ",
        source.get("init_rot", [0.0, 0.0, 0.0]),
        degrees=True,
    ).apply(vertices)
    return 0.5 * float(np.ptp(world_vertices[:, world_axis]))


def _support_origin_z(
    source: dict[str, Any],
    *,
    axis_aligned: bool,
    table_top_z: float | None,
) -> float:
    """Return the object-origin height when resting on the scene table."""
    if not axis_aligned:
        return _position(source)[2]
    if table_top_z is None:
        raise ValueError("Axis-aligned placement requires a derived tabletop height.")
    bottom, _ = _vertical_mesh_bounds(source, axis_aligned=True)
    return float(table_top_z) - bottom + _PLACEMENT_CLEARANCE


def _dominant_local_axis(source: dict[str, Any]) -> list[float]:
    """Return the unique major mesh axis in DexSim's object-local basis."""
    vertices = _mesh_vertices(source)
    extents = np.ptp(vertices, axis=0)
    order = np.argsort(extents)
    major = int(order[-1])
    if extents[major] <= max(float(extents[order[-2]]) * 1.25, 1.0e-6):
        raise ValueError(
            f"Scene object {source.get('runtime_uid')!r} has no unique dominant "
            "local axis for semantic upright alignment."
        )
    axis = [0.0, 0.0, 0.0]
    axis[major] = 1.0
    return axis


def _vertical_mesh_bounds(
    source: dict[str, Any],
    *,
    axis_aligned: bool,
) -> tuple[float, float]:
    """Return bottom/top offsets for the intended object orientation."""
    vertices = _mesh_vertices(source)
    if axis_aligned:
        axis = np.asarray(_dominant_local_axis(source), dtype=np.float64)
        heights = vertices @ axis
    else:
        from scipy.spatial.transform import Rotation

        heights = Rotation.from_euler(
            "XYZ",
            source.get("init_rot", [0.0, 0.0, 0.0]),
            degrees=True,
        ).apply(vertices)[:, 2]
    return float(heights.min()), float(heights.max())


def _mesh_vertices(source: dict[str, Any]) -> np.ndarray:
    """Load one configured mesh in DexSim's object-local coordinate basis."""
    shape = source.get("shape")
    if not isinstance(shape, dict) or not shape.get("fpath"):
        raise ValueError(
            f"Scene object {source.get('runtime_uid')!r} has no mesh geometry."
        )
    try:
        import trimesh

        loaded = trimesh.load(str(shape["fpath"]), force="scene")
        geometry = (
            loaded.to_geometry()
            if hasattr(loaded, "to_geometry")
            else loaded.dump(concatenate=True)
        )
        vertices = np.asarray(geometry.vertices, dtype=np.float64)
    except Exception as exc:
        raise ValueError(
            f"Could not inspect scene mesh for {source.get('runtime_uid')!r}: {exc}"
        ) from exc
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not vertices.size:
        raise ValueError(
            f"Scene object {source.get('runtime_uid')!r} has empty mesh geometry."
        )
    # DexSim converts glTF's Y-up vertices into its Z-up object-local basis.
    result = np.column_stack((vertices[:, 0], -vertices[:, 2], vertices[:, 1]))
    scale = np.asarray(source.get("body_scale", [1.0, 1.0, 1.0]), dtype=np.float64)
    if scale.shape != (3,) or not np.isfinite(scale).all():
        raise ValueError(
            f"Scene object {source.get('runtime_uid')!r} has invalid body_scale."
        )
    return result * scale


def _position(source: dict[str, Any]) -> tuple[float, float, float]:
    """Return one finite source position."""
    value = source.get("init_pos")
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(
            f"Scene object {source.get('runtime_uid')!r} has no three-value init_pos."
        )
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError(
            f"Scene object {source.get('runtime_uid')!r} has invalid init_pos."
        )
    return result


def _handover_position_z(table_top_z: float | None) -> float:
    """Place the dual-arm exchange at a scene-relative reachable clearance."""
    if table_top_z is None or not math.isfinite(float(table_top_z)):
        raise ValueError("Hand-over grounding requires a derived tabletop height.")
    return float(table_top_z) + _DUAL_FRANKA_HANDOVER_CLEARANCE


def _translation_pose(x: float, y: float, z: float) -> list[float]:
    return [
        1.0,
        0.0,
        0.0,
        float(x),
        0.0,
        1.0,
        0.0,
        float(y),
        0.0,
        0.0,
        1.0,
        float(z),
        0.0,
        0.0,
        0.0,
        1.0,
    ]


def _program_identifier(task_id: str) -> str:
    normalized = "".join(
        character if character.isalnum() or character == "_" else "_"
        for character in str(task_id).strip().lower()
    ).strip("_")
    if not normalized:
        raise ValueError("task_id does not contain a usable program identifier.")
    return f"gen_sim_{normalized}"


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
