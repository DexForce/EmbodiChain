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
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Final

import numpy as np

from embodichain.lab.task_program.language import load_task_program
from embodichain.utils.utility import load_config, save_config

from .semantic_graph import SemanticTaskGraph, validate_semantic_task_graph
from ._task_program.assembly import ADAPTER_CONTRACT, load_deployment

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
_DUAL_FRANKA_PLACE_TCP_CLEARANCE: Final = 0.22
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
_STACK_PLACE_CALL_ID: Final = "gen_sim.stack_place"
_STACK_PICK_CALL_ID: Final = "gen_sim.stack_pick"


_MOVE_HELD_OBJECT_CALL_ID: Final = "simulation.move_held_object"
_ALIGN_HELD_CALL_ID: Final = "gen_sim.align_held"
_CLEAR_RELEASED_CALL_ID: Final = "gen_sim.clear_released"


_PICK_CALL_ID: Final = "simulation.pick"

_POUR_CALL_ID: Final = "simulation.pour"

_COORDINATED_HOLD_CALL_ID: Final = "simulation.coordinated_hold"


_UPRIGHT_RELEASE_CLEARANCE: Final = 0.04

_SLENDER_UPRIGHT_RELEASE_CLEARANCE: Final = 0.01

_UPRIGHT_STAGING_CLEARANCE: Final = 0.20

_E2_RELEASE_SAFETY_MARGIN: Final = 0.02


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
    task_template: dict[str, Any] | None = None,
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
        task_template: Explicit bounded upright template for independent E2
            acceptance. Legacy graphs keep their original step-hash semantics.

    Returns:
        Final fingerprint-bound graph and all generated paths.

    Raises:
        ValueError: If the robot profile is unsupported or graph/scene
            integration cannot be composed and preflighted.
    """
    selected_graph = validate_semantic_task_graph(graph)
    task_binding = None
    if task_template is not None:
        from ._task_spec import binding_for_graph

        task_binding = binding_for_graph(task_template, selected_graph)
    if "task_spec" in selected_graph:
        raise ValueError(
            "TaskSpec bundle export requires final task evaluation before data "
            "submission; that runtime integration is not yet available."
        )
    unsupported = sorted(
        {node["task_type"] for node in selected_graph["nodes"]}
        - {"E1", "E2", "E3", "E4", "E5"}
    )
    if unsupported:
        raise ValueError(f"Task Engine supports only E1-E5, not {unsupported}.")
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
    selected_graph = _refine_upright_targets(selected_graph, scene)
    selected_graph = _refine_coordinated_targets(selected_graph, scene)
    for node in selected_graph["nodes"]:
        if node["call"].get("call_id") == _PICK_CALL_ID:
            node["call"]["call_id"] = f"gen_sim.pick.{node['task_instance_id']}"
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
    policy_payload = load_config(policy_source)
    # Centimetre-scale object goals need a settled arm before the core captures
    # the next measured grasp transform. Grasp endpoints use effect evidence,
    # not this arm-only terminal tracking metric.
    policy_payload["tracking"]["terminal_max_abs_error"] = 0.05
    policy_payload["tracking"]["consecutive_acceptances"] = 5
    policy_payload["tracking"]["terminal_settle_timeout"] = 3.0
    if any(node["call"]["kind"] == "hand_over" for node in selected_graph["nodes"]):
        # Preserve 44 frames each for transfer and receiver approach after the
        # generated hand-close, hold, release-settle, and retreat allocations.
        policy_payload["motion"]["sample_count"] = 180
    save_config(paths.execution_policy, policy_payload)

    program_id = _program_identifier(selected_graph["task_id"])
    scene_contract = f"{program_id}_scene_v1"
    stability = _task_stability_payload(selected_graph, scene, embodiment_payload)
    if task_binding is not None:
        from embodichain.lab.task_evaluation import UprightTaskEvaluator

        minimum_alignment = math.cos(UprightTaskEvaluator(task_template).max_tilt)
        for preset in stability["presets"].values():
            if (
                preset.get("entity") == task_binding["entity_id"]
                and "local_axis" in preset
            ):
                if list(preset["local_axis"]) != [0.0, 0.0, 1.0]:
                    raise ValueError(
                        "TaskSpec upright requires the asset local +Z axis."
                    )
                preset["minimum_alignment"] = minimum_alignment
    save_config(
        paths.program,
        _program_payload(
            selected_graph,
            program_id,
            scene,
            stability_presets=set(stability["presets"]),
        ),
    )
    _write_json(
        task_program_root / "constraints.json",
        stability,
    )
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
            "max_episode_steps": int(
                max_episode_steps
                or (
                    8000
                    if any(
                        cfg["kind"] == "stack" for cfg in stability["presets"].values()
                    )
                    else 6000
                )
            ),
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
    deployment = load_deployment(
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
            "schema_version": "semantic_integration_fingerprint/v2",
            "adapter_contract": ADAPTER_CONTRACT,
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
    if task_binding is not None:
        from ._task_spec import binding_for_graph, write_evidence

        binding = binding_for_graph(task_template, selected_graph)
        reference = write_evidence(root / "task_spec_binding.json", binding)
        fingerprint_payload = json.loads(paths.integration_fingerprint.read_text())
        fingerprint_payload["schema_version"] = "semantic_integration_fingerprint/v3"
        fingerprint_payload["task_spec_binding"] = reference
        _write_json(paths.integration_fingerprint, fingerprint_payload)
    return selected_graph, paths


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
        displacement[2] = max(0.05, min(0.10, 0.5 * (top - bottom) + 0.03))
        call["arguments"]["world_displacement"] = displacement
    return validate_semantic_task_graph(result)


def _task_stability_payload(
    graph: SemanticTaskGraph,
    scene: Any,
    embodiment: dict[str, Any],
) -> dict[str, Any]:
    """Declare task conditions without adding public validator or runtime types."""
    objects = {str(item["runtime_uid"]): item for item in scene.planner_objects}
    upright = {
        str(node["call"]["arguments"]["object"])
        for node in graph["nodes"]
        if node["call"]["kind"] == "registered"
        and (
            node["task_type"] == "E2" or node["call"]["call_id"] == _ALIGN_HELD_CALL_ID
        )
        and "object" in node["call"]["arguments"]
    }
    oriented_groups = {
        node["task_instance_id"]
        for node in graph["nodes"]
        if node["task_type"] in {"E1", "E4"}
        and node["call"].get("call_id") == _ALIGN_HELD_CALL_ID
        and node["call"]["arguments"].get("target") == "current_object_pose"
    }
    terminal_nodes = {group["node_ids"][-1] for group in graph["task_groups"]}
    routes = {
        (r["object_id"], r["reference_entity_id"], r["relation"]): r
        for r in _relative_place_route_payloads(graph, scene, settled=True)
    }
    motion_parts = {
        resource["resource_id"]: next(
            endpoint["control_part"]
            for endpoint in resource["endpoints"]
            if endpoint["endpoint_id"] == "motion"
        )
        for resource in embodiment["skill_profile"]["resources"]
    }
    presets: dict[str, Any] = {}
    for node in graph["nodes"]:
        call = node["call"]
        if call["kind"] == "place" and node.get("task_instance_id") in oriented_groups:
            object_id = str(call["object"])
            presets[f"gen_sim.{node['id']}.stable"] = {
                "kind": "upright",
                "entity": object_id,
                "local_axis": _longest_local_axis(objects[object_id]),
            }
        if call["kind"] != "registered":
            continue
        arguments = call["arguments"]
        if call["call_id"] in {_PLACE_RELATIVE_CALL_ID, _STACK_PLACE_CALL_ID}:
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
                route = routes[(object_id, reference_id, arguments["relation"])]
                presets[f"gen_sim.{node['id']}.stable"] = {
                    "kind": "upright",
                    "entity": object_id,
                    "local_axis": _longest_local_axis(objects[object_id]),
                    "reference": reference_id,
                    "displacement": route["world_displacement"],
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
                    "local_axis": _longest_local_axis(objects[object_id]),
                    "reference_axis": _longest_local_axis(objects[reference_id]),
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
        elif (
            call["call_id"] == _ALIGN_HELD_CALL_ID
            and node["task_type"] == "E4"
            and node["id"] in terminal_nodes
        ):
            object_id = str(arguments["object"])
            presets[f"gen_sim.{node['id']}.stable"] = {
                "kind": "hold",
                "entity": object_id,
                "local_axis": _longest_local_axis(objects[object_id]),
                "motion_parts": [motion_parts[call["resources"]["primary"]]],
            }
        elif call["call_id"] in {
            _COORDINATED_HOLD_CALL_ID,
            _COORDINATED_TRANSPORT_CALL_ID,
        }:
            object_id = str(arguments["object"])
            position = list(_position(objects[object_id]))
            support = objects[object_id].get("attributes", {}).get("final_support", {})
            if support.get("parent_uid") == "table" and scene.table_top_z is not None:
                # The skill offsets the live resting pose, not the import hover gap.
                bottom, _ = _vertical_mesh_bounds(
                    objects[object_id], axis_aligned=False
                )
                position[2] = float(scene.table_top_z) - bottom
            displacement = arguments["world_displacement"]
            holding = call["call_id"] == _COORDINATED_HOLD_CALL_ID
            presets[f"gen_sim.{node['id']}.stable"] = {
                "kind": "hold" if holding else "placement",
                "entity": object_id,
                "target_position": [
                    position[i] + float(displacement[i]) for i in range(3)
                ],
                "motion_parts": (
                    [
                        motion_parts[call["resources"][slot]]
                        for slot in ("left", "right")
                    ]
                    if holding
                    else []
                ),
                "position_tolerance": 0.02 if holding else _RELATIVE_POSITION_TOLERANCE,
            }
    return {"schema_version": "gen_sim_task_constraints/v1", "presets": presets}


def _program_payload(
    graph: SemanticTaskGraph,
    program_id: str,
    scene: Any | None = None,
    *,
    stability_presets: set[str] | None = None,
) -> dict[str, Any]:
    axis_by_object = (
        {}
        if scene is None
        else {
            str(item["runtime_uid"]): _longest_local_axis(item)
            for item in scene.planner_objects
        }
    )
    relative_routes = (
        {}
        if scene is None
        else {
            (route["object_id"], route["reference_entity_id"], route["relation"]): route
            for route in _relative_place_route_payloads(graph, scene, settled=True)
        }
    )
    items = [
        _program_node(
            node,
            relative_routes=relative_routes,
            axis_by_object=axis_by_object,
            task_stability=f"gen_sim.{node['id']}.stable"
            in (stability_presets or set()),
        )
        for node in graph["nodes"]
    ]
    by_name = {item["name"]: item for item in items}
    for group in graph["task_groups"]:
        terminal = by_name[group["node_ids"][-1]]
        for node_id in reversed(group["node_ids"]):
            accepted = by_name[node_id]
            policies = [
                policy
                for policy in accepted.get("post", ())
                if policy["preset"].startswith("gen_sim.")
            ]
            if not policies:
                policies = list(accepted.get("post", ()))
            validators = accepted.get("validators", ())
            if not policies and not validators:
                continue
            if accepted is not terminal:
                # Cleanup is still physical work; it can invalidate placement.
                terminal.setdefault("post", []).extend(deepcopy(policies))
                terminal.setdefault("validators", []).extend(deepcopy(validators))
            break
    return {
        "program_id": program_id,
        "targets": deepcopy(graph["targets"]),
        "program": {"kind": "sequence", "items": items},
    }


def _program_node(
    node: dict[str, Any],
    *,
    relative_routes: dict[tuple[str, str, str], dict[str, Any]] | None = None,
    axis_by_object: dict[str, list[float]] | None = None,
    task_stability: bool = False,
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
        _STACK_PLACE_CALL_ID,
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
    if call["kind"] == "registered" and call["call_id"] in {
        _PLACE_RELATIVE_CALL_ID,
        _STACK_PLACE_CALL_ID,
    }:
        arguments = call["arguments"]
        selector = (
            str(arguments["object"]),
            str(arguments["reference"]),
            str(arguments["relation"]),
        )
        try:
            route = (relative_routes or {})[selector]
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
    if (
        not task_stability
        and node.get("task_type") == "E2"
        and (
            call["kind"] == "place"
            or (
                call["kind"] == "registered"
                and call["call_id"] == _PLACE_RELATIVE_CALL_ID
            )
        )
    ):
        raise ValueError("E2 release requires a task-owned upright stability preset.")
    if node.get("task_type") == "E3" and call["kind"] == "place":
        target = call.get("at")
        if type(target) is not dict or set(target) != {"kind", "target"}:
            raise ValueError("Generated E3 Place requires one exact return target.")
        segment["validators"] = [
            {
                "kind": "object_near_target",
                "object": str(call["object"]),
                "target": str(target["target"]),
                "position_tolerance": 0.06,
            }
        ]
    if task_stability:
        arguments = call.get("arguments", {})
        entity = call.get("object", arguments.get("object"))
        segment.setdefault("post", []).append(
            {
                "kind": "wait_stable",
                "entity": entity,
                "preset": f"gen_sim.{node['id']}.stable",
            }
        )
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
    on_routes: list[tuple[str, str, str]] = []
    coordinated_routes: list[tuple[str, str, tuple[float, float, float]]] = []
    coordinated_hold_routes: list[tuple[str, str, tuple[float, float, float]]] = []
    move_held_routes: list[dict[str, Any]] = []
    upright_move_objects: set[str] = set()
    pick_routes: dict[str, list[dict[str, Any]]] = {}
    pick_options: dict[str, dict[str, Any]] = {}
    axis_align_objects: set[str] = set()
    pour_objects: set[str] = set()
    relative_lowerer_routes = _relative_place_route_payloads(graph, scene)
    has_relative_place = False
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
        if call["kind"] == "place" and "on" in call:
            affordance = str(call["on"])
            parts = affordance.split("__")
            if len(parts) != 3 or parts[0] != "on":
                raise ValueError(f"Unsupported generated on affordance {affordance!r}.")
            support_id, object_id = parts[1], parts[2]
            referenced_objects.add(support_id)
            on_routes.append((affordance, support_id, object_id))
        if call["kind"] == "registered" and call["call_id"] in {
            "simulation.coordinated_transport",
            _COORDINATED_HOLD_CALL_ID,
        }:
            arguments = call["arguments"]
            object_id = str(arguments["object"])
            referenced_objects.add(object_id)
            displacement = tuple(
                float(value)
                for value in arguments.get(
                    "world_displacement",
                    (-_DUAL_FRANKA_COORDINATED_TRANSPORT_DISTANCE, 0.0, 0.0),
                )
            )
            route = (object_id, str(arguments["target"]), displacement)
            if call["call_id"] == _COORDINATED_HOLD_CALL_ID:
                coordinated_hold_routes.append(route)
            else:
                coordinated_routes.append(route)
        elif call["kind"] == "registered" and (
            call["call_id"] == _PICK_CALL_ID
            or call["call_id"].startswith("gen_sim.pick.")
        ):
            arguments = call["arguments"]
            object_id = str(arguments["object"])
            final_target_id = str(arguments["target"])
            final_pose = _single_target_pose(graph, final_target_id)
            table_top = scene.table_top_z
            if table_top is None or not math.isfinite(float(table_top)):
                raise ValueError(
                    "Upright pickup requires the scene's measured table_top_z."
                )
            referenced_objects.add(object_id)
            referenced_objects.add("table")
            call_id = call["call_id"]
            source_axis = _longest_local_axis(scene_objects[object_id])
            if object_id in upright_move_objects | axis_align_objects:
                world_axis = np.array([0.0, 0.0, 1.0])
            else:
                from scipy.spatial.transform import Rotation

                world_axis = Rotation.from_euler(
                    "XYZ",
                    scene_objects[object_id].get("init_rot", [0.0, 0.0, 0.0]),
                    degrees=True,
                ).apply(source_axis)
            approach = np.array([0.0, 0.0, -1.0]) - world_axis
            length = np.linalg.norm(approach)
            approach = (
                np.array([0.0, 0.0, -1.0]) if length <= 1e-6 else approach / length
            )
            options = {
                "kind": "pick_up",
                "pre_grasp_distance": 0.08,
                "grasp_settle_steps": 16,
                "pick_object_part": "center",
                "approach_direction": approach.tolist(),
            }
            if call_id in pick_options and pick_options[call_id] != options:
                raise ValueError(
                    "Distinct Pick policies require task-scoped aliases; regenerate the bundle."
                )
            pick_options[call_id] = options
            pick_routes.setdefault(call_id, []).append(
                {
                    "object_id": object_id,
                    "target_id": final_target_id,
                    # E2 permits all declared world-yaw alternatives. Requiring
                    # the nominal yaw here would reject feasible upright grasps.
                    # Transport checks live reachability; yaw does not change
                    # the fingertip heights used by release-clearance screening.
                    "release_clearance_object_pose": {"kind": "pose", **final_pose},
                    "release_clearance_plane_z": float(table_top),
                    "release_clearance_safety_margin": _E2_RELEASE_SAFETY_MARGIN,
                    "grasp_region": "upper_half",
                }
            )
        elif call["kind"] == "registered" and call["call_id"] == _AXIS_ALIGN_CALL_ID:
            object_id = str(call["arguments"]["object"])
            referenced_objects.add(object_id)
            axis_align_objects.add(object_id)
        elif call["kind"] == "registered" and call["call_id"] in {
            _PLACE_RELATIVE_CALL_ID,
            _STACK_PLACE_CALL_ID,
        }:
            arguments = call["arguments"]
            referenced_objects.add(str(arguments["object"]))
            referenced_objects.add(str(arguments["reference"]))
            has_relative_place = True
        elif (
            call["kind"] == "registered"
            and call["call_id"] == _MOVE_HELD_OBJECT_CALL_ID
        ):
            arguments = call["arguments"]
            object_id = str(arguments["object"])
            target_id = str(arguments["target"])
            referenced_objects.add(object_id)
            if node["task_type"] == "E2":
                upright_move_objects.add(object_id)
                pose = _single_target_pose(graph, target_id)
                move_held_routes.append(
                    {
                        "object_id": object_id,
                        "target_id": target_id,
                        "pose": {"kind": "pose", **pose},
                    }
                )
            else:
                reference = str(arguments["reference"])
                referenced_objects.add(reference)
                move_held_routes.append(
                    {
                        "object_id": object_id,
                        "target_id": target_id,
                        "pose": {
                            "kind": "scene_entity",
                            "entity_id": reference,
                            "relative_pose": _translation_pose(0.05, -0.10, 0.125),
                        },
                    }
                )
        elif call["kind"] == "registered" and call["call_id"] == _POUR_CALL_ID:
            object_id = str(call["arguments"]["object"])
            referenced_objects.add(object_id)
            pour_objects.add(object_id)
        elif call["kind"] == "registered" and call["call_id"] == _ALIGN_HELD_CALL_ID:
            referenced_objects.add(str(call["arguments"]["object"]))
            upright_move_objects.add(str(call["arguments"]["object"]))
        elif call["kind"] == "registered" and call["call_id"] in {
            _STACK_PICK_CALL_ID,
            _CLEAR_RELEASED_CALL_ID,
        }:
            referenced_objects.add(str(call["arguments"]["object"]))
        elif call["kind"] == "registered" and call["call_id"] == "simulation.park":
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
            grasp_affordance = {
                "entity_id": f"{entity_id}_grasp",
                "kind": "antipodal_grasp",
            }
            if entity_id in axis_align_objects | pour_objects | upright_move_objects:
                grasp_affordance["internal_axis"] = _longest_local_axis(source)
            affordances.append(grasp_affordance)
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
        for affordance_id, support_id, object_id in on_routes:
            if support_id != entity_id:
                continue
            child = scene_objects.get(object_id)
            if child is None:
                raise ValueError(
                    f"On relation references missing scene entity {object_id!r}."
                )
            affordances.append(
                {
                    "entity_id": affordance_id,
                    "kind": "support_surface",
                    "native_name": affordance_id,
                    "object_target_pose": _support_target_pose(
                        source,
                        child,
                        axis_aligned=entity_id
                        in (axis_align_objects | upright_move_objects),
                    ),
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

    def coordinated_lowerer_routes(
        routes: list[tuple[str, str, tuple[float, float, float]]],
    ) -> list[dict[str, Any]]:
        return [
            {
                "object_id": object_id,
                "target_id": target_id,
                "world_displacement": list(displacement),
            }
            for object_id, target_id, displacement in routes
        ]

    lowerer_routes = coordinated_lowerer_routes(coordinated_routes)
    hold_lowerer_routes = coordinated_lowerer_routes(coordinated_hold_routes)
    if len(pour_objects) > 1:
        raise ValueError("One generated bundle currently supports one Pour object.")
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
            "articulations": [],
            "links": [],
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
                    "pick_object_part": "center",
                    "pre_grasp_distance": 0.15,
                    "lift_height": 0.16,
                    "approach_alignment_max_angle": 0.10,
                    "hand_interp_steps": 5,
                    "grasp_settle_steps": (
                        16
                        if any(
                            node["call"]["kind"] == "hand_over"
                            for node in graph["nodes"]
                        )
                        else 0
                    ),
                    "grasp_commit_fraction": 1.0,
                },
                "place": {
                    "kind": "place",
                    "hand_interp_steps": 12,
                    "release_settle_steps": 60,
                    "lift_height": 0.18,
                    "cartesian_waypoint_count": 2,
                    "preserve_current_object_orientation": True,
                },
                "hand_over": {
                    "kind": "hand_over",
                    "pre_grasp_distance": 0.15,
                    "lift_height": 0.0,
                    "hand_interp_steps": 16,
                    "hold_steps": 8,
                    "retreat_steps": 36,
                    "retreat_distance": 0.10,
                    "receive_pick_object_part": "bottom",
                    "release_at_target": False,
                    "arm_selection": "bound",
                },
                **pick_options,
                **(
                    {
                        _PLACE_RELATIVE_CALL_ID: {
                            "kind": "place",
                            "hand_interp_steps": 12,
                            "release_settle_steps": 60,
                            "lift_height": 0.10,
                            "max_approach_retract_z": (
                                float(scene.table_top_z)
                                + _DUAL_FRANKA_PLACE_TCP_CLEARANCE
                            ),
                            "cartesian_waypoint_count": 2,
                            "preserve_current_object_orientation": True,
                        }
                    }
                    if has_relative_place
                    else {}
                ),
                **(
                    {_MOVE_HELD_OBJECT_CALL_ID: {"kind": "move_held_object"}}
                    if move_held_routes
                    else {}
                ),
                **(
                    {_POUR_CALL_ID: {"kind": "pour", "rotate_angle": -1.0471975512}}
                    if pour_objects
                    else {}
                ),
                **(
                    {
                        _COORDINATED_HOLD_CALL_ID: {
                            "kind": "coordinated_pickment",
                            "object_motion_keyframes": 8,
                            "pre_grasp_distance": 0.10,
                            "lift_height": 0.08,
                            "hand_interp_steps": 10,
                            "hold_steps": 4,
                            "release": False,
                            "approach_direction": [0.0, 0.0, -1.0],
                            "left_to_right_arm_direction": [0.0, 1.0, 0.0],
                            "middle_empty_ratio": 0.4,
                        }
                    }
                    if hold_lowerer_routes
                    else {}
                ),
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
                    {"simulation.park": {"kind": "move_joints"}}
                    if has_park_call
                    else {}
                ),
                **(
                    {
                        "simulation.coordinated_transport": {
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
                    if lowerer_routes
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
                    "simulation.coordinated_transport",
                    _COORDINATED_HOLD_CALL_ID,
                    _AXIS_ALIGN_CALL_ID,
                    _MOVE_HELD_OBJECT_CALL_ID,
                    *pick_routes,
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
                    "final_position": [
                        0.0,
                        -0.08,
                        _handover_position_z(scene.table_top_z),
                    ],
                    "final_quaternion_wxyz": [
                        0.7071067812,
                        0.7071067812,
                        0.0,
                        0.0,
                    ],
                }
            ],
            "registered_semantic_lowerers": [
                *[
                    {"kind": "pick", "call_id": call_id, "routes": routes}
                    for call_id, routes in pick_routes.items()
                ],
                *(
                    [{"kind": "place_relative", "routes": relative_lowerer_routes}]
                    if has_relative_place
                    else []
                ),
                *(
                    [
                        {
                            "kind": "move_held_object",
                            "routes": move_held_routes,
                            "phase_protection": "held_object_v1",
                        }
                    ]
                    if move_held_routes
                    else []
                ),
                *(
                    [
                        {
                            "kind": "pour",
                            "object_id": next(iter(pour_objects)),
                        }
                    ]
                    if pour_objects
                    else []
                ),
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
                    [{"kind": "coordinated_hold", "routes": hold_lowerer_routes}]
                    if hold_lowerer_routes
                    else []
                ),
                *(
                    [{"kind": "coordinated_transport", "routes": lowerer_routes}]
                    if lowerer_routes
                    else []
                ),
            ],
        },
    }


def _scene_payload(scene: Any, *, program_id: str) -> dict[str, Any]:
    articulations = [deepcopy(value) for value in scene.articulations]
    for articulation in articulations:
        for semantic_only_key in (
            "affordances",
            "attributes",
            "category",
            "description",
            "initial_state",
            "is_articulated",
            "name",
            "proxy_body_scale",
            "proxy_glb_fpath",
        ):
            articulation.pop(semantic_only_key, None)
        suffix = Path(str(articulation.get("fpath", ""))).suffix.lower()
        if suffix in {".usd", ".usda", ".usdc"}:
            # pytorch-kinematics accepts URDF XML only.  These task skills use
            # DexSim's native link poses/geometry and do not need a PK chain.
            articulation["build_pk_chain"] = False
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
            "articulation": articulations,
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
    *,
    settled: bool = False,
) -> list[dict[str, Any]]:
    """Project release targets or their expected post-settle support positions."""
    axis_align_objects: set[str] = set()
    selectors: set[tuple[str, str, str]] = set()
    stack_selectors: set[tuple[str, str, str]] = set()
    upright_targets: dict[tuple[str, str, str], str] = {}
    for node in graph["nodes"]:
        call = node["call"]
        if call["kind"] != "registered":
            continue
        if call["call_id"] in {_AXIS_ALIGN_CALL_ID, _ALIGN_HELD_CALL_ID} or (
            node.get("task_type") == "E2"
            and call["call_id"] == _MOVE_HELD_OBJECT_CALL_ID
        ):
            axis_align_objects.add(str(call["arguments"]["object"]))
        elif call["call_id"] in {_PLACE_RELATIVE_CALL_ID, _STACK_PLACE_CALL_ID}:
            arguments = call["arguments"]
            if call["call_id"] == _STACK_PLACE_CALL_ID:
                stack_selectors.add(
                    (
                        str(arguments["object"]),
                        str(arguments["reference"]),
                        str(arguments["relation"]),
                    )
                )
            if node.get("task_type") == "E2":
                upright_targets[
                    (
                        str(arguments["object"]),
                        str(arguments["reference"]),
                        str(arguments["relation"]),
                    )
                ] = f"{node['task_instance_id']}_upright_target"
            selectors.add(
                (
                    str(arguments["object"]),
                    str(arguments["reference"]),
                    str(arguments["relation"]),
                )
            )
    scene_objects = {str(item["runtime_uid"]): item for item in scene.planner_objects}
    routes: list[dict[str, Any]] = []
    for selector in sorted(selectors):
        object_id, reference_id, relation = selector
        if selector in upright_targets:
            target = _single_target_pose(graph, upright_targets[selector])
            reference_position = _position(scene_objects[reference_id])
            displacement = [
                float(target["position"][index]) - reference_position[index]
                for index in range(3)
            ]
            if settled:
                # Release clearance is a command target, not a resting height.
                displacement[2] -= _upright_release_clearance(scene_objects[object_id])
        else:
            displacement = _relative_world_displacement(
                relation,
                object_id=object_id,
                reference_id=reference_id,
                scene_objects=scene_objects,
                axis_align_objects=axis_align_objects,
                table_top_z=scene.table_top_z,
            )
            if settled and selector in stack_selectors:
                displacement[2] -= _PLACEMENT_CLEARANCE
        routes.append(
            {
                "object_id": object_id,
                "reference_entity_id": reference_id,
                "relation": relation,
                "world_displacement": displacement,
            }
        )
    return routes


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
        or not isinstance(pose.get("quaternion_wxyz"), list)
        or len(pose["quaternion_wxyz"]) != 4
    ):
        raise ValueError(f"Generated target {target_id!r} has an invalid pose.")
    return pose


def _longest_local_axis(source: dict[str, Any]) -> list[float]:
    """Return the longest axis in the simulator rigid-body local frame."""
    source_axis = _longest_source_axis(source)
    shape = source.get("shape", {})
    path = shape.get("fpath") if isinstance(shape, dict) else None
    if path and Path(str(path)).suffix.lower() in {".glb", ".gltf"}:
        # glTF is Y-up. DexSim converts it to Z-up while importing the render
        # mesh, so semantic axes must follow the same source-to-body mapping.
        x_axis, y_axis, z_axis = source_axis
        return [x_axis, -z_axis, y_axis]
    return source_axis


def _longest_source_axis(source: dict[str, Any]) -> list[float]:
    """Return the longest axis in the source mesh coordinate frame."""
    extents = _local_extents(source)
    if extents is None:
        return [0.0, 0.0, 1.0]
    axis = [0.0, 0.0, 0.0]
    axis[max(range(3), key=extents.__getitem__)] = 1.0
    return axis


def _local_extents(source: dict[str, Any]) -> list[float] | None:
    """Return finite positive local geometry extents when available."""
    bounds = _local_bounds(source)
    if bounds is None:
        return None
    return [maximum - minimum for minimum, maximum in zip(*bounds, strict=True)]


def _local_bounds(source: dict[str, Any]) -> tuple[list[float], list[float]] | None:
    """Return finite local geometry bounds when available."""
    shape = source.get("shape", {})
    if not isinstance(shape, dict):
        return None
    if shape.get("shape_type") == "Cube":
        size = shape.get("size")
        if isinstance(size, list) and len(size) == 3:
            extents = [float(value) for value in size]
            if min(extents) <= 0.0:
                return None
            return (
                [-value / 2.0 for value in extents],
                [value / 2.0 for value in extents],
            )
    path = shape.get("fpath")
    if not path:
        return None
    try:
        import trimesh

        geometry = trimesh.load(str(path), force="scene").to_geometry()
        bounds = [[float(value) for value in row] for row in geometry.bounds]
    except (OSError, TypeError, ValueError):
        return None
    if (
        len(bounds) != 2
        or any(len(row) != 3 for row in bounds)
        or any(maximum <= minimum for minimum, maximum in zip(*bounds, strict=True))
    ):
        return None
    return bounds[0], bounds[1]


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


def _refine_upright_targets(
    graph: SemanticTaskGraph,
    scene: Any,
) -> SemanticTaskGraph:
    """Use normalized mesh origins to place E2 objects on the measured table."""
    selected = deepcopy(graph)
    objects = {str(item["runtime_uid"]): item for item in scene.planner_objects}
    table = objects.get("table")
    if table is None:
        return selected
    if scene.table_top_z is None:
        return selected
    table_top = float(scene.table_top_z)
    target_routes: dict[str, tuple[str, str]] = {}
    for node in selected["nodes"]:
        call = node["call"]
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
        # Place owns the final descent; its release target still needs the
        # same mesh-origin refinement when no separate final Move is emitted.
        target_routes[f"{node['task_instance_id']}_upright_target"] = (
            object_id,
            resource,
        )
    for target_id, (object_id, resource) in target_routes.items():
        source = objects.get(object_id)
        target = selected["targets"].get(target_id) if target_id is not None else None
        bounds = _local_bounds(source) if source is not None else None
        if bounds is None or not isinstance(target, dict):
            continue
        source_axis = _longest_source_axis(source)
        axis = _longest_local_axis(source)
        axis_index = source_axis.index(1.0)
        local_minimum = bounds[0][axis_index]
        values = target.get("values")
        if not isinstance(values, list) or len(values) != 1:
            continue
        position = values[0].get("position")
        if not isinstance(position, list) or len(position) != 3:
            continue
        clearance = _upright_release_clearance(source)
        position[2] = table_top + clearance - local_minimum
        if target_id.endswith("_upright_staging_target"):
            position[2] += _UPRIGHT_STAGING_CLEARANCE
        values[0]["quaternion_wxyz"] = _upright_target_quaternion(
            source,
            axis,
            world_yaw=(math.pi if resource == "right" else 0.0),
        )
    return validate_semantic_task_graph(selected)


def _upright_target_quaternion(
    source: dict[str, Any],
    local_axis: list[float],
    *,
    world_yaw: float,
) -> list[float]:
    """Rotate the current object frame so its selected positive axis is world +Z."""
    import torch

    from embodichain.utils.math import (
        axis_angle_to_rotation_matrix,
        matrix_from_quat,
        quat_from_euler_xyz,
        quat_from_matrix,
    )

    init_rot = source.get("init_rot")
    if not isinstance(init_rot, list) or len(init_rot) != 3:
        raise ValueError(
            f"E2 object {source.get('runtime_uid')!r} requires three-value init_rot."
        )
    angles = torch.deg2rad(torch.tensor(init_rot, dtype=torch.float32))
    quaternion = quat_from_euler_xyz(angles[0], angles[1], angles[2])
    rotation = matrix_from_quat(quaternion)
    axis = torch.tensor(local_axis, dtype=torch.float32)
    source_axis = torch.nn.functional.normalize(rotation @ axis, dim=0)
    target_axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
    cross = torch.linalg.cross(source_axis, target_axis)
    dot = torch.clamp(torch.dot(source_axis, target_axis), -1.0, 1.0)
    cross_norm = torch.linalg.vector_norm(cross)
    if float(cross_norm) <= 1.0e-6:
        if float(dot) >= 0.0:
            delta = torch.eye(3, dtype=torch.float32)
        else:
            basis = torch.eye(3, dtype=torch.float32)[
                torch.argmin(torch.abs(source_axis))
            ]
            perpendicular = torch.nn.functional.normalize(
                torch.linalg.cross(source_axis, basis), dim=0
            )
            delta = axis_angle_to_rotation_matrix(perpendicular * math.pi)
    else:
        rotation_axis = cross / cross_norm
        angle = torch.atan2(cross_norm, dot)
        delta = axis_angle_to_rotation_matrix(rotation_axis * angle)
    target_rotation = delta @ rotation
    if world_yaw:
        target_rotation = (
            axis_angle_to_rotation_matrix(
                torch.tensor([0.0, 0.0, world_yaw], dtype=torch.float32)
            )
            @ target_rotation
        )
    target_quaternion = quat_from_matrix(target_rotation)
    return [float(value) for value in target_quaternion]


def _support_target_pose(
    support: dict[str, Any],
    child: dict[str, Any],
    *,
    axis_aligned: bool,
) -> list[float]:
    """Build a conservative object-center target above a support entity."""
    support_extents = _local_extents(support) or [0.1, 0.1, 0.1]
    child_extents = _local_extents(child) or [0.05, 0.05, 0.05]
    child_half_height = max(child_extents) / 2.0
    translation = [0.0, 0.0, 0.0]
    if axis_aligned:
        normal = _longest_local_axis(support)
        support_bounds = _local_bounds(support)
        source_axis = _longest_source_axis(support)
        source_axis_index = source_axis.index(1.0)
        support_top = (
            support_extents[source_axis_index] / 2.0
            if support_bounds is None
            else support_bounds[1][source_axis_index]
        )
        distance = support_top + child_half_height + 0.01
        translation = [component * distance for component in normal]
    else:
        attributes = support.get("attributes", {})
        aabb = (
            attributes.get("final_world_aabb") if isinstance(attributes, dict) else None
        )
        position = support.get("init_pos", [0.0, 0.0, 0.0])
        if isinstance(aabb, dict) and isinstance(aabb.get("max"), list):
            surface_height = float(aabb["max"][2]) - float(position[2])
        else:
            surface_height = support_extents[2] / 2.0
        translation[2] = surface_height + child_half_height + 0.01
    return _translation_pose(*translation)
