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
from pathlib import Path
import shutil
from typing import Any, Final

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
    shutil.copy2(embodiment_source, paths.embodiment)
    shutil.copy2(policy_source, paths.execution_policy)

    program_id = _program_identifier(selected_graph["task_id"])
    scene_contract = f"{program_id}_scene_v1"
    save_config(paths.program, _program_payload(selected_graph, program_id))
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
                            "max_steps": 120,
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


def _program_payload(graph: SemanticTaskGraph, program_id: str) -> dict[str, Any]:
    return {
        "program_id": program_id,
        "targets": deepcopy(graph["targets"]),
        "program": {
            "kind": "sequence",
            "items": [_program_node(node) for node in graph["nodes"]],
        },
    }


def _program_node(node: dict[str, Any]) -> dict[str, Any]:
    """Materialize one task node as one canonical runtime segment."""
    call = deepcopy(node["call"])
    segment: dict[str, Any] = {
        "kind": "segment",
        "name": str(node["id"]),
        "steps": {"kind": "invoke", "call": call},
    }
    settle_entities: list[str] = []
    if call["kind"] == "place":
        inside = call.get("inside")
        if inside is not None:
            parts = str(inside).split("__")
            if len(parts) != 3 or parts[0] != "inside":
                raise ValueError(f"Unsupported generated inside affordance {inside!r}.")
        settle_entities.append(str(call["object"]))
    elif (
        call["kind"] == "registered"
        and call["call_id"] == "simulation.coordinated_transport"
    ):
        settle_entities.append(str(call["arguments"]["object"]))
    if settle_entities:
        segment["post"] = [
            {
                "kind": "wait_stable",
                "entity": settle_entity,
                "preset": (
                    "contained_rigid_object"
                    if call["kind"] == "place" and call.get("inside") is not None
                    else "rigid_object"
                ),
            }
            for settle_entity in dict.fromkeys(settle_entities)
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
            and call["call_id"] == "simulation.coordinated_transport"
        ):
            arguments = call["arguments"]
            object_id = str(arguments["object"])
            referenced_objects.add(object_id)
            coordinated_routes.append((object_id, str(arguments["target"])))
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
            affordances.append(
                {
                    "entity_id": f"{entity_id}_grasp",
                    "kind": "antipodal_grasp",
                }
            )
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

    lowerer_routes = []
    for object_id, target_id in coordinated_routes:
        lowerer_routes.append(
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
                    "lift_height": 0.08,
                    "hand_interp_steps": 10,
                    "hold_steps": 4,
                    "retreat_steps": 28,
                    "retreat_distance": 0.12,
                    "receive_pick_object_part": "bottom",
                    "release_at_target": False,
                    "arm_selection": "bound",
                },
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
                    "final_position": [0.0, -0.08, 0.866],
                    "final_quaternion_wxyz": [
                        0.7071067812,
                        0.7071067812,
                        0.0,
                        0.0,
                    ],
                }
            ],
            "registered_semantic_lowerers": [
                *([{"kind": "park"}] if has_park_call else []),
                *(
                    [{"kind": "coordinated_transport", "routes": lowerer_routes}]
                    if lowerer_routes
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
