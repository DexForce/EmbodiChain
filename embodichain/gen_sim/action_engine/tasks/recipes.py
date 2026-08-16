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

"""Direct AtomicAction recipes for E1-E9 TaskSpec instances."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from embodichain.gen_sim.action_engine.capabilities import (
    AtomicCapabilityRegistry,
    build_atomic_capability_registry,
    capability_precondition,
)
from embodichain.gen_sim.action_engine.domain import (
    TERMINAL_BEHAVIORS,
    TRANSPORT_DIRECTIONS,
    motion_policy,
    task_success_type,
    validate_task_spec,
)
from embodichain.gen_sim.action_engine.planning.linker import (
    link_seed_graph,
    link_task_dependencies,
)
from embodichain.gen_sim.action_engine.protocol import SEED_GRAPH_SCHEMA

__all__ = ["instantiate_seed_graph"]


def instantiate_seed_graph(
    task_spec: Mapping[str, Any],
    role_bindings: Mapping[str, str],
    *,
    planner_route: str = "offline",
    registry: AtomicCapabilityRegistry | None = None,
) -> dict[str, Any]:
    """Instantiate a coordinate-free SeedGraph after Scene Engine hand-off."""
    task = validate_task_spec(task_spec)
    bindings = _validate_bindings(task, role_bindings)
    capabilities = registry or build_atomic_capability_registry()
    task, payload_links = _propagate_direct_payloads(task, bindings)
    task = link_task_dependencies(task, bindings, registry=capabilities)
    instances = _topological_instances(task["task_instances"])
    nodes: list[dict[str, Any]] = []
    groups = []
    terminal_by_group: dict[str, list[str]] = {}
    held_after_group: dict[str, tuple[str, str] | None] = {}
    for instance in instances:
        group_id = str(instance["id"])
        task_type = str(instance["task_type"])
        params = _resolve_params(instance["params"], bindings)
        object_uid = _primary_object(task_type, params)
        incoming_held_arm = _incoming_held_arm(
            task_type,
            object_uid,
            instance["depends_on"],
            held_after_group,
        )
        actor = _actor(task_type, params, incoming_held_arm=incoming_held_arm)
        dependency_nodes = [
            node_id
            for dependency in instance["depends_on"]
            for node_id in terminal_by_group[str(dependency)]
        ]
        recipe_nodes, operator, goal, success = _recipe(
            group_id,
            task_type,
            object_uid,
            actor,
            params,
            dependency_nodes,
            role=str(instance["role"]),
            incoming_held_arm=incoming_held_arm,
        )
        for node in recipe_nodes:
            node["precondition"] = capability_precondition(
                capabilities.get(str(node["atomic_action"])),
                object_uid=str(node["object_uid"]),
                actor=node["actor"],
                target_binding=node["target_binding"],
            )
        nodes.extend(recipe_nodes)
        terminal_by_group[group_id] = _terminal_nodes(recipe_nodes)
        held_after_group[group_id] = _terminal_hold(
            task_type,
            object_uid,
            params,
        )
        groups.append(
            {
                "id": group_id,
                "task_type": task_type,
                "role": str(instance["role"]),
                "operator": operator,
                "object_uid": object_uid,
                "actor": actor,
                "goal": goal,
                "depends_on": list(instance["depends_on"]),
                "parent_task_instance_id": str(
                    params.get("parent_task_instance_id", group_id)
                ),
                "node_ids": [node["id"] for node in recipe_nodes],
                "success": success,
            }
        )

    graph_metadata = {
        "task_spec_id": task["task_id"],
        "role_bindings": dict(sorted(bindings.items())),
        "allocation_groups": deepcopy(
            task.get("metadata", {}).get("allocation_groups", [])
        ),
        "direct_payload_links": payload_links,
        "oracle_exposed": False,
        "planning_latency_seconds": 0.0,
        "vlm_call_count": 0,
    }
    task_linker = task.get("metadata", {}).get("action_contract_task_linker")
    if isinstance(task_linker, Mapping):
        graph_metadata["action_contract_task_linker"] = deepcopy(dict(task_linker))

    graph = {
        "schema_version": SEED_GRAPH_SCHEMA,
        "task_id": task["task_id"],
        "instruction": task["instruction"],
        "level": task["level"],
        "reasoning_type": task["reasoning_type"],
        "planner_route": planner_route,
        "nodes": nodes,
        "task_groups": groups,
        "success": {
            "op": "all",
            "terms": [deepcopy(group["success"]) for group in groups],
        },
        "capability_catalog_hash": capabilities.catalog_hash(),
        "metadata": graph_metadata,
    }
    known_objects = set(bindings.values()) | {"table"}
    graph = link_seed_graph(
        graph,
        registry=capabilities,
        task_order=[str(instance["id"]) for instance in instances],
        known_objects=known_objects,
    )
    for node in graph["nodes"]:
        capabilities.validate_binding(node)
    return graph


def _topological_instances(
    instances: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Emit task groups in dependency order even for externally authored specs."""
    by_id = {str(instance["id"]): dict(instance) for instance in instances}
    original = [str(instance["id"]) for instance in instances]
    pending = set(by_id)
    ordered: list[dict[str, Any]] = []
    while pending:
        ready = [
            instance_id
            for instance_id in original
            if instance_id in pending
            and all(
                str(dependency) not in pending
                for dependency in by_id[instance_id]["depends_on"]
            )
        ]
        if not ready:
            raise ValueError("TaskSpec task instances contain a dependency cycle.")
        for instance_id in ready:
            ordered.append(by_id[instance_id])
            pending.remove(instance_id)
    return ordered


def _propagate_direct_payloads(
    task: Mapping[str, Any],
    bindings: Mapping[str, str],
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Carry direct E1 support relations into a later single-arm E1 move.

    This is intentionally a one-hop physical relation rather than a general
    scene-state planner: an object placed on or inside a carrier becomes that
    carrier's direct payload until the object itself is manipulated again.
    """
    result = deepcopy(dict(task))
    role_by_uid = {uid: role for role, uid in bindings.items()}
    direct_by_carrier: dict[str, list[tuple[str, str, str]]] = {}
    carrier_by_payload: dict[str, str] = {}
    links: list[dict[str, str]] = []
    changed = False

    for instance in _topological_instances(result["task_instances"]):
        task_type = str(instance["task_type"])
        params = instance["params"]
        primary_key = "source_role" if task_type == "E3" else "object_role"
        primary_role = params.get(primary_key)
        if not isinstance(primary_role, str) or not primary_role:
            continue
        primary_uid = bindings.get(primary_role, primary_role)
        direct_payloads = list(direct_by_carrier.get(primary_uid, ()))
        if direct_payloads:
            if task_type != "E1":
                raise ValueError(
                    f"TaskGroup {instance['id']!r} moves carrier {primary_uid!r} "
                    "with direct payloads, but payload propagation currently "
                    "supports only single-arm E1 placement."
                )
            payload_roles = [payload_role for _, payload_role, _ in direct_payloads]
            if params.get("payload_roles") != payload_roles:
                params["payload_roles"] = payload_roles
                changed = True
            for payload_uid, _payload_role, producer_id in direct_payloads:
                links.append(
                    {
                        "producer": producer_id,
                        "consumer": str(instance["id"]),
                        "carrier": primary_uid,
                        "payload": payload_uid,
                        "relation": "direct_support",
                    }
                )

        if task_type in {"E1", "E2", "E3", "E4", "E5"}:
            old_carrier = carrier_by_payload.pop(primary_uid, None)
            if old_carrier is not None:
                direct_by_carrier[old_carrier] = [
                    item
                    for item in direct_by_carrier.get(old_carrier, ())
                    if item[0] != primary_uid
                ]

        if task_type != "E1" or str(params.get("relation")) not in {"on", "inside"}:
            continue
        target_role = params.get("target_role")
        if not isinstance(target_role, str) or not target_role:
            continue
        target_uid = bindings.get(target_role, target_role)
        if target_uid in {"table", "table_center"} or target_uid == primary_uid:
            continue
        payload_role = role_by_uid.get(primary_uid, primary_role)
        direct_by_carrier.setdefault(target_uid, []).append(
            (primary_uid, payload_role, str(instance["id"]))
        )
        carrier_by_payload[primary_uid] = target_uid

    if changed:
        metadata = dict(result.get("metadata", {}))
        metadata.pop("action_contract_task_linker", None)
        result["metadata"] = metadata
    return validate_task_spec(result), links


def _payload_goal(params: Mapping[str, Any], object_uid: str) -> list[dict[str, str]]:
    raw_payloads = params.get("payload_roles", [])
    if not isinstance(raw_payloads, Sequence) or isinstance(
        raw_payloads, (str, bytes, bytearray)
    ):
        raise ValueError("E1 payload_roles must be a list.")
    payloads = [str(value) for value in raw_payloads]
    if any(not value for value in payloads):
        raise ValueError("E1 payload_roles must contain non-empty object IDs.")
    if object_uid in payloads:
        raise ValueError("An E1 carrier cannot be its own payload.")
    if len(payloads) != len(set(payloads)):
        raise ValueError("E1 direct payload objects must be unique.")
    return [{"object": value, "slot": "center"} for value in payloads]


def _orientation_extensions(params: Mapping[str, Any]) -> dict[str, Any]:
    """Copy optional compiled-orientation fields from one task instance."""
    return {
        key: deepcopy(params[key])
        for key in ("orientation_constraint", "orientation_directed")
        if key in params
    }


def _recipe(
    group_id: str,
    task_type: str,
    object_uid: str,
    actor: Mapping[str, Any],
    params: Mapping[str, Any],
    dependencies: list[str],
    *,
    role: str,
    incoming_held_arm: str | None = None,
) -> tuple[list[dict[str, Any]], str, dict[str, Any], dict[str, Any]]:
    if task_type == "E1":
        target = str(params.get("target_role", "table"))
        relation = str(params.get("relation", "on"))
        layout = str(params.get("layout", ""))
        if layout == "line":
            goal = {
                "layout": "line",
                "objects": list(params["objects_roles"]),
                "axis": str(params.get("axis", "world_y")),
                "anchor": "table_center",
                "order_by": str(params.get("order_by", "explicit")),
                "order_direction": str(params.get("order_direction", "given")),
                "order_constraint": str(params.get("order_constraint", "free")),
                "participation": str(params.get("participation", "auto")),
                "orientation_goal": str(params.get("orientation_goal", "none")),
                "orientation_axis": str(params.get("orientation_axis", "none")),
                **_orientation_extensions(params),
                "nominal_slot_index": int(params["nominal_slot_index"]),
                "slot_constraint": str(
                    params.get("slot_constraint", "free_reassignable")
                ),
            }
            payloads = _payload_goal(params, object_uid)
            if payloads:
                goal["payloads"] = payloads
            success = {
                "type": "line_member_placed",
                "nominal_slot_index": goal["nominal_slot_index"],
                "slot_constraint": goal["slot_constraint"],
                "order_constraint": goal["order_constraint"],
            }
            return (
                _single_arm_manipulation(
                    group_id,
                    task_type,
                    object_uid,
                    actor,
                    dependencies,
                    role=role,
                    already_held=incoming_held_arm is not None,
                    payloads=payloads,
                ),
                "arrange_line",
                goal,
                success,
            )
        goal = {
            "reference_object": target,
            "reference_state": "live",
            "relation": relation,
            "relation_frame": str(params.get("relation_frame", "world")),
            "orientation_goal": str(params.get("orientation_goal", "none")),
            "orientation_axis": str(params.get("orientation_axis", "none")),
            **_orientation_extensions(params),
            "slot": str(params.get("slot", "auto")),
        }
        if "visual_constraint" in params:
            goal["visual_constraint"] = deepcopy(params["visual_constraint"])
        payloads = _payload_goal(params, object_uid)
        if payloads:
            goal["payloads"] = payloads
        success = {
            "type": "semantic_goal",
            "relation": relation,
            "reference_object": target,
        }
        return (
            _single_arm_manipulation(
                group_id,
                task_type,
                object_uid,
                actor,
                dependencies,
                role=role,
                already_held=incoming_held_arm is not None,
                payloads=payloads,
            ),
            "place_relative",
            goal,
            success,
        )
    if task_type == "E2":
        terminal_behavior = str(params.get("terminal_behavior", "place"))
        if terminal_behavior == "hold" and role != "recovery":
            raise ValueError(
                "Ordinary E2 groups must release their supported object at the "
                "TaskGroup boundary."
            )
        goal = {
            "relation": "none",
            "reference_state": "live",
            "orientation_goal": str(params.get("orientation_goal", "upright")),
            "orientation_axis": str(params.get("orientation_axis", "none")),
            "position_anchor": "initial_xy",
            "support_object": str(params.get("support_role", "table")),
            "upright_local_axis": str(params.get("upright_local_axis", "long_axis")),
            **_orientation_extensions(params),
        }
        if terminal_behavior == "hold":
            goal["terminal_behavior"] = "hold"
        success = {
            "type": task_success_type(task_type, params),
            "object": object_uid,
            "local_axis": goal["upright_local_axis"],
        }
        return (
            _single_arm_manipulation(
                group_id,
                task_type,
                object_uid,
                actor,
                dependencies,
                role=role,
                already_held=incoming_held_arm is not None,
                leave_held=str(params.get("terminal_behavior", "place")) == "hold",
            ),
            "orient_object",
            goal,
            success,
        )
    if task_type == "E3":
        target = str(params["target_role"])
        goal = {
            "reference_object": target,
            "relation": "above",
            "amount": "task_defined",
        }
        return (
            [
                _node(
                    group_id,
                    1,
                    "Pour",
                    task_type,
                    object_uid,
                    actor,
                    "arm",
                    {
                        "kind": "pour_goal",
                        "object": object_uid,
                        "reference_object": target,
                    },
                    dependencies,
                    role,
                    {
                        "type": "poured",
                        "object": object_uid,
                        "reference_object": target,
                    },
                    motion_policy(),
                )
            ],
            "pour",
            goal,
            {"type": "poured", "object": object_uid, "reference_object": target},
        )
    if task_type == "E4":
        transfer = str(params.get("transfer_arm", "left_arm"))
        receive = str(params.get("receive_arm", "right_arm"))
        if incoming_held_arm == "coordinated":
            raise ValueError(
                "E4 cannot consume a coordinated hold; an explicit single-arm "
                "handover state is required."
            )
        if incoming_held_arm is not None and transfer != incoming_held_arm:
            raise ValueError(
                f"E4 transfer_arm {transfer!r} conflicts with the predecessor "
                f"holder {incoming_held_arm!r}."
            )
        pickup_actor = {"mode": "required", "arm": transfer}
        pickup = None
        if incoming_held_arm is None:
            pickup = _node(
                group_id,
                1,
                "PickUp",
                task_type,
                object_uid,
                pickup_actor,
                "arm",
                {"kind": "object", "object": object_uid},
                dependencies,
                role,
                {"type": "object_held", "object": object_uid, "arm": transfer},
                motion_policy(("handover_role", "transfer")),
            )
        staging = _node(
            group_id,
            1 if pickup is None else 2,
            "MoveHeldObject",
            task_type,
            object_uid,
            pickup_actor,
            "arm",
            {
                "kind": "handover_staging",
                "object": object_uid,
                "transfer_arm": transfer,
                "receive_arm": receive,
            },
            dependencies if pickup is None else [pickup["id"]],
            role,
            {"type": "object_held", "object": object_uid, "arm": transfer},
            motion_policy(),
        )
        handover = _node(
            group_id,
            2 if pickup is None else 3,
            "HandOver",
            task_type,
            object_uid,
            {"mode": "coordinated", "arms": ["left_arm", "right_arm"]},
            "coordinated",
            {
                "kind": "handover_goal",
                "object": object_uid,
                "transfer_arm": transfer,
                "receive_arm": receive,
            },
            [staging["id"]],
            role,
            {"type": "handover_complete", "object": object_uid, "arm": receive},
            motion_policy(),
        )
        # Grounding configures HandOver as exchange-to-exchange, so its receiver
        # stays at the grasp while the transfer arm performs the built-in lift.
        # This ordered retreat/home suffix then verifies and completes clearance
        # before any receiver-side continuation may carry the object away.
        retreat = _node(
            group_id,
            3 if pickup is None else 4,
            "MoveEndEffector",
            task_type,
            object_uid,
            pickup_actor,
            "arm",
            {
                "kind": "policy_pose",
                "source": "handover",
                "operation": "retreat",
            },
            [handover["id"]],
            "cleanup",
            {},
            motion_policy(),
        )
        home = _node(
            group_id,
            4 if pickup is None else 5,
            "MoveJoints",
            task_type,
            object_uid,
            pickup_actor,
            "arm",
            {
                "kind": "joint_state",
                "source": "initial",
                "operation": "handover_home",
            },
            [retreat["id"]],
            "cleanup",
            {},
            motion_policy(),
        )
        return (
            [
                item
                for item in (pickup, staging, handover, retreat, home)
                if item is not None
            ],
            "handover",
            {
                "relation": "handover",
                "orientation_goal": str(params.get("orientation_goal", "none")),
                "orientation_axis": "none",
                **_orientation_extensions(params),
                "transfer_arm": transfer,
                "receive_arm": receive,
            },
            {"type": "handover_complete", "object": object_uid, "arm": receive},
        )
    if task_type == "E5":
        terminal_behavior = str(params.get("terminal_behavior", "hold"))
        if terminal_behavior not in TERMINAL_BEHAVIORS - {"none"}:
            raise ValueError("E5 terminal_behavior must be 'hold' or 'place'.")
        direction = str(params.get("direction", "up"))
        if direction not in TRANSPORT_DIRECTIONS:
            raise ValueError(f"E5 direction {direction!r} is unsupported.")
        goal = {
            "direction": direction,
            "terminal_behavior": terminal_behavior,
            "orientation_goal": str(params.get("orientation_goal", "none")),
            "orientation_axis": "none",
            **_orientation_extensions(params),
            "relation_frame": str(params.get("relation_frame", "robot")),
        }
        target = params.get("target_role")
        relation = str(params.get("relation", "none"))
        if isinstance(target, str) and target:
            if relation == "none":
                raise ValueError("E5 target_role requires a symbolic relation.")
            goal.update(
                {
                    "reference_object": target,
                    "reference_state": "live",
                    "relation": relation,
                    "direction": "none",
                }
            )
        elif direction == "none" and terminal_behavior != "place":
            raise ValueError("E5 requires a direction or target_role relation.")
        pick = _node(
            group_id,
            1,
            "CoordinatedPickment",
            task_type,
            object_uid,
            actor,
            "coordinated",
            {"kind": "coordinated_goal", "object": object_uid},
            dependencies,
            role,
            {"type": "held_by_both_grippers", "object": object_uid},
            motion_policy(),
        )
        nodes = [pick]
        if terminal_behavior == "place":
            release_sync_group = f"{group_id}__dual_release"
            for index, arm, release_role in (
                (2, "left_arm", "participant"),
                (3, "right_arm", "commit"),
            ):
                release = _node(
                    group_id,
                    index,
                    "MoveJoints",
                    task_type,
                    object_uid,
                    {"mode": "required", "arm": arm},
                    "hand",
                    {
                        "kind": "joint_state",
                        "source": "gripper_open",
                        "coordinated_release_role": release_role,
                    },
                    [pick["id"]],
                    role,
                    {},
                    motion_policy(),
                )
                release["sync_group"] = release_sync_group
                nodes.append(release)
        success_type = task_success_type(task_type, params)
        success = (
            {"type": success_type, "object": object_uid}
            if success_type == "held_by_both_grippers"
            else {
                "type": success_type,
                "relation": relation,
                **(
                    {"reference_object": target}
                    if isinstance(target, str) and target
                    else {}
                ),
            }
        )
        return (
            nodes,
            "coordinated_transport",
            goal,
            success,
        )
    planning = {
        "E6": ("PullArticulatedPart", "pull_articulated_part"),
        "E7": ("PushArticulatedPart", "push_articulated_part"),
        "E8": ("TurnKnob", "turn_knob"),
    }
    if task_type in planning:
        action_name, operator = planning[task_type]
        success = {
            "type": "articulation_joint_near",
            "object": object_uid,
            "target_state": params.get("target_state", params.get("target_setting")),
        }
        return (
            [
                _node(
                    group_id,
                    1,
                    action_name,
                    task_type,
                    object_uid,
                    actor,
                    "arm",
                    {"kind": "articulation_goal", "object": object_uid},
                    dependencies,
                    role,
                    success,
                    motion_policy(),
                )
            ],
            operator,
            {
                key: deepcopy(value)
                for key, value in params.items()
                if not key.endswith("_role")
            },
            success,
        )
    if task_type == "E9":
        success = {
            "type": "pressed",
            "object": object_uid,
            "terminal_state": str(params.get("terminal_state", "activated")),
        }
        return (
            [
                _node(
                    group_id,
                    1,
                    "Press",
                    task_type,
                    object_uid,
                    actor,
                    "arm",
                    {"kind": "object", "object": object_uid},
                    dependencies,
                    role,
                    success,
                    motion_policy(),
                )
            ],
            "press",
            {"terminal_state": success["terminal_state"]},
            success,
        )
    raise ValueError(f"Unsupported task type {task_type!r}.")


def _single_arm_manipulation(
    group_id: str,
    task_type: str,
    object_uid: str,
    actor: Mapping[str, Any],
    dependencies: list[str],
    *,
    role: str,
    already_held: bool = False,
    leave_held: bool = False,
    payloads: Sequence[Mapping[str, Any]] = (),
) -> list[dict[str, Any]]:
    orientation_modifiers: tuple[tuple[str, str], ...] = (
        (("orientation", "upright"),) if task_type == "E2" else ()
    )
    payload_binding = deepcopy(list(payloads))
    specs = (
        (
            "PickUp",
            {
                "kind": "object",
                "object": object_uid,
                **({"payloads": payload_binding} if payload_binding else {}),
            },
            motion_policy(*orientation_modifiers),
        ),
        (
            "MoveHeldObject",
            {
                "kind": "semantic_goal",
                "semantic_step": group_id,
                "phase": "staging",
                **({"payloads": payload_binding} if payload_binding else {}),
            },
            motion_policy(*orientation_modifiers),
        ),
        (
            "MoveHeldObject",
            {
                "kind": "semantic_goal",
                "semantic_step": group_id,
                "phase": "final",
                **({"payloads": payload_binding} if payload_binding else {}),
            },
            motion_policy(*orientation_modifiers),
        ),
        (
            "Place",
            {
                "kind": "current_held_pose",
                **({"payloads": payload_binding} if payload_binding else {}),
            },
            motion_policy(*orientation_modifiers),
        ),
        (
            "MoveEndEffector",
            {"kind": "policy_pose", "source": "release", "operation": "retreat"},
            motion_policy(*orientation_modifiers),
        ),
        (
            "MoveJoints",
            {"kind": "joint_state", "source": "initial"},
            motion_policy(),
        ),
    )
    if already_held:
        specs = specs[1:]
    if leave_held:
        # A held continuation must not retreat or home the arm after the final
        # semantic move: those cleanup phases would move away from the
        # handover staging state while still owning the object.
        place_index = next(
            (index for index, spec in enumerate(specs) if spec[0] == "Place"),
            len(specs),
        )
        specs = specs[:place_index]
    nodes = []
    previous = list(dependencies)
    for index, (action, binding, policy) in enumerate(specs, start=1):
        node_role = "cleanup" if action in {"MoveEndEffector", "MoveJoints"} else role
        node = _node(
            group_id,
            index,
            action,
            task_type,
            object_uid,
            actor,
            "arm",
            binding,
            previous,
            node_role,
            {},
            policy,
        )
        nodes.append(node)
        previous = [node["id"]]
    return nodes


def _node(
    group_id: str,
    index: int,
    action: str,
    task_type: str,
    object_uid: str,
    actor: Mapping[str, Any],
    control: str,
    binding: Mapping[str, Any],
    dependencies: list[str],
    role: str,
    postcondition: Mapping[str, Any],
    motion_policy: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "id": f"{group_id}__a{index:02d}",
        "atomic_action": action,
        "object_uid": object_uid,
        "actor": deepcopy(dict(actor)),
        "control": control,
        "target_binding": deepcopy(dict(binding)),
        "depends_on": list(dependencies),
        "task_instance_id": group_id,
        "task_type": task_type,
        "role": role,
        "precondition": {},
        "postcondition": deepcopy(dict(postcondition)),
        "motion_policy": deepcopy(dict(motion_policy)),
    }


def _terminal_nodes(nodes: list[Mapping[str, Any]]) -> list[str]:
    depended = {dependency for node in nodes for dependency in node["depends_on"]}
    return [str(node["id"]) for node in nodes if node["id"] not in depended]


def _primary_object(task_type: str, params: Mapping[str, Any]) -> str:
    key = "source_role" if task_type == "E3" else "object_role"
    value = params.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{task_type} requires resolved parameter {key!r}.")
    return value


def _actor(
    task_type: str,
    params: Mapping[str, Any],
    *,
    incoming_held_arm: str | None = None,
) -> dict[str, Any]:
    required_arm = params.get("required_arm")
    if (
        incoming_held_arm is not None
        and required_arm in {"left_arm", "right_arm"}
        and str(required_arm) != incoming_held_arm
    ):
        raise ValueError(
            f"Continuation requires {incoming_held_arm!r}, but the task "
            f"requested {required_arm!r}."
        )
    if incoming_held_arm is not None:
        return {"mode": "required", "arm": incoming_held_arm}
    if required_arm in {"left_arm", "right_arm"}:
        return {"mode": "required", "arm": str(required_arm)}
    if task_type == "E5":
        return {"mode": "coordinated", "arms": ["left_arm", "right_arm"]}
    if task_type == "E4":
        return {"mode": "required", "arm": str(params.get("transfer_arm", "left_arm"))}
    return {"mode": "auto"}


def _incoming_held_arm(
    task_type: str,
    object_uid: str,
    dependencies: list[str],
    held_after_group: Mapping[str, tuple[str, str] | None],
) -> str | None:
    """Resolve a predecessor-provided hold for a continuation recipe."""
    if task_type not in {"E1", "E2", "E4"}:
        return None
    candidates = {
        held[1]
        for dependency in dependencies
        if (held := held_after_group.get(str(dependency))) is not None
        and held[0] == object_uid
    }
    if len(candidates) > 1:
        raise ValueError(
            f"Task instance has conflicting predecessor holders for {object_uid!r}."
        )
    return next(iter(candidates), None)


def _terminal_hold(
    task_type: str,
    object_uid: str,
    params: Mapping[str, Any],
) -> tuple[str, str] | None:
    if task_type == "E4":
        return object_uid, str(params.get("receive_arm", "right_arm"))
    if task_type == "E2" and str(params.get("terminal_behavior", "place")) == "hold":
        arm = str(params.get("required_arm", ""))
        if arm in {"left_arm", "right_arm"}:
            return object_uid, arm
    if task_type == "E5" and str(params.get("terminal_behavior", "hold")) == "hold":
        return object_uid, "coordinated"
    return None


def _validate_bindings(
    task: Mapping[str, Any],
    role_bindings: Mapping[str, str],
) -> dict[str, str]:
    bindings = dict(role_bindings)
    for role, uid in bindings.items():
        if not isinstance(role, str) or not role or not isinstance(uid, str) or not uid:
            raise ValueError("role_bindings must map non-empty role IDs to scene UIDs.")
    required = set()
    for instance in task["task_instances"]:
        required.update(_role_references(instance["params"]))
    required.discard("table")
    missing = sorted(required - set(bindings))
    if missing:
        raise ValueError(f"Scene hand-off is missing role bindings: {missing}.")
    if len(bindings.values()) != len(set(bindings.values())):
        raise ValueError("Scene role bindings must resolve to unique object UIDs.")
    return bindings


def _role_references(value: Any, key: str = "") -> set[str]:
    if isinstance(value, Mapping):
        return {
            role
            for child_key, child in value.items()
            for role in _role_references(child, str(child_key))
        }
    if isinstance(value, list):
        return {role for child in value for role in _role_references(child, key)}
    if isinstance(value, str) and (key.endswith("_role") or key.endswith("_roles")):
        return {value}
    return set()


def _resolve_params(value: Any, bindings: Mapping[str, str], key: str = "") -> Any:
    if isinstance(value, Mapping):
        return {
            child_key: _resolve_params(child, bindings, str(child_key))
            for child_key, child in value.items()
        }
    if isinstance(value, list):
        return [_resolve_params(child, bindings, key) for child in value]
    if isinstance(value, str) and (key.endswith("_role") or key.endswith("_roles")):
        return bindings.get(value, value)
    return deepcopy(value)
