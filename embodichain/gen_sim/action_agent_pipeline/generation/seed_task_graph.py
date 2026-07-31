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

"""Build and validate executable, environment-independent seed task graphs."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import re
from typing import Any

from embodichain.gen_sim.action_agent_pipeline.domain.seed_task_graph import (
    MOTION_POLICY_VERSION,
    SEED_TASK_GRAPH_SCHEMA_VERSION,
    SEMANTIC_STEP_SCHEMA_VERSION,
    seed_task_graph_hash,
    validate_seed_task_graph,
)
from embodichain.gen_sim.action_agent_pipeline.generation.arrangement_intent import (
    _arrangement_order_is_constrained,
)
from embodichain.gen_sim.action_agent_pipeline.generation.builder_protocols import (
    ArrangementSpecLike,
    RelativePlacementLike,
    RelativeSpecLike,
    StackingSpecLike,
)

__all__ = [
    "MOTION_POLICY_VERSION",
    "SEED_TASK_GRAPH_SCHEMA_VERSION",
    "SEMANTIC_STEP_SCHEMA_VERSION",
    "make_arrangement_seed_task_graph",
    "make_relative_seed_task_graph",
    "make_stacking_seed_task_graph",
    "seed_task_graph_hash",
    "validate_seed_task_graph",
]

_UNSAFE_ID_RE = re.compile(r"[^0-9a-z]+")


def make_relative_seed_task_graph(
    task_name: str,
    spec: RelativeSpecLike,
) -> dict[str, Any]:
    """Expand ordered manipulation semantics into a symbolic action topology."""
    semantic_steps: list[dict[str, Any]] = []
    previous_step_id: str | None = None
    for index, placement in enumerate(spec.placements, start=1):
        operator = "hold_hover" if spec.intent == "hold_hover" else placement.intent
        step_id = _relative_step_id(placement, index)
        goal = _relative_seed_goal(placement, operator=operator)
        if operator == "coordinated_pickment":
            goal["direction"] = spec.coordinated_direction
            goal["terminal_behavior"] = spec.coordinated_terminal_behavior
        configured_dependencies = tuple(getattr(placement, "depends_on", ()))
        depends_on = list(
            configured_dependencies
            or ((previous_step_id,) if previous_step_id is not None else ())
        )
        semantic_steps.append(
            {
                "id": step_id,
                "operator": operator,
                "object": placement.moved_runtime_uid,
                "actor": _relative_seed_actor(placement, operator=operator),
                "goal": goal,
                "depends_on": depends_on,
                "postcondition": {
                    "type": "semantic_goal",
                    "operator": operator,
                    "relation": goal["relation"],
                },
            }
        )
        previous_step_id = step_id
    allocation_groups = _relative_allocation_groups(
        semantic_steps,
        parallel_pickup_requested=bool(
            getattr(spec, "parallel_pickup_requested", False)
        ),
    )
    return _build_executable_seed(
        task_name=task_name,
        route="object_manipulation",
        program=spec.intent,
        semantic_steps=semantic_steps,
        allocation_groups=allocation_groups,
    )


def make_arrangement_seed_task_graph(
    task_name: str,
    spec: ArrangementSpecLike,
) -> dict[str, Any]:
    """Expand each arrangement member into an independently grounded step."""
    order_is_constrained = _arrangement_order_is_constrained(
        spec.order_by,
        task_description=spec.task_description,
    )
    source_steps = list(spec.steps)
    runtime_uids = [str(step.runtime_uid) for step in source_steps]
    if len(runtime_uids) != len(set(runtime_uids)):
        raise ValueError("Arrangement seed objects must be distinct.")

    ordered_source_steps = sorted(
        source_steps,
        key=lambda item: int(item.slot_index),
    )
    ordered_uids = [str(step.runtime_uid) for step in ordered_source_steps]
    if order_is_constrained:
        configured_order = tuple(getattr(spec, "semantic_order", ()))
        if configured_order and list(configured_order) != ordered_uids:
            raise ValueError(
                "Ordered arrangement nominal slots must preserve semantic_order."
            )

    source_by_uid = {str(step.runtime_uid): step for step in source_steps}
    slot_constraint = "required" if order_is_constrained else "free_reassignable"
    anchor = str(spec.anchor)
    if anchor == "center":
        anchor = "table_center"
    if anchor != "table_center":
        raise ValueError("Arrangement Seed v5 requires anchor='table_center'.")
    semantic_steps: list[dict[str, Any]] = []
    previous_step_id: str | None = None
    for slot_index, object_uid in enumerate(ordered_uids):
        source_step = source_by_uid[object_uid]
        step_id = _seed_step_id(slot_index + 1, "place_in_line", object_uid)
        semantic_steps.append(
            {
                "id": step_id,
                "operator": "place_in_line",
                "object": object_uid,
                "actor": {"mode": "auto"},
                "goal": {
                    "layout": "line",
                    "objects": ordered_uids,
                    "order_constraint": ("ordered" if order_is_constrained else "free"),
                    "axis": spec.axis,
                    "anchor": anchor,
                    "order_by": spec.order_by,
                    "order_direction": spec.order_direction,
                    "nominal_slot_index": slot_index,
                    "slot_constraint": slot_constraint,
                    "orientation_goal": str(source_step.orientation_goal),
                    "orientation_axis": str(source_step.orientation_axis),
                },
                "depends_on": (
                    [previous_step_id] if previous_step_id is not None else []
                ),
                "postcondition": {
                    "type": "line_member_placed",
                    "nominal_slot_index": slot_index,
                    "slot_constraint": slot_constraint,
                    "order_constraint": ("ordered" if order_is_constrained else "free"),
                },
            }
        )
        previous_step_id = step_id
    return _build_executable_seed(
        task_name=task_name,
        route="arrangement_line",
        program="arrange_in_line",
        semantic_steps=semantic_steps,
    )


def make_stacking_seed_task_graph(
    task_name: str,
    spec: StackingSpecLike,
) -> dict[str, Any]:
    """Expand bottom-to-top symbolic support relations into an action topology."""
    ordered_steps = sorted(spec.steps, key=lambda step: int(step.layer_index))
    semantic_steps: list[dict[str, Any]] = []
    previous_step_id: str | None = None
    for index, step in enumerate(ordered_steps, start=1):
        step_id = _seed_step_id(index, "place_on_stack", step.runtime_uid)
        reference_object = step.support_runtime_uid or spec.anchor_runtime_uid
        semantic_steps.append(
            {
                "id": step_id,
                "operator": "place_on_stack",
                "object": step.runtime_uid,
                "actor": {"mode": "auto"},
                "goal": {
                    "relation": "on",
                    "reference_object": reference_object,
                    "reference_state": (
                        "live" if reference_object else "symbolic_anchor"
                    ),
                    "layer_index": int(step.layer_index),
                    "stack_mode": spec.stack_mode,
                    "orientation_goal": str(step.orientation_goal),
                    "orientation_axis": str(step.orientation_axis),
                },
                "depends_on": (
                    [previous_step_id] if previous_step_id is not None else []
                ),
                "postcondition": {
                    "type": "stack_layer_supported",
                    "layer_index": int(step.layer_index),
                    "reference_object": reference_object,
                },
            }
        )
        previous_step_id = step_id
    return _build_executable_seed(
        task_name=task_name,
        route="stacking",
        program="build_stack",
        semantic_steps=semantic_steps,
    )


def _build_executable_seed(
    *,
    task_name: str,
    route: str,
    program: str,
    semantic_steps: list[dict[str, Any]],
    allocation_groups: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if not semantic_steps:
        raise ValueError("Executable Seed v5 requires at least one semantic step.")
    # Edge IDs belong to the generated protocol artifact. Work on a deep copy so
    # callers can safely reuse their semantic input for diagnostics or retries.
    semantic_steps = deepcopy(semantic_steps)
    nodes = [
        {
            "id": "v0_start",
            "semantic": "Initial state before executing the symbolic action graph",
        }
    ]
    edges: list[dict[str, Any]] = []
    previous_node_id = "v0_start"
    previous_edge_id: str | None = None
    edge_index = 0
    node_index = 0

    def new_node(action_name: str, state_semantic: str) -> str:
        nonlocal node_index
        node_index += 1
        slug = _slugify(action_name)
        node_id = f"v{node_index}_{slug}"
        nodes.append({"id": node_id, "semantic": state_semantic})
        return node_id

    def append_edge(
        step: Mapping[str, Any],
        action_record: tuple[str, str, list[dict[str, Any]]],
        *,
        source: str,
        dependencies: list[str],
        target: str | None = None,
    ) -> tuple[str, str]:
        nonlocal edge_index
        action_name, state_semantic, actions = action_record
        edge_index += 1
        slug = _slugify(f"{step['id']}_{action_name}")
        edge_id = f"e{edge_index:02d}_{slug}"
        target_id = target or new_node(slug, state_semantic)
        edges.append(
            {
                "id": edge_id,
                "source": source,
                "target": target_id,
                "actions": actions,
                "depends_on": list(dependencies),
                "resources": _symbolic_edge_resources(step, actions),
            }
        )
        return edge_id, target_id

    step_index = 0
    while step_index < len(semantic_steps):
        first_step = semantic_steps[step_index]
        second_step = (
            semantic_steps[step_index + 1]
            if step_index + 1 < len(semantic_steps)
            else None
        )
        if second_step is not None and _can_prefetch_pickups(
            first_step,
            second_step,
            route=route,
            allocation_groups=allocation_groups or [],
        ):
            first_records = _symbolic_actions_for_step(first_step)
            second_records = _symbolic_actions_for_step(second_step)
            shared_dependencies = (
                [previous_edge_id] if previous_edge_id is not None else []
            )
            first_pick_id, first_holding_node = append_edge(
                first_step,
                first_records[0],
                source=previous_node_id,
                dependencies=shared_dependencies,
            )
            join_node = new_node(
                f"{first_step['id']}_{second_step['id']}_join",
                (
                    f"Step `{first_step['id']}` complete; "
                    f"`{second_step['object']}` remains held"
                ),
            )
            second_pick_id, _ = append_edge(
                second_step,
                second_records[0],
                source=previous_node_id,
                dependencies=shared_dependencies,
                target=join_node,
            )
            first_edge_ids = [first_pick_id]
            source = first_holding_node
            dependency_ids = [first_pick_id, second_pick_id]
            for record_index, record in enumerate(first_records[1:], start=1):
                is_last = record_index == len(first_records) - 1
                edge_id, source = append_edge(
                    first_step,
                    record,
                    source=source,
                    dependencies=dependency_ids,
                    target=join_node if is_last else None,
                )
                first_edge_ids.append(edge_id)
                dependency_ids = [edge_id]
            first_step["edge_ids"] = first_edge_ids

            second_edge_ids = [second_pick_id]
            source = join_node
            dependency_ids = [first_edge_ids[-1], second_pick_id]
            for record in second_records[1:]:
                edge_id, source = append_edge(
                    second_step,
                    record,
                    source=source,
                    dependencies=dependency_ids,
                )
                second_edge_ids.append(edge_id)
                dependency_ids = [edge_id]
            second_step["edge_ids"] = second_edge_ids
            previous_node_id = source
            previous_edge_id = second_edge_ids[-1]
            step_index += 2
            continue

        step_edge_ids: list[str] = []
        for record in _symbolic_actions_for_step(first_step):
            dependencies = [previous_edge_id] if previous_edge_id is not None else []
            edge_id, previous_node_id = append_edge(
                first_step,
                record,
                source=previous_node_id,
                dependencies=dependencies,
            )
            step_edge_ids.append(edge_id)
            previous_edge_id = edge_id
        first_step["edge_ids"] = step_edge_ids
        step_index += 1

    goal_id = f"v{node_index}_done"
    nodes[-1]["id"] = goal_id
    for edge in edges:
        if edge["target"] == previous_node_id:
            edge["target"] = goal_id
    previous_node_id = goal_id
    graph = {
        "schema_version": SEED_TASK_GRAPH_SCHEMA_VERSION,
        "task": task_name,
        "route": route,
        "program": program,
        "start": "v0_start",
        "goal": previous_node_id,
        "nodes": nodes,
        "edges": edges,
        "semantic_step_schema_version": SEMANTIC_STEP_SCHEMA_VERSION,
        "semantic_steps": semantic_steps,
        "allocation_groups": allocation_groups or [],
        "motion_policy_version": MOTION_POLICY_VERSION,
    }
    validate_seed_task_graph(graph)
    return graph


def _can_prefetch_pickups(
    first_step: Mapping[str, Any],
    second_step: Mapping[str, Any],
    *,
    route: str,
    allocation_groups: list[dict[str, Any]],
) -> bool:
    """Allow only declared, resource-safe dual-PickUp parallelism."""
    if route != "object_manipulation":
        return False
    if any(
        step["goal"].get("placement_mode") == "upright_in_place"
        for step in (first_step, second_step)
    ):
        return False
    first_actor = first_step["actor"]
    second_actor = second_step["actor"]
    group_declares_pair = any(
        group["semantic_step_ids"] == [first_step["id"], second_step["id"]]
        for group in allocation_groups
    )
    required_opposite = (
        first_actor.get("mode") == "required"
        and second_actor.get("mode") == "required"
        and first_actor.get("arm") != second_actor.get("arm")
    )
    if not group_declares_pair and not required_opposite:
        return False
    if first_step["object"] == second_step["object"]:
        return False
    if second_step.get("depends_on") != [first_step["id"]]:
        return False
    moved_objects = {str(first_step["object"]), str(second_step["object"])}
    references = {
        str(step["goal"].get("reference_object"))
        for step in (first_step, second_step)
        if step["goal"].get("reference_object") is not None
    }
    return moved_objects.isdisjoint(references)


def _relative_allocation_groups(
    semantic_steps: list[dict[str, Any]],
    *,
    parallel_pickup_requested: bool,
) -> list[dict[str, Any]]:
    """Preserve structured dual-arm intent without fixing automatic arm identity."""
    if len(semantic_steps) != 2:
        return []
    first, second = semantic_steps
    if any(
        step["goal"].get("placement_mode") == "upright_in_place"
        for step in (first, second)
    ):
        # Do not prefetch one upright object while the other arm completes its
        # full manipulation. Upright parallelism requires a complete joint plan.
        return []
    first_actor = first["actor"]
    second_actor = second["actor"]
    explicitly_opposite = (
        first_actor.get("mode") == "required"
        and second_actor.get("mode") == "required"
        and first_actor.get("arm") != second_actor.get("arm")
    )
    if not explicitly_opposite and not parallel_pickup_requested:
        return []
    if first["object"] == second["object"]:
        return []
    moved_objects = {str(first["object"]), str(second["object"])}
    references = {
        str(step["goal"].get("reference_object"))
        for step in (first, second)
        if step["goal"].get("reference_object") is not None
    }
    if not moved_objects.isdisjoint(references):
        return []
    return [
        {
            "id": "g01_dual_pickup",
            "semantic_step_ids": [first["id"], second["id"]],
            "arm_constraint": "distinct_arms",
            "execution_policy": "parallel_if_feasible",
            "parallel_action_classes": ["PickUp"],
            "workspace_policy": "shared_target_serial",
        }
    ]


def _symbolic_edge_resources(
    step: Mapping[str, Any],
    actions: list[dict[str, Any]],
) -> list[str]:
    """Declare conservative resources consumed by one symbolic action edge."""
    resources = {f"object:{step['object']}"}
    actor = step["actor"]
    if actor.get("mode") == "required":
        resources.add(f"arm:{actor['arm']}")
    elif actor.get("mode") == "coordinated":
        resources.update({"arm:left_arm", "arm:right_arm"})
    else:
        resources.add("arm:auto")

    action_classes = {str(action["atomic_action_class"]) for action in actions}
    if action_classes & {"MoveHeldObject", "Place", "CoordinatedPickment"}:
        reference = step["goal"].get("reference_object")
        if reference is not None:
            resources.add(f"workspace:{reference}")
        elif step["operator"] == "place_in_line":
            resources.add("workspace:table")
        else:
            resources.add("workspace:world")
    return sorted(resources)


def _symbolic_actions_for_step(
    step: Mapping[str, Any],
) -> list[tuple[str, str, list[dict[str, Any]]]]:
    """Expand one semantic operator through small route-neutral action blocks."""
    actor = deepcopy(dict(step["actor"]))
    if actor["mode"] == "coordinated":
        return _coordinated_symbolic_actions_for_step(step, actor)

    actions = [_pickup_action_record(step, actor)]
    actions.extend(_transport_action_records(step, actor))
    if step["operator"] == "hold_hover":
        actions.append(_hold_action_record(step, actor))
        return actions
    actions.extend(_release_action_records(step, actor))
    return actions


def _coordinated_symbolic_actions_for_step(
    step: Mapping[str, Any],
    actor: Mapping[str, Any],
) -> list[tuple[str, str, list[dict[str, Any]]]]:
    """Build the shared-object action sequence owned by both physical arms."""
    object_uid = str(step["object"])
    actions = [
        (
            "coordinated_manipulation",
            f"`{object_uid}` held by both arms at its semantic goal",
            [
                _symbolic_action(
                    "CoordinatedPickment",
                    actor,
                    "coordinated_goal",
                    "default_transport",
                    object=object_uid,
                )
            ],
        )
    ]
    if step["goal"].get("terminal_behavior") != "place":
        return actions
    actions.extend(
        [
            (
                "dual_release",
                f"`{object_uid}` released at its semantic goal",
                _dual_arm_symbolic_actions(
                    "MoveJoints",
                    binding_kind="joint_state",
                    policy="default_release",
                    source="gripper_open",
                ),
            ),
            (
                "dual_retreat",
                f"Both end effectors retreated after placing `{object_uid}`",
                _dual_arm_symbolic_actions(
                    "MoveEndEffector",
                    binding_kind="policy_pose",
                    policy="default_retreat",
                ),
            ),
            (
                "dual_home",
                f"Step `{step['id']}` complete; both arms at initial state",
                _dual_arm_symbolic_actions(
                    "MoveJoints",
                    binding_kind="joint_state",
                    policy="default_home",
                    source="initial",
                ),
            ),
        ]
    )
    return actions


def _pickup_action_record(
    step: Mapping[str, Any],
    actor: Mapping[str, Any],
) -> tuple[str, str, list[dict[str, Any]]]:
    object_uid = str(step["object"])
    upright = step["goal"].get("placement_mode") == "upright_in_place"
    return (
        "pick_up",
        f"Holding `{object_uid}`",
        [
            _symbolic_action(
                "PickUp",
                actor,
                "object",
                "upright_in_place_pickup" if upright else "default_pickup",
                object=object_uid,
                affordance="antipodal",
            )
        ],
    )


def _transport_action_records(
    step: Mapping[str, Any],
    actor: Mapping[str, Any],
) -> list[tuple[str, str, list[dict[str, Any]]]]:
    """Build route-specific transport while keeping terminal actions separate."""
    object_uid = str(step["object"])
    step_id = str(step["id"])
    if step["operator"] == "place_in_line":
        return [
            (
                "move_to_staging",
                f"`{object_uid}` held above its nominal line slot",
                [
                    _symbolic_action(
                        "MoveHeldObject",
                        actor,
                        "semantic_goal",
                        "default_transport",
                        semantic_step=step_id,
                        phase="staging",
                    )
                ],
            ),
            (
                "move_to_final",
                f"`{object_uid}` held at its nominal line slot",
                [
                    _symbolic_action(
                        "MoveHeldObject",
                        actor,
                        "semantic_goal",
                        "default_transport",
                        semantic_step=step_id,
                        phase="final",
                    )
                ],
            ),
        ]
    if step["goal"].get("placement_mode") == "upright_in_place":
        return [
            (
                "upright_at_staging",
                f"`{object_uid}` held upright above its initial position",
                [
                    _symbolic_action(
                        "MoveHeldObject",
                        actor,
                        "semantic_goal",
                        "upright_in_place_transport",
                        semantic_step=step_id,
                        phase="staging",
                    )
                ],
            ),
            (
                "move_to_semantic_goal",
                f"`{object_uid}` held upright at its initial XY",
                [
                    _symbolic_action(
                        "MoveHeldObject",
                        actor,
                        "semantic_goal",
                        "upright_in_place_transport",
                        semantic_step=step_id,
                        phase="final",
                    )
                ],
            ),
        ]
    return [
        (
            "move_to_semantic_goal",
            f"`{object_uid}` held at its semantic goal",
            [
                _symbolic_action(
                    "MoveHeldObject",
                    actor,
                    "semantic_goal",
                    "default_transport",
                    semantic_step=step_id,
                )
            ],
        )
    ]


def _hold_action_record(
    step: Mapping[str, Any],
    actor: Mapping[str, Any],
) -> tuple[str, str, list[dict[str, Any]]]:
    object_uid = str(step["object"])
    return (
        "keep_holding",
        f"`{object_uid}` remains held at its semantic goal",
        [
            _symbolic_action(
                "MoveJoints",
                actor,
                "joint_state",
                "default_release",
                source="gripper_closed",
            )
        ],
    )


def _release_action_records(
    step: Mapping[str, Any],
    actor: Mapping[str, Any],
) -> list[tuple[str, str, list[dict[str, Any]]]]:
    object_uid = str(step["object"])
    upright = step["goal"].get("placement_mode") == "upright_in_place"
    return [
        (
            "release",
            f"`{object_uid}` released at its semantic goal",
            [
                _symbolic_action(
                    "Place",
                    actor,
                    "current_held_pose",
                    "upright_in_place_release" if upright else "default_release",
                )
            ],
        ),
        (
            "retreat",
            f"End effector retreated after placing `{object_uid}`",
            [
                _symbolic_action(
                    "MoveEndEffector",
                    actor,
                    "policy_pose",
                    "upright_in_place_retreat" if upright else "default_retreat",
                )
            ],
        ),
        (
            "home",
            f"Step `{step['id']}` complete; arm at initial state",
            [
                _symbolic_action(
                    "MoveJoints",
                    actor,
                    "joint_state",
                    "default_home",
                    source="initial",
                )
            ],
        ),
    ]


def _dual_arm_symbolic_actions(
    action_class: str,
    *,
    binding_kind: str,
    policy: str,
    **binding: Any,
) -> list[dict[str, Any]]:
    return [
        _symbolic_action(
            action_class,
            {"mode": "required", "arm": arm},
            binding_kind,
            policy,
            **binding,
        )
        for arm in ("left_arm", "right_arm")
    ]


def _symbolic_action(
    action_class: str,
    actor: Mapping[str, Any],
    binding_kind: str,
    policy: str,
    **binding: Any,
) -> dict[str, Any]:
    return {
        "atomic_action_class": action_class,
        "actor": deepcopy(dict(actor)),
        "control": (
            "coordinated"
            if action_class == "CoordinatedPickment"
            else (
                "hand"
                if binding.get("source") in {"gripper_closed", "gripper_open"}
                else "arm"
            )
        ),
        "target_binding": {"kind": binding_kind, **binding},
        "motion_policy": policy,
    }


def _relative_seed_actor(
    placement: RelativePlacementLike,
    *,
    operator: str,
) -> dict[str, Any]:
    if operator == "coordinated_pickment":
        return {"mode": "coordinated", "arms": ["left_arm", "right_arm"]}
    arm_request = str(getattr(placement, "arm_request", "auto"))
    if arm_request == "auto":
        return {"mode": "auto"}
    return {"mode": "required", "arm": f"{arm_request}_arm"}


def _relative_seed_goal(
    placement: RelativePlacementLike,
    *,
    operator: str,
) -> dict[str, Any]:
    goal = {
        "relation": placement.relation,
        "reference_object": placement.reference_runtime_uid,
        "reference_state": (
            "initial" if placement.reference_is_initial_pose else "live"
        ),
        "orientation_goal": placement.orientation_goal,
        "orientation_axis": placement.orientation_axis,
    }
    if operator == "hold_hover":
        goal["relation"] = "held_above_initial"
        goal["reference_object"] = placement.moved_runtime_uid
        goal["reference_state"] = "initial"
    if placement.orientation_align_to_runtime_uid is not None:
        goal["orientation_reference_object"] = (
            placement.orientation_align_to_runtime_uid
        )
    if bool(getattr(placement, "upright_in_place", False)):
        goal["placement_mode"] = "upright_in_place"
        pickup_direction = getattr(placement, "pickup_upright_direction", None)
        if pickup_direction is not None:
            axis_index = max(
                range(3),
                key=lambda index: abs(float(pickup_direction[index])),
            )
            goal["upright_local_axis"] = ("x", "y", "z")[axis_index]
    return goal


def _relative_step_id(placement: RelativePlacementLike, index: int) -> str:
    configured = str(getattr(placement, "step_id", "")).strip()
    if configured:
        return configured
    return _seed_step_id(
        index,
        placement.intent,
        placement.moved_runtime_uid,
        placement.relation,
        placement.reference_runtime_uid,
    )


def _seed_step_id(index: int, *parts: Any) -> str:
    identity = "_".join(str(part) for part in parts).lower()
    slug = _UNSAFE_ID_RE.sub("_", identity).strip("_") or "step"
    return f"s{index:02d}_{slug[:72].rstrip('_')}"


def _slugify(text: str) -> str:
    slug = _UNSAFE_ID_RE.sub("_", text.lower()).strip("_")
    return slug[:64].rstrip("_") or "step"
