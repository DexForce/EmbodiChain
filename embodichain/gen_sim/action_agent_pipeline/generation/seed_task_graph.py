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
    return _build_executable_seed(
        task_name=task_name,
        route="object_manipulation",
        program=spec.intent,
        semantic_steps=semantic_steps,
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

    if order_is_constrained:
        configured_order = tuple(getattr(spec, "semantic_order", ()))
        ordered_uids = list(
            configured_order
            or (
                step.runtime_uid
                for step in sorted(source_steps, key=lambda item: int(item.slot_index))
            )
        )
        if set(ordered_uids) != set(runtime_uids) or len(ordered_uids) != len(
            runtime_uids
        ):
            raise ValueError(
                "Arrangement semantic_order must contain every selected object once."
            )
    else:
        ordered_uids = sorted(runtime_uids)

    source_by_uid = {str(step.runtime_uid): step for step in source_steps}
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
                    "anchor": spec.anchor,
                    "order_by": spec.order_by,
                    "order_direction": spec.order_direction,
                    "slot_index": slot_index,
                    "orientation_goal": str(source_step.orientation_goal),
                    "orientation_axis": str(source_step.orientation_axis),
                },
                "depends_on": (
                    [previous_step_id] if previous_step_id is not None else []
                ),
                "postcondition": {
                    "type": "line_member_placed",
                    "slot_index": slot_index,
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
) -> dict[str, Any]:
    if not semantic_steps:
        raise ValueError("Executable Seed v2 requires at least one semantic step.")
    nodes = [
        {
            "id": "v0_start",
            "semantic": "Initial state before executing the symbolic action graph",
        }
    ]
    edges: list[dict[str, Any]] = []
    previous_node_id = "v0_start"
    edge_index = 0
    for step in semantic_steps:
        step_edge_ids: list[str] = []
        for action_name, state_semantic, actions in _symbolic_actions_for_step(step):
            edge_index += 1
            slug = _slugify(f"{step['id']}_{action_name}")
            edge_id = f"e{edge_index:02d}_{slug}"
            target_id = f"v{edge_index}_{slug}"
            nodes.append({"id": target_id, "semantic": state_semantic})
            edges.append(
                {
                    "id": edge_id,
                    "source": previous_node_id,
                    "target": target_id,
                    "actions": actions,
                }
            )
            step_edge_ids.append(edge_id)
            previous_node_id = target_id
        step["edge_ids"] = step_edge_ids
    nodes[-1]["id"] = f"v{edge_index}_done"
    edges[-1]["target"] = nodes[-1]["id"]
    graph = {
        "schema_version": SEED_TASK_GRAPH_SCHEMA_VERSION,
        "task": task_name,
        "route": route,
        "program": program,
        "start": "v0_start",
        "goal": nodes[-1]["id"],
        "nodes": nodes,
        "edges": edges,
        "semantic_step_schema_version": SEMANTIC_STEP_SCHEMA_VERSION,
        "semantic_steps": semantic_steps,
        "motion_policy_version": MOTION_POLICY_VERSION,
    }
    validate_seed_task_graph(graph)
    return graph


def _symbolic_actions_for_step(
    step: Mapping[str, Any],
) -> list[tuple[str, str, list[dict[str, Any]]]]:
    actor = deepcopy(dict(step["actor"]))
    object_uid = str(step["object"])
    operator = str(step["operator"])
    if actor["mode"] == "coordinated":
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

    actions = [
        (
            "pick_up",
            f"Holding `{object_uid}`",
            [
                _symbolic_action(
                    "PickUp",
                    actor,
                    "object",
                    "default_pickup",
                    object=object_uid,
                    affordance="antipodal",
                )
            ],
        ),
        (
            "move_to_semantic_goal",
            f"`{object_uid}` held at its semantic goal",
            [
                _symbolic_action(
                    "MoveHeldObject",
                    actor,
                    "semantic_goal",
                    "default_transport",
                    semantic_step=str(step["id"]),
                )
            ],
        ),
    ]
    if operator == "hold_hover":
        actions.append(
            (
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
        )
        return actions
    actions.extend(
        [
            (
                "release",
                f"`{object_uid}` released at its semantic goal",
                [
                    _symbolic_action(
                        "Place",
                        actor,
                        "current_held_pose",
                        "default_release",
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
                        "default_retreat",
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
    )
    return actions


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
