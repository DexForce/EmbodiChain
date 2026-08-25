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

"""Built-in semantic operators lowered to public atomic-action contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from embodichain.gen_sim.action_engine.domain.motion import (
    motion_policy as build_motion_policy,
)
from embodichain.gen_sim.action_engine.domain.task_contracts import (
    PLACEMENT_RELATIONS,
    TERMINAL_BEHAVIORS,
    TRANSPORT_DIRECTIONS,
    normalize_placement_relation,
)
from embodichain.gen_sim.action_engine.orientation import (
    compile_orientation_constraint,
)

from .registry import (
    ActionCapability,
    ActionTemplate,
    CapabilityRegistry,
    OperatorCapability,
    PhaseTemplate,
)

__all__ = ["build_default_registry"]

_SINGLE_ARM_PHASE_OPERATORS = frozenset(
    {"build_stack", "hold_hover", "orient_object", "place_relative"}
)


def build_default_registry() -> CapabilityRegistry:
    """Build a fresh registry containing all Action Engine v1 capabilities."""
    registry = CapabilityRegistry()
    from .atomic import build_atomic_capability_registry

    for capability in build_atomic_capability_registry().catalog().values():
        registry.register_action(
            ActionCapability(
                str(capability["name"]),
                frozenset(capability["binding_kinds"]),
                frozenset(capability["controls"]),
            )
        )

    definitions = (
        OperatorCapability(
            "arrange_line",
            "Arrange two or more movable objects into one live-grounded line.",
            _expand_arrange_line,
            _build_arrange_line_phases,
            expansion_topology="parallel_children",
        ),
        OperatorCapability(
            "build_stack",
            "Build one ordered vertical or nested stack.",
            _expand_build_stack,
            _build_single_arm_phases,
        ),
        OperatorCapability(
            "place_relative",
            "Place one object at a symbolic relation to another object.",
            _expand_place_relative,
            _build_single_arm_phases,
        ),
        OperatorCapability(
            "orient_object",
            "Reorient one object in place and release it in a stable pose.",
            _expand_orient_object,
            _build_orient_object_phases,
        ),
        OperatorCapability(
            "coordinated_transport",
            "Use both arms to pick and transport one shared object.",
            _expand_coordinated_transport,
            _build_coordinated_transport_phases,
        ),
        # These internal operators preserve runtime characterization coverage
        # for public Atomic Actions. They are intentionally absent from the
        # planner catalog during the five-skill first phase.
        OperatorCapability(
            "hold_hover",
            "Internal terminal-hold compatibility operator.",
            _expand_hold_hover,
            _build_single_arm_phases,
            lifecycle="terminal_hold",
            planner_visible=False,
        ),
        OperatorCapability(
            "press",
            "Internal press compatibility operator.",
            _expand_press,
            _build_press_phases,
            planner_visible=False,
        ),
        OperatorCapability(
            "coordinated_place",
            "Internal coordinated-placement compatibility operator.",
            _expand_coordinated_place,
            _build_coordinated_place_phases,
            planner_visible=False,
        ),
    )
    for definition in definitions:
        registry.register_operator(definition)
    return registry


def _expand_arrange_line(step: Mapping[str, Any]) -> list[dict[str, Any]]:
    objects = _collective_objects(step, "arrange_line", minimum=2)
    goal = _goal(
        step,
        allowed={
            "anchor",
            "axis",
            "order_by",
            "order_constraint",
            "order_direction",
            "orientation_axis",
            "orientation_constraint",
            "orientation_directed",
            "orientation_goal",
            "participation",
        },
    )
    orientation_goal, orientation_axis = _orientation(goal, "arrange_line")
    axis = str(goal.get("axis", "world_y"))
    if axis not in {"world_x", "world_y", "table_long_axis"}:
        raise ValueError("arrange_line goal.axis must be a symbolic table axis.")
    anchor = str(goal.get("anchor", "table_center"))
    if anchor != "table_center":
        raise ValueError("arrange_line currently requires anchor='table_center'.")
    order_constraint = str(goal.get("order_constraint", "free"))
    if order_constraint not in {"free", "ordered"}:
        raise ValueError(
            "arrange_line goal.order_constraint must be 'free' or 'ordered'."
        )
    order_by = str(goal.get("order_by", "explicit"))
    if order_by not in {"explicit", "size", "color"}:
        raise ValueError("arrange_line order_by must be explicit, size, or color.")
    order_direction = str(goal.get("order_direction", "given"))
    if order_direction not in {"given", "ascending", "descending"}:
        raise ValueError(
            "arrange_line order_direction must be given, ascending, or descending."
        )
    participation = str(goal.get("participation", "auto"))
    if participation not in {"auto", "both_arms"}:
        raise ValueError("arrange_line participation must be auto or both_arms.")

    common_goal = {
        "layout": "line",
        "objects": objects,
        "axis": axis,
        "anchor": anchor,
        "order_by": order_by,
        "order_direction": order_direction,
        "order_constraint": order_constraint,
        "participation": participation,
        "orientation_goal": orientation_goal,
        "orientation_axis": orientation_axis,
        **_orientation_extensions(goal),
    }
    actor = _single_arm_actor(step)
    expanded: list[dict[str, Any]] = []
    for slot_index, object_uid in enumerate(objects):
        child_goal = {
            **deepcopy(common_goal),
            "nominal_slot_index": slot_index,
            "slot_constraint": (
                "required" if order_constraint == "ordered" else "free_reassignable"
            ),
        }
        expanded.append(
            _execution_step(
                step,
                child_id=f"{step['id']}__{slot_index + 1:02d}",
                object_uid=object_uid,
                actor=(
                    {
                        **actor,
                        "allocation_group": f"{step['id']}_both_arms",
                    }
                    if participation == "both_arms" and slot_index < 2
                    else actor
                ),
                goal=child_goal,
                postcondition={
                    "type": "line_member_placed",
                    "nominal_slot_index": slot_index,
                    "slot_constraint": child_goal["slot_constraint"],
                    "order_constraint": order_constraint,
                },
            )
        )
    return expanded


def _expand_build_stack(step: Mapping[str, Any]) -> list[dict[str, Any]]:
    objects = _collective_objects(step, "build_stack", minimum=1)
    goal = _goal(
        step,
        allowed={
            "anchor",
            "orientation_axis",
            "orientation_constraint",
            "orientation_directed",
            "orientation_goal",
            "stack_mode",
        },
    )
    orientation_goal, orientation_axis = _orientation(goal, "build_stack")
    stack_mode = str(goal.get("stack_mode", "on_top"))
    if stack_mode not in {"on_top", "nested"}:
        raise ValueError("build_stack goal.stack_mode must be 'on_top' or 'nested'.")
    anchor = goal.get("anchor", "table_center")
    if not isinstance(anchor, str) or not anchor:
        raise ValueError("build_stack goal.anchor must be an object or table_center.")

    actor = _single_arm_actor(step)
    expanded: list[dict[str, Any]] = []
    for layer_index, object_uid in enumerate(objects):
        reference = objects[layer_index - 1] if layer_index else anchor
        support_reference = "table" if reference == "table_center" else reference
        child_goal: dict[str, Any] = {
            "relation": (
                "inside" if stack_mode == "nested" and layer_index > 0 else "on"
            ),
            "reference_object": support_reference,
            "reference_state": "live",
            "layer_index": layer_index,
            "stack_mode": stack_mode,
            "orientation_goal": orientation_goal,
            "orientation_axis": orientation_axis,
            **_orientation_extensions(goal),
        }
        expanded.append(
            _execution_step(
                step,
                child_id=f"{step['id']}__{layer_index + 1:02d}",
                object_uid=object_uid,
                actor=actor,
                goal=child_goal,
                postcondition={
                    "type": "stack_layer_supported",
                    "layer_index": layer_index,
                    "reference_object": support_reference,
                },
            )
        )
    return expanded


def _expand_place_relative(step: Mapping[str, Any]) -> list[dict[str, Any]]:
    object_uid = _single_object(step, "place_relative")
    goal = _goal(
        step,
        allowed={
            "orientation_axis",
            "orientation_constraint",
            "orientation_directed",
            "orientation_goal",
            "orientation_reference_object",
            "payloads",
            "reference_object",
            "reference_state",
            "relation",
            "slot",
        },
    )
    orientation_goal, orientation_axis = _orientation(goal, "place_relative")
    reference = _required_string(goal, "reference_object", "place_relative")
    relation = normalize_placement_relation(goal.get("relation", "on"))
    normalized_goal = {
        "reference_object": reference,
        "reference_state": str(goal.get("reference_state", "live")),
        "relation": relation,
        "orientation_goal": orientation_goal,
        "orientation_axis": orientation_axis,
        **_orientation_extensions(goal),
        "slot": str(goal.get("slot", "auto")),
    }
    if normalized_goal["reference_state"] not in {"initial", "live"}:
        raise ValueError("place_relative reference_state must be 'initial' or 'live'.")
    if normalized_goal["slot"] not in {"auto", "left", "center", "right"}:
        raise ValueError("place_relative slot must be left, center, right, or auto.")
    if "orientation_reference_object" in goal:
        normalized_goal["orientation_reference_object"] = goal[
            "orientation_reference_object"
        ]
    payloads = _normalize_payloads(
        goal.get("payloads", []),
        object_uid,
        "place_relative",
    )
    if payloads:
        normalized_goal["payloads"] = payloads
    return [
        _execution_step(
            step,
            object_uid=object_uid,
            actor=_single_arm_actor(step),
            goal=normalized_goal,
            postcondition={
                "type": "semantic_goal",
                "relation": relation,
                "reference_object": reference,
            },
        )
    ]


def _expand_hold_hover(step: Mapping[str, Any]) -> list[dict[str, Any]]:
    object_uid = _single_object(step, "hold_hover")
    goal = _goal(
        step,
        allowed={
            "orientation_axis",
            "orientation_constraint",
            "orientation_directed",
            "orientation_goal",
            "reference_object",
            "reference_state",
        },
    )
    orientation_goal, orientation_axis = _orientation(
        goal,
        "hold_hover",
    )
    reference = str(goal.get("reference_object", object_uid))
    return [
        _execution_step(
            step,
            object_uid=object_uid,
            actor=_single_arm_actor(step),
            goal={
                "relation": "held_above_initial",
                "reference_object": reference,
                "reference_state": str(goal.get("reference_state", "initial")),
                "orientation_goal": orientation_goal,
                "orientation_axis": orientation_axis,
                **_orientation_extensions(goal),
            },
            postcondition={"type": "object_held", "object": object_uid},
        )
    ]


def _expand_orient_object(step: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Normalize an in-place orientation request into one executable step.

    Keeping the target position symbolic is important: runtime observes the
    object's live position immediately before grounding, so prior independent
    operations and simulator settling cannot make this plan stale.
    """
    object_uid = _single_object(step, "orient_object")
    goal = _goal(
        step,
        allowed={
            "orientation_axis",
            "orientation_constraint",
            "orientation_directed",
            "orientation_goal",
            "position_anchor",
            "support_object",
            "upright_local_axis",
        },
    )
    orientation_goal, orientation_axis = _orientation(goal, "orient_object")
    if orientation_goal not in {"upright", "lay_flat", "axis_align"}:
        raise ValueError(
            "orient_object requires upright, lay_flat, or axis_align orientation."
        )
    position_anchor = str(goal.get("position_anchor", "initial_xy"))
    if position_anchor not in {"initial_xy", "live_xy"}:
        raise ValueError(
            "orient_object position_anchor must be 'initial_xy' or 'live_xy'."
        )
    upright_local_axis = str(goal.get("upright_local_axis", "auto"))
    if upright_local_axis not in {"auto", "long_axis", "x", "y", "z"}:
        raise ValueError(
            "orient_object upright_local_axis must be auto, long_axis, x, y, or z."
        )
    support_object = str(goal.get("support_object", "table"))
    if not support_object:
        raise ValueError("orient_object support_object must be a non-empty string.")
    return [
        _execution_step(
            step,
            object_uid=object_uid,
            actor=_single_arm_actor(step),
            goal={
                "relation": "none",
                "reference_state": "live",
                "orientation_goal": orientation_goal,
                "orientation_axis": orientation_axis,
                **_orientation_extensions(goal),
                "position_anchor": position_anchor,
                "support_object": support_object,
                "upright_local_axis": upright_local_axis,
            },
            postcondition={
                "type": "semantic_goal",
                "relation": "none",
                "orientation_goal": orientation_goal,
            },
        )
    ]


def _expand_coordinated_transport(
    step: Mapping[str, Any],
) -> list[dict[str, Any]]:
    object_uid = _single_object(step, "coordinated_transport")
    goal = _goal(
        step,
        allowed={
            "direction",
            "orientation_axis",
            "orientation_constraint",
            "orientation_directed",
            "orientation_goal",
            "payloads",
            "reference_object",
            "relation",
            "terminal_behavior",
        },
    )
    orientation_goal, orientation_axis = _orientation(
        goal,
        "coordinated_transport",
    )
    terminal_behavior = str(goal.get("terminal_behavior", "hold"))
    if terminal_behavior not in TERMINAL_BEHAVIORS - {"none"}:
        raise ValueError(
            "coordinated_transport terminal_behavior must be 'hold' or 'place'."
        )
    direction = str(goal.get("direction", "none"))
    if direction not in TRANSPORT_DIRECTIONS:
        raise ValueError(
            f"coordinated_transport direction {direction!r} is unsupported."
        )
    relation = goal.get("relation")
    if relation is not None and str(relation) not in PLACEMENT_RELATIONS:
        raise ValueError(
            f"coordinated_transport relation {str(relation)!r} is unsupported."
        )
    normalized_goal = {
        "direction": direction,
        "terminal_behavior": terminal_behavior,
        "orientation_goal": orientation_goal,
        "orientation_axis": orientation_axis,
        **_orientation_extensions(goal),
    }
    normalized_payloads = _normalize_payloads(
        goal.get("payloads", []),
        object_uid,
        "coordinated_transport",
    )
    if normalized_payloads:
        normalized_goal["payloads"] = normalized_payloads
    for key in ("reference_object", "relation"):
        if key in goal:
            normalized_goal[key] = goal[key]
    postcondition = (
        {"type": "semantic_goal", "relation": normalized_goal.get("relation", "at")}
        if terminal_behavior == "place"
        else {"type": "held_by_both_grippers", "object": object_uid}
    )
    return [
        _execution_step(
            step,
            object_uid=object_uid,
            actor={"mode": "coordinated", "arms": ["left_arm", "right_arm"]},
            goal=normalized_goal,
            postcondition=postcondition,
        )
    ]


def _expand_press(step: Mapping[str, Any]) -> list[dict[str, Any]]:
    object_uid = _single_object(step, "press")
    goal = _goal(
        step,
        allowed={"interaction", "reference_object", "terminal_state"},
    )
    return [
        _execution_step(
            step,
            object_uid=object_uid,
            actor=_single_arm_actor(step),
            goal={
                "interaction": str(goal.get("interaction", "press")),
                "terminal_state": str(goal.get("terminal_state", "activated")),
                **(
                    {"reference_object": goal["reference_object"]}
                    if "reference_object" in goal
                    else {}
                ),
            },
            postcondition={
                "type": "pressed",
                "object": object_uid,
                "terminal_state": str(goal.get("terminal_state", "activated")),
            },
        )
    ]


def _expand_coordinated_place(step: Mapping[str, Any]) -> list[dict[str, Any]]:
    object_uid = _single_object(step, "coordinated_place")
    goal = _goal(
        step,
        allowed={"relation", "release", "support_object"},
    )
    support_object = _required_string(goal, "support_object", "coordinated_place")
    if support_object == object_uid:
        raise ValueError("coordinated_place requires two distinct objects.")
    relation = str(goal.get("relation", "on"))
    if relation not in {"on", "inside"}:
        raise ValueError("coordinated_place relation must be 'on' or 'inside'.")
    release = goal.get("release", True)
    if not isinstance(release, bool):
        raise ValueError("coordinated_place goal.release must be boolean.")
    return [
        _execution_step(
            step,
            object_uid=object_uid,
            actor={"mode": "coordinated", "arms": ["left_arm", "right_arm"]},
            goal={
                "support_object": support_object,
                "relation": relation,
                "release": release,
            },
            postcondition={
                "type": "coordinated_placed",
                "object": object_uid,
                "support_object": support_object,
                "relation": relation,
            },
        )
    ]


def _build_arrange_line_phases(
    step: Mapping[str, Any],
) -> tuple[PhaseTemplate, ...]:
    return (
        _pickup_phase(step),
        _move_phase(step, "staging"),
        _move_phase(step, "final"),
        *_release_retreat_home(step),
    )


def _build_single_arm_phases(
    step: Mapping[str, Any],
) -> tuple[PhaseTemplate, ...]:
    if step["operator"] not in _SINGLE_ARM_PHASE_OPERATORS:
        raise ValueError(f"Unexpected single-arm operator {step['operator']!r}.")
    phases: tuple[PhaseTemplate, ...] = (
        _pickup_phase(step),
        _move_phase(step),
    )
    if step["operator"] == "hold_hover":
        return phases + (
            PhaseTemplate(
                name="keep_holding",
                state_semantic=f"`{step['object']}` remains held",
                actions=(
                    ActionTemplate(
                        "MoveJoints",
                        {"kind": "joint_state", "source": "gripper_closed"},
                        build_motion_policy(),
                        control="hand",
                    ),
                ),
            ),
        )
    return phases + _release_retreat_home(step)


def _build_orient_object_phases(
    step: Mapping[str, Any],
) -> tuple[PhaseTemplate, ...]:
    """Rotate at a clearance waypoint before descending to the support."""
    upright = build_motion_policy(("orientation", "upright"))
    return (
        _pickup_phase(step, motion_policy=upright),
        _move_phase(
            step,
            "staging",
            motion_policy=upright,
        ),
        _move_phase(
            step,
            "final",
            motion_policy=upright,
        ),
        *_release_retreat_home(
            step,
            release_policy=upright,
            retreat_policy=upright,
        ),
    )


def _build_coordinated_transport_phases(
    step: Mapping[str, Any],
) -> tuple[PhaseTemplate, ...]:
    phases: tuple[PhaseTemplate, ...] = (
        PhaseTemplate(
            name="coordinated_transport",
            state_semantic=f"`{step['object']}` reaches its coordinated goal",
            actions=(
                ActionTemplate(
                    "CoordinatedPickment",
                    {
                        "kind": "coordinated_goal",
                        "semantic_step": step["id"],
                        "object": step["object"],
                        "payloads": deepcopy(step["goal"].get("payloads", [])),
                    },
                    build_motion_policy(),
                    control="coordinated",
                ),
            ),
        ),
    )
    if step["goal"]["terminal_behavior"] != "place":
        return phases
    return phases + (
        PhaseTemplate(
            name="dual_release",
            state_semantic="Both grippers release the transported object",
            actions=tuple(
                ActionTemplate(
                    "MoveJoints",
                    {
                        "kind": "joint_state",
                        "source": "gripper_open",
                        "coordinated_release_role": release_role,
                    },
                    build_motion_policy(),
                    control="hand",
                    actor={"mode": "required", "arm": arm},
                )
                for arm, release_role in (
                    ("left_arm", "participant"),
                    ("right_arm", "commit"),
                )
            ),
        ),
    )


def _build_press_phases(step: Mapping[str, Any]) -> tuple[PhaseTemplate, ...]:
    return (
        PhaseTemplate(
            name="press",
            state_semantic=f"`{step['object']}` has been pressed",
            actions=(
                ActionTemplate(
                    "Press",
                    {
                        "kind": "semantic_goal",
                        "semantic_step": step["id"],
                        "object": step["object"],
                        "interaction": "press",
                    },
                    build_motion_policy(),
                ),
            ),
        ),
    )


def _build_coordinated_place_phases(
    step: Mapping[str, Any],
) -> tuple[PhaseTemplate, ...]:
    return (
        PhaseTemplate(
            name="dual_pick_up",
            state_semantic=(
                f"`{step['object']}` is held by the left arm and "
                f"`{step['goal']['support_object']}` is held by the right arm"
            ),
            actions=(
                ActionTemplate(
                    "PickUp",
                    {
                        "kind": "object",
                        "object": step["object"],
                        "affordance": "antipodal",
                    },
                    build_motion_policy(),
                    actor={"mode": "required", "arm": "left_arm"},
                ),
                ActionTemplate(
                    "PickUp",
                    {
                        "kind": "object",
                        "object": step["goal"]["support_object"],
                        "affordance": "antipodal",
                    },
                    build_motion_policy(),
                    actor={"mode": "required", "arm": "right_arm"},
                ),
            ),
        ),
        PhaseTemplate(
            name="coordinated_place",
            state_semantic=(
                f"`{step['object']}` is coordinated with "
                f"`{step['goal']['support_object']}`"
            ),
            actions=(
                ActionTemplate(
                    "CoordinatedPlacement",
                    {
                        "kind": "coordinated_placement_goal",
                        "semantic_step": step["id"],
                        "placing_object": step["object"],
                        "support_object": step["goal"]["support_object"],
                    },
                    build_motion_policy(),
                    control="coordinated",
                ),
            ),
        ),
    )


def _pickup_phase(
    step: Mapping[str, Any],
    *,
    motion_policy: Mapping[str, Any] | None = None,
) -> PhaseTemplate:
    payloads = deepcopy(step["goal"].get("payloads", []))
    return PhaseTemplate(
        name="pick_up",
        state_semantic=f"Holding `{step['object']}`",
        actions=(
            ActionTemplate(
                "PickUp",
                {
                    "kind": "object",
                    "object": step["object"],
                    "affordance": "antipodal",
                    **({"payloads": payloads} if payloads else {}),
                },
                motion_policy or build_motion_policy(),
            ),
        ),
    )


def _move_phase(
    step: Mapping[str, Any],
    phase: str | None = None,
    *,
    motion_policy: Mapping[str, Any] | None = None,
) -> PhaseTemplate:
    target_binding = {
        "kind": "semantic_goal",
        "semantic_step": step["id"],
    }
    if phase is not None:
        target_binding["phase"] = phase
    payloads = deepcopy(step["goal"].get("payloads", []))
    if payloads:
        target_binding["payloads"] = payloads
    return PhaseTemplate(
        name=f"move_to_{phase or 'semantic_goal'}",
        state_semantic=f"`{step['object']}` is held at {phase or 'its semantic goal'}",
        actions=(
            ActionTemplate(
                "MoveHeldObject",
                target_binding,
                motion_policy or build_motion_policy(),
            ),
        ),
    )


def _release_retreat_home(
    step: Mapping[str, Any],
    *,
    release_policy: Mapping[str, Any] | None = None,
    retreat_policy: Mapping[str, Any] | None = None,
) -> tuple[PhaseTemplate, ...]:
    payloads = deepcopy(step["goal"].get("payloads", []))
    return (
        PhaseTemplate(
            name="release",
            state_semantic=f"`{step['object']}` is released at its semantic goal",
            actions=(
                ActionTemplate(
                    "Place",
                    {
                        "kind": "current_held_pose",
                        **({"payloads": payloads} if payloads else {}),
                    },
                    release_policy or build_motion_policy(),
                ),
            ),
        ),
        PhaseTemplate(
            name="retreat",
            state_semantic=f"The end effector retreats from `{step['object']}`",
            actions=(
                ActionTemplate(
                    "MoveEndEffector",
                    {
                        "kind": "policy_pose",
                        "source": "release",
                        "operation": "retreat",
                    },
                    retreat_policy or build_motion_policy(),
                ),
            ),
        ),
        PhaseTemplate(
            name="home",
            state_semantic="The selected arm returns to its initial state",
            actions=(
                ActionTemplate(
                    "MoveJoints",
                    {"kind": "joint_state", "source": "initial"},
                    build_motion_policy(),
                ),
            ),
        ),
    )


def _dual_arm_phase(
    name: str,
    state_semantic: str,
    action_class: str,
    target_binding: Mapping[str, Any],
    motion_policy: Mapping[str, Any],
    *,
    control: str = "arm",
) -> PhaseTemplate:
    return PhaseTemplate(
        name=name,
        state_semantic=state_semantic,
        actions=tuple(
            ActionTemplate(
                action_class,
                target_binding,
                motion_policy,
                control=control,
                actor={"mode": "required", "arm": arm},
            )
            for arm in ("left_arm", "right_arm")
        ),
    )


def _execution_step(
    parent: Mapping[str, Any],
    *,
    object_uid: str,
    actor: Mapping[str, Any],
    goal: Mapping[str, Any],
    postcondition: Mapping[str, Any],
    child_id: str | None = None,
) -> dict[str, Any]:
    return {
        "id": child_id or parent["id"],
        "parent_step_id": parent["id"],
        "operator": parent["operator"],
        "object": object_uid,
        "actor": deepcopy(dict(actor)),
        "goal": deepcopy(dict(goal)),
        "depends_on": [],
        "postcondition": deepcopy(dict(postcondition)),
        "edge_ids": [],
    }


def _single_object(step: Mapping[str, Any], operator: str) -> str:
    if "object" not in step:
        raise ValueError(f"{operator} requires one 'object', not 'objects'.")
    return str(step["object"])


def _collective_objects(
    step: Mapping[str, Any],
    operator: str,
    *,
    minimum: int,
) -> list[str]:
    if "objects" not in step:
        raise ValueError(f"{operator} requires an 'objects' list.")
    objects = [str(value) for value in step["objects"]]
    if len(objects) < minimum:
        raise ValueError(f"{operator} requires at least {minimum} object(s).")
    return objects


def _single_arm_actor(step: Mapping[str, Any]) -> dict[str, Any]:
    actor = deepcopy(dict(step["actor"]))
    if actor["mode"] == "coordinated":
        raise ValueError(f"{step['operator']} requires one arm, not coordinated arms.")
    if actor["mode"] == "required":
        arm = str(actor["arm"])
        if arm in {"left", "right"}:
            actor["arm"] = f"{arm}_arm"
    return actor


def _goal(step: Mapping[str, Any], *, allowed: set[str]) -> dict[str, Any]:
    goal = deepcopy(dict(step["goal"]))
    unknown = sorted(set(goal) - allowed)
    if unknown:
        raise ValueError(
            f"{step['operator']} goal contains unsupported fields: {unknown}."
        )
    return goal


def _normalize_payloads(
    value: Any,
    carrier_uid: str,
    operator: str,
) -> list[dict[str, str]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{operator} payloads must be a list.")
    if len(value) > 4:
        raise ValueError(f"{operator} supports at most four payloads.")
    result = []
    for index, payload in enumerate(value):
        item = {"object": payload} if isinstance(payload, str) else dict(payload)
        uid = item.get("object")
        slot = str(item.get("slot", "auto"))
        if not isinstance(uid, str) or not uid:
            raise ValueError(f"payloads[{index}] requires an object UID.")
        if uid == carrier_uid:
            raise ValueError(f"A {operator} carrier cannot be its own payload.")
        if slot not in {"left", "right", "center", "auto"}:
            raise ValueError(f"Unsupported payload slot {slot!r}.")
        result.append({"object": uid, "slot": slot})
    payload_uids = [item["object"] for item in result]
    if len(payload_uids) != len(set(payload_uids)):
        raise ValueError(f"{operator} payload objects must be unique.")
    return result


def _required_string(
    value: Mapping[str, Any],
    key: str,
    operator: str,
) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result.strip():
        raise ValueError(f"{operator} goal.{key} must be a non-empty string.")
    return result


def _orientation(
    goal: Mapping[str, Any],
    operator: str,
) -> tuple[str, str]:
    orientation_goal = str(goal.get("orientation_goal", "none"))
    orientation_axis = str(goal.get("orientation_axis", "none"))
    allowed_goals = {"none", "preserve", "upright", "lay_flat", "axis_align"}
    if orientation_goal not in allowed_goals:
        raise ValueError(
            f"{operator} orientation_goal {orientation_goal!r} is unsupported."
        )
    if orientation_axis not in {"none", "x", "y", "long_axis", "short_axis"}:
        raise ValueError(
            f"{operator} orientation_axis {orientation_axis!r} is unsupported."
        )
    if orientation_goal == "axis_align" and orientation_axis == "none":
        raise ValueError(f"{operator} axis_align requires an orientation_axis.")
    compile_orientation_constraint(goal)
    return orientation_goal, orientation_axis


def _orientation_extensions(goal: Mapping[str, Any]) -> dict[str, Any]:
    """Copy optional composable fields after operator-level validation."""
    return {
        key: deepcopy(goal[key])
        for key in ("orientation_constraint", "orientation_directed")
        if key in goal
    }
