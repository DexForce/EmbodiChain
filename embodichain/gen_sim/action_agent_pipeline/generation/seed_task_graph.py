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

"""Build and validate environment-independent semantic seed task graphs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import re
from typing import Any

from embodichain.gen_sim.action_agent_pipeline.generation.builder_protocols import (
    ArrangementSpecLike,
    RelativePlacementLike,
    RelativeSpecLike,
    StackingSpecLike,
)
from embodichain.gen_sim.action_agent_pipeline.generation.arrangement_intent import (
    _arrangement_order_is_constrained,
)

__all__ = [
    "SEED_TASK_GRAPH_SCHEMA_VERSION",
    "compile_seed_graph_metadata",
    "make_arrangement_seed_task_graph",
    "make_relative_seed_task_graph",
    "make_stacking_seed_task_graph",
    "seed_task_graph_hash",
    "validate_seed_task_graph",
]

SEED_TASK_GRAPH_SCHEMA_VERSION = "seed_task_graph_v1"

_ROUTES = {"arrangement_line", "object_manipulation", "stacking"}
_ACTOR_MODES = {"auto", "coordinated", "required"}
_ARM_NAMES = {"left_arm", "right_arm"}
_TOP_LEVEL_KEYS = {"program", "route", "schema_version", "steps", "task"}
_STEP_KEYS = {
    "actor",
    "depends_on",
    "goal",
    "id",
    "object",
    "operator",
    "postcondition",
}
_UNSAFE_ID_RE = re.compile(r"[^0-9a-z]+")
# Seed graphs deliberately stop before geometry and motion grounding. Rejecting
# these names recursively makes accidental leakage visible at generation time.
_GROUNDED_FIELD_NAMES = {
    "edge_ids",
    "edges",
    "high_position",
    "joint_state",
    "nodes",
    "position",
    "qpos",
    "release_position",
    "target_object_pose",
    "target_pose",
    "target_position",
    "target_qpos",
    "target_xy",
    "trajectory",
}


def make_relative_seed_task_graph(
    task_name: str,
    spec: RelativeSpecLike,
) -> dict[str, Any]:
    """Serialize normalized LLM manipulation steps without grounded actions."""
    steps: list[dict[str, Any]] = []
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
        steps.append(
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
    return _validated_seed_graph(
        {
            "schema_version": SEED_TASK_GRAPH_SCHEMA_VERSION,
            "task": task_name,
            "route": "object_manipulation",
            "program": spec.intent,
            "steps": steps,
        }
    )


def make_arrangement_seed_task_graph(
    task_name: str,
    spec: ArrangementSpecLike,
) -> dict[str, Any]:
    """Serialize the symbolic line order without generated slot coordinates."""
    order_is_constrained = _arrangement_order_is_constrained(
        spec.order_by,
        task_description=spec.task_description,
    )
    step_runtime_uids = [step.runtime_uid for step in spec.steps]
    runtime_uids = set(step_runtime_uids)
    if len(step_runtime_uids) != len(runtime_uids):
        raise ValueError("Arrangement seed objects must be distinct.")
    if order_is_constrained:
        configured_order = tuple(getattr(spec, "semantic_order", ()))
        ordered_uids = list(
            configured_order
            or (
                step.runtime_uid
                for step in sorted(spec.steps, key=lambda step: int(step.slot_index))
            )
        )
        if set(ordered_uids) != runtime_uids or len(ordered_uids) != len(runtime_uids):
            raise ValueError(
                "Arrangement semantic_order must contain every selected object once."
            )
    else:
        # A generic "form one row" goal owns membership, not an arbitrary LLM
        # array order. Sorting makes that unordered set hash deterministically.
        ordered_uids = sorted(runtime_uids)
    steps = [
        {
            "id": "s01_arrange_objects_in_line",
            "operator": "arrange_in_line",
            "object": "__arrangement__",
            "actor": {"mode": "auto"},
            "goal": {
                "layout": "line",
                "objects": ordered_uids,
                "order_constraint": ("ordered" if order_is_constrained else "free"),
                "axis": spec.axis,
                "anchor": spec.anchor,
                "order_by": spec.order_by,
                "order_direction": spec.order_direction,
            },
            "depends_on": [],
            "postcondition": {
                "type": (
                    "objects_in_ordered_line"
                    if order_is_constrained
                    else "objects_in_line"
                ),
            },
        }
    ]
    return _validated_seed_graph(
        {
            "schema_version": SEED_TASK_GRAPH_SCHEMA_VERSION,
            "task": task_name,
            "route": "arrangement_line",
            "program": "arrange_in_line",
            "steps": steps,
        }
    )


def make_stacking_seed_task_graph(
    task_name: str,
    spec: StackingSpecLike,
) -> dict[str, Any]:
    """Serialize bottom-to-top support relations without target coordinates."""
    ordered_steps = sorted(spec.steps, key=lambda step: int(step.layer_index))
    steps: list[dict[str, Any]] = []
    previous_step_id: str | None = None
    for index, step in enumerate(ordered_steps, start=1):
        step_id = _seed_step_id(index, "place_on_stack", step.runtime_uid)
        reference_object = step.support_runtime_uid or spec.anchor_runtime_uid
        steps.append(
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
                },
                "depends_on": (
                    [previous_step_id] if previous_step_id is not None else []
                ),
                "postcondition": {
                    "type": "stack_layer_supported",
                    "layer_index": int(step.layer_index),
                },
            }
        )
        previous_step_id = step_id
    return _validated_seed_graph(
        {
            "schema_version": SEED_TASK_GRAPH_SCHEMA_VERSION,
            "task": task_name,
            "route": "stacking",
            "program": "build_stack",
            "steps": steps,
        }
    )


def seed_task_graph_hash(seed_graph: Mapping[str, Any]) -> str:
    """Return the stable SHA-256 identity of one validated seed graph."""
    validate_seed_task_graph(seed_graph)
    canonical = json.dumps(
        seed_graph,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compile_seed_graph_metadata(
    task_graph: Mapping[str, Any],
    seed_graph: Mapping[str, Any],
) -> dict[str, Any]:
    """Attach immutable seed provenance to a deterministic atomic task graph."""
    validate_seed_task_graph(seed_graph)
    compiled = deepcopy(dict(task_graph))
    compiled["seed_graph_schema_version"] = seed_graph["schema_version"]
    compiled["seed_graph_hash"] = seed_task_graph_hash(seed_graph)
    return compiled


def validate_seed_task_graph(
    seed_graph: Mapping[str, Any],
    *,
    task_name: str | None = None,
    route: str | None = None,
) -> None:
    """Validate the symbolic seed schema and reject grounded data leakage."""
    if not isinstance(seed_graph, Mapping):
        raise TypeError("Seed task graph must be a mapping.")
    unknown_graph_keys = set(seed_graph) - _TOP_LEVEL_KEYS
    if unknown_graph_keys:
        raise ValueError(
            "Seed task graph contains unsupported top-level fields: "
            f"{sorted(unknown_graph_keys)}."
        )
    if seed_graph.get("schema_version") != SEED_TASK_GRAPH_SCHEMA_VERSION:
        raise ValueError(
            "Seed task graph schema_version must be "
            f"{SEED_TASK_GRAPH_SCHEMA_VERSION!r}."
        )
    graph_task = seed_graph.get("task")
    if not isinstance(graph_task, str) or not graph_task.strip():
        raise ValueError("Seed task graph requires a non-empty task name.")
    if task_name is not None and graph_task != task_name:
        raise ValueError(
            f"Seed task graph task {graph_task!r} does not match {task_name!r}."
        )
    graph_route = seed_graph.get("route")
    if graph_route not in _ROUTES:
        raise ValueError(f"Unsupported seed task graph route: {graph_route!r}.")
    if route is not None and graph_route != route:
        raise ValueError(
            f"Seed task graph route {graph_route!r} does not match {route!r}."
        )
    program = seed_graph.get("program")
    if not isinstance(program, str) or not program:
        raise ValueError("Seed task graph requires a non-empty program.")

    _reject_grounded_fields(seed_graph)
    steps = seed_graph.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("Seed task graph requires a non-empty steps list.")

    known_ids: set[str] = set()
    for index, step in enumerate(steps):
        if not isinstance(step, Mapping):
            raise TypeError(f"Seed task graph step {index} must be a mapping.")
        missing = _STEP_KEYS - set(step)
        if missing:
            raise ValueError(
                f"Seed task graph step {index} is missing: {sorted(missing)}."
            )
        unknown_step_keys = set(step) - _STEP_KEYS
        if unknown_step_keys:
            raise ValueError(
                f"Seed task graph step {index} contains unsupported fields: "
                f"{sorted(unknown_step_keys)}."
            )
        step_id = step["id"]
        if not isinstance(step_id, str) or not step_id:
            raise ValueError(f"Seed task graph step {index} requires an id.")
        if step_id in known_ids:
            raise ValueError(f"Duplicate seed task graph step id: {step_id!r}.")
        _validate_seed_actor(step_id, step["actor"])
        if not isinstance(step["goal"], Mapping):
            raise TypeError(f"Seed task graph step {step_id!r} goal must be a mapping.")
        if not isinstance(step["postcondition"], Mapping):
            raise TypeError(
                f"Seed task graph step {step_id!r} postcondition must be a mapping."
            )
        dependencies = step["depends_on"]
        if not isinstance(dependencies, list) or not all(
            isinstance(item, str) for item in dependencies
        ):
            raise TypeError(
                f"Seed task graph step {step_id!r} depends_on must be a string list."
            )
        unknown_dependencies = set(dependencies) - known_ids
        if unknown_dependencies:
            raise ValueError(
                f"Seed task graph step {step_id!r} depends on non-prior steps: "
                f"{sorted(unknown_dependencies)}."
            )
        known_ids.add(step_id)
    _validate_route_seed_steps(graph_route, program, steps)


def _validated_seed_graph(seed_graph: dict[str, Any]) -> dict[str, Any]:
    validate_seed_task_graph(seed_graph)
    return seed_graph


def _relative_seed_actor(
    placement: RelativePlacementLike,
    *,
    operator: str,
) -> dict[str, Any]:
    if operator == "coordinated_pickment":
        return {
            "mode": "coordinated",
            "arms": ["left_arm", "right_arm"],
        }
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


def _validate_seed_actor(step_id: str, actor: Any) -> None:
    if not isinstance(actor, Mapping) or actor.get("mode") not in _ACTOR_MODES:
        raise ValueError(f"Seed task graph step {step_id!r} has an invalid actor.")
    mode = actor["mode"]
    if mode == "required" and actor.get("arm") not in _ARM_NAMES:
        raise ValueError(
            f"Seed task graph step {step_id!r} required actor needs a valid arm."
        )
    if mode == "auto" and set(actor) != {"mode"}:
        raise ValueError(
            f"Seed task graph step {step_id!r} auto actor must remain unresolved."
        )
    if mode == "coordinated":
        arms = actor.get("arms")
        if not isinstance(arms, list) or set(arms) != _ARM_NAMES:
            raise ValueError(
                f"Seed task graph step {step_id!r} coordinated actor needs both arms."
            )


def _validate_route_seed_steps(
    route: str,
    program: str,
    steps: list[Mapping[str, Any]],
) -> None:
    """Validate the exact symbolic vocabulary owned by each route."""
    if route == "arrangement_line":
        if program != "arrange_in_line":
            raise ValueError("Arrangement seed program must be 'arrange_in_line'.")
        _validate_arrangement_seed_steps(steps)
        return
    if route == "stacking":
        if program != "build_stack":
            raise ValueError("Stacking seed program must be 'build_stack'.")
        _validate_stacking_seed_steps(steps)
        return
    if program not in {
        "coordinated_pickment",
        "hold_hover",
        "place_relative",
    }:
        raise ValueError(f"Unsupported relative seed program: {program!r}.")
    _validate_relative_seed_steps(steps)


def _validate_arrangement_seed_steps(steps: list[Mapping[str, Any]]) -> None:
    if len(steps) != 1:
        raise ValueError("Arrangement seed graph requires exactly one step.")
    step = steps[0]
    if (
        step["operator"] != "arrange_in_line"
        or step["object"] != "__arrangement__"
        or step["actor"] != {"mode": "auto"}
        or step["depends_on"] != []
    ):
        raise ValueError("Arrangement seed graph contains invalid step semantics.")
    _require_exact_mapping_keys(
        step["goal"],
        {
            "anchor",
            "axis",
            "layout",
            "objects",
            "order_by",
            "order_constraint",
            "order_direction",
        },
        context="Arrangement seed goal",
    )
    if step["goal"]["layout"] != "line":
        raise ValueError("Arrangement seed goal layout must be 'line'.")
    objects = step["goal"]["objects"]
    if (
        not isinstance(objects, list)
        or not objects
        or not all(isinstance(uid, str) and uid for uid in objects)
    ):
        raise ValueError("Arrangement seed goal objects must be a non-empty UID list.")
    if len(objects) != len(set(objects)):
        raise ValueError("Arrangement seed goal objects must be distinct.")
    order_constraint = step["goal"]["order_constraint"]
    if order_constraint not in {"free", "ordered"}:
        raise ValueError(
            "Arrangement seed goal order_constraint must be 'free' or 'ordered'."
        )
    expected_postcondition = {
        "type": (
            "objects_in_ordered_line"
            if order_constraint == "ordered"
            else "objects_in_line"
        )
    }
    if step["postcondition"] != expected_postcondition:
        raise ValueError("Arrangement seed graph has an invalid postcondition.")
    if order_constraint == "free":
        if objects != sorted(objects):
            raise ValueError(
                "Free-order arrangement seed objects must use canonical UID order."
            )
        if step["goal"]["order_by"] != "explicit":
            raise ValueError(
                "Free-order arrangement seed must use order_by='explicit'."
            )


def _validate_stacking_seed_steps(steps: list[Mapping[str, Any]]) -> None:
    for step in steps:
        if step["operator"] != "place_on_stack" or step["actor"] != {"mode": "auto"}:
            raise ValueError("Stacking seed graph contains invalid step semantics.")
        _require_exact_mapping_keys(
            step["goal"],
            {
                "layer_index",
                "reference_object",
                "reference_state",
                "relation",
                "stack_mode",
            },
            context=f"Stacking seed goal {step['id']!r}",
        )
        if step["goal"]["relation"] != "on":
            raise ValueError("Stacking seed goal relation must be 'on'.")
        expected_postcondition = {
            "type": "stack_layer_supported",
            "layer_index": step["goal"]["layer_index"],
        }
        if step["postcondition"] != expected_postcondition:
            raise ValueError(
                f"Stacking seed step {step['id']!r} has an invalid postcondition."
            )


def _validate_relative_seed_steps(steps: list[Mapping[str, Any]]) -> None:
    supported_operators = {
        "coordinated_pickment",
        "hold_hover",
        "place_relative",
    }
    for step in steps:
        operator = step["operator"]
        if operator not in supported_operators:
            raise ValueError(
                f"Relative seed step {step['id']!r} has unsupported operator "
                f"{operator!r}."
            )
        goal_keys = {
            "orientation_axis",
            "orientation_goal",
            "reference_object",
            "reference_state",
            "relation",
        }
        if "orientation_reference_object" in step["goal"]:
            goal_keys.add("orientation_reference_object")
        if operator == "coordinated_pickment":
            goal_keys.update({"direction", "terminal_behavior"})
        _require_exact_mapping_keys(
            step["goal"],
            goal_keys,
            context=f"Relative seed goal {step['id']!r}",
        )
        expected_postcondition = {
            "type": "semantic_goal",
            "operator": operator,
            "relation": step["goal"]["relation"],
        }
        if step["postcondition"] != expected_postcondition:
            raise ValueError(
                f"Relative seed step {step['id']!r} has an invalid postcondition."
            )


def _require_exact_mapping_keys(
    value: Mapping[str, Any],
    expected: set[str],
    *,
    context: str,
) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{context} fields must be {sorted(expected)}; got {sorted(actual)}."
        )


def _reject_grounded_fields(value: Any, path: str = "seed_task_graph") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            if key_text in _GROUNDED_FIELD_NAMES:
                raise ValueError(
                    f"Seed task graph must not contain grounded field "
                    f"{path}.{key_text}."
                )
            _reject_grounded_fields(child, f"{path}.{key_text}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            _reject_grounded_fields(child, f"{path}[{index}]")
