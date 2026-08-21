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

"""Deterministically lower a route-free TaskAgent into an action DAG."""

from __future__ import annotations

import re
from collections import deque
from collections.abc import Collection, Mapping, Sequence
from copy import deepcopy
from typing import Any

from embodichain.gen_sim.action_engine.capabilities import (
    ActionTemplate,
    CapabilityRegistry,
    PhaseTemplate,
    build_default_registry,
)
from embodichain.gen_sim.action_engine.domain import (
    EXECUTION_PROGRAM_SCHEMA,
    MOTION_POLICY_VERSION,
    validate_execution_program,
    validate_task_agent,
)

__all__ = ["compile_task_agent"]

_UNSAFE_ID_RE = re.compile(r"[^0-9a-z]+")


def compile_task_agent(
    program: Mapping[str, Any],
    *,
    registry: CapabilityRegistry | None = None,
    known_objects: Collection[str] | None = None,
) -> dict[str, Any]:
    """Compile semantic steps into a complete coordinate-free action DAG.

    Compilation never reads simulator state and never calls an LLM. Collective
    operators such as ``arrange_line`` and ``build_stack`` expand into one
    execution semantic step per object, while dependencies are rewritten to
    point at the terminal expanded step of each parent operation.

    Args:
        program: Valid or validation-ready TaskAgent mapping.
        registry: Optional capability registry for controlled extensions.
        known_objects: Optional runtime scene UIDs used for pre-simulator
            object-reference validation.

    Returns:
        A validated ``action_engine_execution_graph_v1`` mapping.
    """
    task_agent = validate_task_agent(program, known_objects=known_objects)
    capabilities = registry or build_default_registry()
    ordered_task_steps = _stable_topological_steps(task_agent["semantic_steps"])

    expanded_by_parent: dict[str, list[dict[str, Any]]] = {}
    all_expanded_ids: set[str] = set()
    for task_step in ordered_task_steps:
        definition = capabilities.operator(task_step["operator"])
        expanded = definition.expand(task_step)
        if not expanded:
            raise ValueError(
                f"Operator {task_step['operator']!r} produced no execution steps."
            )
        for child in expanded:
            child_id = str(child.get("id", ""))
            if not child_id or child_id in all_expanded_ids:
                raise ValueError(
                    f"Operator {task_step['operator']!r} produced duplicate or "
                    f"empty execution step ID {child_id!r}."
                )
            all_expanded_ids.add(child_id)
        expanded_by_parent[task_step["id"]] = expanded

    # Operator expansion validates each step's shape first, so held-state
    # diagnostics never mask a more direct capability-contract error.
    _validate_held_state_contract(ordered_task_steps)

    terminal_children: dict[str, list[str]] = {}
    for task_step in ordered_task_steps:
        definition = capabilities.operator(task_step["operator"])
        children = expanded_by_parent[task_step["id"]]
        terminal_children[task_step["id"]] = (
            [child["id"] for child in children]
            if definition.expansion_topology == "parallel_children"
            else [children[-1]["id"]]
        )
    expanded_steps: list[dict[str, Any]] = []
    for task_step in ordered_task_steps:
        definition = capabilities.operator(task_step["operator"])
        children = expanded_by_parent[task_step["id"]]
        parent_dependencies = [
            child_id
            for parent_id in task_step["depends_on"]
            for child_id in terminal_children[parent_id]
        ]
        for index, child in enumerate(children):
            child["depends_on"] = (
                parent_dependencies
                if index == 0 or definition.expansion_topology == "parallel_children"
                else [children[index - 1]["id"]]
            )
            expanded_steps.append(child)

    phases_by_step: dict[str, tuple[PhaseTemplate, ...]] = {}
    for step in expanded_steps:
        definition = capabilities.operator(step["operator"])
        phases = tuple(definition.build_phases(step))
        if not phases or any(not phase.actions for phase in phases):
            raise ValueError(
                f"Operator {step['operator']!r} produced an empty action phase."
            )
        for phase in phases:
            for action in phase.actions:
                capabilities.validate_action_template(action)
        phases_by_step[step["id"]] = phases

    graph = _build_graph(
        task=task_agent["task"],
        goal_description=task_agent["goal"],
        semantic_steps=expanded_steps,
        phases_by_step=phases_by_step,
    )
    graph["allocation_groups"] = _merge_allocation_groups(
        _compile_task_allocation_groups(
            task_agent["allocation_groups"],
            expanded_by_parent,
        ),
        _derive_allocation_groups(
            expanded_steps,
            phases_by_step,
        ),
    )
    return validate_execution_program(graph)


def _compile_task_allocation_groups(
    groups: Sequence[Mapping[str, Any]],
    expanded_by_parent: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for group in groups:
        members = [
            expanded_by_parent[parent_id][0]["id"]
            for parent_id in group["semantic_step_ids"]
        ]
        result.append(
            {
                "id": group["id"],
                "semantic_step_ids": members,
                "arm_constraint": "distinct_arms",
                "execution_policy": "parallel_if_feasible",
                "parallel_action_classes": ["PickUp"],
                "workspace_policy": "shared_target_serial",
            }
        )
    return result


def _merge_allocation_groups(
    explicit: Sequence[Mapping[str, Any]],
    derived: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result = [deepcopy(dict(group)) for group in explicit]
    assigned = {step_id for group in result for step_id in group["semantic_step_ids"]}
    used_ids = {group["id"] for group in result}
    for group in derived:
        if set(group["semantic_step_ids"]) & assigned:
            continue
        candidate = deepcopy(dict(group))
        base_id = candidate["id"]
        suffix = 2
        while candidate["id"] in used_ids:
            candidate["id"] = f"{base_id}_{suffix}"
            suffix += 1
        result.append(candidate)
        used_ids.add(candidate["id"])
        assigned.update(candidate["semantic_step_ids"])
    return result


def _build_graph(
    *,
    task: str,
    goal_description: str,
    semantic_steps: list[dict[str, Any]],
    phases_by_step: Mapping[str, tuple[PhaseTemplate, ...]],
) -> dict[str, Any]:
    start_id = "v0_start"
    goal_id = "v_goal"
    dependents: dict[str, list[str]] = {step["id"]: [] for step in semantic_steps}
    for step in semantic_steps:
        for dependency in step["depends_on"]:
            dependents[dependency].append(step["id"])

    terminal_node = {
        step["id"]: (
            f"v_{_slug(step['id'])}_done" if dependents[step["id"]] else goal_id
        )
        for step in semantic_steps
    }
    nodes: list[dict[str, str]] = [
        {
            "id": start_id,
            "semantic": "Initial state before executing the semantic action DAG",
        }
    ]
    node_ids = {start_id}
    edges: list[dict[str, Any]] = []
    final_edge_by_step: dict[str, str] = {}

    def add_node(node_id: str, semantic: str) -> None:
        if node_id in node_ids or node_id == goal_id:
            return
        node_ids.add(node_id)
        nodes.append({"id": node_id, "semantic": semantic})

    for step in semantic_steps:
        phases = phases_by_step[step["id"]]
        if step["depends_on"]:
            source_id = terminal_node[step["depends_on"][0]]
        else:
            source_id = start_id
        add_node(
            source_id,
            f"Dependencies for semantic step `{step['id']}` are complete",
        )

        step_edge_ids: list[str] = []
        previous_edge_id: str | None = None
        for phase_index, phase in enumerate(phases, start=1):
            is_last = phase_index == len(phases)
            target_id = (
                terminal_node[step["id"]]
                if is_last
                else f"v_{_slug(step['id'])}_{phase_index:02d}_{_slug(phase.name)}"
            )
            add_node(target_id, phase.state_semantic)
            edge_id = f"e{len(edges) + 1:03d}_{_slug(step['id'])}_{_slug(phase.name)}"
            edge_dependencies = (
                [final_edge_by_step[item] for item in step["depends_on"]]
                if previous_edge_id is None
                else [previous_edge_id]
            )
            actions = [
                _materialize_action(action, default_actor=step["actor"])
                for action in phase.actions
            ]
            edges.append(
                {
                    "id": edge_id,
                    "source": source_id,
                    "target": target_id,
                    "semantic_step_id": step["id"],
                    "actions": actions,
                    "depends_on": edge_dependencies,
                    "resources": _edge_resources(step, actions),
                }
            )
            step_edge_ids.append(edge_id)
            previous_edge_id = edge_id
            source_id = target_id
        step["edge_ids"] = step_edge_ids
        final_edge_by_step[step["id"]] = step_edge_ids[-1]

    nodes.append(
        {
            "id": goal_id,
            "semantic": "All required semantic steps have reached their postconditions",
        }
    )
    return {
        "schema_version": EXECUTION_PROGRAM_SCHEMA,
        "task": task,
        "goal_description": goal_description,
        "start": start_id,
        "goal": goal_id,
        "nodes": nodes,
        "edges": edges,
        "semantic_steps": semantic_steps,
        "allocation_groups": [],
        "motion_policy_version": MOTION_POLICY_VERSION,
    }


def _materialize_action(
    template: ActionTemplate,
    *,
    default_actor: Mapping[str, Any],
) -> dict[str, Any]:
    actor = template.actor if template.actor is not None else default_actor
    return {
        "atomic_action_class": template.atomic_action_class,
        "actor": deepcopy(dict(actor)),
        "control": template.control,
        "target_binding": deepcopy(dict(template.target_binding)),
        "motion_policy": deepcopy(dict(template.motion_policy)),
    }


def _edge_resources(
    step: Mapping[str, Any],
    actions: Sequence[Mapping[str, Any]],
) -> list[str]:
    resources = {f"object:{step['object']}"}
    reference = step["goal"].get("reference_object")
    support = step["goal"].get("support_object")

    for action in actions:
        actor = action["actor"]
        if actor["mode"] == "auto":
            resources.add("arm:auto")
        elif actor["mode"] == "required":
            resources.add(f"arm:{actor['arm']}")
        else:
            resources.update(f"arm:{arm}" for arm in actor["arms"])

        binding = action["target_binding"]
        for key in ("object", "placing_object", "support_object"):
            object_uid = binding.get(key)
            if isinstance(object_uid, str) and object_uid:
                resources.add(f"object:{object_uid}")
        for payload in binding.get("payloads", []):
            object_uid = (
                payload.get("object") if isinstance(payload, Mapping) else payload
            )
            if isinstance(object_uid, str) and object_uid:
                resources.add(f"object:{object_uid}")

    action_classes = {action["atomic_action_class"] for action in actions}
    uses_goal_workspace = bool(
        action_classes
        & {
            "MoveHeldObject",
            "MoveEndEffector",
            "Place",
            "CoordinatedPickment",
            "CoordinatedPlacement",
            "Press",
        }
    )
    if isinstance(reference, str) and reference and uses_goal_workspace:
        resources.add(f"workspace:{reference}")
    elif isinstance(support, str) and support:
        # Passive supports such as a table may be shared by independent
        # pickups. Only a coordinated placement manipulates and owns its
        # support object throughout the semantic step.
        if step["operator"] == "coordinated_place":
            resources.add(f"object:{support}")
        if uses_goal_workspace:
            resources.add(f"workspace:{support}")

    if action_classes & {
        "MoveHeldObject",
        "Place",
        "CoordinatedPickment",
        "CoordinatedPlacement",
        "Press",
    }:
        if step["operator"] == "arrange_line":
            resources.add("workspace:table")
        elif reference is None and support is None:
            resources.add("workspace:world")
    return sorted(resources)


def _derive_allocation_groups(
    semantic_steps: Sequence[Mapping[str, Any]],
    phases_by_step: Mapping[str, tuple[PhaseTemplate, ...]],
) -> list[dict[str, Any]]:
    """Declare only explicit, independent distinct-arm pickup pairs."""
    groups: list[dict[str, Any]] = []
    ancestor_ids = _ancestor_sets(semantic_steps)
    used_steps: set[str] = set()
    for index, first in enumerate(semantic_steps):
        if first["id"] in used_steps or not _starts_with_pickup(
            phases_by_step[first["id"]]
        ):
            continue
        for second in semantic_steps[index + 1 :]:
            if second["id"] in used_steps or not _starts_with_pickup(
                phases_by_step[second["id"]]
            ):
                continue
            if not _actors_request_distinct_arms(
                first["actor"],
                second["actor"],
            ):
                continue
            if (
                second["id"] in ancestor_ids[first["id"]]
                or first["id"] in ancestor_ids[second["id"]]
            ):
                continue
            if first["object"] == second["object"]:
                continue
            groups.append(
                {
                    "id": f"g{len(groups) + 1:02d}_distinct_arms",
                    "semantic_step_ids": [first["id"], second["id"]],
                    "arm_constraint": "distinct_arms",
                    "execution_policy": "parallel_if_feasible",
                    "parallel_action_classes": ["PickUp"],
                    "workspace_policy": "shared_target_serial",
                }
            )
            used_steps.update({first["id"], second["id"]})
            break
    return groups


def _validate_held_state_contract(
    semantic_steps: Sequence[Mapping[str, Any]],
) -> None:
    """Validate persistent object ownership and required-arm reservations.

    ``hold_hover`` is terminal behavior for its object and reserves the
    selected arm through task completion. Unrelated downstream work remains
    legal because runtime can assign it to another free arm. Action Engine v1
    does not expose a "continue with currently held object" operator, however,
    so any second step that references the held object would imply an unsafe
    pickup, handover, or use of a moving reference. Planner-produced
    hold/place pairs are fused before this boundary.
    """
    ancestors = _ancestor_sets(semantic_steps)
    for hold in semantic_steps:
        terminal_coordinated = (
            hold["operator"] == "coordinated_transport"
            and hold["goal"].get("terminal_behavior", "hold") == "hold"
        )
        if hold["operator"] != "hold_hover" and not terminal_coordinated:
            continue
        hold_id = hold["id"]
        held_object = hold["object"]
        for other in semantic_steps:
            other_id = other["id"]
            if other_id == hold_id or other_id in ancestors[hold_id]:
                continue
            if held_object in _step_object_references(other):
                raise ValueError(
                    f"hold_hover step {hold_id!r} reserves object "
                    f"{held_object!r} through task completion, but step "
                    f"{other_id!r} also references it."
                )
            hold_actor = hold["actor"]
            other_actor = other["actor"]
            if terminal_coordinated:
                raise ValueError(
                    f"Terminal coordinated step {hold_id!r} reserves both arms, "
                    f"but step {other_id!r} is not an ancestor."
                )
            if hold_actor["mode"] != "required":
                continue
            reserved_arm = _canonical_arm(hold_actor["arm"])
            conflicts = other_actor["mode"] == "coordinated" or (
                other_actor["mode"] == "required"
                and _canonical_arm(other_actor["arm"]) == reserved_arm
            )
            if conflicts:
                raise ValueError(
                    f"hold_hover step {hold_id!r} reserves arm "
                    f"{reserved_arm!r}, but non-ancestor step {other_id!r} "
                    "also requires it."
                )


def _step_object_references(step: Mapping[str, Any]) -> set[str]:
    """Return object UIDs whose ownership or workspace a step may require."""
    result = {step["object"]} if "object" in step else set(step.get("objects", ()))
    goal = step["goal"]
    for key in (
        "anchor",
        "orientation_reference_object",
        "reference_object",
        "support_object",
    ):
        value = goal.get(key)
        if isinstance(value, str):
            result.add(value)
    for payload in goal.get("payloads", []):
        value = payload.get("object") if isinstance(payload, Mapping) else payload
        if isinstance(value, str):
            result.add(value)
    return result


def _ancestor_sets(
    semantic_steps: Sequence[Mapping[str, Any]],
) -> dict[str, set[str]]:
    direct = {step["id"]: set(step["depends_on"]) for step in semantic_steps}
    ancestors: dict[str, set[str]] = {}
    for step in semantic_steps:
        pending = list(direct[step["id"]])
        result: set[str] = set()
        while pending:
            dependency = pending.pop()
            if dependency in result:
                continue
            result.add(dependency)
            pending.extend(direct[dependency])
        ancestors[step["id"]] = result
    return ancestors


def _starts_with_pickup(phases: Sequence[PhaseTemplate]) -> bool:
    return bool(
        phases
        and phases[0].actions
        and phases[0].actions[0].atomic_action_class == "PickUp"
    )


def _actors_request_distinct_arms(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
) -> bool:
    """Return whether actors explicitly request a distinct-arm assignment."""
    first_group = first.get("allocation_group")
    same_group = first_group is not None and first_group == second.get(
        "allocation_group"
    )
    required_opposite = (
        first["mode"] == "required"
        and second["mode"] == "required"
        and _canonical_arm(first["arm"]) != _canonical_arm(second["arm"])
    )
    if same_group and not required_opposite:
        both_required = first["mode"] == second["mode"] == "required"
        if both_required:
            raise ValueError(
                f"Allocation group {first_group!r} requires distinct arms, "
                "but both steps require the same arm."
            )
    return same_group or required_opposite


def _canonical_arm(value: Any) -> str:
    arm = str(value)
    return f"{arm}_arm" if arm in {"left", "right"} else arm


def _stable_topological_steps(
    semantic_steps: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    original = [deepcopy(dict(step)) for step in semantic_steps]
    order = {step["id"]: index for index, step in enumerate(original)}
    by_id = {step["id"]: step for step in original}
    indegree = {step["id"]: len(step["depends_on"]) for step in original}
    dependents: dict[str, list[str]] = {step["id"]: [] for step in original}
    for step in original:
        for dependency in step["depends_on"]:
            dependents[dependency].append(step["id"])

    ready = deque(
        sorted(
            (step_id for step_id, degree in indegree.items() if degree == 0),
            key=order.__getitem__,
        )
    )
    result: list[dict[str, Any]] = []
    while ready:
        step_id = ready.popleft()
        result.append(by_id[step_id])
        newly_ready: list[str] = []
        for dependent in dependents[step_id]:
            indegree[dependent] -= 1
            if indegree[dependent] == 0:
                newly_ready.append(dependent)
        ready.extend(sorted(newly_ready, key=order.__getitem__))
    return result


def _slug(value: Any) -> str:
    slug = _UNSAFE_ID_RE.sub("_", str(value).lower()).strip("_")
    return slug[:64].rstrip("_") or "step"
