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

from __future__ import annotations

import importlib
import re
from collections.abc import Mapping
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any

from embodichain.utils.llm_json import extract_json_object

__all__ = [
    "compile_agent_graph_from_file",
    "compile_agent_graph_spec",
    "expand_recovery_spec",
    "load_agent_graph_bundle",
    "normalize_recovery_spec",
]

MONITOR_TYPE_ALIASES = {
    "object_moved": "object_moved",
    "object_shifted": "object_moved",
    "object_displaced": "object_moved",
    "object_pose_changed": "object_moved",
    "moved_object": "object_moved",
    "hold_lost": "hold_lost",
    "object_held": "hold_lost",
    "grasp_lost": "hold_lost",
    "object_dropped": "hold_lost",
    "not_held": "hold_lost",
    "slip": "hold_lost",
    "slipped": "hold_lost",
}

RECOVERY_STEP_TYPE_ALIASES = {
    "retry": "retry_failed_edge",
    "retry_edge": "retry_failed_edge",
    "retry_failed_edge": "retry_failed_edge",
    "replay": "replay_edge",
    "repeat_edge": "replay_edge",
    "replay_edge": "replay_edge",
    "regrasp": "regrasp",
    "re_grasp": "regrasp",
    "grasp_again": "regrasp",
    "regrasp_both": "regrasp_both",
    "dual_regrasp": "regrasp_both",
    "regrasp_objects": "regrasp_both",
    "regrasp_all": "regrasp_both",
    "action": "action",
    "atomic_action": "action",
}


def load_agent_graph_bundle(path: str | Path) -> dict[str, Any]:
    """Load a compiled graph JSON bundle from disk."""
    return extract_json_object(Path(path).read_text(encoding="utf-8"))


def compile_agent_graph_from_file(
    path: str | Path,
    *,
    env: Any = None,
    graph_cls: type | None = None,
    action_module: Any = None,
    monitor_module: Any = None,
) -> Any:
    """Compile a graph JSON bundle from disk into an executable graph."""
    bundle = load_agent_graph_bundle(path)
    task_graph = bundle.get("task_graph", bundle)
    recovery_graph = bundle.get("recovery_graph")
    return compile_agent_graph_spec(
        task_graph,
        recovery_graph,
        env=env,
        graph_cls=graph_cls,
        action_module=action_module,
        monitor_module=monitor_module,
    )


def compile_agent_graph_spec(
    task_graph: str | Mapping[str, Any],
    recovery_graph: str | Mapping[str, Any] | None = None,
    *,
    env: Any = None,
    graph_cls: type | None = None,
    action_module: Any = None,
    monitor_module: Any = None,
) -> Any:
    """Compile nominal and explicit recovery JSON graphs into ``AgentTaskGraph``.

    Args:
        task_graph: Nominal graph spec with ``nodes`` and ``edges``.
        recovery_graph: Optional compiled recovery graph with ``recovery_nodes``,
            ``recovery_edges``, and ``recovery_branches``.
        env: Runtime environment bound into monitor partials.
        graph_cls: Optional graph class override for tests.
        action_module: Optional atomic action namespace override.
        monitor_module: Optional monitor namespace override.

    Returns:
        Executable ``AgentTaskGraph`` instance.
    """
    task_spec = extract_json_object(task_graph)
    recovery_spec = extract_json_object(
        recovery_graph or _empty_recovery_graph(task_spec)
    )
    _reject_legacy_recovery_schema(recovery_spec)
    nominal_node_ids, nominal_edge_ids = _validate_task_spec(task_spec)
    graph_cls, action_module, monitor_module = _resolve_runtime(
        graph_cls=graph_cls,
        action_module=action_module,
        monitor_module=monitor_module,
    )

    graph = graph_cls(
        start=task_spec["start"],
        goal=task_spec["goal"],
        max_transitions=int(task_spec.get("max_transitions", 1000)),
    )

    for node in task_spec.get("nodes", []):
        graph.add_node(node["id"], node.get("semantic", ""))

    for edge in task_spec.get("edges", []):
        graph.add_edge(
            edge["id"],
            edge["source"],
            edge["target"],
            left_arm_action=_compile_action(edge.get("left_arm_action"), action_module),
            right_arm_action=_compile_action(
                edge.get("right_arm_action"), action_module
            ),
        )

    recovery_node_ids = _add_recovery_nodes(graph, recovery_spec, nominal_node_ids)
    recovery_edge_ids = _add_recovery_edges(
        graph,
        recovery_spec,
        action_module=action_module,
        defined_node_ids=nominal_node_ids | recovery_node_ids,
    )
    _add_recovery_branches(
        graph,
        recovery_spec,
        env=env,
        monitor_module=monitor_module,
        nominal_edge_ids=nominal_edge_ids,
        recovery_edge_ids=recovery_edge_ids,
    )

    return graph


def expand_recovery_spec(
    task_graph: str | Mapping[str, Any],
    recovery_spec: str | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Expand a lightweight recovery authoring spec into an explicit graph.

    The LLM-facing schema should stay compact: each binding says which nominal
    edge to monitor, which failure monitor to use, and which reusable recovery
    policy steps to apply. This compiler expands those bindings into concrete
    ``recovery_nodes``, ``recovery_edges``, and ``recovery_branches`` that the
    runtime can execute.
    """
    task_spec = extract_json_object(task_graph)
    spec, issues = normalize_recovery_spec(task_spec, recovery_spec or {})
    if issues:
        issue_text = "; ".join(issues)
        raise ValueError(
            f"Recovery spec needs canonicalization before expansion: {issue_text}"
        )
    _reject_compiled_recovery_schema_in_authoring_spec(spec)

    expanded = {
        "task": spec.get("task", task_spec.get("task", "")),
        "recovery_nodes": [],
        "recovery_edges": [],
        "recovery_branches": [],
    }

    _validate_recovery_authoring_spec(spec)
    bindings = list(spec.get("recovery_bindings", []))
    if not bindings:
        return expanded
    nominal_edges = {edge["id"]: edge for edge in task_spec.get("edges", [])}
    used_node_ids = {node["id"] for node in task_spec.get("nodes", [])}
    used_edge_ids = set(nominal_edges)

    for binding_index, binding in enumerate(bindings):
        _expand_recovery_binding(
            expanded,
            binding,
            binding_index=binding_index,
            nominal_edges=nominal_edges,
            used_node_ids=used_node_ids,
            used_edge_ids=used_edge_ids,
        )

    return expanded


def normalize_recovery_spec(
    task_graph: str | Mapping[str, Any],
    recovery_spec: str | Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Canonicalize a lightweight recovery spec before graph expansion.

    The compiler keeps common aliases and small omissions deterministic, while
    returning unresolved semantic gaps for CompileAgent to canonicalize with one
    optional LLM call.
    """
    task_spec = extract_json_object(task_graph)
    spec = extract_json_object(recovery_spec or {})
    _reject_compiled_recovery_schema_in_authoring_spec(spec)

    issues: list[str] = []
    nominal_edges = {edge["id"]: edge for edge in task_spec.get("edges", [])}
    bindings = spec.get("recovery_bindings", spec.get("bindings", []))
    if not isinstance(bindings, list):
        return (
            {
                "task": spec.get("task", task_spec.get("task", "")),
                "recovery_bindings": [],
            },
            ["recovery_bindings must be a list"],
        )

    normalized = {
        "task": spec.get("task", task_spec.get("task", "")),
        "recovery_bindings": [],
    }
    allowed_top_keys = {"task", "recovery_bindings", "bindings"}
    unknown_top_keys = set(spec) - allowed_top_keys
    if unknown_top_keys:
        issues.append(f"unknown top-level keys: {', '.join(sorted(unknown_top_keys))}")

    for binding_index, binding in enumerate(bindings):
        if not isinstance(binding, Mapping):
            issues.append(f"binding {binding_index} must be an object")
            continue
        normalized_binding, binding_issues = _normalize_recovery_binding(
            binding,
            binding_index=binding_index,
            nominal_edges=nominal_edges,
        )
        if normalized_binding is not None:
            normalized["recovery_bindings"].append(normalized_binding)
        issues.extend(binding_issues)

    return normalized, issues


def _normalize_recovery_binding(
    binding: Mapping[str, Any],
    *,
    binding_index: int,
    nominal_edges: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any] | None, list[str]]:
    issues: list[str] = []
    edge_id = binding.get("edge_id", binding.get("edge"))
    if not edge_id:
        return None, [f"binding {binding_index} is missing edge_id"]
    edge_id = str(edge_id)
    monitored_edge = nominal_edges.get(edge_id)
    if monitored_edge is None:
        issues.append(f"binding {binding_index} references unknown edge_id '{edge_id}'")

    monitors, monitor_issues = _normalize_monitor_list(binding, monitored_edge)
    recovery_steps, recovery_issues = _normalize_recovery_steps(
        binding,
        monitored_edge,
        monitors,
    )
    issues.extend(monitor_issues)
    issues.extend(recovery_issues)

    merge = _normalize_merge(binding.get("merge", "target"))
    if merge is None:
        issues.append(
            f"binding {binding_index} has unsupported merge value "
            f"'{binding.get('merge')}'"
        )
        merge = "target"

    allowed_binding_keys = {
        "edge_id",
        "edge",
        "failure_name",
        "failure",
        "monitors",
        "monitor",
        "monitor_intent",
        "recovery",
        "recovery_steps",
        "recovery_intent",
        "policy",
        "merge",
        "repeat_until_success",
        "repeat",
    }
    unknown_binding_keys = set(binding) - allowed_binding_keys
    if unknown_binding_keys:
        issues.append(
            f"binding {binding_index} has unknown keys: "
            f"{', '.join(sorted(unknown_binding_keys))}"
        )

    return (
        {
            "edge_id": edge_id,
            "failure_name": str(
                binding.get(
                    "failure_name",
                    binding.get("failure", f"failure_{binding_index}"),
                )
            ),
            "monitors": monitors,
            "recovery": recovery_steps,
            "merge": merge,
            "repeat_until_success": bool(
                binding.get(
                    "repeat_until_success",
                    binding.get("repeat", True),
                )
            ),
        },
        issues,
    )


def _normalize_monitor_list(
    binding: Mapping[str, Any],
    monitored_edge: Mapping[str, Any] | None,
) -> tuple[list[dict[str, Any]], list[str]]:
    monitors = binding.get("monitors", binding.get("monitor"))
    issues: list[str] = []
    if monitors is None:
        if "monitor_intent" in binding:
            issues.append(
                f"binding for edge '{binding.get('edge_id', binding.get('edge'))}' "
                "has monitor_intent but no canonical monitors"
            )
        else:
            issues.append(
                f"binding for edge '{binding.get('edge_id', binding.get('edge'))}' "
                "has no monitors"
            )
        return [], issues

    normalized_monitors: list[dict[str, Any]] = []
    for monitor_index, monitor in enumerate(_as_list(monitors)):
        normalized_monitor, monitor_issues = _normalize_monitor(
            monitor,
            monitor_index=monitor_index,
            monitored_edge=monitored_edge,
        )
        if normalized_monitor is not None:
            normalized_monitors.append(normalized_monitor)
        issues.extend(monitor_issues)

    return normalized_monitors, issues


def _normalize_monitor(
    monitor: Mapping[str, Any],
    *,
    monitor_index: int,
    monitored_edge: Mapping[str, Any] | None,
) -> tuple[dict[str, Any] | None, list[str]]:
    if not isinstance(monitor, Mapping):
        monitor = {"type": monitor}
    monitor = dict(monitor)
    issues: list[str] = []
    if "fn" in monitor:
        return {
            "fn": monitor["fn"],
            "kwargs": dict(monitor.get("kwargs", {})),
        }, issues

    monitor_type = _canonical_type(
        monitor.get("type", monitor.get("monitor_type", monitor.get("name"))),
        MONITOR_TYPE_ALIASES,
    )
    if monitor_type is None:
        return None, [f"monitor {monitor_index} has no supported type"]

    if monitor_type == "object_moved":
        objects = monitor.get("objects", monitor.get("obj_name", monitor.get("object")))
        if objects is None:
            objects = _edge_action_objects(monitored_edge)
        objects = [str(obj_name) for obj_name in _as_list(objects) if obj_name]
        if not objects:
            return None, [
                f"monitor {monitor_index} object_moved needs objects or obj_name"
            ]
        return {
            "type": "object_moved",
            "objects": objects,
            "threshold": monitor.get("threshold", 0.02),
        }, issues

    robot_name = monitor.get("robot_name", monitor.get("arm"))
    obj_name = monitor.get("obj_name", monitor.get("object"))
    if robot_name is None or obj_name is None:
        inferred = _infer_single_robot_object(monitored_edge)
        if inferred is not None:
            robot_name = robot_name or inferred[0]
            obj_name = obj_name or inferred[1]
    if robot_name is None or obj_name is None:
        return None, [
            f"monitor {monitor_index} hold_lost needs robot_name and obj_name"
        ]
    return {
        "type": "hold_lost",
        "robot_name": str(robot_name),
        "obj_name": str(obj_name),
        "threshold": monitor.get("threshold", 0.05),
    }, issues


def _normalize_recovery_steps(
    binding: Mapping[str, Any],
    monitored_edge: Mapping[str, Any] | None,
    monitors: list[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    recovery_steps = (
        binding.get("recovery")
        or binding.get("recovery_steps")
        or binding.get("policy")
    )
    issues: list[str] = []
    if recovery_steps is None:
        if "recovery_intent" in binding:
            issues.append(
                f"binding for edge '{binding.get('edge_id', binding.get('edge'))}' "
                "has recovery_intent but no canonical recovery steps"
            )
            return [], issues
        recovery_steps = [{"type": "retry_failed_edge"}]

    normalized_steps: list[dict[str, Any]] = []
    for step_index, step in enumerate(_as_list(recovery_steps)):
        normalized_step, step_issues = _normalize_recovery_step(
            step,
            step_index=step_index,
            monitored_edge=monitored_edge,
            monitors=monitors,
        )
        if normalized_step is not None:
            normalized_steps.append(normalized_step)
        issues.extend(step_issues)

    return normalized_steps, issues


def _normalize_recovery_step(
    step: Mapping[str, Any],
    *,
    step_index: int,
    monitored_edge: Mapping[str, Any] | None,
    monitors: list[Mapping[str, Any]],
) -> tuple[dict[str, Any] | None, list[str]]:
    if not isinstance(step, Mapping):
        step = {"type": step}
    step = dict(step)
    if "left_arm_action" in step or "right_arm_action" in step:
        return {
            "type": "action",
            "left_arm_action": deepcopy(step.get("left_arm_action")),
            "right_arm_action": deepcopy(step.get("right_arm_action")),
            **_optional_step_metadata(step),
        }, []

    step_type = _canonical_type(
        step.get("type", step.get("name")), RECOVERY_STEP_TYPE_ALIASES
    )
    if step_type is None:
        return None, [f"recovery step {step_index} has no supported type"]

    base = {
        "type": step_type,
        **_optional_step_metadata(step),
    }
    if step_type == "retry_failed_edge":
        if "force_valid" in step:
            base["force_valid"] = step["force_valid"]
        return base, []

    if step_type == "action":
        return None, [
            f"recovery step {step_index} action needs left_arm_action or right_arm_action"
        ]

    if step_type == "replay_edge":
        edge_id = step.get("edge_id", step.get("edge"))
        if edge_id is None:
            return None, [f"recovery step {step_index} replay_edge needs edge_id"]
        base["edge_id"] = str(edge_id)
        if "force_valid" in step:
            base["force_valid"] = step["force_valid"]
        return base, []

    if step_type == "regrasp_both":
        arms = step.get("arms")
        if arms is None:
            arms = _infer_regrasp_arms(monitored_edge, monitors)
        if not arms:
            return None, [f"recovery step {step_index} regrasp_both needs arms"]
        base.update(
            {
                "arms": {str(robot): str(obj) for robot, obj in dict(arms).items()},
                "pre_grasp_dis": step.get("pre_grasp_dis", 0.1),
                "force_valid": step.get("force_valid", False),
            }
        )
        return base, []

    robot_name = step.get("robot_name", step.get("arm"))
    obj_name = step.get("obj_name", step.get("object"))
    if robot_name is None or obj_name is None:
        inferred = _infer_regrasp_target(monitored_edge, monitors)
        if inferred is not None:
            robot_name = robot_name or inferred[0]
            obj_name = obj_name or inferred[1]
    if robot_name is None or obj_name is None:
        return None, [
            f"recovery step {step_index} regrasp needs robot_name and obj_name"
        ]

    base.update(
        {
            "robot_name": str(robot_name),
            "obj_name": str(obj_name),
            "pre_grasp_dis": step.get("pre_grasp_dis", 0.1),
            "force_valid": step.get("force_valid", False),
        }
    )
    return base, []


def _optional_step_metadata(step: Mapping[str, Any]) -> dict[str, Any]:
    metadata = {}
    for key in ("name", "semantic"):
        if key in step:
            metadata[key] = step[key]
    return metadata


def _canonical_type(value: Any, aliases: Mapping[str, str]) -> str | None:
    if value is None:
        return None
    key = _sanitize_id(str(value))
    return aliases.get(key)


def _normalize_merge(value: Any) -> str | None:
    key = _sanitize_id(str(value))
    if key in {"target", "complete", "completed", "finish", "success"}:
        return "target"
    if key in {"source", "retry", "precondition", "restore_precondition"}:
        return "source"
    return None


def _edge_action_objects(edge: Mapping[str, Any] | None) -> list[str]:
    objects: list[str] = []
    if edge is None:
        return objects
    for action_key in ("left_arm_action", "right_arm_action"):
        action = edge.get(action_key)
        if not action:
            continue
        obj_name = dict(action.get("kwargs", {})).get("obj_name")
        if obj_name is not None:
            objects.append(str(obj_name))
    return objects


def _edge_robot_objects(edge: Mapping[str, Any] | None) -> dict[str, str]:
    robot_objects: dict[str, str] = {}
    if edge is None:
        return robot_objects
    for action_key in ("left_arm_action", "right_arm_action"):
        action = edge.get(action_key)
        if not action:
            continue
        kwargs = dict(action.get("kwargs", {}))
        robot_name = kwargs.get("robot_name")
        obj_name = kwargs.get("obj_name")
        if robot_name is not None and obj_name is not None:
            robot_objects[str(robot_name)] = str(obj_name)
    return robot_objects


def _infer_single_robot_object(
    edge: Mapping[str, Any] | None,
) -> tuple[str, str] | None:
    robot_objects = _edge_robot_objects(edge)
    if len(robot_objects) == 1:
        return next(iter(robot_objects.items()))
    return None


def _infer_regrasp_target(
    edge: Mapping[str, Any] | None,
    monitors: list[Mapping[str, Any]],
) -> tuple[str, str] | None:
    hold_monitors = [
        (str(monitor["robot_name"]), str(monitor["obj_name"]))
        for monitor in monitors
        if monitor.get("type") == "hold_lost"
    ]
    if len(hold_monitors) == 1:
        return hold_monitors[0]
    return _infer_single_robot_object(edge)


def _infer_regrasp_arms(
    edge: Mapping[str, Any] | None,
    monitors: list[Mapping[str, Any]],
) -> dict[str, str]:
    arms = _edge_robot_objects(edge)
    for monitor in monitors:
        if monitor.get("type") == "hold_lost":
            arms[str(monitor["robot_name"])] = str(monitor["obj_name"])
    return arms


def _expand_recovery_binding(
    expanded: dict[str, Any],
    binding: Mapping[str, Any],
    *,
    binding_index: int,
    nominal_edges: Mapping[str, Mapping[str, Any]],
    used_node_ids: set[str],
    used_edge_ids: set[str],
) -> None:
    edge_id = binding["edge_id"]
    if edge_id not in nominal_edges:
        raise ValueError(
            f"Recovery binding references unknown nominal edge_id '{edge_id}'."
        )

    monitored_edge = nominal_edges[edge_id]
    failure_name = binding.get("failure_name", f"failure_{binding_index}")
    monitor_sequence = _expand_monitor_sequence(binding)
    if not monitor_sequence:
        raise ValueError(f"Recovery binding for edge '{edge_id}' must define monitors.")

    merge = binding.get("merge", "target")
    if merge not in {"source", "target"}:
        raise ValueError("Recovery binding merge must be 'source' or 'target'.")

    recovery_steps = list(
        binding.get("recovery")
        or binding.get("recovery_steps")
        or [{"type": "retry_failed_edge"}]
    )
    if not recovery_steps:
        raise ValueError(
            f"Recovery binding for edge '{edge_id}' has no recovery steps."
        )

    current_node = monitored_edge["source"]
    merge_node = monitored_edge[merge]
    branch_edges: list[str] = []
    generated_edges: list[tuple[dict[str, Any], Mapping[str, Any]]] = []

    for step_index, step in enumerate(recovery_steps):
        step = dict(step)
        step_label = _recovery_step_label(step, monitored_edge)
        is_last_step = step_index == len(recovery_steps) - 1
        target_node = merge_node
        if not is_last_step:
            target_node = _unique_id(
                f"rn_{_sanitize_id(edge_id)}_{step_index + 1}_{step_label}",
                used_node_ids,
            )
            expanded["recovery_nodes"].append(
                {
                    "id": target_node,
                    "semantic": step.get("semantic", step_label.replace("_", " ")),
                }
            )

        recovery_edge = {
            "id": _unique_id(
                f"re_{_sanitize_id(edge_id)}_{step_index + 1}_{step_label}",
                used_edge_ids,
            ),
            "source": current_node,
            "target": target_node,
            **_expand_recovery_step_actions(step, monitored_edge, nominal_edges),
        }
        expanded["recovery_edges"].append(recovery_edge)
        branch_edges.append(recovery_edge["id"])
        generated_edges.append((recovery_edge, step))
        current_node = target_node

    expanded["recovery_branches"].append(
        {
            "edge_id": edge_id,
            "failure_name": failure_name,
            "monitor_sequence": monitor_sequence,
            "recovery_edges": branch_edges,
        }
    )

    if binding.get("repeat_until_success", True):
        for recovery_edge, step in generated_edges:
            recovery_monitor_sequence = _recovery_step_monitor_sequence(
                step,
                fallback_monitor_sequence=monitor_sequence,
            )
            _add_reusable_recovery_branch(
                expanded,
                recovery_edge,
                step,
                binding=binding,
                monitor_sequence=recovery_monitor_sequence,
                used_edge_ids=used_edge_ids,
            )


def _add_reusable_recovery_branch(
    expanded: dict[str, Any],
    recovery_edge: Mapping[str, Any],
    step: Mapping[str, Any],
    *,
    binding: Mapping[str, Any],
    monitor_sequence: list[dict[str, Any]],
    used_edge_ids: set[str],
) -> None:
    failure_name = binding.get("failure_name", "recovery_failure")
    repair_edges = [recovery_edge["id"]]

    hold_monitor = _first_hold_lost_monitor(binding)
    if hold_monitor is not None and _step_type(step) not in {"regrasp", "regrasp_both"}:
        repair_edge = {
            "id": _unique_id(
                f"{recovery_edge['id']}_repair_regrasp_{_sanitize_id(hold_monitor['obj_name'])}",
                used_edge_ids,
            ),
            "source": recovery_edge["source"],
            "target": recovery_edge["source"],
            **_regrasp_actions(
                robot_name=hold_monitor["robot_name"],
                obj_name=hold_monitor["obj_name"],
                pre_grasp_dis=float(step.get("pre_grasp_dis", 0.1)),
                force_valid=bool(step.get("force_valid", False)),
            ),
        }
        expanded["recovery_edges"].append(repair_edge)
        repair_edges = [repair_edge["id"]]
        repair_monitor_sequence = _object_moved_monitors([hold_monitor["obj_name"]])
        expanded["recovery_branches"].append(
            {
                "edge_id": repair_edge["id"],
                "failure_name": f"{failure_name}_during_{repair_edge['id']}",
                "monitor_sequence": repair_monitor_sequence,
                "recovery_edges": [repair_edge["id"]],
            }
        )

    expanded["recovery_branches"].append(
        {
            "edge_id": recovery_edge["id"],
            "failure_name": f"{failure_name}_during_{recovery_edge['id']}",
            "monitor_sequence": deepcopy(monitor_sequence),
            "recovery_edges": repair_edges,
        }
    )


def _expand_monitor_sequence(binding: Mapping[str, Any]) -> list[dict[str, Any]]:
    monitors = binding.get("monitors", [])
    monitor_sequence: list[dict[str, Any]] = []
    for monitor in monitors:
        monitor = dict(monitor)
        if "fn" in monitor:
            monitor_sequence.append(
                {"fn": monitor["fn"], "kwargs": dict(monitor.get("kwargs", {}))}
            )
            continue

        monitor_type = monitor.get("type")
        if monitor_type == "object_moved":
            objects = monitor.get("objects")
            if objects is None:
                objects = [monitor["obj_name"]]
            monitor_sequence.extend(
                _object_moved_monitors(
                    _as_list(objects),
                    threshold=monitor.get("threshold", 0.02),
                )
            )
        elif monitor_type in {"hold_lost", "object_held"}:
            monitor_sequence.append(
                {
                    "fn": "monitor_object_held",
                    "kwargs": {
                        "robot_name": monitor["robot_name"],
                        "obj_name": monitor["obj_name"],
                        "threshold": monitor.get("threshold", 0.05),
                    },
                }
            )
        else:
            raise ValueError(f"Unsupported recovery monitor type '{monitor_type}'.")

    return monitor_sequence


def _recovery_step_monitor_sequence(
    step: Mapping[str, Any],
    *,
    fallback_monitor_sequence: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    step_type = _step_type(step)
    if step_type in {"regrasp", "regrasp_both"}:
        objects = _regrasp_step_objects(step)
        if objects:
            return _object_moved_monitors(objects)
    return deepcopy(fallback_monitor_sequence)


def _object_moved_monitors(
    objects: list[str],
    *,
    threshold: float = 0.02,
) -> list[dict[str, Any]]:
    return [
        {
            "fn": "monitor_object_moved",
            "kwargs": {"obj_name": obj_name, "threshold": threshold},
        }
        for obj_name in objects
    ]


def _expand_recovery_step_actions(
    step: Mapping[str, Any],
    monitored_edge: Mapping[str, Any],
    nominal_edges: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    step_type = _step_type(step)

    if step_type in {"retry_failed_edge", "retry_edge"}:
        return _copy_edge_actions(
            monitored_edge,
            force_valid=step.get("force_valid"),
        )

    if step_type == "replay_edge":
        replay_edge_id = step["edge_id"]
        if replay_edge_id not in nominal_edges:
            raise ValueError(f"Unknown replay_edge id '{replay_edge_id}'.")
        return _copy_edge_actions(
            nominal_edges[replay_edge_id],
            force_valid=step.get("force_valid"),
        )

    if step_type in {"regrasp", "regrasp_both"}:
        arms = step.get("arms")
        if arms:
            return _multi_regrasp_actions(
                arms,
                pre_grasp_dis=float(step.get("pre_grasp_dis", 0.1)),
                force_valid=bool(step.get("force_valid", False)),
            )
        return _regrasp_actions(
            robot_name=step["robot_name"],
            obj_name=step["obj_name"],
            pre_grasp_dis=float(step.get("pre_grasp_dis", 0.1)),
            force_valid=bool(step.get("force_valid", False)),
        )

    if step_type == "action" or "left_arm_action" in step or "right_arm_action" in step:
        return {
            "left_arm_action": deepcopy(step.get("left_arm_action")),
            "right_arm_action": deepcopy(step.get("right_arm_action")),
        }

    raise ValueError(f"Unsupported recovery step type '{step_type}'.")


def _copy_edge_actions(
    edge: Mapping[str, Any],
    *,
    force_valid: Any = None,
) -> dict[str, Any]:
    left_action = deepcopy(edge.get("left_arm_action"))
    right_action = deepcopy(edge.get("right_arm_action"))
    if force_valid is not None:
        _set_force_valid(left_action, bool(force_valid))
        _set_force_valid(right_action, bool(force_valid))
    return {
        "left_arm_action": left_action,
        "right_arm_action": right_action,
    }


def _regrasp_actions(
    *,
    robot_name: str,
    obj_name: str,
    pre_grasp_dis: float,
    force_valid: bool,
) -> dict[str, Any]:
    action = {
        "fn": "grasp",
        "kwargs": {
            "robot_name": robot_name,
            "obj_name": obj_name,
            "pre_grasp_dis": pre_grasp_dis,
            "force_valid": force_valid,
        },
    }
    if "left" in robot_name:
        return {"left_arm_action": action, "right_arm_action": None}
    return {"left_arm_action": None, "right_arm_action": action}


def _multi_regrasp_actions(
    arms: Mapping[str, str],
    *,
    pre_grasp_dis: float,
    force_valid: bool,
) -> dict[str, Any]:
    actions = {"left_arm_action": None, "right_arm_action": None}
    for robot_name, obj_name in arms.items():
        action = {
            "fn": "grasp",
            "kwargs": {
                "robot_name": robot_name,
                "obj_name": obj_name,
                "pre_grasp_dis": pre_grasp_dis,
                "force_valid": force_valid,
            },
        }
        if "left" in robot_name:
            actions["left_arm_action"] = action
        else:
            actions["right_arm_action"] = action
    return actions


def _regrasp_step_objects(step: Mapping[str, Any]) -> list[str]:
    arms = step.get("arms")
    if arms:
        return [str(obj_name) for obj_name in arms.values()]
    obj_name = step.get("obj_name")
    return [str(obj_name)] if obj_name is not None else []


def _set_force_valid(action: dict[str, Any] | None, value: bool) -> None:
    if action is not None:
        action.setdefault("kwargs", {})["force_valid"] = value


def _first_hold_lost_monitor(binding: Mapping[str, Any]) -> dict[str, Any] | None:
    for monitor in binding.get("monitors", []):
        monitor = dict(monitor)
        if monitor.get("type") in {"hold_lost", "object_held"}:
            return {
                "robot_name": monitor["robot_name"],
                "obj_name": monitor["obj_name"],
            }
    return None


def _recovery_step_label(
    step: Mapping[str, Any],
    monitored_edge: Mapping[str, Any],
) -> str:
    step_type = _step_type(step)
    if step_type in {"retry_failed_edge", "retry_edge"}:
        return f"retry_{_sanitize_id(monitored_edge['id'])}"
    if step_type == "replay_edge":
        return f"replay_{_sanitize_id(step['edge_id'])}"
    if step_type in {"regrasp", "regrasp_both"}:
        if step.get("arms"):
            return "regrasp_objects"
        return f"regrasp_{_sanitize_id(step['obj_name'])}"
    return _sanitize_id(step.get("name", step_type))


def _step_type(step: Mapping[str, Any]) -> str:
    return step.get("type", "action")


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _unique_id(base: str, used_ids: set[str]) -> str:
    candidate = _sanitize_id(base)
    if candidate not in used_ids:
        used_ids.add(candidate)
        return candidate

    index = 2
    while f"{candidate}_{index}" in used_ids:
        index += 1
    unique_candidate = f"{candidate}_{index}"
    used_ids.add(unique_candidate)
    return unique_candidate


def _sanitize_id(value: str) -> str:
    return re.sub(r"[^0-9a-zA-Z_]+", "_", str(value)).strip("_").lower()


def _empty_recovery_graph(task_graph: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "task": task_graph.get("task", ""),
        "recovery_nodes": [],
        "recovery_edges": [],
        "recovery_branches": [],
    }


def _reject_compiled_recovery_schema_in_authoring_spec(
    recovery_spec: Mapping[str, Any],
) -> None:
    compiled_keys = {"recovery_nodes", "recovery_edges", "recovery_branches"}
    present_keys = compiled_keys & set(recovery_spec)
    if present_keys:
        present = ", ".join(sorted(present_keys))
        raise ValueError(
            "RecoveryAgent must output the lightweight recovery_bindings schema. "
            f"Compiled recovery graph keys are not accepted here: {present}."
        )
    if "recoveries" in recovery_spec or "error_functions" in recovery_spec:
        raise ValueError(
            "Recovery specs must not contain legacy recoveries or executable error_functions."
        )


def _validate_recovery_authoring_spec(recovery_spec: Mapping[str, Any]) -> None:
    allowed_top_keys = {"task", "recovery_bindings"}
    unknown_top_keys = set(recovery_spec) - allowed_top_keys
    if unknown_top_keys:
        unknown = ", ".join(sorted(unknown_top_keys))
        raise ValueError(f"Unknown recovery spec keys: {unknown}.")

    allowed_binding_keys = {
        "edge_id",
        "failure_name",
        "monitors",
        "recovery",
        "recovery_steps",
        "merge",
        "repeat_until_success",
    }
    allowed_monitor_keys = {
        "type",
        "objects",
        "obj_name",
        "robot_name",
        "threshold",
        "fn",
        "kwargs",
    }
    allowed_step_keys = {
        "type",
        "robot_name",
        "obj_name",
        "arms",
        "pre_grasp_dis",
        "force_valid",
        "edge_id",
        "left_arm_action",
        "right_arm_action",
        "name",
        "semantic",
    }
    for binding in recovery_spec.get("recovery_bindings", []):
        unknown_binding_keys = set(binding) - allowed_binding_keys
        if unknown_binding_keys:
            unknown = ", ".join(sorted(unknown_binding_keys))
            raise ValueError(f"Unknown recovery binding keys: {unknown}.")
        for monitor in binding.get("monitors", []):
            unknown_monitor_keys = set(monitor) - allowed_monitor_keys
            if unknown_monitor_keys:
                unknown = ", ".join(sorted(unknown_monitor_keys))
                raise ValueError(f"Unknown recovery monitor keys: {unknown}.")
        for step in binding.get("recovery", binding.get("recovery_steps", [])):
            unknown_step_keys = set(step) - allowed_step_keys
            if unknown_step_keys:
                unknown = ", ".join(sorted(unknown_step_keys))
                raise ValueError(f"Unknown recovery step keys: {unknown}.")


def _reject_legacy_recovery_schema(recovery_spec: Mapping[str, Any]) -> None:
    if "recovery_bindings" in recovery_spec:
        raise ValueError(
            "compile_agent_graph_spec expects a compiled recovery_graph. "
            "Call expand_recovery_spec(task_graph, recovery_spec) first."
        )
    if "recoveries" in recovery_spec:
        raise ValueError(
            "Legacy recovery schema key 'recoveries' is no longer supported. "
            "Use recovery_bindings before compilation or recovery_graph after compilation."
        )
    if "error_functions" in recovery_spec:
        raise ValueError(
            "Recovery graphs do not execute error_functions. "
            "Use monitor_sequence and recovery_edges only."
        )

    for branch in recovery_spec.get("recovery_branches", []):
        if "error_functions" in branch:
            raise ValueError(
                "Recovery branches do not execute error_functions. "
                "Use monitor_sequence and recovery_edges only."
            )
        if "recovery_actions" in branch or "merge" in branch:
            raise ValueError(
                "Legacy recovery branch keys 'recovery_actions' and 'merge' are no "
                "longer supported. Define explicit recovery_edges instead."
            )


def _resolve_runtime(
    *,
    graph_cls: type | None,
    action_module: Any,
    monitor_module: Any,
) -> tuple[type, Any, Any]:
    if graph_cls is None:
        graph_cls = _resolve_attr(
            importlib.import_module("embodichain.lab.sim.agent.agent_graph"),
            "AgentTaskGraph",
        )
    if action_module is None:
        action_module = importlib.import_module(
            "embodichain.lab.sim.agent.atom_actions"
        )
    if monitor_module is None:
        monitor_module = importlib.import_module(
            "embodichain.lab.sim.agent.monitor_functions"
        )
    return graph_cls, action_module, monitor_module


def _validate_task_spec(task_spec: Mapping[str, Any]) -> tuple[set[str], set[str]]:
    node_ids = set()
    for node in task_spec.get("nodes", []):
        node_id = node["id"]
        if node_id in node_ids:
            raise ValueError(f"Duplicate graph node id '{node_id}'.")
        node_ids.add(node_id)

    for required_node in (task_spec["start"], task_spec["goal"]):
        if required_node not in node_ids:
            raise ValueError(f"Graph node '{required_node}' is not defined.")

    edge_specs = list(task_spec.get("edges", []))
    edge_ids = set()
    for edge in edge_specs:
        edge_id = edge["id"]
        if edge_id in edge_ids:
            raise ValueError(f"Duplicate graph edge id '{edge_id}'.")
        edge_ids.add(edge_id)
        if edge.get("left_arm_action") is None and edge.get("right_arm_action") is None:
            raise ValueError(f"Nominal edge '{edge_id}' must define an arm action.")

        for node_key in ("source", "target"):
            node_id = edge[node_key]
            if node_id not in node_ids:
                raise ValueError(
                    f"Edge '{edge_id}' references unknown {node_key} node '{node_id}'."
                )

    _validate_nominal_path(task_spec, edge_specs)
    return node_ids, edge_ids


def _validate_nominal_path(
    task_spec: Mapping[str, Any],
    edge_specs: list[Mapping[str, Any]],
) -> None:
    outgoing_edges: dict[str, Mapping[str, Any]] = {}
    for edge in edge_specs:
        source = edge["source"]
        if source in outgoing_edges:
            raise ValueError(
                f"Nominal node '{source}' has multiple outgoing edges. "
                "The current graph executor expects one deterministic nominal path."
            )
        outgoing_edges[source] = edge

    current = task_spec["start"]
    goal = task_spec["goal"]
    visited_edges = set()
    visited_nodes = {current}

    while current != goal:
        edge = outgoing_edges.get(current)
        if edge is None:
            raise ValueError(
                f"Nominal graph has no start-to-goal path from node '{current}'."
            )
        edge_id = edge["id"]
        if edge_id in visited_edges:
            raise ValueError("Nominal graph contains a cycle.")

        visited_edges.add(edge_id)
        current = edge["target"]
        if current in visited_nodes and current != goal:
            raise ValueError("Nominal graph contains a cycle.")
        visited_nodes.add(current)

    all_edge_ids = {edge["id"] for edge in edge_specs}
    unused_edge_ids = all_edge_ids - visited_edges
    if unused_edge_ids:
        unused = ", ".join(sorted(unused_edge_ids))
        raise ValueError(
            f"Nominal graph contains edges outside the start-to-goal path: {unused}."
        )


def _add_recovery_nodes(
    graph: Any,
    recovery_spec: Mapping[str, Any],
    nominal_node_ids: set[str],
) -> set[str]:
    recovery_node_ids = set()
    recovery_nodes = recovery_spec.get("recovery_nodes", [])
    if isinstance(recovery_nodes, Mapping):
        recovery_nodes = [
            {"id": node_id, **dict(node_spec)}
            for node_id, node_spec in recovery_nodes.items()
        ]

    for node in recovery_nodes:
        node_id = node["id"]
        if node_id in nominal_node_ids or node_id in recovery_node_ids:
            raise ValueError(f"Duplicate recovery node id '{node_id}'.")
        recovery_node_ids.add(node_id)
        graph.add_node(node_id, node.get("semantic", ""))

    return recovery_node_ids


def _add_recovery_edges(
    graph: Any,
    recovery_spec: Mapping[str, Any],
    *,
    action_module: Any,
    defined_node_ids: set[str],
) -> set[str]:
    recovery_edge_ids = set()
    for edge in recovery_spec.get("recovery_edges", []):
        edge_id = edge["id"]
        if edge_id in graph.edges or edge_id in recovery_edge_ids:
            raise ValueError(f"Duplicate recovery edge id '{edge_id}'.")
        if edge.get("left_arm_action") is None and edge.get("right_arm_action") is None:
            raise ValueError(f"Recovery edge '{edge_id}' must define an arm action.")

        for node_key in ("source", "target"):
            node_id = edge[node_key]
            if node_id not in defined_node_ids:
                raise ValueError(
                    f"Recovery edge '{edge_id}' references unknown {node_key} node '{node_id}'."
                )

        graph.add_edge(
            edge_id,
            edge["source"],
            edge["target"],
            left_arm_action=_compile_action(edge.get("left_arm_action"), action_module),
            right_arm_action=_compile_action(
                edge.get("right_arm_action"), action_module
            ),
            is_recovery=True,
        )
        recovery_edge_ids.add(edge_id)

    return recovery_edge_ids


def _add_recovery_branches(
    graph: Any,
    recovery_spec: Mapping[str, Any],
    *,
    env: Any,
    monitor_module: Any,
    nominal_edge_ids: set[str],
    recovery_edge_ids: set[str],
) -> None:
    recoverable_edge_ids = nominal_edge_ids | recovery_edge_ids
    for branch_index, branch in enumerate(recovery_spec.get("recovery_branches", [])):
        edge_id = branch["edge_id"]
        if edge_id not in recoverable_edge_ids:
            raise ValueError(f"Recovery branch references unknown edge_id '{edge_id}'.")

        edge = graph.edges[edge_id]
        monitor_sequence = [
            _compile_monitor(monitor, monitor_module, env=env)
            for monitor in branch.get("monitor_sequence", [])
        ]
        if not monitor_sequence:
            raise ValueError(
                f"Recovery branch for edge '{edge_id}' must define monitor_sequence."
            )

        branch_recovery_edges = list(branch.get("recovery_edges", []))
        _validate_recovery_branch_path(
            graph,
            branch=branch,
            monitored_edge=edge,
            recovery_edge_ids=recovery_edge_ids,
        )

        edge.monitor_sequences = list(edge.monitor_sequences or [])
        edge.monitor_labels = list(edge.monitor_labels or [])
        monitor_index = len(edge.monitor_sequences)
        edge.monitor_sequences.append(monitor_sequence)
        edge.monitor_labels.append(
            branch.get("failure_name", f"recovery_{branch_index}")
        )
        graph.add_recovery(
            edge.id,
            monitor_index=monitor_index,
            recovery_edges=branch_recovery_edges,
        )


def _validate_recovery_branch_path(
    graph: Any,
    *,
    branch: Mapping[str, Any],
    monitored_edge: Any,
    recovery_edge_ids: set[str],
) -> None:
    branch_edges = list(branch.get("recovery_edges", []))
    if not branch_edges:
        raise ValueError(
            f"Recovery branch for edge '{monitored_edge.id}' must define recovery_edges."
        )

    previous_target = None
    for index, edge_id in enumerate(branch_edges):
        if edge_id not in recovery_edge_ids:
            raise ValueError(
                f"Recovery branch for edge '{monitored_edge.id}' references unknown recovery edge '{edge_id}'."
            )

        recovery_edge = graph.edges[edge_id]
        if index == 0 and recovery_edge.source != monitored_edge.source:
            raise ValueError(
                f"Recovery branch for edge '{monitored_edge.id}' must start from '{monitored_edge.source}'."
            )
        if previous_target is not None and recovery_edge.source != previous_target:
            raise ValueError(
                f"Recovery branch for edge '{monitored_edge.id}' is not path-contiguous at '{edge_id}'."
            )
        previous_target = recovery_edge.target

    if previous_target not in {monitored_edge.source, monitored_edge.target}:
        raise ValueError(
            f"Recovery branch for edge '{monitored_edge.id}' must merge into the failed edge source or target node."
        )


def _compile_action(spec: Any, action_module: Any) -> Any:
    if spec is None:
        return None
    return _compile_call(spec, action_module)


def _compile_monitor(spec: Mapping[str, Any], monitor_module: Any, *, env: Any) -> Any:
    kwargs = dict(spec.get("kwargs", {}))
    kwargs.setdefault("env", env)
    return _compile_call({"fn": spec["fn"], "kwargs": kwargs}, monitor_module)


def _compile_call(spec: Mapping[str, Any], namespace: Any) -> Any:
    return partial(_resolve_attr(namespace, spec["fn"]), **dict(spec.get("kwargs", {})))


def _resolve_attr(namespace: Any, name: str) -> Any:
    if isinstance(namespace, Mapping):
        return namespace[name]
    return getattr(namespace, name)
