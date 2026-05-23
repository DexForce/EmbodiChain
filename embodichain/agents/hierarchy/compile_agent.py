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

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from embodichain.agents.hierarchy.agent_base import AgentBase
from embodichain.data import database_agent_prompt_dir
from embodichain.utils.llm_json import extract_json_object, normalize_json_content

__all__ = ["CompileAgent"]

COMPILED_GRAPH_SCHEMA_VERSION = "recovery_bindings_atomic_v3"


class CompileAgent(AgentBase):
    """Compile and execute atomic-action graph specs.

    The compile agent expands the LLM-facing recovery spec into an explicit
    monitor/recovery graph artifact, then executes that artifact through the
    graph runtime.
    """

    query_prefix = "# "
    query_suffix = "."
    prompt_kwargs: dict[str, dict[str, Any]]

    def __init__(self, llm, **kwargs) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.prompt_kwargs = kwargs.get("prompt_kwargs", {})
        self.llm = llm

    def generate(self, **kwargs):
        log_dir = kwargs.get(
            "log_dir", Path(database_agent_prompt_dir) / self.task_name
        )
        file_path = Path(log_dir) / "agent_compiled_graph.json"
        recovery_enabled = bool(kwargs.get("recovery_enabled", False))
        task_graph = extract_json_object(kwargs["task_graph"])
        raw_recovery_spec = extract_json_object(
            kwargs.get("recovery_spec") or _empty_recovery_spec(task_graph)
        )
        task_graph_hash = _stable_json_hash(task_graph)
        raw_recovery_spec_hash = _stable_json_hash(raw_recovery_spec)

        if not kwargs.get("regenerate", False) and file_path.exists():
            existing_bundle = extract_json_object(file_path.read_text(encoding="utf-8"))
            metadata = existing_bundle.get("metadata", {})
            if (
                metadata.get("recovery_enabled") == recovery_enabled
                and metadata.get("schema_version") == COMPILED_GRAPH_SCHEMA_VERSION
                and metadata.get("task_graph_hash") == task_graph_hash
                and metadata.get("raw_recovery_spec_hash") == raw_recovery_spec_hash
            ):
                print(f"Compiled graph artifact already exists at {file_path}.")
                return file_path, kwargs, None

        from embodichain.lab.sim.agent.graph_spec import (
            expand_recovery_spec,
            normalize_recovery_spec,
        )

        recovery_spec, issues = normalize_recovery_spec(task_graph, raw_recovery_spec)
        if issues:
            recovery_spec = _canonicalize_recovery_spec_with_llm(
                self.llm,
                task_graph=task_graph,
                recovery_spec=raw_recovery_spec,
                issues=issues,
            )
            recovery_spec, issues = normalize_recovery_spec(task_graph, recovery_spec)
            if issues:
                issue_text = "; ".join(issues)
                raise ValueError(
                    "CompileAgent could not canonicalize recovery_spec: "
                    f"{issue_text}"
                )

        recovery_spec_hash = _stable_json_hash(recovery_spec)
        if not kwargs.get("regenerate", False) and file_path.exists():
            existing_bundle = extract_json_object(file_path.read_text(encoding="utf-8"))
            metadata = existing_bundle.get("metadata", {})
            if (
                metadata.get("recovery_enabled") == recovery_enabled
                and metadata.get("schema_version") == COMPILED_GRAPH_SCHEMA_VERSION
                and metadata.get("task_graph_hash") == task_graph_hash
                and metadata.get("recovery_spec_hash") == recovery_spec_hash
            ):
                print(f"Compiled graph artifact already exists at {file_path}.")
                return file_path, kwargs, None

        recovery_graph = expand_recovery_spec(task_graph, recovery_spec)
        content = normalize_json_content(
            {
                "task_graph": task_graph,
                "recovery_spec": recovery_spec,
                "recovery_graph": recovery_graph,
                "metadata": {
                    "recovery_enabled": recovery_enabled,
                    "schema_version": COMPILED_GRAPH_SCHEMA_VERSION,
                    "task_graph_hash": task_graph_hash,
                    "raw_recovery_spec_hash": raw_recovery_spec_hash,
                    "recovery_spec_hash": recovery_spec_hash,
                },
            }
        )

        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        print(f"Compiled graph artifact saved to {file_path}")
        return file_path, kwargs, content

    def act(self, graph_file_path, **kwargs):
        graph_file_path = Path(graph_file_path)
        if graph_file_path.suffix != ".json":
            raise ValueError("CompileAgent executes compiled graph JSON artifacts.")

        from embodichain.lab.sim.agent.graph_spec import (
            compile_agent_graph_from_file,
        )

        runtime_kwargs = _runtime_kwargs(kwargs, getattr(self, "prompt_kwargs", {}))
        runtime_recovery_agent = runtime_kwargs.pop("runtime_recovery_agent", None)
        runtime_recovery_enabled = bool(
            runtime_kwargs.get("runtime_llm_recovery", False)
        )
        if (
            runtime_recovery_enabled
            and runtime_recovery_agent is not None
            and "runtime_recovery_planner" not in runtime_kwargs
        ):
            graph_bundle = extract_json_object(
                graph_file_path.read_text(encoding="utf-8")
            )
            runtime_kwargs["runtime_recovery_planner"] = (
                _build_runtime_recovery_planner(
                    runtime_recovery_agent,
                    task_graph=graph_bundle.get("task_graph", graph_bundle),
                    use_llm=bool(runtime_kwargs.get("runtime_recovery_use_llm", True)),
                )
            )
            runtime_kwargs["runtime_state_collector"] = _collect_runtime_state
            runtime_kwargs["prefer_runtime_llm_recovery"] = True

        graph = compile_agent_graph_from_file(
            graph_file_path,
            env=runtime_kwargs.get("env"),
        )
        result = graph.run(**runtime_kwargs)
        print("Compiled agent graph executed successfully.")
        return result

    def get_composed_observations(self, **kwargs):
        return dict(kwargs)


def _canonicalize_recovery_spec_with_llm(
    llm,
    *,
    task_graph: dict[str, Any],
    recovery_spec: dict[str, Any],
    issues: list[str],
) -> dict[str, Any]:
    """Use one LLM call to map ambiguous recovery intent to canonical templates."""
    if llm is None:
        raise ValueError(
            "Recovery spec is ambiguous and CompileAgent has no LLM for canonicalization."
        )

    from langchain_core.messages import HumanMessage, SystemMessage

    human_content = (
        "Convert the recovery spec into this canonical authoring schema:\n"
        "{\n"
        '  "task": "<same task name>",\n'
        '  "recovery_bindings": [\n'
        "    {\n"
        '      "edge_id": "<nominal edge id>",\n'
        '      "failure_name": "<short label>",\n'
        '      "monitors": [\n'
        '        {"type": "object_moved", "objects": ["<object>"], "threshold": 0.02}\n'
        "      ],\n"
        '      "recovery": [\n'
        '        {"type": "regrasp", "robot_name": "<arm>", "obj_name": "<object>"}\n'
        "      ],\n"
        '      "merge": "target",\n'
        '      "repeat_until_success": true\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Supported monitor templates:\n"
        "- object_moved: objects or obj_name, optional threshold\n"
        "- hold_lost: robot_name, obj_name, optional threshold\n\n"
        "Supported recovery step templates:\n"
        "- regrasp: robot_name, obj_name, optional pre_grasp_dis\n"
        "- regrasp_both: arms mapping robot_name to obj_name, optional pre_grasp_dis\n"
        "- retry_failed_edge\n"
        "- replay_edge: edge_id\n"
        "- move_to_safe_pose: robot_name or arms\n"
        "- action: left_arm_action and/or right_arm_action atomic action specs\n\n"
        "Rules:\n"
        "- Use only edge ids from the nominal task graph.\n"
        "- Prefer canonical templates over direct action specs.\n"
        "- For grasp-phase displacement use object_moved, not hold_lost.\n"
        "- For an object lost after it is already held use hold_lost.\n"
        "- Use merge target when recovery completes the failed edge state.\n"
        "- Use merge source when recovery only restores the failed edge precondition.\n"
        "- Keep the output compact; do not emit explicit graph topology.\n\n"
        "Unresolved compiler issues:\n"
        f"{json.dumps(issues, ensure_ascii=False, indent=2)}\n\n"
        "Nominal task graph JSON:\n"
        f"{json.dumps(task_graph, ensure_ascii=False, indent=2)}\n\n"
        "Input recovery spec JSON:\n"
        f"{json.dumps(recovery_spec, ensure_ascii=False, indent=2)}\n"
    )
    response = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You canonicalize robotic recovery specs. Return only JSON. "
                    "Do not create recovery_nodes, recovery_edges, or "
                    "recovery_branches."
                )
            ),
            HumanMessage(content=human_content),
        ]
    )
    content = getattr(response, "content", response)
    return extract_json_object(content)


def _build_runtime_recovery_planner(
    recovery_agent,
    *,
    task_graph: dict[str, Any],
    use_llm: bool = True,
):
    """Build a planner callable consumed by ``AgentTaskGraph.run``."""

    def planner(
        *,
        graph,
        edge,
        monitor_index: int,
        monitor_name: str | None,
        step_index: int | None,
        env,
        runtime_kwargs: dict[str, Any],
        failure_context: dict[str, Any] | None = None,
        runtime_state: dict[str, Any] | None = None,
        recovery_history: list[dict[str, Any]] | None = None,
    ):
        runtime_task_graph = _task_graph_with_runtime_edge(task_graph, edge)
        runtime_state = runtime_state or _collect_runtime_state(env)
        failure_context = failure_context or {
            "edge_id": edge.id,
            "monitor_index": monitor_index,
            "monitor_name": monitor_name,
            "step_index": step_index,
        }
        recovery_history = list(recovery_history or [])
        nominal_edges = {
            str(edge_spec["id"]): edge_spec
            for edge_spec in task_graph.get("edges", [])
            if isinstance(edge_spec, dict) and edge_spec.get("id") is not None
        }

        print(
            "Runtime recovery planner requested for "
            f"edge='{edge.id}', monitor='{monitor_name}', "
            f"monitor_index={monitor_index}, step_index={step_index}."
        )
        try:
            if use_llm:
                raw_spec = _call_runtime_recovery_agent(
                    recovery_agent,
                    env=env,
                    runtime_kwargs=runtime_kwargs,
                    task_graph=runtime_task_graph,
                    edge=edge,
                    failure_context=failure_context,
                    runtime_state=runtime_state,
                    recovery_history=recovery_history,
                )
            else:
                raw_spec = _heuristic_runtime_recovery_spec(
                    runtime_task_graph=runtime_task_graph,
                    edge=edge,
                    failure_context=failure_context,
                )

            recovery_spec = _sanitize_runtime_recovery_spec(
                raw_spec,
                current_edge_id=edge.id,
                edge=edge,
                failure_context=failure_context,
                runtime_state=runtime_state,
                recovery_history=recovery_history,
                nominal_edges=nominal_edges,
            )
            if recovery_spec is None:
                return None
            recovery_signature = _runtime_recovery_signature(recovery_spec)
            if _runtime_recovery_repeats_strategy(
                recovery_signature,
                failure_context=failure_context,
                recovery_history=recovery_history,
                max_attempts=int(
                    runtime_kwargs.get(
                        "runtime_recovery_max_repeated_strategy_attempts",
                        1,
                    )
                ),
            ):
                print(
                    "Runtime recovery spec rejected: repeated strategy "
                    f"for {failure_context.get('origin_edge_id', edge.id)} "
                    f"{failure_context.get('failure_class', monitor_name)}."
                )
                return None

            from embodichain.lab.sim.agent.graph_spec import (
                compile_agent_graph_spec,
                expand_recovery_spec,
                normalize_recovery_spec,
            )

            normalized_spec, issues = normalize_recovery_spec(
                runtime_task_graph,
                recovery_spec,
            )
            if issues:
                print(
                    "Runtime recovery spec rejected: "
                    + "; ".join(str(issue) for issue in issues)
                )
                return None

            recovery_graph = expand_recovery_spec(runtime_task_graph, normalized_spec)
            branch_edges = _first_runtime_branch_edges(
                recovery_graph,
                current_edge_id=edge.id,
            )
            if not branch_edges:
                return None

            compiled_graph = compile_agent_graph_spec(
                runtime_task_graph,
                recovery_graph,
                env=env,
            )
            recovery_edges = [compiled_graph.edges[edge_id] for edge_id in branch_edges]
            for recovery_edge in recovery_edges:
                setattr(
                    recovery_edge,
                    "runtime_recovery_signature",
                    recovery_signature,
                )
                setattr(
                    recovery_edge,
                    "runtime_recovery_origin_edge_id",
                    failure_context.get("origin_edge_id", edge.id),
                )
                setattr(
                    recovery_edge,
                    "runtime_recovery_failure_class",
                    failure_context.get("failure_class"),
                )
            branch_final_target = recovery_edges[-1].target
            if branch_final_target not in {edge.source, edge.target}:
                print(
                    "Runtime recovery branch rejected: "
                    f"final target '{branch_final_target}' does not merge into "
                    f"'{edge.source}' or '{edge.target}'."
                )
                return None
            return recovery_edges
        except Exception as exc:
            print(
                "Runtime recovery planner failed; falling back to static recovery: "
                f"{type(exc).__name__}: {exc}"
            )
            return None

    return planner


def _call_runtime_recovery_agent(
    recovery_agent,
    *,
    env,
    runtime_kwargs: dict[str, Any],
    task_graph: dict[str, Any],
    edge,
    failure_context: dict[str, Any],
    runtime_state: dict[str, Any],
    recovery_history: list[dict[str, Any]],
) -> str:
    agent_input = recovery_agent.get_composed_observations(
        env=env,
        regenerate=True,
        log_dir=runtime_kwargs.get("log_dir"),
        task_graph=json.dumps(task_graph, ensure_ascii=False, indent=2),
        current_edge=json.dumps(_edge_to_dict(edge), ensure_ascii=False, indent=2),
        triggered_monitor=json.dumps(failure_context, ensure_ascii=False, indent=2),
        runtime_state=json.dumps(runtime_state, ensure_ascii=False, indent=2),
        recovery_history=json.dumps(recovery_history, ensure_ascii=False, indent=2),
    )
    return recovery_agent.generate_runtime(**agent_input)


def _heuristic_runtime_recovery_spec(
    *,
    runtime_task_graph: dict[str, Any],
    edge,
    failure_context: dict[str, Any],
) -> dict[str, Any]:
    return {
        "task": runtime_task_graph.get("task", ""),
        "recovery_bindings": [
            {
                "edge_id": edge.id,
                "failure_name": "runtime_retry",
                "monitors": [_monitor_from_failure_context(failure_context)],
                "recovery": [{"type": "retry_failed_edge"}],
                "merge": "target",
                "repeat_until_success": False,
            }
        ],
    }


def _sanitize_runtime_recovery_spec(
    recovery_spec: Any,
    *,
    current_edge_id: str,
    edge: Any | None = None,
    failure_context: dict[str, Any] | None = None,
    runtime_state: dict[str, Any] | None = None,
    recovery_history: list[dict[str, Any]] | None = None,
    nominal_edges: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    spec = extract_json_object(recovery_spec or {})
    bindings = spec.get("recovery_bindings", spec.get("bindings", []))
    if not isinstance(bindings, list):
        return None

    sanitized_bindings = []
    for binding in bindings:
        if not isinstance(binding, dict):
            continue
        edge_id = str(binding.get("edge_id", binding.get("edge", "")))
        if edge_id and edge_id != current_edge_id:
            continue

        sanitized_steps = []
        for step in binding.get("recovery", binding.get("recovery_steps", [])):
            sanitized_step_list = _sanitize_runtime_recovery_step(
                step,
                current_edge_id=current_edge_id,
                edge=edge,
                failure_context=failure_context,
                runtime_state=runtime_state,
                recovery_history=recovery_history,
                nominal_edges=nominal_edges or {},
            )
            if sanitized_step_list is None:
                return None
            sanitized_steps.extend(sanitized_step_list)
        if not sanitized_steps:
            continue

        inferred_merge = "target"
        final_step_type = sanitized_steps[-1].get("type")
        if final_step_type in {"regrasp", "move_to_safe_pose"}:
            inferred_merge = "source"
        elif final_step_type in {"retry_failed_edge", "replay_edge", "action"}:
            inferred_merge = "target"

        sanitized_bindings.append(
            {
                "edge_id": current_edge_id,
                "failure_name": str(binding.get("failure_name", "runtime_recovery")),
                "monitors": deepcopy(
                    binding.get("monitors", binding.get("monitor", []))
                ),
                "recovery": sanitized_steps,
                "merge": inferred_merge,
                "repeat_until_success": bool(
                    binding.get("repeat_until_success", binding.get("repeat", False))
                ),
            }
        )

    if not sanitized_bindings:
        return None
    return {
        "task": str(spec.get("task", "")),
        "recovery_bindings": sanitized_bindings[:1],
    }


def _runtime_recovery_signature(recovery_spec: dict[str, Any]) -> str:
    bindings = recovery_spec.get("recovery_bindings", [])
    if not bindings:
        return "[]"
    steps = bindings[0].get("recovery", [])
    signature_steps = []
    for step in steps:
        if not isinstance(step, dict):
            signature_steps.append({"type": str(step)})
            continue
        signature_step = {"type": step.get("type")}
        for key in (
            "robot_name",
            "obj_name",
            "arms",
            "edge_id",
            "name",
            "pre_grasp_dis",
            "force_valid",
        ):
            if key in step:
                signature_step[key] = _jsonable(step[key])
        signature_steps.append(signature_step)
    return json.dumps(
        signature_steps,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _runtime_recovery_repeats_strategy(
    recovery_signature: str,
    *,
    failure_context: dict[str, Any],
    recovery_history: list[dict[str, Any]],
    max_attempts: int,
) -> bool:
    if max_attempts <= 0:
        return False
    origin_edge_id = str(
        failure_context.get("origin_edge_id") or failure_context.get("edge_id")
    )
    failure_class = str(failure_context.get("failure_class") or "")
    matches = 0
    for attempt in recovery_history:
        if attempt.get("strategy") != "runtime":
            continue
        if (
            str(attempt.get("origin_edge_id") or attempt.get("edge_id"))
            != origin_edge_id
        ):
            continue
        if failure_class and str(attempt.get("failure_class") or "") != failure_class:
            continue
        if attempt.get("recovery_signature") != recovery_signature:
            continue
        matches += 1
    return matches >= max_attempts


def _sanitize_runtime_recovery_step(
    step: Any,
    *,
    current_edge_id: str,
    edge: Any | None = None,
    failure_context: dict[str, Any] | None = None,
    runtime_state: dict[str, Any] | None = None,
    recovery_history: list[dict[str, Any]] | None = None,
    nominal_edges: dict[str, Any] | None = None,
) -> list[dict[str, Any]] | None:
    if not isinstance(step, dict):
        step = {"type": step}
    step_type = _runtime_recovery_step_type(step.get("type", step.get("name")))
    if step_type is None:
        return None

    allowed_keys = {
        "type",
        "robot_name",
        "obj_name",
        "arms",
        "pre_grasp_dis",
        "force_valid",
        "edge_id",
        "name",
        "semantic",
    }
    sanitized = {
        key: deepcopy(value) for key, value in step.items() if key in allowed_keys
    }
    sanitized["type"] = step_type
    if step_type == "retry_failed_edge":
        return [sanitized]

    if step_type == "replay_edge":
        replay_edge_id = sanitized.get("edge_id")
        if not replay_edge_id:
            return None
        replay_edge = (nominal_edges or {}).get(str(replay_edge_id))
        if replay_edge is None:
            return None
        return [
            {
                "type": "action",
                "name": f"replay_{replay_edge_id}",
                "left_arm_action": deepcopy(replay_edge.get("left_arm_action")),
                "right_arm_action": deepcopy(replay_edge.get("right_arm_action")),
            }
        ]

    if step_type == "move_to_safe_pose":
        arms = sanitized.get("arms")
        if isinstance(arms, dict):
            sanitized["arms"] = [str(arm) for arm in arms.keys()]
        elif arms is not None:
            sanitized["arms"] = [str(arm) for arm in arms if arm]
        return [sanitized]

    if step_type == "regrasp_both":
        arms = sanitized.get("arms")
        if arms is None:
            robot_name = sanitized.get("robot_name")
            obj_name = sanitized.get("obj_name")
            if robot_name is not None and obj_name is not None:
                arms = {str(robot_name): str(obj_name)}
        if arms is None:
            return None
        if isinstance(arms, dict):
            arm_items = [(str(robot), str(obj)) for robot, obj in arms.items()]
        else:
            arm_items = []
            for arm in arms:
                if isinstance(arm, dict):
                    robot_name = arm.get("robot_name", arm.get("arm"))
                    obj_name = arm.get("obj_name", arm.get("object"))
                    if robot_name is not None and obj_name is not None:
                        arm_items.append((str(robot_name), str(obj_name)))
                elif isinstance(arm, (list, tuple)) and len(arm) >= 2:
                    arm_items.append((str(arm[0]), str(arm[1])))
        if not arm_items:
            return None
        arm_items.sort(
            key=lambda item: (
                0 if "left" in item[0] else 1 if "right" in item[0] else 2,
                item[0],
                item[1],
            )
        )
        normalized_steps = []
        for robot_name, obj_name in arm_items:
            normalized_steps.append(
                {
                    "type": "regrasp",
                    "robot_name": robot_name,
                    "obj_name": obj_name,
                    "pre_grasp_dis": sanitized.get("pre_grasp_dis", 0.1),
                    "force_valid": sanitized.get("force_valid", False),
                }
            )
        return normalized_steps

    if step_type == "regrasp":
        if "arms" in sanitized:
            arms = sanitized.get("arms")
            if isinstance(arms, dict):
                arm_items = [(str(robot), str(obj)) for robot, obj in arms.items()]
            else:
                arm_items = []
                for arm in arms:
                    if isinstance(arm, dict):
                        robot_name = arm.get("robot_name", arm.get("arm"))
                        obj_name = arm.get("obj_name", arm.get("object"))
                        if robot_name is not None and obj_name is not None:
                            arm_items.append((str(robot_name), str(obj_name)))
                    elif isinstance(arm, (list, tuple)) and len(arm) >= 2:
                        arm_items.append((str(arm[0]), str(arm[1])))
            if not arm_items:
                return None
            normalized_steps = []
            for robot_name, obj_name in arm_items:
                normalized_steps.append(
                    {
                        "type": "regrasp",
                        "robot_name": robot_name,
                        "obj_name": obj_name,
                        "pre_grasp_dis": sanitized.get("pre_grasp_dis", 0.1),
                        "force_valid": sanitized.get("force_valid", False),
                    }
                )
            return normalized_steps

        if not sanitized.get("robot_name") or not sanitized.get("obj_name"):
            return None
        return [sanitized]

    return [sanitized]


def _runtime_recovery_step_type(value: Any) -> str | None:
    aliases = {
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
        "move_to_safe_pose": "move_to_safe_pose",
        "safe_pose": "move_to_safe_pose",
        "move_safe": "move_to_safe_pose",
    }
    if value is None:
        return None
    return aliases.get(str(value))


def _first_runtime_branch_edges(
    recovery_graph: dict[str, Any],
    *,
    current_edge_id: str,
) -> list[str]:
    branches = recovery_graph.get("recovery_branches", [])
    if not branches:
        return []
    for branch in branches:
        if branch.get("edge_id") == current_edge_id:
            return list(branch.get("recovery_edges", []))
    return list(branches[0].get("recovery_edges", []))


def _task_graph_with_runtime_edge(
    task_graph: dict[str, Any],
    edge,
) -> dict[str, Any]:
    runtime_task_graph = {
        "task": task_graph.get("task", ""),
        "start": edge.source,
        "goal": edge.target if edge.target != edge.source else edge.source,
        "nodes": [
            {"id": edge.source, "semantic": ""},
            {"id": edge.target, "semantic": ""} if edge.target != edge.source else None,
        ],
        "edges": [_edge_to_dict(edge)],
    }
    runtime_task_graph["nodes"] = [
        node for node in runtime_task_graph["nodes"] if node is not None
    ]
    return runtime_task_graph


def _edge_to_dict(edge) -> dict[str, Any]:
    return {
        "id": edge.id,
        "source": edge.source,
        "target": edge.target,
        "left_arm_action": _action_to_jsonable(getattr(edge, "left_arm_action", None)),
        "right_arm_action": _action_to_jsonable(
            getattr(edge, "right_arm_action", None)
        ),
        "is_recovery": bool(getattr(edge, "is_recovery", False)),
    }


def _action_to_jsonable(action: Any) -> Any:
    if action is None:
        return None
    return _jsonable(getattr(action, "spec", action))


def _collect_runtime_state(env) -> dict[str, Any]:
    env = getattr(env, "unwrapped", env)
    if hasattr(env, "update_obj_info"):
        try:
            env.update_obj_info()
        except Exception:
            pass

    objects = {}
    obj_info = getattr(env, "obj_info", {}) or {}
    for obj_name, info in obj_info.items():
        if obj_name == "table":
            continue
        objects[str(obj_name)] = {
            "pose": _jsonable(info.get("pose")),
            "initial_pose": _jsonable(info.get("initial_pose")),
            "height": _jsonable(info.get("height")),
            "has_grasp_pose": info.get("grasp_pose_obj") is not None,
        }

    arms = {}
    for arm_name, is_left in (("left_arm", True), ("right_arm", False)):
        arm_state = {
            "qpos": _jsonable(
                getattr(
                    env,
                    "left_arm_current_qpos" if is_left else "right_arm_current_qpos",
                    None,
                )
            ),
            "eef_pose": _jsonable(
                getattr(
                    env,
                    "left_arm_current_xpos" if is_left else "right_arm_current_xpos",
                    None,
                )
            ),
            "cached_gripper_state": _jsonable(
                getattr(
                    env,
                    (
                        "left_arm_current_gripper_state"
                        if is_left
                        else "right_arm_current_gripper_state"
                    ),
                    None,
                )
            ),
        }
        try:
            from embodichain.lab.sim.agent.monitor_utils import get_gripper_distance

            arm_state["gripper_distance"] = get_gripper_distance(env, arm_name)
        except Exception:
            arm_state["gripper_distance"] = None
        arms[arm_name] = arm_state

    hold_states = []
    for arm_name in ("left_arm", "right_arm"):
        for obj_name in objects:
            try:
                from embodichain.lab.sim.agent.monitor_utils import (
                    get_arm_object_distance,
                )

                distance = get_arm_object_distance(env, arm_name, obj_name)
            except Exception:
                distance = None
            hold_states.append(
                {
                    "robot_name": arm_name,
                    "obj_name": obj_name,
                    "distance": _jsonable(distance),
                }
            )

    return {"objects": objects, "arms": arms, "hold_states": hold_states}


def _monitor_from_failure_context(context: dict[str, Any]) -> dict[str, Any]:
    monitor_name = str(context.get("monitor_name") or "monitor_object_moved")
    if monitor_name == "monitor_object_held":
        return {"type": "hold_lost"}
    if monitor_name == "monitor_object_fallen":
        return {"type": "object_fallen"}
    return {"type": "object_moved"}


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    detach = getattr(value, "detach", None)
    if callable(detach):
        return _jsonable(value.detach().cpu().tolist())
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _jsonable(tolist())
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _jsonable(item())
        except Exception:
            pass
    return str(value)


def _empty_recovery_spec(task_graph: dict[str, Any]) -> dict[str, Any]:
    return {
        "task": task_graph.get("task", ""),
        "recovery_bindings": [],
    }


def _stable_json_hash(content: dict[str, Any]) -> str:
    payload = json.dumps(
        content, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _runtime_kwargs(
    kwargs: dict[str, Any],
    prompt_kwargs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    prompt_only_keys = set(prompt_kwargs)
    prompt_only_keys.update(
        {
            "task_graph",
            "recovery_spec",
            "recovery_graph",
            "recovery_enabled",
            "observations",
            "regenerate",
        }
    )
    return {key: value for key, value in kwargs.items() if key not in prompt_only_keys}
