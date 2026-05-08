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
from pathlib import Path
from typing import Any

from embodichain.agents.hierarchy.agent_base import AgentBase
from embodichain.data import database_agent_prompt_dir
from embodichain.utils.llm_json import extract_json_object, normalize_json_content

__all__ = ["CompileAgent"]

COMPILED_GRAPH_SCHEMA_VERSION = "recovery_bindings_v1"


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
