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

from embodichain.gen_sim.action_agent_pipeline.agents.agent_base import AgentBase
from embodichain.gen_sim.action_agent_pipeline.utils.llm_json import (
    extract_json_object,
    normalize_json_content,
)
from embodichain.data import database_agent_prompt_dir

__all__ = ["CompileAgent"]

COMPILED_GRAPH_SCHEMA_VERSION = "nominal_graph_v1"


class CompileAgent(AgentBase):
    """Compile and execute nominal atomic-action graph specs."""

    query_prefix = "# "
    query_suffix = "."
    prompt_kwargs: dict[str, dict[str, Any]]

    def __init__(self, llm, **kwargs) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.prompt_kwargs = kwargs.get("prompt_kwargs", {})
        self.llm = llm

    def generate(self, **kwargs):
        if kwargs.get("recovery_enabled") or kwargs.get("recovery_spec"):
            raise NotImplementedError("Recovery graph generation has been removed.")

        log_dir = kwargs.get(
            "log_dir", Path(database_agent_prompt_dir) / self.task_name
        )
        file_path = Path(log_dir) / "agent_compiled_graph.json"
        task_graph = extract_json_object(kwargs["task_graph"])
        task_graph_hash = _stable_json_hash(task_graph)

        if not kwargs.get("regenerate", False) and file_path.exists():
            existing_bundle = extract_json_object(file_path.read_text(encoding="utf-8"))
            metadata = existing_bundle.get("metadata", {})
            if (
                metadata.get("schema_version") == COMPILED_GRAPH_SCHEMA_VERSION
                and metadata.get("task_graph_hash") == task_graph_hash
            ):
                print(f"Compiled graph artifact already exists at {file_path}.")
                return file_path, kwargs, None

        content = normalize_json_content(
            {
                "task_graph": task_graph,
                "metadata": {
                    "schema_version": COMPILED_GRAPH_SCHEMA_VERSION,
                    "task_graph_hash": task_graph_hash,
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

        from embodichain.gen_sim.action_agent_pipeline.runtime.graph_compiler import (
            compile_agent_graph_from_file,
        )

        runtime_kwargs = _runtime_kwargs(kwargs, getattr(self, "prompt_kwargs", {}))
        graph = compile_agent_graph_from_file(graph_file_path)
        result = graph.run(**runtime_kwargs)
        print("Compiled agent graph executed successfully.")
        return result

    def get_composed_observations(self, **kwargs):
        return dict(kwargs)


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
