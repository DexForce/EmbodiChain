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

from embodichain.gen_sim.action_agent_pipeline.contracts import (
    COMPILED_GRAPH_FILENAME,
    TASK_GRAPH_FILENAME,
)
from embodichain.gen_sim.action_agent_pipeline.utils.llm_json import (
    extract_json_object,
    normalize_json_content,
)
from embodichain.data import database_agent_prompt_dir
from embodichain.utils.logger import log_info

__all__ = ["CompileAgent", "resolve_precomputed_task_graph_path"]

COMPILED_GRAPH_SCHEMA_VERSION = "nominal_graph_v1"


class CompileAgent:
    """Compile and execute nominal atomic-action graph specs."""

    def __init__(
        self,
        *,
        task_name: str,
        config_dir: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.task_name = task_name
        self.config_dir = config_dir
        for key, value in kwargs.items():
            setattr(self, key, value)

    def generate(self, **kwargs: Any):
        file_path = self._compiled_graph_path(kwargs.get("log_dir"))
        task_graph = extract_json_object(kwargs["task_graph"])
        task_graph_hash = _stable_json_hash(task_graph)

        # The cache is valid only for the exact source graph and compiler
        # schema. ``regenerate`` bypasses this artifact cache; it never mutates
        # the generation-stage task_graph.json supplied by the caller.
        if not kwargs.get("regenerate", False) and file_path.exists():
            existing_bundle = extract_json_object(file_path.read_text(encoding="utf-8"))
            metadata = existing_bundle.get("metadata", {})
            if (
                metadata.get("schema_version") == COMPILED_GRAPH_SCHEMA_VERSION
                and metadata.get("task_graph_hash") == task_graph_hash
            ):
                log_info(f"Compiled graph artifact already exists at {file_path}.")
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
        log_info(f"Compiled graph artifact saved to {file_path}")
        return file_path, kwargs, content

    def _compiled_graph_path(self, log_dir: str | Path | None) -> Path:
        """Resolve the compiled artifact beside its owning config by default.

        An explicit ``log_dir`` remains authoritative for compatibility.
        Callers without an agent config retain the historical global fallback,
        while normal CLI runs produce a self-contained config directory.
        """
        if log_dir is not None:
            directory = Path(log_dir).expanduser()
        elif self.config_dir:
            directory = Path(self.config_dir).expanduser()
            # Older direct callers sometimes passed agent_config.json itself
            # despite the historical ``config_dir`` name.
            if directory.suffix.lower() == ".json":
                directory = directory.parent
        else:
            directory = Path(database_agent_prompt_dir) / self.task_name
        return directory / COMPILED_GRAPH_FILENAME

    def act(self, graph_file_path, **kwargs: Any):
        graph_file_path = Path(graph_file_path)
        if graph_file_path.suffix != ".json":
            raise ValueError("CompileAgent executes compiled graph JSON artifacts.")

        from embodichain.gen_sim.action_agent_pipeline.runtime.graph_compiler import (
            compile_agent_graph_from_file,
        )

        runtime_kwargs = _runtime_kwargs(kwargs)
        graph = compile_agent_graph_from_file(graph_file_path)
        result = graph.run(**runtime_kwargs)
        log_info("Compiled agent graph executed successfully.")
        return result


def resolve_precomputed_task_graph_path(
    *,
    configured_path: str | None,
    agent_config_path: str | None,
) -> Path:
    """Resolve the immutable task graph consumed by the runtime.

    An explicit ``precomputed_task_graph`` remains authoritative. Legacy
    configs without that field fall back to ``task_graph.json`` beside the
    agent config, which preserves compatibility without reviving online graph
    generation.

    Args:
        configured_path: Path from ``TaskAgent.precomputed_task_graph``.
        agent_config_path: Path to the loaded ``agent_config.json``.

    Returns:
        The resolved task graph path.

    Raises:
        FileNotFoundError: If no configured or adjacent task graph exists.
    """
    candidates: list[Path] = []
    config_file = (
        Path(agent_config_path).expanduser().resolve() if agent_config_path else None
    )

    if configured_path:
        graph_path = Path(configured_path).expanduser()
        if not graph_path.is_absolute():
            graph_path = (
                config_file.parent / graph_path
                if config_file is not None
                else graph_path.resolve()
            )
        candidates.append(graph_path.resolve())
    elif config_file is not None:
        candidates.append(config_file.parent / TASK_GRAPH_FILENAME)
    else:
        candidates.append(Path(TASK_GRAPH_FILENAME).resolve())

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    searched = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        "Precomputed task graph not found. Searched:\n"
        f"{searched}\n"
        "Generate the action-agent config before running the agent."
    )


def _stable_json_hash(content: dict[str, Any]) -> str:
    payload = json.dumps(
        content, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _runtime_kwargs(
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    prompt_only_keys = {"task_graph", "observations", "regenerate"}
    return {key: value for key, value in kwargs.items() if key not in prompt_only_keys}
