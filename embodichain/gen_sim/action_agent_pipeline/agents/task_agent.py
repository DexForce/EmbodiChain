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
    normalize_json_content,
)
from embodichain.gen_sim.action_agent_pipeline.utils.timing import timing_scope
from embodichain.gen_sim.action_agent_pipeline.prompts import TaskPrompt
from embodichain.data import database_agent_prompt_dir
from embodichain.utils.logger import log_info
from embodichain.utils.utility import load_txt

__all__ = ["TaskAgent"]

TASK_GRAPH_CACHE_SCHEMA_VERSION = "task_graph_prompt_v1"


class TaskAgent(AgentBase):
    """Generate the nominal atomic-action task graph."""

    prompt_name: str
    prompt_kwargs: dict[str, dict[str, Any]]
    precomputed_task_graph: str | None

    def __init__(self, llm: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if llm is None and not getattr(self, "precomputed_task_graph", None):
            raise ValueError(
                "LLM is None. Configure the shared MLLM entry point "
                "`embodichain.gen_sim.action_agent_pipeline.utils.mllm` with "
                "OPENAI_API_KEY, optional "
                "OPENAI_MODEL/OPENAI_BASE_URL, or the gen-sim LLM config."
            )
        self.llm = llm

    def generate(self, **kwargs) -> str:
        log_dir = kwargs.get(
            "log_dir", Path(database_agent_prompt_dir) / self.task_name
        )
        file_path = Path(log_dir) / "agent_task_graph.json"
        metadata_path = file_path.with_suffix(".metadata.json")
        precomputed_path = _resolve_precomputed_task_graph_path(self)
        if precomputed_path is not None:
            with timing_scope("action_agent.task_graph.precomputed_read"):
                content = normalize_json_content(
                    precomputed_path.read_text(encoding="utf-8")
                )
            with timing_scope("action_agent.task_graph.cache_write"):
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(content, encoding="utf-8")
                _write_metadata(
                    metadata_path,
                    prompt_hash=_stable_text_hash(content),
                    prompt_name="precomputed_task_graph",
                    task_name=self.task_name,
                )
            log_info(
                f"Using precomputed task graph from {precomputed_path}; "
                f"cached at {file_path}."
            )
            log_info(
                f"Task agent output (precomputed):\n```json\n{content}\n```",
                color="green",
            )
            return content

        with timing_scope(
            "action_agent.task_graph.prompt_build",
            metadata={"prompt_name": self.prompt_name},
        ):
            prompt = getattr(TaskPrompt, self.prompt_name)(**kwargs)
            prompt_hash = _stable_text_hash(prompt)

        with timing_scope(
            "action_agent.task_graph.cache_lookup",
            metadata={"regenerate": bool(kwargs.get("regenerate", False))},
        ):
            cache_hit = (
                not kwargs.get("regenerate", False)
                and file_path.exists()
                and _metadata_matches(
                    metadata_path,
                    prompt_hash=prompt_hash,
                    prompt_name=self.prompt_name,
                    task_name=self.task_name,
                )
            )

        if cache_hit:
            log_info(f"Task graph already exists at {file_path}.")
            with timing_scope("action_agent.task_graph.cache_read"):
                return load_txt(file_path)

        with timing_scope(
            "action_agent.task_graph.llm_invoke",
            metadata={"prompt_name": self.prompt_name},
        ):
            response = self.llm.invoke(prompt)
        log_info(f"Task agent output:\n{response.content}", color="green")

        with timing_scope("action_agent.task_graph.output_parse"):
            content = normalize_json_content(response.content)
        with timing_scope("action_agent.task_graph.cache_write"):
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")
            _write_metadata(
                metadata_path,
                prompt_hash=prompt_hash,
                prompt_name=self.prompt_name,
                task_name=self.task_name,
            )
        log_info(f"Generated task graph saved to {file_path}")

        return content

    def act(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("TaskAgent only generates task graphs.")


def _stable_text_hash(content: Any) -> str:
    text = _prompt_to_hash_text(content)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _prompt_to_hash_text(prompt: Any) -> str:
    to_string = getattr(prompt, "to_string", None)
    if callable(to_string):
        return str(to_string())
    return str(prompt)


def _metadata_matches(
    metadata_path: Path,
    *,
    prompt_hash: str,
    prompt_name: str,
    task_name: str,
) -> bool:
    if not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if not isinstance(metadata, dict):
        return False
    return (
        metadata.get("schema_version") == TASK_GRAPH_CACHE_SCHEMA_VERSION
        and metadata.get("prompt_hash") == prompt_hash
        and metadata.get("prompt_name") == prompt_name
        and metadata.get("task_name") == task_name
    )


def _write_metadata(
    metadata_path: Path,
    *,
    prompt_hash: str,
    prompt_name: str,
    task_name: str,
) -> None:
    metadata = {
        "schema_version": TASK_GRAPH_CACHE_SCHEMA_VERSION,
        "prompt_hash": prompt_hash,
        "prompt_name": prompt_name,
        "task_name": task_name,
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _resolve_precomputed_task_graph_path(agent: TaskAgent) -> Path | None:
    graph_name = getattr(agent, "precomputed_task_graph", None)
    if not graph_name:
        return None

    graph_path = Path(graph_name).expanduser()
    if graph_path.is_absolute():
        return graph_path

    config_dir = getattr(agent, "config_dir", None)
    if config_dir:
        config_base = Path(config_dir).expanduser()
        if config_base.suffix:
            config_base = config_base.parent
        return (config_base / graph_path).resolve()
    return graph_path.resolve()
