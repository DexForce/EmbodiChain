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

from pathlib import Path
from typing import Any

from embodichain.gen_sim.action_agent_pipeline.protocol.artifacts import (
    SEED_TASK_GRAPH_FILENAME,
    TASK_GRAPH_FILENAME,
)
from embodichain.gen_sim.action_agent_pipeline.utils.llm_json import (
    extract_json_object,
)
from embodichain.utils.logger import log_info

__all__ = [
    "CompileAgent",
    "resolve_precomputed_seed_task_graph_path",
]


class CompileAgent:
    """Validate and execute immutable Seed Graph v3 specs in memory."""

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
        from embodichain.gen_sim.action_agent_pipeline.domain.seed_task_graph import (
            validate_seed_task_graph,
        )

        if "task_graph" in kwargs:
            raise ValueError(
                "precomputed task_graph input is no longer supported. Regenerate "
                "the action-agent config with --overwrite."
            )
        if "seed_task_graph" not in kwargs:
            raise ValueError(
                "CompileAgent requires seed_task_graph_v3. Regenerate the "
                "action-agent config with --overwrite."
            )
        seed_graph = extract_json_object(kwargs["seed_task_graph"])
        validate_seed_task_graph(seed_graph, task_name=self.task_name)
        log_info("Validated executable Seed Graph v3 for runtime grounding.")
        return seed_graph, kwargs, None

    def act(self, seed_graph, **kwargs: Any):
        from embodichain.gen_sim.action_agent_pipeline.runtime.graph_compiler import (
            compile_agent_graph_spec,
        )

        runtime_kwargs = _runtime_kwargs(kwargs)
        graph = compile_agent_graph_spec(seed_graph)
        result = graph.run(**runtime_kwargs)
        log_info("Executable Seed Graph v3 completed runtime execution.")
        return result


def resolve_precomputed_seed_task_graph_path(
    *,
    configured_path: str | None,
    agent_config_path: str | None,
) -> Path:
    """Resolve the required executable Seed v3 runtime input."""
    config_file = (
        Path(agent_config_path).expanduser().resolve() if agent_config_path else None
    )
    if configured_path:
        seed_path = Path(configured_path).expanduser()
        if not seed_path.is_absolute():
            seed_path = (
                config_file.parent / seed_path
                if config_file is not None
                else seed_path.resolve()
            )
        resolved = seed_path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(
                f"Configured seed task graph not found: {resolved}."
            )
        return resolved

    adjacent = (
        config_file.parent / SEED_TASK_GRAPH_FILENAME
        if config_file is not None
        else Path(SEED_TASK_GRAPH_FILENAME).resolve()
    )
    if not adjacent.is_file():
        legacy = adjacent.with_name(TASK_GRAPH_FILENAME)
        if legacy.is_file():
            raise ValueError(
                "Found legacy task_graph.json without Seed v3. Regenerate the "
                "action-agent config with --overwrite."
            )
        raise FileNotFoundError(
            f"Executable Seed task graph not found: {adjacent}. Generate the "
            "action-agent config before running."
        )
    return adjacent


def _runtime_kwargs(
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    prompt_only_keys = {
        "seed_task_graph",
        "observations",
        "regenerate",
    }
    return {key: value for key, value in kwargs.items() if key not in prompt_only_keys}
