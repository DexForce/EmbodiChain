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

"""Compile executable Seed Graph v5 into the live runtime graph."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import importlib
from pathlib import Path
from typing import Any

from embodichain.gen_sim.action_agent_pipeline.domain.seed_task_graph import (
    validate_seed_task_graph,
)
from embodichain.gen_sim.action_agent_pipeline.utils.llm_json import extract_json_object

__all__ = [
    "compile_agent_graph_from_file",
    "compile_agent_graph_spec",
    "load_agent_graph_bundle",
]


def load_agent_graph_bundle(path: str | Path) -> dict[str, Any]:
    """Load a Seed graph from disk and reject removed compiled bundles."""
    spec = extract_json_object(Path(path).read_text(encoding="utf-8"))
    if "task_graph" in spec or "metadata" in spec:
        raise ValueError(
            "Compiled/precomputed task graphs are no longer supported. Regenerate "
            "the action-agent config with --overwrite."
        )
    return spec


def compile_agent_graph_from_file(
    path: str | Path,
    *,
    graph_cls: type | None = None,
    action_module: Any = None,
) -> Any:
    """Compile Seed v5 from disk into an executable runtime graph."""
    return compile_agent_graph_spec(
        load_agent_graph_bundle(path),
        graph_cls=graph_cls,
        action_module=action_module,
    )


def compile_agent_graph_spec(
    seed_graph: str | Mapping[str, Any],
    *,
    graph_cls: type | None = None,
    action_module: Any = None,
) -> Any:
    """Compile a validated Seed v5 mapping without grounding its actions."""
    del action_module
    seed_spec = extract_json_object(seed_graph)
    validate_seed_task_graph(seed_spec)
    if graph_cls is None:
        graph_cls = getattr(
            importlib.import_module(
                "embodichain.gen_sim.action_agent_pipeline.runtime.task_graph"
            ),
            "AgentTaskGraph",
        )
    graph = graph_cls(
        start=seed_spec["start"],
        goal=seed_spec["goal"],
        max_transitions=len(seed_spec["edges"]) + 1,
        seed_graph=seed_spec,
    )
    for node in seed_spec["nodes"]:
        graph.add_node(node["id"], node.get("semantic", ""))
    for edge in seed_spec["edges"]:
        graph.add_edge(
            edge["id"],
            edge["source"],
            edge["target"],
            symbolic_actions=edge["actions"],
            depends_on=edge["depends_on"],
            resources=edge["resources"],
        )
    for step in seed_spec["semantic_steps"]:
        graph.add_semantic_step(
            step["id"],
            operator=step["operator"],
            object_uid=step["object"],
            actor=step["actor"],
            goal=step["goal"],
            depends_on=step["depends_on"],
            postcondition=step["postcondition"],
            edge_ids=step["edge_ids"],
        )
    return graph
