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

from langgraph.graph import END, StateGraph

from embodichain.gen_sim.prompt2scene.llms import (
    OpenAICompatibleLLMCfg,
    build_chat_model,
)
from embodichain.gen_sim.prompt2scene.utils import log
from embodichain.gen_sim.prompt2scene.workflows.stage_errors import (
    format_result_missing_error,
)
from embodichain.gen_sim.prompt2scene.workflows.scene_intake.schema import (
    SceneIntakeSpec,
)
from embodichain.gen_sim.prompt2scene.workflows.text_relations.nodes import (
    call_llm_text_relations_node,
    normalize_text_relations_node,
    prepare_text_relation_messages_node,
)
from embodichain.gen_sim.prompt2scene.workflows.text_relations.schema import (
    TextRelationSpec,
)
from embodichain.gen_sim.prompt2scene.workflows.text_relations.state import (
    TextRelationsState,
)
from embodichain.gen_sim.prompt2scene.workflows.request import Prompt2SceneInput

__all__ = ["build_text_relations_graph", "run_text_relations"]


def route_after_text_relation_normalize(state: TextRelationsState) -> str:
    """Route to retry or finish after text relation normalization."""
    if state["text_relations"] is not None:
        return "end"
    if state["attempt_count"] < state["max_attempts"]:
        return "retry"
    return "end"


def build_text_relations_graph(llm: Any) -> Any:
    """Build the fixed text spatial-relation extraction workflow."""
    graph = StateGraph(TextRelationsState)
    graph.add_node(
        "prepare_text_relation_messages",
        prepare_text_relation_messages_node,
    )
    graph.add_node(
        "call_llm_text_relations",
        lambda state: call_llm_text_relations_node(state, llm=llm),
    )
    graph.add_node("normalize_text_relations", normalize_text_relations_node)

    graph.set_entry_point("prepare_text_relation_messages")
    graph.add_edge("prepare_text_relation_messages", "call_llm_text_relations")
    graph.add_edge("call_llm_text_relations", "normalize_text_relations")
    graph.add_conditional_edges(
        "normalize_text_relations",
        route_after_text_relation_normalize,
        {
            "retry": "call_llm_text_relations",
            "end": END,
        },
    )
    return graph.compile()


def run_text_relations(
    request: Prompt2SceneInput,
    *,
    scene_intake: SceneIntakeSpec,
    llm_cfg: OpenAICompatibleLLMCfg,
    output_root: Path,
) -> TextRelationSpec:
    """Run text spatial-relation extraction for one prompt2scene request."""
    llm = build_chat_model(llm_cfg)
    graph = build_text_relations_graph(llm)
    result = graph.invoke(
        {
            "request": request,
            "scene_intake": scene_intake,
            "output_root": output_root,
            "messages": [],
            "raw_model_output": None,
            "text_relations": None,
            "attempt_count": 0,
            "max_attempts": llm_cfg.max_attempts,
            "last_error": None,
            "errors": [],
        }
    )

    text_relations = result.get("text_relations")
    if text_relations is not None:
        return text_relations

    error = format_result_missing_error(
        "Text relations",
        "TextRelationSpec",
        attempt_count=result.get("attempt_count", 0),
        last_error=result.get("last_error"),
        errors=result.get("errors", []),
    )
    log.log_warning(error)
    raise RuntimeError(error)
