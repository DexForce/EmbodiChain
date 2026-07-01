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
from embodichain.gen_sim.prompt2scene.workflows.scene_intake.nodes import (
    call_vlm_scene_intake_node,
    call_vlm_verify_scene_intake_node,
    normalize_scene_intake_node,
    normalize_verified_scene_intake_node,
    prepare_input_node,
)
from embodichain.gen_sim.prompt2scene.workflows.scene_intake.state import (
    SceneIntakeState,
)
from embodichain.gen_sim.prompt2scene.workflows.request import Prompt2SceneInput

__all__ = ["build_scene_intake_graph", "run_scene_intake"]


def route_after_normalize(state: SceneIntakeState) -> str:
    """Route to retry or verify after draft scene intake normalization."""
    if state["draft_scene_intake"] is not None:
        return "verify"
    if state["attempt_count"] < state["max_attempts"]:
        return "retry"
    return "end"


def route_after_verified_normalize(state: SceneIntakeState) -> str:
    """Route to retry or finish after scene intake verifier normalization."""
    if state["scene_intake"] is not None:
        return "end"
    if state["attempt_count"] < state["max_attempts"]:
        return "retry"
    return "end"


def build_scene_intake_graph(llm: Any) -> Any:
    """Build the fixed LangGraph scene intake workflow."""
    graph = StateGraph(SceneIntakeState)
    graph.add_node("prepare_input", prepare_input_node)
    graph.add_node(
        "call_vlm_scene_intake",
        lambda state: call_vlm_scene_intake_node(state, llm=llm),
    )
    graph.add_node("normalize_scene_intake", normalize_scene_intake_node)
    graph.add_node(
        "call_vlm_verify_scene_intake",
        lambda state: call_vlm_verify_scene_intake_node(state, llm=llm),
    )
    graph.add_node(
        "normalize_verified_scene_intake",
        normalize_verified_scene_intake_node,
    )

    graph.set_entry_point("prepare_input")
    graph.add_edge("prepare_input", "call_vlm_scene_intake")
    graph.add_edge("call_vlm_scene_intake", "normalize_scene_intake")
    graph.add_conditional_edges(
        "normalize_scene_intake",
        route_after_normalize,
        {
            "retry": "call_vlm_scene_intake",
            "verify": "call_vlm_verify_scene_intake",
            "end": END,
        },
    )
    graph.add_edge("call_vlm_verify_scene_intake", "normalize_verified_scene_intake")
    graph.add_conditional_edges(
        "normalize_verified_scene_intake",
        route_after_verified_normalize,
        {
            "retry": "call_vlm_verify_scene_intake",
            "end": END,
        },
    )
    return graph.compile()


def run_scene_intake(
    request: Prompt2SceneInput,
    llm_cfg: OpenAICompatibleLLMCfg,
) -> SceneIntakeSpec:
    """Run fixed VLM-based scene intake for one prompt2scene request."""
    llm = build_chat_model(llm_cfg)
    graph = build_scene_intake_graph(llm)
    result = graph.invoke(
        {
            "request": request,
            "messages": [],
            "raw_model_output": None,
            "draft_scene_intake": None,
            "scene_intake": None,
            "attempt_count": 0,
            "max_attempts": llm_cfg.max_attempts,
            "last_error": None,
            "errors": [],
        }
    )

    scene_intake = result.get("scene_intake")
    if scene_intake is not None:
        return scene_intake

    error = format_result_missing_error(
        "Scene intake",
        "SceneIntakeSpec",
        attempt_count=result.get("attempt_count", 0),
        last_error=result.get("last_error"),
        errors=result.get("errors", []),
    )
    log.log_warning(error)
    raise RuntimeError(error)
