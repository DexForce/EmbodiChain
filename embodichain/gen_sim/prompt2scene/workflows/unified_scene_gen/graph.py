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

from embodichain.gen_sim.prompt2scene.llms import build_chat_model
from embodichain.gen_sim.prompt2scene.llms.config import OpenAICompatibleLLMCfg
from embodichain.gen_sim.prompt2scene.workflows.unified_scene_gen.nodes import (
    fit_image_table_to_clutter_node,
    generate_image_assets_node,
    load_unified_scene_input_kind_node,
)
from embodichain.gen_sim.prompt2scene.workflows.unified_scene_gen.state import (
    UnifiedSceneGenState,
)
__all__ = [
    "build_unified_scene_gen_graph",
    "run_unified_scene_gen",
]


def build_unified_scene_gen_graph() -> Any:
    """Build the unified-scene generation graph."""
    graph = StateGraph(UnifiedSceneGenState)
    graph.add_node("load_unified_scene_input_kind", load_unified_scene_input_kind_node)
    graph.add_node("generate_image_assets", generate_image_assets_node)
    graph.add_node("fit_image_table_to_clutter", fit_image_table_to_clutter_node)

    graph.set_entry_point("load_unified_scene_input_kind")
    graph.add_edge("load_unified_scene_input_kind", "generate_image_assets")
    graph.add_edge("generate_image_assets", "fit_image_table_to_clutter")
    graph.add_edge("fit_image_table_to_clutter", END)
    return graph.compile()


def run_unified_scene_gen(
    output_root: Path,
    *,
    unified_scene_result_path: Path | None = None,
    llm_cfg: OpenAICompatibleLLMCfg | None = None,
) -> UnifiedSceneGenState:
    """Run downstream generation routing from a unified-scene result."""
    llm = build_chat_model(llm_cfg) if llm_cfg is not None else None
    initial_state: UnifiedSceneGenState = {
        "output_root": output_root,
        "unified_scene_result_path": unified_scene_result_path,
        "llm": llm,
        "unified_scene": None,
        "input_kind": None,
        "table_result": None,
        "image_object_results": [],
        "image_objects_layout_result": None,
        "table_fit_result": None,
        "generation_status": None,
        "attempt_count": 0,
        "max_attempts": 1,
        "last_error": None,
        "errors": [],
    }
    return build_unified_scene_gen_graph().invoke(initial_state)
