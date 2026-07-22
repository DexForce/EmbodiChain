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
from embodichain.gen_sim.prompt2scene.workflows.image_relations.schema import (
    ImageRelationSpec,
)
from embodichain.gen_sim.prompt2scene.workflows.image_relations.nodes import (
    call_vlm_filter_initial_segments_node,
    call_vlm_spatial_layout_node,
    normalize_asset_segments_node,
    prepare_segmentation_input_node,
    retry_missing_by_candidates_node,
    segment_table_node,
    segment_by_name_node,
)
from embodichain.gen_sim.prompt2scene.workflows.scene_intake.schema import (
    SceneIntakeSpec,
)
from embodichain.gen_sim.prompt2scene.workflows.request import Prompt2SceneInput
from embodichain.gen_sim.prompt2scene.workflows.image_relations.state import (
    ImageRelationsState,
)

__all__ = ["build_image_relations_graph", "run_image_relations"]


def route_after_filter_extra_instances(state: ImageRelationsState) -> str:
    """Route to retry or continue after VLM extra-instance filtering."""
    if state["last_error"] is None:
        return "continue"
    if state["attempt_count"] < state["max_attempts"]:
        return "retry"
    return "continue"


def route_after_spatial_layout(state: ImageRelationsState) -> str:
    """Route to retry or finish after spatial-layout extraction."""
    if state["last_error"] is None:
        return "end"
    if state["attempt_count"] < state["max_attempts"]:
        return "retry"
    return "end"


def build_image_relations_graph(llm: Any) -> Any:
    """Build the fixed LangGraph image asset segmentation workflow."""
    graph = StateGraph(ImageRelationsState)
    graph.add_node("prepare_segmentation_input", prepare_segmentation_input_node)
    graph.add_node("segment_by_name", segment_by_name_node)
    graph.add_node(
        "call_vlm_filter_initial_segments",
        lambda state: call_vlm_filter_initial_segments_node(state, llm=llm),
    )
    graph.add_node(
        "retry_missing_by_candidates",
        lambda state: retry_missing_by_candidates_node(state, llm=llm),
    )
    graph.add_node("normalize_asset_segments", normalize_asset_segments_node)
    graph.add_node(
        "segment_table",
        lambda state: segment_table_node(state, llm=llm),
    )
    graph.add_node(
        "call_vlm_spatial_layout",
        lambda state: call_vlm_spatial_layout_node(state, llm=llm),
    )

    graph.set_entry_point("prepare_segmentation_input")
    graph.add_edge("prepare_segmentation_input", "segment_by_name")
    graph.add_edge("segment_by_name", "call_vlm_filter_initial_segments")
    graph.add_conditional_edges(
        "call_vlm_filter_initial_segments",
        route_after_filter_extra_instances,
        {
            "retry": "call_vlm_filter_initial_segments",
            "continue": "retry_missing_by_candidates",
        },
    )
    graph.add_edge("retry_missing_by_candidates", "normalize_asset_segments")
    graph.add_edge("normalize_asset_segments", "segment_table")
    graph.add_edge("segment_table", "call_vlm_spatial_layout")
    graph.add_conditional_edges(
        "call_vlm_spatial_layout",
        route_after_spatial_layout,
        {
            "retry": "call_vlm_spatial_layout",
            "end": END,
        },
    )
    return graph.compile()


def run_image_relations(
    request: Prompt2SceneInput,
    *,
    scene_intake: SceneIntakeSpec,
    llm_cfg: OpenAICompatibleLLMCfg,
    output_root: Path,
) -> ImageRelationSpec:
    """Run image asset segmentation alignment for one prompt2scene request."""
    llm = build_chat_model(llm_cfg)
    graph = build_image_relations_graph(llm)
    result = graph.invoke(
        {
            "request": request,
            "scene_intake": scene_intake,
            "output_root": output_root,
            "segment_groups": [],
            "raw_model_output": None,
            "image_relations": None,
            "attempt_count": 0,
            "max_attempts": llm_cfg.max_attempts,
            "last_error": None,
            "errors": [],
        }
    )

    image_relations = result.get("image_relations")
    if (
        image_relations is not None
        and image_relations.status == "ok"
        and image_relations.anchor is not None
    ):
        return image_relations
    if image_relations is not None and image_relations.status == "ok":
        error = format_result_missing_error(
            "Image relations",
            "spatial layout",
            attempt_count=result.get("attempt_count", 0),
            last_error=result.get("last_error"),
            errors=result.get("errors", []),
        )
        log.log_warning(error)
        raise RuntimeError(error)
    if image_relations is not None:
        failed_groups = [
            group.to_manifest()
            for group in image_relations.groups
            if group.status != "ok"
        ]
        if (
            image_relations.table_group is not None
            and image_relations.table_group.status != "ok"
        ):
            failed_groups.append(image_relations.table_group.to_manifest())
        error = (
            "Image relations failed to align all image segments. "
            f"Failed groups: {failed_groups}"
        )
        log.log_warning(error)
        raise RuntimeError(error)

    error = format_result_missing_error(
        "Image relations",
        "ImageRelationSpec",
        attempt_count=result.get("attempt_count", 0),
        last_error=result.get("last_error"),
        errors=result.get("errors", []),
    )
    log.log_warning(error)
    raise RuntimeError(error)
