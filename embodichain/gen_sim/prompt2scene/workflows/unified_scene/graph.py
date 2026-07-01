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

from embodichain.gen_sim.prompt2scene.utils import log
from embodichain.gen_sim.prompt2scene.workflows.stage_errors import (
    format_result_missing_error,
)
from embodichain.gen_sim.prompt2scene.workflows.image_relations.schema import (
    ImageRelationSpec,
)
from embodichain.gen_sim.prompt2scene.workflows.scene_intake.schema import (
    SceneIntakeSpec,
)
from embodichain.gen_sim.prompt2scene.workflows.text_relations.schema import (
    TextRelationSpec,
)
from embodichain.gen_sim.prompt2scene.workflows.unified_scene.schema import (
    UnifiedSceneSpec,
)
from embodichain.gen_sim.prompt2scene.workflows.unified_scene.nodes import (
    build_unified_scene_node,
)
from embodichain.gen_sim.prompt2scene.workflows.unified_scene.state import (
    UnifiedSceneState,
)
from embodichain.gen_sim.prompt2scene.workflows.request import Prompt2SceneInput

__all__ = ["build_unified_scene_graph", "run_unified_scene"]


def build_unified_scene_graph() -> Any:
    """Build the fixed unified-scene assembly workflow."""
    graph = StateGraph(UnifiedSceneState)
    graph.add_node("build_unified_scene", build_unified_scene_node)
    graph.set_entry_point("build_unified_scene")
    graph.add_edge("build_unified_scene", END)
    return graph.compile()


def run_unified_scene(
    request: Prompt2SceneInput,
    *,
    scene_intake: SceneIntakeSpec,
    image_relations: ImageRelationSpec | None = None,
    text_relations: TextRelationSpec | None = None,
    output_root: Path,
) -> UnifiedSceneSpec:
    """Run final unified-scene assembly for one prompt2scene request."""
    graph = build_unified_scene_graph()
    result = graph.invoke(
        {
            "request": request,
            "scene_intake": scene_intake,
            "output_root": output_root,
            "image_relations": image_relations,
            "text_relations": text_relations,
            "unified_scene": None,
            "attempt_count": 0,
            "max_attempts": 1,
            "last_error": None,
            "errors": [],
        }
    )

    unified_scene = result.get("unified_scene")
    if unified_scene is not None:
        return unified_scene

    error = format_result_missing_error(
        "Unified scene",
        "UnifiedSceneSpec",
        attempt_count=result.get("attempt_count", 0),
        last_error=result.get("last_error"),
        errors=result.get("errors", []),
    )
    log.log_warning(error)
    raise RuntimeError(error)
