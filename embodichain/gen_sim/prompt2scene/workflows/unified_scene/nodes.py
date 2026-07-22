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

from embodichain.gen_sim.prompt2scene.workflows.artifact_writer import (
    UNIFIED_SCENE_STEP,
    WorkflowArtifactWriter,
)
from embodichain.gen_sim.prompt2scene.workflows.unified_scene.state import (
    UnifiedSceneState,
)
from embodichain.gen_sim.prompt2scene.workflows.unified_scene.utils import (
    build_unified_scene_from_image_relations,
)

__all__ = ["build_unified_scene_node"]


def build_unified_scene_node(state: UnifiedSceneState) -> dict[str, object]:
    """Assemble the final unified scene manifest."""
    scene_intake = state["scene_intake"]
    image_relations = state.get("image_relations")

    if image_relations is not None and image_relations.status == "ok":
        unified_scene = build_unified_scene_from_image_relations(
            scene_intake=scene_intake,
            image_relations=image_relations,
        )
    else:
        raise ValueError("Unified scene requires image_relations.")

    WorkflowArtifactWriter(
        state["output_root"],
        UNIFIED_SCENE_STEP,
    ).write_step_result(unified_scene.to_manifest())
    return {"unified_scene": unified_scene}
