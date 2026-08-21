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

import shutil
from pathlib import Path

from embodichain.gen_sim.scene_engine.core.scene import Scene
from embodichain.gen_sim.scene_engine.core.scene_edit_plan import SceneEditPlan
from embodichain.gen_sim.scene_engine.core.scene_graph import GeneratedSceneGraph
from embodichain.gen_sim.scene_engine.core.scene_object import SceneObject
from embodichain.gen_sim.scene_engine.pipeline.utils.scene_layout_constructor import (
    SceneLayoutConstructor,
)


def edit_layout(
    *,
    scene: Scene,
    scene_edit_plan: SceneEditPlan,
    updated_scene_graph: GeneratedSceneGraph,
    added_assets: list[SceneObject],
    output_root: str | Path,
) -> Scene:
    """Dispatch one edit-layout optimization from the goal scene graph."""
    formal_scene = scene
    goal_scene_graph = updated_scene_graph
    generated_scene_objects = added_assets
    # Recreate this stage only when new assets need image, segmentation, and geometry outputs.
    stage_output_root = (
        Path(output_root).expanduser().resolve()
        / "scene_editing"
        / "layout_optimization"
    )
    if stage_output_root.exists():
        shutil.rmtree(stage_output_root)
    stage_output_root.mkdir(parents=True, exist_ok=True)

    # Add and move operations are the only layout variables in this edit pass.
    layout_variable_ids = {
        operation.object_id
        for operation in scene_edit_plan.operations
        if operation.op in {"add", "move"}
    }
    if None in layout_variable_ids:
        raise ValueError("Add and move operations must identify an object.")
    layout_variable_ids = {
        object_id for object_id in layout_variable_ids if object_id is not None
    }

    # Optimize the layout constrained by the goal scene graph.
    layout_constructor = SceneLayoutConstructor(
        formal_scene=formal_scene,
        goal_scene_graph=goal_scene_graph,
        layout_variable_ids=layout_variable_ids,
        generated_scene_objects=generated_scene_objects,
        output_root=stage_output_root,
    )
    # Optimize.
    post_edit_scene = layout_constructor.construct()

    return post_edit_scene
