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
from typing import TYPE_CHECKING

from embodichain.gen_sim.prompt2scene.utils.io import write_json
from embodichain.gen_sim.prompt2scene.workflows.artifact_writer import (
    write_step_result,
)
from embodichain.gen_sim.prompt2scene.workflows.paths import SCENE_EDIT_STEP
from embodichain.gen_sim.prompt2scene.workflows.scene_edit.nodes import (
    analyze_scene_edit_intent_node,
    generate_edit_assets_node,
    optimize_edit_layout_node,
    update_scene_files_node,
)
from embodichain.gen_sim.prompt2scene.workflows.scene_edit.schema import (
    SceneEditRequest,
    SceneEditResult,
)

__all__ = ["run_scene_edit"]

if TYPE_CHECKING:
    from embodichain.gen_sim.prompt2scene.llms import OpenAICompatibleLLMCfg


def run_scene_edit(
    request: SceneEditRequest,
    *,
    llm_cfg: OpenAICompatibleLLMCfg | None = None,
) -> SceneEditResult:
    """Run the scene edit workflow."""
    output_root = request.output_root.expanduser().resolve()
    scene_state_path = output_root / "gym_export" / "scene_state" / "result.json"
    if not scene_state_path.is_file():
        raise FileNotFoundError(
            "Scene edit requires an existing exported scene state: "
            f"{scene_state_path}"
        )
    output_dir = output_root / SCENE_EDIT_STEP
    output_dir.mkdir(parents=True, exist_ok=True)
    if llm_cfg is None:
        raise ValueError("Scene edit requires an LLM config for intent analysis.")
    from embodichain.gen_sim.prompt2scene.llms import build_chat_model

    llm = build_chat_model(llm_cfg)

    intent_analysis = analyze_scene_edit_intent_node(
        request=request,
        output_dir=output_dir,
        llm=llm,
    )
    generated_assets = generate_edit_assets_node(
        intent_analysis=intent_analysis,
        output_dir=output_dir,
        llm=llm,
    )
    layout_result = optimize_edit_layout_node(
        request=request,
        intent_analysis=intent_analysis,
        generated_assets=generated_assets,
        output_dir=output_dir,
    )
    file_updates = update_scene_files_node(
        intent_analysis=intent_analysis,
        generated_assets=generated_assets,
        layout_result=layout_result,
        output_dir=output_dir,
    )

    result = SceneEditResult(
        status="ok" if file_updates.get("status") == "ok" else "partial",
        prompt=request.prompt,
        scene_state_path=scene_state_path,
        reason=(
            "Scene edit intent analysis, asset generation, layout optimization, "
            "and gym_export file updates completed."
        ),
        steps={
            "intent_analysis": intent_analysis,
            "generated_assets": generated_assets,
            "layout_optimization": layout_result,
            "file_updates": file_updates,
        },
    )
    resolved_intent = intent_analysis.get("resolved_intent")
    if isinstance(resolved_intent, dict):
        write_json(output_dir / "resolved_intent.json", resolved_intent)
    write_step_result(output_root, SCENE_EDIT_STEP, result.to_manifest())
    if request.cleanup_scene_edit_dir and output_dir.is_dir():
        shutil.rmtree(output_dir)
    return result
