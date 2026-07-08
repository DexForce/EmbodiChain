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

from embodichain.gen_sim.prompt2scene.prompts.builders import (
    build_scene_edit_intent_messages,
    build_text_metric_scale_messages,
)
from embodichain.gen_sim.prompt2scene.prompts.schemas import (
    IMAGE_METRIC_SCALE_JSON_SCHEMA,
    SCENE_EDIT_INTENT_JSON_SCHEMA,
)
from embodichain.gen_sim.prompt2scene.agent_tools.tools.text_scene_metric_scale import (
    estimate_text_scene_metric_scale,
)
from embodichain.gen_sim.prompt2scene.utils import (
    log,
    log_api_request_start,
)
from embodichain.gen_sim.prompt2scene.workflows.stage_errors import (
    format_attempt_error,
    format_result_missing_error,
)
from embodichain.gen_sim.prompt2scene.workflows.scene_edit.schema import (
    SceneEditRequest,
)
from embodichain.gen_sim.prompt2scene.workflows.scene_edit.utils import (
    build_scene_edit_layout,
    export_scene_edit_gym_state,
    extract_current_grids,
    extract_current_relations,
    extract_scene_objects,
    generate_scene_edit_object_assets,
    load_json_object,
    match_prompt_scene_objects,
    resolve_scene_edit_intent,
    scene_state_path,
)

__all__ = [
    "analyze_scene_edit_intent_node",
    "generate_edit_assets_node",
    "optimize_edit_layout_node",
    "update_scene_files_node",
]


def analyze_scene_edit_intent_node(
    *,
    request: SceneEditRequest,
    output_dir: Path,
    llm: Any,
) -> dict[str, Any]:
    """Analyze existing scene state plus user prompt into structured edit intent."""
    state_path = scene_state_path(request.output_root)
    if not state_path.is_file():
        raise FileNotFoundError(
            "Scene edit requires an existing exported scene state: "
            f"{state_path}"
        )
    scene_state = load_json_object(state_path)
    scene_objects = extract_scene_objects(scene_state)
    current_relations = extract_current_relations(
        output_root=request.output_root,
        scene_state=scene_state,
    )
    current_grids = extract_current_grids(
        output_root=request.output_root,
        scene_state=scene_state,
    )
    messages = build_scene_edit_intent_messages(
        prompt=request.prompt,
        scene_objects=scene_objects,
        current_relations=current_relations,
    )
    from embodichain.gen_sim.prompt2scene.llms.llm_output import (
        StructuredModelCallError,
        call_structured_json_model_step,
    )

    attempt_count = 0
    max_attempts = 3
    errors: list[str] = []
    raw_model_output: dict[str, Any] | None = None
    retry_messages = list(messages)
    persist_raw_model_output = False
    while attempt_count < max_attempts:
        attempt_count += 1
        try:
            log_api_request_start(
                step="scene_edit",
                request="intent_analysis",
                attempt=attempt_count,
            )
            raw_model_output = call_structured_json_model_step(
                llm=llm,
                schema=SCENE_EDIT_INTENT_JSON_SCHEMA,
                messages=retry_messages,
                context="Scene edit intent",
                attempt_count=attempt_count,
                raw_output_writer=None,
            )
            break
        except StructuredModelCallError as exc:
            error = format_attempt_error("Scene edit intent", attempt_count, exc)
            errors.append(error)
            log.log_warning(error)
            persist_raw_model_output = True
            retry_messages = list(messages) + [
                {
                    "role": "user",
                    "content": (
                        "The previous JSON output failed schema validation. "
                        f"Fix this exact error and output the full JSON again: {exc}"
                    ),
                }
            ]

    if raw_model_output is None:
        raise RuntimeError(
            format_result_missing_error(
                "Scene edit intent",
                "SceneEditIntentOutput",
                attempt_count=attempt_count,
                last_error=errors[-1] if errors else None,
                errors=errors,
            )
        )

    resolved_intent = resolve_scene_edit_intent(
        intent=raw_model_output,
        scene_objects=scene_objects,
        current_relations=current_relations,
        current_grids=current_grids,
    )
    source_snapshots = scene_state.get("source_snapshots") or {}
    object_matches = match_prompt_scene_objects(
        prompt=request.prompt,
        scene_state=scene_state,
    )
    analysis = {
        "status": "ok",
        "node": "analyze_scene_edit_intent",
        "prompt": request.prompt,
        "scene_state_path": str(state_path),
        "source_snapshots": source_snapshots,
        "scene_summary": {
            "object_count": len(scene_objects),
            "objects": scene_objects,
        },
        "current_relations": current_relations,
        "current_grid_assignments": current_grids,
        "prompt_object_matches": object_matches,
        "llm_intent": raw_model_output,
        "resolved_intent": resolved_intent,
    }
    if persist_raw_model_output:
        analysis["debug"] = {"retry_errors": errors}
    return analysis


def generate_edit_assets_node(
    *,
    request: SceneEditRequest,
    intent_analysis: dict[str, Any],
    output_dir: Path,
    llm: Any | None = None,
) -> dict[str, Any]:
    """Generate simready assets for add/replace objects in a scene edit."""
    intent = intent_analysis.get("resolved_intent")
    if not isinstance(intent, dict):
        intent = {}
    generated_objects = intent.get("generated_objects")
    if not isinstance(generated_objects, list):
        generated_objects = []
    if not generated_objects:
        return {
            "status": "ok",
            "node": "generate_edit_assets",
            "input_intent_status": intent_analysis.get("status"),
            "objects_to_generate": [],
            "generated_assets": [],
            "reason": "No new objects were requested by the edit intent.",
        }
    generation_result = generate_scene_edit_object_assets(
        generated_objects=generated_objects,
        output_root=output_dir.parent,
        output_dir=output_dir,
        gravity_settle_mode=request.gravity_settle_mode,
    )
    generated_assets = generation_result.get("generated_assets", [])
    if isinstance(generated_assets, list) and generated_assets:
        metric_prompt_objects = [
            {
                "object_id": str(obj.get("id", "")),
                "object_name": str(obj.get("name", "")),
                "object_description": str(obj.get("description", "")),
            }
            for obj in generated_assets
        ]
        prompt_text = str(intent_analysis.get("prompt") or "")
        metric_scale_result = estimate_text_scene_metric_scale(
            object_results=generated_assets,
            user_text=prompt_text,
            messages=build_text_metric_scale_messages(
                user_text=prompt_text,
                objects_json=metric_prompt_objects,
            ),
            schema=IMAGE_METRIC_SCALE_JSON_SCHEMA,
            output_dir=output_dir / "glb_gen" / "metric_scale",
            output_root=output_dir.parent,
            llm=llm,
            step_name="scene_edit",
        )
    else:
        metric_scale_result = {
            "status": "skipped",
            "reason": "no_generated_assets",
            "objects": [],
        }
    result = {
        "status": generation_result.get("status", "partial"),
        "node": "generate_edit_assets",
        "input_intent_status": intent_analysis.get("status"),
        "objects_to_generate": generated_objects,
        "generated_assets": generated_assets,
        "object_count": generation_result.get("object_count", 0),
        "metric_scale": metric_scale_result,
        "reason": (
            "Generated simready assets for scene-edit add/replace objects."
        ),
    }
    return result


def optimize_edit_layout_node(
    *,
    request: SceneEditRequest,
    intent_analysis: dict[str, Any],
    generated_assets: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Load the previous 2D layout and optimize an edited scene layout."""
    scene_state_value = intent_analysis.get("scene_state_path", "")
    scene_state = load_json_object(Path(str(scene_state_value)))
    resolved_intent = intent_analysis.get("resolved_intent")
    if not isinstance(resolved_intent, dict):
        resolved_intent = {}
    generated_asset_items = generated_assets.get("generated_assets")
    if not isinstance(generated_asset_items, list):
        generated_asset_items = []
    layout = build_scene_edit_layout(
        scene_state=scene_state,
        resolved_intent=resolved_intent,
        generated_assets=generated_asset_items,
        output_root=output_dir.parent,
        optimize_new_objects_only=request.optimize_new_objects_only,
    )
    return {
        "status": layout.get("status", "ok"),
        "node": "optimize_edit_layout",
        "existing_scene_state_path": scene_state_value,
        "generated_asset_count": len(generated_asset_items),
        "deleted_object_ids": layout.get("deleted_object_ids", []),
        "moved_object_ids": layout.get("moved_object_ids", []),
        "support_region": layout.get("support_region", {}),
        "layout_updates": layout.get("layout_updates", []),
        "optimization": layout.get("optimization", {}),
        "reason": (
            "Loaded the previous scene_state 2D footprints, inherited replacement "
            "object centers, computed generated-object XY sizes from simready GLBs, "
            "and applied relation/grid-based local layout optimization."
        ),
    }


def update_scene_files_node(
    *,
    intent_analysis: dict[str, Any],
    generated_assets: dict[str, Any],
    layout_result: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Update gym_export outputs so future scene edits read the edited scene."""
    scene_state_value = intent_analysis.get("scene_state_path", "")
    scene_state = load_json_object(Path(str(scene_state_value)))
    generated_asset_items = generated_assets.get("generated_assets")
    if not isinstance(generated_asset_items, list):
        generated_asset_items = []
    layout_updates = layout_result.get("layout_updates")
    if not isinstance(layout_updates, list):
        layout_updates = []
    export_result = export_scene_edit_gym_state(
        output_root=output_dir.parent,
        scene_state=scene_state,
        generated_assets=generated_asset_items,
        layout_updates=layout_updates,
        output_dir=output_dir,
    )
    return {
        "status": export_result.get("status", "ok"),
        "node": "update_scene_files",
        "updated_files": export_result.get("updated_files", []),
        "reason": (
            "Updated gym_export outputs from the edited scene layout, including "
            "gym_config, scene_state/result.json, topdown_2d.png, and any new "
            "simready mesh assets."
        ),
        "inputs": {
            "intent_status": intent_analysis.get("status"),
            "generated_assets_status": generated_assets.get("status"),
            "layout_status": layout_result.get("status"),
        },
    }
