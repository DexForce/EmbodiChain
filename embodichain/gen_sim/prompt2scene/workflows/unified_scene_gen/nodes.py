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

import json

from embodichain.gen_sim.prompt2scene.utils.log import log_info
from embodichain.gen_sim.prompt2scene.utils.io import write_json
from embodichain.gen_sim.prompt2scene.workflows.unified_scene_gen.state import (
    UnifiedSceneGenState,
)
from embodichain.gen_sim.prompt2scene.workflows.artifact_writer import (
    UNIFIED_SCENE_GEN_STEP,
    UNIFIED_SCENE_STEP,
    WorkflowArtifactWriter,
)
from embodichain.gen_sim.prompt2scene.agent_tools.tools.text_asset_generation import (
    generate_text_object_assets,
    generate_text_table_asset,
)
from embodichain.gen_sim.prompt2scene.agent_tools.tools.text_scene_metric_scale import (
    estimate_text_scene_metric_scale,
)
from embodichain.gen_sim.prompt2scene.agent_tools.tools.text_clutter_layout import (
    generate_text_clutter_layout,
)
from embodichain.gen_sim.prompt2scene.agent_tools.tools.table_fit_scene import (
    fit_image_scene_table,
    fit_text_scene_table,
)
from embodichain.gen_sim.prompt2scene.agent_tools.tools.image_scene_asset_generation import (
    generate_image_scene_assets,
)
from embodichain.gen_sim.prompt2scene.workflows.paths import (
    PipelinePaths,
)
from embodichain.gen_sim.prompt2scene.prompts.builders import (
    build_text_metric_scale_messages,
)
from embodichain.gen_sim.prompt2scene.prompts.schemas import (
    IMAGE_METRIC_SCALE_JSON_SCHEMA,
)
from embodichain.gen_sim.prompt2scene.workflows.unified_scene_gen.utils import (
    update_unified_scene,
)

__all__ = [
    "fit_image_table_to_clutter_node",
    "fit_text_table_to_clutter_node",
    "generate_image_assets_node",
    "generate_text_assets_node",
    "generate_text_clutter_layout_node",
    "load_unified_scene_input_kind_node",
]


def load_unified_scene_input_kind_node(
    state: UnifiedSceneGenState,
) -> dict[str, object]:
    """Load unified-scene output and determine the generation route."""
    paths = PipelinePaths(state["output_root"])
    result_path = paths.resolve_scene_result(state["unified_scene_result_path"])
    if not result_path.is_file():
        raise FileNotFoundError(f"Unified scene result not found: {result_path}")

    with result_path.open("r", encoding="utf-8") as f:
        unified_scene = json.load(f)
    if not isinstance(unified_scene, dict):
        raise ValueError("Unified scene result must be a JSON object.")

    input_record = unified_scene.get("input")
    if not isinstance(input_record, dict):
        raise ValueError("Unified scene result requires input object.")

    input_kind = str(input_record.get("input_kind") or "").strip()
    if input_kind not in {"text", "image"}:
        raise ValueError(
            "Unified scene input.input_kind must be 'text' or 'image', "
            f"got {input_kind!r}."
        )

    return {
        "unified_scene_result_path": result_path,
        "unified_scene": unified_scene,
        "input_kind": input_kind,
    }


def generate_text_assets_node(
    state: UnifiedSceneGenState,
) -> dict[str, object]:
    """Generate images, RGBA cutouts, geometry, and sim-ready GLBs for a
    text-origin unified scene.
    """
    unified_scene = state["unified_scene"]
    if unified_scene is None:
        return {"generation_status": "no_unified_scene"}

    paths = PipelinePaths(state["output_root"])
    output_root = paths.output_root
    image_gen_dir, glb_gen_dir, debug_dir = paths.prepare_generation_dirs()
    log_info(
        "generate_text_assets started "
        f"output_dir={paths.unified_scene_gen_dir}"
    )

    table_spec = unified_scene.get("table") or {}
    table_result = generate_text_table_asset(
        table_spec=table_spec,
        image_gen_dir=image_gen_dir,
        glb_gen_dir=glb_gen_dir,
        debug_dir=debug_dir,
    )

    object_specs = unified_scene.get("objects") or []
    object_results = generate_text_object_assets(
        object_specs=object_specs,
        image_gen_dir=image_gen_dir,
        glb_gen_dir=glb_gen_dir,
        debug_dir=debug_dir,
    )
    metric_prompt_objects = [
        {
            "object_id": str(obj.get("id", "")),
            "object_name": str(obj.get("name", "")),
            "object_description": str(obj.get("description", "")),
        }
        for obj in object_results
    ]
    user_text = str((unified_scene.get("input") or {}).get("text") or "")
    text_metric_scale_result = estimate_text_scene_metric_scale(
        object_results=object_results,
        user_text=user_text,
        messages=build_text_metric_scale_messages(
            user_text=user_text,
            objects_json=metric_prompt_objects,
        ),
        schema=IMAGE_METRIC_SCALE_JSON_SCHEMA,
        output_dir=glb_gen_dir / "metric_scale",
        output_root=output_root,
        llm=state.get("llm"),
        step_name=UNIFIED_SCENE_STEP,
    )

    result_path = paths.resolve_scene_result(state["unified_scene_result_path"])
    update_unified_scene(unified_scene, table_result, object_results, output_root)
    write_json(result_path, unified_scene)
    WorkflowArtifactWriter(output_root, UNIFIED_SCENE_GEN_STEP).write_step_result(
        {
            "table": table_result,
            "objects": object_results,
            "text_metric_scale": text_metric_scale_result,
            "generation_status": "ok",
        }
    )
    log_info(
        "generate_text_assets completed "
        f"table_status={table_result.get('status')} "
        f"object_count={len(object_results)}"
    )

    return {
        "unified_scene": unified_scene,
        "table_result": table_result,
        "text_object_results": object_results,
        "generation_status": "ok",
    }


def generate_image_assets_node(state: UnifiedSceneGenState) -> dict[str, object]:
    """Generate table assets and layout-aware object GLBs for image input.

    Table/support and objects are generated in one multi-object call from the
    original image and existing segmentation masks.
    """
    unified_scene = state["unified_scene"]
    if unified_scene is None:
        return {"generation_status": "no_unified_scene"}

    paths = PipelinePaths(state["output_root"])
    output_root = paths.output_root
    image_gen_dir, glb_gen_dir, debug_dir = paths.prepare_generation_dirs()
    log_info(
        "generate_image_assets started "
        f"output_dir={paths.unified_scene_gen_dir}"
    )

    segments_path = paths.image_segments_result
    if not segments_path.is_file():
        raise FileNotFoundError(
            f"Image segments result not found: {segments_path}"
        )
    with segments_path.open("r", encoding="utf-8") as _f:
        segments_data = json.load(_f)
    if not isinstance(segments_data, dict):
        raise ValueError("Image segments result must be a JSON object.")

    table_spec = unified_scene.get("table") or {}
    # Image input uses the segmented table/support mask in the multi-object
    # SAM3D call below. Text table generation belongs to the text branch.
    object_specs = unified_scene.get("objects") or []
    object_layout_result = generate_image_scene_assets(
        object_specs=object_specs,
        table_spec=table_spec,
        spatial_relations=(unified_scene.get("spatial") or {}).get("relations", []),
        segments_data=segments_data,
        image_gen_dir=image_gen_dir,
        glb_gen_dir=glb_gen_dir,
        debug_dir=debug_dir,
        output_root=output_root,
        llm=state.get("llm"),
    )
    table_result = object_layout_result.get("table") or {
        "id": str(table_spec.get("id", "table")),
        "name": str(table_spec.get("name", "table")),
        "status": "missing_table_generation",
    }
    object_results = object_layout_result.get("objects") or []
    generation_status = str(object_layout_result.get("status", "failed"))
    if table_result.get("status") != "ok":
        generation_status = str(table_result.get("status") or generation_status)
    result_path = paths.resolve_scene_result(state["unified_scene_result_path"])
    update_unified_scene(unified_scene, table_result, object_results, output_root)
    write_json(result_path, unified_scene)
    WorkflowArtifactWriter(output_root, UNIFIED_SCENE_GEN_STEP).write_step_result(
        {
            "table": table_result,
            "objects_layout": object_layout_result,
            "objects": object_results,
            "table_fit_to_clutter": None,
            "generation_status": generation_status,
        }
    )
    log_info(f"generate_image_assets completed status={generation_status}")

    return {
        "unified_scene": unified_scene,
        "table_result": table_result,
        "text_object_results": object_results,
        "image_objects_layout_result": object_layout_result,
        "generation_status": generation_status,
    }


def fit_image_table_to_clutter_node(state: UnifiedSceneGenState) -> dict[str, object]:
    """Resize the final table to fit the aligned image-object clutter."""
    if state.get("input_kind") != "image":
        return {}

    paths = PipelinePaths(state["output_root"])
    output_root = paths.output_root
    output_dir = paths.table_fit_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    log_info(f"fit_image_table_to_clutter started output_dir={output_dir}")
    layout_result = dict(state.get("image_objects_layout_result") or {})
    table_fit_result = fit_image_scene_table(
        layout_result=layout_result,
        fallback_table_result=state.get("table_result"),
        output_root=output_root,
        output_dir=output_dir,
    )
    layout_result["table_fit_to_clutter"] = table_fit_result
    WorkflowArtifactWriter(output_root, UNIFIED_SCENE_GEN_STEP).write_step_result(
        {
            "table": state.get("table_result"),
            "objects_layout": layout_result,
            "objects": state.get("text_object_results") or [],
            "table_fit_to_clutter": table_fit_result,
            "generation_status": state.get("generation_status"),
        }
    )
    log_info(
        f"fit_image_table_to_clutter completed status={table_fit_result.get('status')}"
    )
    return {
        "image_objects_layout_result": layout_result,
        "table_fit_result": table_fit_result,
    }


def generate_text_clutter_layout_node(
    state: UnifiedSceneGenState,
) -> dict[str, object]:
    """Scale text objects to real-world size, gravity-settle, centre at origin.

    Produces per-object settled GLBs and 2D AABB footprints for downstream
    spatial layout optimisation and table fitting.
    """
    if state.get("input_kind") != "text":
        return {}

    paths = PipelinePaths(state["output_root"])
    output_root = paths.output_root
    output_dir = paths.text_clutter_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    log_info(f"generate_text_clutter_layout started output_dir={output_dir}")

    text_object_results = state.get("text_object_results") or []
    if not text_object_results:
        return {
            "text_clutter_settle_result": {
                "status": "skipped",
                "reason": "no_text_objects",
            }
        }

    unified_scene = state.get("unified_scene") or {}
    spatial_data = unified_scene.get("spatial") or {}
    spatial_relations = spatial_data.get("relations", [])
    table_constraints = spatial_data.get("table_constraints", [])

    settle_result = generate_text_clutter_layout(
        object_results=text_object_results,
        spatial_relations=spatial_relations,
        table_constraints=table_constraints,
        output_dir=output_dir,
        output_root=output_root,
    )
    WorkflowArtifactWriter(output_root, UNIFIED_SCENE_GEN_STEP).write_step_result(
        {
            "table": state.get("table_result"),
            "objects": text_object_results,
            "text_clutter_settle": settle_result,
            "generation_status": state.get("generation_status"),
        }
    )
    log_info(
        f"generate_text_clutter_layout completed status={settle_result.get('status')}"
    )
    return {
        "text_clutter_settle_result": settle_result,
    }


def fit_text_table_to_clutter_node(
    state: UnifiedSceneGenState,
) -> dict[str, object]:
    """Resize the text-scene table to fit the laid-out clutter footprint."""
    if state.get("input_kind") != "text":
        return {}

    paths = PipelinePaths(state["output_root"])
    output_root = paths.output_root
    table_result = state.get("table_result")
    settle_result = state.get("text_clutter_settle_result")

    if table_result is None or settle_result is None:
        return {
            "table_fit_result": {
                "status": "skipped",
                "reason": "missing_table_or_settle_result",
            }
        }

    output_dir = paths.table_fit_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    log_info(f"fit_text_table_to_clutter started output_dir={output_dir}")
    table_fit_result = fit_text_scene_table(
        table_result=table_result,
        clutter_layout_result=settle_result,
        output_root=output_root,
        output_dir=output_dir,
    )
    WorkflowArtifactWriter(output_root, UNIFIED_SCENE_GEN_STEP).write_step_result(
        {
            "table": table_result,
            "objects": state.get("text_object_results") or [],
            "text_clutter_settle": settle_result,
            "table_fit_to_clutter": table_fit_result,
            "generation_status": state.get("generation_status"),
        }
    )
    log_info(
        f"fit_text_table_to_clutter completed status={table_fit_result.get('status')}"
    )
    return {
        "table_fit_result": table_fit_result,
    }
