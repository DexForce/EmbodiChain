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

from embodichain.gen_sim.prompt2scene.agent_tools.clients.image_segmentation_client import (
    decode_rle_mask,
    draw_numbered_masks,
)
from embodichain.gen_sim.prompt2scene.workflows.image_relations.schema import (
    ImageAssetSegment,
    ImageRelationGroup,
    ImageRelationSpec,
)
from embodichain.gen_sim.prompt2scene.workflows.request import InputKind
from embodichain.gen_sim.prompt2scene.workflows.image_relations.schema import (
    FILTER_EXTRA_INSTANCES_JSON_SCHEMA,
    SPATIAL_LAYOUT_JSON_SCHEMA,
)
from embodichain.gen_sim.prompt2scene.utils import (
    log_api_request_start,
    log,
)
from embodichain.gen_sim.prompt2scene.workflows.artifact_writer import (
    IMAGE_SEGMENTS_STEP,
    IMAGE_SPATIAL_RELATIONS_STEP,
    WorkflowArtifactWriter,
)
from embodichain.gen_sim.prompt2scene.workflows.image_relations.utils import (
    append_unique,
    apply_spatial_layout_output,
    asset_bbox_label,
    draw_labeled_bboxes,
    expand_asset_ids,
    filter_group_segments_with_vlm,
    filter_segments_with_vlm,
    merge_non_overlapping_segments,
    prompt_text,
    path_token,
    require_image_path,
    segment_prompt,
    segments_from_response,
)
from embodichain.gen_sim.prompt2scene.workflows.image_relations.prompts import (
    build_filter_extra_instances_messages,
    build_spatial_layout_messages,
)
from embodichain.gen_sim.prompt2scene.workflows.image_relations.state import (
    ImageRelationsState,
)
from embodichain.gen_sim.prompt2scene.workflows.llm_output import (
    call_structured_json_model_step,
    is_model_output_error,
)
from embodichain.gen_sim.prompt2scene.workflows.stage_errors import (
    format_attempt_error,
)

__all__ = [
    "call_vlm_filter_extra_instances_node",
    "call_vlm_filter_initial_segments_node",
    "call_vlm_spatial_layout_node",
    "normalize_asset_segments_node",
    "prepare_segmentation_input_node",
    "retry_missing_by_candidates_node",
    "segment_table_node",
    "segment_by_name_node",
]

def prepare_segmentation_input_node(state: ImageRelationsState) -> dict[str, object]:
    """Prepare scene-intake asset groups for class-level segmentation."""
    request = state["request"]
    if request.input_kind != InputKind.IMAGE or request.image_path is None:
        raise ValueError("Image relations requires an image input.")

    segment_groups = []
    for asset in state["scene_intake"].assets:
        group = {
            "name": asset.name,
            "description": asset.description,
            "asset_ids": expand_asset_ids(asset.id, asset.count),
            "class_candidate": list(asset.class_candidate),
            "segments": [],
            "tried_prompts": [],
            "debug_images": [],
            "status": "pending",
            "error": None,
            "expected_count": asset.count,
        }
        segment_groups.append(group)
    return {"segment_groups": segment_groups}


def segment_by_name_node(state: ImageRelationsState) -> dict[str, object]:
    """Run SAM3 once per object name."""
    image_path = require_image_path(state)
    segment_groups = []
    for group in state["segment_groups"]:
        prompt = prompt_text(group["name"])
        response = segment_prompt(image_path=image_path, prompt=prompt)
        group = dict(group)
        group["tried_prompts"] = append_unique(group["tried_prompts"], prompt)
        group["segments"] = segments_from_response(
            group=group,
            response=response,
            source_prompt=prompt,
        )
        segment_groups.append(group)
    return {"segment_groups": segment_groups}


def call_vlm_filter_extra_instances_node(
    state: ImageRelationsState,
    *,
    llm: Any,
) -> dict[str, object]:
    """Compatibility wrapper for the initial VLM segment filter."""
    return call_vlm_filter_initial_segments_node(state, llm=llm)


def call_vlm_filter_initial_segments_node(
    state: ImageRelationsState,
    *,
    llm: Any,
) -> dict[str, object]:
    """Ask VLM to remove wrong masks from initial name-based SAM3 output."""
    return filter_segments_with_vlm(state=state, llm=llm, stage="initial")
def retry_missing_by_candidates_node(
    state: ImageRelationsState,
    *,
    llm: Any,
) -> dict[str, object]:
    """Use remaining class candidates to add missing segment instances."""
    image_path = require_image_path(state)
    artifact_writer = WorkflowArtifactWriter(state["output_root"], IMAGE_SEGMENTS_STEP)
    segment_groups = []
    for group in state["segment_groups"]:
        group = dict(group)
        segments = group["segments"]
        expected_count = group["expected_count"]
        for candidate_name in group["class_candidate"][1:]:
            if len(segments) >= expected_count:
                break
            prompt = prompt_text(candidate_name)
            if prompt in group["tried_prompts"]:
                continue
            response = segment_prompt(image_path=image_path, prompt=prompt)
            group["tried_prompts"] = append_unique(group["tried_prompts"], prompt)
            new_segments = segments_from_response(
                group=group,
                response=response,
                source_prompt=prompt,
            )
            new_segments = filter_group_segments_with_vlm(
                llm=llm,
                image_path=image_path,
                artifact_writer=artifact_writer,
                group=group,
                segments=new_segments,
                stage=f"fallback_{path_token(prompt)}",
            )
            segments = merge_non_overlapping_segments(
                existing=segments,
                incoming=new_segments,
                limit=expected_count,
            )
        if len(segments) < expected_count:
            description_prompt = str(group.get("description") or "").strip()
            if description_prompt and description_prompt not in group["tried_prompts"]:
                response = segment_prompt(
                    image_path=image_path,
                    prompt=description_prompt,
                )
                group["tried_prompts"] = append_unique(
                    group["tried_prompts"],
                    description_prompt,
                )
                new_segments = segments_from_response(
                    group=group,
                    response=response,
                    source_prompt=description_prompt,
                )
                new_segments = filter_group_segments_with_vlm(
                    llm=llm,
                    image_path=image_path,
                    artifact_writer=artifact_writer,
                    group=group,
                    segments=new_segments,
                    stage="fallback_description",
                )
                segments = merge_non_overlapping_segments(
                    existing=segments,
                    incoming=new_segments,
                    limit=expected_count,
                )
        group["segments"] = segments
        segment_groups.append(group)
    return {"segment_groups": segment_groups}


def normalize_asset_segments_node(state: ImageRelationsState) -> dict[str, object]:
    """Assign final segments to scene-intake asset IDs."""
    image_path = require_image_path(state)
    asset_segments: list[ImageAssetSegment] = []
    relation_groups: list[ImageRelationGroup] = []
    status = "ok"

    for group in state["segment_groups"]:
        expected_count = group["expected_count"]
        segments = group["segments"]
        group_status = "ok"
        error = None
        if len(segments) < expected_count:
            group_status = "failed"
            error = "missing_segments"
            status = "failed"
        elif len(segments) > expected_count:
            group_status = "failed"
            error = "extra_segments"
            status = "failed"

        relation_groups.append(
            ImageRelationGroup(
                name=group["name"],
                expected_count=expected_count,
                detected_count=len(segments),
                status=group_status,
                tried_prompts=list(group["tried_prompts"]),
                asset_ids=list(group["asset_ids"]),
                debug_images=list(group["debug_images"]),
                error=error,
            )
        )

        if group_status != "ok":
            continue
        for asset_id, segment in zip(group["asset_ids"], segments):
            asset_segments.append(
                ImageAssetSegment(
                    asset_id=asset_id,
                    name=group["name"],
                    segment_id=segment["segment_id"],
                    bbox_xyxy=list(segment["bbox_xyxy"]),
                    score=float(segment["score"]),
                    source_prompt=segment["source_prompt"],
                    mask_rle=segment.get("mask_rle"),
                )
            )

    bbox_name_image_path = None
    if status == "ok":
        artifact_writer = WorkflowArtifactWriter(
            state["output_root"],
            IMAGE_SEGMENTS_STEP,
        )
        bbox_name_image_path = str(
            draw_labeled_bboxes(
                image_path=image_path,
                boxes=[
                    {
                        "bbox_xyxy": segment.bbox_xyxy,
                        "label": asset_bbox_label(segment.asset_id),
                    }
                    for segment in asset_segments
                ],
                output_path=artifact_writer.step_dir / "asset_segments_bbox_name.png",
            )
        )

    image_relations = ImageRelationSpec(
        status=status,
        image_path=str(image_path),
        asset_segments=asset_segments,
        groups=relation_groups,
        bbox_name_image_path=bbox_name_image_path,
    )
    WorkflowArtifactWriter(
        state["output_root"],
        IMAGE_SEGMENTS_STEP,
    ).write_step_result(image_relations.to_segmentation_manifest())
    return {"image_relations": image_relations}


def segment_table_node(
    state: ImageRelationsState,
    *,
    llm: Any,
) -> dict[str, object]:
    """Segment the table/support target after object segmentation is complete."""
    image_relations = state["image_relations"]
    if image_relations is None or image_relations.status != "ok":
        return {}

    image_path = require_image_path(state)
    table = state["scene_intake"].table
    artifact_writer = WorkflowArtifactWriter(state["output_root"], IMAGE_SEGMENTS_STEP)
    group = {
        "name": table.name,
        "description": table.description,
        "asset_ids": [table.id],
        "class_candidate": list(table.class_candidate),
        "segments": [],
        "tried_prompts": [],
        "debug_images": [],
        "status": "pending",
        "error": None,
        "expected_count": 1,
    }
    segments: list[dict[str, Any]] = []

    for prompt in _table_segmentation_prompts(group):
        if len(segments) >= 1:
            break
        response = segment_prompt(image_path=image_path, prompt=prompt)
        group["tried_prompts"] = append_unique(group["tried_prompts"], prompt)
        new_segments = segments_from_response(
            group=group,
            response=response,
            source_prompt=prompt,
        )
        _write_table_candidate_debug_image(
            image_path=image_path,
            artifact_writer=artifact_writer,
            group=group,
            segments=new_segments,
            stage=f"table_{path_token(prompt)}",
        )
        selected_segment = _select_largest_table_segment(new_segments)
        if selected_segment is not None:
            segments = [selected_segment]

    group_status = "ok" if len(segments) == 1 else "failed"
    error = None if group_status == "ok" else "missing_table_segment"
    table_group = ImageRelationGroup(
        name=group["name"],
        expected_count=1,
        detected_count=len(segments),
        status=group_status,
        tried_prompts=list(group["tried_prompts"]),
        asset_ids=[table.id],
        debug_images=list(group["debug_images"]),
        error=error,
    )
    table_segment = None
    if group_status == "ok":
        segment = segments[0]
        table_segment = ImageAssetSegment(
            asset_id=table.id,
            name=table.name,
            segment_id=segment["segment_id"],
            bbox_xyxy=list(segment["bbox_xyxy"]),
            score=float(segment["score"]),
            source_prompt=segment["source_prompt"],
            mask_rle=segment.get("mask_rle"),
        )

    updated_image_relations = ImageRelationSpec(
        status="ok" if group_status == "ok" else "failed",
        image_path=image_relations.image_path,
        asset_segments=image_relations.asset_segments,
        groups=image_relations.groups,
        table_segment=table_segment,
        table_group=table_group,
        bbox_name_image_path=image_relations.bbox_name_image_path,
        anchor=image_relations.anchor,
        x_order=image_relations.x_order,
        y_order=image_relations.y_order,
        asset_layouts=image_relations.asset_layouts,
    )
    artifact_writer.write_step_result(updated_image_relations.to_segmentation_manifest())
    return {"image_relations": updated_image_relations}


def _table_segmentation_prompts(group: dict[str, Any]) -> list[str]:
    """Return table/support segmentation prompts in object-style fallback order."""
    prompts = [prompt_text(group["name"])]
    for candidate_name in group["class_candidate"][1:]:
        prompts.append(prompt_text(candidate_name))
    description_prompt = str(group.get("description") or "").strip()
    if description_prompt:
        prompts.append(description_prompt)

    unique_prompts: list[str] = []
    for prompt in prompts:
        if prompt and prompt not in unique_prompts:
            unique_prompts.append(prompt)
    return unique_prompts


def _write_table_candidate_debug_image(
    *,
    image_path: Path,
    artifact_writer: WorkflowArtifactWriter,
    group: dict[str, Any],
    segments: list[dict[str, Any]],
    stage: str,
) -> None:
    """Write table/support candidate mask debug image without VLM filtering."""
    if not segments:
        return
    round_name = artifact_writer.next_debug_round_name(label=f"{stage}_{group['name']}")
    round_dir = artifact_writer.debug_round_dir(round_name)
    debug_image_path = draw_numbered_masks(
        image_path=image_path,
        segments=segments,
        output_path=round_dir / "mask.png",
    )
    group["debug_images"] = append_unique(
        group["debug_images"],
        str(debug_image_path),
    )


def _select_largest_table_segment(
    segments: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Select the largest SAM3 table/support candidate without VLM filtering."""
    if not segments:
        return None
    return max(segments, key=_segment_area)


def _segment_area(segment: dict[str, Any]) -> float:
    mask_rle = segment.get("mask_rle")
    if mask_rle is not None:
        try:
            mask = decode_rle_mask(mask_rle).convert("L")
            histogram = mask.histogram()
            return float(sum(count for value, count in enumerate(histogram) if value))
        except Exception:
            pass
    x1, y1, x2, y2 = segment["bbox_xyxy"]
    return max(0.0, float(x2) - float(x1)) * max(0.0, float(y2) - float(y1))


def call_vlm_spatial_layout_node(
    state: ImageRelationsState,
    *,
    llm: Any,
) -> dict[str, object]:
    """Ask VLM for object ordering, anchor grid, and per-object layout states."""
    image_relations = state["image_relations"]
    if image_relations is None or image_relations.status != "ok":
        return {}
    if image_relations.bbox_name_image_path is None:
        raise ValueError("Image spatial layout requires bbox_name_image_path.")

    attempt_count = state["attempt_count"] + 1
    asset_ids = [segment.asset_id for segment in image_relations.asset_segments]
    artifact_writer = WorkflowArtifactWriter(
        state["output_root"],
        IMAGE_SPATIAL_RELATIONS_STEP,
    )
    messages = build_spatial_layout_messages(
        bbox_name_image_path=Path(image_relations.bbox_name_image_path),
        asset_ids=asset_ids,
    )

    try:
        log_api_request_start(
            step=IMAGE_SPATIAL_RELATIONS_STEP,
            request="spatial_layout",
            attempt=attempt_count,
        )
        raw_model_output = call_structured_json_model_step(
            llm=llm,
            schema=SPATIAL_LAYOUT_JSON_SCHEMA,
            messages=messages,
            context="Image spatial layout",
            step_name=IMAGE_SPATIAL_RELATIONS_STEP,
            output_root=None,
            attempt_count=attempt_count,
            raw_output_label="spatial_layout",
            artifact_writer=artifact_writer,
        )
        updated_image_relations = apply_spatial_layout_output(
            image_relations=image_relations,
            raw_model_output=raw_model_output,
        )
        artifact_writer.write_step_result(updated_image_relations.to_spatial_manifest())
    except Exception as exc:
        if is_model_output_error(exc) or isinstance(exc, ValueError):
            error = format_attempt_error("Image relations spatial layout", attempt_count, exc)
            log.log_warning(error)
            return {
                "attempt_count": attempt_count,
                "last_error": error,
                "errors": state["errors"] + [error],
            }
        raise
    return {
        "attempt_count": attempt_count,
        "image_relations": updated_image_relations,
        "last_error": None,
    }
