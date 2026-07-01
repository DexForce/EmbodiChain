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
    ImageSegmentationClient,
    ImageSegmentationError,
    ImageSegmentationServerRequest,
    ImageSegmentationServerResponse,
    bbox_iou,
    draw_labeled_bboxes,
    draw_numbered_masks,
    is_usable_segmentation_candidate,
    sort_segments_by_bbox,
)
from embodichain.gen_sim.prompt2scene.workflows.image_relations.schema import (
    FILTER_EXTRA_INSTANCES_JSON_SCHEMA,
    ImageAnchor,
    ImageAssetLayout,
    ImageAssetSegment,
    ImageRelationGroup,
    ImageRelationSpec,
    SPATIAL_LAYOUT_JSON_SCHEMA,
)
from embodichain.gen_sim.prompt2scene.workflows.spatial import (
    GRID_VALUES,
    validate_exact_asset_id_coverage,
)
from embodichain.gen_sim.prompt2scene.utils import log_api_request_start, log
from embodichain.gen_sim.prompt2scene.workflows.artifact_writer import (
    IMAGE_SEGMENTS_STEP,
    IMAGE_SPATIAL_RELATIONS_STEP,
    RAW_MODEL_OUTPUT_FILENAME,
    WorkflowArtifactWriter,
)
from embodichain.gen_sim.prompt2scene.workflows.image_relations.prompts import (
    build_filter_extra_instances_messages,
    build_spatial_layout_messages,
)
from embodichain.gen_sim.prompt2scene.workflows.llm_output import (
    call_structured_json_model_step,
    is_model_output_error,
)
from embodichain.gen_sim.prompt2scene.workflows.stage_errors import (
    format_attempt_error,
)

__all__ = [
    "MAX_SEGMENT_RETRIES",
    "OVERLAP_IOU_THRESHOLD",
    "append_unique",
    "apply_spatial_layout_output",
    "asset_bbox_label",
    "expand_asset_ids",
    "filter_group_segments_with_vlm",
    "filter_segments_with_vlm",
    "merge_non_overlapping_segments",
    "draw_labeled_bboxes",
    "parse_anchor",
    "parse_asset_states",
    "parse_order_groups",
    "path_token",
    "prompt_text",
    "remove_extra_numbered_segments",
    "require_image_path",
    "segment_prompt",
    "segments_from_response",
    "sort_segments_by_bbox",
]

MAX_SEGMENT_RETRIES = 1
OVERLAP_IOU_THRESHOLD = 0.5


def require_image_path(state: dict[str, Any]) -> Path:
    """Return the request image path or raise if the input is invalid."""
    image_path = state["request"].image_path
    if image_path is None:
        raise ValueError("Image relations requires request.image_path.")
    return image_path


def prompt_text(name: str) -> str:
    """Convert an asset name to a natural-language prompt."""
    return name.replace("_", " ")


def asset_bbox_label(asset_id: str) -> str:
    """Convert an internal asset id into a display label."""
    prefix = "interact_"
    return asset_id[len(prefix) :] if asset_id.startswith(prefix) else asset_id


def expand_asset_ids(asset_id: str, count: int) -> list[str]:
    """Expand a grouped asset id into instance ids."""
    return [f"{asset_id}_{index}" for index in range(count)]


def path_token(value: str) -> str:
    """Convert a label into a filesystem-safe token."""
    token = "".join(character if character.isalnum() else "_" for character in value)
    return token.strip("_")[:80] or "prompt"


def append_unique(values: list[str], value: str) -> list[str]:
    """Append a string only if it does not already exist in the list."""
    return values if value in values else values + [value]


def segment_prompt(
    *,
    image_path: Path,
    prompt: str,
) -> ImageSegmentationServerResponse:
    """Call the segmentation server with a single prompt."""
    client = ImageSegmentationClient()
    log_api_request_start(
        step=IMAGE_SEGMENTS_STEP,
        request="sam3_segment",
        prompt=prompt,
    )
    result = client.segment(
        ImageSegmentationServerRequest(prompt=prompt, image_path=image_path),
        max_retries=MAX_SEGMENT_RETRIES,
    )
    if isinstance(result, ImageSegmentationError):
        log.log_warning(result.error_message)
        raise RuntimeError(result.error_message)
    return result


def segments_from_response(
    *,
    group: dict[str, Any],
    response: ImageSegmentationServerResponse,
    source_prompt: str,
) -> list[dict[str, Any]]:
    """Convert segmentation server output into internal segment dicts."""
    segments = []
    for candidate in response.result.candidates:
        if not is_usable_segmentation_candidate(candidate):
            continue
        segments.append(
            {
                "segment_id": f"{group['name']}_{len(segments)}",
                "bbox_xyxy": list(candidate.bbox_xyxy),
                "score": float(candidate.score),
                "mask_rle": candidate.mask_rle,
                "source_prompt": source_prompt,
            }
        )
    return sort_segments_by_bbox(segments)


def apply_spatial_layout_output(
    *,
    image_relations: ImageRelationSpec,
    raw_model_output: dict[str, Any],
) -> ImageRelationSpec:
    """Apply VLM spatial-layout output to an image-relations spec."""
    asset_ids = [segment.asset_id for segment in image_relations.asset_segments]
    asset_id_set = set(asset_ids)

    anchor = parse_anchor(raw_model_output.get("anchor"), asset_id_set=asset_id_set)
    x_order = parse_order_groups(
        raw_model_output.get("x_order"),
        asset_ids=asset_ids,
        field_name="x_order",
    )
    y_order = parse_order_groups(
        raw_model_output.get("y_order"),
        asset_ids=asset_ids,
        field_name="y_order",
    )
    state_by_asset_id = parse_asset_states(
        raw_model_output.get("asset_states"),
        asset_ids=asset_ids,
    )
    asset_layouts = [
        ImageAssetLayout(
            asset_id=asset_id,
            is_arbitrary_layout=state_by_asset_id[asset_id]["is_arbitrary_layout"],
            reason=state_by_asset_id[asset_id]["reason"],
        )
        for asset_id in asset_ids
    ]
    return ImageRelationSpec(
        status=image_relations.status,
        image_path=image_relations.image_path,
        asset_segments=image_relations.asset_segments,
        groups=image_relations.groups,
        table_segment=image_relations.table_segment,
        table_group=image_relations.table_group,
        bbox_name_image_path=image_relations.bbox_name_image_path,
        anchor=anchor,
        x_order=x_order,
        y_order=y_order,
        asset_layouts=asset_layouts,
    )


def parse_anchor(raw_anchor: Any, *, asset_id_set: set[str]) -> ImageAnchor:
    """Parse and validate the anchor entry."""
    if not isinstance(raw_anchor, dict):
        raise ValueError("anchor must be an object.")
    asset_id = str(raw_anchor.get("asset_id") or "").strip()
    grid = str(raw_anchor.get("grid") or "").strip()
    reason = str(raw_anchor.get("reason") or "").strip()
    if asset_id not in asset_id_set:
        raise ValueError(f"anchor.asset_id is not a known asset: {asset_id!r}.")
    if grid not in GRID_VALUES:
        raise ValueError(f"anchor.grid is not valid: {grid!r}.")
    return ImageAnchor(asset_id=asset_id, grid=grid, reason=reason)


def parse_order_groups(
    raw_order: Any,
    *,
    asset_ids: list[str],
    field_name: str,
) -> list[list[str]]:
    """Parse ordered asset-id groups from VLM output."""
    if not isinstance(raw_order, list) or not raw_order:
        raise ValueError(f"{field_name} must be a non-empty list.")

    groups: list[list[str]] = []
    flattened: list[str] = []
    for group_index, raw_group in enumerate(raw_order):
        if not isinstance(raw_group, list) or not raw_group:
            raise ValueError(f"{field_name}[{group_index}] must be a non-empty list.")
        group: list[str] = []
        for raw_asset_id in raw_group:
            asset_id = str(raw_asset_id).strip()
            group.append(asset_id)
            flattened.append(asset_id)
        groups.append(group)

    validate_exact_asset_id_coverage(
        values=flattened,
        expected_asset_ids=asset_ids,
        context=field_name,
    )
    return groups


def parse_asset_states(
    raw_asset_states: Any,
    *,
    asset_ids: list[str],
) -> dict[str, dict[str, Any]]:
    """Parse per-asset layout state annotations."""
    if not isinstance(raw_asset_states, list):
        raise ValueError("asset_states must be a list.")

    state_by_asset_id: dict[str, dict[str, Any]] = {}
    for state_index, raw_state in enumerate(raw_asset_states):
        if not isinstance(raw_state, dict):
            raise ValueError(f"asset_states[{state_index}] must be an object.")
        asset_id = str(raw_state.get("asset_id") or "").strip()
        is_arbitrary_layout = raw_state.get("is_arbitrary_layout")
        reason = str(raw_state.get("reason") or "").strip()
        if not isinstance(is_arbitrary_layout, bool):
            raise ValueError(
                f"asset_states[{state_index}].is_arbitrary_layout must be boolean."
            )
        if not reason:
            raise ValueError(f"asset_states[{state_index}].reason must be non-empty.")
        if asset_id in state_by_asset_id:
            raise ValueError(f"asset_states has duplicate asset_id: {asset_id!r}.")
        state_by_asset_id[asset_id] = {
            "is_arbitrary_layout": is_arbitrary_layout,
            "reason": reason,
        }

    validate_exact_asset_id_coverage(
        values=list(state_by_asset_id),
        expected_asset_ids=asset_ids,
        context="asset_states",
    )
    return state_by_asset_id


def filter_group_segments_with_vlm(
    *,
    llm: Any,
    image_path: Path,
    artifact_writer: WorkflowArtifactWriter,
    group: dict[str, Any],
    segments: list[dict[str, Any]],
    stage: str,
) -> list[dict[str, Any]]:
    """Ask VLM to remove wrong or duplicate instances from one SAM3 result."""
    segments = sort_segments_by_bbox(segments)
    if not segments:
        return segments

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
    log_api_request_start(
        step=IMAGE_SEGMENTS_STEP,
        request=f"vlm_filter_{stage}",
        debug_image=str(debug_image_path),
    )
    messages = build_filter_extra_instances_messages(
        debug_image_path=debug_image_path,
        name=group["name"],
        description=group["description"],
        expected_count=group["expected_count"],
        class_candidate=group["class_candidate"],
    )
    raw_model_output = call_structured_json_model_step(
        llm=llm,
        schema=FILTER_EXTRA_INSTANCES_JSON_SCHEMA,
        messages=messages,
        context=f"Image relation {stage} segmentation filtering",
        step_name=IMAGE_SEGMENTS_STEP,
        output_root=None,
        attempt_count=0,
        raw_output_writer=lambda payload: artifact_writer.write_debug_round_json(
            round_name=round_name,
            filename=RAW_MODEL_OUTPUT_FILENAME,
            payload=payload,
        ),
    )
    return remove_extra_numbered_segments(
        segments=segments,
        raw_model_output=raw_model_output,
    )


def filter_segments_with_vlm(
    *,
    state: dict[str, Any],
    llm: Any,
    stage: str,
) -> dict[str, object]:
    """Filter all segment groups with VLM and return an updated state patch."""
    segment_groups = []
    attempt_count = state["attempt_count"] + 1
    image_path = require_image_path(state)
    artifact_writer = WorkflowArtifactWriter(state["output_root"], IMAGE_SEGMENTS_STEP)

    try:
        for group in state["segment_groups"]:
            group = dict(group)
            group["segments"] = filter_group_segments_with_vlm(
                llm=llm,
                image_path=image_path,
                artifact_writer=artifact_writer,
                group=group,
                segments=group["segments"],
                stage=stage,
            )
            segment_groups.append(group)
    except Exception as exc:
        if is_model_output_error(exc) or isinstance(exc, ValueError):
            error = format_attempt_error("Image relations VLM filter", attempt_count, exc)
            log.log_warning(error)
            return {
                "attempt_count": attempt_count,
                "last_error": error,
                "errors": state["errors"] + [error],
            }
        raise

    return {
        "attempt_count": attempt_count,
        "segment_groups": segment_groups,
        "last_error": None,
    }


def remove_extra_numbered_segments(
    *,
    segments: list[dict[str, Any]],
    raw_model_output: dict[str, Any],
) -> list[dict[str, Any]]:
    """Remove numbered masks flagged as extra by the VLM."""
    extra_numbers = raw_model_output.get("extra_instance_numbers")
    if not isinstance(extra_numbers, list):
        raise ValueError("extra_instance_numbers must be a list.")
    extra_indices = {int(number) - 1 for number in extra_numbers}
    if any(index < 0 or index >= len(segments) for index in extra_indices):
        raise ValueError("VLM returned an out-of-range extra mask number.")
    kept = [
        segment for index, segment in enumerate(segments) if index not in extra_indices
    ]
    return kept


def merge_non_overlapping_segments(
    *,
    existing: list[dict[str, Any]],
    incoming: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    """Merge non-overlapping segments until a limit is reached."""
    merged = list(existing)
    for segment in sorted(
        incoming, key=lambda item: float(item["score"]), reverse=True
    ):
        if len(merged) >= limit:
            break
        if all(
            bbox_iou(segment["bbox_xyxy"], other["bbox_xyxy"]) < OVERLAP_IOU_THRESHOLD
            for other in merged
        ):
            merged.append(segment)
    return sort_segments_by_bbox(merged)
