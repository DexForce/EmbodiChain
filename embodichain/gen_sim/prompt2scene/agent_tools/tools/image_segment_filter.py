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
from typing import Any, Callable

import numpy as np

from embodichain.gen_sim.prompt2scene.agent_tools.clients.image_segmentation_client import (
    decode_rle_mask,
    draw_numbered_masks,
    sort_segments_by_bbox,
)
from embodichain.gen_sim.prompt2scene.llms.llm_output import (
    call_structured_json_model_step,
    is_model_output_error,
)
from embodichain.gen_sim.prompt2scene.prompts.builders import (
    build_filter_extra_instances_messages,
)
from embodichain.gen_sim.prompt2scene.prompts.schemas import (
    FILTER_EXTRA_INSTANCES_JSON_SCHEMA,
)
from embodichain.gen_sim.prompt2scene.utils import log_api_request_start, log
from embodichain.gen_sim.prompt2scene.workflows.stage_errors import (
    format_attempt_error,
)

__all__ = [
    "filter_group_segments_with_vlm",
    "filter_overlapping_confirmed_segments",
    "filter_segments_with_vlm",
    "remove_extra_numbered_segments",
]

DebugWriter = Callable[[str, str, dict[str, Any]], Path]
CONFIRMED_MASK_COVERAGE_THRESHOLD = 0.85
CONFIRMED_MASK_IOU_THRESHOLD = 0.70


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
    return [
        segment for index, segment in enumerate(segments) if index not in extra_indices
    ]


def filter_group_segments_with_vlm(
    *,
    llm: Any,
    image_path: Path,
    step_name: str,
    group: dict[str, Any],
    segments: list[dict[str, Any]],
    stage: str,
    debug_round_name: str,
    debug_round_dir: Path,
    write_debug_json: DebugWriter,
    confirmed_segments: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Ask VLM to remove wrong or duplicate instances from one SAM3 result.

    All path concerns are injected via *step_name*, *debug_round_name*,
    *debug_round_dir*, and *write_debug_json* so the tool does not depend
    on workflow internals.
    """
    segments = filter_overlapping_confirmed_segments(
        segments=segments,
        confirmed_segments=confirmed_segments or [],
    )
    segments = sort_segments_by_bbox(segments)
    if not segments:
        return segments

    debug_image_path = draw_numbered_masks(
        image_path=image_path,
        segments=segments,
        output_path=debug_round_dir / "mask.png",
    )
    debug_images = list(group.get("debug_images") or [])
    if str(debug_image_path) not in debug_images:
        debug_images.append(str(debug_image_path))
    group["debug_images"] = debug_images

    log_api_request_start(
        step=step_name,
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
        attempt_count=0,
        raw_output_writer=lambda payload: write_debug_json(
            round_name=debug_round_name,
            filename="raw_model_output.json",
            payload=payload,
        ),
    )
    return remove_extra_numbered_segments(
        segments=segments,
        raw_model_output=raw_model_output,
    )


def filter_segments_with_vlm(
    *,
    llm: Any,
    image_path: Path,
    step_name: str,
    segment_groups: list[dict[str, Any]],
    attempt_count: int,
    errors: list[str],
    stage: str,
    next_debug_round_name: Callable[[str], str],
    debug_round_dir: Callable[[str], Path],
    write_debug_json: DebugWriter,
) -> dict[str, object]:
    """Filter all segment groups with VLM and return an updated state patch.

    All path concerns are injected via callbacks so the tool does not
    depend on workflow internals.
    """
    result_groups: list[dict[str, Any]] = []
    current_attempt = attempt_count + 1
    confirmed_segments: list[dict[str, Any]] = []

    try:
        for group in segment_groups:
            group = dict(group)
            name = str(group.get("name", "unknown"))
            round_name = next_debug_round_name(f"{stage}_{name}")
            round_dir = debug_round_dir(round_name)
            group["segments"] = filter_group_segments_with_vlm(
                llm=llm,
                image_path=image_path,
                step_name=step_name,
                group=group,
                segments=group["segments"],
                stage=stage,
                debug_round_name=round_name,
                debug_round_dir=round_dir,
                write_debug_json=write_debug_json,
                confirmed_segments=confirmed_segments,
            )
            confirmed_segments.extend(group["segments"])
            result_groups.append(group)
    except Exception as exc:
        if is_model_output_error(exc) or isinstance(exc, ValueError):
            error = format_attempt_error(
                "Image relations VLM filter", current_attempt, exc
            )
            log.log_warning(error)
            return {
                "attempt_count": current_attempt,
                "last_error": error,
                "errors": errors + [error],
            }
        raise

    return {
        "attempt_count": current_attempt,
        "segment_groups": result_groups,
        "last_error": None,
    }


def filter_overlapping_confirmed_segments(
    *,
    segments: list[dict[str, Any]],
    confirmed_segments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Drop candidates that almost exactly duplicate an already accepted mask."""
    if not segments or not confirmed_segments:
        return segments
    filtered: list[dict[str, Any]] = []
    confirmed_masks = [
        _segment_mask_array(segment)
        for segment in confirmed_segments
        if segment.get("mask_rle") is not None
    ]
    confirmed_masks = [mask for mask in confirmed_masks if mask is not None]
    if not confirmed_masks:
        return segments

    for segment in segments:
        candidate_mask = _segment_mask_array(segment)
        if candidate_mask is None:
            filtered.append(segment)
            continue
        if any(
            _is_almost_confirmed_duplicate(
                candidate_mask=candidate_mask,
                confirmed_mask=confirmed_mask,
            )
            for confirmed_mask in confirmed_masks
        ):
            continue
        filtered.append(segment)
    removed_count = len(segments) - len(filtered)
    if removed_count:
        log.log_info(
            "removed overlapping confirmed segment candidates "
            f"before VLM count={removed_count}"
        )
    return filtered


def _segment_mask_array(segment: dict[str, Any]) -> np.ndarray | None:
    mask_rle = segment.get("mask_rle")
    if mask_rle is None:
        return None
    try:
        return np.asarray(decode_rle_mask(mask_rle).convert("L"), dtype=np.uint8) > 0
    except Exception:
        return None


def _is_almost_confirmed_duplicate(
    *,
    candidate_mask: np.ndarray,
    confirmed_mask: np.ndarray,
) -> bool:
    if candidate_mask.shape != confirmed_mask.shape:
        return False
    candidate_area = int(np.count_nonzero(candidate_mask))
    confirmed_area = int(np.count_nonzero(confirmed_mask))
    if candidate_area <= 0 or confirmed_area <= 0:
        return False
    intersection = int(np.count_nonzero(candidate_mask & confirmed_mask))
    union = candidate_area + confirmed_area - intersection
    candidate_covered = intersection / float(candidate_area)
    iou = intersection / float(union) if union > 0 else 0.0
    return (
        candidate_covered >= CONFIRMED_MASK_COVERAGE_THRESHOLD
        or iou >= CONFIRMED_MASK_IOU_THRESHOLD
    )
