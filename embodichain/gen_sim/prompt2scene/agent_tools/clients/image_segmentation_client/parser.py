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

from typing import Any

import requests

from embodichain.gen_sim.prompt2scene.agent_tools.clients.common import (
    parse_json_object_response,
)
from embodichain.gen_sim.prompt2scene.agent_tools.clients.image_segmentation_client.schemas import (
    ImageSegmentationCandidate,
    ImageSegmentationResult,
    ImageSegmentationServerRequest,
    ImageSegmentationServerResponse,
)

__all__ = ["parse_segmentation_response"]

SERVER_NAME = "Image segmentation server"


def parse_segmentation_response(
    response: requests.Response,
    request: ImageSegmentationServerRequest,
) -> ImageSegmentationServerResponse:
    """Parse a SAM3 server JSON response into typed segmentation records."""
    response_data = parse_json_object_response(
        response,
        server_name=SERVER_NAME,
    )
    result = _parse_segmentation_result(response_data, request)
    return ImageSegmentationServerResponse(
        ok=bool(response_data.get("ok", True)),
        status=_string_or_none(response_data.get("status")) or "ok",
        result=result,
        status_code=response.status_code,
        content_type=response.headers.get("Content-Type"),
        headers=dict(response.headers),
    )


def _parse_segmentation_result(
    response_data: dict[str, Any],
    request: ImageSegmentationServerRequest,
) -> ImageSegmentationResult:
    result_data = response_data.get("result")
    if not isinstance(result_data, dict):
        result_data = response_data.get("data")
    if not isinstance(result_data, dict):
        result_data = response_data

    return ImageSegmentationResult(
        image_path=_string_or_none(result_data.get("image_path"))
        or str(request.image_path),
        prompt=_string_or_none(result_data.get("prompt")) or request.prompt,
        candidates=_parse_candidates(result_data),
        request_id=_string_or_none(result_data.get("request_id")),
        elapsed_sec=_float_or_none(result_data.get("elapsed_sec")),
        count=_int_or_none(result_data.get("count")),
        image_width=_parse_image_width(result_data),
        image_height=_parse_image_height(result_data),
        box_format=_string_or_none(result_data.get("box_format")) or "xyxy",
        mask_format=_string_or_none(result_data.get("mask_format")) or "rle",
    )


def _parse_candidates(result_data: dict[str, Any]) -> list[ImageSegmentationCandidate]:
    for key in ("instances", "candidates", "segmentations", "detections"):
        items = result_data.get(key)
        if isinstance(items, list):
            return [
                _parse_candidate_item(item, index)
                for index, item in enumerate(items)
                if isinstance(item, dict)
            ]

    boxes = result_data.get("boxes", [])
    scores = result_data.get("scores", [])
    masks = result_data.get("masks", [])
    if not isinstance(boxes, list):
        return []

    candidates: list[ImageSegmentationCandidate] = []
    for index, box in enumerate(boxes):
        candidates.append(
            ImageSegmentationCandidate(
                candidate_id=f"candidate_{index}",
                bbox_xyxy=_float_list(box),
                score=_float_or_zero(_list_get(scores, index)),
                mask_rle=_mask_or_none(_list_get(masks, index)),
            )
        )
    return candidates


def _parse_candidate_item(
    item: dict[str, Any],
    index: int,
) -> ImageSegmentationCandidate:
    known_keys = {
        "candidate_id",
        "id",
        "index",
        "bbox_xyxy",
        "box_xyxy",
        "box",
        "bbox",
        "score",
        "mask_rle",
        "mask",
        "segmentation",
        "mask_path",
        "label",
    }
    mask_value = item.get("mask_rle") or item.get("mask") or item.get("segmentation")
    return ImageSegmentationCandidate(
        candidate_id=_string_or_none(item.get("candidate_id"))
        or _string_or_none(item.get("id"))
        or _index_id_or_none(item.get("index"))
        or f"candidate_{index}",
        bbox_xyxy=_float_list(
            item.get("bbox_xyxy")
            or item.get("box_xyxy")
            or item.get("box")
            or item.get("bbox")
        ),
        score=_float_or_zero(item.get("score")),
        mask_rle=_mask_or_none(mask_value),
        mask_path=_string_or_none(item.get("mask_path")),
        label=_string_or_none(item.get("label")),
        metadata={k: v for k, v in item.items() if k not in known_keys},
    )


def _list_get(values: Any, index: int) -> Any:
    if not isinstance(values, list) or index >= len(values):
        return None
    return values[index]


def _float_list(value: Any) -> list[float]:
    if not isinstance(value, list):
        return []
    parsed: list[float] = []
    for item in value:
        try:
            parsed.append(float(item))
        except (TypeError, ValueError):
            continue
    return parsed


def _float_or_zero(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _string_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _mask_or_none(value: Any) -> dict[str, Any] | None:
    return value if isinstance(value, dict) else None


def _index_id_or_none(value: Any) -> str | None:
    index = _int_or_none(value)
    return f"candidate_{index}" if index is not None else None


def _parse_image_width(result_data: dict[str, Any]) -> int | None:
    image_size = result_data.get("image_size")
    if isinstance(image_size, dict):
        width = _int_or_none(image_size.get("width"))
        if width is not None:
            return width
    return _int_or_none(result_data.get("image_width"))


def _parse_image_height(result_data: dict[str, Any]) -> int | None:
    image_size = result_data.get("image_size")
    if isinstance(image_size, dict):
        height = _int_or_none(image_size.get("height"))
        if height is not None:
            return height
    return _int_or_none(result_data.get("image_height"))
