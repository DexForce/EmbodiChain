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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from embodichain.gen_sim.prompt2scene.agent_tools.clients.common import ClientError

__all__ = [
    "ImageSegmentationCandidate",
    "ImageSegmentationError",
    "ImageSegmentationResult",
    "ImageSegmentationServerRequest",
    "ImageSegmentationServerResponse",
]


@dataclass(frozen=True)
class ImageSegmentationServerRequest:
    """Request sent to the SAM3 server.

    Args:
        prompt: Short text concept prompt.
        image_path: Local input image path.
    """

    prompt: str
    image_path: str | Path

    def to_form_data(self) -> dict[str, str]:
        """Convert the request to the SAM3 server multipart form fields."""
        return {
            "prompt": self.prompt,
            "score_threshold": "0.0",
            "max_instances": "5",
        }


@dataclass(frozen=True)
class ImageSegmentationCandidate:
    """One SAM3 segmentation candidate for a prompted concept.

    SAM3 image inference returns parallel masks, boxes, and scores. The client
    normalizes one aligned mask/box/score item into this candidate record.
    """

    candidate_id: str
    bbox_xyxy: list[float]
    score: float
    mask_rle: dict[str, Any] | None = None
    mask_path: str | None = None
    label: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ImageSegmentationResult:
    """Successful SAM3 segmentation result."""

    image_path: str
    prompt: str
    candidates: list[ImageSegmentationCandidate]
    request_id: str | None = None
    elapsed_sec: float | None = None
    count: int | None = None
    image_width: int | None = None
    image_height: int | None = None
    box_format: str = "xyxy"
    mask_format: str | None = None


@dataclass(frozen=True)
class ImageSegmentationServerResponse:
    """Parsed successful response from the SAM3 server."""

    ok: bool
    result: ImageSegmentationResult
    status: str | None = None
    error: str | None = None
    status_code: int | None = None
    content_type: str | None = None
    headers: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ImageSegmentationError(ClientError):
    """Image segmentation failure returned by the server."""
