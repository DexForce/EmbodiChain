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

from embodichain.gen_sim.prompt2scene.agent_tools.clients.image_segmentation_client import (
    ImageSegmentationClient,
    ImageSegmentationError,
    ImageSegmentationServerRequest,
    apply_mask_to_alpha,
    decode_rle_mask,
)
from embodichain.gen_sim.prompt2scene.agent_tools.managers.image_segmentation_manager.schemas import (
    AssetImageToRgbaRequest,
    ImageSegmentationRequest,
    ImageSegmentationResult,
)


class ImageSegmentationManager:
    """Image segmentation domain operations."""

    def __init__(self, *, client: ImageSegmentationClient | None = None) -> None:
        self.client = client or ImageSegmentationClient()

    def segment_image(
        self,
        request: ImageSegmentationRequest,
    ) -> ImageSegmentationResult:
        image_path = request.image_path.expanduser().resolve()
        _validate_segment_request(image_path=image_path, prompt=request.prompt)

        response = self.client.segment(
            ImageSegmentationServerRequest(
                prompt=request.prompt.strip(),
                image_path=image_path,
            ),
        )
        if isinstance(response, ImageSegmentationError):
            raise RuntimeError(response.error_message)

        return ImageSegmentationResult(candidates=list(response.result.candidates))

    def convert_asset_image_to_rgba(
        self,
        request: AssetImageToRgbaRequest,
    ) -> Path:
        segmentation_result = self.segment_image(
            ImageSegmentationRequest(
                image_path=request.image_path,
                prompt=request.prompt,
            )
        )
        if not segmentation_result.candidates:
            raise ValueError("Image segmentation returned no candidates.")

        candidate = segmentation_result.candidates[0]
        if candidate.mask_rle is None:
            raise ValueError(f"Candidate {candidate.candidate_id} has no mask_rle.")

        output_path = request.output_path.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mask = decode_rle_mask(candidate.mask_rle)
        rgba = apply_mask_to_alpha(request.image_path, mask)
        rgba.save(output_path)
        if not output_path.is_file():
            raise FileNotFoundError(f"RGBA image was not written: {output_path}")
        return output_path


def _validate_segment_request(*, image_path: Path, prompt: str) -> None:
    if not image_path.is_file():
        raise FileNotFoundError(f"Image segmentation input not found: {image_path}")
    if not prompt.strip():
        raise ValueError("Image segmentation prompt must be non-empty.")
