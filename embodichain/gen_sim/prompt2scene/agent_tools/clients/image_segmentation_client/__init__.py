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

from embodichain.gen_sim.prompt2scene.agent_tools.clients.config import (
    DEFAULT_CLIENT_CONFIG_PATH,
)
from embodichain.gen_sim.prompt2scene.agent_tools.clients.image_segmentation_client.client import (
    ImageSegmentationClient,
)
from embodichain.gen_sim.prompt2scene.agent_tools.clients.image_segmentation_client.schemas import (
    ImageSegmentationCandidate,
    ImageSegmentationError,
    ImageSegmentationResult,
    ImageSegmentationServerRequest,
    ImageSegmentationServerResponse,
)
from embodichain.gen_sim.prompt2scene.agent_tools.clients.image_segmentation_client.utils import (
    apply_mask_to_alpha,
    bbox_iou,
    decode_rle_mask,
    draw_labeled_bboxes,
    draw_numbered_bboxes,
    draw_numbered_masks,
    is_usable_segmentation_candidate,
    save_candidate_rgba_and_mask,
    sort_segments_by_bbox,
)

__all__ = [
    "DEFAULT_CLIENT_CONFIG_PATH",
    "ImageSegmentationCandidate",
    "ImageSegmentationClient",
    "ImageSegmentationError",
    "ImageSegmentationResult",
    "ImageSegmentationServerRequest",
    "ImageSegmentationServerResponse",
    "apply_mask_to_alpha",
    "bbox_iou",
    "decode_rle_mask",
    "draw_labeled_bboxes",
    "draw_numbered_bboxes",
    "draw_numbered_masks",
    "is_usable_segmentation_candidate",
    "save_candidate_rgba_and_mask",
    "sort_segments_by_bbox",
]
