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

from dataclasses import dataclass
from pathlib import Path

from embodichain.gen_sim.prompt2scene.agent_tools.clients.image_segmentation_client import (
    ImageSegmentationCandidate,
)


@dataclass(frozen=True)
class AssetImageToRgbaRequest:
    """Request for converting an asset image to an RGBA cutout."""

    image_path: Path
    prompt: str
    output_path: Path


@dataclass(frozen=True)
class ImageSegmentationRequest:
    """Request for segmenting one image with one text prompt."""

    image_path: Path
    prompt: str


@dataclass(frozen=True)
class ImageSegmentationResult:
    """Segmentation candidates."""

    candidates: list[ImageSegmentationCandidate]
