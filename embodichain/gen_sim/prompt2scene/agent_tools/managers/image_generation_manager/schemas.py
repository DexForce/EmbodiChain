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


@dataclass(frozen=True)
class TextToAssetImageRequest:
    """Request for generating an asset image from a text prompt."""

    prompt: str
    output_path: Path


@dataclass(frozen=True)
class ImageGenerationRequest:
    """Request for generating one image from text."""

    prompt: str
    output_path: Path


@dataclass(frozen=True)
class ImageGenerationResult:
    """Generated image path."""

    image_path: Path
