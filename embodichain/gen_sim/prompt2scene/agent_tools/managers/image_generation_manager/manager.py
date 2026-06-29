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

from embodichain.gen_sim.prompt2scene.agent_tools.clients.image_generation_client import (
    ImageGenerationClient,
    ImageGenerationError,
    ImageGenerationServerRequest,
)
from embodichain.gen_sim.prompt2scene.agent_tools.managers.image_generation_manager.schemas import (
    ImageGenerationRequest,
    ImageGenerationResult,
    TextToAssetImageRequest,
)

ASSET_IMAGE_PROMPT_SUFFIX = (
    "single isolated object, centered, fully visible, "
    "on a high contrast colored background. "
)


class ImageGenerationManager:
    """Image generation domain operations."""

    def __init__(self, *, client: ImageGenerationClient | None = None) -> None:
        self.client = client or ImageGenerationClient()

    def generate_image(self, request: ImageGenerationRequest) -> ImageGenerationResult:
        output_path = request.output_path.expanduser().resolve()
        response = self.client.generate(
            ImageGenerationServerRequest(
                prompt=request.prompt,
                output_path=output_path,
            ),
        )
        if isinstance(response, ImageGenerationError):
            raise RuntimeError(response.error_message)

        return ImageGenerationResult(
            image_path=Path(response.result.image_path).expanduser().resolve(),
        )

    def generate_asset_image_from_text(
        self,
        request: TextToAssetImageRequest,
    ) -> Path:
        prompt = _build_asset_image_prompt(request.prompt)
        result = self.generate_image(
            ImageGenerationRequest(prompt=prompt, output_path=request.output_path)
        )
        return result.image_path


def _build_asset_image_prompt(prompt: str) -> str:
    prompt = prompt.strip()
    if not prompt:
        raise ValueError("Text-to-asset image prompt must be non-empty.")
    if ASSET_IMAGE_PROMPT_SUFFIX in prompt:
        return prompt
    return f"{prompt}, {ASSET_IMAGE_PROMPT_SUFFIX}"
