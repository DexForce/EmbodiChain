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

"""Client for the Z-Image image generation server."""

from __future__ import annotations

from pathlib import Path

import requests

from embodichain.gen_sim.prompt2scene.agent_tools.clients.base import BaseHttpClient
from embodichain.gen_sim.prompt2scene.agent_tools.clients.common import (
    validate_required_strings,
)
from embodichain.gen_sim.prompt2scene.agent_tools.clients.config import (
    DEFAULT_CLIENT_CONFIG_PATH,
)
from embodichain.gen_sim.prompt2scene.agent_tools.clients.image_generation_client.parser import (
    parse_generation_response,
)
from embodichain.gen_sim.prompt2scene.agent_tools.clients.image_generation_client.schemas import (
    ImageGenerationError,
    ImageGenerationResult,
    ImageGenerationServerRequest,
    ImageGenerationServerResponse,
)

__all__ = [
    "DEFAULT_CLIENT_CONFIG_PATH",
    "ImageGenerationClient",
    "ImageGenerationError",
    "ImageGenerationResult",
    "ImageGenerationServerRequest",
    "ImageGenerationServerResponse",
]


class ImageGenerationClient(BaseHttpClient):
    """Client for making single-image Z-Image generation requests."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        timeout_s: int | None = None,
        config_path: Path | None = None,
        config_key: str = "zimage",
        session: requests.Session | None = None,
    ) -> None:
        """Initialize the image generation client."""
        super().__init__(
            config_key=config_key,
            server_name="Image generation server",
            base_url=base_url,
            timeout_s=timeout_s,
            config_path=config_path,
            session=session,
        )
        self.generate_single_object_path = str(
            self.config.get("generate_single_object_path", "/generate.png")
        )

    def generate(
        self,
        request: ImageGenerationServerRequest,
        *,
        max_retries: int = 3,
    ) -> ImageGenerationServerResponse | ImageGenerationError:
        """Generate one image and save the returned PNG locally."""
        _validate_request(request)
        url = f"{self.base_url}{self.generate_single_object_path}"
        response = self.post_with_retries(
            lambda: _post_generation_request(self, url, request),
            max_retries=max_retries,
            error_cls=ImageGenerationError,
            request_label="image_generation",
        )
        if isinstance(response, ImageGenerationError):
            return response
        return parse_generation_response(response, request)


def _validate_request(request: ImageGenerationServerRequest) -> None:
    validate_required_strings(
        {
            "Image generation prompt": request.prompt,
            "Image generation output_path": request.output_path,
        }
    )
    if not str(request.output_path).lower().endswith(".png"):
        raise ValueError("Image generation output_path must be a PNG file path.")


def _post_generation_request(
    client: ImageGenerationClient,
    url: str,
    request: ImageGenerationServerRequest,
) -> requests.Response:
    return client.session.post(
        url,
        json=request.to_dict(),
        timeout=(10, client.timeout_s),
    )
