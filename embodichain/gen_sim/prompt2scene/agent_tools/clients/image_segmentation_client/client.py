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

"""Client for the SAM3 image segmentation server."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import requests

from embodichain.gen_sim.prompt2scene.agent_tools.clients.base import BaseHttpClient
from embodichain.gen_sim.prompt2scene.agent_tools.clients.common import (
    validate_required_strings,
)
from embodichain.gen_sim.prompt2scene.agent_tools.clients.config import (
    DEFAULT_CLIENT_CONFIG_PATH,
)
from embodichain.gen_sim.prompt2scene.agent_tools.clients.image_segmentation_client.parser import (
    parse_segmentation_response,
)
from embodichain.gen_sim.prompt2scene.agent_tools.clients.image_segmentation_client.schemas import (
    ImageSegmentationCandidate,
    ImageSegmentationError,
    ImageSegmentationResult,
    ImageSegmentationServerRequest,
    ImageSegmentationServerResponse,
)

__all__ = [
    "DEFAULT_CLIENT_CONFIG_PATH",
    "ImageSegmentationCandidate",
    "ImageSegmentationClient",
    "ImageSegmentationError",
    "ImageSegmentationResult",
    "ImageSegmentationServerRequest",
    "ImageSegmentationServerResponse",
]


class ImageSegmentationClient(BaseHttpClient):
    """Client for making single-image SAM3 segmentation requests."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        timeout_s: int | None = None,
        config_path: Path | None = None,
        config_key: str = "sam3_segmentation",
        session: requests.Session | None = None,
    ) -> None:
        """Initialize the image segmentation client."""
        super().__init__(
            config_key=config_key,
            server_name="Image segmentation server",
            base_url=base_url,
            timeout_s=timeout_s,
            config_path=config_path,
            session=session,
            trust_env=False,
        )
        self.segmentation_path = str(
            self.config.get("segment_single_object_path", "/segment_single_object")
        )

    def segment(
        self,
        request: ImageSegmentationServerRequest,
        *,
        max_retries: int = 3,
    ) -> ImageSegmentationServerResponse | ImageSegmentationError:
        """Segment one image with a text prompt."""
        _validate_request(request)
        url = f"{self.base_url}{self.segmentation_path}"
        response = self.post_with_retries(
            lambda: _post_segmentation_request(self, url, request),
            max_retries=max_retries,
            error_cls=ImageSegmentationError,
            request_label="image_segmentation",
        )
        if isinstance(response, ImageSegmentationError):
            return response
        return parse_segmentation_response(response, request)


def _validate_request(request: ImageSegmentationServerRequest) -> None:
    validate_required_strings(
        {
            "Image segmentation image_path": request.image_path,
        }
    )
    image_path = Path(request.image_path).expanduser()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image segmentation input not found: {image_path}")


def _post_segmentation_request(
    client: ImageSegmentationClient,
    url: str,
    request: ImageSegmentationServerRequest,
) -> requests.Response:
    with _open_image_file(request.image_path) as image_file:
        return client.session.post(
            url,
            data=request.to_form_data(),
            files={
                "image": (
                    Path(request.image_path).name,
                    image_file,
                )
            },
            timeout=(10, client.timeout_s),
        )


def _open_image_file(image_path: str | Path) -> Any:
    return Path(image_path).expanduser().resolve().open("rb")
