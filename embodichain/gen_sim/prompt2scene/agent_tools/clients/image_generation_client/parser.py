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

import requests

from embodichain.gen_sim.prompt2scene.agent_tools.clients.common import (
    validate_png_response,
)
from embodichain.gen_sim.prompt2scene.agent_tools.clients.image_generation_client.schemas import (
    ImageGenerationResult,
    ImageGenerationServerRequest,
    ImageGenerationServerResponse,
)
from embodichain.gen_sim.prompt2scene.utils.log import log_info

__all__ = ["parse_generation_response"]


def parse_generation_response(
    response: requests.Response,
    request: ImageGenerationServerRequest,
) -> ImageGenerationServerResponse:
    """Parse a Z-Image PNG response and save it to the request output path."""
    png_bytes = response.content
    validate_png_response(response, png_bytes)
    output_path = _write_png_output(request, png_bytes)
    result = ImageGenerationResult(image_path=str(output_path))
    return ImageGenerationServerResponse(
        ok=True,
        status="ok",
        result=result,
        status_code=response.status_code,
        content_type=response.headers.get("Content-Type"),
        headers=dict(response.headers),
    )


def _write_png_output(
    request: ImageGenerationServerRequest,
    png_bytes: bytes,
) -> Path:
    output_path = Path(request.output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(png_bytes)
    if not output_path.is_file():
        raise FileNotFoundError(f"Generated image was not written: {output_path}")
    log_info(f"Generated image written: {output_path}")
    return output_path
