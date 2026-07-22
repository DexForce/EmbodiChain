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
    "ImageGenerationError",
    "ImageGenerationResult",
    "ImageGenerationServerRequest",
    "ImageGenerationServerResponse",
]


@dataclass(frozen=True)
class ImageGenerationServerRequest:
    """Request sent to the Z-Image server.

    Args:
        prompt: Text prompt used to generate the image.
        output_path: Local output PNG path where the client saves the response.
    """

    prompt: str
    output_path: str | Path

    def to_dict(self) -> dict[str, Any]:
        """Convert the request to the Z-Image server JSON payload."""
        return {"prompt": self.prompt}


@dataclass(frozen=True)
class ImageGenerationResult:
    """Successful Z-Image generation result."""

    image_path: str


@dataclass(frozen=True)
class ImageGenerationServerResponse:
    """Parsed successful response from the Z-Image server."""

    ok: bool
    result: ImageGenerationResult
    status: str | None = None
    error: str | None = None
    status_code: int | None = None
    content_type: str | None = None
    headers: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ImageGenerationError(ClientError):
    """Image generation failure returned by the server."""
