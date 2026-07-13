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

from embodichain.gen_sim.prompt2scene.agent_tools.clients.common import ClientError

__all__ = [
    "GeometryGenerationError",
    "GeometryGenerationResult",
    "GeometryGenerationServerRequest",
    "GeometryGenerationServerResponse",
    "MultiObjectGenerationError",
    "MultiObjectGenerationObject",
    "MultiObjectGenerationResult",
    "MultiObjectGenerationServerRequest",
    "MultiObjectGenerationServerResponse",
]


@dataclass(frozen=True)
class GeometryGenerationServerRequest:
    """Request sent to the Geometry Generation server.

    Args:
        image_path: Local object image path.
        output_path: Local output GLB path where the client saves the generated geometry.
    """

    image_path: str | Path
    output_path: str | Path

    def to_form_data(self) -> dict[str, str]:
        """Convert the request to the geometry server multipart form fields."""
        return {}


@dataclass(frozen=True)
class GeometryGenerationResult:
    """Successful Geometry Generation result."""

    geometry_path: str


@dataclass(frozen=True)
class GeometryGenerationServerResponse:
    """Parsed successful response from the Geometry Generation server."""

    ok: bool
    result: GeometryGenerationResult
    status: str | None = None
    error: str | None = None
    status_code: int | None = None
    content_type: str | None = None
    headers: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class GeometryGenerationError(ClientError):
    """Geometry generation failure returned by the server."""


@dataclass(frozen=True)
class MultiObjectGenerationServerRequest:
    """Request sent to the Geometry Generation server (multi-object).

    Args:
        image_path: Local scene RGB image path.
        mask_paths: Local mask PNG file paths (one per object).
    """

    image_path: str | Path
    mask_paths: list[Path]

    def to_form_data(self) -> dict[str, str]:
        """Convert the request to the geometry server multipart form fields."""
        return {"json": "1"}


@dataclass(frozen=True)
class MultiObjectGenerationObject:
    """Successful Multi-Object Geometry Generation result."""

    name: str
    geometry_path: str
    rotation_quaternion_wxyz: list[float]
    translation: list[float]
    scale: list[float]


@dataclass(frozen=True)
class MultiObjectGenerationResult:
    """Successful Multi-Object Geometry Generation result."""

    objects: list[MultiObjectGenerationObject]

    @property
    def geometry_paths(self) -> list[str]:
        """Paths to the generated GLB files."""
        return [item.geometry_path for item in self.objects]


@dataclass(frozen=True)
class MultiObjectGenerationServerResponse:
    """Parsed successful response from the Geometry Generation server."""

    ok: bool
    result: MultiObjectGenerationResult
    status: str | None = None
    error: str | None = None
    status_code: int | None = None
    content_type: str | None = None
    headers: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class MultiObjectGenerationError(ClientError):
    """Multi-object geometry generation failure returned by the server."""
