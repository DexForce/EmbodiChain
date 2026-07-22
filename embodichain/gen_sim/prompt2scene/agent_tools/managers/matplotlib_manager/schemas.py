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
from typing import Any

__all__ = [
    "RenderFootprintLayoutRequest",
    "RenderFootprintLayoutResult",
    "RenderImageComparisonRequest",
    "RenderImageComparisonResult",
    "RenderSupportRegionRequest",
    "RenderSupportRegionResult",
    "RenderXYComparisonRequest",
    "RenderXYComparisonResult",
]


@dataclass(frozen=True)
class RenderFootprintLayoutRequest:
    """Request to render labeled top-down object footprints."""

    object_ids: list[str]
    centers: dict[str, Any]
    xy_sizes: dict[str, Any]
    output_path: Path
    title: str = ""


@dataclass(frozen=True)
class RenderFootprintLayoutResult:
    """Result of rendering a footprint layout."""

    output_path: Path


@dataclass(frozen=True)
class RenderImageComparisonRequest:
    """Request to render two labeled images side by side."""

    first_image_path: Path
    second_image_path: Path
    output_path: Path
    first_label: str = "1: normal"
    second_label: str = "2: flipped"


@dataclass(frozen=True)
class RenderImageComparisonResult:
    """Result of rendering an image comparison."""

    output_path: Path


@dataclass(frozen=True)
class RenderSupportRegionRequest:
    """Request to render a mesh with the selected support region highlighted."""

    mesh: Any
    face_indices: list[int]
    output_path: Path


@dataclass(frozen=True)
class RenderSupportRegionResult:
    """Result of rendering the support region."""

    output_path: Path


@dataclass(frozen=True)
class RenderXYComparisonRequest:
    """Request to render before/after XY projections for PCA yaw alignment."""

    before_mesh: Any
    after_mesh: Any
    angle_degrees: float
    output_path: Path


@dataclass(frozen=True)
class RenderXYComparisonResult:
    """Result of rendering the XY alignment comparison."""

    output_path: Path
