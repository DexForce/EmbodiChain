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
class RgbaImageToGeometryRequest:
    """Request for converting one RGBA asset image to one mesh."""

    image_path: Path
    output_path: Path


@dataclass(frozen=True)
class RgbaImagesToGeometriesRequest:
    """Request for converting a scene image with object masks to meshes."""

    image_path: Path
    mask_paths: list[Path]
    output_dir: Path


@dataclass(frozen=True)
class RgbaImagesToGeometriesObject:
    """One generated object mesh and its scene placement."""

    name: str
    geometry_path: Path
    rotation_quaternion_wxyz: list[float]
    translation: list[float]
    scale: list[float]


@dataclass(frozen=True)
class RgbaImagesToGeometriesResult:
    """Result of multi-object geometry generation."""

    objects: list[RgbaImagesToGeometriesObject]
    sam3d_generation_elapsed_seconds: float = 0.0

    @property
    def geometry_paths(self) -> list[Path]:
        return [item.geometry_path for item in self.objects]


@dataclass(frozen=True)
class GeometryGenerationRequest:
    """Request for generating one object mesh from one image."""

    image_path: Path
    output_path: Path


@dataclass(frozen=True)
class GeometryGenerationResult:
    """Generated mesh path."""

    output_path: Path
    sam3d_generation_elapsed_seconds: float = 0.0


@dataclass(frozen=True)
class MultiObjectGenerationRequest:
    """Request to generate multiple object meshes from one image and masks."""

    image_path: Path
    mask_paths: list[Path]
    output_dir: Path


@dataclass(frozen=True)
class MultiObjectGenerationObject:
    """One generated object mesh and its scene placement."""

    name: str
    geometry_path: Path
    rotation_quaternion_wxyz: list[float]
    translation: list[float]
    scale: list[float]


@dataclass(frozen=True)
class MultiObjectGenerationResult:
    """Result of multi-object geometry generation."""

    objects: list[MultiObjectGenerationObject]
    sam3d_generation_elapsed_seconds: float = 0.0

    @property
    def geometry_paths(self) -> list[Path]:
        return [item.geometry_path for item in self.objects]
