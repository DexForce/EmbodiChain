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
    "AlignToAxisRequest",
    "AlignToAxisResult",
    "AlignXYLongAxisRequest",
    "AlignXYLongAxisResult",
    "CenterMeshRequest",
    "NormalizeRequest",
    "NormalizeResult",
    "CenterMeshResult",
    "ConvertUpAxisRequest",
    "ConvertUpAxisResult",
    "DetectTabletopRequest",
    "DetectTabletopResult",
    "ExportMeshRequest",
    "ExportMeshResult",
    "LoadMeshRequest",
    "LoadMeshResult",
    "PlaceAbovePlaneRequest",
    "PlaceAbovePlaneResult",
    "SupportPlaneCandidate",
]


@dataclass(frozen=True)
class SupportPlaneCandidate:
    """Candidate planar tabletop support surface."""

    normal: list[float]
    center: list[float]
    area: float
    face_indices: list[int]
    below_vertex_count: int
    above_vertex_count: int
    below_area_score: float
    above_area_score: float
    score: float


@dataclass(frozen=True)
class LoadMeshRequest:
    """Request to load a GLB/mesh file."""

    mesh_path: Path


@dataclass(frozen=True)
class LoadMeshResult:
    """Result of loading a mesh file."""

    mesh: Any


@dataclass(frozen=True)
class ExportMeshRequest:
    """Request to export a mesh to a file."""

    mesh: Any
    output_path: Path


@dataclass(frozen=True)
class ExportMeshResult:
    """Result of exporting a mesh."""

    output_path: Path


@dataclass(frozen=True)
class ConvertUpAxisRequest:
    """Request to convert a mesh from one up-axis convention to another."""

    mesh: Any
    input_up_axis: list[float] | None = None
    output_up_axis: list[float] | None = None


@dataclass(frozen=True)
class ConvertUpAxisResult:
    """Result of converting a mesh up-axis."""

    mesh: Any


@dataclass(frozen=True)
class CenterMeshRequest:
    """Request to center a mesh by its bounding-box center."""

    mesh: Any


@dataclass(frozen=True)
class CenterMeshResult:
    """Result of centering a mesh."""

    mesh: Any
    bbox_center: list[float]


@dataclass(frozen=True)
class AlignToAxisRequest:
    """Request to rotate a mesh so a source axis aligns to a target axis."""

    mesh: Any
    source_axis: list[float]
    target_axis: list[float]


@dataclass(frozen=True)
class AlignToAxisResult:
    """Result of aligning a mesh vector to an axis."""

    mesh: Any


@dataclass(frozen=True)
class PlaceAbovePlaneRequest:
    """Request to translate a mesh so its AABB bottom sits above the XY plane."""

    mesh: Any
    clearance: float = 0.01


@dataclass(frozen=True)
class PlaceAbovePlaneResult:
    """Result of placing a mesh above the XY plane."""

    mesh: Any


@dataclass(frozen=True)
class DetectTabletopRequest:
    """Request to detect the most likely tabletop plane in a mesh."""

    mesh: Any
    normal_angle_tol_deg: float = 8.0
    plane_distance_tol: float | None = None
    min_area_ratio: float = 0.02
    max_candidates: int = 24


@dataclass(frozen=True)
class DetectTabletopResult:
    """Result of detecting the tabletop plane with oriented normal."""

    selected: SupportPlaneCandidate
    oriented_normal: list[float]
    candidates: list[SupportPlaneCandidate]


@dataclass(frozen=True)
class AlignXYLongAxisRequest:
    """Request to align a mesh XY long axis to the Y axis via PCA."""

    mesh: Any
    face_indices: list[int] | None = None


@dataclass(frozen=True)
class AlignXYLongAxisResult:
    """Result of PCA yaw alignment."""

    mesh: Any
    yaw_angle_degrees: float


@dataclass(frozen=True)
class NormalizeRequest:
    """Request to normalize a mesh to a target size."""

    mesh: Any
    target_size: float = 1.0


@dataclass(frozen=True)
class NormalizeResult:
    """Result of normalizing a mesh."""

    mesh: Any
    scale_factor: float
