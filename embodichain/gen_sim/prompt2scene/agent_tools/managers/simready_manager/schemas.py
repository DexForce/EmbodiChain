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
class MakeAssetSimreadyRequest:
    """Request to prepare a general asset GLB for simulation placement."""

    input_path: Path
    output_path: Path
    input_up_axis: list[float] | None = None
    up_axis: list[float] | None = None
    ground_clearance: float = 0.01


@dataclass(frozen=True)
class MakeAssetSimreadyResult:
    """Result of making an asset simulation-ready."""

    output_path: Path
    transform_matrix: list[list[float]]


@dataclass(frozen=True)
class MakeTableSimreadyRequest:
    """Request to prepare a generated table GLB for simulation placement."""

    input_path: Path
    output_path: Path
    input_up_axis: list[float] | None = None
    up_axis: list[float] | None = None
    ground_clearance: float = 0.01


@dataclass(frozen=True)
class MakeTableSimreadyResult:
    """Result of making a table simulation-ready."""

    output_path: Path
    transform_matrix: list[list[float]]
