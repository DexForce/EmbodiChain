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
    "EstimateMetricScalesRequest",
    "EstimateMetricScalesResult",
    "GlobalMetricScaleRequest",
    "MakeAssetSimreadyRequest",
    "MakeAssetSimreadyResult",
    "MakeTableSimreadyRequest",
    "MakeTableSimreadyResult",
    "MetricScaleObjectInput",
]


@dataclass(frozen=True)
class MakeAssetSimreadyRequest:
    input_path: Path
    output_path: Path
    input_up_axis: list[float] | None = None
    up_axis: list[float] | None = None
    ground_clearance: float = 0.01


@dataclass(frozen=True)
class MakeAssetSimreadyResult:
    output_path: Path
    transform_matrix: list[list[float]]


@dataclass(frozen=True)
class MakeTableSimreadyRequest:
    input_path: Path
    output_path: Path
    input_up_axis: list[float] | None = None
    up_axis: list[float] | None = None
    ground_clearance: float = 0.01


@dataclass(frozen=True)
class MakeTableSimreadyResult:
    output_path: Path
    transform_matrix: list[list[float]]


@dataclass(frozen=True)
class MetricScaleObjectInput:
    object_id: str
    object_name: str
    object_description: str
    mesh_path: Path


@dataclass(frozen=True)
class EstimateMetricScalesRequest:
    objects: list[MetricScaleObjectInput]
    messages: list[dict[str, Any]]
    schema: dict[str, Any]
    llm: Any
    context: str
    method: str
    step_name: str = "metric_scale"
    raw_output_path: Path | None = None


@dataclass(frozen=True)
class EstimateMetricScalesResult:
    status: str
    object_scales: list[dict[str, Any]]
    object_payload: list[dict[str, Any]]
    raw_model_output: dict[str, Any] | None = None
    reason: str = ""


@dataclass(frozen=True)
class GlobalMetricScaleRequest:
    objects: list[dict[str, Any]]
    object_scenes: list[tuple[str, Any]]
    min_scale: float = 0.10
    max_scale: float = 10.00
