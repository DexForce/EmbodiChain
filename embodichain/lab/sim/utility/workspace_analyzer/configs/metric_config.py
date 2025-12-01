# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
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

from dataclasses import dataclass
from typing import List
from enum import Enum


class MetricType(Enum):
    """Types of workspace metrics."""

    REACHABILITY = "reachability"
    """Measures reachable workspace volume and coverage."""

    MANIPULABILITY = "manipulability"
    """Measures dexterity/manipulability throughout workspace."""

    DENSITY = "density"
    """Measures point density and workspace distribution."""

    ALL = "all"
    """Compute all available metrics."""


@dataclass
class ReachabilityConfig:
    """Configuration for reachability metric."""

    voxel_size: float = 0.01
    """Size of voxels for volume calculation in meters."""

    min_points_per_voxel: int = 1
    """Minimum number of points in a voxel to consider it reachable."""

    compute_coverage: bool = True
    """Whether to compute coverage percentage relative to bounding box."""


@dataclass
class ManipulabilityConfig:
    """Configuration for manipulability metric."""

    jacobian_threshold: float = 0.01
    """Minimum manipulability value to consider valid."""

    compute_isotropy: bool = True
    """Whether to compute isotropy index (condition number)."""

    compute_heatmap: bool = False
    """Whether to generate manipulability heatmap."""


@dataclass
class DensityConfig:
    """Configuration for density metric."""

    radius: float = 0.05
    """Radius for local density estimation in meters."""

    k_neighbors: int = 30
    """Number of nearest neighbors for density estimation."""

    compute_distribution: bool = True
    """Whether to compute density distribution statistics."""


@dataclass
class MetricConfig:
    """Configuration for workspace analysis metrics."""

    enabled_metrics: List[MetricType] = None
    """List of metrics to compute. If None, computes all metrics."""

    reachability: ReachabilityConfig = None
    """Configuration for reachability metric."""

    manipulability: ManipulabilityConfig = None
    """Configuration for manipulability metric."""

    density: DensityConfig = None
    """Configuration for density metric."""

    save_results: bool = True
    """Whether to save metric results to file."""

    output_format: str = "json"
    """Output format for metrics. Options: 'json', 'yaml', 'pkl'."""

    def __post_init__(self):
        """Initialize sub-configs with defaults if not provided."""
        if self.enabled_metrics is None:
            self.enabled_metrics = [MetricType.ALL]

        if self.reachability is None:
            self.reachability = ReachabilityConfig()

        if self.manipulability is None:
            self.manipulability = ManipulabilityConfig()

        if self.density is None:
            self.density = DensityConfig()
