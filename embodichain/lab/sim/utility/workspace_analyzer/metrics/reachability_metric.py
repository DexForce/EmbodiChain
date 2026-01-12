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

from typing import Dict, Any
import numpy as np
from embodichain.lab.sim.utility.workspace_analyzer.metrics.base_metric import (
    BaseMetric,
)
from embodichain.lab.sim.utility.workspace_analyzer.configs.metric_config import (
    ReachabilityConfig,
)


class ReachabilityMetric(BaseMetric):
    """Reachability metric for workspace analysis.

    Computes reachable workspace volume and coverage statistics.
    """

    def __init__(self, config: ReachabilityConfig | None = None):
        """Initialize reachability metric.

        Args:
            config: Reachability configuration.
        """
        super().__init__(config or ReachabilityConfig())

    def compute(
        self,
        workspace_points: np.ndarray,
        joint_configurations: np.ndarray | None = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Compute reachability metrics.

        Args:
            workspace_points: Workspace points in Cartesian space, shape (N, 3).
            joint_configurations: Joint configurations, shape (N, num_joints).
            **kwargs: Additional arguments.

        Returns:
            Dictionary containing:
                - volume: Estimated reachable volume (m³)
                - num_voxels: Number of occupied voxels
                - coverage: Coverage percentage relative to bounding box
                - bounding_box: Min and max bounds
                - centroid: Center of workspace
        """
        points = self._to_numpy(workspace_points)

        if len(points) == 0:
            return {
                "volume": 0.0,
                "num_voxels": 0,
                "coverage": 0.0,
                "bounding_box": {"min": None, "max": None},
                "centroid": None,
            }

        # Compute bounding box
        min_bounds = points.min(axis=0)
        max_bounds = points.max(axis=0)

        # Compute centroid
        centroid = points.mean(axis=0)

        # Voxelize the workspace
        voxel_size = self.config.voxel_size
        voxel_grid = self._voxelize_points(points, voxel_size)

        # Count occupied voxels
        num_voxels = len(voxel_grid)

        # Estimate volume
        volume = num_voxels * (voxel_size**3)

        # Compute coverage if requested
        coverage = 0.0
        if self.config.compute_coverage:
            # Bounding box volume
            bbox_volume = np.prod(max_bounds - min_bounds)
            if bbox_volume > 0:
                coverage = (volume / bbox_volume) * 100.0

        self.results = {
            "volume": float(volume),
            "num_voxels": int(num_voxels),
            "coverage": float(coverage),
            "bounding_box": {
                "min": min_bounds.tolist(),
                "max": max_bounds.tolist(),
            },
            "centroid": centroid.tolist(),
            "voxel_size": voxel_size,
        }

        return self.results

    def _voxelize_points(
        self, points: np.ndarray, voxel_size: float
    ) -> Dict[tuple, int]:
        """Convert points to voxel grid.

        Args:
            points: Point cloud, shape (N, 3).
            voxel_size: Size of each voxel.

        Returns:
            Dictionary mapping voxel coordinates to point count.
        """
        # Convert points to voxel indices
        voxel_indices = np.floor(points / voxel_size).astype(int)

        # Count points in each voxel
        voxel_grid = {}
        for idx in voxel_indices:
            key = tuple(idx)
            voxel_grid[key] = voxel_grid.get(key, 0) + 1

        # Filter by minimum points threshold
        min_points = self.config.min_points_per_voxel
        voxel_grid = {k: v for k, v in voxel_grid.items() if v >= min_points}

        return voxel_grid
