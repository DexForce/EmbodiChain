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
    DensityConfig,
)


class DensityMetric(BaseMetric):
    """Density metric for workspace analysis.

    Computes point density and workspace distribution statistics.
    """

    def __init__(self, config: DensityConfig | None = None):
        """Initialize density metric.

        Args:
            config: Density configuration.
        """
        super().__init__(config or DensityConfig())

    def compute(
        self,
        workspace_points: np.ndarray,
        joint_configurations: np.ndarray | None = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Compute density metrics.

        Args:
            workspace_points: Workspace points in Cartesian space, shape (N, 3).
            joint_configurations: Joint configurations, shape (N, num_joints).
            **kwargs: Additional arguments.

        Returns:
            Dictionary containing:
                - mean_density: Average local density
                - std_density: Standard deviation
                - min_density: Minimum density
                - max_density: Maximum density
                - density_distribution: Histogram of density values (if enabled)
        """
        points = self._to_numpy(workspace_points)

        if len(points) < 2:
            return {
                "mean_density": 0.0,
                "std_density": 0.0,
                "min_density": 0.0,
                "max_density": 0.0,
            }

        # Compute local density for each point
        densities = self._compute_local_density(points)

        self.results = {
            "mean_density": float(densities.mean()),
            "std_density": float(densities.std()),
            "min_density": float(densities.min()),
            "max_density": float(densities.max()),
        }

        # Compute distribution statistics if requested
        if self.config.compute_distribution:
            hist, bin_edges = np.histogram(densities, bins=10)
            self.results["density_distribution"] = {
                "histogram": hist.tolist(),
                "bin_edges": bin_edges.tolist(),
            }

        return self.results

    def _compute_local_density(self, points: np.ndarray) -> np.ndarray:
        """Compute local density for each point.

        Args:
            points: Point cloud, shape (N, 3).

        Returns:
            Local densities, shape (N,).
        """
        n_points = len(points)
        densities = np.zeros(n_points)

        # Use radius-based density estimation for better performance
        radius = self.config.radius

        for i in range(n_points):
            # Compute distances to all other points
            distances = np.linalg.norm(points - points[i], axis=1)

            # Count neighbors within radius
            num_neighbors = np.sum(distances <= radius) - 1  # Exclude self

            # Density = neighbors / volume of sphere
            volume = (4.0 / 3.0) * np.pi * (radius**3)
            densities[i] = num_neighbors / volume if volume > 0 else 0.0

        return densities

    def _compute_knn_density(self, points: np.ndarray) -> np.ndarray:
        """Compute k-nearest neighbors density.

        Alternative method using k-nearest neighbors instead of fixed radius.

        Args:
            points: Point cloud, shape (N, 3).

        Returns:
            Local densities, shape (N,).
        """
        n_points = len(points)
        k = min(self.config.k_neighbors, n_points - 1)

        if k <= 0:
            return np.zeros(n_points)

        densities = np.zeros(n_points)

        for i in range(n_points):
            # Compute distances to all other points
            distances = np.linalg.norm(points - points[i], axis=1)

            # Find k-nearest neighbors (excluding self)
            distances[i] = np.inf
            knn_distances = np.partition(distances, k)[:k]

            # Density = k / volume of sphere containing k neighbors
            max_distance = knn_distances.max()
            volume = (4.0 / 3.0) * np.pi * (max_distance**3)
            densities[i] = k / volume if volume > 0 else 0.0

        return densities
