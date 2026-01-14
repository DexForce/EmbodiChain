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
    ManipulabilityConfig,
)


class ManipulabilityMetric(BaseMetric):
    """Manipulability metric for workspace analysis.

    Computes dexterity and manipulability measures throughout the workspace.
    Note: Full implementation requires robot Jacobian computation.
    """

    def __init__(self, config: ManipulabilityConfig | None = None):
        """Initialize manipulability metric.

        Args:
            config: Manipulability configuration.
        """
        super().__init__(config or ManipulabilityConfig())

    def compute(
        self,
        workspace_points: np.ndarray,
        joint_configurations: np.ndarray | None = None,
        jacobians: np.ndarray | None = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Compute manipulability metrics.

        Args:
            workspace_points: Workspace points in Cartesian space, shape (N, 3).
            joint_configurations: Joint configurations, shape (N, num_joints).
            jacobians: Precomputed Jacobian matrices, shape (N, 6, num_joints).
            **kwargs: Additional arguments.

        Returns:
            Dictionary containing:
                - mean_manipulability: Average manipulability index
                - std_manipulability: Standard deviation
                - min_manipulability: Minimum value
                - max_manipulability: Maximum value
                - mean_condition: Average condition number (if isotropy enabled)
        """
        points = self._to_numpy(workspace_points)

        if len(points) == 0:
            return {
                "mean_manipulability": 0.0,
                "std_manipulability": 0.0,
                "min_manipulability": 0.0,
                "max_manipulability": 0.0,
            }

        # If Jacobians are not provided, we cannot compute true manipulability
        # Return placeholder statistics
        if jacobians is None:
            # Estimate based on distance from centroid (simple heuristic)
            centroid = points.mean(axis=0)
            distances = np.linalg.norm(points - centroid, axis=1)

            # Normalize to [0, 1] range (higher manipulability near center)
            max_dist = distances.max() if distances.max() > 0 else 1.0
            manipulability_scores = 1.0 - (distances / max_dist)

            # Filter by threshold
            valid_mask = manipulability_scores >= self.config.jacobian_threshold
            valid_scores = manipulability_scores[valid_mask]

            if len(valid_scores) == 0:
                valid_scores = np.array([0.0])
        else:
            # Compute true manipulability from Jacobians
            manipulability_scores = self._compute_manipulability_index(jacobians)
            valid_mask = manipulability_scores >= self.config.jacobian_threshold
            valid_scores = manipulability_scores[valid_mask]

        self.results = {
            "mean_manipulability": float(valid_scores.mean()),
            "std_manipulability": float(valid_scores.std()),
            "min_manipulability": float(valid_scores.min()),
            "max_manipulability": float(valid_scores.max()),
            "num_valid_points": int(len(valid_scores)),
        }

        # Compute isotropy if requested
        if self.config.compute_isotropy and jacobians is not None:
            condition_numbers = self._compute_condition_numbers(jacobians)
            self.results["mean_condition"] = float(condition_numbers.mean())
            self.results["std_condition"] = float(condition_numbers.std())

        return self.results

    def _compute_manipulability_index(self, jacobians: np.ndarray) -> np.ndarray:
        """Compute Yoshikawa manipulability index.

        Args:
            jacobians: Jacobian matrices, shape (N, 6, num_joints).

        Returns:
            Manipulability indices, shape (N,).
        """
        # Manipulability index: sqrt(det(J * J^T))
        manipulability = np.zeros(len(jacobians))

        for i, J in enumerate(jacobians):
            JJT = J @ J.T
            det = np.linalg.det(JJT)
            manipulability[i] = np.sqrt(max(det, 0))

        return manipulability

    def _compute_condition_numbers(self, jacobians: np.ndarray) -> np.ndarray:
        """Compute condition numbers of Jacobian matrices.

        Args:
            jacobians: Jacobian matrices, shape (N, 6, num_joints).

        Returns:
            Condition numbers, shape (N,).
        """
        condition_numbers = np.zeros(len(jacobians))

        for i, J in enumerate(jacobians):
            try:
                condition_numbers[i] = np.linalg.cond(J)
            except np.linalg.LinAlgError:
                # Singular matrix, use infinity as condition number
                condition_numbers[i] = np.inf

        return condition_numbers
