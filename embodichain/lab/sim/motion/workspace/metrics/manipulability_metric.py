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

from typing import Dict, Any
import numpy as np
from embodichain.utils import logger
from embodichain.lab.sim.motion.workspace.metrics.base_metric import (
    BaseMetric,
)
from embodichain.lab.sim.motion.workspace.configs.metric_config import (
    ManipulabilityConfig,
)


class ManipulabilityMetric(BaseMetric):
    """Manipulability metric for workspace analysis.

    Computes Yoshikawa manipulability statistics from robot Jacobians or from
    precomputed per-point scores. Without either input no statistics are
    produced: an earlier centroid-distance placeholder was measured to be
    *negatively* correlated with true manipulability and has been removed.
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
        manipulability_scores: np.ndarray | None = None,
        condition_numbers: np.ndarray | None = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Compute manipulability metrics.

        Args:
            workspace_points: Workspace points in Cartesian space, shape (N, 3).
            joint_configurations: Joint configurations, shape (N, num_joints).
            jacobians: Precomputed Jacobian matrices, shape (N, 6, num_joints).
            manipulability_scores: Precomputed per-point Yoshikawa scores,
                shape (N,). Takes precedence over ``jacobians``.
            condition_numbers: Precomputed per-point Jacobian condition
                numbers, shape (N,). Used for isotropy statistics when
                ``jacobians`` is not provided.
            **kwargs: Additional arguments.

        Returns:
            Dictionary containing:
                - mean_manipulability: Average manipulability index
                - std_manipulability: Standard deviation
                - min_manipulability: Minimum value
                - max_manipulability: Maximum value
                - mean_condition: Average condition number (if isotropy enabled
                  and Jacobians were provided)

            Without ``jacobians`` or ``manipulability_scores`` an empty dict is
            returned: true manipulability cannot be derived from Cartesian
            points alone, and fabricated statistics are worse than none.
        """
        points = self._to_numpy(workspace_points)

        if len(points) == 0:
            return {
                "mean_manipulability": 0.0,
                "std_manipulability": 0.0,
                "min_manipulability": 0.0,
                "max_manipulability": 0.0,
            }

        if manipulability_scores is not None:
            manipulability_scores = self._to_numpy(manipulability_scores)
        elif jacobians is not None:
            manipulability_scores = self._compute_manipulability_index(jacobians)
        else:
            logger.log_warning(
                "ManipulabilityMetric needs jacobians or precomputed scores; "
                "skipping (no placeholder statistics are produced)."
            )
            self.results = {}
            return self.results

        valid_mask = manipulability_scores >= self.config.jacobian_threshold
        valid_scores = manipulability_scores[valid_mask]
        if len(valid_scores) == 0:
            valid_scores = np.array([0.0])

        self.results = {
            "mean_manipulability": float(valid_scores.mean()),
            "std_manipulability": float(valid_scores.std()),
            "min_manipulability": float(valid_scores.min()),
            "max_manipulability": float(valid_scores.max()),
            "num_valid_points": int(len(valid_scores)),
        }

        # Compute isotropy if requested
        if self.config.compute_isotropy:
            if condition_numbers is None and jacobians is not None:
                condition_numbers = self._compute_condition_numbers(jacobians)
            if condition_numbers is not None:
                condition_numbers = self._to_numpy(condition_numbers)
                self.results["mean_condition"] = float(condition_numbers.mean())
                self.results["std_condition"] = float(condition_numbers.std())

        return self.results

    def _compute_manipulability_index(self, jacobians: np.ndarray) -> np.ndarray:
        """Compute Yoshikawa manipulability index with batched operations.

        Args:
            jacobians: Jacobian matrices, shape (N, rows, cols).

        Returns:
            Manipulability indices, shape (N,).
        """
        # Batch matrix multiply: J @ J^T for all samples
        JJT = np.matmul(jacobians, np.swapaxes(jacobians, -2, -1))

        # Batch determinant
        dets = np.linalg.det(JJT)

        # sqrt(max(0, det))
        return np.sqrt(np.maximum(dets, 0.0))

    def _compute_condition_numbers(self, jacobians: np.ndarray) -> np.ndarray:
        """Compute condition numbers of Jacobian matrices with batched SVD.

        Args:
            jacobians: Jacobian matrices, shape (N, rows, cols).

        Returns:
            Condition numbers, shape (N,).
        """
        try:
            _, singular_values, _ = np.linalg.svd(jacobians, full_matrices=False)
            # Condition number = max singular value / min singular value
            max_sv = singular_values[:, 0]
            min_sv = singular_values[:, -1]
            # Avoid division by zero
            min_sv = np.maximum(min_sv, 1e-15)
            return max_sv / min_sv
        except np.linalg.LinAlgError:
            # Fallback to per-matrix computation if batch SVD fails
            condition_numbers = np.zeros(len(jacobians))
            for i, J in enumerate(jacobians):
                try:
                    condition_numbers[i] = np.linalg.cond(J)
                except np.linalg.LinAlgError:
                    condition_numbers[i] = np.inf
            return condition_numbers
