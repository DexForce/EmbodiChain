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

from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np
import torch


class BaseMetric(ABC):
    """Base class for workspace metrics.

    All metrics should inherit from this class and implement the compute method.
    """

    def __init__(self, config: Any | None = None):
        """Initialize the metric.

        Args:
            config: Configuration object for the metric.
        """
        self.config = config
        self.results: Dict[str, Any] = {}

    @abstractmethod
    def compute(
        self,
        workspace_points: np.ndarray,
        joint_configurations: np.ndarray | None = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Compute the metric.

        Args:
            workspace_points: Workspace points in Cartesian space, shape (N, 3).
            joint_configurations: Joint configurations, shape (N, num_joints).
            **kwargs: Additional arguments specific to the metric.

        Returns:
            Dictionary containing metric results.
        """
        pass

    def reset(self) -> None:
        """Reset metric results."""
        self.results = {}

    def get_results(self) -> Dict[str, Any]:
        """Get computed metric results.

        Returns:
            Dictionary containing metric results.
        """
        return self.results

    def _to_numpy(self, data: Any) -> np.ndarray:
        """Convert data to numpy array.

        Args:
            data: Input data (numpy array or torch tensor).

        Returns:
            Numpy array.
        """
        if isinstance(data, torch.Tensor):
            return data.cpu().numpy()
        elif isinstance(data, np.ndarray):
            return data
        else:
            return np.array(data)
