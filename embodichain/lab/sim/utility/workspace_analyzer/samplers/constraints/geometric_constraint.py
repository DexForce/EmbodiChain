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

import numpy as np
import torch
from abc import ABC, abstractmethod
from typing import Union, Optional

__all__ = [
    "GeometricConstraint",
]


class GeometricConstraint(ABC):
    """Abstract base class for geometric sampling constraints.

    This class defines the interface for geometric constraints that can be used
    to limit sampling to specific geometric regions (e.g., boxes, spheres).
    """

    def __init__(self, device: Optional[torch.device] = None):
        """Initialize the geometric constraint.

        Args:
            device: PyTorch device for tensor operations. Defaults to cpu.
        """
        self.device = device if device is not None else torch.device("cpu")

    @abstractmethod
    def contains(self, points: torch.Tensor) -> torch.Tensor:
        """Check if points are within the geometric constraint.

        Args:
            points: Tensor of shape (N, n_dims) containing point positions.

        Returns:
            Boolean tensor of shape (N,) indicating which points are within the constraint.
        """
        pass

    @abstractmethod
    def sample_uniform(self, num_samples: int) -> torch.Tensor:
        """Generate uniformly distributed samples within the constraint.

        Args:
            num_samples: Number of samples to generate.

        Returns:
            Tensor of shape (num_samples, n_dims) containing uniformly sampled points.
        """
        pass

    @abstractmethod
    def get_bounds(self) -> torch.Tensor:
        """Get the axis-aligned bounding box of the constraint.

        Returns:
            Tensor of shape (n_dims, 2) containing [min, max] bounds for each dimension.
        """
        pass

    @abstractmethod
    def get_volume(self) -> float:
        """Calculate the volume/area of the constrained region.

        Returns:
            Volume of the constraint region.
        """
        pass

    def filter_points(self, points: torch.Tensor) -> torch.Tensor:
        """Filter points to keep only those within the constraint.

        Args:
            points: Tensor of shape (N, n_dims) containing point positions.

        Returns:
            Filtered tensor of shape (M, n_dims) where M <= N.
        """
        valid = self.contains(points)
        return points[valid]

    def _to_tensor(self, data: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Convert data to torch.Tensor.

        Args:
            data: Input data (numpy array or torch tensor).

        Returns:
            PyTorch tensor on the configured device.
        """
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        return torch.tensor(data, dtype=torch.float32, device=self.device)
