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
from typing import Union, Optional

from .geometric_constraint import GeometricConstraint

__all__ = [
    "BoxConstraint",
]


class BoxConstraint(GeometricConstraint):
    """Box (axis-aligned bounding box) geometric constraint.

    This constraint defines a rectangular region in n-dimensional space
    specified by min/max bounds for each dimension.
    """

    def __init__(
        self,
        bounds: Union[torch.Tensor, np.ndarray],
        device: Optional[torch.device] = None,
    ):
        """Initialize the box constraint.

        Args:
            bounds: Tensor/Array of shape (n_dims, 2) containing [lower, upper] bounds.
            device: PyTorch device for tensor operations. Defaults to cpu.
        """
        super().__init__(device)
        self.bounds = self._validate_and_convert_bounds(bounds)
        self.n_dims = self.bounds.shape[0]

    def _validate_and_convert_bounds(
        self, bounds: Union[torch.Tensor, np.ndarray]
    ) -> torch.Tensor:
        """Validate and convert bounds to tensor format.

        Args:
            bounds: Input bounds array.

        Returns:
            Validated bounds tensor.

        Raises:
            ValueError: If bounds are invalid.
        """
        bounds_tensor = self._to_tensor(bounds)

        if bounds_tensor.ndim != 2 or bounds_tensor.shape[1] != 2:
            raise ValueError(
                f"Bounds must have shape (n_dims, 2), got {bounds_tensor.shape}"
            )

        if torch.any(bounds_tensor[:, 0] >= bounds_tensor[:, 1]):
            raise ValueError(
                "Lower bounds must be strictly less than upper bounds. "
                f"Got bounds: {bounds_tensor}"
            )

        return bounds_tensor

    def contains(self, points: torch.Tensor) -> torch.Tensor:
        """Check if points are within the box constraint.

        Args:
            points: Tensor of shape (N, n_dims) containing point positions.

        Returns:
            Boolean tensor of shape (N,) indicating which points are within the box.
        """
        if points.shape[1] != self.n_dims:
            raise ValueError(
                f"Points dimension ({points.shape[1]}) doesn't match constraint dimension ({self.n_dims})"
            )

        lower_bounds = self.bounds[:, 0]  # (n_dims,)
        upper_bounds = self.bounds[:, 1]  # (n_dims,)

        # Check if all dimensions are within bounds
        within_lower = torch.all(points >= lower_bounds, dim=1)  # (N,)
        within_upper = torch.all(points <= upper_bounds, dim=1)  # (N,)

        return within_lower & within_upper

    def sample_uniform(self, num_samples: int) -> torch.Tensor:
        """Generate uniformly distributed samples within the box.

        Args:
            num_samples: Number of samples to generate.

        Returns:
            Tensor of shape (num_samples, n_dims) containing uniformly sampled points.
        """
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}")

        # Generate random samples in [0, 1]^n_dims
        samples_unit = torch.rand(num_samples, self.n_dims, device=self.device)

        # Scale to the actual bounds
        lower_bounds = self.bounds[:, 0]
        upper_bounds = self.bounds[:, 1]
        samples = lower_bounds + samples_unit * (upper_bounds - lower_bounds)

        return samples

    def get_bounds(self) -> torch.Tensor:
        """Get the axis-aligned bounding box of the constraint.

        Returns:
            Tensor of shape (n_dims, 2) containing [min, max] bounds for each dimension.
        """
        return self.bounds.clone()

    def get_volume(self) -> float:
        """Calculate the volume of the box.

        Returns:
            Volume of the box region.
        """
        dimensions = self.bounds[:, 1] - self.bounds[:, 0]  # (n_dims,)
        return float(torch.prod(dimensions).item())

    def get_center(self) -> torch.Tensor:
        """Get the center point of the box.

        Returns:
            Tensor of shape (n_dims,) containing the center coordinates.
        """
        return (self.bounds[:, 0] + self.bounds[:, 1]) / 2.0

    def get_dimensions(self) -> torch.Tensor:
        """Get the dimensions (size) of the box.

        Returns:
            Tensor of shape (n_dims,) containing the size in each dimension.
        """
        return self.bounds[:, 1] - self.bounds[:, 0]

    def __repr__(self) -> str:
        """String representation of the box constraint."""
        return (
            f"{self.__class__.__name__}("
            f"bounds={self.bounds.tolist()}, "
            f"n_dims={self.n_dims}, "
            f"volume={self.get_volume():.6f})"
        )
