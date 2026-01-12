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
from typing import Union
import numpy as np
import torch

from embodichain.utils import logger


__all__ = [
    "IConstraintChecker",
    "BaseConstraintChecker",
]


class IConstraintChecker:
    """Interface for constraint checkers.

    This protocol defines the contract that all constraint checkers must follow.
    """

    def check_bounds(
        self, points: torch.Tensor | np.ndarray
    ) -> torch.Tensor | np.ndarray:
        """Check if points are within bounds.

        Args:
            points: Array of shape (N, 3) containing point positions.

        Returns:
            Boolean array of shape (N,) indicating which points satisfy bounds.
        """
        ...

    def filter_points(
        self, points: torch.Tensor | np.ndarray
    ) -> torch.Tensor | np.ndarray:
        """Filter points to keep only those satisfying all constraints.

        Args:
            points: Array of shape (N, 3) containing point positions.

        Returns:
            Filtered array of shape (M, 3) where M <= N.
        """
        ...


class BaseConstraintChecker(ABC):
    """Abstract base class for workspace constraint checkers.

    Provides common functionality for checking dimensional constraints,
    bounds, excluded zones, and other workspace limitations.
    """

    def __init__(
        self,
        min_bounds: np.ndarray | None = None,
        max_bounds: np.ndarray | None = None,
        ground_height: float = 0.0,
        device: torch.device | None = None,
    ):
        """Initialize the constraint checker.

        Args:
            min_bounds: Minimum bounds [x_min, y_min, z_min]. If None, no lower limit.
            max_bounds: Maximum bounds [x_max, y_max, z_max]. If None, no upper limit.
            ground_height: Ground plane height. Points below this are filtered.
            device: PyTorch device for tensor operations. Defaults to cpu.
        """
        self.min_bounds = np.array(min_bounds) if min_bounds is not None else None
        self.max_bounds = np.array(max_bounds) if max_bounds is not None else None
        self.ground_height = ground_height
        self.device = device if device is not None else torch.device("cpu")

        # Validate bounds
        if self.min_bounds is not None and self.max_bounds is not None:
            if not np.all(self.min_bounds < self.max_bounds):
                logger.log_warning(
                    "min_bounds should be less than max_bounds in all dimensions"
                )

    def check_bounds(
        self, points: torch.Tensor | np.ndarray
    ) -> torch.Tensor | np.ndarray:
        """Check if points are within bounds.

        Args:
            points: Array of shape (N, 3) containing point positions.

        Returns:
            Boolean array of shape (N,) indicating which points are within bounds.
        """
        is_tensor = isinstance(points, torch.Tensor)

        if is_tensor:
            valid = torch.ones(len(points), dtype=torch.bool, device=points.device)
        else:
            valid = np.ones(len(points), dtype=bool)

        # Check minimum bounds
        if self.min_bounds is not None:
            if is_tensor:
                min_bounds_t = torch.tensor(
                    self.min_bounds, dtype=points.dtype, device=points.device
                )
                valid &= torch.all(points >= min_bounds_t, dim=1)
            else:
                valid &= np.all(points >= self.min_bounds, axis=1)

        # Check maximum bounds
        if self.max_bounds is not None:
            if is_tensor:
                max_bounds_t = torch.tensor(
                    self.max_bounds, dtype=points.dtype, device=points.device
                )
                valid &= torch.all(points <= max_bounds_t, dim=1)
            else:
                valid &= np.all(points <= self.max_bounds, axis=1)

        # Check ground height
        if is_tensor:
            valid &= points[:, 2] >= self.ground_height
        else:
            valid &= points[:, 2] >= self.ground_height

        return valid

    def filter_points(
        self, points: torch.Tensor | np.ndarray
    ) -> torch.Tensor | np.ndarray:
        """Filter points to keep only those satisfying all constraints.

        Args:
            points: Array of shape (N, 3) containing point positions.

        Returns:
            Filtered array of shape (M, 3) where M <= N.
        """
        valid = self.check_bounds(points)
        return points[valid]

    @abstractmethod
    def check_collision(
        self, points: torch.Tensor | np.ndarray
    ) -> torch.Tensor | np.ndarray:
        """Check if points are collision-free (to be implemented by subclasses).

        Args:
            points: Array of shape (N, 3) containing point positions.

        Returns:
            Boolean array of shape (N,) indicating which points are collision-free.
        """
        pass

    def get_bounds_volume(self) -> float:
        """Calculate the volume of the bounded workspace.

        Returns:
            Volume in cubic meters, or inf if unbounded.
        """
        if self.min_bounds is None or self.max_bounds is None:
            return float("inf")

        dimensions = self.max_bounds - self.min_bounds
        return float(np.prod(dimensions))

    def get_bounds_info(self) -> dict:
        """Get information about the workspace bounds.

        Returns:
            Dictionary containing bounds information.
        """
        return {
            "min_bounds": (
                self.min_bounds.tolist() if self.min_bounds is not None else None
            ),
            "max_bounds": (
                self.max_bounds.tolist() if self.max_bounds is not None else None
            ),
            "ground_height": self.ground_height,
            "volume": self.get_bounds_volume(),
        }
