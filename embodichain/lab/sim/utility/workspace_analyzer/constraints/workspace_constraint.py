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

from typing import Union, List, Tuple, Optional
import numpy as np
import torch

from embodichain.lab.sim.utility.workspace_analyzer.constraints.base_constraint import (
    BaseConstraintChecker,
)
from embodichain.lab.sim.utility.workspace_analyzer.configs.dimension_constraint import (
    DimensionConstraint,
)


__all__ = [
    "WorkspaceConstraintChecker",
]


class WorkspaceConstraintChecker(BaseConstraintChecker):
    """Concrete implementation of constraint checker for workspace analysis.

    Extends base checker with support for excluded zones and configuration-based setup.
    """

    def __init__(
        self,
        min_bounds: Optional[np.ndarray] = None,
        max_bounds: Optional[np.ndarray] = None,
        ground_height: float = 0.0,
        exclude_zones: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
        device: Optional[torch.device] = None,
    ):
        """Initialize the workspace constraint checker.

        Args:
            min_bounds: Minimum bounds [x_min, y_min, z_min].
            max_bounds: Maximum bounds [x_max, y_max, z_max].
            ground_height: Ground plane height.
            exclude_zones: List of excluded zones as [(min_bounds, max_bounds), ...].
            device: PyTorch device for tensor operations.
        """
        super().__init__(min_bounds, max_bounds, ground_height, device)
        self.exclude_zones = exclude_zones or []

    @classmethod
    def from_config(
        cls, config: DimensionConstraint, device: Optional[torch.device] = None
    ):
        """Create a constraint checker from a DimensionConstraint config.

        Args:
            config: DimensionConstraint instance with constraint settings.
            device: PyTorch device for tensor operations.

        Returns:
            Configured WorkspaceConstraintChecker instance.
        """
        return cls(
            min_bounds=config.min_bounds,
            max_bounds=config.max_bounds,
            ground_height=config.ground_height,
            exclude_zones=config.exclude_zones,
            device=device,
        )

    def check_collision(
        self, points: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        """Check if points are not in excluded zones.

        Args:
            points: Array of shape (N, 3) containing point positions.

        Returns:
            Boolean array of shape (N,) indicating which points are collision-free.
        """
        is_tensor = isinstance(points, torch.Tensor)

        if is_tensor:
            valid = torch.ones(len(points), dtype=torch.bool, device=points.device)
        else:
            valid = np.ones(len(points), dtype=bool)

        # Check each excluded zone
        for min_zone, max_zone in self.exclude_zones:
            if is_tensor:
                min_zone_t = torch.tensor(
                    min_zone, dtype=points.dtype, device=points.device
                )
                max_zone_t = torch.tensor(
                    max_zone, dtype=points.dtype, device=points.device
                )
                # Points inside excluded zone
                in_zone = torch.all(points >= min_zone_t, dim=1) & torch.all(
                    points <= max_zone_t, dim=1
                )
                valid &= ~in_zone  # Exclude these points
            else:
                in_zone = np.all(points >= min_zone, axis=1) & np.all(
                    points <= max_zone, axis=1
                )
                valid &= ~in_zone

        return valid

    def filter_points(
        self, points: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        """Filter points to keep only those satisfying all constraints including collision check.

        Args:
            points: Array of shape (N, 3) containing point positions.

        Returns:
            Filtered array of shape (M, 3) where M <= N.
        """
        # First filter by bounds
        valid_bounds = self.check_bounds(points)

        # Then filter by collision
        valid_collision = self.check_collision(points)

        # Combine both checks
        valid = valid_bounds & valid_collision

        return points[valid]

    def add_exclude_zone(self, min_bounds: np.ndarray, max_bounds: np.ndarray) -> None:
        """Add an excluded zone to the workspace.

        Args:
            min_bounds: Minimum bounds of the excluded zone [x_min, y_min, z_min].
            max_bounds: Maximum bounds of the excluded zone [x_max, y_max, z_max].
        """
        self.exclude_zones.append((np.array(min_bounds), np.array(max_bounds)))

    def clear_exclude_zones(self) -> None:
        """Remove all excluded zones."""
        self.exclude_zones = []

    def get_num_exclude_zones(self) -> int:
        """Get the number of excluded zones.

        Returns:
            Number of excluded zones.
        """
        return len(self.exclude_zones)
