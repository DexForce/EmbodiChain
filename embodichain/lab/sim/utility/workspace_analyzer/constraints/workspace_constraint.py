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
    """Workspace constraint checker for robotic workspace analysis.

    Main features:
    - Boundary constraint checking (prevents robot from exceeding safe working range)
    - Excluded zone checking (avoids obstacles, tables, and other fixed objects)
    - Ground height constraints (prevents robot collision with ground or work surface)
    - Configuration-based creation for easy reuse across different scenarios

    Extends base checker with excluded zones and configuration-based setup.
    """

    def __init__(
        self,
        min_bounds: np.ndarray | None = None,
        max_bounds: np.ndarray | None = None,
        ground_height: float = 0.0,
        exclude_zones: list[tuple[np.ndarray, np.ndarray]] | None = None,
        device: torch.device | None = None,
    ):
        """Initialize the workspace constraint checker.

        Args:
            min_bounds: Minimum bounds [x_min, y_min, z_min] in meters
            max_bounds: Maximum bounds [x_max, y_max, z_max] in meters
            ground_height: Ground plane height in meters
            exclude_zones: List of excluded zones as [(min_bounds, max_bounds), ...]
            device: PyTorch device (CPU/GPU)
        """
        super().__init__(min_bounds, max_bounds, ground_height, device)
        self.exclude_zones = exclude_zones or []

    @classmethod
    def from_config(
        cls, config: DimensionConstraint, device: torch.device | None = None
    ):
        """Create a constraint checker from a DimensionConstraint config (recommended approach).

        This is the recommended way to create constraint checkers. Using configuration objects
        allows for easy reuse of the same constraint settings across different scenarios.

        Args:
            config: Dimension constraint configuration object containing bounds, excluded zones, etc.
            device: PyTorch device for tensor operations

        Returns:
            Configured WorkspaceConstraintChecker instance

        Example:
            >>> from embodichain.lab.sim.utility.workspace_analyzer.configs import DimensionConstraint
            >>> config = DimensionConstraint(
            ...     min_bounds=np.array([-1, -1, 0]),
            ...     max_bounds=np.array([1, 1, 2]),
            ...     exclude_zones=[]
            ... )
            >>> checker = WorkspaceConstraintChecker.from_config(config)
        """
        return cls(
            min_bounds=config.min_bounds,
            max_bounds=config.max_bounds,
            ground_height=config.ground_height,
            exclude_zones=config.exclude_zones,
            device=device,
        )

    def check_collision(
        self, points: torch.Tensor | np.ndarray
    ) -> torch.Tensor | np.ndarray:
        """Check if points are not in excluded zones (collision checking).

        This method checks whether the robot end-effector positions would collide with
        predefined obstacle zones such as cabinets, tables, walls, and other fixed obstacles.

        Args:
            points: Array of shape (N, 3) containing 3D point positions

        Returns:
            Boolean array of shape (N,) indicating which points are collision-free
            True = collision-free (safe), False = collision detected (dangerous)
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
        self, points: torch.Tensor | np.ndarray
    ) -> torch.Tensor | np.ndarray:
        """Filter points to keep only those satisfying all constraints.

        This is a comprehensive filtering method that checks both boundary constraints
        and collision constraints, returning only safe and reachable workspace points.

        Args:
            points: Array of shape (N, 3) containing 3D point positions

        Returns:
            Filtered array of shape (M, 3) where M <= N
            Contains only points that satisfy all constraint conditions
        """
        # First filter by bounds
        valid_bounds = self.check_bounds(points)

        # Then filter by collision
        valid_collision = self.check_collision(points)

        # Combine both checks
        valid = valid_bounds & valid_collision

        return points[valid]

    def add_exclude_zone(self, min_bounds: np.ndarray, max_bounds: np.ndarray) -> None:
        """Add an excluded zone (obstacle region) to the workspace.

        Used for dynamically adding obstacles such as temporarily placed objects,
        newly installed equipment, etc.

        Args:
            min_bounds: Minimum bounds of the excluded zone [x_min, y_min, z_min] in meters
            max_bounds: Maximum bounds of the excluded zone [x_max, y_max, z_max] in meters

        Example:
            >>> # Add a cubic obstacle
            >>> checker.add_exclude_zone(
            ...     min_bounds=np.array([0.2, 0.2, 0.0]),
            ...     max_bounds=np.array([0.4, 0.4, 0.2])
            ... )
        """
        self.exclude_zones.append((np.array(min_bounds), np.array(max_bounds)))

    def clear_exclude_zones(self) -> None:
        """Remove all excluded zones.

        Used to reset the workspace environment by clearing all previously
        configured obstacle zones.
        """
        self.exclude_zones = []

    def get_num_exclude_zones(self) -> int:
        """Get the number of current excluded zones.

        Returns:
            Number of excluded zones
        """
        return len(self.exclude_zones)


# ====================
# Usage Examples
# ====================
"""
Typical usage scenarios:

1. Basic usage - Setting workspace boundaries:
```python
import numpy as np
from embodichain.lab.sim.utility.workspace_analyzer.configs import DimensionConstraint
from embodichain.lab.sim.utility.workspace_analyzer.constraints import WorkspaceConstraintChecker

# Create via configuration (recommended approach)
config = DimensionConstraint(
    min_bounds=np.array([-0.8, -0.6, 0.0]),  # Work range: ±80cm left/right, ±60cm front/back, height ≥ 0cm
    max_bounds=np.array([0.8, 0.6, 1.5]),   # Max height 150cm
    ground_height=0.0,                      # Ground height
    exclude_zones=[]                        # No obstacles initially
)
checker = WorkspaceConstraintChecker.from_config(config)

# Check if points are within safe range
test_points = np.array([
    [0.5, 0.3, 0.8],   # Safe point
    [1.0, 0.0, 0.5],   # Exceeds x boundary
    [0.0, 0.0, -0.1],  # Below ground level
])
valid_mask = checker.check_bounds(test_points)
print(f"Safe points: {valid_mask}")  # [True, False, False]
```

2. Advanced usage - Adding obstacle zones:
```python
# Add a table as an obstacle
checker.add_exclude_zone(
    min_bounds=np.array([0.2, 0.1, 0.0]),
    max_bounds=np.array([0.7, 0.5, 0.8])
)

# Check both boundaries and collisions simultaneously
test_points = np.array([
    [0.1, 0.0, 0.5],   # Within boundaries, not in obstacle zone
    [0.4, 0.3, 0.4],   # Within boundaries, but inside table zone
])
valid_bounds = checker.check_bounds(test_points)
valid_collision = checker.check_collision(test_points) 
overall_valid = valid_bounds & valid_collision
print(f"Comprehensive safety check: {overall_valid}")  # [True, False]

# Or use the comprehensive filtering method directly
safe_points = checker.filter_points(test_points)
print(f"Filtered safe points: {safe_points}")  # Only returns [0.1, 0.0, 0.5]
```

3. Application in robot workspace analysis:
```python
# Usage in WorkspaceAnalyzer
from embodichain.lab.sim.utility.workspace_analyzer import WorkspaceAnalyzer

# Configure work environment with complex obstacles
config = DimensionConstraint(
    min_bounds=np.array([-1.0, -1.0, 0.0]),
    max_bounds=np.array([1.0, 1.0, 2.0]),
    exclude_zones=[
        # Table
        (np.array([0.3, 0.2, 0.0]), np.array([0.8, 0.7, 0.8])),
        # Pillar
        (np.array([-0.1, -0.1, 0.0]), np.array([0.1, 0.1, 2.0])),
    ]
)

# WorkspaceAnalyzer will automatically use these constraints to filter invalid workspace points
analyzer = WorkspaceAnalyzer(robot, config=analyzer_config)
results = analyzer.analyze(visualize=True)
```
"""
