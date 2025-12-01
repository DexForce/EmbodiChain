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
import math
from typing import Union, Optional

from .geometric_constraint import GeometricConstraint

__all__ = [
    "SphereConstraint",
]


class SphereConstraint(GeometricConstraint):
    """Sphere (n-dimensional ball) geometric constraint.

    This constraint defines a spherical region in n-dimensional space
    specified by center and radius.
    """

    def __init__(
        self,
        center: Union[torch.Tensor, np.ndarray],
        radius: Optional[float] = None,
        bounds: Optional[Union[torch.Tensor, np.ndarray]] = None,
        radius_mode: str = "inscribed",
        device: Optional[torch.device] = None,
    ):
        """Initialize the sphere constraint.

        Args:
            center: Center point of the sphere. Shape is (n_dims,).
            radius: Radius of the sphere. Must be positive. If None, will be computed from bounds.
            bounds: Bounding box for auto-calculating radius. Shape is (n_dims, 2).
                   Only used if radius is None.
            radius_mode: How to calculate radius from bounds. Options:
                        - "inscribed": Largest sphere that fits inside bounds
                        - "circumscribed": Smallest sphere that contains bounds
                        Only used if radius is None.
            device: PyTorch device for tensor operations. Defaults to cpu.
        """
        super().__init__(device)

        self.center = self._to_tensor(center)
        self.n_dims = self.center.shape[0]

        if self.center.ndim != 1:
            raise ValueError(f"Center must be 1D tensor, got shape {self.center.shape}")

        # Calculate radius
        if radius is not None:
            if radius <= 0:
                raise ValueError(f"Radius must be positive, got {radius}")
            self.radius = float(radius)
        elif bounds is not None:
            self.radius = self._calculate_radius_from_bounds(bounds, radius_mode)
        else:
            raise ValueError("Either radius or bounds must be provided")

    def _calculate_radius_from_bounds(
        self, bounds: Union[torch.Tensor, np.ndarray], mode: str = "inscribed"
    ) -> float:
        """Calculate radius from bounding box.

        Args:
            bounds: Tensor of shape (n_dims, 2) with [min, max] bounds.
            mode: "inscribed" or "circumscribed".

        Returns:
            Calculated radius.
        """
        bounds_tensor = self._to_tensor(bounds)

        if bounds_tensor.ndim != 2 or bounds_tensor.shape[1] != 2:
            raise ValueError(
                f"Bounds must have shape (n_dims, 2), got {bounds_tensor.shape}"
            )

        if bounds_tensor.shape[0] != self.n_dims:
            raise ValueError(
                f"Bounds dimensions ({bounds_tensor.shape[0]}) don't match center dimensions ({self.n_dims})"
            )

        dimensions = bounds_tensor[:, 1] - bounds_tensor[:, 0]

        if mode == "inscribed":
            # Largest sphere that fits inside bounds
            radius = float(torch.min(dimensions).item() / 2.0)
        elif mode == "circumscribed":
            # Smallest sphere that contains entire bounds
            radius = float(torch.norm(dimensions).item() / 2.0)
        else:
            raise ValueError(
                f"Unknown radius_mode: {mode}. Use 'inscribed' or 'circumscribed'"
            )

        return radius

    def contains(self, points: torch.Tensor, tolerance: float = 1e-6) -> torch.Tensor:
        """Check if points are within the sphere constraint.

        Args:
            points: Tensor of shape (N, n_dims) containing point positions.
            tolerance: Numerical tolerance for boundary points. Points within
                      tolerance of the boundary are considered inside.

        Returns:
            Boolean tensor of shape (N,) indicating which points are within the sphere.
        """
        if points.shape[1] != self.n_dims:
            raise ValueError(
                f"Points dimension ({points.shape[1]}) doesn't match constraint dimension ({self.n_dims})"
            )

        # Calculate squared distance from center
        distances_sq = torch.sum((points - self.center) ** 2, dim=1)

        # Check if within radius (with tolerance)
        radius_sq_with_tolerance = (self.radius + tolerance) ** 2
        return distances_sq <= radius_sq_with_tolerance

    def sample_uniform(self, num_samples: int) -> torch.Tensor:
        """Generate uniformly distributed samples within the sphere.

        Uses the standard algorithm for uniform sampling in n-dimensional ball:
        1. Sample direction uniformly on unit sphere using Gaussian method
        2. Sample radius with appropriate distribution: r^(1/n) for uniform volume

        Args:
            num_samples: Number of samples to generate.

        Returns:
            Tensor of shape (num_samples, n_dims) containing uniformly sampled points.
        """
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}")

        # Step 1: Generate random directions on unit sphere using Gaussian method
        # Sample from standard normal distribution and normalize
        directions = torch.randn(num_samples, self.n_dims, device=self.device)
        directions = directions / torch.norm(directions, dim=1, keepdim=True)

        # Step 2: Sample radial distances for uniform distribution in n-ball
        # For uniform distribution in n-dimensional ball, use r^(1/n) scaling
        u = torch.rand(num_samples, device=self.device)  # uniform [0, 1]
        radii = self.radius * (u ** (1.0 / self.n_dims))

        # Step 3: Combine direction and radius
        samples = self.center + directions * radii.unsqueeze(1)

        return samples

    def get_bounds(self) -> torch.Tensor:
        """Get the axis-aligned bounding box of the sphere.

        Returns:
            Tensor of shape (n_dims, 2) containing [min, max] bounds for each dimension.
        """
        bounds = torch.zeros(self.n_dims, 2, device=self.device)
        bounds[:, 0] = self.center - self.radius  # min bounds
        bounds[:, 1] = self.center + self.radius  # max bounds
        return bounds

    def get_volume(self) -> float:
        """Calculate the volume of the n-dimensional sphere.

        Uses the formula: V_n(r) = π^(n/2) * r^n / Γ(n/2 + 1)
        where Γ is the gamma function.

        Returns:
            Volume of the sphere region.
        """
        # Use the relationship: Γ(n/2 + 1) = (n/2)! for integer n/2
        # For general case, use the gamma function
        n = self.n_dims

        if n == 1:
            # Line segment: length = 2r
            return 2.0 * self.radius
        elif n == 2:
            # Circle: area = πr²
            return math.pi * (self.radius**2)
        elif n == 3:
            # Sphere: volume = (4/3)πr³
            return (4.0 / 3.0) * math.pi * (self.radius**3)
        else:
            # General n-dimensional formula
            # V_n(r) = π^(n/2) * r^n / Γ(n/2 + 1)
            coefficient = (math.pi ** (n / 2.0)) / math.gamma(n / 2.0 + 1)
            return float(coefficient * (self.radius**n))

    def get_surface_area(self) -> float:
        """Calculate the surface area of the (n-1)-dimensional sphere boundary.

        Uses the formula: A_n(r) = n * V_n(r) / r

        Returns:
            Surface area of the sphere boundary.
        """
        if self.radius == 0:
            return 0.0

        volume = self.get_volume()
        return self.n_dims * volume / self.radius

    def sample_surface_uniform(self, num_samples: int) -> torch.Tensor:
        """Generate uniformly distributed samples on the sphere surface.

        Args:
            num_samples: Number of samples to generate.

        Returns:
            Tensor of shape (num_samples, n_dims) containing surface samples.
        """
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}")

        # Generate random directions on unit sphere using Gaussian method
        directions = torch.randn(num_samples, self.n_dims, device=self.device)
        directions = directions / torch.norm(directions, dim=1, keepdim=True)

        # Scale to sphere radius and translate to center
        samples = self.center + directions * self.radius

        return samples

    def get_circumscribed_box_bounds(self) -> torch.Tensor:
        """Get bounds of the circumscribed (smallest enclosing) axis-aligned box.

        This is the same as get_bounds() for a sphere.

        Returns:
            Tensor of shape (n_dims, 2) containing [min, max] bounds.
        """
        return self.get_bounds()

    def get_inscribed_box_bounds(self) -> torch.Tensor:
        """Get bounds of the inscribed (largest enclosed) axis-aligned box.

        Returns:
            Tensor of shape (n_dims, 2) containing [min, max] bounds.
        """
        # For n-dimensional sphere, inscribed box has side length 2r/√n
        half_side = self.radius / math.sqrt(self.n_dims)

        bounds = torch.zeros(self.n_dims, 2, device=self.device)
        bounds[:, 0] = self.center - half_side  # min bounds
        bounds[:, 1] = self.center + half_side  # max bounds
        return bounds

    @classmethod
    def from_bounds_inscribed(
        cls,
        bounds: Union[torch.Tensor, np.ndarray],
        device: Optional[torch.device] = None,
    ) -> "SphereConstraint":
        """Create inscribed sphere from bounding box.

        Args:
            bounds: Tensor of shape (n_dims, 2) with [min, max] bounds.
            device: PyTorch device.

        Returns:
            SphereConstraint inscribed within the bounds.

        Examples:
            >>> bounds = torch.tensor([[-1, 1], [-1, 1]])
            >>> sphere = SphereConstraint.from_bounds_inscribed(bounds)
            >>> print(sphere.radius)  # 1.0 (min dimension / 2)
        """
        bounds_tensor = (
            torch.tensor(bounds, dtype=torch.float32, device=device)
            if not isinstance(bounds, torch.Tensor)
            else bounds
        )
        center = (bounds_tensor[:, 0] + bounds_tensor[:, 1]) / 2.0
        return cls(center, bounds=bounds_tensor, radius_mode="inscribed", device=device)

    @classmethod
    def from_center_and_bounds(
        cls,
        center: Union[torch.Tensor, np.ndarray],
        bounds: Union[torch.Tensor, np.ndarray],
        radius_mode: str = "inscribed",
        device: Optional[torch.device] = None,
    ) -> "SphereConstraint":
        """Create sphere with custom center and radius calculated from bounds.

        Args:
            center: Center point of the sphere.
            bounds: Bounding box for radius calculation.
            radius_mode: "inscribed" or "circumscribed".
            device: PyTorch device.

        Returns:
            SphereConstraint with calculated radius.

        Examples:
            >>> center = [0, 0]
            >>> bounds = torch.tensor([[-2, 2], [-1, 1]])
            >>> sphere = SphereConstraint.from_center_and_bounds(center, bounds, "inscribed")
            >>> print(sphere.radius)  # 1.0 (min dimension / 2)
        """
        return cls(center, bounds=bounds, radius_mode=radius_mode, device=device)

    @classmethod
    def from_bounds_circumscribed(
        cls,
        bounds: Union[torch.Tensor, np.ndarray],
        device: Optional[torch.device] = None,
    ) -> "SphereConstraint":
        """Create circumscribed sphere from bounding box.

        Args:
            bounds: Tensor of shape (n_dims, 2) with [min, max] bounds.
            device: PyTorch device.

        Returns:
            SphereConstraint circumscribed around the bounds.

        Examples:
            >>> bounds = torch.tensor([[-1, 1], [-1, 1]])
            >>> sphere = SphereConstraint.from_bounds_circumscribed(bounds)
            >>> print(sphere.radius)  # sqrt(2) (diagonal / 2)
        """
        bounds_tensor = (
            torch.tensor(bounds, dtype=torch.float32, device=device)
            if not isinstance(bounds, torch.Tensor)
            else bounds
        )
        center = (bounds_tensor[:, 0] + bounds_tensor[:, 1]) / 2.0
        return cls(
            center, bounds=bounds_tensor, radius_mode="circumscribed", device=device
        )

    @classmethod
    def unit_sphere(
        cls,
        center: Union[torch.Tensor, np.ndarray],
        device: Optional[torch.device] = None,
    ) -> "SphereConstraint":
        """Create a unit sphere (radius=1.0) at given center.

        Args:
            center: Center point of the sphere.
            device: PyTorch device.

        Returns:
            SphereConstraint with radius=1.0.

        Examples:
            >>> sphere = SphereConstraint.unit_sphere([0, 0, 0])
            >>> print(sphere.radius)  # 1.0
        """
        return cls(center, radius=1.0, device=device)

    def __repr__(self) -> str:
        """String representation of the sphere constraint."""
        return (
            f"{self.__class__.__name__}("
            f"center={self.center.tolist()}, "
            f"radius={self.radius}, "
            f"n_dims={self.n_dims}, "
            f"volume={self.get_volume():.6f})"
        )
