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
from typing import Optional, Union, TYPE_CHECKING

from embodichain.lab.sim.utility.workspace_analyzer.configs.sampling_config import (
    SamplingStrategy,
)
from embodichain.lab.sim.utility.workspace_analyzer.samplers.base_sampler import (
    BaseSampler,
)


from embodichain.utils import logger

if TYPE_CHECKING:
    from embodichain.lab.sim.utility.workspace_analyzer.samplers.constraints.geometric_constraint import (
        GeometricConstraint,
    )

__all__ = ["UniformSampler"]


class UniformSampler(BaseSampler):
    """Uniform grid sampler.

    This sampler generates samples on a regular grid within the specified bounds.
    It ensures even coverage of the entire space, but suffers from the curse of
    dimensionality - the number of samples grows exponentially with the number
    of dimensions.

    For geometric constraints, it uses grid-based filtering: generates a uniform
    grid in the constraint's bounding box and filters out points outside the constraint.
    This provides regular, predictable sampling patterns within complex geometries.

    Important: When using constraints, the actual number of samples returned depends
    on the grid density (samples_per_dim) and the constraint shape, not the num_samples
    parameter. For predictable results, always specify samples_per_dim.

    Attributes:
        samples_per_dim: Number of samples to generate per dimension. When specified,
            this controls the grid density and takes precedence over num_samples.
            Total grid points = samples_per_dim^n_dims.
    """

    def __init__(
        self,
        seed: int = 42,
        samples_per_dim: Optional[int] = None,
        constraint: Optional["GeometricConstraint"] = None,
        device: Optional[torch.device] = None,
    ):
        """Initialize the uniform sampler.

        Args:
            seed: Random seed for reproducibility. Defaults to 42.
            samples_per_dim: Fixed number of samples per dimension. If None,
                will be calculated automatically from num_samples. Defaults to None.
            constraint: Optional geometric constraint for sampling (e.g., SphereConstraint).
                When constraint is provided, uses grid-based filtering approach:
                generates uniform grid in constraint's bounding box and filters
                out points outside the constraint.
            device: PyTorch device for tensor operations.
        """
        super().__init__(seed, device, constraint)
        self.samples_per_dim = samples_per_dim

    def _sample_from_bounds(
        self, bounds: Union[torch.Tensor, np.ndarray], num_samples: int
    ) -> torch.Tensor:
        """Generate uniform grid samples within the given bounds.

        Args:
            bounds: Tensor/Array of shape (n_dims, 2) containing [lower, upper] bounds for each dimension.
            num_samples: Total number of samples to generate. This is used to calculate
                samples_per_dim if not explicitly provided during initialization.
                Note: The actual number of samples may differ slightly from this value
                to maintain a uniform grid.

        Returns:
            Tensor of shape (actual_num_samples, n_dims) containing the sampled points.
            The actual number of samples will be samples_per_dim^n_dims.

        Raises:
            ValueError: If bounds are invalid.

        Examples:
            >>> sampler = UniformSampler(samples_per_dim=3)
            >>> bounds = torch.tensor([[-1, 1], [-1, 1]], dtype=torch.float32)
            >>> samples = sampler.sample(bounds, num_samples=10)
            >>> samples.shape
            torch.Size([9, 2])  # 3^2 = 9 samples
        """
        bounds = self._validate_bounds(bounds)

        n_dims = bounds.shape[0]

        # Calculate samples per dimension if not provided
        if self.samples_per_dim is None:
            # Compute samples_per_dim to approximate the desired num_samples
            samples_per_dim = max(2, int(np.ceil(num_samples ** (1.0 / n_dims))))
        else:
            samples_per_dim = self.samples_per_dim

        actual_num_samples = samples_per_dim**n_dims

        if actual_num_samples != num_samples and self.samples_per_dim is None:
            logger.log_info(
                f"Uniform grid: requested {num_samples} samples, "
                f"generating {actual_num_samples} samples "
                f"({samples_per_dim}^{n_dims}) for uniform coverage."
            )

        # Create uniform grid for each dimension
        samples = self._create_grid(bounds, samples_per_dim)

        # Validate samples
        self._validate_samples(samples, bounds)

        return samples

    def _create_grid(self, bounds: torch.Tensor, samples_per_dim: int) -> torch.Tensor:
        """Create a uniform grid of samples.

        Args:
            bounds: Tensor of shape (n_dims, 2) containing [lower, upper] bounds.
            samples_per_dim: Number of samples per dimension.

        Returns:
            Tensor of shape (samples_per_dim^n_dims, n_dims) containing grid points.
        """
        n_dims = bounds.shape[0]

        # Create linspace for each dimension
        grids = []
        for i in range(n_dims):
            grid = torch.linspace(
                bounds[i, 0].item(),
                bounds[i, 1].item(),
                samples_per_dim,
                device=self.device,
            )
            grids.append(grid)

        # Create meshgrid and flatten
        mesh = torch.meshgrid(*grids, indexing="ij")
        samples = torch.stack([m.flatten() for m in mesh], dim=-1)

        return samples

    def _sample_from_constraint(self, num_samples: int) -> torch.Tensor:
        """Sample from the geometric constraint.

        Uses the constraint's own sampling method, passing along the samples_per_dim
        parameter for grid density control when available.

        Args:
            num_samples: Target number of samples to generate.

        Returns:
            Tensor containing sampled points within the constraint.
        """
        # Check if constraint supports samples_per_dim parameter
        try:
            return self.constraint.sample_uniform(
                num_samples, samples_per_dim=self.samples_per_dim
            )
        except TypeError:
            # Fallback for constraints that don't support samples_per_dim
            return self.constraint.sample_uniform(num_samples)

    def _sample_from_constraint_with_grid_filtering(
        self, num_samples: int
    ) -> torch.Tensor:
        """Sample using grid-based filtering approach.

        This method generates a uniform grid within the constraint's bounding box,
        then filters out points that don't satisfy the constraint. The grid density
        is controlled by samples_per_dim parameter for predictable results.

        Args:
            num_samples: Target number of samples to generate (used only for estimation if samples_per_dim is None).

        Returns:
            Tensor of shape (actual_num_samples, n_dims) containing filtered grid points.
            Note: actual_num_samples depends on grid density and constraint shape.
        """
        # Get constraint bounds
        bounds = self.constraint.get_bounds()
        n_dims = bounds.shape[0]

        # Calculate samples per dimension for the grid
        if self.samples_per_dim is None:
            # If no specific grid density is set, ensure sufficient density for IK analysis
            acceptance_rate = self._estimate_constraint_acceptance_rate()
            # Use higher multiplier for adequate workspace coverage for IK
            target_grid_samples = int(num_samples / max(acceptance_rate, 0.1) * 2.5)
            samples_per_dim = max(
                15,
                int(
                    np.ceil(target_grid_samples ** (1.0 / n_dims))
                ),  # minimum 15 per dim for IK
            )
        else:
            # Use the explicitly specified grid density
            samples_per_dim = self.samples_per_dim

        # Generate uniform grid
        grid_samples = self._create_grid(bounds, samples_per_dim)

        # Filter samples that satisfy the constraint
        valid_mask = self.constraint.contains(grid_samples)
        valid_samples = grid_samples[valid_mask]

        total_grid_samples = samples_per_dim**n_dims
        num_valid = len(valid_samples)

        logger.log_info(
            f"Grid filtering: {samples_per_dim}^{n_dims} = {total_grid_samples} grid samples, "
            f"{num_valid} valid samples "
            f"(acceptance rate: {num_valid/total_grid_samples:.3f})"
        )

        # If we have samples_per_dim specified but got too few samples, warn user
        if self.samples_per_dim is not None and num_valid < num_samples * 0.5:
            logger.log_warning(
                f"Grid density {samples_per_dim}^{n_dims} generated only {num_valid} samples, "
                f"much less than requested {num_samples}. Consider increasing samples_per_dim."
            )

        return valid_samples

    def _estimate_constraint_acceptance_rate(self) -> float:
        """Estimate acceptance rate for constraint filtering.

        This provides a rough estimate of how many grid points will be
        accepted by the constraint filter.

        Returns:
            Estimated acceptance rate between 0 and 1.
        """
        try:
            # Try to get volume-based estimate if both constraint and its bounds support it
            constraint_volume = self.constraint.get_volume()
            bounds = self.constraint.get_bounds()
            box_dimensions = bounds[:, 1] - bounds[:, 0]
            box_volume = float(torch.prod(box_dimensions).item())

            if box_volume > 0:
                return min(1.0, constraint_volume / box_volume)
        except (AttributeError, NotImplementedError):
            pass

        # Fallback: conservative estimate for common constraint types
        constraint_type = type(self.constraint).__name__.lower()
        if "sphere" in constraint_type:
            # For sphere in box: roughly π/4 ≈ 0.785 in 2D, 4π/3/8 ≈ 0.524 in 3D
            n_dims = self.constraint.get_bounds().shape[0]
            if n_dims == 2:
                return 0.785  # π/4
            elif n_dims == 3:
                return 0.524  # 4π/3 / 8
            else:
                # General n-dimensional estimate (gets lower as dimensions increase)
                return max(0.1, 0.8 / (1.5 ** (n_dims - 2)))

        # Conservative default for unknown constraint types
        return 0.5

    def get_strategy_name(self) -> str:
        """Get the name of the sampling strategy.

        Returns:
            String identifier for the sampling strategy.
        """
        return SamplingStrategy.UNIFORM.value

    def __repr__(self) -> str:
        """String representation of the sampler."""
        constraint_info = (
            f", constraint={type(self.constraint).__name__}" if self.constraint else ""
        )
        return (
            f"{self.__class__.__name__}("
            f"strategy={self.get_strategy_name()}, "
            f"samples_per_dim={self.samples_per_dim}{constraint_info}, "
            f"seed={self.seed})"
        )
