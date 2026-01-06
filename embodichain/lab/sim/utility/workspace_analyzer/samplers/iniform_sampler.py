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
from typing import Union, TYPE_CHECKING

from embodichain.lab.sim.utility.workspace_analyzer.configs.sampling_config import (
    SamplingStrategy,
)
from embodichain.lab.sim.utility.workspace_analyzer.samplers.base_sampler import (
    BaseSampler,
)


from embodichain.utils import logger

# Note: Geometric constraint imports temporarily disabled

__all__ = ["UniformSampler"]


class UniformSampler(BaseSampler):
    """Uniform grid sampler.

    This sampler generates samples on a regular grid within the specified bounds.
    It ensures even coverage of the entire space, but suffers from the curse of
    dimensionality - the number of samples grows exponentially with the number
    of dimensions.

    Note: Geometric constraint sampling is temporarily disabled.

    Attributes:
        samples_per_dim: Number of samples to generate per dimension. When specified,
            this controls the grid density and takes precedence over num_samples.
            Total grid points = samples_per_dim^n_dims.
    """

    def __init__(
        self,
        seed: int = 42,
        samples_per_dim: int | None = None,
        device: torch.device | None = None,
    ):
        """Initialize the uniform sampler.

        Args:
            seed: Random seed for reproducibility. Defaults to 42.
            samples_per_dim: Fixed number of samples per dimension. If None,
                will be calculated automatically from num_samples. Defaults to None.
            device: PyTorch device for tensor operations.
        """
        super().__init__(seed, device)
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

    # Note: Constraint-based sampling methods temporarily disabled

    def get_strategy_name(self) -> str:
        """Get the name of the sampling strategy.

        Returns:
            String identifier for the sampling strategy.
        """
        return SamplingStrategy.UNIFORM.value

    def __repr__(self) -> str:
        """String representation of the sampler."""
        return (
            f"{self.__class__.__name__}("
            f"strategy={self.get_strategy_name()}, "
            f"samples_per_dim={self.samples_per_dim}, "
            f"seed={self.seed})"
        )
