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

__all__ = ["RandomSampler"]


class RandomSampler(BaseSampler):
    """Random sampler using uniform distribution.

    This sampler generates samples uniformly at random within the specified bounds.
    It's simple and fast, but doesn't guarantee uniform coverage. Samples may
    cluster in some regions while leaving other regions sparse.

    Advantages:
        - Fast and simple
        - No assumptions about the space
        - Works well for any number of dimensions

    Disadvantages:
        - Uneven coverage (clusters and gaps)
        - Slower convergence than quasi-random methods
        - May miss important regions
    """

    def __init__(
        self,
        seed: int = 42,
        device: torch.device | None = None,
    ):
        """Initialize the random sampler.

        Args:
            seed: Random seed for reproducibility. Defaults to 42.
            device: PyTorch device for tensor operations.
        """
        super().__init__(seed, device)

    def _sample_from_bounds(
        self, bounds: Union[torch.Tensor, np.ndarray], num_samples: int
    ) -> torch.Tensor:
        """Generate random samples within the given bounds.

        Samples are drawn from a uniform distribution within each dimension's bounds.

        Args:
            bounds: Tensor/Array of shape (n_dims, 2) containing [lower, upper] bounds for each dimension.
            num_samples: Number of samples to generate.

        Returns:
            Tensor of shape (num_samples, n_dims) containing the sampled points.

        Raises:
            ValueError: If bounds are invalid or num_samples is non-positive.

        Examples:
            >>> sampler = RandomSampler(seed=42)
            >>> bounds = torch.tensor([[-1, 1], [-1, 1]], dtype=torch.float32)
            >>> samples = sampler.sample(bounds, num_samples=100)
            >>> samples.shape
            torch.Size([100, 2])
            >>> torch.all(samples >= -1) and torch.all(samples <= 1)
            True
        """
        bounds = self._validate_bounds(bounds)

        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}")

        n_dims = bounds.shape[0]

        # Generate random samples in [0, 1]^n_dims using torch
        samples_unit = torch.rand(num_samples, n_dims, device=self.device)

        # Scale to the actual bounds
        samples = self._scale_samples(samples_unit, bounds)

        # Validate samples
        self._validate_samples(samples, bounds)

        return samples

    def get_strategy_name(self) -> str:
        """Get the name of the sampling strategy.

        Returns:
            String identifier for the sampling strategy.
        """
        return SamplingStrategy.RANDOM.value
