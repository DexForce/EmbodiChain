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
from typing import Union

try:
    from scipy.stats import qmc

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

__all__ = ["LatinHypercubeSampler"]

from embodichain.lab.sim.utility.workspace_analyzer.configs.sampling_config import (
    SamplingStrategy,
)
from embodichain.lab.sim.utility.workspace_analyzer.samplers.base_sampler import (
    BaseSampler,
)

from embodichain.utils import logger


class LatinHypercubeSampler(BaseSampler):
    """Latin Hypercube Sampler (LHS) for stratified sampling.

    Latin Hypercube Sampling ensures that each dimension is divided into
    equally probable intervals, with exactly one sample in each interval.
    This provides better coverage with fewer samples compared to random sampling.

    Advantages:
        - Excellent coverage with small sample sizes
        - Each dimension is uniformly sampled
        - No sample clustering in any dimension
        - Works well in high dimensions
        - Popular in experimental design and sensitivity analysis

    Disadvantages:
        - Samples may align in projections (can be mitigated with optimization)
        - Not deterministic across different sample sizes
        - May have correlation between dimensions (unless optimized)

    Attributes:
        strength: Strength of the LHS (1 or 2). Higher strength reduces correlation.
        optimization: Optimization method ('random-cd', 'lloyd', None).
                     'random-cd': Random coordinate descent (fast, good quality)
                     'lloyd': Lloyd's algorithm (slower, better quality)
                     None: No optimization (fastest, may have correlation)
    """

    def __init__(
        self,
        seed: int = 42,
        device: torch.device | None = None,
        strength: int = 1,
        optimization: str | None = "random-cd",
    ):
        """Initialize the Latin Hypercube sampler.

        Args:
            seed: Random seed for reproducibility. Defaults to 42.
            device: PyTorch device (cpu/cuda). Defaults to cpu.
            strength: Strength of the LHS (1 or 2). Defaults to 1.
                     Strength 1: Standard LHS
                     Strength 2: Improved spacing (requires more samples)
            optimization: Optimization method to reduce correlation.
                         'random-cd': Fast, good quality (recommended)
                         'lloyd': Better quality, slower
                         None: No optimization (fastest)
                         Defaults to 'random-cd'.
            constraint: Optional geometric constraint for sampling (e.g., SphereConstraint).
        """
        super().__init__(seed, device)
        self.strength = strength
        self.optimization = optimization

        if not SCIPY_AVAILABLE and optimization is not None:
            logger.log_warning(
                "scipy is not available. LHS optimization will be disabled. "
                "Install scipy for optimized sampling: pip install scipy"
            )
            self.optimization = None

    def sample(
        self, bounds: torch.Tensor | np.ndarray, num_samples: int
    ) -> torch.Tensor:
        """Generate Latin Hypercube samples within the given bounds.

        Args:
            bounds: Tensor/Array of shape (n_dims, 2) containing [lower, upper] bounds.
            num_samples: Number of samples to generate.

        Returns:
            Tensor of shape (num_samples, n_dims) containing the sampled points.

        Raises:
            ValueError: If bounds are invalid or num_samples is non-positive.

        Examples:
            >>> sampler = LatinHypercubeSampler(seed=42, optimization='random-cd')
            >>> bounds = torch.tensor([[-1.0, 1.0], [-1.0, 1.0]], dtype=torch.float32)
            >>> samples = sampler.sample(bounds, num_samples=50)
            >>> samples.shape
            torch.Size([50, 2])
        """
        bounds = self._validate_bounds(bounds)

        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}")

        n_dims = bounds.shape[0]

        # Check strength constraints
        if self.strength == 2:
            min_samples = n_dims + 1
            if num_samples < min_samples:
                logger.log_warning(
                    f"Strength 2 LHS requires at least {min_samples} samples for {n_dims} dimensions. "
                    f"Got {num_samples}. Falling back to strength 1."
                )
                strength = 1
            else:
                strength = self.strength
        else:
            strength = self.strength

        # Generate Latin Hypercube samples
        if SCIPY_AVAILABLE and self.optimization is not None:
            samples_unit = self._generate_lhs_scipy(n_dims, num_samples, strength)
        else:
            samples_unit = self._generate_lhs_basic(n_dims, num_samples)

        # Convert to tensor and scale to bounds
        samples_unit_tensor = self._to_tensor(samples_unit)
        samples = self._scale_samples(samples_unit_tensor, bounds)

        # Validate samples
        self._validate_samples(samples, bounds)

        return samples

    def _generate_lhs_scipy(
        self, n_dims: int, num_samples: int, strength: int
    ) -> np.ndarray:
        """Generate optimized LHS using scipy.

        Args:
            n_dims: Number of dimensions.
            num_samples: Number of samples to generate.
            strength: Strength of the LHS (1 or 2).

        Returns:
            Array of shape (num_samples, n_dims) with values in [0, 1].
        """
        # Create LHS engine
        lhs_engine = qmc.LatinHypercube(
            d=n_dims, strength=strength, optimization=self.optimization, seed=self.seed
        )

        # Generate samples
        samples = lhs_engine.random(n=num_samples)

        return samples.astype(np.float32)

    def _generate_lhs_basic(self, n_dims: int, num_samples: int) -> np.ndarray:
        """Generate basic LHS without optimization.

        This is a simple implementation that doesn't optimize for correlation
        but still provides the stratification property of LHS.

        Args:
            n_dims: Number of dimensions.
            num_samples: Number of samples to generate.

        Returns:
            Array of shape (num_samples, n_dims) with values in [0, 1].
        """
        samples = np.zeros((num_samples, n_dims), dtype=np.float32)

        for dim in range(n_dims):
            # Divide [0, 1] into num_samples equally-sized intervals
            intervals = np.arange(num_samples, dtype=np.float32) / num_samples

            # Sample uniformly within each interval
            samples[:, dim] = (
                intervals + self.rng.rand(num_samples).astype(np.float32) / num_samples
            )

            # Randomly permute to decorrelate dimensions
            self.rng.shuffle(samples[:, dim])

        return samples

    def get_strategy_name(self) -> str:
        """Get the name of the sampling strategy.

        Returns:
            String identifier for the sampling strategy.
        """
        return SamplingStrategy.LATIN_HYPERCUBE.value

    def __repr__(self) -> str:
        """String representation of the sampler."""
        return (
            f"{self.__class__.__name__}("
            f"strategy={self.get_strategy_name()}, "
            f"strength={self.strength}, "
            f"optimization={self.optimization}, "
            f"seed={self.seed})"
        )
