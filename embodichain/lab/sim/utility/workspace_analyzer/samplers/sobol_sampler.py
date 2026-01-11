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

from embodichain.lab.sim.utility.workspace_analyzer.configs.sampling_config import (
    SamplingStrategy,
)
from embodichain.lab.sim.utility.workspace_analyzer.samplers.base_sampler import (
    BaseSampler,
)

from embodichain.utils import logger


class SobolSampler(BaseSampler):
    """Sobol sequence sampler using quasi-random low-discrepancy sequences.

    The Sobol sequence is a low-discrepancy sequence that provides excellent
    uniformity in high-dimensional spaces. It's widely used in finance,
    engineering, and scientific computing for Monte Carlo simulations.

    Advantages:
        - Excellent uniformity in high dimensions (up to ~40 dimensions)
        - Industry standard for quasi-Monte Carlo methods
        - Better convergence than random sampling (O(1/n) vs O(1/√n))
        - Well-suited for integration and optimization

    Disadvantages:
        - Requires scipy library
        - Sequential generation (but can be scrambled for randomization)
        - Initial points may not be well-distributed (use skip parameter)

    Attributes:
        scramble: Whether to scramble the sequence for better randomization.
        skip: Number of initial samples to skip.

    Notes:
        This implementation uses scipy.stats.qmc.Sobol for efficient generation.
        Falls back to a basic implementation if scipy is not available.
    """

    def __init__(
        self,
        seed: int = 42,
        device: torch.device | None = None,
        scramble: bool = True,
        skip: int = 0,
    ):
        """Initialize the Sobol sampler.

        Args:
            seed: Random seed for scrambling. Defaults to 42.
            device: PyTorch device (cpu/cuda). Defaults to cpu.
            scramble: Whether to scramble the sequence. Defaults to True.
                     Scrambling improves randomization while maintaining low discrepancy.
            skip: Number of initial samples to skip. Defaults to 0.
                  Recommended: 0 for scrambled, >0 (e.g., 100) for unscrambled.
            constraint: Optional geometric constraint for sampling (e.g., SphereConstraint).
        """
        super().__init__(seed, device)
        self.scramble = scramble
        self.skip = skip

        if not SCIPY_AVAILABLE:
            logger.log_warning(
                "scipy is not available. Sobol sampler will use a basic fallback implementation. "
                "For optimal performance, install scipy: pip install scipy"
            )

    def sample(
        self, bounds: torch.Tensor | np.ndarray, num_samples: int
    ) -> torch.Tensor:
        """Generate Sobol sequence samples within the given bounds.

        Args:
            bounds: Tensor/Array of shape (n_dims, 2) containing [lower, upper] bounds.
            num_samples: Number of samples to generate.

        Returns:
            Tensor of shape (num_samples, n_dims) containing the sampled points.

        Raises:
            ValueError: If bounds are invalid or num_samples is non-positive.

        Examples:
            >>> sampler = SobolSampler(scramble=True, seed=42)
            >>> bounds = torch.tensor([[-1.0, 1.0], [-1.0, 1.0]], dtype=torch.float32)
            >>> samples = sampler.sample(bounds, num_samples=100)
            >>> samples.shape
            torch.Size([100, 2])
        """
        bounds = self._validate_bounds(bounds)

        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}")

        n_dims = bounds.shape[0]

        if n_dims > 21201:  # Maximum dimension for scipy's Sobol
            raise ValueError(
                f"Sobol sequence supports up to 21201 dimensions, got {n_dims}"
            )

        # Generate Sobol sequence
        if SCIPY_AVAILABLE:
            samples_unit = self._generate_sobol_scipy(n_dims, num_samples)
        else:
            samples_unit = self._generate_sobol_fallback(n_dims, num_samples)

        # Convert to tensor and scale to bounds
        samples_unit_tensor = self._to_tensor(samples_unit)
        samples = self._scale_samples(samples_unit_tensor, bounds)

        # Validate samples
        self._validate_samples(samples, bounds)

        return samples

    def _generate_sobol_scipy(self, n_dims: int, num_samples: int) -> np.ndarray:
        """Generate Sobol sequence using scipy.

        Args:
            n_dims: Number of dimensions.
            num_samples: Number of samples to generate.

        Returns:
            Array of shape (num_samples, n_dims) with values in [0, 1].
        """
        # Create Sobol engine
        sobol_engine = qmc.Sobol(d=n_dims, scramble=self.scramble, seed=self.seed)

        # Skip initial samples if requested
        if self.skip > 0:
            sobol_engine.fast_forward(self.skip)

        # Generate samples
        samples = sobol_engine.random(n=num_samples)

        return samples.astype(np.float32)

    def _generate_sobol_fallback(self, n_dims: int, num_samples: int) -> np.ndarray:
        """Fallback Sobol generator when scipy is not available.

        This is a basic implementation and may not match scipy's quality.
        For production use, install scipy.

        Args:
            n_dims: Number of dimensions.
            num_samples: Number of samples to generate.

        Returns:
            Array of shape (num_samples, n_dims) with values in [0, 1].
        """
        logger.log_warning(
            "Using fallback Sobol generator. Results may not match scipy quality."
        )

        # Simple fallback: use stratified random sampling
        # This is NOT a true Sobol sequence but provides reasonable coverage
        samples = np.zeros((num_samples, n_dims), dtype=np.float32)

        for dim in range(n_dims):
            # Divide [0, 1] into num_samples intervals
            intervals = np.linspace(0, 1, num_samples + 1)
            # Sample randomly within each interval
            samples[:, dim] = intervals[:-1] + self.rng.rand(num_samples) / num_samples

        # Shuffle to reduce correlation
        for dim in range(n_dims):
            self.rng.shuffle(samples[:, dim])

        return samples

    def get_strategy_name(self) -> str:
        """Get the name of the sampling strategy.

        Returns:
            String identifier for the sampling strategy.
        """
        return SamplingStrategy.SOBOL.value

    def __repr__(self) -> str:
        """String representation of the sampler."""
        return (
            f"{self.__class__.__name__}("
            f"strategy={self.get_strategy_name()}, "
            f"scramble={self.scramble}, "
            f"skip={self.skip}, "
            f"seed={self.seed})"
        )


__all__ = ["SobolSampler"]
