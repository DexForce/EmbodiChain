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
from typing import List, Union, TYPE_CHECKING

from embodichain.lab.sim.utility.workspace_analyzer.configs.sampling_config import (
    SamplingStrategy,
)
from embodichain.lab.sim.utility.workspace_analyzer.samplers.base_sampler import (
    BaseSampler,
)

__all__ = ["HaltonSampler"]


class HaltonSampler(BaseSampler):
    """Halton sequence sampler using quasi-random low-discrepancy sequences.

    The Halton sequence is a deterministic low-discrepancy sequence that provides
    better coverage than random sampling. It uses coprime bases (primes) for each
    dimension to generate well-distributed points.

    Advantages:
        - Better uniformity than random sampling
        - Deterministic (reproducible)
        - Fast convergence for Monte Carlo integration
        - Works well in low to medium dimensions (2-10)

    Disadvantages:
        - Performance degrades in high dimensions (>10)
        - Correlation between dimensions can appear
        - Sequential generation (not easily parallelizable)

    Attributes:
        bases: Prime bases for each dimension. If None, uses first n primes.
        skip: Number of initial samples to skip (helps reduce correlation).
    """

    # First 100 prime numbers for Halton bases
    _PRIMES = [
        2,
        3,
        5,
        7,
        11,
        13,
        17,
        19,
        23,
        29,
        31,
        37,
        41,
        43,
        47,
        53,
        59,
        61,
        67,
        71,
        73,
        79,
        83,
        89,
        97,
        101,
        103,
        107,
        109,
        113,
        127,
        131,
        137,
        139,
        149,
        151,
        157,
        163,
        167,
        173,
        179,
        181,
        191,
        193,
        197,
        199,
        211,
        223,
        227,
        229,
        233,
        239,
        241,
        251,
        257,
        263,
        269,
        271,
        277,
        281,
        283,
        293,
        307,
        311,
        313,
        317,
        331,
        337,
        347,
        349,
        353,
        359,
        367,
        373,
        379,
        383,
        389,
        397,
        401,
        409,
        419,
        421,
        431,
        433,
        439,
        443,
        449,
        457,
        461,
        463,
        467,
        479,
        487,
        491,
        499,
        503,
        509,
        521,
        523,
        541,
    ]

    def __init__(
        self,
        seed: int = 42,
        device: torch.device | None = None,
        bases: List[int] | None = None,
        skip: int = 0,
    ):
        """Initialize the Halton sampler.

        Args:
            seed: Random seed (used for consistency, but Halton is deterministic).
            device: PyTorch device (cpu/cuda). Defaults to cpu.
            bases: List of prime bases for each dimension. If None, uses first n primes.
            skip: Number of initial samples to skip. Defaults to 0.
                  Higher values (e.g., 100-1000) can improve distribution quality.

        """
        super().__init__(seed, device)
        self.bases = bases
        self.skip = skip

    def sample(
        self, bounds: Union[torch.Tensor, np.ndarray], num_samples: int
    ) -> torch.Tensor:
        """Generate Halton sequence samples within the given bounds.

        Args:
            bounds: Tensor/Array of shape (n_dims, 2) containing [lower, upper] bounds.
            num_samples: Number of samples to generate.

        Returns:
            Tensor of shape (num_samples, n_dims) containing the sampled points.

        Raises:
            ValueError: If bounds are invalid or num_samples is non-positive.

        Examples:
            >>> sampler = HaltonSampler(skip=100)
            >>> bounds = torch.tensor([[-1.0, 1.0], [-1.0, 1.0]], dtype=torch.float32)
            >>> samples = sampler.sample(bounds, num_samples=100)
            >>> samples.shape
            torch.Size([100, 2])
        """
        bounds = self._validate_bounds(bounds)

        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}")

        n_dims = bounds.shape[0]

        # Get bases for each dimension
        if self.bases is None:
            if n_dims > len(self._PRIMES):
                raise ValueError(
                    f"Number of dimensions ({n_dims}) exceeds available primes ({len(self._PRIMES)}). "
                    "Please provide custom bases."
                )
            bases = self._PRIMES[:n_dims]
        else:
            if len(self.bases) < n_dims:
                raise ValueError(
                    f"Number of bases ({len(self.bases)}) is less than dimensions ({n_dims})"
                )
            bases = self.bases[:n_dims]

        # Generate Halton sequence
        samples_unit = np.zeros((num_samples, n_dims), dtype=np.float32)

        for dim in range(n_dims):
            base = bases[dim]
            for i in range(num_samples):
                index = i + self.skip + 1  # Start from 1, apply skip
                samples_unit[i, dim] = self._halton_number(index, base)

        # Convert to tensor and scale to bounds
        samples_unit_tensor = self._to_tensor(samples_unit)
        samples = self._scale_samples(samples_unit_tensor, bounds)

        # Validate samples
        self._validate_samples(samples, bounds)

        return samples

    @staticmethod
    def _halton_number(index: int, base: int) -> float:
        """Compute a single Halton number.

        The Halton sequence is generated by reversing the base-n representation
        of the index.

        Args:
            index: Sequence index (starting from 1).
            base: Prime base for this dimension.

        Returns:
            Halton number in [0, 1].
        """
        result = 0.0
        f = 1.0 / base
        i = index

        while i > 0:
            result += f * (i % base)
            i //= base
            f /= base

        return result

    def get_strategy_name(self) -> str:
        """Get the name of the sampling strategy.

        Returns:
            String identifier for the sampling strategy.
        """
        return SamplingStrategy.HALTON.value

    def __repr__(self) -> str:
        """String representation of the sampler."""
        return (
            f"{self.__class__.__name__}("
            f"strategy={self.get_strategy_name()}, "
            f"skip={self.skip}, "
            f"seed={self.seed})"
        )
