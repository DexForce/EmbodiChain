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
from typing import Callable, Union, TYPE_CHECKING

from embodichain.lab.sim.utility.workspace_analyzer.configs.sampling_config import (
    SamplingStrategy,
)
from embodichain.lab.sim.utility.workspace_analyzer.samplers.base_sampler import (
    BaseSampler,
)

from embodichain.utils import logger

__all__ = ["ImportanceSampler"]


class ImportanceSampler(BaseSampler):
    """Importance sampler using weighted distribution.

    Importance sampling generates samples according to a user-defined importance
    (weight) function, concentrating samples in regions of interest. This is
    particularly useful for workspace analysis where certain regions (e.g., task space,
    reachable space) are more important than others.

    Advantages:
        - Focuses samples on important regions
        - More efficient for targeted analysis
        - Reduces variance in Monte Carlo estimation
        - Flexible - user defines importance function

    Disadvantages:
        - Requires domain knowledge for weight function
        - May miss unexpected important regions
        - Can be computationally expensive if weight function is complex
        - Quality depends heavily on weight function design

    Attributes:
        weight_fn: Callable that takes positions (N, n_dims) and returns weights (N,).
                   Higher weights = higher probability of sampling.
        num_candidates: Number of candidate samples to generate for rejection sampling.
                       Higher values = better approximation but slower.
        method: Sampling method ('rejection' or 'transform').
    """

    def __init__(
        self,
        weight_fn: Callable[[torch.Tensor], torch.Tensor],
        seed: int = 42,
        device: torch.device | None = None,
        num_candidates: int = 10,
        method: str = "rejection",
    ):
        """Initialize the importance sampler.

        Args:
            weight_fn: Function that computes importance weights for positions.
                      Signature: weight_fn(positions: Tensor[N, n_dims]) -> Tensor[N]
                      Returns non-negative weights (higher = more important).
            seed: Random seed for reproducibility. Defaults to 42.
            device: PyTorch device (cpu/cuda). Defaults to cpu.
            num_candidates: Number of candidates per desired sample. Defaults to 10.
                          Higher values improve quality but increase computation.
            method: Sampling method. Defaults to 'rejection'.
                   'rejection': Rejection sampling (simple, unbiased)
                   'transform': Inverse transform sampling (requires normalized weights)
            constraint: Optional geometric constraint for sampling (e.g., SphereConstraint).
        """
        super().__init__(seed, device)
        self.weight_fn = weight_fn
        self.num_candidates = num_candidates
        self.method = method

        if method not in ["rejection", "transform"]:
            raise ValueError(
                f"Invalid method '{method}'. Use 'rejection' or 'transform'."
            )

    def sample(
        self, bounds: torch.Tensor | np.ndarray, num_samples: int
    ) -> torch.Tensor:
        """Generate importance-weighted samples within the given bounds.

        Args:
            bounds: Tensor/Array of shape (n_dims, 2) containing [lower, upper] bounds.
            num_samples: Number of samples to generate.

        Returns:
            Tensor of shape (num_samples, n_dims) containing the sampled points.

        Raises:
            ValueError: If bounds are invalid or num_samples is non-positive.

        Examples:
            >>> # Define weight function (favor center)
            >>> def center_weight(positions):
            ...     distances = torch.sqrt(torch.sum(positions**2, dim=1))
            ...     return torch.exp(-distances)  # Higher weight near origin
            >>>
            >>> sampler = ImportanceSampler(weight_fn=center_weight, num_candidates=20)
            >>> bounds = torch.tensor([[-1.0, 1.0], [-1.0, 1.0]], dtype=torch.float32)
            >>> samples = sampler.sample(bounds, num_samples=100)
        """
        bounds = self._validate_bounds(bounds)

        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}")

        if self.method == "rejection":
            samples = self._rejection_sampling(bounds, num_samples)
        else:  # transform
            samples = self._transform_sampling(bounds, num_samples)

        # Validate samples
        self._validate_samples(samples, bounds)

        return samples

    def _rejection_sampling(
        self, bounds: torch.Tensor, num_samples: int
    ) -> torch.Tensor:
        """Rejection sampling using importance weights.

        Algorithm:
        1. Generate candidate samples uniformly
        2. Compute importance weights for all candidates
        3. Accept/reject based on weights
        4. Repeat until enough samples

        Args:
            bounds: Tensor of shape (n_dims, 2) containing [lower, upper] bounds.
            num_samples: Number of samples to generate.

        Returns:
            Tensor of shape (num_samples, n_dims) containing accepted samples.
        """
        n_dims = bounds.shape[0]
        accepted_samples = []

        while len(accepted_samples) < num_samples:
            # Generate candidate samples
            num_needed = num_samples - len(accepted_samples)
            num_candidates_batch = max(num_needed * self.num_candidates, 100)

            candidates = torch.rand(num_candidates_batch, n_dims, device=self.device)
            candidates = self._scale_samples(candidates, bounds)

            # Compute weights
            weights = self.weight_fn(candidates)

            if torch.any(weights < 0):
                logger.log_warning(
                    "Weight function returned negative values. Using absolute values."
                )
                weights = torch.abs(weights)

            # Normalize weights to [0, 1] for rejection
            if weights.max() > 0:
                weights_normalized = weights / weights.max()
            else:
                logger.log_warning(
                    "All weights are zero. Falling back to uniform sampling."
                )
                weights_normalized = torch.ones_like(weights)

            # Rejection sampling
            accept_probs = torch.rand(num_candidates_batch, device=self.device)
            accepted_mask = accept_probs < weights_normalized

            accepted_batch = candidates[accepted_mask]
            accepted_samples.append(accepted_batch)

            # Safety check to avoid infinite loop
            if (
                len(accepted_samples) > 1000
                and sum(s.shape[0] for s in accepted_samples) < num_samples * 0.1
            ):
                logger.log_warning(
                    "Rejection sampling is very inefficient. Consider adjusting weight function "
                    "or increasing num_candidates."
                )

        # Concatenate and trim to exact size
        all_samples = torch.cat(accepted_samples, dim=0)
        return all_samples[:num_samples]

    def _transform_sampling(
        self, bounds: torch.Tensor, num_samples: int
    ) -> torch.Tensor:
        """Inverse transform sampling using importance weights.

        This method generates candidates and then selects based on cumulative weights.
        More stable than rejection sampling but requires more memory.

        Args:
            bounds: Tensor of shape (n_dims, 2) containing [lower, upper] bounds.
            num_samples: Number of samples to generate.

        Returns:
            Tensor of shape (num_samples, n_dims) containing sampled points.
        """
        n_dims = bounds.shape[0]

        # Generate candidate samples
        num_candidates_total = num_samples * self.num_candidates
        candidates = torch.rand(num_candidates_total, n_dims, device=self.device)
        candidates = self._scale_samples(candidates, bounds)

        # Compute weights
        weights = self.weight_fn(candidates)

        if torch.any(weights < 0):
            logger.log_warning(
                "Weight function returned negative values. Using absolute values."
            )
            weights = torch.abs(weights)

        # Normalize to probability distribution
        if weights.sum() > 0:
            probabilities = weights / weights.sum()
        else:
            logger.log_warning(
                "All weights are zero. Falling back to uniform sampling."
            )
            probabilities = (
                torch.ones(num_candidates_total, device=self.device)
                / num_candidates_total
            )

        # Sample indices according to probabilities
        try:
            indices = torch.multinomial(probabilities, num_samples, replacement=False)
        except RuntimeError:
            # If probabilities are problematic, add small epsilon
            probabilities = probabilities + 1e-10
            probabilities = probabilities / probabilities.sum()
            indices = torch.multinomial(probabilities, num_samples, replacement=False)

        selected_samples = candidates[indices]

        return selected_samples

    def get_strategy_name(self) -> str:
        """Get the name of the sampling strategy.

        Returns:
            String identifier for the sampling strategy.
        """
        return SamplingStrategy.IMPORTANCE.value

    def __repr__(self) -> str:
        """String representation of the sampler."""
        return (
            f"{self.__class__.__name__}("
            f"strategy={self.get_strategy_name()}, "
            f"method={self.method}, "
            f"num_candidates={self.num_candidates}, "
            f"seed={self.seed})"
        )
