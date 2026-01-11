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

from embodichain.lab.sim.utility.workspace_analyzer.configs.sampling_config import (
    SamplingStrategy,
)

from embodichain.lab.sim.utility.workspace_analyzer.samplers.base_sampler import (
    BaseSampler,
)

__all__ = ["GaussianSampler"]


class GaussianSampler(BaseSampler):
    """Gaussian (Normal distribution) sampler.

    This sampler generates samples from a multivariate Gaussian (normal) distribution,
    which is useful for local workspace exploration around a specific point or for
    generating samples with controlled spread.

    Advantages:
        - Natural for local exploration
        - Concentrates samples near center
        - Smooth density falloff
        - Good for sensitivity analysis
        - Matches many physical phenomena

    Disadvantages:
        - May not cover entire space well
        - Can generate samples outside bounds (clipped)
        - Not uniform - biased toward center
        - Parameters require tuning

    Attributes:
        mean: Center point of the distribution (n_dims,). If None, uses bounds center.
        std: Standard deviation(s). Can be scalar (isotropic) or per-dimension (n_dims,).
        clip_to_bounds: Whether to clip samples to bounds. If False, rejects out-of-bounds.
    """

    def __init__(
        self,
        seed: int = 42,
        device: torch.device | None = None,
        mean: torch.Tensor | np.ndarray | None = None,
        std: Union[float, torch.Tensor, np.ndarray] = 0.3,
        clip_to_bounds: bool = True,
    ):
        """Initialize the Gaussian sampler.

        Args:
            seed: Random seed for reproducibility. Defaults to 42.
            device: PyTorch device (cpu/cuda). Defaults to cpu.
            mean: Mean of the distribution (center point). If None, uses midpoint of bounds.
                 Can be Tensor or ndarray of shape (n_dims,).
            std: Standard deviation(s). Defaults to 0.3.
                - Scalar: Isotropic (same std for all dimensions)
                - Tensor/array (n_dims,): Per-dimension std
            clip_to_bounds: Whether to clip samples to bounds. Defaults to True.
                           If False, regenerates out-of-bounds samples (slower but unbiased).

        """
        super().__init__(seed, device)
        self.mean = self._to_tensor(mean) if mean is not None else None

        if isinstance(std, (int, float)):
            self.std = float(std)
        else:
            self.std = self._to_tensor(std)

        self.clip_to_bounds = clip_to_bounds

    def _sample_from_bounds(
        self, bounds: torch.Tensor | np.ndarray, num_samples: int
    ) -> torch.Tensor:
        """Generate Gaussian-distributed samples within the given bounds.

        Args:
            bounds: Tensor/Array of shape (n_dims, 2) containing [lower, upper] bounds.
            num_samples: Number of samples to generate.

        Returns:
            Tensor of shape (num_samples, n_dims) containing the sampled points.

        Raises:
            ValueError: If bounds are invalid or num_samples is non-positive.

        Examples:
            >>> # Sample around center with std=0.2
            >>> sampler = GaussianSampler(seed=42, std=0.2, clip_to_bounds=True)
            >>> bounds = torch.tensor([[-1.0, 1.0], [-1.0, 1.0]], dtype=torch.float32)
            >>> samples = sampler.sample(bounds, num_samples=100)
            >>>
            >>> # Sample around custom point
            >>> mean = torch.tensor([0.5, 0.5], dtype=torch.float32)
            >>> sampler = GaussianSampler(mean=mean, std=0.15)
            >>> samples = sampler.sample(bounds, num_samples=100)
        """
        bounds = self._validate_bounds(bounds)

        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}")

        n_dims = bounds.shape[0]

        # Determine mean (center of distribution)
        if self.mean is None:
            # Use center of bounds
            mean = (bounds[:, 0] + bounds[:, 1]) / 2.0
        else:
            mean = self.mean
            if mean.shape[0] != n_dims:
                raise ValueError(
                    f"Mean dimension ({mean.shape[0]}) doesn't match bounds ({n_dims})"
                )

        # Determine std
        if isinstance(self.std, float):
            std = torch.full((n_dims,), self.std, device=self.device)
        else:
            std = self.std
            if std.shape[0] != n_dims:
                raise ValueError(
                    f"Std dimension ({std.shape[0]}) doesn't match bounds ({n_dims})"
                )

        if self.clip_to_bounds:
            # Generate samples and clip
            samples = self._generate_and_clip(mean, std, bounds, num_samples)
        else:
            # Rejection sampling to avoid clipping bias
            samples = self._generate_with_rejection(mean, std, bounds, num_samples)

        # Validate samples
        self._validate_samples(samples, bounds)

        return samples

    def _generate_and_clip(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        bounds: torch.Tensor,
        num_samples: int,
    ) -> torch.Tensor:
        """Generate Gaussian samples and clip to bounds.

        Args:
            mean: Mean vector (n_dims,).
            std: Standard deviation vector (n_dims,).
            bounds: Bounds tensor (n_dims, 2).
            num_samples: Number of samples to generate.

        Returns:
            Clipped samples (num_samples, n_dims).
        """
        # Generate Gaussian samples
        samples = torch.randn(num_samples, mean.shape[0], device=self.device)
        samples = mean + samples * std

        # Clip to bounds
        lower_bounds = bounds[:, 0]
        upper_bounds = bounds[:, 1]
        samples = torch.clamp(samples, lower_bounds, upper_bounds)

        return samples

    def _generate_with_rejection(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        bounds: torch.Tensor,
        num_samples: int,
    ) -> torch.Tensor:
        """Generate Gaussian samples with rejection sampling for out-of-bounds.

        This avoids the bias introduced by clipping but is slower.

        Args:
            mean: Mean vector (n_dims,).
            std: Standard deviation vector (n_dims,).
            bounds: Bounds tensor (n_dims, 2).
            num_samples: Number of samples to generate.

        Returns:
            Valid samples within bounds (num_samples, n_dims).
        """
        lower_bounds = bounds[:, 0]
        upper_bounds = bounds[:, 1]

        accepted_samples = []
        max_iterations = 100
        iteration = 0

        while len(accepted_samples) < num_samples and iteration < max_iterations:
            # Generate more samples than needed to reduce iterations
            num_needed = num_samples - len(accepted_samples)
            num_generate = max(num_needed * 2, 100)  # Generate 2x to reduce rejections

            # Generate Gaussian samples
            samples = torch.randn(num_generate, mean.shape[0], device=self.device)
            samples = mean + samples * std

            # Check which samples are within bounds
            within_bounds = torch.all(
                (samples >= lower_bounds) & (samples <= upper_bounds), dim=1
            )

            valid_samples = samples[within_bounds]
            accepted_samples.append(valid_samples)

            iteration += 1

        if (
            len(accepted_samples) == 0
            or sum(s.shape[0] for s in accepted_samples) < num_samples
        ):
            raise RuntimeError(
                f"Failed to generate {num_samples} samples within bounds after {max_iterations} iterations. "
                "Consider using clip_to_bounds=True or adjusting mean/std parameters."
            )

        # Concatenate and trim to exact size
        all_samples = torch.cat(accepted_samples, dim=0)
        return all_samples[:num_samples]

    def get_strategy_name(self) -> str:
        """Get the name of the sampling strategy.

        Returns:
            String identifier for the sampling strategy.
        """
        return SamplingStrategy.GAUSSIAN.value

    def __repr__(self) -> str:
        """String representation of the sampler."""
        mean_str = "auto" if self.mean is None else f"shape={self.mean.shape}"
        std_str = (
            f"{self.std}" if isinstance(self.std, float) else f"shape={self.std.shape}"
        )
        return (
            f"{self.__class__.__name__}("
            f"strategy={self.get_strategy_name()}, "
            f"mean={mean_str}, "
            f"std={std_str}, "
            f"clip={self.clip_to_bounds}, "
            f"seed={self.seed})"
        )
