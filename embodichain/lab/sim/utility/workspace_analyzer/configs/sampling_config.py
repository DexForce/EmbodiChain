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
from enum import Enum
from dataclasses import dataclass
from typing import Callable


class SamplingStrategy(Enum):
    """Sampling strategy for joint space."""

    UNIFORM = "uniform"  # Uniform grid sampling
    RANDOM = "random"  # Random sampling
    HALTON = "halton"  # Quasi-random Halton sequence
    SOBOL = "sobol"  # Quasi-random Sobol sequence
    LATIN_HYPERCUBE = "lhs"  # Latin Hypercube Sampling
    IMPORTANCE = "importance"  # Importance sampling (requires weight function)
    GAUSSIAN = "gaussian"  # Gaussian (normal) distribution sampling
    SPHERE = "sphere"  # Sphere-constrained uniform sampling


@dataclass
class SamplingConfig:
    """Configuration for sampling strategies in workspace analysis."""

    strategy: SamplingStrategy = None
    """Sampling strategy to use."""

    num_samples: int = 1000
    """Number of samples to generate."""

    grid_resolution: int = 10
    """Resolution for grid sampling (used with UNIFORM strategy)."""

    batch_size: int = 1000
    """Number of samples to process in each batch."""

    seed: int = 42
    """Random seed for reproducibility."""

    importance_weight_func: Callable | None = None
    """Weight function for importance sampling (used with IMPORTANCE strategy)."""

    gaussian_mean: float | None = None
    """Mean for Gaussian sampling (used with GAUSSIAN strategy). If None, uses center of bounds."""

    gaussian_std: float | None = None
    """Standard deviation for Gaussian sampling (used with GAUSSIAN strategy). If None, uses 1/6 of range."""

    # Sphere sampling parameters
    sphere_center_mode: str = "bounds_center"
    """How to determine sphere center for SPHERE strategy. Options: 'bounds_center', 'custom', 'auto'."""

    sphere_radius_mode: str = "inscribed"
    """How to determine sphere radius for SPHERE strategy. Options: 'inscribed', 'circumscribed', 'custom'."""

    sphere_boundary_handling: str = "reject"
    """How to handle boundary violations for SPHERE strategy. Options: 'clip', 'reject', 'extend'."""

    sphere_center: list | None = None
    """Custom sphere center for SPHERE strategy (used when sphere_center_mode='custom')."""

    sphere_radius: float | None = None
    """Custom sphere radius for SPHERE strategy (used when sphere_radius_mode='custom')."""

    def __post_init__(self):
        """Set default strategy after initialization."""
        if self.strategy is None:
            self.strategy = SamplingStrategy.UNIFORM
