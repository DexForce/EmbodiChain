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

from dataclasses import dataclass, field
from typing import Optional, Callable, TYPE_CHECKING

# Import SamplingStrategy from samplers to avoid duplication
if TYPE_CHECKING:
    from ..samplers.base_sampler import SamplingStrategy
else:
    try:
        from ..samplers.base_sampler import SamplingStrategy
    except ImportError:
        from enum import Enum

        class SamplingStrategy(Enum):
            """Fallback SamplingStrategy if samplers not available."""

            UNIFORM = "uniform"
            RANDOM = "random"
            HALTON = "halton"
            SOBOL = "sobol"
            LATIN_HYPERCUBE = "lhs"
            IMPORTANCE = "importance"
            GAUSSIAN = "gaussian"


@dataclass
class SamplingConfig:
    """Configuration for sampling strategies in workspace analysis."""

    strategy: "SamplingStrategy" = (
        None  # Will be set to UNIFORM by default in __post_init__
    )
    num_samples: int = 1000
    """Number of samples to generate."""

    grid_resolution: int = 10
    """Resolution for grid sampling (used with UNIFORM strategy)."""

    batch_size: int = 1000
    """Number of samples to process in each batch."""

    seed: int = 42
    """Random seed for reproducibility."""

    importance_weight_func: Optional[Callable] = None
    """Weight function for importance sampling (used with IMPORTANCE strategy)."""

    gaussian_mean: Optional[float] = None
    """Mean for Gaussian sampling (used with GAUSSIAN strategy). If None, uses center of bounds."""

    gaussian_std: Optional[float] = None
    """Standard deviation for Gaussian sampling (used with GAUSSIAN strategy). If None, uses 1/6 of range."""

    def __post_init__(self):
        """Set default strategy after initialization."""
        if self.strategy is None:
            self.strategy = SamplingStrategy.UNIFORM
