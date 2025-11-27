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

from .cache_config import CacheConfig
from .dimension_constraint import DimensionConstraint
from .sampling_config import SamplingConfig
from .visualization_config import VisualizationConfig
from .metric_config import (
    MetricConfig,
    MetricType,
    ReachabilityConfig,
    ManipulabilityConfig,
    DensityConfig,
)

# Import SamplingStrategy from samplers module (avoid duplication)
try:
    from ..samplers.base_sampler import SamplingStrategy
except ImportError:
    # Fallback if samplers module is not available yet
    from enum import Enum

    class SamplingStrategy(Enum):
        """Fallback SamplingStrategy."""

        UNIFORM = "uniform"
        RANDOM = "random"
        HALTON = "halton"
        SOBOL = "sobol"
        LATIN_HYPERCUBE = "lhs"
        IMPORTANCE = "importance"
        GAUSSIAN = "gaussian"


__all__ = [
    # Cache
    "CacheConfig",
    # Constraints
    "DimensionConstraint",
    # Sampling
    "SamplingConfig",
    "SamplingStrategy",
    # Visualization
    "VisualizationConfig",
    # Metrics
    "MetricConfig",
    "MetricType",
    "ReachabilityConfig",
    "ManipulabilityConfig",
    "DensityConfig",
]
