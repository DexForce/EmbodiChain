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
from dataclasses import dataclass, field
from typing import Optional, Callable


class SamplingStrategy(Enum):
    """Sampling strategy for joint space"""

    UNIFORM = "uniform"  # Uniform grid sampling
    RANDOM = "random"  # Random sampling


@dataclass
class sampling_config:
    """Configuration for sampling strategies in workspace analysis."""

    strategy: SamplingStrategy = SamplingStrategy.UNIFORM
    num_samples: int = 1000  # Number of samples to generate
    grid_resolution: int = 10  # Resolution for grid sampling
    importance_weight_func: Optional[
        Callable
    ] = None  # Weight function for importance sampling
