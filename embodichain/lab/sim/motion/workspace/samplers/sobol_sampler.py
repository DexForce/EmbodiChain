# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
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

from __future__ import annotations

import numpy as np
import torch

from .base_sampler import BaseSampler
from ..configs.sampling_config import SamplingStrategy

__all__ = ["SobolSampler"]


class SobolSampler(BaseSampler):
    """Scrambled Sobol sampling with a persistent stream for each dimension.

    Successive draws continue the sequence. Power-of-two total sample counts
    preserve Sobol's balance properties. Samples are drawn on CPU by Torch's
    Sobol engine, then transferred as one tensor to the requested device.
    """

    def __init__(
        self,
        seed: int = 42,
        device: torch.device | None = None,
        scramble: bool = True,
        skip: int = 0,
    ) -> None:
        """Initialize the sequence.

        Args:
            seed: Scrambling seed.
            device: Destination device for samples.
            scramble: Whether to scramble the Sobol sequence.
            skip: Number of leading sequence points to skip in each dimension.
        """
        super().__init__(seed, device)
        if skip < 0:
            raise ValueError("skip must be non-negative")
        self.scramble = scramble
        self.skip = skip
        self._engines: dict[int, torch.quasirandom.SobolEngine] = {}

    def _sample_from_bounds(
        self, bounds: torch.Tensor | np.ndarray, num_samples: int
    ) -> torch.Tensor:
        bounds = self._validate_bounds(bounds)
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        dimension = bounds.shape[0]
        if dimension not in self._engines:
            engine = torch.quasirandom.SobolEngine(
                dimension, scramble=self.scramble, seed=self.seed
            )
            if self.skip:
                engine.fast_forward(self.skip)
            self._engines[dimension] = engine
        unit = self._engines[dimension].draw(num_samples).to(self.device)
        return self._scale_samples(unit, bounds)

    def get_strategy_name(self) -> str:
        """Return the registered strategy name."""
        return SamplingStrategy.SOBOL.value
