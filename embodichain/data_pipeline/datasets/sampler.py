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

import random
from abc import ABC, abstractmethod
from typing import Callable, Iterator, List, Optional, Union

__all__ = [
    "ChunkSizeSampler",
    "UniformChunkSampler",
    "GMMChunkSampler",
]


class ChunkSizeSampler(ABC):
    """Abstract base class for chunk-size samplers.

    Subclasses implement :meth:`__call__` to return an integer chunk size on
    demand.  A sampler is called once per :meth:`OnlineDataset.__iter__` step,
    so consecutive samples / batches may have different time dimensions.

    When used in **batch mode** the same chunk size is drawn once and applied
    to every trajectory in the batch so that the resulting TensorDict has a
    consistent shape ``[batch_size, chunk_size]``.
    """

    @abstractmethod
    def __call__(self) -> int:
        """Return the next chunk size (positive integer).

        Returns:
            A positive integer representing the number of timesteps to include
            in the next trajectory chunk.
        """
        ...


class UniformChunkSampler(ChunkSizeSampler):
    """Discrete-uniform chunk-size sampler over ``[low, high]``.

    Draws an integer uniformly at random from the closed interval
    ``[low, high]`` on every call.

    Args:
        low: Minimum chunk size (inclusive, must be ≥ 1).
        high: Maximum chunk size (inclusive, must be ≥ ``low``).

    Raises:
        ValueError: If ``low < 1`` or ``high < low``.

    Example::

        sampler = UniformChunkSampler(low=16, high=64)
        chunk_size = sampler()  # e.g. 37
    """

    def __init__(self, low: int, high: int) -> None:
        if low < 1:
            raise ValueError(f"low must be ≥ 1, got {low}.")
        if high < low:
            raise ValueError(f"high must be ≥ low ({low}), got {high}.")
        self._low = low
        self._high = high

    def __call__(self) -> int:
        return random.randint(self._low, self._high)

    def __repr__(self) -> str:
        return f"UniformChunkSampler(low={self._low}, high={self._high})"


class GMMChunkSampler(ChunkSizeSampler):
    """Gaussian Mixture Model chunk-size sampler.

    Selects a mixture component according to ``weights``, samples a value from
    the corresponding ``Normal(mean, std)`` distribution, rounds to the nearest
    integer, and optionally clamps the result to ``[low, high]``.

    Args:
        means: Mean of each Gaussian component (number of elements = K).
        stds: Standard deviation of each component (must be > 0, same length
            as ``means``).
        weights: Unnormalised mixture weights (same length as ``means``).
            Defaults to a uniform distribution over all components.
        low: Optional lower bound for clamping the sampled value (inclusive,
            must be ≥ 1 if provided).
        high: Optional upper bound for clamping the sampled value (inclusive,
            must be ≥ ``low`` if both are provided).

    Raises:
        ValueError: If ``means``, ``stds``, or ``weights`` have mismatched
            lengths, if any ``std ≤ 0``, or if the bounds are inconsistent.

    Example — two-component mixture favouring short and long chunks::

        sampler = GMMChunkSampler(
            means=[16.0, 64.0],
            stds=[4.0, 8.0],
            weights=[0.6, 0.4],
            low=8,
            high=96,
        )
        chunk_size = sampler()  # e.g. 18
    """

    def __init__(
        self,
        means: List[float],
        stds: List[float],
        weights: Optional[List[float]] = None,
        low: Optional[int] = None,
        high: Optional[int] = None,
    ) -> None:
        if len(means) == 0:
            raise ValueError("means must not be empty.")
        if len(stds) != len(means):
            raise ValueError(
                f"stds length ({len(stds)}) must match means length ({len(means)})."
            )
        if any(s <= 0 for s in stds):
            raise ValueError("All stds must be > 0.")
        if weights is not None:
            if len(weights) != len(means):
                raise ValueError(
                    f"weights length ({len(weights)}) must match means length ({len(means)})."
                )
            if any(w < 0 for w in weights):
                raise ValueError("All weights must be ≥ 0.")
            total = sum(weights)
            if total <= 0:
                raise ValueError("Sum of weights must be > 0.")
            self._weights = [w / total for w in weights]
        else:
            k = len(means)
            self._weights = [1.0 / k] * k

        if low is not None and low < 1:
            raise ValueError(f"low must be ≥ 1, got {low}.")
        if low is not None and high is not None and high < low:
            raise ValueError(f"high must be ≥ low ({low}), got {high}.")

        self._means = means
        self._stds = stds
        self._low = low
        self._high = high
        # Precompute cumulative weights for component selection.
        self._cumulative = []
        acc = 0.0
        for w in self._weights:
            acc += w
            self._cumulative.append(acc)

    def __call__(self) -> int:
        # Select component via inverse CDF on the cumulative weight table.
        u = random.random()
        component = len(self._cumulative) - 1
        for i, cdf in enumerate(self._cumulative):
            if u <= cdf:
                component = i
                break

        # Sample from the selected Gaussian using Box-Muller.
        value = random.gauss(self._means[component], self._stds[component])

        # Round to nearest integer, ensuring at least 1.
        chunk = max(1, round(value))

        # Clamp to [low, high] if bounds are specified.
        if self._low is not None:
            chunk = max(self._low, chunk)
        if self._high is not None:
            chunk = min(self._high, chunk)

        return chunk

    def __repr__(self) -> str:
        return (
            f"GMMChunkSampler(means={self._means}, stds={self._stds}, "
            f"weights={self._weights}, low={self._low}, high={self._high})"
        )
