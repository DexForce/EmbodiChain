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

"""Shared numeric helpers for benchmark metric aggregation."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Iterable

__all__ = ["nearest_rank_percentile"]


def _seed_cluster_interval(
    values: list[tuple[int, float]], confidence_level: float
) -> tuple[float | None, float | None, int]:
    """Bootstrap case rates by seed, keeping environment/repeat rows together."""
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between zero and one.")
    groups: dict[int, list[float]] = defaultdict(list)
    for seed, rate in values:
        groups[seed].append(rate)
    clusters = [(sum(group), len(group)) for _, group in sorted(groups.items())]
    if len(clusters) < 2:
        return None, None, len(clusters)
    generator = random.Random(0)
    estimates = []
    for _ in range(1000):
        sample = generator.choices(clusters, k=len(clusters))
        estimates.append(
            sum(total for total, _ in sample) / sum(count for _, count in sample)
        )
    tail = 100.0 * (1.0 - confidence_level) / 2.0
    return (
        nearest_rank_percentile(estimates, tail),
        nearest_rank_percentile(estimates, 100.0 - tail),
        len(clusters),
    )


def nearest_rank_percentile(
    values: Iterable[float | None], percentile: float
) -> float | None:
    """Return a nearest-rank percentile over finite values, or ``None`` if empty."""
    finite = sorted(
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    )
    if not finite:
        return None
    index = max(
        0, min(len(finite) - 1, math.ceil(percentile / 100.0 * len(finite)) - 1)
    )
    return finite[index]
