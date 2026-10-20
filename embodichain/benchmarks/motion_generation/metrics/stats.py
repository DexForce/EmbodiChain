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
from collections.abc import Iterable

__all__ = ["nearest_rank_percentile"]


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
