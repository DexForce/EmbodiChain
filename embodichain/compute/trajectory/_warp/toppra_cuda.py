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

"""Cooperative float64 CUDA inference; one 64-lane block owns each path.

The serial recorded solver remains responsible for autograd. Bounds and
feasibility tolerances match it; reductions only parallelize plane scans.
"""

from __future__ import annotations

import warp as wp

from .toppra import _forward_bound, _speed

__all__: list[str] = []

_BLOCK_DIM = 64


@wp.kernel(enable_backward=False)
def _solve(
    a: wp.array3d(dtype=wp.float64),
    b: wp.array3d(dtype=wp.float64),
    c: wp.array3d(dtype=wp.float64),
    intervals: wp.array(dtype=int),
    caps: wp.array2d(dtype=wp.float64),
    controllable: wp.array2d(dtype=wp.float64),
    x: wp.array2d(dtype=wp.float64),
    times: wp.array2d(dtype=wp.float64),
    durations: wp.array(dtype=wp.float64),
    success: wp.array(dtype=int),
):
    # Every CTA owns one complete row. All returns and reductions are
    # uniform across its 64 lanes, including invalid/stationary rows.
    row, lane = wp.tid()
    total = intervals[row]
    if success[row] == 0 or total == 0:
        return
    next_cap = wp.float64(0.0)
    for reverse in range(total):
        i = total - 1 - reverse
        candidate = wp.float64(wp.inf)
        for group in range((a.shape[2] + 63) // 64):
            j = group * 64 + lane
            if j < a.shape[2]:
                aj, bj = a[row, i, j], b[row, i, j]
                if aj > wp.float64(0.0) and bj < wp.float64(0.0):
                    bound = (c[row, i, j] - bj * next_cap) / aj
                    if bound < candidate:
                        candidate = bound
        limit = wp.tile_extract(wp.tile_min(wp.tile(candidate)), 0)
        cap = wp.min(caps[row, i], limit)
        if not wp.isfinite(cap) or cap < wp.float64(0.0):
            if lane == 0:
                success[row] = 0
            return
        if lane == 0:
            controllable[row, i] = cap
        next_cap = cap
    # A cross-warp reduction synchronizes each preceding controllable write.
    # Forward step i reads K[i+1], written before the final backward reduction.
    current, arrival = wp.float64(0.0), wp.float64(0.0)
    for i in range(total):
        candidate = wp.float64(wp.inf)
        for group in range((a.shape[2] + 63) // 64):
            j = group * 64 + lane
            if j < a.shape[2] and b[row, i, j] > wp.float64(0.0):
                bound = _forward_bound(
                    a[row, i, j], b[row, i, j], c[row, i, j], current
                )
                if bound < candidate:
                    candidate = bound
        limit = wp.tile_extract(wp.tile_min(wp.tile(candidate)), 0)
        high = wp.max(wp.float64(0.0), wp.min(controllable[row, i + 1], limit))
        invalid = int(0)
        for group in range((a.shape[2] + 63) // 64):
            j = group * 64 + lane
            if j < a.shape[2]:
                left, right, check_bound = (
                    a[row, i, j] * current,
                    b[row, i, j] * high,
                    c[row, i, j],
                )
                tolerance = wp.float64(1e-10) * wp.max(
                    wp.max(wp.abs(left), wp.abs(right)), wp.abs(check_bound)
                )
                if left + right - check_bound > tolerance:
                    invalid = 1
        failed = wp.tile_extract(wp.tile_max(wp.tile(invalid)), 0)
        denominator = _speed(current) + _speed(high)
        if failed != 0 or denominator <= wp.float64(0.0):
            if lane == 0:
                success[row] = 0
            return
        next_arrival = arrival + wp.float64(2.0) / (wp.float64(total) * denominator)
        if not wp.isfinite(next_arrival) or next_arrival <= arrival:
            if lane == 0:
                success[row] = 0
            return
        if lane == 0:
            x[row, i + 1] = high
            times[row, i + 1] = next_arrival
        current, arrival = high, next_arrival
    if lane == 0:
        durations[row] = arrival
