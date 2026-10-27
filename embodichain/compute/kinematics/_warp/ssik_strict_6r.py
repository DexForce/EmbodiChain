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

"""Isolated strict6R branch generation without SRS search or FK stack frames."""

from __future__ import annotations

import warp as wp

from .ssik_6r import mat73d, mat83d, ssik_three_parallel_limited, vec7d

wp.set_module_options({"enable_backward": False})

__all__: list[str] = []


@wp.kernel
def three_parallel_raw(
    axes: mat73d,
    offsets: mat83d,
    home: wp.mat33d,
    targets: wp.array(dtype=wp.mat44d),
    inverse_tcp: wp.mat44d,
    seed: wp.array2d(dtype=wp.float64),
    lower: wp.array(dtype=wp.float64),
    upper: wp.array(dtype=wp.float64),
    raw_valid: wp.array2d(dtype=wp.int32),
    raw: wp.array3d(dtype=wp.float64),
):
    """Write every raw branch; a separate original-model kernel accepts it."""
    i = wp.tid()
    origin, lo, hi = vec7d(), vec7d(), vec7d()
    finite = bool(True)
    for j in range(6):
        lo[j] = lower[j]
        hi[j] = upper[j]
        value = seed[i, j]
        finite = finite and wp.isfinite(value)
        if not wp.isfinite(value):
            value = (lo[j] + hi[j]) * wp.float64(0.5)
        origin[j] = wp.clamp(value, lo[j], hi[j])
    goal = targets[i] @ inverse_tcp
    rotation = wp.mat33d()
    for j in range(4):
        for k in range(4):
            finite = finite and wp.isfinite(goal[j, k])
    if not finite:
        goal = wp.mat44d(wp.float64(1))
    for j in range(3):
        for k in range(3):
            rotation[j, k] = goal[j, k]
    position = wp.vec3d(goal[0, 3], goal[1, 3], goal[2, 3])
    branches, flags = ssik_three_parallel_limited(
        axes, offsets, position, rotation @ wp.transpose(home), origin, lo, hi
    )
    for branch in range(8):
        raw_valid[i, branch] = wp.int32(finite and flags[branch] != 0)
        for j in range(6):
            raw[i, branch, j] = branches[branch, j]
