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


"""Inspect every measured native contact without materializing Python pairs."""

from __future__ import annotations

import warp as wp

__all__: list[str] = []


@wp.func
def actor_index(actors: wp.array(dtype=wp.int32), value: int):
    lower = int(0)
    upper = actors.shape[0]
    while lower < upper:
        middle = (lower + upper) // 2
        if actors[middle] < value:
            lower = middle + 1
        else:
            upper = middle
    return lower


@wp.kernel(enable_backward=False)
def classify_contacts(
    pairs: wp.array3d(dtype=wp.int32),
    valid: wp.array2d(dtype=wp.bool),
    impulse_valid: wp.array2d(dtype=wp.bool),
    actors: wp.array(dtype=wp.int32),
    memberships: wp.array2d(dtype=wp.bool),
    codes: wp.array(dtype=wp.int64),
    moving: wp.array(dtype=wp.bool),
    allowed: wp.array2d(dtype=wp.bool),
    dropped: wp.array(dtype=wp.bool),
    first_error: wp.array(dtype=wp.int32),
    any_drop: wp.array(dtype=wp.int32),
):
    row, slot = wp.tid()
    if slot == 0 and dropped[row]:
        wp.atomic_max(any_drop, 0, 1)
    if valid[row, slot]:
        reason = int(0)
        if not impulse_valid[row, slot]:
            reason = 1
        else:
            left, right = pairs[row, slot, 0], pairs[row, slot, 1]
            a, b = actor_index(actors, left), actor_index(actors, right)
            if a == actors.shape[0] or b == actors.shape[0]:
                reason = 2
            elif actors[a] != left or actors[b] != right:
                reason = 2
            elif not memberships[a, row] or not memberships[b, row]:
                reason = 2
            else:
                x, y = codes[a], codes[b]
                if (moving[x] or moving[y]) and not allowed[x, y]:
                    reason = 3
        if reason != 0:
            wp.atomic_min(first_error, row, slot * 8 + reason)


@wp.kernel(enable_backward=False)
def contact_failures(
    pairs: wp.array3d(dtype=wp.int32),
    first_error: wp.array(dtype=wp.int32),
    any_drop: wp.array(dtype=wp.int32),
    output: wp.array2d(dtype=wp.int32),
):
    row = wp.tid()
    if any_drop[0] != 0:
        output[row, 0] = 4
        output[row, 1] = 0
        output[row, 2] = 0
    elif first_error[row] < pairs.shape[1] * 8:
        slot = first_error[row] // 8
        output[row, 0] = first_error[row] % 8
        output[row, 1] = pairs[row, slot, 0]
        output[row, 2] = pairs[row, slot, 1]
    else:
        output[row, 0] = 0
        output[row, 1] = 0
        output[row, 2] = 0
