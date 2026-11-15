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

"""Sparse geometry-pair net contact forces for tactile observations."""

from __future__ import annotations

import warp as wp

__all__: list[str] = []


@wp.kernel
def _reduce_geometry_netforce(
    count: wp.array(dtype=wp.int32),
    shape0: wp.array(dtype=wp.int32),
    shape1: wp.array(dtype=wp.int32),
    forces: wp.array(dtype=wp.spatial_vector),
    target_slots: wp.array(dtype=wp.int32),
    object_rows: wp.array(dtype=wp.int32),
    targets_per_env: int,
    output: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    if i >= count[0]:
        return
    a, b = shape0[i], shape1[i]
    if a < 0 or b < 0 or a >= target_slots.shape[0] or b >= target_slots.shape[0]:
        return
    first, second = target_slots[a], target_slots[b]
    force = wp.spatial_top(forces[i])
    if first >= 0 and object_rows[b] == first // targets_per_env:
        wp.atomic_add(output, first, force)
    elif second >= 0 and object_rows[a] == second // targets_per_env:
        wp.atomic_add(output, second, -force)
