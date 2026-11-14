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


"""Per-environment reduction of one independent batched Newton query."""

from __future__ import annotations

import warp as wp

__all__: list[str] = []


@wp.kernel(enable_backward=False)
def contact_summary(
    count: wp.array(dtype=wp.int32),
    first: wp.array(dtype=wp.int32),
    second: wp.array(dtype=wp.int32),
    point0: wp.array(dtype=wp.vec3),
    point1: wp.array(dtype=wp.vec3),
    normal: wp.array(dtype=wp.vec3),
    margin0: wp.array(dtype=float),
    margin1: wp.array(dtype=float),
    shape_body: wp.array(dtype=wp.int32),
    poses: wp.array(dtype=wp.transform),
    active: wp.array(dtype=wp.int32),
    width: int,
    batches: int,
    required: float,
    tolerance: wp.array(dtype=float),
    capacity: int,
    broad_count: wp.array(dtype=wp.int32),
    broad_capacity: int,
    gjk_count: wp.array(dtype=wp.int32),
    gjk_capacity: int,
    split_count: wp.array(dtype=wp.int32),
    split_capacity: int,
    manifold_count: wp.array(dtype=wp.int32),
    manifold_capacity: int,
    output: wp.array2d(dtype=float),
):
    index = wp.tid()
    if index == 0:
        if (
            count[0] >= capacity
            or broad_count[0] > broad_capacity
            or gjk_count[0] > gjk_capacity
            or split_count[0] > split_capacity
            or manifold_count[0] > manifold_capacity
        ):
            for row in range(batches):
                wp.atomic_min(output, row, 2, 0.0)
    if index < count[0] and index < capacity:
        a, b = first[index], second[index]
        if a < 0 or b < 0 or a >= width * batches or b >= width * batches:
            for row in range(batches):
                wp.atomic_min(output, row, 2, 0.0)
        elif a // width != b // width:
            for row in range(batches):
                wp.atomic_min(output, row, 2, 0.0)
        else:
            row = a // width
            x, y = a % width, b % width
            if active[x * width + y] != 0:
                p0 = wp.transform_point(poses[shape_body[a]], point0[index])
                p1 = wp.transform_point(poses[shape_body[b]], point1[index])
                distance = (
                    wp.dot(normal[index], p1 - p0) - margin0[index] - margin1[index]
                )
                if not wp.isfinite(distance):
                    wp.atomic_min(output, row, 2, 0.0)
                else:
                    wp.atomic_min(output, row, 0, distance)
                    if distance <= required + tolerance[row]:
                        wp.atomic_min(
                            output, row, 1, float(wp.min(x, y) * width + wp.max(x, y))
                        )
