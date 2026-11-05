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

"""Numerical reduction of isolated rigid-contact queries, without host objects."""

from __future__ import annotations

import warp as wp

__all__ = []


@wp.kernel(enable_backward=False)
def _contact_summary(
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
    shapes: int,
    required: float,
    tolerance: wp.array(dtype=float),
    contact_capacity: int,
    broad_count: wp.array(dtype=wp.int32),
    broad_capacity: int,
    gjk_count: wp.array(dtype=wp.int32),
    gjk_capacity: int,
    split_count: wp.array(dtype=wp.int32),
    split_capacity: int,
    manifold_count: wp.array(dtype=wp.int32),
    manifold_capacity: int,
    mesh_count: wp.array(dtype=wp.int32),
    mesh_capacity: int,
    triangle_count: wp.array(dtype=wp.int32),
    triangle_capacity: int,
    mesh_mesh_count: wp.array(dtype=wp.int32),
    mesh_mesh_capacity: int,
    reduction_failures: wp.array(dtype=wp.int32),
    pair_distances: wp.array(dtype=float),
    output: wp.array(dtype=float),
):
    index = wp.tid()
    if index == 0:
        output[2] = float(count[0])
        if (
            count[0] >= contact_capacity
            or broad_count[0] > broad_capacity
            or gjk_count[0] > gjk_capacity
            or split_count[0] > split_capacity
            or manifold_count[0] > manifold_capacity
            or mesh_count[0] > mesh_capacity
            or triangle_count[0] > triangle_capacity
            or mesh_mesh_count[0] > mesh_mesh_capacity
            or reduction_failures[0] > 0
        ):
            wp.atomic_min(output, 3, 0.0)
    if index < count[0] and index < contact_capacity:
        a, b = first[index], second[index]
        if a < 0 or b < 0 or a >= shapes or b >= shapes:
            wp.atomic_min(output, 3, 0.0)
        elif active[a * shapes + b] != 0:
            p0 = wp.vec3(point0[index][0], point0[index][1], point0[index][2])
            p1 = wp.vec3(point1[index][0], point1[index][1], point1[index][2])
            owner0, owner1 = shape_body[a], shape_body[b]
            if owner0 >= 0:
                p0 = wp.transform_point(poses[owner0], p0)
            if owner1 >= 0:
                p1 = wp.transform_point(poses[owner1], p1)
            distance = wp.dot(normal[index], p1 - p0) - margin0[index] - margin1[index]
            if not wp.isfinite(distance):
                wp.atomic_min(output, 3, 0.0)
            else:
                wp.atomic_min(output, 0, distance)
                if pair_distances.shape[0] > 0:
                    wp.atomic_min(
                        pair_distances, wp.min(a, b) * shapes + wp.max(a, b), distance
                    )
                if distance <= required + tolerance[0]:
                    wp.atomic_min(
                        output, 1, float(wp.min(a, b) * shapes + wp.max(a, b))
                    )
