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


"""Accumulate measured contact impulses on a planar sensor grid."""

from __future__ import annotations

import warp as wp


@wp.kernel(enable_backward=False)
def scatter_grid(
    positions: wp.array2d(dtype=wp.vec3),
    normals: wp.array2d(dtype=wp.vec3),
    friction: wp.array2d(dtype=wp.vec3),
    impulse: wp.array2d(dtype=wp.float32),
    ids: wp.array2d(dtype=wp.vec2i),
    populated: wp.array2d(dtype=wp.bool),
    impulse_valid: wp.array2d(dtype=wp.bool),
    friction_valid: wp.array2d(dtype=wp.bool),
    actors: wp.array(dtype=wp.int32),
    mounts: wp.array(dtype=wp.mat44),
    size_x: float,
    size_y: float,
    slab: float,
    width: int,
    height: int,
    use_normal: bool,
    counts: wp.array3d(dtype=wp.int32),
    measured: wp.array4d(dtype=wp.int32),
    magnitude_out: wp.array3d(dtype=wp.float32),
    normal_out: wp.array3d(dtype=wp.vec3),
    friction_out: wp.array3d(dtype=wp.vec3),
):
    env, contact = wp.tid()
    if not populated[env, contact]:
        return
    pair = ids[env, contact]
    first = pair[0] == actors[env]
    second = pair[1] == actors[env]
    if first == second:
        return
    transform = mounts[env]
    rotation = wp.mat33(
        transform[0, 0],
        transform[0, 1],
        transform[0, 2],
        transform[1, 0],
        transform[1, 1],
        transform[1, 2],
        transform[2, 0],
        transform[2, 1],
        transform[2, 2],
    )
    inverse = wp.transpose(rotation)
    center = wp.vec3(transform[0, 3], transform[1, 3], transform[2, 3])
    local = inverse * (positions[env, contact] - center)
    nx = local[0] / size_x + 0.5
    ny = local[1] / size_y + 0.5
    if not (
        nx >= 0.0 and nx < 1.0 and ny >= 0.0 and ny < 1.0 and wp.abs(local[2]) <= slab
    ):
        return
    x = wp.min(int(nx * float(width)), width - 1)
    y = wp.min(int(ny * float(height)), height - 1)
    wp.atomic_add(counts, env, y, x, 1)
    sign = wp.where(first, -1.0, 1.0)
    value = impulse[env, contact]
    if impulse_valid[env, contact] and wp.isfinite(value):
        wp.atomic_add(measured, env, y, x, 0, 1)
        wp.atomic_add(magnitude_out, env, y, x, value)
        n = normals[env, contact]
        if use_normal and wp.isfinite(n[0]) and wp.isfinite(n[1]) and wp.isfinite(n[2]):
            wp.atomic_add(measured, env, y, x, 1, 1)
            wp.atomic_add(normal_out, env, y, x, inverse * n * (value * sign))
    tangent = friction[env, contact]
    if (
        friction_valid[env, contact]
        and wp.isfinite(tangent[0])
        and wp.isfinite(tangent[1])
        and wp.isfinite(tangent[2])
    ):
        wp.atomic_add(measured, env, y, x, 2, 1)
        wp.atomic_add(friction_out, env, y, x, inverse * tangent * -sign)
