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

"""Closest-hit intersection for independently transformed mesh instances."""

from __future__ import annotations

import warp as wp


@wp.kernel(enable_backward=False)
def cast_mesh(
    mesh: wp.uint64,
    origins: wp.array2d(dtype=wp.vec3),
    directions: wp.array2d(dtype=wp.vec3),
    poses: wp.array3d(dtype=wp.float32),
    max_distance: float,
    shape_id: int,
    distance: wp.array2d(dtype=wp.float32),
    shape: wp.array2d(dtype=wp.int32),
):
    env, ray = wp.tid()
    direction = directions[env, ray]
    length = wp.length(direction)
    if length == 0.0:
        return
    direction = direction / length
    row = env
    if poses.shape[0] == 1:
        row = 0
    r = wp.mat33(
        poses[row, 0, 0],
        poses[row, 0, 1],
        poses[row, 0, 2],
        poses[row, 1, 0],
        poses[row, 1, 1],
        poses[row, 1, 2],
        poses[row, 2, 0],
        poses[row, 2, 1],
        poses[row, 2, 2],
    )
    origin = origins[env, ray] - wp.vec3(
        poses[row, 0, 3], poses[row, 1, 3], poses[row, 2, 3]
    )
    inverse = wp.transpose(r)
    cutoff = wp.min(distance[env, ray], max_distance)
    hit = wp.mesh_query_ray(mesh, inverse * origin, inverse * direction, cutoff)
    if hit.result and hit.t <= max_distance:
        if shape[env, ray] < 0 or hit.t < distance[env, ray]:
            distance[env, ray] = hit.t
            shape[env, ray] = shape_id
