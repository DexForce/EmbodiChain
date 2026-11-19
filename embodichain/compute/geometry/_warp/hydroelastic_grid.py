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


"""Rasterize completed hydroelastic triangles onto a local tactile grid."""

from __future__ import annotations

import warp as wp


@wp.kernel(enable_backward=False)
def scatter_faces(
    face_count: wp.array(dtype=wp.int32),
    triangles: wp.array2d(dtype=wp.vec3),
    shape_pairs: wp.array(dtype=wp.vec2i),
    separation: wp.array(dtype=wp.float32),
    body_poses: wp.array(dtype=wp.transform),
    pad_env: wp.array(dtype=wp.int32),
    target_env: wp.array(dtype=wp.int32),
    body_ids: wp.array(dtype=wp.int32),
    static_inverse: wp.array(dtype=wp.mat44),
    mount_inverse: wp.mat44,
    points: wp.array(dtype=wp.vec3),
    normals: wp.array(dtype=wp.vec3),
    regular: bool,
    size_x: float,
    size_y: float,
    width: int,
    height: int,
    max_distance: float,
    counts: wp.array2d(dtype=wp.int32),
    closest: wp.array2d(dtype=wp.float32),
):
    face = wp.tid()
    if face >= face_count[0]:
        return
    pair = shape_pairs[face]
    a = pair[0]
    b = pair[1]
    if a < 0 or a >= pad_env.shape[0] or b < 0 or b >= pad_env.shape[0]:
        return
    env = pad_env[a]
    if env < 0 or target_env[b] != env:
        env = pad_env[b]
        if env < 0 or target_env[a] != env:
            return
    center = (triangles[face, 0] + triangles[face, 1] + triangles[face, 2]) / 3.0
    body = body_ids[env]
    if body >= 0:
        center = wp.transform_point(wp.transform_inverse(body_poses[body]), center)
    else:
        center = wp.transform_point(static_inverse[env], center)
    local = wp.transform_point(mount_inverse, center)
    probe = int(-1)
    if regular:
        x = local[0] / size_x + 0.5
        y = local[1] / size_y + 0.5
        if x < 0.0 or x >= 1.0 or y < 0.0 or y >= 1.0:
            return
        probe = int(y * float(height)) * width + int(x * float(width))
        if wp.abs(wp.dot(local - points[probe], normals[probe])) > max_distance:
            return
    else:
        nearest = max_distance * max_distance
        for cell in range(points.shape[0]):
            distance_sq = wp.length_sq(local - points[cell])
            if distance_sq < nearest:
                nearest = distance_sq
                probe = cell
        if probe < 0:
            return
    wp.atomic_add(counts, env, probe, 1)
    wp.atomic_min(closest, env, probe, separation[face])


@wp.kernel(enable_backward=False)
def finish_raster(
    face_count: wp.array(dtype=wp.int32),
    capacity: int,
    counts: wp.array2d(dtype=wp.int32),
    closest: wp.array2d(dtype=wp.float32),
    complete: wp.array(dtype=wp.bool),
):
    env, probe = wp.tid()
    if counts[env, probe] == 0:
        closest[env, probe] = 0.0
    if probe == 0:
        complete[env] = face_count[0] <= capacity
