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


"""Ray intervals on closed, outward-oriented collider meshes."""

from __future__ import annotations

import warp as wp


@wp.func
def _rotation(pose: wp.mat44) -> wp.mat33:
    return wp.mat33(
        pose[0, 0],
        pose[0, 1],
        pose[0, 2],
        pose[1, 0],
        pose[1, 1],
        pose[1, 2],
        pose[2, 0],
        pose[2, 1],
        pose[2, 2],
    )


@wp.func
def _translation(pose: wp.mat44) -> wp.vec3:
    return wp.vec3(pose[0, 3], pose[1, 3], pose[2, 3])


@wp.kernel(enable_backward=False)
def select_pad_candidates(
    mounts: wp.array(dtype=wp.mat44),
    poses: wp.array(dtype=wp.mat44),
    pad_center: wp.vec3,
    pad_radius: float,
    mesh_min: wp.vec3,
    mesh_max: wp.vec3,
    candidates: wp.array(dtype=wp.bool),
):
    env = wp.tid()
    pose = poses[wp.where(poses.shape[0] == 1, 0, env)]
    mount = mounts[wp.where(mounts.shape[0] == 1, 0, env)]
    center = wp.transpose(_rotation(pose)) * (
        _rotation(mount) * pad_center + _translation(mount) - _translation(pose)
    )
    closest = wp.vec3(
        wp.clamp(center[0], mesh_min[0], mesh_max[0]),
        wp.clamp(center[1], mesh_min[1], mesh_max[1]),
        wp.clamp(center[2], mesh_min[2], mesh_max[2]),
    )
    candidates[env] = wp.length_sq(closest - center) <= pad_radius * pad_radius


@wp.kernel(enable_backward=False)
def cast_indentation(
    mesh: wp.uint64,
    mounts: wp.array(dtype=wp.mat44),
    poses: wp.array(dtype=wp.mat44),
    points: wp.array(dtype=wp.vec3),
    normals: wp.array(dtype=wp.vec3),
    candidates: wp.array(dtype=wp.bool),
    use_candidates: bool,
    search_distance: float,
    depth_limit: float,
    shape_id: int,
    depth: wp.array2d(dtype=wp.float32),
    shape: wp.array2d(dtype=wp.int32),
):
    env, probe = wp.tid()
    if use_candidates and not candidates[env]:
        return
    pose = poses[wp.where(poses.shape[0] == 1, 0, env)]
    mount = mounts[wp.where(mounts.shape[0] == 1, 0, env)]
    inv = wp.transpose(_rotation(pose))
    rotation = inv * _rotation(mount)
    point = rotation * points[probe] + inv * (_translation(mount) - _translation(pose))
    normal = rotation * normals[probe]
    outer = wp.mesh_query_ray(mesh, point, normal, search_distance)
    if not outer.result or outer.sign >= 0.0 or outer.t <= 0.0:
        return
    # Test the reverse ray on the same mesh. An outside point in a cavity
    # cannot combine an exit from one shape with an entry from another.
    inner = wp.mesh_query_ray(mesh, point, -normal, search_distance)
    if not inner.result or inner.sign >= 0.0 or inner.t <= 0.0:
        return
    value = wp.min(outer.t, depth_limit)
    if value > depth[env, probe]:
        depth[env, probe] = value
        shape[env, probe] = shape_id
