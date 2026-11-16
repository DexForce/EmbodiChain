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


"""Warp kernels for pose-driven tactile load and shear."""

from __future__ import annotations

import warp as wp

from .indentation import _rotation, _translation


@wp.kernel(enable_backward=False)
def _relative_pad_pose(
    mount: wp.array(dtype=wp.mat44),
    target: wp.array(dtype=wp.mat44),
    shape_id: int,
    current: wp.array2d(dtype=wp.mat44),
    previous: wp.array2d(dtype=wp.mat44),
):
    env = wp.tid()
    target_pose = target[wp.where(target.shape[0] == 1, 0, env)]
    mount_pose = mount[wp.where(mount.shape[0] == 1, 0, env)]
    inverse_rotation = wp.transpose(_rotation(target_pose))
    rotation = inverse_rotation * _rotation(mount_pose)
    position = inverse_rotation * (_translation(mount_pose) - _translation(target_pose))
    previous[shape_id, env] = current[shape_id, env]
    current[shape_id, env] = wp.mat44(
        rotation[0, 0],
        rotation[0, 1],
        rotation[0, 2],
        position[0],
        rotation[1, 0],
        rotation[1, 1],
        rotation[1, 2],
        position[1],
        rotation[2, 0],
        rotation[2, 1],
        rotation[2, 2],
        position[2],
        0.0,
        0.0,
        0.0,
        1.0,
    )


@wp.kernel(enable_backward=False)
def _update_load(
    points: wp.array(dtype=wp.vec3),
    tangents: wp.array(dtype=wp.vec3),
    bitangents: wp.array(dtype=wp.vec3),
    depth: wp.array3d(dtype=wp.float32),
    shape_ids: wp.array3d(dtype=wp.int32),
    current: wp.array2d(dtype=wp.mat44),
    previous: wp.array2d(dtype=wp.mat44),
    normal_stiffness: float,
    shear_stiffness: float,
    friction: float,
    last_shape: wp.array3d(dtype=wp.int32),
    normal_force: wp.array3d(dtype=wp.float32),
    shear_force: wp.array3d(dtype=wp.vec2),
    displacement: wp.array3d(dtype=wp.vec2),
):
    env, row, col = wp.tid()
    shape = shape_ids[env, row, col]
    normal = normal_stiffness * depth[env, row, col]
    deformation = wp.vec2(0.0)
    if shape >= 0 and normal > 0.0:
        if last_shape[env, row, col] == shape:
            probe = row * depth.shape[2] + col
            pad_now = current[shape, env]
            pad_before = previous[shape, env]
            tangent = tangents[probe]
            bitangent = bitangents[probe]
            history = displacement[env, row, col]
            stored_local = tangent * history[0] + bitangent * history[1]
            stored_target = _rotation(pad_before) * stored_local
            travel = wp.transform_point(pad_now, points[probe]) - wp.transform_point(
                pad_before, points[probe]
            )
            # Transport history through the shared target frame before adding travel.
            local = wp.transpose(_rotation(pad_now)) * (stored_target - travel)
            deformation = wp.vec2(wp.dot(local, tangent), wp.dot(local, bitangent))
            magnitude = wp.length(deformation)
            slip_limit = friction * normal / shear_stiffness
            if magnitude > slip_limit:
                deformation = deformation * (slip_limit / magnitude)
        last_shape[env, row, col] = shape
    else:
        last_shape[env, row, col] = -1
    normal_force[env, row, col] = normal
    shear_force[env, row, col] = shear_stiffness * deformation
    displacement[env, row, col] = deformation
