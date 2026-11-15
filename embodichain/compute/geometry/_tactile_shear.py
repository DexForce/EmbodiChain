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


"""Pose-driven load and elastic shear on a device-resident tactile grid."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import warp as wp

from embodichain.utils.device_utils import current_warp_stream

from ._warp.indentation import _rotation, _translation


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
            travel = wp.transform_point(pad_now, points[probe]) - wp.transform_point(
                previous[shape, env], points[probe]
            )
            local = wp.transpose(_rotation(pad_now)) * travel
            deformation = displacement[env, row, col] - wp.vec2(
                wp.dot(local, tangents[probe]), wp.dot(local, bitangents[probe])
            )
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


class _TactileShear:
    """Track per-cell elastic displacement and report force on the pad."""

    def __init__(
        self,
        points: torch.Tensor,
        normals: torch.Tensor,
        depth: torch.Tensor,
        shape_ids: torch.Tensor,
        normal_force: torch.Tensor,
        shear_force: torch.Tensor,
        displacement: torch.Tensor,
        *,
        num_meshes: int,
        normal_stiffness: float,
        shear_stiffness: float,
        friction: float,
    ) -> None:
        self._device = depth.device
        self._shape = tuple(depth.shape)
        u = torch.tensor([1.0, 0.0, 0.0], device=self._device)
        v = torch.tensor([0.0, 1.0, 0.0], device=self._device)
        reference = torch.where(normals[:, :1].abs() > 0.95, v, u)
        tangent = reference - (reference * normals).sum(dim=-1, keepdim=True) * normals
        tangent = torch.nn.functional.normalize(tangent, dim=-1)
        bitangent = torch.linalg.cross(normals, tangent)
        self._current = wp.zeros(
            (num_meshes, depth.shape[0]), dtype=wp.mat44, device=str(self._device)
        )
        self._previous = wp.empty_like(self._current)
        self._last_shape = torch.full_like(shape_ids, -1)
        self._inputs = [
            wp.from_torch(points, dtype=wp.vec3),
            wp.from_torch(tangent, dtype=wp.vec3),
            wp.from_torch(bitangent, dtype=wp.vec3),
            wp.from_torch(depth),
            wp.from_torch(shape_ids),
            self._current,
            self._previous,
            normal_stiffness,
            shear_stiffness,
            friction,
        ]
        self._outputs = [
            wp.from_torch(self._last_shape),
            wp.from_torch(normal_force),
            wp.from_torch(shear_force, dtype=wp.vec2),
            wp.from_torch(displacement, dtype=wp.vec2),
        ]

    def reset(self, rows: slice | torch.Tensor) -> None:
        self._last_shape[rows] = -1

    def sample(self, mount: torch.Tensor, poses: Sequence[torch.Tensor]) -> None:
        device = str(self._device)
        stream = current_warp_stream(self._device)
        mount_array = wp.from_torch(mount, dtype=wp.mat44)
        for shape_id, pose in enumerate(poses):
            wp.launch(
                _relative_pad_pose,
                dim=self._shape[0],
                inputs=[mount_array, wp.from_torch(pose, dtype=wp.mat44), shape_id],
                outputs=[self._current, self._previous],
                device=device,
                stream=stream,
            )
        wp.launch(
            _update_load,
            dim=self._shape,
            inputs=self._inputs,
            outputs=self._outputs,
            device=device,
            stream=stream,
        )
