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

"""Yaw-aligned height rays over reference collision terrain."""

from __future__ import annotations

import numpy as np
import newton
import torch
import warp as wp

__all__ = ["ReferenceHeightScanner"]


@wp.kernel(enable_backward=False)
def _prepare_rays(
    pose: wp.array2d(dtype=wp.float32),
    pattern: wp.array(dtype=wp.vec3),
    origins: wp.array(dtype=wp.vec3),
    directions: wp.array(dtype=wp.vec3),
):
    env, ray = wp.tid()
    x, y, z, w = pose[env, 3], pose[env, 4], pose[env, 5], pose[env, 6]
    yaw = wp.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    rotation = wp.quat(0.0, 0.0, wp.sin(yaw * 0.5), wp.cos(yaw * 0.5))
    point = pattern[ray] + wp.vec3(0.0, 0.0, 20.0)
    index = env * pattern.shape[0] + ray
    origins[index] = wp.quat_rotate(rotation, point) + wp.vec3(
        pose[env, 0], pose[env, 1], pose[env, 2]
    )
    directions[index] = wp.vec3(0.0, 0.0, -1.0)


@wp.kernel(enable_backward=False)
def _resolve_heights(
    pose: wp.array2d(dtype=wp.float32),
    origins: wp.array(dtype=wp.vec3),
    distances: wp.array(dtype=wp.float32),
    height: wp.array2d(dtype=wp.float32),
):
    env, ray = wp.tid()
    index = env * height.shape[1] + ray
    distance = distances[index]
    value = wp.float32(wp.inf)
    if distance >= 0.0 and distance <= 1000000.0:
        value = pose[env, 2] - (origins[index][2] - distance) - 0.5
    height[env, ray] = value


@wp.kernel
def _scan(
    mesh: wp.uint64,
    pose: wp.array2d(dtype=wp.float32),
    pattern: wp.array(dtype=wp.vec3),
    height: wp.array2d(dtype=wp.float32),
):
    env, ray = wp.tid()
    x, y, z, w = pose[env, 3], pose[env, 4], pose[env, 5], pose[env, 6]
    yaw = wp.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    p = pattern[ray]
    start = wp.vec3(
        pose[env, 0] + wp.cos(yaw) * p[0] - wp.sin(yaw) * p[1],
        pose[env, 1] + wp.sin(yaw) * p[0] + wp.cos(yaw) * p[1],
        pose[env, 2] + 20.0,
    )
    direction = wp.vec3(0.0, 0.0, -1.0)
    query = wp.mesh_query_ray(mesh, start, direction, 1000000.0)
    value = wp.float32(wp.inf)
    if query.result:
        value = pose[env, 2] - (start[2] - query.t) - 0.5
    height[env, ray] = value


class ReferenceHeightScanner:
    """187 terrain rays with the reference torso frame, XY order and offset."""

    def __init__(
        self,
        vertices: np.ndarray,
        triangles: np.ndarray,
        num_envs: int,
        device: torch.device,
        *,
        physics_model: newton.Model | None = None,
        physics_state: newton.State | None = None,
    ) -> None:
        self.physics_model = physics_model
        self.mesh = (
            None
            if physics_model is not None
            else wp.Mesh(
                points=wp.array(vertices, dtype=wp.vec3, device=str(device)),
                indices=wp.array(
                    triangles.reshape(-1), dtype=wp.int32, device=str(device)
                ),
            )
        )
        x = torch.arange(-0.8, 0.8 + 1.0e-9, 0.1, device=device)
        y = torch.arange(-0.5, 0.5 + 1.0e-9, 0.1, device=device)
        xx, yy = torch.meshgrid(x, y, indexing="xy")
        pattern = torch.stack(
            (xx.flatten(), yy.flatten(), torch.zeros_like(xx.flatten())), -1
        )
        self.pattern = wp.from_torch(pattern, dtype=wp.vec3)
        self.height = torch.empty((num_envs, 187), device=device)
        self._pose = torch.empty((num_envs, 7), device=device)
        self._pose_warp = wp.from_torch(self._pose)
        self._graph = None
        self._scan_stream = wp.Stream(str(device)) if device.type == "cuda" else None
        self._torch_stream = (
            wp.stream_to_torch(self._scan_stream)
            if self._scan_stream is not None
            else None
        )
        if physics_model is not None:
            if physics_state is None:
                raise ValueError("The physics height scanner requires the model state.")
            physics_model.bvh_build_shapes(
                physics_state,
                shape_flags=newton.ShapeFlags.COLLIDE_SHAPES
                | newton.ShapeFlags.VISIBLE,
            )
            count = num_envs * 187
            self._origins = wp.empty(count, dtype=wp.vec3, device=str(device))
            self._directions = wp.empty(count, dtype=wp.vec3, device=str(device))
            self._distances = wp.empty(count, dtype=wp.float32, device=str(device))
            self._worlds = wp.full(count, -1, dtype=wp.int32, device=str(device))

    def compute(self, pose: torch.Tensor) -> torch.Tensor:
        """Return torso height minus terrain hit height minus 0.5 m.

        Physics-backed deployments query the global collision heightfield through
        the same Newton BVH ray interface as the reference sensor.
        """
        if pose.is_cuda:
            caller = torch.cuda.current_stream(pose.device)
            self._torch_stream.wait_stream(caller)
            with (
                wp.ScopedStream(self._scan_stream),
                torch.cuda.stream(self._torch_stream),
            ):
                self._pose.copy_(pose)
                if self._graph is None:
                    self._compute_height()
                    with wp.ScopedCapture(stream=self._scan_stream) as capture:
                        self._compute_height()
                    self._graph = capture.graph
                else:
                    wp.capture_launch(self._graph, stream=self._scan_stream)
            caller.wait_stream(self._torch_stream)
        else:
            self._pose.copy_(pose)
            self._compute_height()
        return self.height

    def _compute_height(self) -> None:
        """Launch the fixed-buffer ray pipeline without changing its geometry."""
        if self.physics_model is not None:
            wp.launch(
                _prepare_rays,
                dim=self.height.shape,
                inputs=[
                    self._pose_warp,
                    self.pattern,
                    self._origins,
                    self._directions,
                ],
                device=str(self._pose.device),
            )
            newton.intersect_ray(
                self.physics_model,
                ray_origins=self._origins,
                ray_directions=self._directions,
                ray_worlds=self._worlds,
                enable_global_world=False,
                out_dist=self._distances,
            )
            wp.launch(
                _resolve_heights,
                dim=self.height.shape,
                inputs=[
                    self._pose_warp,
                    self._origins,
                    self._distances,
                    wp.from_torch(self.height),
                ],
                device=str(self._pose.device),
            )
            return
        wp.launch(
            _scan,
            dim=self.height.shape,
            inputs=[
                self.mesh.id,
                self._pose_warp,
                self.pattern,
                wp.from_torch(self.height),
            ],
            device=str(self._pose.device),
        )
