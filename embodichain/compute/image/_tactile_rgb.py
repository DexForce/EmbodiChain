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


"""Device-resident RGB response from a planar tactile depth image."""

from __future__ import annotations

import math

import torch
import warp as wp

from embodichain.utils.device_utils import current_warp_stream


@wp.func
def _depth_at_hit(
    depth: wp.array3d(dtype=wp.float32),
    hit: wp.array3d(dtype=wp.bool),
    env: int,
    row: int,
    col: int,
) -> float:
    if hit[env, row, col]:
        return depth[env, row, col]
    return 0.0


@wp.kernel(enable_backward=False)
def _shade(
    depth: wp.array3d(dtype=wp.float32),
    hit: wp.array3d(dtype=wp.bool),
    background: wp.array2d(dtype=wp.vec3),
    x_response: wp.vec3,
    y_response: wp.vec3,
    inv_dx: float,
    inv_dy: float,
    rgb: wp.array3d(dtype=wp.vec3),
):
    env, row, col = wp.tid()
    base = background[row, col]
    if not hit[env, row, col]:
        rgb[env, row, col] = base
        return
    left = wp.max(0, col - 1)
    right = wp.min(depth.shape[2] - 1, col + 1)
    front = wp.max(0, row - 1)
    back = wp.min(depth.shape[1] - 1, row + 1)
    dx = (
        _depth_at_hit(depth, hit, env, row, right)
        - _depth_at_hit(depth, hit, env, row, left)
    ) * (inv_dx / float(wp.max(1, right - left)))
    dy = (
        _depth_at_hit(depth, hit, env, back, col)
        - _depth_at_hit(depth, hit, env, front, col)
    ) * (inv_dy / float(wp.max(1, back - front)))
    color = base + dx * x_response + dy * y_response
    rgb[env, row, col] = wp.vec3(
        wp.clamp(color[0], 0.0, 1.0),
        wp.clamp(color[1], 0.0, 1.0),
        wp.clamp(color[2], 0.0, 1.0),
    )


def _render_tactile_rgb(
    depth: torch.Tensor,
    hit: torch.Tensor,
    background: torch.Tensor,
    *,
    spacing: tuple[float, float],
    x_response: tuple[float, float, float],
    y_response: tuple[float, float, float],
    rgb: torch.Tensor,
) -> None:
    """Write RGB in-place using a first-order calibrated slope response.

    Depth and hit have shape [env, row, col]. Spacing is (dx, dy) in meters,
    with positive x along columns and positive y along rows. The background
    has shape [row, col, 3] and is shared by the batch. Response triples
    give the change in normalized RGB per unit slope. Misses preserve the
    background and contribute zero depth to adjacent slopes. All buffers
    are contiguous, on one device, and images use float32 with RGB in [0, 1].
    """
    if depth.ndim != 3 or hit.shape != depth.shape or min(depth.shape[1:]) <= 0:
        raise ValueError("Expected depth and hit with shape [env, row, col].")
    if background.shape != (*depth.shape[1:], 3) or rgb.shape != (*depth.shape, 3):
        raise ValueError("Background and output must match the depth layout.")
    if any(
        not tensor.is_contiguous() or tensor.device != depth.device
        for tensor in (depth, hit, background, rgb)
    ):
        raise ValueError("Input and output must be contiguous on one device.")
    if hit.dtype != torch.bool or any(
        tensor.dtype != torch.float32 for tensor in (depth, background, rgb)
    ):
        raise ValueError("Expected a boolean hit mask and float32 image tensors.")
    dx, dy = spacing
    if not all(math.isfinite(step) and step > 0 for step in spacing):
        raise ValueError("Pixel spacing must be finite and positive.")
    if depth.shape[0] == 0:
        return
    wp.init()
    wp.launch(
        _shade,
        dim=tuple(depth.shape),
        inputs=[
            wp.from_torch(depth),
            wp.from_torch(hit),
            wp.from_torch(background, dtype=wp.vec3),
            wp.vec3(*x_response),
            wp.vec3(*y_response),
            1.0 / dx,
            1.0 / dy,
        ],
        outputs=[wp.from_torch(rgb, dtype=wp.vec3)],
        device=str(depth.device),
        stream=current_warp_stream(depth.device),
    )
