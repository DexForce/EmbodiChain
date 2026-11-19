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


"""Device-resident calibrated RGB response from a tactile depth grid."""

from __future__ import annotations

import torch
import warp as wp

from embodichain.utils.device_utils import current_warp_stream

from ._warp.tactile_rgb import _shade


class _TactileRGB:
    """Bind grid storage once and shade into its caller-owned RGB buffer."""

    def __init__(
        self,
        depth: torch.Tensor,
        hit: torch.Tensor,
        rgb: torch.Tensor,
        background: torch.Tensor,
        points: torch.Tensor,
        x_response: tuple[float, float, float],
        y_response: tuple[float, float, float],
    ) -> None:
        _, height, width = depth.shape
        grid = points.reshape(height, width, 3)
        inverse_span = torch.zeros((height, width, 2), device=depth.device)
        for axis, size, channel in ((1, width, 0), (0, height, 1)):
            if size > 1:
                idx = torch.arange(size, device=depth.device)
                left = grid.index_select(axis, (idx - 1).clamp_min(0))
                right = grid.index_select(axis, (idx + 1).clamp_max(size - 1))
                distance = (right - left).norm(dim=-1)
                if not bool((distance > 0).all()):
                    raise ValueError(
                        "RGB calibration requires distinct slope neighbors."
                    )
                inverse_span[..., channel] = distance.reciprocal()

        wp.init()
        self._device = depth.device
        self._shape = tuple(depth.shape)
        self._inputs = [
            wp.from_torch(depth),
            wp.from_torch(hit),
            wp.from_torch(background, dtype=wp.vec3),
            wp.vec3(*x_response),
            wp.vec3(*y_response),
            wp.from_torch(inverse_span, dtype=wp.vec2),
        ]
        self._output = wp.from_torch(rgb, dtype=wp.vec3)

    def render(self) -> None:
        wp.launch(
            _shade,
            dim=self._shape,
            inputs=self._inputs,
            outputs=[self._output],
            device=str(self._device),
            stream=current_warp_stream(self._device),
        )
