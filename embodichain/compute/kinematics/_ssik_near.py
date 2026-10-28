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

"""Borrowed near-parallel candidates and bounded numerical workspace."""

from __future__ import annotations

import numpy as np
import torch
import warp as wp

from ._ssik_model import _SSIKModel
from ._warp.ssik_near import CANDIDATE_COUNT

__all__: list[str] = []


class _NearParallelCandidates:
    """Generate eight geometric slots and one seed-based bounded solve.

    Returned arrays are private borrowed buffers, valid until the next call.
    The host adapter serializes stream use, validates original-model FK at
    returned precision, and copies every public result into its owned output.
    """

    candidate_count = CANDIDATE_COUNT
    max_chunk_size = 512

    def __init__(
        self,
        model: _SSIKModel,
        device: torch.device | str,
        axis_tolerance: float,
    ) -> None:
        from ._warp.ssik_6r import mat73d, mat83d

        if model.family != "near_parallel" or model.dof != 6:
            raise ValueError(
                "Near-parallel candidates require an explicitly certified 6R model"
            )
        wp.init()
        self.device = torch.device(device)
        axes = model.axes.copy()
        for joint in (2, 3):
            axes[joint] = axes[1] * np.sign(axes[joint] @ axes[1])
        self._h = mat73d(axes)
        self._p = mat83d(model.offsets)
        self._home = wp.mat33d(model.home_rotation)
        self._frames = torch.as_tensor(
            model.frames.copy(), dtype=torch.float64, device=self.device
        )
        self._axes = torch.as_tensor(
            model.local_axes.copy(), dtype=torch.float64, device=self.device
        )
        self._axis_tolerance = axis_tolerance
        self._workspace: dict[str, torch.Tensor] = {}
        self._capacity = 0

    def prepare_buffers(self, batch_size: int) -> None:
        from ._warp.ssik_near import WORKSPACE_COLUMNS

        capacity = min(batch_size, self.max_chunk_size)
        if capacity <= self._capacity:
            return
        self._workspace = {
            "work": torch.empty(
                (capacity * self.candidate_count, WORKSPACE_COLUMNS),
                dtype=torch.float64,
                device=self.device,
            ),
            "valid": torch.empty(
                (capacity, self.candidate_count), dtype=torch.int32, device=self.device
            ),
            "raw": torch.empty(
                (capacity, self.candidate_count, 6),
                dtype=torch.float64,
                device=self.device,
            ),
        }
        self._capacity = capacity

    def generate(
        self,
        targets: torch.Tensor,
        seeds: torch.Tensor,
        tcp_translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
        position_tolerance: float = 1e-5,
        rotation_tolerance: float = 1e-5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return borrowed flags and joints for flange goals and projected seeds."""
        from ._warp.ssik_near import generate_near

        count = len(targets)
        if not count:
            return (
                torch.empty(
                    (0, self.candidate_count), dtype=torch.int32, device=self.device
                ),
                torch.empty(
                    (0, self.candidate_count, 6),
                    dtype=torch.float64,
                    device=self.device,
                ),
            )
        if count > self.max_chunk_size:
            raise ValueError(
                "Near-parallel candidate calls require chunks of at most 512 targets"
            )
        self.prepare_buffers(count)
        if self.device.type == "cuda":
            stream = torch.cuda.current_stream(self.device)
            for value in (*self._workspace.values(), self._frames, self._axes):
                value.record_stream(stream)
        valid = self._workspace["valid"][:count]
        raw = self._workspace["raw"][:count]
        if count:
            wp.launch(
                generate_near,
                dim=(count, self.candidate_count),
                inputs=[
                    self._h,
                    self._p,
                    self._home,
                    wp.from_torch(self._frames),
                    wp.from_torch(self._axes),
                    wp.from_torch(targets, dtype=wp.mat44d),
                    wp.from_torch(seeds),
                    self._axis_tolerance,
                    wp.vec3d(tcp_translation),
                    position_tolerance,
                    rotation_tolerance,
                    wp.from_torch(self._workspace["work"]),
                    wp.from_torch(valid),
                    wp.from_torch(raw),
                ],
                device=str(self.device),
            )
        return valid, raw
