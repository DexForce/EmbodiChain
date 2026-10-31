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

"""A fixed-budget Torch layer for differentiable joint-path retiming."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from embodichain.compute.trajectory import JointPathTimingResult, retime_joint_paths
from embodichain.compute.trajectory._toppra_common import (
    _validate_grid_size,
    _validate_sampling,
)

__all__ = ["DifferentiableRetimingLayer"]


class DifferentiableRetimingLayer(nn.Module):
    """Retime paths with TOPPRA, retaining first-order gradients to their generator.

    Args:
        velocity_limits: Fixed scalar, per-joint or signed velocity bounds.
        acceleration_limits: Fixed scalar, per-joint or signed acceleration bounds.
        sample_count: Fixed rectangular output width, at least two.
        grid_size: Minimum timing grid size, at least three.

    Motion limits are buffers, not trainable parameters or robot-derived caps.
    Per-call overrides use the same joint order and are shared by the batch.
    ``eval()`` retains gradients; use ``torch.no_grad()`` for inference. Inputs
    must share this layer's device and floating precision. Buffers and numerical
    settings are checkpointed; no per-call graph or optimizer state is retained.
    """

    def __init__(
        self,
        velocity_limits: float | torch.Tensor,
        acceleration_limits: float | torch.Tensor,
        *,
        sample_count: int = 64,
        grid_size: int = 100,
    ) -> None:
        super().__init__()
        _validate_sampling(sample_count, None)
        _validate_grid_size(grid_size)
        self.sample_count = int(sample_count)
        self.grid_size = int(grid_size)
        for name, value in (
            ("velocity_limits", velocity_limits),
            ("acceleration_limits", acceleration_limits),
        ):
            tensor = torch.as_tensor(value).detach().clone()
            if not tensor.is_floating_point():
                tensor = tensor.float()
            if tensor.dtype not in (torch.float32, torch.float64):
                raise ValueError("TOPPRA limits support float32 and float64")
            if not torch.isfinite(tensor).all():
                raise ValueError("TOPPRA limits must be finite")
            self.register_buffer(name, tensor)

    def get_extra_state(self) -> dict[str, int]:
        """Save numerical settings alongside fixed motion-limit buffers."""
        return {"sample_count": self.sample_count, "grid_size": self.grid_size}

    def set_extra_state(self, state: dict[str, int]) -> None:
        """Restore and validate checkpointed numerical settings."""
        _validate_sampling(state["sample_count"], None)
        _validate_grid_size(state["grid_size"])
        self.sample_count = int(state["sample_count"])
        self.grid_size = int(state["grid_size"])

    def _apply(self, fn: Any, recurse: bool = True) -> DifferentiableRetimingLayer:
        if fn(self.velocity_limits).dtype not in (torch.float32, torch.float64):
            raise ValueError("DifferentiableRetimingLayer supports float32 and float64")
        return super()._apply(fn, recurse=recurse)

    def forward(
        self,
        positions: torch.Tensor,
        *,
        path_lengths: torch.Tensor | None = None,
        velocity_limits: float | torch.Tensor | None = None,
        acceleration_limits: float | torch.Tensor | None = None,
    ) -> JointPathTimingResult:
        """Return a fixed-width trajectory with native validity counts.

        Args:
            positions: Graph-connected paths, shape ``(B, N, J)``.
            path_lengths: Valid input prefixes, including initial and final points.
            velocity_limits: Optional shared request-level velocity bounds.
            acceleration_limits: Optional shared request-level acceleration bounds.

        Returns:
            Timed paths of width ``sample_count``. Trailing positions repeat
            their endpoint; derivatives and intervals are zero. Failure lengths
            remain zero, stationary success lengths remain two.
        """
        if any(
            positions.device != buffer.device or positions.dtype != buffer.dtype
            for buffer in (self.velocity_limits, self.acceleration_limits)
        ):
            raise ValueError("paths must match the layer's device and dtype")
        result = retime_joint_paths(
            positions,
            constraints={
                "velocity": (
                    self.velocity_limits if velocity_limits is None else velocity_limits
                ),
                "acceleration": (
                    self.acceleration_limits
                    if acceleration_limits is None
                    else acceleration_limits
                ),
            },
            path_lengths=path_lengths,
            sample_count=self.sample_count,
            grid_size=self.grid_size,
        )
        extra = self.sample_count - result.positions.shape[1]
        if extra == 0:
            return result
        values = {}
        for key in ("positions", "velocities", "accelerations", "dt"):
            value = getattr(result, key)
            shape = (value.shape[0], extra, *value.shape[2:])
            padding = (
                value[:, -1:].expand(shape)
                if key == "positions"
                else value.new_zeros(shape)
            )
            values[key] = torch.cat((value, padding), dim=1)
        return JointPathTimingResult(
            **values, success=result.success, sample_lengths=result.sample_lengths
        )
