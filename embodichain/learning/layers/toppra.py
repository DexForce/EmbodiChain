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

"""A stateless Torch module for differentiable joint-path retiming."""

from __future__ import annotations

import torch
from torch import nn

from embodichain.compute.trajectory import retime_joint_paths
from embodichain.compute.trajectory._toppra_common import (
    _validate_grid_size,
    _validate_sampling,
)

__all__ = ["DifferentiableRetimingLayer"]


class DifferentiableRetimingLayer(nn.Module):
    """Apply TOPPRA as a tensor-in/tensor-tuple-out numerical layer.

    Args:
        sample_count: Fixed output width, at least two.
        grid_size: Minimum timing grid size, at least three.

    Motion limits are forward inputs. Models with fixed robot limits can own
    them through ``register_buffer``. This layer has no parameters, buffers,
    per-call cache or custom checkpoint/device protocol. Reconstruct numerical
    settings at construction, as for other stateless Torch modules. ``eval()``
    retains input gradients; ``torch.no_grad()`` selects inference.
    """

    def __init__(self, sample_count: int = 64, grid_size: int = 100) -> None:
        super().__init__()
        _validate_sampling(sample_count, None)
        _validate_grid_size(grid_size)
        self.sample_count = int(sample_count)
        self.grid_size = int(grid_size)

    def forward(
        self,
        positions: torch.Tensor,
        velocity_limits: torch.Tensor,
        acceleration_limits: torch.Tensor,
        *,
        path_lengths: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Return retimed tensors while retaining first-order input gradients.

        Args:
            positions: Paths ``(B,N,J)``, float32 or float64.
            velocity_limits: Shared scalar, ``(J,)`` or signed ``(J,2)`` bounds.
            acceleration_limits: Acceleration bounds in the same supported forms.
            path_lengths: Optional valid input-prefix lengths ``(B,)``.

        Returns:
            ``(q, qd, qdd, dt, success, lengths)`` with fixed width
            ``sample_count``. Duration is ``dt.sum(-1)``; failed-row validity
            remains explicit rather than being hidden behind a zero duration.
        """
        return retime_joint_paths(
            positions,
            velocity_limits,
            acceleration_limits,
            path_lengths=path_lengths,
            sample_count=self.sample_count,
            grid_size=self.grid_size,
        )

    def extra_repr(self) -> str:
        """Show numerical hyperparameters in the standard module representation."""
        return f"sample_count={self.sample_count}, grid_size={self.grid_size}"
