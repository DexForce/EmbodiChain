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

"""Gradient stabilization primitives for differentiable rollouts."""

from __future__ import annotations

import torch

__all__ = ["BatchedGradientNormStats", "clip_batched_gradient_norm"]


class BatchedGradientNormStats:
    """Accumulate row-wise adjoint norm and clipping statistics on-device.

    Args:
        device: Device on which hook-side counters are accumulated.
    """

    def __init__(self, device: torch.device | str) -> None:
        self.device = torch.device(device)
        self.norm_sum = torch.zeros((), device=self.device)
        self.norm_max = torch.zeros((), device=self.device)
        self.finite_rows = torch.zeros((), device=self.device)
        self.rows = torch.zeros((), device=self.device)
        self.clipped_rows = torch.zeros((), device=self.device)
        self.nonfinite_rows = torch.zeros((), device=self.device)

    def metrics(self) -> dict[str, float]:
        """Return statistics after backward has invoked registered hooks.

        Returns:
            Mean and maximum pre-clip norm plus clipped/non-finite fractions.
        """
        rows = float(self.rows)
        finite_rows = float(self.finite_rows)
        return {
            "action_adjoint_preclip_mean_norm": (
                float(self.norm_sum) / finite_rows if finite_rows > 0.0 else 0.0
            ),
            "action_adjoint_preclip_max_norm": float(self.norm_max),
            "action_adjoint_clipped_fraction": (
                float(self.clipped_rows) / rows if rows > 0.0 else 0.0
            ),
            "action_adjoint_nonfinite_fraction": (
                float(self.nonfinite_rows) / rows if rows > 0.0 else 0.0
            ),
        }


def clip_batched_gradient_norm(
    gradient: torch.Tensor,
    max_norm: float,
    stats: BatchedGradientNormStats | None = None,
) -> torch.Tensor:
    """Clip each batch row without shortening the differentiable time horizon.

    Norms are computed with max-absolute-value scaling to avoid overflow in
    float32. A non-finite row is replaced with zeros while finite rows remain
    independent from one another.

    Args:
        gradient: Tensor whose first dimension identifies independent rows.
        max_norm: Maximum L2 norm per row. Zero disables clipping.
        stats: Optional on-device accumulator populated before clipping.

    Returns:
        A finite tensor with the same shape, dtype, and device as ``gradient``.

    Raises:
        ValueError: If ``max_norm`` is negative or ``gradient`` is not batched.
    """
    if max_norm < 0.0:
        raise ValueError("max_norm cannot be negative.")
    if max_norm == 0.0:
        return gradient
    if gradient.ndim < 2:
        raise ValueError("gradient must have a leading batch dimension.")

    flat = gradient.flatten(start_dim=1)
    finite_rows = torch.isfinite(flat).all(dim=1, keepdim=True)
    finite_values = torch.where(finite_rows, flat, torch.zeros_like(flat))
    max_abs = finite_values.abs().amax(dim=1, keepdim=True)
    safe_max_abs = max_abs.clamp_min(1.0e-12)
    scaled_norm = (finite_values / safe_max_abs).norm(dim=1, keepdim=True)
    raw_norm = max_abs * scaled_norm

    if stats is not None:
        with torch.no_grad():
            detached_norm = raw_norm.detach().flatten()
            finite = finite_rows.detach().flatten() & torch.isfinite(detached_norm)
            finite_norm = torch.where(
                finite,
                detached_norm,
                torch.zeros_like(detached_norm),
            )
            stats.norm_sum.add_(finite_norm.sum())
            stats.finite_rows.add_(finite.sum())
            stats.rows.add_(detached_norm.numel())
            stats.clipped_rows.add_((finite & (detached_norm > max_norm)).sum())
            stats.nonfinite_rows.add_((~finite).sum())
            stats.norm_max.copy_(torch.maximum(stats.norm_max, finite_norm.max()))

    scale = ((float(max_norm) / safe_max_abs) / scaled_norm.clamp_min(1.0)).clamp(
        max=1.0
    )
    scale = torch.where(finite_rows, scale, torch.zeros_like(scale))
    broadcast_shape = (-1,) + (1,) * (gradient.ndim - 1)
    safe_gradient = torch.where(
        finite_rows.view(broadcast_shape),
        gradient,
        torch.zeros_like(gradient),
    )
    return safe_gradient * scale.view(broadcast_shape)
