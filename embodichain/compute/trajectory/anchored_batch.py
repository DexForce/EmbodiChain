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

"""Prepare auditable ragged clocks once and evaluate batched joint paths."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch

from .anchored import AnchoredTimeMap, _prepare_time_map

__all__ = [
    "AnchoredBatchPlan",
    "prepare_anchored_batch",
    "resample_with_anchors_batched",
]


@dataclass(frozen=True)
class AnchoredBatchPlan:
    """Reusable interpolation maps on one execution device.

    Host rows retain immutable float64 timing evidence. Device buffers contain
    only interpolation indices/weights; no robot or trajectory values are owned.
    Padding copies the last valid source sample and is excluded by valid_counts.
    Plans can be shared by trajectories with the same clocks and source lengths.

    Args:
        rows: Nonempty tuple of independently validated source/execution maps.
        device: Device used by every tensor passed to :meth:`apply`.
        max_samples: Bound on the total rectangular grid size B * M_max.
    """

    rows: tuple[AnchoredTimeMap, ...]
    device: torch.device | str = "cpu"
    max_samples: int = 1_000_000
    _left: torch.Tensor = field(init=False, repr=False, compare=False)
    _right: torch.Tensor = field(init=False, repr=False, compare=False)
    _weights: torch.Tensor = field(init=False, repr=False, compare=False)
    _source_lengths: torch.Tensor = field(init=False, repr=False, compare=False)
    _valid_counts: torch.Tensor = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        rows = tuple(self.rows)
        if not rows or any(not isinstance(r, AnchoredTimeMap) for r in rows):
            raise ValueError("rows must contain nonempty AnchoredTimeMap values")
        if type(self.max_samples) is not int or self.max_samples < 2:
            raise ValueError("max_samples must be an integer >= 2")
        width = max(len(row.execution_times) for row in rows)
        if len(rows) * width > self.max_samples:
            raise ValueError("Padded execution grid exceeds max_samples")
        device = torch.device(self.device)
        if device.type not in ("cpu", "cuda"):
            raise ValueError("Anchored batches support CPU and CUDA devices")
        left = np.empty((len(rows), width), dtype=np.int64)
        right = np.empty_like(left)
        weights = np.zeros((len(rows), width), dtype=np.float64)
        for b, row in enumerate(rows):
            size = len(row.execution_times)
            left[b] = right[b] = len(row.source_times) - 1
            left[b, :size] = row.left_indices
            right[b, :size] = row.right_indices
            weights[b, :size] = row.weights
        object.__setattr__(self, "rows", rows)
        for name, value in (
            ("_left", left),
            ("_right", right),
            ("_weights", weights),
            ("_source_lengths", np.array([len(r.source_times) for r in rows])),
            ("_valid_counts", np.array([len(r.execution_times) for r in rows])),
        ):
            object.__setattr__(self, name, torch.tensor(value, device=device))
        object.__setattr__(self, "device", self._left.device)

    @property
    def source_lengths(self) -> torch.Tensor:
        """Return an owned integer (B,) tensor of valid source sample counts."""
        return self._source_lengths.clone()

    @property
    def valid_counts(self) -> torch.Tensor:
        """Return an owned integer (B,) tensor of valid execution sample counts."""
        return self._valid_counts.clone()

    @property
    def valid_mask(self) -> torch.Tensor:
        """Return a boolean (B,M_max) mask excluding terminal-hold padding."""
        return (
            torch.arange(self._left.shape[1], device=self.device)[None]
            < self._valid_counts[:, None]
        )

    def apply(
        self,
        positions: torch.Tensor,
        *,
        backend: Literal["torch", "warp"] = "torch",
        max_output_elements: int = 100_000_000,
    ) -> torch.Tensor:
        """Evaluate a batch using prepared clocks, preserving dtype and device.

        Args:
            positions: Float32/float64 (B,N_max,D) tensor. Only each row's
                declared valid source prefix must be finite; padding is ignored.
            backend: Explicit execution backend. Warp errors propagate without
                silently selecting another backend. Torch supports qpos autograd;
                Warp rejects gradient-bearing inputs in this implementation.
            max_output_elements: Bound on B * M_max * D before output allocation.

        Returns:
            Owned (B,M_max,D) tensor with exact anchor copies and terminal holds.

        Raises:
            ValueError: Invalid shape, device, dtype, backend or finite samples.
        """
        if backend not in ("torch", "warp"):
            raise ValueError("backend must be torch or warp")
        if (
            not isinstance(positions, torch.Tensor)
            or positions.ndim != 3
            or positions.dtype not in (torch.float32, torch.float64)
            or positions.shape[0] != len(self.rows)
            or positions.shape[2] < 1
            or positions.device != self.device
            or positions.shape[1] < max(len(r.source_times) for r in self.rows)
        ):
            raise ValueError(
                "Expected float32/float64 (B,N_max,D) positions on the plan device"
            )
        if type(max_output_elements) is not int or max_output_elements < 1:
            raise ValueError("max_output_elements must be a positive integer")
        if self._left.numel() * positions.shape[-1] > max_output_elements:
            raise ValueError("Resampled positions exceed max_output_elements")
        source_mask = (
            torch.arange(positions.shape[1], device=self.device)[None]
            < self._source_lengths[:, None]
        )
        if not (torch.isfinite(positions).all(dim=-1) | ~source_mask).all():
            raise ValueError("Valid source positions must be finite")
        if backend == "warp":
            if positions.requires_grad:
                raise ValueError(
                    "Warp anchored resampling does not support autograd; use torch"
                )
            from ._warp.anchored import evaluate

            return evaluate(positions, self._left, self._right, self._weights)
        dimensions = positions.shape[-1]
        q0 = positions.gather(1, self._left[..., None].expand(-1, -1, dimensions))
        q1 = positions.gather(1, self._right[..., None].expand(-1, -1, dimensions))
        # Match the single NumPy reference: subtract in qpos dtype, accumulate
        # with float64 time weights, then cast once. Never demote float64 qpos.
        interpolated = (q0.double() + self._weights[..., None] * (q1 - q0).double()).to(
            positions.dtype
        )
        return torch.where((self._left == self._right)[..., None], q0, interpolated)


def prepare_anchored_batch(
    source_times: torch.Tensor,
    control_dt: float,
    anchor_indices: torch.Tensor,
    *,
    source_lengths: torch.Tensor,
    anchor_counts: torch.Tensor,
    device: torch.device | str | None = None,
    max_anchor_delay: float | None = None,
    max_samples: int = 1_000_000,
) -> AnchoredBatchPlan:
    """Prepare independent ragged clocks and reusable device interpolation maps.

    Mapping/anchor decisions run on the CPU in float64, using exactly the same
    implementation as the single-path API. Valid prefixes alone are inspected.
    Preparation may synchronize CUDA metadata once; apply reuses device buffers.

    Args:
        source_times: Floating (B,N_max) relative seconds; each valid prefix
            starts at zero and strictly increases. Padding may contain NaN.
        control_dt: Shared positive control period in seconds; physics is untouched.
        anchor_indices: Integer (B,K_max) source indices. Unused entries are ignored.
        source_lengths: Integer (B,) valid source counts, each >= 2.
        anchor_counts: Integer (B,) valid anchor counts, possibly zero. First and
            last source samples are always added, and duplicate anchors coalesce.
        device: Evaluation device; defaults to source_times.device.
        max_anchor_delay: Permitted forward event delay; None means one control tick.
        max_samples: Bound on both per-row and total padded execution samples.

    Returns:
        A plan retaining per-row audit maps, device buffers and valid lengths.

    Raises:
        ValueError: Invalid inputs or any row whose events cannot fit the clock.
            Row-level errors identify the batch row; no failed row is discarded.
    """
    if (
        not isinstance(source_times, torch.Tensor)
        or source_times.ndim != 2
        or source_times.dtype not in (torch.float32, torch.float64)
        or source_times.shape[0] < 1
    ):
        raise ValueError("source_times must be floating (B,N_max), B >= 1")
    batch, width = source_times.shape
    if type(max_samples) is not int or max_samples < 2 * batch:
        raise ValueError("max_samples must allow at least two samples per batch row")
    for name, value, shape in (
        ("source_lengths", source_lengths, (batch,)),
        ("anchor_counts", anchor_counts, (batch,)),
    ):
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != shape
            or value.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError(f"{name} must be an integer (B,) tensor")
    if (
        not isinstance(anchor_indices, torch.Tensor)
        or anchor_indices.ndim != 2
        or anchor_indices.shape[0] != batch
        or anchor_indices.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("anchor_indices must be an integer (B,K_max) tensor")
    lengths = source_lengths.detach().cpu().tolist()
    counts = anchor_counts.detach().cpu().tolist()
    if any(n < 2 or n > width for n in lengths):
        raise ValueError("source_lengths must be between 2 and N_max")
    if any(k < 0 or k > anchor_indices.shape[1] for k in counts):
        raise ValueError("anchor_counts must be between 0 and K_max")
    times = source_times.detach().to(device="cpu", dtype=torch.float64).numpy()
    anchors = anchor_indices.detach().cpu().numpy()
    rows = []
    for b, (n, k) in enumerate(zip(lengths, counts)):
        try:
            rows.append(
                _prepare_time_map(
                    times[b, :n],
                    control_dt,
                    anchors[b, :k].tolist(),
                    max_anchor_delay=max_anchor_delay,
                    max_samples=max_samples // batch,
                )
            )
        except ValueError as error:
            raise ValueError(f"Batch row {b}: {error}") from error
    return AnchoredBatchPlan(
        tuple(rows), source_times.device if device is None else device, max_samples
    )


def resample_with_anchors_batched(
    positions: torch.Tensor,
    source_times: torch.Tensor,
    control_dt: float,
    anchor_indices: torch.Tensor,
    *,
    source_lengths: torch.Tensor,
    anchor_counts: torch.Tensor,
    backend: Literal["torch", "warp"] = "torch",
    max_anchor_delay: float | None = None,
    max_samples: int = 1_000_000,
    max_output_elements: int = 100_000_000,
) -> tuple[torch.Tensor, AnchoredBatchPlan]:
    """Resample ragged joint paths onto fixed clocks with exact event samples.

    Args:
        positions: Float32/float64 (B,N_max,D) joint paths.
        source_times: Floating (B,N_max) relative seconds; padding is ignored.
        control_dt: Shared destination control period, in seconds.
        anchor_indices: Integer (B,K_max) hard source sample indices.
        source_lengths: Integer (B,) valid source counts.
        anchor_counts: Integer (B,) valid anchor counts.
        backend: Explicit torch (default) or warp execution backend.
        max_anchor_delay: Maximum forward anchor delay; defaults to one tick.
        max_samples: Maximum padded execution grid size B * M_max.
        max_output_elements: Maximum output allocation B * M_max * D.

    Returns:
        Resampled positions and reusable plan. Use plan.valid_counts or
        plan.valid_mask to exclude padding. This does not compute qvel or
        qualify physical motion; those remain caller responsibilities.
    """
    if not isinstance(positions, torch.Tensor):
        raise ValueError("positions must be a tensor")
    plan = prepare_anchored_batch(
        source_times,
        control_dt,
        anchor_indices,
        source_lengths=source_lengths,
        anchor_counts=anchor_counts,
        device=positions.device,
        max_anchor_delay=max_anchor_delay,
        max_samples=max_samples,
    )
    return (
        plan.apply(positions, backend=backend, max_output_elements=max_output_elements),
        plan,
    )
