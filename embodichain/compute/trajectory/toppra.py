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

"""Pure tensor operations for differentiable joint-path time parameterization."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from ._toppra_warp import _retime_toppra_warp

__all__ = ["retime_joint_paths"]


def retime_joint_paths(
    positions: torch.Tensor,
    velocity_limits: float | Sequence[float] | torch.Tensor,
    acceleration_limits: float | Sequence[float] | torch.Tensor,
    *,
    path_lengths: torch.Tensor | None = None,
    sample_count: int | None = None,
    sample_dt: float | None = None,
    grid_size: int = 100,
    output_dtype: torch.dtype | None = None,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Fit and time joint paths with CPU/CUDA Warp TOPPRA.

    Args:
        positions: Joint paths ``(B >= 1, N >= 2, J >= 1)``, float32 or float64.
        velocity_limits: Scalar, ``(J,)`` magnitudes or signed ``(J,2)`` bounds
            containing zero, shared by the batch. Robot caps are caller-owned.
        acceleration_limits: Acceleration bounds with the same supported forms.
        path_lengths: Integer input-prefix lengths ``(B,)``, each in ``[2,N]``.
            Padding is excluded; existing filtering applies inside the prefix.
            Repeated final nodes retain the solver's final-knot geometry.
        sample_count: Fixed output width. Defaults to ``N`` if neither sampling
            argument is supplied. Mutually exclusive with ``sample_dt``.
        sample_dt: Positive upper bound on seconds between samples. Each row is
            sampled uniformly over its solved duration; this is not a fixed
            controller clock. Shorter rows are padded to the batch maximum.
        grid_size: Minimum timing grid size, at least three. Actual work also
            depends on the input node count.
        output_dtype: Optional float32/float64 output precision. Defaults to the
            input precision. Float64 avoids position quantization when dense
            outputs are differentiated. Input filtering keeps input precision.

    Returns:
        Plain tensor tuple ``(q, qd, qdd, dt, success, lengths)``. The first
        three tensors are ``(B,M,J)``, ``dt`` is ``(B,M)``, and the final tensors
        are ``(B,)`` bool/int64. Failed rows are zero-filled with length zero;
        stationary successes retain two valid zero-time samples. Padding repeats
        endpoints with zero derivatives/intervals. Duration is ``dt.sum(-1)``.
        Solver success does not certify task completion or geometric validity.

    Raises:
        ValueError: For malformed shapes, precision, lengths, limits or sampling.

    Note:
        Positions and tensor limits retain first-order gradients. Filtering,
        grid/sampling topology and active constraints are held fixed during
        backward; gradients need not exist at their switches. Stationary
        durations and failed rows supply no timing gradient. Higher-order
        derivatives are unsupported. Internal computation is float64.
    """
    if not isinstance(positions, torch.Tensor) or positions.ndim != 3:
        raise ValueError("positions must be a (B, N, J) tensor")
    if sample_count is None and sample_dt is None:
        sample_count = positions.shape[1]
    values = _retime_toppra_warp(
        positions,
        velocity_limits,
        acceleration_limits,
        path_lengths=path_lengths,
        sample_count=sample_count,
        sample_dt=sample_dt,
        grid_size=grid_size,
        output_dtype=output_dtype,
    )
    lengths = (values["dt"] > 0).sum(-1).add(1).clamp_min(2)
    lengths = torch.where(values["success"], lengths, 0)
    # Quantity is a rectangular tensor contract even for all-static/failed
    # batches. Validity remains explicit in success and lengths.
    if sample_count is not None:
        extra = sample_count - values["positions"].shape[1]
        if extra:
            for key in ("positions", "velocities", "accelerations", "dt"):
                value = values[key]
                shape = (value.shape[0], extra, *value.shape[2:])
                padding = (
                    value[:, -1:].expand(shape)
                    if key == "positions"
                    else value.new_zeros(shape)
                )
                values[key] = torch.cat((value, padding), dim=1)
    return (
        values["positions"],
        values["velocities"],
        values["accelerations"],
        values["dt"],
        values["success"],
        lengths,
    )
