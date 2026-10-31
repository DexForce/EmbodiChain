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

"""Simulation-independent, differentiable joint-path time parameterization."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from ._toppra_warp import _retime_toppra_warp

__all__ = ["JointPathTimingResult", "retime_joint_paths"]


@dataclass(frozen=True)
class JointPathTimingResult:
    """A batch of timed paths, in the input joint order, device and selected precision.

    Args:
        positions: Sampled joint positions, shape ``(B, M, J)``.
        velocities: Corresponding physical joint velocities.
        accelerations: Corresponding physical joint accelerations.
        dt: Arrival intervals, shape ``(B, M)``, starting with zero.
        success: Boolean solver status, shape ``(B,)``.
        sample_lengths: Valid output counts, shape ``(B,)`` int64. Failed rows
            have length zero; stationary successes have two zero-time samples.

    Shorter outputs repeat the endpoint with zero derivatives and intervals.
    A solver success certifies neither task completion nor joint-position or
    collision validity. Recompute FK from the returned positions.
    """

    positions: torch.Tensor
    velocities: torch.Tensor
    accelerations: torch.Tensor
    dt: torch.Tensor
    success: torch.Tensor
    sample_lengths: torch.Tensor

    @property
    def duration(self) -> torch.Tensor:
        """Return physical durations in seconds, shape ``(B,)``, retaining gradients."""
        return self.dt.sum(dim=-1)


def retime_joint_paths(
    positions: torch.Tensor,
    *,
    constraints: Mapping[str, float | Sequence[float] | torch.Tensor],
    path_lengths: torch.Tensor | None = None,
    sample_count: int | None = None,
    sample_dt: float | None = None,
    grid_size: int = 100,
    output_dtype: torch.dtype | None = None,
) -> JointPathTimingResult:
    """Fit and time joint paths with the shared CPU/CUDA Warp TOPPRA solver.

    Args:
        positions: Joint paths ``(B >= 1, N >= 2, J >= 1)``, float32 or float64.
        constraints: Required ``velocity`` and ``acceleration`` limits, each a
            scalar, ``(J,)`` magnitudes, or signed ``(J, 2)`` bounds containing
            zero. One set is shared by the batch; no robot caps are implicit.
        path_lengths: Integer valid input prefix lengths ``(B,)``, in ``[2,N]``.
            Padding is ignored. Existing point filtering applies inside the
            prefix; an explicit repeated final node retains its final-knot geometry.
        sample_count: Output quantity for moving successful rows. Defaults to
            ``N`` when neither sampling argument is supplied.
        sample_dt: Positive upper bound on time spacing in seconds. Samples
            are uniform over each solved duration. Mutually exclusive with
            ``sample_count``; this does not define a fixed controller clock.
        output_dtype: Optional float32/float64 output precision. Defaults to the
            input dtype; float64 avoids position quantization when densely
            sampled outputs are differentiated. Input filtering is unchanged.
        grid_size: Minimum timing grid size, at least three. Actual work also
            depends on the number of path nodes.

    Returns:
        Timed paths with isolated, zero-filled failures and native output
        lengths. An all-stationary/failed quantity batch has width two.

    Raises:
        ValueError: For malformed shapes, dtypes, lengths, limits or sampling.

    Note:
        Inputs and tensor limits support first-order gradients through the
        fitted spline and timing. Filtering, active constraints, grid topology
        and output counts are held fixed for each backward; gradients need not
        exist at their switches. Higher-order derivatives are unsupported.
        Stationary durations and failed rows supply no timing gradient.
        Internal numerical work uses float64 on both CPU and CUDA.
    """
    if not isinstance(positions, torch.Tensor) or positions.ndim != 3:
        raise ValueError("positions must be a (B, N, J) tensor")
    missing = {"velocity", "acceleration"}.difference(constraints)
    if missing:
        raise ValueError(f"constraints is missing required keys: {sorted(missing)}")
    if sample_count is None and sample_dt is None:
        sample_count = positions.shape[1]
    values = _retime_toppra_warp(
        positions,
        constraints["velocity"],
        constraints["acceleration"],
        path_lengths=path_lengths,
        sample_count=sample_count,
        sample_dt=sample_dt,
        grid_size=grid_size,
        output_dtype=output_dtype,
    )
    # The native sampler emits strictly positive intervals for moving samples,
    # a leading zero, then zero padding. Stationary successes retain two points.
    lengths = (values["dt"] > 0).sum(-1).add(1).clamp_min(2)
    lengths = torch.where(values["success"], lengths, 0)
    return JointPathTimingResult(**values, sample_lengths=lengths)
