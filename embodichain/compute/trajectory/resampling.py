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

"""Resample batched paths at uniform cumulative-distance positions."""

from __future__ import annotations

import torch
import warp as wp

from embodichain.utils.device_utils import standardize_device_string
from ._warp.resampling import (
    pairwise_distances,
    cumsum_distances,
    repeat_first_point,
    interpolate_along_distance,
)

__all__ = ["resample_with_distance"]


def resample_with_distance(
    trajectory: torch.Tensor,
    interp_num: int,
    device: torch.device | str = torch.device("cuda"),
) -> torch.Tensor:
    """Resample a batched path at uniform cumulative-distance positions.

    Unlike :func:`interpolate_with_distance`, interior input samples are not
    required output points, so this function supports both upsampling and
    downsampling. It is intended for dense planner paths rather than required
    waypoint sequences.

    Args:
        trajectory: Path tensor with shape ``(B, N, M)``.
        interp_num: Target number of samples ``T``.
        device: Device on which to perform interpolation.

    Returns:
        Resampled trajectories with shape ``(B, T, M)``.

    Raises:
        ValueError: If ``trajectory`` is not three-dimensional, contains no
            points for a non-empty output, or ``interp_num`` is negative.
    """
    if trajectory.ndim != 3:
        raise ValueError("`trajectory` must have shape (B, N, M).")

    trajectory = trajectory.contiguous().to(device)
    if not torch.is_floating_point(trajectory) or trajectory.dtype != torch.float32:
        trajectory = trajectory.float()

    batch_size, point_count, dimension = trajectory.shape
    sample_count = int(interp_num)
    if sample_count < 0:
        raise ValueError("`interp_num` must be non-negative.")
    if point_count == 0:
        if sample_count == 0:
            return trajectory.new_empty((batch_size, 0, dimension))
        raise ValueError("Cannot resample a trajectory with no points.")
    if batch_size == 0 or sample_count == 0:
        return trajectory.new_empty((batch_size, sample_count, dimension))

    # Flatten input trajectory for Warp kernels (avoids multidimensional
    # wp.array interop issues).
    trajectory_flat = trajectory.view(-1)
    points = wp.from_torch(trajectory_flat)

    out = wp.empty(
        (batch_size * sample_count * dimension,),
        dtype=wp.float32,
        device=standardize_device_string(device),
    )

    if point_count == 1:
        wp.launch(
            kernel=repeat_first_point,
            dim=batch_size * sample_count,
            inputs=[
                points,
                out,
                batch_size,
                sample_count,
                dimension,
                point_count,
            ],
            device=standardize_device_string(device),
        )
        return wp.to_torch(out).view(batch_size, sample_count, dimension)

    dists = wp.empty(
        (batch_size * (point_count - 1),),
        dtype=wp.float32,
        device=standardize_device_string(device),
    )
    wp.launch(
        kernel=pairwise_distances,
        dim=batch_size * (point_count - 1),
        inputs=[points, dists, batch_size, point_count, dimension],
        device=standardize_device_string(device),
    )

    cumulative = wp.empty(
        (batch_size * point_count,),
        dtype=wp.float32,
        device=standardize_device_string(device),
    )
    wp.launch(
        kernel=cumsum_distances,
        dim=batch_size,
        inputs=[dists, cumulative, batch_size, point_count],
        device=standardize_device_string(device),
    )

    wp.launch(
        kernel=interpolate_along_distance,
        dim=batch_size * sample_count,
        inputs=[
            points,
            cumulative,
            out,
            batch_size,
            point_count,
            dimension,
            sample_count,
        ],
        device=standardize_device_string(device),
    )
    return wp.to_torch(out).view(batch_size, sample_count, dimension)
