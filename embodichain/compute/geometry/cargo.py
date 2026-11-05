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


"""Pure carrier-envelope geometry, independent of simulation and tasks."""

from __future__ import annotations

from typing import Literal
import torch

__all__ = ["inside_carrier_envelope", "first_carrier_envelope_failure"]


def inside_carrier_envelope(
    vertices: torch.Tensor,
    object_pose: torch.Tensor,
    carrier_pose: torch.Tensor,
    bounds: torch.Tensor,
) -> torch.Tensor:
    """Check horizontal mesh extent and its lowest point against a carrier AABB.

    Args:
        vertices: Object-local points, shape ``(V, 3)`` or ``(B, V, 3)``.
        object_pose: World object transforms, shape ``(B, 4, 4)``.
        carrier_pose: World carrier transforms, shape ``(B, 4, 4)``.
        bounds: Carrier-local lower/upper bounds, shape ``(2, 3)``.

    Returns:
        Boolean mask per row, using the original two-millimetre tolerance.
        The upper extent is deliberately unconstrained: this is an envelope
        condition, not an inner-cavity or physical retention certificate.
    """
    relative = torch.linalg.solve(carrier_pose, object_pose)
    local = vertices @ relative[:, :3, :3].transpose(-1, -2) + relative[:, None, :3, 3]
    minimum, maximum = local.amin(1), local.amax(1)
    margin = 0.002
    return (
        torch.isfinite(local).all(dim=-1).all(dim=-1)
        & (minimum[:, :2] >= bounds[0, :2] - margin).all(-1)
        & (maximum[:, :2] <= bounds[1, :2] + margin).all(-1)
        & (minimum[:, 2] >= bounds[0, 2] - margin)
        & (minimum[:, 2] <= bounds[1, 2] + margin)
    )


def first_carrier_envelope_failure(
    vertices: torch.Tensor,
    object_poses: torch.Tensor,
    carrier_poses: torch.Tensor,
    bounds: torch.Tensor,
    *,
    frame_count: int,
    pose_format: Literal["xyz_xyzw", "matrix"] = "xyz_xyzw",
) -> int | None:
    """Find the first recorded envelope violation using the source 64-frame grid.

    Args:
        vertices: Object-local points, shape (V, 3).
        object_poses: Recorded samples (T, 7) or explicit rigid matrices (T, 4, 4).
        carrier_poses: Carrier samples/matrices in the same explicit format.
        bounds: Source carrier-local bounds, shape (2, 3).
        frame_count: Exact valid prefix length of this environment's recording.
        pose_format: Raw recorded XYZ/XYZW values or already-converted rigid
            matrices from the public recording owner. No matrix is reserialized.

    Returns:
        First rejected zero-based frame, or None when every supplied frame passes.
        Empty/incomplete recordings raise rather than establish retention.
    """
    from embodichain.utils.math import xyz_quat_to_4x4_matrix

    shape = (7,) if pose_format == "xyz_xyzw" else (4, 4)
    if (
        pose_format not in {"xyz_xyzw", "matrix"}
        or type(frame_count) is not int
        or frame_count <= 0
        or object_poses.shape[1:] != shape
        or carrier_poses.shape[1:] != shape
        or frame_count > min(len(object_poses), len(carrier_poses))
    ):
        raise ValueError("Cargo trajectory has no complete valid frame range.")
    for poses in (object_poses, carrier_poses):
        prefix = poses[:frame_count]
        if not torch.isfinite(prefix).all() or (
            pose_format == "xyz_xyzw"
            and (torch.linalg.vector_norm(prefix[:, 3:], dim=-1) <= 1e-8).any()
        ):
            raise ValueError(
                "Cargo recorded pose is unavailable or has invalid orientation."
            )

    def matrices(values: torch.Tensor) -> torch.Tensor:
        values = values.to(vertices)
        return xyz_quat_to_4x4_matrix(values) if pose_format == "xyz_xyzw" else values

    for start in range(0, frame_count, 64):
        stop = min(start + 64, frame_count)
        valid = inside_carrier_envelope(
            vertices,
            matrices(object_poses[start:stop]),
            matrices(carrier_poses[start:stop]),
            bounds,
        )
        failed = (~valid).nonzero()
        if len(failed):
            return start + int(failed[0, 0])
    return None
