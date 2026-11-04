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

"""Internal owned identifiers and proper rigid-transform validation."""

from __future__ import annotations

import math
from numbers import Real
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

__all__ = []


def _identifier(value: str, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{name} must be a nonempty stripped string")
    return value


def _nonnegative_integer(value: int, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _rigid_pose(value: object) -> tuple[tuple[float, ...], ...]:
    try:
        rows = tuple(tuple(row) for row in value)  # type: ignore[union-attr]
    except TypeError as exc:
        raise ValueError("arena_pose must be a 4x4 rigid transform") from exc
    if len(rows) != 4 or any(len(row) != 4 for row in rows):
        raise ValueError("arena_pose must be a 4x4 rigid transform")
    if any(
        isinstance(item, bool) or not isinstance(item, Real) or not math.isfinite(item)
        for row in rows
        for item in row
    ):
        raise ValueError("arena_pose must contain finite real numbers")
    pose = tuple(
        tuple(float(item) if item != 0 else 0.0 for item in row) for row in rows
    )
    tolerance = 1e-6
    if any(
        abs(pose[3][index] - target) > tolerance
        for index, target in enumerate((0, 0, 0, 1))
    ):
        raise ValueError("arena_pose must have homogeneous last row [0, 0, 0, 1]")
    rotation = tuple(row[:3] for row in pose[:3])
    if any(
        abs(sum(rotation[k][i] * rotation[k][j] for k in range(3)) - (i == j))
        > tolerance
        for i in range(3)
        for j in range(3)
    ):
        raise ValueError("arena_pose rotation must be orthonormal")
    determinant = (
        rotation[0][0]
        * (rotation[1][1] * rotation[2][2] - rotation[1][2] * rotation[2][1])
        - rotation[0][1]
        * (rotation[1][0] * rotation[2][2] - rotation[1][2] * rotation[2][0])
        + rotation[0][2]
        * (rotation[1][0] * rotation[2][1] - rotation[1][1] * rotation[2][0])
    )
    if abs(determinant - 1.0) > tolerance:
        raise ValueError("arena_pose rotation must be proper (determinant +1)")
    return (*pose[:3], (0.0, 0.0, 0.0, 1.0))


def _owned_batch_pose(value: torch.Tensor, name: str) -> torch.Tensor:
    """Own nonempty batched rigid tensor poses using the scene/query tolerance contract."""
    import torch

    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != 3
        or value.shape[0] < 1
        or value.shape[1:] != (4, 4)
        or not value.is_floating_point()
    ):
        raise ValueError(f"{name} requires finite floating poses (B, 4, 4).")
    owned = value.detach().clone()
    # Validate small metadata matrices in one CPU transfer, avoiding a
    # cuSOLVER initialization merely to check a 3x3 determinant on the sim GPU.
    matrix = owned.to(device="cpu", dtype=torch.float64)
    rotation = matrix[:, :3, :3]
    if (
        not torch.isfinite(matrix).all()
        or not torch.allclose(
            matrix[:, 3],
            matrix.new_tensor([0, 0, 0, 1]).expand(len(matrix), -1),
            atol=1e-6,
            rtol=0,
        )
        or not torch.allclose(
            rotation.transpose(-1, -2) @ rotation,
            torch.eye(3, device=matrix.device, dtype=matrix.dtype).expand(
                len(matrix), -1, -1
            ),
            atol=1e-5,
            rtol=0,
        )
        or not torch.isclose(
            torch.linalg.det(rotation), matrix.new_tensor(1.0), atol=1e-5, rtol=0
        ).all()
    ):
        raise ValueError(f"{name} must be a rigid transform.")
    return owned


def _owned_b1_pose(value: torch.Tensor, name: str) -> torch.Tensor:
    """Keep the B=1 scene provider boundary unchanged."""
    import torch

    if not isinstance(value, torch.Tensor) or value.shape != (1, 4, 4):
        raise ValueError(f"{name} requires one finite floating pose (1, 4, 4).")
    return _owned_batch_pose(value, name)
