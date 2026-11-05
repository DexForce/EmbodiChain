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

"""Numerical pose residual checks with the existing source tolerance laws."""

from __future__ import annotations

import torch
import numpy as np
from numpy.typing import ArrayLike
from embodichain.utils.math import quat_error_magnitude, quat_from_matrix

__all__ = ["as_rigid_matrix", "fk_pose_error_mask", "sampled_pose_path_matches"]


def fk_pose_error_mask(
    actual: torch.Tensor,
    desired: torch.Tensor,
    position_tolerance: float,
    rotation_tolerance: float,
) -> torch.Tensor:
    """Compare FK rigid poses using the original double-precision atan2 residual.

    Args:
        actual: Already-qualified FK transforms ``(N,4,4)``.
        desired: Required rigid transforms ``(N,4,4)`` or one broadcast pose.
        position_tolerance: Position residual threshold in metres.
        rotation_tolerance: Rotation residual threshold in radians.

    Returns:
        Per-pose residual mask. This compares known rigid poses; it is not a
        substitute for validating an arbitrary matrix as an SE(3) input.
    """
    fk, poses = actual.double(), desired.double()
    position = (fk[:, :3, 3] - poses[:, :3, 3]).norm(dim=-1)
    rotation = fk[:, :3, :3] @ poses[:, :3, :3].transpose(-1, -2)
    skew = torch.stack(
        (
            rotation[:, 2, 1] - rotation[:, 1, 2],
            rotation[:, 0, 2] - rotation[:, 2, 0],
            rotation[:, 1, 0] - rotation[:, 0, 1],
        ),
        dim=-1,
    )
    angle = torch.atan2(
        skew.norm(dim=-1) / 2, (rotation.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
    )
    return (position <= position_tolerance) & (angle <= rotation_tolerance)


def sampled_pose_path_matches(
    actual: torch.Tensor,
    desired: torch.Tensor,
    position_tolerance: float,
    rotation_tolerance: float,
) -> bool:
    """Check every emitted pose with the preserved sampled-manipulation margins.

    Args:
        actual: Emitted FK transforms, with arbitrary leading path dimensions.
        desired: Matching requested rigid transforms.
        position_tolerance: Solver position budget before the original 1e-6 margin.
        rotation_tolerance: Solver angular budget, retaining the source factor 2
            and additional 1e-5 rounding margin.

    Returns:
        Whether all samples agree. This aggregate check does not widen the
        environment support of a caller; B1 callers still reject batched actions.
    """
    if actual.shape != desired.shape or not torch.isfinite(actual).all():
        return False
    position = torch.linalg.vector_norm(
        actual[..., :3, 3] - desired[..., :3, 3], dim=-1
    )
    rotation = quat_error_magnitude(
        quat_from_matrix(actual[..., :3, :3]), quat_from_matrix(desired[..., :3, :3])
    )
    return bool(
        (position <= position_tolerance + 1e-6).all()
        and (rotation <= rotation_tolerance * 2 + 1e-5).all()
    )


def as_rigid_matrix(value: ArrayLike | torch.Tensor) -> np.ndarray:
    """Validate one rigid matrix without changing its numeric dtype.

    Accept (4,4) or one explicit (1,4,4) row. NumPy inputs are borrowed read-only;
    Tensor inputs are detached and transferred to CPU when necessary. The source
    residuals use 1e-6 for the last row and 1e-5 for rotation/determinant, with
    zero relative tolerance. This does not normalize or repair a pose.
    """
    value = (
        value.detach().cpu().numpy()
        if isinstance(value, torch.Tensor)
        else np.asarray(value)
    )
    if value.shape == (1, 4, 4):
        value = value[0]
    if (
        value.shape != (4, 4)
        or not np.isfinite(value).all()
        or not np.allclose(value[3], [0, 0, 0, 1], atol=1e-6, rtol=0)
        or not np.allclose(
            value[:3, :3].T @ value[:3, :3], np.eye(3), atol=1e-5, rtol=0
        )
        or not np.isclose(np.linalg.det(value[:3, :3]), 1, atol=1e-5, rtol=0)
    ):
        raise ValueError("non_rigid_or_missing_staging_pose")
    return value
