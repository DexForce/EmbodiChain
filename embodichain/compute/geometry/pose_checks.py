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

"""Pure rigid input validation and emitted-pose residual checks."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
import torch

from embodichain.utils.math import quat_error_magnitude, quat_from_matrix

__all__ = ["as_rigid_matrix", "sampled_pose_path_matches"]


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
