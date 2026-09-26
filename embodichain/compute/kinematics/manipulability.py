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

"""Pure batched manipulability computations over robot Jacobians.

These helpers are the single numerical source shared by workspace analysis and
IK candidate ranking. They operate on batched Torch tensors, preserve the input
dtype and device, and never import simulation or workspace modules. Aggregation
and result assembly belong to the consuming layers.

A Jacobian batch has shape ``(N, R, DOF)`` where ``R`` is 6 for a full spatial
Jacobian; :func:`select_jacobian_rows` extracts task-specific row subsets (for
example the translational rows) before scoring.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

__all__ = [
    "select_jacobian_rows",
    "yoshikawa_manipulability",
    "condition_number",
]

# Row indices of a spatial ``(6, DOF)`` Jacobian: linear velocity first.
_TRANSLATIONAL_ROWS = (0, 1, 2)
_ROTATIONAL_ROWS = (3, 4, 5)


def select_jacobian_rows(
    jacobian: torch.Tensor,
    rows: str | Sequence[int] = "all",
) -> torch.Tensor:
    """Select task-relevant rows of a batched spatial Jacobian.

    Args:
        jacobian: Batched Jacobian with shape ``(N, R, DOF)``.
        rows: ``"all"`` (default) keeps every row; ``"translational"`` keeps the
            linear-velocity rows ``(0, 1, 2)``; ``"rotational"`` keeps the
            angular-velocity rows ``(3, 4, 5)``; a sequence of ints selects
            exactly those row indices.

    Returns:
        The row-subset Jacobian with shape ``(N, len(rows), DOF)``.

    Raises:
        ValueError: If ``jacobian`` is not 3D, ``rows`` names an unknown
            selection, or a requested index is out of range.
    """
    if jacobian.ndim != 3:
        raise ValueError(
            f"jacobian must be batched (N, R, DOF); got shape {tuple(jacobian.shape)}."
        )
    if isinstance(rows, str):
        if rows == "all":
            return jacobian
        if rows == "translational":
            selected = _TRANSLATIONAL_ROWS
        elif rows == "rotational":
            selected = _ROTATIONAL_ROWS
        else:
            raise ValueError(
                "rows string must be 'all', 'translational', or 'rotational'; "
                f"got {rows!r}."
            )
    else:
        selected = tuple(int(r) for r in rows)
        if not selected:
            raise ValueError("rows sequence must not be empty.")
    num_rows = jacobian.shape[1]
    if any(r < 0 or r >= num_rows for r in selected):
        raise ValueError(
            f"row indices {selected} out of range for {num_rows}-row Jacobian."
        )
    index = torch.as_tensor(selected, dtype=torch.long, device=jacobian.device)
    return jacobian.index_select(dim=1, index=index)


def yoshikawa_manipulability(jacobian: torch.Tensor) -> torch.Tensor:
    """Compute the Yoshikawa manipulability index ``sqrt(det(J @ J^T))``.

    The measure is the volume of the velocity manipulability ellipsoid. It is
    zero at singular configurations and positive elsewhere.

    Args:
        jacobian: Batched Jacobian with shape ``(N, R, DOF)`` (``R <= DOF`` for
            a non-degenerate ``J @ J^T``).

    Returns:
        Scores with shape ``(N,)``, same dtype and device as ``jacobian``. The
        determinant is clamped at zero before the square root so numerical
        round-off near singularities never yields NaN.

    Raises:
        ValueError: If ``jacobian`` is not batched 3D.
    """
    if jacobian.ndim != 3:
        raise ValueError(
            f"jacobian must be batched (N, R, DOF); got shape {tuple(jacobian.shape)}."
        )
    jjt = jacobian @ jacobian.transpose(-2, -1)
    det = torch.linalg.det(jjt)
    return torch.sqrt(torch.clamp(det, min=0.0))


def condition_number(
    jacobian: torch.Tensor,
    eps: float = 1e-15,
) -> torch.Tensor:
    """Compute the Jacobian condition number ``sigma_max / sigma_min``.

    A large condition number marks an ill-conditioned (near-singular) posture
    whose manipulability ellipsoid is highly anisotropic.

    Args:
        jacobian: Batched Jacobian with shape ``(N, R, DOF)``.
        eps: Lower clamp on the smallest singular value to avoid division by
            zero; an exactly singular Jacobian yields a large finite ratio.

    Returns:
        Condition numbers with shape ``(N,)``, same dtype and device as
        ``jacobian``.

    Raises:
        ValueError: If ``jacobian`` is not batched 3D.
    """
    if jacobian.ndim != 3:
        raise ValueError(
            f"jacobian must be batched (N, R, DOF); got shape {tuple(jacobian.shape)}."
        )
    singular_values = torch.linalg.svdvals(jacobian)
    max_sv = singular_values[:, 0]
    min_sv = singular_values[:, -1].clamp_min(eps)
    return max_sv / min_sv
