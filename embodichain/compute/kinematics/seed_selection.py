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

"""Source-order separated selection from an already qualified CPU IK population."""

from __future__ import annotations

import torch

__all__ = ["select_separated_joint_branches"]


def select_separated_joint_branches(
    qpos: torch.Tensor, valid: torch.Tensor, start: torch.Tensor
) -> tuple[int, ...]:
    """Keep the nearest qualified row, then at most three farthest separated rows.

    Args:
        qpos: CPU candidate matrix ``(N,D)`` with at most 32 rows.
        valid: Boolean validity for every candidate; invalid rows are not chosen.
        start: One CPU starting posture ``(1,D)``.

    Returns:
        Original candidate indices in the source stable order. Ties retain
        the first source index; separation below 0.02 joint-coordinate units stops expansion.
        This fixed 32-seed/four-branch selection makes no collision claim.
    """
    if (
        qpos.ndim != 2
        or not 0 < qpos.shape[0] <= 32
        or not qpos.shape[1]
        or start.shape != (1, qpos.shape[1])
        or valid.shape != qpos.shape[:1]
        or valid.dtype != torch.bool
        or any(v.device.type != "cpu" for v in (qpos, valid, start))
        or not qpos.is_floating_point()
        or not start.is_floating_point()
        or not torch.isfinite(start).all()
        or not torch.isfinite(qpos[valid]).all()
    ):
        raise ValueError(
            "Separated branch selection requires one finite CPU start and up to 32 qualified joint rows."
        )
    ids = valid.nonzero().flatten().tolist()
    chosen = [min(ids, key=lambda i: float((qpos[i] - start).norm()))] if ids else []
    while len(chosen) < min(4, len(ids)):
        remaining = [i for i in ids if i not in chosen]
        index = max(
            remaining, key=lambda i: float((qpos[chosen] - qpos[i]).norm(dim=-1).min())
        )
        if float((qpos[chosen] - qpos[index]).norm(dim=-1).min()) < 0.02:
            break
        chosen.append(index)
    return tuple(chosen)
