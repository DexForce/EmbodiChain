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

"""Bilateral contact classification from explicit actor and numeric inputs."""

from __future__ import annotations

import torch

__all__ = ["classify_contacts"]


def classify_contacts(
    pairs: torch.Tensor,
    normals: torch.Tensor,
    impulses: torch.Tensor,
    *,
    target: int,
    pads: tuple[int, int],
) -> tuple[bool, bool]:
    """Distinguish opposed bilateral contact, clear release, and unresolved contact.

    Args:
        pairs: Native actor pairs, shaped (contacts, 2).
        normals: Arena-frame actor-0-to-actor-1 normals, shaped (contacts, 3).
        impulses: Native scalar normal impulses, shaped (contacts,).
        target: Exact object actor ID for this environment.
        pads: Exact two pad actor IDs for this environment.

    Returns:
        Measured bilateral contact and whether this positive or negative
        observation is available. Availability is separate from a task verdict.
    """
    if not all(torch.isfinite(v).all() for v in (normals, impulses)):
        return False, False
    sides = []
    touching = False
    for pad in pads:
        selected = (
            ((pairs[:, 0] == target) & (pairs[:, 1] == pad))
            | ((pairs[:, 1] == target) & (pairs[:, 0] == pad))
        ) & (impulses > 0)
        touching |= bool(selected.any())
        outward = torch.where(
            (pairs[selected, 0] == target)[:, None],
            normals[selected],
            -normals[selected],
        )
        lengths = torch.linalg.vector_norm(outward, dim=-1)
        if bool((lengths <= 1e-8).any()):
            return False, False
        sides.append(outward / lengths[:, None])
    if (
        sides[0].numel()
        and sides[1].numel()
        and bool((sides[0] @ sides[1].T <= -0.5).any())
    ):
        return True, True
    return False, not touching
