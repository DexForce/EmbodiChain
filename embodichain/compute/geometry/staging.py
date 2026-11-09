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

"""Ordered distal-envelope translations used as staging hints."""

from __future__ import annotations
from collections.abc import Mapping
from typing import Any
import numpy as np
import torch
from .pose_checks import as_rigid_matrix as _pose

__all__ = ["distal_staging_variants"]


def distal_staging_variants(
    targets: torch.Tensor,
    object_pose: torch.Tensor,
    attachment: np.ndarray,
    *,
    active: str,
    passive: str,
    tcp: Mapping[str, np.ndarray],
    roots: Mapping[str, np.ndarray],
    points: Mapping[str, np.ndarray],
    hand_owner_fk_error: Mapping[str, float],
) -> list[tuple[str, torch.Tensor | None, dict[str, Any]]]:
    """Build the original ordered TCP/root-anchored outward translations.

    Args:
        targets: Original two staging poses, shape ``(1,2,4,4)``.
        object_pose: Actual object pose, shape ``(1,4,4)``.
        attachment: Object-to-tool transform, shape ``(4,4)``.
        active: Active numeric-map key.
        passive: Passive numeric-map key.
        tcp: World tool frames for both keys.
        roots: World root frames for both keys.
        points: Distal collision-input world points for both keys.
        hand_owner_fk_error: Observed/model discrepancy diagnostics to retain.

    Returns:
        Ordered B/C tuples containing optional three-pose targets and numeric
        reports. A bound failure returns None for that candidate; no collision,
        kinematic or physical success is inferred.
    """
    variants = []
    for label, anchors in (("B", tcp), ("C", roots)):
        report = {
            "variant": label,
            "scope": "measured_distal_mesh_hint_only",
            "hand_owner_fk_error": dict(hand_owner_fk_error),
        }
        direction = anchors[active][:2, 3] - anchors[passive][:2, 3]
        length = float(np.linalg.norm(direction))
        if not np.isfinite(length) or length <= np.finfo(np.float32).eps:
            variants.append(
                (label, None, {**report, "reason": "degenerate_xy_direction"})
            )
            continue
        direction = direction / length
        active_projection = points[active][:, :2] @ direction
        passive_projection = points[passive][:, :2] @ direction
        active_span, passive_span = float(np.ptp(active_projection)), float(
            np.ptp(passive_projection)
        )
        stage_delta = (_pose(targets[:, 0]) @ attachment)[:2, 3] - tcp[active][:2, 3]
        required = max(
            float(passive_projection.max())
            - float(active_projection.min())
            - min(float(stage_delta @ direction), 0),
            0,
        )
        bound = active_span + passive_span
        shift = max(required + 0.0002, active_span / 2)
        if min(active_span, passive_span) <= np.finfo(np.float32).eps or shift > bound:
            variants.append(
                (label, None, {**report, "reason": "distal_mesh_bound_exceeded"})
            )
            continue
        offset = targets.new_tensor([direction[0] * shift, direction[1] * shift, 0.0])
        outward, elevated = object_pose.clone(), targets[:, 0].clone()
        outward[:, :3, 3] += offset
        elevated[:, :3, 3] += offset
        poses = torch.stack((outward, elevated, targets[:, -1]), dim=1)
        variants.append(
            (
                label,
                poses,
                {
                    **report,
                    "direction_xy": direction.tolist(),
                    "shift_m": shift,
                    "shift_bound_m": bound,
                },
            )
        )
    return variants
