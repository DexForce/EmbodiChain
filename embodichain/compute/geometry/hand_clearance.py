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

"""Numeric released-hand nominal envelope targets, not collision certificates."""

from __future__ import annotations
import torch

__all__ = ["released_hand_targets", "released_hand_height"]


def released_hand_targets(
    current: torch.Tensor,
    root: torch.Tensor,
    *,
    height: float | torch.Tensor,
    retreat: float,
) -> torch.Tensor:
    """First clear vertically, then move toward the owning arm's root.

    Args:
        current: Observed world TCP transforms, shape ``(B,4,4)``.
        root: Corresponding observed world arm-root transforms.
        height: Minimum world-Z TCP clearance height per row.
        retreat: Explicit horizontal distance in metres.

    Returns:
        Ordered raised and withdrawn TCP targets, shape ``(B,2,4,4)``.
    """
    raised = current.clone()
    raised[:, 2, 3] = torch.maximum(
        raised[:, 2, 3], torch.as_tensor(height).to(current)
    )
    direction = root[:, :2, 3] - current[:, :2, 3]
    direction = torch.nn.functional.normalize(direction, dim=-1)
    withdrawn = raised.clone()
    withdrawn[:, :2, 3] += retreat * direction
    return torch.stack((raised, withdrawn), dim=1)


def released_hand_height(
    current: torch.Tensor,
    object_pose: torch.Tensor,
    vertices: torch.Tensor,
    *,
    opening_width: float,
    finger_length: float,
    finger_width: float,
    finger_thickness: float,
    palm_depth: float,
) -> torch.Tensor:
    """Measure nominal release height without replacing collision qualification.

    Args:
        current: Observed world TCP transforms.
        object_pose: Observed object transforms, preserving environment rows.
        vertices: Scaled object-local geometry vertices.
        opening_width: Declared gripper opening width in metres.
        finger_length: Explicit nominal finger length.
        finger_width: Explicit nominal finger width.
        finger_thickness: Explicit nominal finger thickness.
        palm_depth: Explicit nominal depth behind the finger origin.

    Returns:
        World-Z clearance height with the source 0.01 m reserve.
    """
    half_open = opening_width / 2 + finger_thickness
    corners = current.new_tensor(
        [
            [x, y, z]
            for x in (-half_open, half_open)
            for y in (-finger_width / 2, finger_width / 2)
            for z in (
                -finger_length / 2 - palm_depth,
                finger_length / 2,
            )
        ]
    )
    gripper_low = (corners @ current[:, :3, :3].transpose(-1, -2))[:, :, 2].amin(1)
    object_points = (
        vertices.to(current) @ object_pose[:, :3, :3].transpose(-1, -2)
        + object_pose[:, None, :3, 3]
    )
    return object_points[:, :, 2].amax(1) - gripper_low + 0.01
