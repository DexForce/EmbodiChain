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

"""Heading-frame features shared by dynamics models and tracking tasks."""

from __future__ import annotations

import torch

from .matrix import matrix_product, matrix_vector_product

__all__ = ["heading_rotation", "heading_kinematic_features"]


def heading_rotation(rotation: torch.Tensor) -> torch.Tensor:
    """Extract Z-up heading from the horizontal projection of the root X axis.

    Args:
        rotation: Body-to-world matrices with arbitrary leading batch axes.

    Returns:
        Yaw rotation matrices. Heading is undefined when the root X axis is
        vertical; callers use this representation away from that singularity.
    """
    yaw = torch.atan2(rotation[..., 1, 0], rotation[..., 0, 0])
    c, s = yaw.cos(), yaw.sin()
    z, o = torch.zeros_like(c), torch.ones_like(c)
    return torch.stack((c, -s, z, s, c, z, z, z, o), -1).unflatten(-1, (3, 3))


def heading_kinematic_features(
    position: torch.Tensor,
    rotation: torch.Tensor,
    linear_velocity: torch.Tensor,
    angular_velocity: torch.Tensor,
    joint_position: torch.Tensor,
    joint_velocity: torch.Tensor,
    resting_position: torch.Tensor,
    link_position: torch.Tensor,
    link_linear_velocity: torch.Tensor,
) -> torch.Tensor:
    """Encode root/joint state and ordered link origins in the root heading frame.

    Args:
        position: World root-origin positions, ending in 3.
        rotation: Body-to-world root rotations, ending in (3, 3).
        linear_velocity: World root-origin linear velocities.
        angular_velocity: World root angular velocities.
        joint_position: Ordered joint positions, ending in J.
        joint_velocity: Ordered joint velocities.
        resting_position: Joint offsets, ending in J.
        link_position: World positions, ending in (L, 3), usually excluding root.
        link_linear_velocity: World link-origin velocities, ending in (L, 3).

    Returns:
        Features of width ``13 + 2*J + 6*L``: height, heading-relative orientation
        (first two columns), heading linear velocity, body angular velocity,
        centered joints, joint velocities, relative link positions and velocities.
    """
    heading = heading_rotation(rotation)
    local = lambda r, v: matrix_vector_product(r.transpose(-1, -2), v)
    orientation = matrix_product(heading.transpose(-1, -2), rotation)
    return torch.cat(
        (
            position[..., 2:3],
            orientation[..., :, :2].transpose(-1, -2).flatten(-2),
            local(heading, linear_velocity),
            local(rotation, angular_velocity),
            joint_position - resting_position,
            joint_velocity,
            local(
                heading.unsqueeze(-3), link_position - position.unsqueeze(-2)
            ).flatten(-2),
            local(heading.unsqueeze(-3), link_linear_velocity).flatten(-2),
        ),
        -1,
    )
