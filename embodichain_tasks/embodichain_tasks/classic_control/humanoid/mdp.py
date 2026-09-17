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

"""Backend-independent Humanoid observation and reward functions."""

from __future__ import annotations

import torch

__all__ = ["humanoid_observation", "humanoid_reward"]


def humanoid_observation(
    torso_height: torch.Tensor,
    linear_velocity_local: torch.Tensor,
    angular_velocity_local: torch.Tensor,
    yaw: torch.Tensor,
    roll: torch.Tensor,
    angle_to_target: torch.Tensor,
    up_projection: torch.Tensor,
    heading_projection: torch.Tensor,
    scaled_joint_position: torch.Tensor,
    joint_velocity: torch.Tensor,
    action: torch.Tensor,
    *,
    angular_velocity_scale: float,
    joint_velocity_scale: float,
) -> torch.Tensor:
    """Build the 63-element observation for the 17-DoF Humanoid asset.

    Args:
        torso_height: Root height for each environment.
        linear_velocity_local: Root linear velocity in the torso frame.
        angular_velocity_local: Root angular velocity in the torso frame.
        yaw: Root yaw angles in radians.
        roll: Root roll angles in radians.
        angle_to_target: Heading error to the target, in radians.
        up_projection: Torso up-axis projection on world up.
        heading_projection: Heading-axis projection toward the target.
        scaled_joint_position: Joint positions normalized by their limits.
        joint_velocity: Joint velocities in policy joint order.
        action: Current clipped effort actions in policy joint order.
        angular_velocity_scale: Multiplier for angular velocity observations.
        joint_velocity_scale: Multiplier for joint velocity observations.

    Returns:
        A tensor of shape ``(num_envs, 63)`` in the policy observation order.
    """
    normalize = lambda value: torch.atan2(torch.sin(value), torch.cos(value))
    return torch.cat(
        (
            torso_height.unsqueeze(-1),
            linear_velocity_local,
            angular_velocity_local * angular_velocity_scale,
            normalize(yaw).unsqueeze(-1),
            normalize(roll).unsqueeze(-1),
            normalize(angle_to_target).unsqueeze(-1),
            up_projection.unsqueeze(-1),
            heading_projection.unsqueeze(-1),
            scaled_joint_position,
            joint_velocity * joint_velocity_scale,
            action,
        ),
        dim=-1,
    )


def humanoid_reward(
    *,
    action: torch.Tensor,
    terminated: torch.Tensor,
    heading_projection: torch.Tensor,
    up_projection: torch.Tensor,
    joint_velocity: torch.Tensor,
    scaled_joint_position: torch.Tensor,
    progress: torch.Tensor,
    motor_effort_ratio: torch.Tensor,
    heading_weight: float,
    up_weight: float,
    actions_cost_scale: float,
    energy_cost_scale: float,
    joint_velocity_scale: float,
    death_cost: float,
    alive_reward_scale: float,
) -> torch.Tensor:
    """Compute the 17-DoF Humanoid reward.

    Args:
        action: Current clipped effort actions.
        terminated: Per-environment failure mask.
        heading_projection: Heading-axis projection toward the target.
        up_projection: Torso up-axis projection on world up.
        joint_velocity: Measured joint velocities.
        scaled_joint_position: Joint positions normalized by their limits.
        progress: Change in the task's progress potential.
        motor_effort_ratio: Per-joint relative motor effort weights.
        heading_weight: Maximum heading reward.
        up_weight: Reward for the upright threshold.
        actions_cost_scale: Weight of squared action cost.
        energy_cost_scale: Weight of action-velocity effort cost.
        joint_velocity_scale: Joint velocity multiplier in effort cost.
        death_cost: Reward assigned to terminated environments.
        alive_reward_scale: Additive survival reward.

    Returns:
        One reward per environment, with death cost replacing failed-row totals.
    """
    heading_reward = torch.where(
        heading_projection > 0.8,
        torch.full_like(heading_projection, heading_weight),
        heading_weight * heading_projection / 0.8,
    )
    up_reward = torch.where(
        up_projection > 0.93,
        torch.full_like(up_projection, up_weight),
        torch.zeros_like(up_projection),
    )
    action_cost = action.square().sum(dim=-1)
    electricity_cost = (
        (action * joint_velocity * joint_velocity_scale).abs()
        * motor_effort_ratio.unsqueeze(0)
    ).sum(dim=-1)
    joint_limit_cost = (scaled_joint_position > 0.98).sum(dim=-1)
    reward = (
        progress
        + alive_reward_scale
        + up_reward
        + heading_reward
        - actions_cost_scale * action_cost
        - energy_cost_scale * electricity_cost
        - joint_limit_cost
    )
    return torch.where(terminated, torch.full_like(reward, death_cost), reward)
