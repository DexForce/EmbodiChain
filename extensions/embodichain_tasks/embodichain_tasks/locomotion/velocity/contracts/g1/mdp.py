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

"""Observation, action, reward and done functions for G1 velocity."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .._math import constant, phase_signal
from .._reward_terms import reward_terms

from .config import G1VelocityConfig

__all__ = [
    "G1Reward",
    "G1State",
    "action_target",
    "build_observations",
    "compute_rewards",
    "compute_termination",
]


@dataclass
class G1State:
    """Physical tensors consumed only by the G1 velocity task."""

    base_lin_vel_b: torch.Tensor
    base_ang_vel_b: torch.Tensor
    projected_gravity_b: torch.Tensor
    command: torch.Tensor
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    joint_acc: torch.Tensor
    action: torch.Tensor
    last_action: torch.Tensor
    episode_step: torch.Tensor
    root_height: torch.Tensor
    foot_height: torch.Tensor
    foot_vel_w: torch.Tensor
    foot_contact: torch.Tensor
    foot_force_w: torch.Tensor
    foot_air_time: torch.Tensor
    first_foot_contact: torch.Tensor
    soft_joint_lower: torch.Tensor
    soft_joint_upper: torch.Tensor
    reward_body_ang_vel_w: torch.Tensor
    angular_momentum_w: torch.Tensor
    self_collision_count: torch.Tensor
    reward_command: torch.Tensor | None = None
    reward_base_lin_vel_b: torch.Tensor | None = None
    reward_base_ang_vel_b: torch.Tensor | None = None
    orientation_projected_gravity_b: torch.Tensor | None = None


@dataclass(frozen=True)
class G1Reward:
    """Raw, weighted and total G1 reward values."""

    raw_terms: dict[str, torch.Tensor]
    weighted_terms: dict[str, torch.Tensor]
    total: torch.Tensor


def action_target(config: G1VelocityConfig, action: torch.Tensor) -> torch.Tensor:
    """Map G1 policy actions to default-offset joint targets.

    Args:
        config: Task dimensions, timing, and robot-specific reward settings.
        action: Policy actions with one row per environment and one column per controlled joint.

    Returns:
        Default-offset joint position targets in policy joint order.
    """
    if action.shape[-1] != config.action_dim:
        raise ValueError(
            f"G1 expected {config.action_dim} actions, got {action.shape[-1]}"
        )
    return constant(config.default_joint_position, action) + action * constant(
        config.action_scale, action
    )


def build_observations(
    config: G1VelocityConfig, state: G1State
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build G1 actor and critic observations without random noise.

    Args:
        config: Task dimensions, timing, and robot-specific reward settings.
        state: Batched physical state in the task schema and its declared coordinate frames.

    Returns:
        Actor and privileged critic tensors, in that order.
    """
    relative_position = state.joint_pos - constant(
        config.default_joint_position, state.joint_pos
    )
    phase = phase_signal(
        state.episode_step,
        config.control_dt,
        config.phase_period,
        state.command,
        zero_when_standing=True,
    )
    actor = torch.cat(
        (
            state.base_ang_vel_b,
            state.projected_gravity_b,
            state.command,
            phase,
            relative_position,
            state.joint_vel,
            state.action,
        ),
        dim=-1,
    )
    logged_force = torch.sign(state.foot_force_w) * torch.log1p(
        torch.abs(state.foot_force_w)
    )
    critic = torch.cat(
        (
            actor,
            state.base_lin_vel_b,
            state.foot_height,
            state.foot_air_time,
            state.foot_contact.to(actor.dtype),
            logged_force.flatten(start_dim=1),
        ),
        dim=-1,
    )
    if actor.shape[-1] != config.actor_observation_dim:
        raise RuntimeError(
            f"G1 actor observation has {actor.shape[-1]} values, "
            f"expected {config.actor_observation_dim}"
        )
    if critic.shape[-1] != config.critic_observation_dim:
        raise RuntimeError(
            f"G1 critic observation has {critic.shape[-1]} values, "
            f"expected {config.critic_observation_dim}"
        )
    return actor, critic


def compute_rewards(
    config: G1VelocityConfig, state: G1State, terminated: torch.Tensor
) -> G1Reward:
    """Compute the official G1 reward terms and dt-scaled total.

    Args:
        config: Task dimensions, timing, and robot-specific reward settings.
        state: Batched physical state in the task schema and its declared coordinate frames.
        terminated: Boolean failure mask for each environment before time-limit truncation.

    Returns:
        Raw terms, weighted terms, and the summed reward for each environment.
    """
    raw = reward_terms(
        config.data,
        config.joint_names,
        config.default_joint_position,
        config.control_dt,
        state,
        terminated,
    )
    weights = {name: item["weight"] for name, item in config.data["rewards"].items()}
    weighted = {
        name: value * float(weights[name]) * config.control_dt
        for name, value in raw.items()
    }
    total = torch.stack(tuple(weighted.values()), dim=0).sum(dim=0)
    return G1Reward(raw_terms=raw, weighted_terms=weighted, total=total)


def compute_termination(
    config: G1VelocityConfig, state: G1State
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute G1 tilt termination and time-limit truncation.

    Args:
        config: Task dimensions, timing, and robot-specific reward settings.
        state: Batched physical state in the task schema and its declared coordinate frames.

    Returns:
        Failure and time-limit masks, respectively, with one value per environment.
    """
    finite = (
        torch.isfinite(state.root_height)
        & torch.isfinite(state.joint_pos).all(dim=-1)
        & torch.isfinite(state.joint_vel).all(dim=-1)
        & torch.isfinite(state.base_lin_vel_b).all(dim=-1)
        & torch.isfinite(state.base_ang_vel_b).all(dim=-1)
    )
    tilt = torch.acos(torch.clamp(-state.projected_gravity_b[:, 2], -1.0, 1.0))
    terminated = ~finite | (tilt > math.radians(70.0))
    truncated = state.episode_step >= config.max_episode_steps
    return terminated, truncated
