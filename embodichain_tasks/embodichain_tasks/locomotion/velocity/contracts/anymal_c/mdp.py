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

"""Observation, action, reward, and termination functions for ANYmal-C."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .._math import constant

from .config import ANYmalCVelocityConfig

__all__ = [
    "ANYmalCReward",
    "ANYmalCState",
    "action_target",
    "build_observations",
    "compute_rewards",
    "compute_termination",
]


@dataclass
class ANYmalCState:
    """Physical tensors consumed by the ANYmal-C flat velocity task."""

    base_lin_vel_b: torch.Tensor
    base_ang_vel_b: torch.Tensor
    projected_gravity_b: torch.Tensor
    command: torch.Tensor
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    joint_acc: torch.Tensor
    joint_torque: torch.Tensor
    action: torch.Tensor
    last_action: torch.Tensor
    episode_step: torch.Tensor
    root_height: torch.Tensor
    foot_air_time: torch.Tensor
    first_foot_contact: torch.Tensor
    illegal_contact_force_by_body: torch.Tensor


@dataclass(frozen=True)
class ANYmalCReward:
    """Raw, weighted, and total ANYmal-C rewards."""

    raw_terms: dict[str, torch.Tensor]
    weighted_terms: dict[str, torch.Tensor]
    total: torch.Tensor


def action_target(
    config: ANYmalCVelocityConfig,
    action: torch.Tensor,
) -> torch.Tensor:
    """Map normalized actions to default-offset joint targets.

    Args:
        config: Task joint, observation, reward and timing settings.
        action: Normalized policy actions in configured joint order.

    Returns:
        Joint-position targets with the configured default pose and action scale.
    """
    if action.shape[-1] != config.action_dim:
        raise ValueError(
            f"ANYmal-C expected {config.action_dim} actions, got {action.shape[-1]}"
        )
    return constant(config.default_joint_position, action) + action * constant(
        config.action_scale, action
    )


def build_observations(
    config: ANYmalCVelocityConfig,
    state: ANYmalCState,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the 48-value ANYmal-C flat policy observation.

    Args:
        config: Task joint, observation, reward and timing settings.
        state: Batched task state in the configured joint and body order.

    Returns:
        Actor and critic observation tensors, each with one row per environment.
    """
    joint_position = state.joint_pos - constant(
        config.default_joint_position, state.joint_pos
    )
    actor = torch.cat(
        (
            state.base_lin_vel_b,
            state.base_ang_vel_b,
            state.projected_gravity_b,
            state.command,
            joint_position,
            state.joint_vel,
            state.action,
        ),
        dim=-1,
    )
    if actor.shape[-1] != config.actor_observation_dim:
        raise RuntimeError(
            f"ANYmal-C actor observation has {actor.shape[-1]} values, "
            f"expected {config.actor_observation_dim}"
        )
    if actor.shape[-1] != config.critic_observation_dim:
        raise RuntimeError(
            f"ANYmal-C critic observation has {actor.shape[-1]} values, "
            f"expected {config.critic_observation_dim}"
        )
    return actor, actor


def _reward_terms(
    config: ANYmalCVelocityConfig,
    state: ANYmalCState,
) -> dict[str, torch.Tensor]:
    rewards = config.data["rewards"]
    linear_error = torch.square(state.command[:, :2] - state.base_lin_vel_b[:, :2]).sum(
        dim=-1
    )
    yaw_error = torch.square(state.command[:, 2] - state.base_ang_vel_b[:, 2])
    contact_threshold = float(rewards["undesired_contacts"]["threshold"])
    thigh_contacts = (
        state.illegal_contact_force_by_body[:, 1:] > contact_threshold
    ).to(state.joint_pos.dtype)
    moving = torch.linalg.vector_norm(state.command[:, :2], dim=-1) > 0.1
    air_time = (
        (state.foot_air_time - float(rewards["feet_air_time"]["threshold"]))
        * state.first_foot_contact
    ).sum(dim=-1) * moving
    return {
        "track_lin_vel_xy_exp": torch.exp(
            -linear_error / float(rewards["track_lin_vel_xy_exp"]["std_squared"])
        ),
        "track_ang_vel_z_exp": torch.exp(
            -yaw_error / float(rewards["track_ang_vel_z_exp"]["std_squared"])
        ),
        "lin_vel_z_l2": torch.square(state.base_lin_vel_b[:, 2]),
        "ang_vel_xy_l2": torch.square(state.base_ang_vel_b[:, :2]).sum(dim=-1),
        "dof_torques_l2": torch.square(state.joint_torque).sum(dim=-1),
        "dof_acc_l2": torch.square(state.joint_acc).sum(dim=-1),
        "action_rate_l2": torch.square(state.action - state.last_action).sum(dim=-1),
        "feet_air_time": air_time,
        "undesired_contacts": thigh_contacts.sum(dim=-1),
        "flat_orientation_l2": torch.square(state.projected_gravity_b[:, :2]).sum(
            dim=-1
        ),
    }


def compute_rewards(
    config: ANYmalCVelocityConfig,
    state: ANYmalCState,
    terminated: torch.Tensor,
) -> ANYmalCReward:
    """Compute dt-scaled ANYmal-C flat velocity rewards.

    Args:
        config: Task joint, observation, reward and timing settings.
        state: Batched task state in the configured joint and body order.
        terminated: Failure mask; unused by ANYmal-C reward terms.

    Returns:
        Raw terms, weighted terms scaled by control timestep, and their total.
    """
    del terminated
    raw = _reward_terms(config, state)
    weighted = {
        name: value * float(config.data["rewards"][name]["weight"]) * config.control_dt
        for name, value in raw.items()
    }
    total = torch.stack(tuple(weighted.values()), dim=0).sum(dim=0)
    return ANYmalCReward(raw_terms=raw, weighted_terms=weighted, total=total)


def compute_termination(
    config: ANYmalCVelocityConfig,
    state: ANYmalCState,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Terminate on base contact or non-finite state and truncate on timeout.

    Args:
        config: Task joint, observation, reward and timing settings.
        state: Batched task state in the configured joint and body order.

    Returns:
        Per-environment failure and episode-timeout masks.
    """
    finite = (
        torch.isfinite(state.root_height)
        & torch.isfinite(state.joint_pos).all(dim=-1)
        & torch.isfinite(state.joint_vel).all(dim=-1)
        & torch.isfinite(state.base_lin_vel_b).all(dim=-1)
        & torch.isfinite(state.base_ang_vel_b).all(dim=-1)
    )
    base_contact = state.illegal_contact_force_by_body[:, 0] > float(
        config.data["terminations"]["base_contact"]["threshold"]
    )
    terminated = ~finite | base_contact
    truncated = state.episode_step >= config.max_episode_steps
    return terminated, truncated
