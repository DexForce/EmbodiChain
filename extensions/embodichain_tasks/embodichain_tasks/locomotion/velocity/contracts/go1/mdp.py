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

"""Official MJLab Go1 flat-velocity observation, reward, and done functions."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .._math import (
    command_active,
    constant,
    joint_limit_cost,
    resolve_pattern_values,
)

from .config import Go1VelocityConfig

__all__ = [
    "Go1Reward",
    "Go1State",
    "action_target",
    "build_observations",
    "compute_rewards",
    "compute_termination",
    "corrupt_actor_observation",
    "termination_causes",
]


@dataclass
class Go1State:
    """Physical tensors consumed by the official MJLab Go1 task."""

    base_lin_vel_b: torch.Tensor
    base_ang_vel_b: torch.Tensor
    projected_gravity_b: torch.Tensor
    command: torch.Tensor
    joint_pos: torch.Tensor
    encoder_bias: torch.Tensor
    joint_vel: torch.Tensor
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
    foot_swing_height_cost: torch.Tensor
    soft_joint_lower: torch.Tensor
    soft_joint_upper: torch.Tensor
    illegal_contact_force: torch.Tensor
    illegal_contact_force_by_body: torch.Tensor


@dataclass(frozen=True)
class Go1Reward:
    """Raw, weighted, and total Go1 reward values."""

    raw_terms: dict[str, torch.Tensor]
    weighted_terms: dict[str, torch.Tensor]
    total: torch.Tensor


def action_target(config: Go1VelocityConfig, action: torch.Tensor) -> torch.Tensor:
    """Map actions to official default-offset joint position targets.

    Args:
        config: Task dimensions, timing, and robot-specific reward settings.
        action: Policy actions with one row per environment and one column per controlled joint.

    Returns:
        Default-offset joint position targets in policy joint order.
    """
    if action.shape[-1] != config.action_dim:
        raise ValueError(
            f"Go1 expected {config.action_dim} actions, got {action.shape[-1]}"
        )
    return constant(config.default_joint_position, action) + action * constant(
        config.action_scale, action
    )


def build_observations(
    config: Go1VelocityConfig, state: Go1State
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the official 48-value actor and 72-value asymmetric critic inputs.

    Args:
        config: Task dimensions, timing, and robot-specific reward settings.
        state: Batched physical state in the task schema and its declared coordinate frames.

    Returns:
        Actor and privileged critic tensors, in that order.
    """
    actor_relative_position = (
        state.joint_pos
        + state.encoder_bias
        - constant(config.default_joint_position, state.joint_pos)
    )
    actor = torch.cat(
        (
            state.base_lin_vel_b,
            state.base_ang_vel_b,
            state.projected_gravity_b,
            actor_relative_position,
            state.joint_vel,
            state.action,
            state.command,
        ),
        dim=-1,
    )
    critic_base = torch.cat(
        (
            state.base_lin_vel_b,
            state.base_ang_vel_b,
            state.projected_gravity_b,
            state.joint_pos - constant(config.default_joint_position, state.joint_pos),
            state.joint_vel,
            state.action,
            state.command,
        ),
        dim=-1,
    )
    logged_force = torch.sign(state.foot_force_w) * torch.log1p(
        torch.abs(state.foot_force_w)
    )
    critic = torch.cat(
        (
            critic_base,
            state.foot_height,
            state.foot_air_time,
            state.foot_contact.to(actor.dtype),
            logged_force.flatten(start_dim=1),
        ),
        dim=-1,
    )
    if actor.shape[-1] != config.actor_observation_dim:
        raise RuntimeError(
            f"Go1 actor observation has {actor.shape[-1]} values, "
            f"expected {config.actor_observation_dim}"
        )
    if critic.shape[-1] != config.critic_observation_dim:
        raise RuntimeError(
            f"Go1 critic observation has {critic.shape[-1]} values, "
            f"expected {config.critic_observation_dim}"
        )
    return actor, critic


def corrupt_actor_observation(
    config: Go1VelocityConfig,
    actor: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    """Apply the official uniform observation noise in place.

    Args:
        config: Task dimensions, timing, and robot-specific reward settings.
        actor: Actor observation tensor to corrupt in place.
        generator: Random generator used to sample the task noise or delays.

    Returns:
        The actor observation tensor after adding the configured noise.
    """
    terms = config.data["observations"]["actor"]["terms"]
    slices = {
        "base_lin_vel": slice(0, 3),
        "base_ang_vel": slice(3, 6),
        "projected_gravity": slice(6, 9),
        "joint_pos": slice(9, 21),
        "joint_vel": slice(21, 33),
    }
    for name, target_slice in slices.items():
        noise = terms[name]["noise"]
        if noise is None:
            continue
        actor[:, target_slice].add_(
            torch.empty_like(actor[:, target_slice]).uniform_(
                float(noise["n_min"]),
                float(noise["n_max"]),
                generator=generator,
            )
        )
    return actor


def compute_rewards(
    config: Go1VelocityConfig, state: Go1State, terminated: torch.Tensor
) -> Go1Reward:
    """Compute the official Go1 flat-velocity reward terms.

    Args:
        config: Task dimensions, timing, and robot-specific reward settings.
        state: Batched physical state in the task schema and its declared coordinate frames.
        terminated: Boolean failure mask for each environment before time-limit truncation.

    Returns:
        Raw terms, weighted terms, and the summed reward for each environment.
    """
    del terminated
    rewards = config.data["rewards"]
    linear_error = torch.square(state.command[:, :2] - state.base_lin_vel_b[:, :2]).sum(
        dim=-1
    ) + torch.square(state.base_lin_vel_b[:, 2])
    angular_error = torch.square(
        state.command[:, 2] - state.base_ang_vel_b[:, 2]
    ) + torch.square(state.base_ang_vel_b[:, :2]).sum(dim=-1)

    pose = rewards["pose"]["params"]
    speed = torch.linalg.vector_norm(state.command[:, :2], dim=-1)
    speed += state.command[:, 2].abs()
    standing = speed < float(pose["walking_threshold"])
    running = speed >= float(pose["running_threshold"])
    standing_std = constant(
        tuple(resolve_pattern_values(pose["std_standing"], config.joint_names)),
        state.joint_pos,
    )
    walking_std = constant(
        tuple(resolve_pattern_values(pose["std_walking"], config.joint_names)),
        state.joint_pos,
    )
    running_std = constant(
        tuple(resolve_pattern_values(pose["std_running"], config.joint_names)),
        state.joint_pos,
    )
    pose_std = torch.where(
        standing.unsqueeze(-1),
        standing_std,
        torch.where(running.unsqueeze(-1), running_std, walking_std),
    )
    pose_error = torch.square(
        state.joint_pos - constant(config.default_joint_position, state.joint_pos)
    )

    clearance = rewards["foot_clearance"]["params"]
    clearance_active = command_active(
        state.command, float(clearance["command_threshold"])
    )
    slip = rewards["foot_slip"]["params"]
    slip_active = command_active(state.command, float(slip["command_threshold"]))
    landing = rewards["soft_landing"]["params"]
    landing_active = command_active(state.command, float(landing["command_threshold"]))
    swing = rewards["foot_swing_height"]["params"]
    swing_active = command_active(state.command, float(swing["command_threshold"]))

    linear_std = float(rewards["track_linear_velocity"]["params"]["std"])
    angular_std = float(rewards["track_angular_velocity"]["params"]["std"])
    upright_std = float(rewards["upright"]["params"]["std"])
    raw = {
        "track_linear_velocity": torch.exp(-linear_error / linear_std**2),
        "track_angular_velocity": torch.exp(-angular_error / angular_std**2),
        "upright": torch.exp(
            -torch.square(state.projected_gravity_b[:, :2]).sum(dim=-1) / upright_std**2
        ),
        "pose": torch.exp(-torch.mean(pose_error / torch.square(pose_std), dim=-1)),
        "dof_pos_limits": joint_limit_cost(
            state.joint_pos, state.soft_joint_lower, state.soft_joint_upper
        ),
        "action_rate_l2": torch.square(state.action - state.last_action).sum(dim=-1),
        "foot_clearance": (
            (
                torch.abs(state.foot_height - float(clearance["target_height"]))
                * torch.linalg.vector_norm(state.foot_vel_w[:, :, :2], dim=-1)
            ).sum(dim=-1)
            * clearance_active
        ),
        "foot_swing_height": state.foot_swing_height_cost.sum(dim=-1) * swing_active,
        "foot_slip": (
            (
                torch.square(
                    torch.linalg.vector_norm(state.foot_vel_w[:, :, :2], dim=-1)
                )
                * state.foot_contact
            ).sum(dim=-1)
            * slip_active
        ),
        "soft_landing": (
            (
                torch.linalg.vector_norm(state.foot_force_w, dim=-1)
                * state.first_foot_contact
            ).sum(dim=-1)
            * landing_active
        ),
    }
    weighted = {
        name: value * float(rewards[name]["weight"]) * config.control_dt
        for name, value in raw.items()
    }
    total = torch.stack(tuple(weighted.values()), dim=0).sum(dim=0)
    return Go1Reward(raw_terms=raw, weighted_terms=weighted, total=total)


def termination_causes(
    config: Go1VelocityConfig, state: Go1State
) -> dict[str, torch.Tensor]:
    """Return the official flat-terrain 70-degree tilt termination.

    Args:
        config: Task dimensions, timing, and robot-specific reward settings.
        state: Batched physical state in the task schema and its declared coordinate frames.

    Returns:
        Named boolean failure masks with one value per environment.
    """
    tilt = torch.acos(torch.clamp(-state.projected_gravity_b[:, 2], min=-1.0, max=1.0))
    return {
        "fell_over": tilt
        > float(config.data["terminations"]["fell_over"]["params"]["limit_angle"]),
    }


def compute_termination(
    config: Go1VelocityConfig, state: Go1State
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute failure termination and time-limit truncation.

    Args:
        config: Task dimensions, timing, and robot-specific reward settings.
        state: Batched physical state in the task schema and its declared coordinate frames.

    Returns:
        Failure and time-limit masks, respectively, with one value per environment.
    """
    causes = termination_causes(config, state)
    terminated = torch.stack(tuple(causes.values()), dim=0).any(dim=0)
    truncated = state.episode_step >= config.max_episode_steps
    return terminated, truncated
