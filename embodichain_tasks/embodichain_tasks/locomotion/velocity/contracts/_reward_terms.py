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

"""Shared math from the official MJLab velocity task family."""

from __future__ import annotations

from typing import Protocol

import torch

from ._math import (
    command_active,
    constant,
    joint_limit_cost,
    required,
    resolve_pattern_values,
)

__all__ = [
    "MjlabVelocityState",
    "corrupt_actor_observation",
    "reward_terms",
]


class MjlabVelocityState(Protocol):
    """Tensor fields required by the shared MJLab reward family."""

    base_lin_vel_b: torch.Tensor
    base_ang_vel_b: torch.Tensor
    projected_gravity_b: torch.Tensor
    command: torch.Tensor
    joint_pos: torch.Tensor
    joint_acc: torch.Tensor
    action: torch.Tensor
    last_action: torch.Tensor
    foot_height: torch.Tensor
    foot_vel_w: torch.Tensor
    foot_contact: torch.Tensor
    foot_force_w: torch.Tensor
    first_foot_contact: torch.Tensor
    soft_joint_lower: torch.Tensor
    soft_joint_upper: torch.Tensor
    episode_step: torch.Tensor
    reward_command: torch.Tensor | None
    reward_base_lin_vel_b: torch.Tensor | None
    reward_base_ang_vel_b: torch.Tensor | None
    orientation_projected_gravity_b: torch.Tensor | None
    reward_body_ang_vel_w: torch.Tensor
    angular_momentum_w: torch.Tensor
    self_collision_count: torch.Tensor | None


def corrupt_actor_observation(
    data: dict,
    action_dim: int,
    actor: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    """Apply the official G1/Go2 actor observation noise in place."""
    expected_dimension = 11 + 3 * action_dim
    if actor.shape[-1] != expected_dimension:
        raise ValueError(
            f"actor observation has {actor.shape[-1]} values, "
            f"expected {expected_dimension}"
        )
    terms = data["observations"]["actor"]["terms"]
    slices = {
        "base_ang_vel": slice(0, 3),
        "projected_gravity": slice(3, 6),
        "joint_pos": slice(11, 11 + action_dim),
        "joint_vel": slice(11 + action_dim, 11 + 2 * action_dim),
    }
    for name, target_slice in slices.items():
        noise = terms[name]["noise"]
        if noise is None:
            continue
        target = actor[:, target_slice]
        target.add_(
            torch.empty_like(target).uniform_(
                float(noise["n_min"]),
                float(noise["n_max"]),
                generator=generator,
            )
        )
    return actor


def reward_terms(
    data: dict,
    joint_names: tuple[str, ...],
    default_joint_position: tuple[float, ...],
    control_dt: float,
    state: MjlabVelocityState,
    terminated: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute raw rewards shared by official G1 and Go2 MJLab tasks."""
    rewards = data["rewards"]
    command = state.command if state.reward_command is None else state.reward_command
    orientation = state.orientation_projected_gravity_b
    if orientation is None:
        orientation = state.projected_gravity_b
    reward_base_lin_vel_b = state.reward_base_lin_vel_b
    if reward_base_lin_vel_b is None:
        reward_base_lin_vel_b = state.base_lin_vel_b
    reward_base_ang_vel_b = state.reward_base_ang_vel_b
    if reward_base_ang_vel_b is None:
        reward_base_ang_vel_b = state.base_ang_vel_b
    lin_std = float(rewards["track_linear_velocity"]["params"]["std"])
    ang_std = float(rewards["track_angular_velocity"]["params"]["std"])
    lin_error = torch.square(command[:, :2] - reward_base_lin_vel_b[:, :2]).sum(
        dim=-1
    ) + 2.0 * torch.square(reward_base_lin_vel_b[:, 2])
    ang_error = torch.square(command[:, 2] - reward_base_ang_vel_b[:, 2]) + 0.05 * (
        torch.square(reward_base_ang_vel_b[:, :2]).sum(dim=-1)
    )

    pose_cfg = rewards["pose"]["params"]
    speed = torch.linalg.vector_norm(command[:, :2], dim=-1) + command[:, 2].abs()
    standing = speed < float(pose_cfg["walking_threshold"])
    running = speed >= float(pose_cfg["running_threshold"])
    standing_std = constant(
        tuple(resolve_pattern_values(pose_cfg["std_standing"], joint_names)),
        state.joint_pos,
    )
    walking_std = constant(
        tuple(resolve_pattern_values(pose_cfg["std_walking"], joint_names)),
        state.joint_pos,
    )
    running_std = constant(
        tuple(resolve_pattern_values(pose_cfg["std_running"], joint_names)),
        state.joint_pos,
    )
    pose_std = torch.where(
        standing.unsqueeze(-1),
        standing_std,
        torch.where(running.unsqueeze(-1), running_std, walking_std),
    )
    default = constant(default_joint_position, state.joint_pos)
    pose_error = torch.square(state.joint_pos - default)

    gait_cfg = rewards["foot_gait"]["params"]
    offsets = state.joint_pos.new_tensor(gait_cfg["offset"])
    global_phase = (
        state.episode_step * control_dt / float(gait_cfg["period"])
    ).unsqueeze(-1)
    expected_contact = (global_phase + offsets).remainder(1.0) < float(
        gait_cfg["threshold"]
    )
    gait_active = command_active(command, float(gait_cfg["command_threshold"]))
    clearance_cfg = rewards["foot_clearance"]["params"]
    slip_cfg = rewards["foot_slip"]["params"]
    landing_cfg = rewards["soft_landing"]["params"]
    stand_cfg = rewards["stand_still"]["params"]

    terms = {
        "track_linear_velocity": torch.exp(-lin_error / lin_std**2),
        "track_angular_velocity": torch.exp(-ang_error / ang_std**2),
        "body_orientation_l2": torch.square(orientation[:, :2]).sum(dim=-1),
        "pose": torch.exp(-torch.mean(pose_error / torch.square(pose_std), dim=-1)),
        "body_ang_vel": torch.square(state.reward_body_ang_vel_w[:, :2]).sum(dim=-1),
        "angular_momentum": torch.square(state.angular_momentum_w).sum(dim=-1),
        "is_terminated": terminated.to(state.joint_pos.dtype),
        "joint_acc_l2": torch.square(state.joint_acc).sum(dim=-1),
        "joint_pos_limits": joint_limit_cost(
            state.joint_pos, state.soft_joint_lower, state.soft_joint_upper
        ),
        "action_rate_l2": torch.square(state.action - state.last_action).sum(dim=-1),
        "foot_gait": (
            (state.foot_contact.bool() == expected_contact)
            .to(state.joint_pos.dtype)
            .mean(dim=-1)
            * gait_active
        ),
        "foot_clearance": (
            (
                torch.abs(state.foot_height - float(clearance_cfg["target_height"]))
                * torch.linalg.vector_norm(state.foot_vel_w[:, :, :2], dim=-1)
            ).sum(dim=-1)
            * command_active(command, float(clearance_cfg["command_threshold"]))
        ),
        "foot_slip": (
            (
                torch.square(
                    torch.linalg.vector_norm(state.foot_vel_w[:, :, :2], dim=-1)
                )
                * state.foot_contact
            ).sum(dim=-1)
            * command_active(command, float(slip_cfg["command_threshold"]))
        ),
        "soft_landing": (
            (
                torch.linalg.vector_norm(state.foot_force_w, dim=-1)
                * state.first_foot_contact
            ).sum(dim=-1)
            * command_active(command, float(landing_cfg["command_threshold"]))
        ),
        "stand_still": pose_error.sum(dim=-1)
        * ~command_active(command, float(stand_cfg["command_threshold"])),
    }
    if "self_collisions" in rewards:
        terms["self_collisions"] = required(
            state.self_collision_count, "self_collision_count"
        )
    return terms
