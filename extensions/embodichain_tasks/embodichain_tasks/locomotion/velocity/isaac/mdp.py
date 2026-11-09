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

"""Tensor computations for the Isaac Lab 3.0 EA velocity task definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .config import IsaacVelocityConfig

__all__ = [
    "IsaacVelocityState",
    "IsaacVelocityReward",
    "IsaacVelocityConstants",
    "action_target",
    "observation_terms",
    "build_observation",
    "compute_rewards",
    "compute_termination",
    "dc_motor_effort",
    "update_contact_times",
]


@dataclass
class IsaacVelocityState:
    """Measured tensors in policy joint order and XYZW quaternion convention."""

    base_lin_vel_b: torch.Tensor
    base_ang_vel_b: torch.Tensor
    projected_gravity_b: torch.Tensor
    base_lin_vel_yaw: torch.Tensor
    base_ang_vel_w: torch.Tensor
    command: torch.Tensor
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    joint_acc: torch.Tensor
    applied_torque: torch.Tensor
    action: torch.Tensor
    previous_action: torch.Tensor
    foot_vel_w: torch.Tensor
    foot_force_history: torch.Tensor
    current_air_time: torch.Tensor
    current_contact_time: torch.Tensor
    last_air_time: torch.Tensor
    illegal_force_history: torch.Tensor
    height_scan: torch.Tensor
    episode_step: torch.Tensor
    root_height: torch.Tensor


@dataclass(frozen=True)
class IsaacVelocityReward:
    """Reward values before and after weight and control-period multiplication."""

    raw_terms: dict[str, torch.Tensor]
    weighted_terms: dict[str, torch.Tensor]
    total: torch.Tensor


@dataclass(frozen=True)
class IsaacVelocityConstants:
    """Device tensors for the fixed task configuration."""

    default_joint_position: torch.Tensor
    action_scale: torch.Tensor
    joint_limits: torch.Tensor
    observation_scales: dict[str, torch.Tensor]
    reward_joint_ids: dict[str, torch.Tensor | None]

    @classmethod
    def from_config(
        cls, config: IsaacVelocityConfig, reference: torch.Tensor
    ) -> IsaacVelocityConstants:
        """Materialize constants once on the environment device and dtype."""
        return cls(
            reference.new_tensor(config.default_joint_position),
            reference.new_tensor(config.action_scale),
            reference.new_tensor(config.data["robot"]["soft_joint_pos_limits"]),
            {
                name: reference.new_tensor(spec["scale"])
                for name, spec in config.data["observations"].items()
                if spec["scale"] is not None
            },
            {
                name: (
                    None
                    if spec["joint_ids"] is None
                    else torch.tensor(
                        spec["joint_ids"], device=reference.device, dtype=torch.long
                    )
                )
                for name, spec in config.data["rewards"].items()
            },
        )


def action_target(
    config: IsaacVelocityConfig,
    action: torch.Tensor,
    *,
    constants: IsaacVelocityConstants | None = None,
) -> torch.Tensor:
    """Apply the reference default joint offset and scale without clipping."""
    scale = (
        action.new_tensor(config.action_scale)
        if constants is None
        else constants.action_scale
    )
    default = (
        action.new_tensor(config.default_joint_position)
        if constants is None
        else constants.default_joint_position
    )
    return action * scale + default


def observation_terms(
    config: IsaacVelocityConfig,
    state: IsaacVelocityState,
    *,
    constants: IsaacVelocityConstants | None = None,
) -> dict[str, torch.Tensor]:
    """Return reference policy terms in their declared order before noise and clipping."""
    default = (
        state.joint_pos.new_tensor(config.default_joint_position)
        if constants is None
        else constants.default_joint_position
    )
    terms = {
        "base_lin_vel": state.base_lin_vel_b,
        "base_ang_vel": state.base_ang_vel_b,
        "projected_gravity": state.projected_gravity_b,
        "velocity_commands": state.command,
        "joint_pos": state.joint_pos - default,
        "joint_vel": state.joint_vel,
        "actions": state.action,
    }
    if config.action_dim == 37:
        terms["height_scan"] = state.height_scan
    return terms


def build_observation(
    config: IsaacVelocityConfig,
    state: IsaacVelocityState,
    *,
    noise: dict[str, torch.Tensor] | None = None,
    constants: IsaacVelocityConstants | None = None,
) -> torch.Tensor:
    """Process term noise, clipping and scaling in reference order.

    Args:
        config: Fixed reference task definition.
        state: Measured state tensors.
        noise: Optional realized additive noise per term for cross-stack replay.
        constants: Optional device tensors for the fixed task configuration.

    Returns:
        The shared actor and critic policy observation.
    """
    values = []
    for name, term in observation_terms(config, state, constants=constants).items():
        spec = config.data["observations"][name]
        if noise is not None and name in noise:
            term = term + noise[name]
        if spec["clip"] is not None:
            term = term.clamp(*spec["clip"])
        if spec["scale"] is not None:
            term = term * (
                term.new_tensor(spec["scale"])
                if constants is None
                else constants.observation_scales[name]
            )
        values.append(term)
    return torch.cat(values, dim=-1)


def compute_termination(
    config: IsaacVelocityConfig, state: IsaacVelocityState
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return illegal body contact and the independent time limit."""
    terminated = state.illegal_force_history.norm(dim=-1).amax(dim=(1, 2)) > 1.0
    return terminated, state.episode_step >= config.max_episode_steps


def compute_rewards(
    config: IsaacVelocityConfig,
    state: IsaacVelocityState,
    terminated: torch.Tensor,
    *,
    constants: IsaacVelocityConstants | None = None,
) -> IsaacVelocityReward:
    """Evaluate active reference terms, including joint selections and step duration."""
    raw = {}
    default = (
        state.joint_pos.new_tensor(config.default_joint_position)
        if constants is None
        else constants.default_joint_position
    )
    limits = (
        state.joint_pos.new_tensor(config.data["robot"]["soft_joint_pos_limits"])
        if constants is None
        else constants.joint_limits
    )
    for name, spec in config.data["rewards"].items():
        function = spec["function"]
        params = spec["params"]
        joints = (
            spec["joint_ids"] if constants is None else constants.reward_joint_ids[name]
        )
        if function.startswith("track_lin_vel"):
            velocity = (
                state.base_lin_vel_yaw
                if "yaw_frame" in function
                else state.base_lin_vel_b
            )
            value = torch.exp(
                -(state.command[:, :2] - velocity[:, :2]).square().sum(-1)
                / params["std"] ** 2
            )
        elif function.startswith("track_ang_vel"):
            velocity = (
                state.base_ang_vel_w if "world" in function else state.base_ang_vel_b
            )
            value = torch.exp(
                -(state.command[:, 2] - velocity[:, 2]).square() / params["std"] ** 2
            )
        elif function == "lin_vel_z_l2":
            value = state.base_lin_vel_b[:, 2].square()
        elif function == "ang_vel_xy_l2":
            value = state.base_ang_vel_b[:, :2].square().sum(-1)
        elif function == "joint_torques_l2":
            value = state.applied_torque[:, joints].square().sum(-1)
        elif function == "joint_acc_l2":
            value = state.joint_acc[:, joints].square().sum(-1)
        elif function == "action_rate_l2":
            value = (state.action - state.previous_action).square().sum(-1)
        elif function == "feet_air_time":
            first = (state.current_contact_time > 0.0) & (
                state.current_contact_time < config.control_dt + config.physics_dt * 0.5
            )
            value = ((state.last_air_time - params["threshold"]) * first).sum(-1)
            value = value * (state.command[:, :2].norm(dim=-1) > 0.1)
        elif function == "feet_air_time_positive_biped":
            contact = state.current_contact_time > 0.0
            mode_time = torch.where(
                contact, state.current_contact_time, state.current_air_time
            )
            single = contact.sum(-1) == 1
            value = (
                torch.where(single[:, None], mode_time, 0.0)
                .amin(-1)
                .clamp(max=params["threshold"])
            )
            value = value * (state.command[:, :2].norm(dim=-1) > 0.1)
        elif function == "flat_orientation_l2":
            value = state.projected_gravity_b[:, :2].square().sum(-1)
        elif function == "joint_pos_limits":
            value = (
                (limits[:, 0] - state.joint_pos).clamp(min=0.0)
                + (state.joint_pos - limits[:, 1]).clamp(min=0.0)
            )[:, joints].sum(-1)
        elif function == "is_terminated":
            value = terminated.to(state.joint_pos.dtype)
        elif function == "feet_slide":
            contacts = state.foot_force_history.norm(dim=-1).amax(dim=1) > 1.0
            value = (state.foot_vel_w[:, :, :2].norm(dim=-1) * contacts).sum(-1)
        elif function == "joint_deviation_l1":
            value = (state.joint_pos - default).abs()[:, joints].sum(-1)
        else:
            raise ValueError(f"Unsupported reference reward function: {function}")
        raw[name] = value
    weighted = {
        name: value * config.data["rewards"][name]["weight"] * config.control_dt
        for name, value in raw.items()
    }
    return IsaacVelocityReward(
        raw, weighted, torch.stack(tuple(weighted.values())).sum(0)
    )


def dc_motor_effort(
    desired_effort: torch.Tensor,
    velocity: torch.Tensor,
    effort_limit: torch.Tensor,
    velocity_limit: torch.Tensor,
    saturation_effort: torch.Tensor,
) -> torch.Tensor:
    """Apply the reference DCMotor torque-speed envelope, including overspeed braking."""
    maximum_velocity = velocity_limit * (1.0 + effort_limit / saturation_effort)
    velocity = velocity.clamp(-maximum_velocity, maximum_velocity)
    maximum = torch.minimum(
        saturation_effort * (1.0 - velocity / velocity_limit), effort_limit
    )
    minimum = torch.maximum(
        saturation_effort * (-1.0 - velocity / velocity_limit), -effort_limit
    )
    return desired_effort.clamp(minimum, maximum)


def update_contact_times(
    force: torch.Tensor,
    elapsed: float,
    current_air: torch.Tensor,
    current_contact: torch.Tensor,
    last_air: torch.Tensor,
    last_contact: torch.Tensor,
) -> None:
    """Update reference air and contact timers from summed normal forces in place."""
    contact = force.norm(dim=-1) > 0.0
    last_air.copy_(
        torch.where(contact & (current_air > 0.0), current_air + elapsed, last_air)
    )
    last_contact.copy_(
        torch.where(
            ~contact & (current_contact > 0.0), current_contact + elapsed, last_contact
        )
    )
    current_air.copy_(torch.where(contact, 0.0, current_air + elapsed))
    current_contact.copy_(torch.where(contact, current_contact + elapsed, 0.0))
