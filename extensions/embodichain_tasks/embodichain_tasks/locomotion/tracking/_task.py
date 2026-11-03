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

"""Differentiable tracking calculations shared by physical and model environments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

import torch

from embodichain.compute.kinematics.matrix import matrix_product, matrix_vector_product

from embodichain.compute.kinematics.features import (
    heading_rotation,
    heading_kinematic_features,
)
from embodichain.compute.kinematics.tree import LinkKinematics, TreeKinematics
from embodichain.utils import configclass

__all__ = ["MotionTrackingCfg", "MotionTrackingTask"]
State = Mapping[str, torch.Tensor]


@configclass
class MotionTrackingCfg:
    """Tracking objective, history and termination parameters."""

    control_dt: float = 0.02
    history_steps: int = 10
    future_offsets: tuple[int, ...] = tuple(range(1, 11))
    reward_weights: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 0.5, 0.5, -0.1, -2.0)
    reward_scales: tuple[float, ...] = (0.3, 0.4, 1.0, 3.14, 0.3, 0.4)
    height_threshold: float = 0.25
    tilt_threshold: float = 0.8
    height_temperature: float = 0.02
    tilt_temperature: float = 0.04
    observation_noise: tuple[float, ...] = (0.01, 0.5, 0.2, 0.05)

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.control_dt)
            or self.control_dt <= 0
            or self.history_steps < 1
        ):
            raise ValueError("Control interval and history length must be positive")
        if (
            not self.future_offsets
            or tuple(sorted(set(self.future_offsets))) != tuple(self.future_offsets)
            or self.future_offsets[0] < 1
        ):
            raise ValueError("Future offsets must be positive and strictly increasing")
        if (
            len(self.reward_weights) != 8
            or len(self.reward_scales) != 6
            or any(x <= 0 for x in self.reward_scales)
        ):
            raise ValueError("Tracking requires eight weights and six positive scales")
        if min(self.height_temperature, self.tilt_temperature) <= 0:
            raise ValueError("Soft termination temperatures must be positive")


class MotionTrackingTask:
    """Evaluate root/body tracking from tensors without owning episode state.

    State dictionaries contain world-frame ``pos``, ``rot``, ``v``, ``w``,
    joint ``q`` and ``qd``. Linear velocity refers to the root body origin.
    ``reference_weight`` selects each joint's reference contribution: zero
    uses the configured resting pose; one adds a reference-relative residual.
    """

    reward_names = (
        "body_pos",
        "body_ori",
        "body_lin",
        "body_ang",
        "root_pos",
        "root_ori",
        "action_rate",
        "joint_limit",
    )

    def __init__(
        self,
        kinematics: TreeKinematics,
        resting_position: torch.Tensor,
        action_scale: torch.Tensor,
        soft_limits: torch.Tensor,
        reference_weight: torch.Tensor,
        tracked_links: Sequence[str],
        cfg: MotionTrackingCfg | None = None,
    ) -> None:
        self.cfg = cfg or MotionTrackingCfg()
        self.kinematics = kinematics
        self.resting_position = resting_position
        self.action_scale = action_scale
        self.soft_limits = soft_limits
        self.reference_weight = reference_weight
        self.dof = len(kinematics.joint_names)
        if any(
            x.shape != (self.dof,)
            for x in (resting_position, action_scale, reference_weight)
        ) or soft_limits.shape != (self.dof, 2):
            raise ValueError(
                "Tracking parameters must match the kinematics joint order"
            )
        if bool((action_scale <= 0).any()):
            raise ValueError("Action scales must be positive")
        self.tracked = kinematics.link_indices(tracked_links)
        self.moving = kinematics.link_indices(kinematics.moving_link_names[1:])
        self.future_offsets = torch.tensor(
            self.cfg.future_offsets, device=kinematics.device, dtype=torch.long
        )
        nq, nv, nw, ng = self.cfg.observation_noise
        self.noise_scale = resting_position.new_tensor(
            [nq] * self.dof + [nv] * self.dof + [nw] * 3 + [ng] * 3 + [0.0] * self.dof
        )

    @property
    def observation_dim(self) -> int:
        """One proprioceptive frame's feature count."""
        return 3 * self.dof + 6

    @property
    def goal_dim(self) -> int:
        """Flattened future reference feature count."""
        return len(self.cfg.future_offsets) * (15 + 2 * self.dof)

    @property
    def max_step_reward(self) -> float:
        """Sum of the six positive tracking weights times control duration."""
        return sum(self.cfg.reward_weights[:6]) * self.cfg.control_dt

    @staticmethod
    def _local(rotation: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
        return matrix_vector_product(rotation.transpose(-1, -2), vector)

    @staticmethod
    def _orientation(rotation: torch.Tensor) -> torch.Tensor:
        return rotation[..., :, :2].transpose(-1, -2).flatten(-2)

    _heading = staticmethod(heading_rotation)

    @staticmethod
    def _angle_squared(rotation: torch.Tensor) -> torch.Tensor:
        # Only the angle magnitude enters the reward. atan2 preserves the
        # shortest angle at zero and pi without constructing an axis/quaternion.
        vector = (
            torch.stack(
                (
                    rotation[..., 2, 1] - rotation[..., 1, 2],
                    rotation[..., 0, 2] - rotation[..., 2, 0],
                    rotation[..., 1, 0] - rotation[..., 0, 1],
                ),
                -1,
            )
            * 0.5
        )
        sine = vector.norm(dim=-1)
        cosine = (rotation.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) * 0.5
        return torch.atan2(sine, cosine).square()

    def links(self, state: State) -> LinkKinematics:
        """Compute all link poses and link-origin velocities."""
        return self.kinematics.evaluate(
            state["pos"], state["rot"], state["q"], state["v"], state["w"], state["qd"]
        )

    def observation(
        self, state: State, previous_action: torch.Tensor, encoder_bias: torch.Tensor
    ) -> torch.Tensor:
        """Joint position/velocity, local gyro/gravity and previous policy action."""
        gravity = -state["rot"][..., 2, :]
        return torch.cat(
            (
                state["q"] - self.resting_position + encoder_bias,
                state["qd"],
                self._local(state["rot"], state["w"]),
                gravity,
                previous_action,
            ),
            -1,
        )

    def control(
        self,
        action: torch.Tensor,
        reference_qpos: torch.Tensor,
        encoder_bias: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return normalized setpoint and the actual encoder-corrected joint target."""
        setpoint = (
            action
            + self.reference_weight
            * (reference_qpos - self.resting_position)
            / self.action_scale
        )
        target = self.resting_position + self.action_scale * setpoint - encoder_bias
        return setpoint, target

    def goal(self, state: State, future: State) -> torch.Tensor:
        """Encode future root pose relative to the robot, joints and local velocities."""
        rotation = state["rot"].unsqueeze(-3)
        position = self._local(rotation, future["pos"] - state["pos"].unsqueeze(-2))
        relative = matrix_product(rotation.transpose(-1, -2), future["rot"])
        return torch.cat(
            (
                position,
                self._orientation(relative),
                future["q"],
                future["qd"],
                self._local(future["rot"], future["v"]),
                self._local(future["rot"], future["w"]),
            ),
            -1,
        ).flatten(-2)

    def features(
        self, state: State, links: LinkKinematics | None = None
    ) -> torch.Tensor:
        """Heading-invariant root/joint state and movable-link positions/velocities."""
        links = links or self.links(state)
        return heading_kinematic_features(
            state["pos"],
            state["rot"],
            state["v"],
            state["w"],
            state["q"],
            state["qd"],
            self.resting_position,
            links.position.index_select(-2, self.moving),
            links.linear_velocity.index_select(-2, self.moving),
        )

    def tracking_errors(
        self,
        state: State,
        reference: State,
        links: LinkKinematics | None = None,
        reference_links: LinkKinematics | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return six squared errors before reward scaling.

        Body pose errors average over tracked links after aligning reference
        heading and horizontal root position. Root pose and body velocities
        use world coordinates. Position and velocity errors sum over axes;
        orientation uses the squared rotation angle in radians.
        """
        links = links or self.links(state)
        reference_links = reference_links or self.links(reference)
        p = links.position.index_select(-2, self.tracked)
        r = links.rotation.index_select(-3, self.tracked)
        rp = reference_links.position.index_select(-2, self.tracked)
        rr = reference_links.rotation.index_select(-3, self.tracked)
        alignment = matrix_product(
            self._heading(state["rot"]),
            self._heading(reference["rot"]).transpose(-1, -2),
        ).unsqueeze(-3)
        anchor = torch.cat(
            (state["pos"][..., :2], reference["pos"][..., 2:3]), -1
        ).unsqueeze(-2)
        aligned = anchor + matrix_vector_product(
            alignment, rp - reference["pos"].unsqueeze(-2)
        )
        errors = (
            (p - aligned).square().sum(-1).mean(-1),
            self._angle_squared(
                matrix_product(matrix_product(alignment, rr), r.transpose(-1, -2))
            ).mean(-1),
            (links.linear_velocity - reference_links.linear_velocity)
            .index_select(-2, self.tracked)
            .square()
            .sum(-1)
            .mean(-1),
            (links.angular_velocity - reference_links.angular_velocity)
            .index_select(-2, self.tracked)
            .square()
            .sum(-1)
            .mean(-1),
            (state["pos"] - reference["pos"]).square().sum(-1),
            self._angle_squared(
                matrix_product(reference["rot"], state["rot"].transpose(-1, -2))
            ),
        )
        return dict(zip(self.reward_names[:6], errors))

    def reward(
        self,
        state: State,
        reference: State,
        action: torch.Tensor,
        previous_action: torch.Tensor,
        links: LinkKinematics | None = None,
        reference_links: LinkKinematics | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute weighted tracking terms and the duration-scaled total reward."""
        errors = self.tracking_errors(state, reference, links, reference_links)
        values = [
            weight * torch.exp(-error / (scale * scale))
            for weight, scale, error in zip(
                self.cfg.reward_weights, self.cfg.reward_scales, errors.values()
            )
        ]
        values.extend(
            (
                self.cfg.reward_weights[6]
                * (action - previous_action).square().sum(-1),
                self.cfg.reward_weights[7]
                * (
                    torch.relu(self.soft_limits[:, 0] - state["q"])
                    + torch.relu(state["q"] - self.soft_limits[:, 1])
                ).sum(-1),
            )
        )
        return sum(values) * self.cfg.control_dt, dict(zip(self.reward_names, values))

    def analytics(
        self,
        state: State,
        reference: State,
        future: State,
        action: torch.Tensor,
        previous_action: torch.Tensor,
        encoder_bias: torch.Tensor,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Evaluate a physical transition, reusing its link kinematics."""
        links = self.links(state)
        reward, terms = self.reward(
            state,
            reference,
            action,
            previous_action,
            links=links,
            reference_links=self.links(reference),
        )
        failed, continuation = self.termination(state, reference)
        return {
            "reward": reward,
            "reward_terms": terms,
            "failed": failed,
            "continuation": continuation,
            "observation": self.observation(state, action, encoder_bias),
            "goal": self.goal(state, future),
            "features": self.features(state, links),
        }

    def failure_reasons(
        self, state: State, reference: State
    ) -> dict[str, torch.Tensor]:
        """Return height and tilt failure masks; both may apply to one state.

        Tilt is the absolute difference between the two root rotations' vertical
        axis dot products with world up, matching the termination criterion.
        """
        return {
            "height": (state["pos"][..., 2] - reference["pos"][..., 2]).abs()
            > self.cfg.height_threshold,
            "tilt": (state["rot"][..., 2, 2] - reference["rot"][..., 2, 2]).abs()
            > self.cfg.tilt_threshold,
        }

    def termination(
        self, state: State, reference: State
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return hard failure and differentiable continuation probability."""
        height = (state["pos"][..., 2] - reference["pos"][..., 2]).abs()
        tilt = (state["rot"][..., 2, 2] - reference["rot"][..., 2, 2]).abs()
        c = self.cfg
        hard = (height > c.height_threshold) | (tilt > c.tilt_threshold)
        continuation = torch.sigmoid(
            (c.height_threshold - height) / c.height_temperature
        ) * torch.sigmoid((c.tilt_threshold - tilt) / c.tilt_temperature)
        return hard, continuation
