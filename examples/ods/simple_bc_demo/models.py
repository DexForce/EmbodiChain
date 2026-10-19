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

"""Small action-chunk BC policy used only by the PickPlace showcase."""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

from embodichain.utils.math import (
    axis_angle_from_quat,
    axis_angle_to_rotation_matrix,
    quat_from_matrix,
)

__all__ = [
    "ChunkBC",
    "ChunkState",
    "MLPBC",
    "GeometryRecurrentBC",
    "pose_targets",
    "rotation_from_6d",
]


def rotation_from_6d(value: torch.Tensor) -> torch.Tensor:
    """Orthonormalize two rotation columns, with a finite degenerate fallback."""
    first = value[..., :3]
    default = torch.zeros_like(first)
    default[..., 0] = 1
    first = F.normalize(
        torch.where(first.norm(dim=-1, keepdim=True) < 1e-6, default, first), dim=-1
    )
    second = value[..., 3:] - (first * value[..., 3:]).sum(-1, keepdim=True) * first
    alternate = F.one_hot(first.abs().argmin(-1), 3).to(value)
    alternate = alternate - (alternate * first).sum(-1, keepdim=True) * first
    second = F.normalize(
        torch.where(second.norm(dim=-1, keepdim=True) < 1e-6, alternate, second), dim=-1
    )
    return torch.stack((first, second, torch.cross(first, second, dim=-1)), dim=-1)


def pose_targets(observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """Recover commanded absolute targets from causal observations and requests.

    Uses the recorded command and its pre-action TCP pose, never a measured
    next state. Position is in arena metres, rotation uses its first two columns,
    and the final component is the normalized gripper request.
    """
    if (
        observations.shape[:-1] != actions.shape[:-1]
        or observations.shape[-1] != 51
        or actions.shape[-1] != 7
    ):
        raise ValueError(
            "Pose targets require aligned geometry observations and seven requests"
        )
    rotation = axis_angle_to_rotation_matrix(
        actions[..., 3:6] * 0.1
    ) @ rotation_from_6d(observations[..., 21:27])
    return torch.cat(
        (
            observations[..., 18:21] + actions[..., :3] * 0.02,
            rotation[..., :, :2].transpose(-1, -2).flatten(-2),
            actions[..., 6:],
        ),
        dim=-1,
    )


class ChunkState(NamedTuple):
    """Policy-owned short history and unexecuted network predictions."""

    previous_observation: torch.Tensor
    targets: torch.Tensor
    index: int


class ChunkBC(nn.Module):
    """Predict absolute TCP/gripper chunks from two measured observations.

    The action method converts queued absolute targets to bounded requests for
    the existing relative-pose controller. There is no expert, stage machine,
    elapsed-time input, or learned success signal in student execution.
    """

    observation_mode = "geometry"
    arm_command_mode = "pose"
    history_length = 2

    def __init__(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        target_mean: torch.Tensor,
        target_std: torch.Tensor,
        prediction_horizon: int = 64,
        action_horizon: int = 16,
        observation_noise: float = 0.01,
    ) -> None:
        super().__init__()
        if (
            mean.shape != (51,)
            or std.shape != (51,)
            or target_mean.shape != (10,)
            or target_std.shape != (10,)
        ):
            raise ValueError(
                "Chunk BC requires 51 observation and 10 target statistics"
            )
        if (
            any(
                type(n) is not int or n < 1
                for n in (prediction_horizon, action_horizon)
            )
            or action_horizon > prediction_horizon
        ):
            raise ValueError(
                "Action horizon must be within the positive prediction horizon"
            )
        if not 0 <= observation_noise < float("inf"):
            raise ValueError("Observation noise must be finite and nonnegative")
        if (
            not all(
                torch.isfinite(v).all() for v in (mean, std, target_mean, target_std)
            )
            or (std <= 0).any()
            or (target_std <= 0).any()
        ):
            raise ValueError("Normalization must be finite with positive scales")
        self.prediction_horizon = prediction_horizon
        self.action_horizon = action_horizon
        self.observation_noise = observation_noise
        for name, value in (
            ("mean", mean),
            ("std", std),
            ("target_mean", target_mean),
            ("target_std", target_std),
        ):
            self.register_buffer(name, value.clone())
        self.net = nn.Sequential(
            nn.Linear(102, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, prediction_horizon * 10),
        )

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """Predict normalized future targets from exactly two causal observations."""
        if history.ndim != 3 or history.shape[1:] != (2, 51):
            raise ValueError("Chunk BC history must have shape [batch, 2, 51]")
        normalized = ((history - self.mean) / self.std).clamp(-10, 10)
        return self.net(normalized.flatten(1)).reshape(-1, self.prediction_horizon, 10)

    def action(
        self, obs: torch.Tensor, hidden: ChunkState | None = None
    ) -> tuple[torch.Tensor, ChunkState]:
        """Replan periodically from measurements and execute one predicted target."""
        previous = obs if hidden is None else hidden.previous_observation
        if hidden is None or hidden.index >= self.action_horizon:
            targets = (
                self(torch.stack((previous, obs), 1)) * self.target_std
                + self.target_mean
            )
            index = 0
        else:
            targets, index = hidden.targets, hidden.index
        target = targets[:, index]
        rotation_error = rotation_from_6d(target[:, 3:9]) @ rotation_from_6d(
            obs[:, 21:27]
        ).transpose(-1, -2)
        request = torch.cat(
            (
                (target[:, :3] - obs[:, 18:21]) / 0.02,
                axis_angle_from_quat(quat_from_matrix(rotation_error)) / 0.1,
                target[:, 9:],
            ),
            dim=-1,
        ).clamp(-1, 1)
        return request, ChunkState(obs, targets, index + 1)

    def loss(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        *,
        loss_mask: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Supervise valid future commands; pad initial history from the current observation."""
        expected = (observations.shape[0], self.prediction_horizon + 1)
        if (
            observations.shape[:2] != expected
            or loss_mask.shape != expected
            or valid_mask.shape != expected
        ):
            raise ValueError(
                "Chunk context and masks must match history plus prediction horizon"
            )
        if (
            loss_mask.dtype != torch.bool
            or valid_mask.dtype != torch.bool
            or not valid_mask[:, 1].all()
            or (loss_mask & ~valid_mask).any()
            or loss_mask[:, 0].any()
            or not loss_mask.any()
        ):
            raise ValueError("Chunk masks must select real current/future targets")
        history = observations[:, :2].clone()
        history[:, 0] = torch.where(valid_mask[:, :1], history[:, 0], history[:, 1])
        if self.training and self.observation_noise:
            history = (
                history + torch.randn_like(history) * self.std * self.observation_noise
            )
        predictions = self(history)
        targets = (
            pose_targets(observations[:, 1:], actions[:, 1:]) - self.target_mean
        ) / self.target_std
        errors = (predictions - targets).square()[loss_mask[:, 1:]]
        return errors[..., :9].mean(), errors[..., 9].mean()


class MLPBC(nn.Module):
    """Two-layer state BC in normalized pose-request or joint-residual space."""

    observation_mode = "geometry"

    def __init__(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        low: torch.Tensor,
        high: torch.Tensor,
        *,
        arm_command_mode: str = "pose",
    ) -> None:
        super().__init__()
        width = 7 if arm_command_mode == "pose" else 8
        if arm_command_mode not in {"pose", "residual"}:
            raise ValueError("MLP actions must be pose requests or joint residuals")
        if (
            mean.shape != (51,)
            or std.shape != (51,)
            or low.shape != (width,)
            or high.shape != low.shape
        ):
            raise ValueError(
                "MLP requires 51 measured features and matching action bounds"
            )
        if (
            not all(torch.isfinite(v).all() for v in (mean, std, low, high))
            or (std <= 0).any()
            or (high <= low).any()
        ):
            raise ValueError("Invalid MLP normalization or action bounds")
        self.arm_command_mode = arm_command_mode
        for name, value in (("mean", mean), ("std", std), ("low", low), ("high", high)):
            self.register_buffer(name, value.clone())
        self.net = nn.Sequential(
            nn.Linear(51, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, width),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(((obs - self.mean) / self.std).clamp(-10, 10))

    def action(
        self,
        obs: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None]:
        request = self(obs).clamp(-1, 1)
        if self.arm_command_mode == "pose":
            return request, None
        arm = (obs[..., :7] + 0.25 * request[..., :7]).clamp(
            self.low[:7], self.high[:7]
        )
        finger = self.low[7:] + (request[..., 7:] + 1) * 0.5 * (
            self.high[7:] - self.low[7:]
        )
        return torch.cat((arm, finger), dim=-1), None

    def loss(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        *,
        loss_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target = actions
        if self.arm_command_mode == "residual":
            target = torch.cat(
                (
                    (actions[..., :7] - obs[..., :7]) / 0.25,
                    2
                    * (actions[..., 7:] - self.low[7:])
                    / (self.high[7:] - self.low[7:])
                    - 1,
                ),
                dim=-1,
            )
        error = (self(obs) - target).square()
        if loss_mask is not None:
            error = error[loss_mask]
        return error[..., :-1].mean(), error[..., -1].mean()


class GeometryRecurrentBC(MLPBC):
    """Causal measured-geometry fallback with normalized pose-request MSE."""

    def __init__(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        low: torch.Tensor,
        high: torch.Tensor,
    ) -> None:
        super().__init__(mean, std, low, high, arm_command_mode="pose")
        del self.net
        self.gru = nn.GRU(51, 128, batch_first=True)
        self.head = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 7))

    def forward(
        self, obs: torch.Tensor, hidden: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict unclamped requests from full causal measured-state sequences."""
        normalized = ((obs - self.mean) / self.std).clamp(-10, 10)
        features, hidden = self.gru(normalized, hidden)
        return self.head(features), hidden

    def action(
        self, obs: torch.Tensor, hidden: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance one measured-state step and carry its recurrent state."""
        requests, hidden = self(obs[:, None], hidden)
        return requests[:, 0].clamp(-1, 1), hidden

    def loss(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        *,
        loss_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Supervise valid requests while backpropagating through their prefixes."""
        if loss_mask is not None and (
            loss_mask.shape != obs.shape[:2]
            or loss_mask.dtype != torch.bool
            or not loss_mask.any()
        ):
            raise ValueError("loss_mask must select supervised sequence steps")
        predictions, _ = self(obs)
        error = (predictions - actions).square()
        if loss_mask is not None:
            error = error[loss_mask]
        return error[..., :6].mean(), error[..., 6].mean()
