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

"""Example-local visual action-chunk BC with a shared trainable CNN."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from embodichain.learning.models import ResNet10Encoder
from examples.ods.simple_bc_demo.models import pose_targets

__all__ = ["VisualChunkBC"]


class VisualChunkBC(nn.Module):
    """Predict TCP/gripper chunks using two frames of vision and proprioception.

    Vision is flattened in history-major, configured-camera-minor order. One
    encoder is shared by all histories and views. Future geometry is accepted
    only by :meth:`loss` for constructing targets, never by :meth:`forward`.
    """

    history_length = 2
    arm_command_mode = "pose"
    observation_mode = "visual"

    def __init__(
        self,
        proprio_mean: torch.Tensor,
        proprio_std: torch.Tensor,
        target_mean: torch.Tensor,
        target_std: torch.Tensor,
        cameras: tuple[str, ...],
        prediction_horizon: int = 32,
    ) -> None:
        super().__init__()
        if (
            not isinstance(cameras, tuple)
            or len(cameras) not in (1, 2)
            or any(not isinstance(camera, str) or not camera for camera in cameras)
            or len(set(cameras)) != len(cameras)
        ):
            raise ValueError(
                "Visual BC requires one or two distinct ordered camera names"
            )
        if type(prediction_horizon) is not int or prediction_horizon < 1:
            raise ValueError("Prediction horizon must be a positive integer")
        for name, value, width, scale in (
            ("proprio_mean", proprio_mean, 27, False),
            ("proprio_std", proprio_std, 27, True),
            ("target_mean", target_mean, 10, False),
            ("target_std", target_std, 10, True),
        ):
            if (
                value.shape != (width,)
                or value.dtype != torch.float32
                or not torch.isfinite(value).all()
                or (scale and (value <= 0).any())
            ):
                raise ValueError(
                    f"Invalid finite float32 normalization statistic: {name}"
                )
            self.register_buffer(name, value.detach().clone())
        self.cameras = cameras
        self.prediction_horizon = prediction_horizon
        self.encoder = ResNet10Encoder()
        input_dim = 2 * (27 + len(cameras) * self.encoder.output_dim)
        self.head = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, prediction_horizon * 10),
        )

    def get_extra_state(self) -> dict[str, Any]:
        """Save the ordered input contract alongside tensor parameters."""
        return {
            "version": 1,
            "cameras": self.cameras,
            "history_length": self.history_length,
            "prediction_horizon": self.prediction_horizon,
        }

    def set_extra_state(self, state: dict[str, Any]) -> None:
        """Reject checkpoint reuse under a different camera or temporal contract."""
        if state != self.get_extra_state():
            raise ValueError(
                "Visual BC checkpoint camera order or horizon does not match"
            )

    def forward(
        self,
        proprio: torch.Tensor,
        images: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        """Return normalized ``[B, F, 10]`` targets from two causal frames."""
        if (
            proprio.ndim != 3
            or proprio.shape[0] < 1
            or proprio.shape[1:] != (2, 27)
            or proprio.dtype != torch.float32
            or proprio.device != self.proprio_mean.device
        ):
            raise ValueError("Visual BC proprioception must be float32 [B, 2, 27]")
        if set(images) != set(self.cameras):
            raise ValueError(
                "Visual BC camera keys must exactly match configured cameras"
            )
        batch_size = proprio.shape[0]
        expected = (batch_size, 2, 3, 256, 256)
        for camera in self.cameras:
            image = images[camera]
            if (
                image.shape != expected
                or image.dtype != torch.float32
                or image.device != proprio.device
            ):
                raise ValueError(f"Camera {camera!r} requires float32 {expected}")
        # [B, H, V, C, W, W] -> [B*H*V, C, W, W]; one shared backbone call.
        stacked = torch.stack([images[camera] for camera in self.cameras], dim=2)
        visual = self.encoder(stacked.flatten(0, 2)).reshape(batch_size, -1)
        normalized = ((proprio - self.proprio_mean) / self.proprio_std).clamp(-10, 10)
        combined = torch.cat((normalized.flatten(1), visual), dim=-1)
        return self.head(combined).reshape(batch_size, self.prediction_horizon, 10)

    def loss(
        self,
        geometry: torch.Tensor,
        actions: torch.Tensor,
        images: Mapping[str, torch.Tensor],
        *,
        loss_mask: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reuse pose-target MSE with geometry kept outside policy inputs."""
        expected = (geometry.shape[0], self.prediction_horizon + 1)
        if (
            geometry.shape != (*expected, 51)
            or actions.shape != (*expected, 7)
            or loss_mask.shape != expected
            or valid_mask.shape != expected
            or loss_mask.dtype != torch.bool
            or valid_mask.dtype != torch.bool
        ):
            raise ValueError("Visual BC geometry/actions/masks must match H=2 and F")
        if not valid_mask.all() or loss_mask[:, 0].any() or not loss_mask[:, 1:].all():
            raise ValueError("Visual BC trains only complete valid future windows")
        predictions = self(geometry[:, :2, :27], images)
        targets = (
            pose_targets(geometry[:, 1:], actions[:, 1:]) - self.target_mean
        ) / self.target_std
        errors = (predictions - targets).square()
        return errors[..., :9].mean(), errors[..., 9].mean()
