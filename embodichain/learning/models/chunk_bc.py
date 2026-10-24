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

"""Configurable causal action-chunk predictors for state and visual BC."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

import torch
from torch import nn

from embodichain.utils import configclass

__all__ = ["ChunkBCModelCfg", "ChunkBCModel", "ChunkBCExecutor"]


@configclass
class ChunkBCModelCfg:
    """Define measured inputs and the model's ordered prediction targets.

    Args:
        observation_dim: Width of each numeric measured observation.
        target_dim: Width of each predicted target; its semantics are explicit.
        history_length: Number of causal input frames including the present.
        prediction_horizon: Number of predicted targets including the present.
        hidden_dims: Hidden action-head widths.
        cameras: Ordered camera keys; empty selects a numeric-only predictor.
        target_semantics: Declared meaning of the target vector.
        normalization_clip: Bound on normalized numeric features.

    Visual inputs use the shared ResNet10 encoder's 256-pixel RGB contract.
    The model contains no controller, task geometry or success classifier.
    """

    observation_dim: int = 1
    target_dim: int = 1
    history_length: int = 1
    prediction_horizon: int = 1
    hidden_dims: tuple[int, ...] = (512, 512)
    cameras: tuple[str, ...] = ()
    target_semantics: str = "vector"
    normalization_clip: float = 10.0


class ChunkBCModel(nn.Module):
    """Predict normalized target chunks from measured causal histories.

    Args:
        cfg: Input, camera and target contract. Normalization starts at zero
            mean/unit scale and is set explicitly using training-only data.
    """

    def __init__(self, cfg: ChunkBCModelCfg | None = None) -> None:
        super().__init__()
        self.cfg = deepcopy(cfg if cfg is not None else ChunkBCModelCfg())
        self.cfg.cameras = tuple(self.cfg.cameras)
        self.cfg.hidden_dims = tuple(self.cfg.hidden_dims)
        self.cfg.validate()
        dimensions = (
            self.cfg.observation_dim,
            self.cfg.target_dim,
            self.cfg.history_length,
            self.cfg.prediction_horizon,
            *self.cfg.hidden_dims,
        )
        if (
            any(type(value) is not int or value < 1 for value in dimensions)
            or not self.cfg.hidden_dims
        ):
            raise ValueError(
                "BC dimensions and history/prediction lengths must be positive"
            )
        if len(set(self.cfg.cameras)) != len(self.cfg.cameras) or any(
            not isinstance(name, str) or not name for name in self.cfg.cameras
        ):
            raise ValueError("Camera names must be ordered, distinct nonempty strings")
        if (
            not self.cfg.target_semantics
            or not 0 < self.cfg.normalization_clip < float("inf")
        ):
            raise ValueError(
                "BC target semantics and normalization clip must be explicit"
            )
        for name, dimension, fill in (
            ("observation_mean", self.cfg.observation_dim, 0),
            ("observation_std", self.cfg.observation_dim, 1),
            ("target_mean", self.cfg.target_dim, 0),
            ("target_std", self.cfg.target_dim, 1),
        ):
            self.register_buffer(name, torch.full((dimension,), float(fill)))
        self.encoder: nn.Module | None = None
        feature_dim = self.cfg.observation_dim
        if self.cfg.cameras:
            from .resnet_encoder import ResNet10Encoder

            self.encoder = ResNet10Encoder()
            feature_dim += len(self.cfg.cameras) * self.encoder.output_dim
        dimensions = (self.cfg.history_length * feature_dim, *self.cfg.hidden_dims)
        layers: list[nn.Module] = []
        for input_dim, output_dim in zip(dimensions[:-1], dimensions[1:]):
            layers.extend((nn.Linear(input_dim, output_dim), nn.ReLU()))
        layers.append(
            nn.Linear(dimensions[-1], self.cfg.prediction_horizon * self.cfg.target_dim)
        )
        self.head = nn.Sequential(*layers)

    @torch.no_grad()
    def set_normalization(
        self,
        observation_mean: torch.Tensor,
        observation_std: torch.Tensor,
        target_mean: torch.Tensor,
        target_std: torch.Tensor,
    ) -> None:
        """Install frozen statistics computed from an explicitly selected corpus.

        Args:
            observation_mean: Ordered numeric input means.
            observation_std: Finite positive input scales.
            target_mean: Ordered target means.
            target_std: Finite positive target scales.
        """
        values = dict(
            observation_mean=observation_mean,
            observation_std=observation_std,
            target_mean=target_mean,
            target_std=target_std,
        )
        for name, value in values.items():
            current = getattr(self, name)
            if (
                value.shape != current.shape
                or not torch.isfinite(value).all()
                or (name.endswith("std") and (value <= 0).any())
            ):
                raise ValueError(f"Invalid finite normalization statistic: {name}")
        for name, value in values.items():
            getattr(self, name).copy_(value)

    def forward(
        self, history: torch.Tensor, images: Mapping[str, torch.Tensor] | None = None
    ) -> torch.Tensor:
        """Return normalized target chunks without accepting future observations.

        Args:
            history: FP32 numeric observations shaped ``[B,H,observation_dim]``.
            images: Configured camera inputs, each FP32 ``[B,H,3,256,256]`` in
                ``[0,1]``. Pixel scaling/finiteness belongs to the data adapter.

        Returns:
            Normalized targets shaped ``[B,prediction_horizon,target_dim]``.
        """
        expected = (self.cfg.history_length, self.cfg.observation_dim)
        if (
            history.ndim != 3
            or history.shape[0] < 1
            or history.shape[1:] != expected
            or history.dtype != torch.float32
            or history.device != self.observation_mean.device
        ):
            raise ValueError(
                f"BC numeric history must be FP32 [B,{expected[0]},{expected[1]}] on the model device"
            )
        images = {} if images is None else images
        if set(images) != set(self.cfg.cameras):
            raise ValueError("BC camera keys differ from the configured input contract")
        normalized = ((history - self.observation_mean) / self.observation_std).clamp(
            -self.cfg.normalization_clip, self.cfg.normalization_clip
        )
        combined = normalized.flatten(1)
        if self.encoder is not None:
            image_shape = (len(history), self.cfg.history_length, 3, 256, 256)
            for camera in self.cfg.cameras:
                image = images[camera]
                if (
                    image.shape != image_shape
                    or image.dtype != torch.float32
                    or image.device != history.device
                ):
                    raise ValueError(
                        f"Camera {camera} must be FP32 {image_shape} on the model device"
                    )
            # Match the reference's history-major, camera-minor feature order.
            stacked = torch.stack(
                [images[camera] for camera in self.cfg.cameras], dim=2
            )
            visual = self.encoder(stacked.flatten(0, 2)).reshape(len(history), -1)
            combined = torch.cat((combined, visual), -1)
        return self.head(combined).reshape(
            len(history), self.cfg.prediction_horizon, self.cfg.target_dim
        )

    def predict_targets(
        self, history: torch.Tensor, images: Mapping[str, torch.Tensor] | None = None
    ) -> torch.Tensor:
        """Return targets in their declared physical/control units.

        Args:
            history: Measured numeric history.
            images: Optional measured camera history.

        Returns:
            Unnormalized target chunks; task adapters own controller decoding.
        """
        return self(history, images) * self.target_std + self.target_mean

    def get_extra_state(self) -> dict[str, Any]:
        """Return the ordered input and target identity for checkpoints.

        Returns:
            Versioned model configuration with camera order and target meaning.
        """
        return {"format": "chunk-bc-v1", "config": self.cfg.to_dict()}

    def set_extra_state(self, state: Mapping[str, Any]) -> None:
        """Validate input, time and target contracts when loading weights.

        Args:
            state: Identity returned by :meth:`get_extra_state`.
        """
        if state != self.get_extra_state():
            raise ValueError("BC checkpoint input or target contract differs")


class ChunkBCExecutor:
    """Maintain measured histories and independent target queues per environment.

    Args:
        model: Frozen or evaluation-mode BC model.
        action_horizon: Number of targets executed before replanning.

    Returned targets retain the model's declared units. The task/controller
    owns conversion to actual environment actions. Every call updates measured
    history, including calls that consume an already predicted target.
    """

    def __init__(self, model: ChunkBCModel, action_horizon: int = 1) -> None:
        if (
            type(action_horizon) is not int
            or not 1 <= action_horizon <= model.cfg.prediction_horizon
        ):
            raise ValueError("Action horizon must fit the model prediction horizon")
        self.model, self.action_horizon = model, action_horizon
        self._history: torch.Tensor | None = None
        self._images: dict[str, torch.Tensor] = {}
        self._targets: torch.Tensor | None = None
        self._index: torch.Tensor | None = None
        self._fresh: torch.Tensor | None = None

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Discard selected queues and histories before the next observation.

        Args:
            env_ids: Optional unique int64 environment indices; None resets all.
        """
        if self._fresh is None:
            return
        if env_ids is None:
            self._fresh.fill_(True)
        else:
            self._fresh[env_ids] = True

    @torch.no_grad()
    def action(
        self,
        observation: torch.Tensor,
        images: Mapping[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Consume measured observations and return one predicted target per row.

        Args:
            observation: Current FP32 numeric inputs ``[B,observation_dim]``.
            images: Current RGB floats ``[B,3,256,256]`` per configured camera.

        Returns:
            Owned unnormalized targets ``[B,target_dim]``.
        """
        images = {} if images is None else images
        if (
            observation.ndim != 2
            or observation.shape[-1] != self.model.cfg.observation_dim
            or observation.dtype != torch.float32
            or observation.device != self.model.observation_mean.device
        ):
            raise ValueError(
                "Executor requires FP32 measured observations on the model device"
            )
        if set(images) != set(self.model.cfg.cameras) or any(
            value.shape != (len(observation), 3, 256, 256)
            or value.dtype != torch.float32
            or value.device != observation.device
            for value in images.values()
        ):
            raise ValueError("Executor cameras require aligned FP32 RGB inputs")
        if self._history is None:
            self._history = (
                observation[:, None]
                .expand(-1, self.model.cfg.history_length, -1)
                .clone()
            )
            self._images = {
                name: value[:, None]
                .expand(-1, self.model.cfg.history_length, -1, -1, -1)
                .clone()
                for name, value in images.items()
            }
            self._fresh = torch.ones(
                len(observation), dtype=torch.bool, device=observation.device
            )
            self._index = torch.zeros(
                len(observation), dtype=torch.int64, device=observation.device
            )
            self._targets = observation.new_zeros(
                len(observation),
                self.model.cfg.prediction_horizon,
                self.model.cfg.target_dim,
            )
        if observation.shape != (
            len(self._history),
            self.model.cfg.observation_dim,
        ) or set(images) != set(self.model.cfg.cameras):
            raise ValueError(
                "Executor observations or cameras differ from its initialized layout"
            )
        self._history = torch.cat((self._history[:, 1:], observation[:, None]), 1)
        self._history[self._fresh] = observation[self._fresh, None]
        for name, value in images.items():
            self._images[name] = torch.cat(
                (self._images[name][:, 1:], value[:, None]), 1
            )
            self._images[name][self._fresh] = value[self._fresh, None]
        replan = self._fresh | (self._index >= self.action_horizon)
        if replan.any():
            targets = self.model.predict_targets(
                self._history[replan],
                {name: value[replan] for name, value in self._images.items()},
            )
            self._targets[replan] = targets
            self._index[replan] = 0
            self._fresh[replan] = False
        result = self._targets[
            torch.arange(len(observation), device=observation.device), self._index
        ].clone()
        self._index.add_(1)
        return result
