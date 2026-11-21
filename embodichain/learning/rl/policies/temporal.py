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

"""Reusable temporal actors, history storage and inference normalization."""

from __future__ import annotations

import torch
from torch import nn

from embodichain.utils import configclass
from embodichain.learning.models import EmpiricalNormalizer, MLP

__all__ = [
    "ObservationHistory",
    "TemporalActorCfg",
    "TemporalActor",
    "TemporalObservationNormalizer",
]


class ObservationHistory:
    """Per-environment chronological history with owned observation snapshots.

    Output contains the current frame followed by ``history_length`` chronological
    frames including current, matching the public browser-policy architecture.
    Call :meth:`append` exactly once per control observation;
    repeated reads of :attr:`observation` never advance the history. Reset
    repeats the first observed state and zeros its previous-action features.
    Storage is FP32 and linear in environments, history length and frame size.
    """

    def __init__(
        self,
        num_envs: int,
        frame_dim: int,
        history_length: int = 30,
        device: torch.device | str = "cpu",
        action_slice: tuple[int, int] | None = None,
    ) -> None:
        if num_envs <= 0 or frame_dim <= 0 or history_length < 0:
            raise ValueError(
                "History dimensions must be positive and length non-negative."
            )
        if action_slice is not None and not (
            0 <= action_slice[0] < action_slice[1] <= frame_dim
        ):
            raise ValueError("action_slice must select features inside a frame.")
        self.num_envs = num_envs
        self.frame_dim = frame_dim
        self.history_length = history_length
        self.action_slice = action_slice
        self._frames = torch.zeros(
            num_envs,
            max(1, history_length),
            frame_dim,
            device=device,
            dtype=torch.float32,
        )

    @torch.no_grad()
    def reset(self, frame: torch.Tensor, env_ids: torch.Tensor | None = None) -> None:
        """Repeat reset frames for all environments or the selected rows.

        ``frame`` may contain the full batch or only selected rows. Boolean
        masks and integer indices are accepted for ``env_ids``.
        """
        if env_ids is None:
            if frame.shape != (self.num_envs, self.frame_dim):
                raise ValueError("Reset frame must match [num_envs, frame_dim].")
            first = frame.to(self._frames).clone()
            if self.action_slice is not None:
                first[:, slice(*self.action_slice)] = 0
            self._frames.copy_(first[:, None].expand_as(self._frames))
            return
        selected = self._frames[env_ids]
        first = frame[env_ids] if frame.shape[0] == self.num_envs else frame
        if first.shape != (selected.shape[0], self.frame_dim):
            raise ValueError("Reset frame must match the selected environment rows.")
        first = first.to(self._frames).clone()
        if self.action_slice is not None:
            first[:, slice(*self.action_slice)] = 0
        self._frames[env_ids] = first[:, None].expand_as(selected)

    @torch.no_grad()
    def append(
        self, frame: torch.Tensor, reset_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Advance once, optionally replace reset rows, and return a snapshot."""
        if frame.shape != (self.num_envs, self.frame_dim):
            raise ValueError("History frame must match [num_envs, frame_dim].")
        if self.history_length:
            self._frames[:, :-1].copy_(self._frames[:, 1:].clone())
        self._frames[:, -1].copy_(frame)
        if reset_mask is not None:
            self.reset(frame, reset_mask)
        return self.observation

    @property
    def observation(self) -> torch.Tensor:
        """Return an owned flattened snapshot in the selected layout."""
        if self.history_length:
            return torch.cat(
                (self._frames[:, -1:], self._frames[:, -self.history_length :]),
                dim=1,
            ).flatten(start_dim=1)
        return self._frames.flatten(start_dim=1).clone()


@configclass
class TemporalActorCfg:
    """Temporal actor layout and network settings.

    History acquisition stays with the environment. Inputs contain current
    followed by chronological history including current (``official_v1``).
    """

    frame_dim: int = 33
    history_length: int = 30
    use_history: bool = True
    latent_dim: int = 32
    hidden_sizes: tuple[int, ...] = (256, 256, 128)
    architecture_version: str = "official_v1"


class TemporalObservationNormalizer(EmpiricalNormalizer):
    """Author-export inference transform with native running-stat updates.

    The public export specifies clipping and epsilon placement, but does not
    publish its training-time statistics update. Native population-moment
    accumulation remains the explicit training convention.

    Args:
        feature_dim: Width of the normalized observation vector.
    """

    def __init__(self, feature_dim: int) -> None:
        super().__init__(feature_dim, epsilon=1e-8)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        return ((value - self.mean) / (self.variance + self.epsilon).sqrt()).clamp(
            -10.0, 10.0
        )


class _TemporalEncoder(nn.Module):
    """Temporal encoder described by the author's public policy JSON.

    Each frame is projected before convolution. The oldest history frame is
    dropped; a 30-frame history therefore reaches the convolutions as 29 frames.
    """

    def __init__(self, frame_dim: int, history_length: int, latent_dim: int) -> None:
        super().__init__()
        channels = 32
        self.frame_transform = nn.Sequential(
            nn.Linear(frame_dim, channels),
            nn.ReLU(),
            nn.Linear(channels, channels),
            nn.ReLU(),
        )
        length = history_length - 1
        layers: list[nn.Module] = []
        for kernel, stride in ((9, 2), (5, 1), (5, 1)):
            length = (length - kernel) // stride + 1
            if length <= 0:
                raise ValueError("history_length is too short for the export encoder.")
            layers.extend((nn.Conv1d(channels, channels, kernel, stride), nn.ReLU()))
        self.temporal_convolution = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(3)
        self.projection = nn.Linear(channels * 3, latent_dim)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        transformed = self.frame_transform(frames.transpose(1, 2)[:, 1:])
        temporal = self.temporal_convolution(transformed.transpose(1, 2))
        return self.projection(self.pool(temporal).flatten(start_dim=1))


class TemporalActor(nn.Module):
    """Produce action means and reusable temporal features from a flat history.

    Args:
        cfg: Actor dimensions, encoder version and history convention.
        action_dim: Number of policy action means.
    """

    def __init__(self, cfg: TemporalActorCfg | dict, action_dim: int) -> None:
        super().__init__()
        cfg = TemporalActorCfg(**cfg) if isinstance(cfg, dict) else cfg
        if cfg.architecture_version != "official_v1":
            raise ValueError("Unknown temporal actor architecture_version.")
        if cfg.frame_dim <= 0 or cfg.history_length < 0:
            raise ValueError(
                "Temporal frame dimensions must be positive and history non-negative."
            )
        self.input_dim = cfg.frame_dim * (cfg.history_length + 1)
        self.architecture_version = cfg.architecture_version
        self.frame_dim = cfg.frame_dim
        self.history_length = cfg.history_length
        self.encoder: nn.Module | None = None
        if self.history_length and cfg.use_history:
            self.encoder = _TemporalEncoder(
                cfg.frame_dim, self.history_length, cfg.latent_dim
            )
        latent_dim = cfg.latent_dim if self.encoder is not None else 0
        self.feature_dims = {
            "latent": cfg.latent_dim if self.encoder is not None else cfg.frame_dim,
            "current": cfg.frame_dim,
        }
        self.head = MLP(self.frame_dim + latent_dim, action_dim, cfg.hidden_sizes)

    def features(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract the decision vector and auxiliary latent without advancing history.

        Args:
            obs: Complete normalized history vectors in the configured layout.

        Returns:
            Current-plus-latent decision features and the auxiliary latent.
            With history disabled, both are the current observation frame.
        """
        frames = obs.reshape(-1, self.history_length + 1, self.frame_dim)
        current = frames[:, 0]
        if self.encoder is None:
            return current, current
        history = frames[:, 1:]
        latent = self.encoder(history.transpose(1, 2))
        return torch.cat((current, latent), dim=-1), latent

    def checkpoint_contract(self) -> dict[str, object]:
        """Describe temporal input semantics even when the encoder is disabled.

        Returns:
            Stable layout/version information for the policy checkpoint contract.
        """
        return {
            "architecture_version": self.architecture_version,
            "frame_dim": self.frame_dim,
            "history_length": self.history_length,
            "use_history": self.encoder is not None,
            "feature_dims": dict(self.feature_dims),
        }

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return action means from complete normalized observation histories."""
        features, _ = self.features(obs)
        return self.head(features)

    def forward_with_features(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return action means and graph-connected features in one encoder pass.

        Args:
            obs: Normalized flattened observation histories.

        Returns:
            Action means and named latent/current features for auxiliary tasks.
        """
        features, latent = self.features(obs)
        return self.head(features), {
            "latent": latent,
            "current": features[:, : self.frame_dim],
        }
