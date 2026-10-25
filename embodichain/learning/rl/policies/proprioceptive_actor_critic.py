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

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy

import torch
from torch import nn
from torch.distributions import Normal
from tensordict import TensorDict

from embodichain.utils import configclass
from embodichain.learning.models import EmpiricalNormalizer
from .actor_critic import ActorCritic

__all__ = [
    "ObservationHistory",
    "ProprioceptiveActorCriticCfg",
    "ProprioceptiveActorCritic",
]


class ObservationHistory:
    """Per-environment chronological history with owned observation snapshots.

    The default output contains ``history_length`` previous frames followed by
    the current frame. ``layout="current_history"`` instead emits the current
    frame followed by ``history_length`` frames including that current frame,
    matching the author's public browser-policy export. Call :meth:`append`
    exactly once per control observation;
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
        layout: str = "previous_current",
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
        if layout not in {"previous_current", "current_history"}:
            raise ValueError("Unknown proprioceptive history layout.")
        self.layout = layout
        self._frames = torch.zeros(
            num_envs, history_length + 1, frame_dim, device=device, dtype=torch.float32
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
        if self.layout == "current_history" and self.history_length:
            return torch.cat(
                (self._frames[:, -1:], self._frames[:, -self.history_length :]),
                dim=1,
            ).flatten(start_dim=1)
        return self._frames.flatten(start_dim=1).clone()


@configclass
class ProprioceptiveActorCriticCfg:
    """Versioned temporal policy settings with explicit checkpoint compatibility.

    ``legacy_v1`` retains the original migration's encoder and observation
    layout. ``official_v1`` uses the author's public browser-export architecture
    and requires ``[current, history]``, with current also at the history's end.
    Existing configurations without a version keep their legacy semantics.

    ``use_history=False`` disables the encoder while preserving the input
    layout; ``history_length=0`` also removes historical input frames.
    ``contact_points=0``
    omits the decoder and training labels for the no-contact-supervision
    ablation. Contact labels are integer class indices for fixed sample points.
    """

    frame_dim: int = 33
    history_length: int = 30
    use_history: bool = True
    latent_dim: int = 32
    actor_hidden_sizes: tuple[int, ...] = (256, 256, 128)
    critic_hidden_sizes: tuple[int, ...] = (256, 256, 128)
    decoder_hidden_sizes: tuple[int, ...] = (128, 128)
    contact_points: int = 0
    contact_classes: int = 2
    contact_class_weights: tuple[float, ...] | None = None
    contact_observation_group: str = "contact_labels"
    # Append fields to preserve the public configuration's positional arguments.
    # New reproductions explicitly select the now-public author architecture.
    architecture_version: str = "legacy_v1"


def _mlp(input_dim: int, hidden: Sequence[int], output_dim: int) -> nn.Sequential:
    modules: list[nn.Module] = []
    for width in hidden:
        modules.extend((nn.Linear(input_dim, width), nn.ELU()))
        input_dim = width
    modules.append(nn.Linear(input_dim, output_dim))
    return nn.Sequential(*modules)


class _ExportNormalizer(EmpiricalNormalizer):
    """Author-export inference transform with native running-stat updates.

    The public export specifies clipping and epsilon placement, but does not
    publish its training-time statistics update. Native population-moment
    accumulation remains the explicit training convention.
    """

    def __init__(self, feature_dim: int) -> None:
        super().__init__(feature_dim, epsilon=1e-8)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        return ((value - self.mean) / (self.variance + self.epsilon).sqrt()).clamp(
            -10.0, 10.0
        )


class _ExportHistoryEncoder(nn.Module):
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


class _TemporalActor(nn.Module):
    def __init__(self, cfg: ProprioceptiveActorCriticCfg, action_dim: int) -> None:
        super().__init__()
        self.frame_dim = cfg.frame_dim
        self.history_length = cfg.history_length
        self.current_first = cfg.architecture_version == "official_v1"
        self.encoder: nn.Module | None = None
        if self.history_length and cfg.use_history and self.current_first:
            self.encoder = _ExportHistoryEncoder(
                cfg.frame_dim, self.history_length, cfg.latent_dim
            )
        elif self.history_length and cfg.use_history:
            layers: list[nn.Module] = []
            channels = self.frame_dim
            length = self.history_length
            for width, kernel, stride in ((64, 5, 2), (64, 3, 2), (32, 3, 1)):
                length = (length - kernel) // stride + 1
                if length <= 0:
                    raise ValueError(
                        "history_length is too short for the temporal encoder."
                    )
                layers.extend((nn.Conv1d(channels, width, kernel, stride), nn.ELU()))
                channels = width
            layers.extend((nn.Flatten(), nn.Linear(channels * length, cfg.latent_dim)))
            self.encoder = nn.Sequential(*layers)
        latent_dim = cfg.latent_dim if self.encoder is not None else 0
        self.head = _mlp(
            self.frame_dim + latent_dim, cfg.actor_hidden_sizes, action_dim
        )

    def features(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        frames = obs.reshape(-1, self.history_length + 1, self.frame_dim)
        current = frames[:, 0] if self.current_first else frames[:, -1]
        if self.encoder is None:
            return current, current
        history = frames[:, 1:] if self.current_first else frames[:, :-1]
        latent = self.encoder(history.transpose(1, 2))
        return torch.cat((current, latent), dim=-1), latent

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        features, _ = self.features(obs)
        return self.head(features)


class ProprioceptiveActorCritic(ActorCritic):
    """Native Gaussian actor-critic with temporal proprioceptive encoding.

    The version selects ``[history, current]`` (legacy) or ``[current, history]``
    (official) actor input; the critic and optional contact labels live in
    explicitly separate observation groups. Labels never
    enter the actor. The encoder is recomputed from stored raw observations in
    every training mini-batch, preserving auxiliary and policy gradients.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        device: torch.device,
        critic_obs_dim: int,
        actor_obs_groups: Sequence[str],
        critic_obs_groups: Sequence[str],
        cfg: ProprioceptiveActorCriticCfg | dict | None = None,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        initial_action_std: float = 1.0,
        action_std_range: tuple[float, float] = (1e-6, 1e6),
    ) -> None:
        cfg = ProprioceptiveActorCriticCfg(**cfg) if isinstance(cfg, dict) else cfg
        cfg = cfg or ProprioceptiveActorCriticCfg()
        if cfg.architecture_version not in {"legacy_v1", "official_v1"}:
            raise ValueError("Unknown proprioceptive policy architecture_version.")
        expected_dim = cfg.frame_dim * (cfg.history_length + 1)
        if cfg.frame_dim <= 0 or cfg.history_length < 0 or obs_dim != expected_dim:
            raise ValueError(
                f"Temporal actor requires obs_dim={expected_dim}, got {obs_dim}."
            )
        if not actor_obs_groups or not critic_obs_groups:
            raise ValueError(
                "Temporal policy requires explicit actor and critic observation groups."
            )

        def overlaps(left: str, right: str) -> bool:
            return (
                left == right
                or left.startswith(right + ".")
                or right.startswith(left + ".")
            )

        if any(
            overlaps(actor, critic)
            for actor in actor_obs_groups
            for critic in critic_obs_groups
        ):
            raise ValueError(
                "Temporal actor and privileged critic groups must be disjoint."
            )
        if cfg.contact_points < 0 or cfg.contact_classes < 2:
            raise ValueError(
                "Contact point count must be non-negative and classes at least two."
            )
        if any(
            overlaps(group, cfg.contact_observation_group) for group in actor_obs_groups
        ):
            raise ValueError("Contact labels cannot enter actor observations.")
        super().__init__(
            obs_dim=obs_dim,
            action_dim=action_dim,
            device=device,
            actor=_TemporalActor(cfg, action_dim),
            critic=_mlp(critic_obs_dim, cfg.critic_hidden_sizes, 1),
            critic_obs_dim=critic_obs_dim,
            actor_obs_groups=actor_obs_groups,
            critic_obs_groups=critic_obs_groups,
            actor_obs_normalization=actor_obs_normalization,
            critic_obs_normalization=critic_obs_normalization,
            initial_action_std=initial_action_std,
            action_std_range=action_std_range,
        )
        self.cfg = cfg
        if cfg.architecture_version == "official_v1":
            if actor_obs_normalization:
                self.actor_obs_normalizer = _ExportNormalizer(obs_dim).to(device)
            if critic_obs_normalization:
                self.critic_obs_normalizer = _ExportNormalizer(critic_obs_dim).to(
                    device
                )
        self.contact_decoder: nn.Module | None = None
        self.training_observation_groups: dict[str, tuple[str, ...]] = {}
        self.training_field_specs: dict[str, tuple[tuple[int, ...], torch.dtype]] = {}
        if cfg.contact_points:
            input_dim = (
                cfg.latent_dim
                if cfg.history_length and cfg.use_history
                else cfg.frame_dim
            )
            self.contact_decoder = _mlp(
                input_dim,
                cfg.decoder_hidden_sizes,
                cfg.contact_points * cfg.contact_classes,
            ).to(device)
            self.training_observation_groups["contact_labels"] = (
                cfg.contact_observation_group,
            )
            self.training_field_specs["contact_labels"] = (
                (cfg.contact_points,),
                torch.long,
            )
        weights = cfg.contact_class_weights
        if weights is not None and (
            len(weights) != cfg.contact_classes
            or any(weight <= 0 for weight in weights)
        ):
            raise ValueError(
                "contact_class_weights must contain one positive weight per class."
            )
        self.register_buffer(
            "contact_class_weights",
            (
                None
                if weights is None
                else torch.tensor(weights, dtype=torch.float32, device=device)
            ),
        )

    def evaluate_actions(self, tensordict: TensorDict) -> TensorDict:
        features, latent = self.actor.features(
            self.actor_obs_normalizer(tensordict["obs"])
        )
        mean = self.actor.head(features)
        dist = Normal(mean, self.action_std.expand_as(mean), validate_args=False)
        fields = {
            "sample_log_prob": dist.log_prob(tensordict["action"]).sum(-1),
            "entropy": dist.entropy().sum(-1),
            "value": self._value(self._critic_observation(tensordict)),
            "action_mean": mean,
            "action_std": dist.stddev,
        }
        if self.contact_decoder is not None:
            fields["contact_logits"] = self.contact_decoder(latent).reshape(
                -1, self.cfg.contact_points, self.cfg.contact_classes
            )
        return TensorDict(
            fields, batch_size=tensordict.batch_size, device=tensordict.device
        )

    def auxiliary_loss(self, batch: TensorDict, evaluated: TensorDict) -> torch.Tensor:
        """Return weighted mean contact cross-entropy for this action's state."""
        if self.contact_decoder is None:
            raise ValueError("Contact supervision requires contact_points > 0.")
        labels = batch["contact_labels"]
        if labels.dtype != torch.long:
            raise TypeError(
                "contact_labels must contain integer class indices (torch.long)."
            )
        return nn.functional.cross_entropy(
            evaluated["contact_logits"].flatten(0, 1),
            labels.flatten(),
            weight=self.contact_class_weights,
        )

    def optimization_parameter_groups(
        self,
    ) -> tuple[tuple[nn.Parameter, ...], tuple[nn.Parameter, ...]]:
        actor, critic = super().optimization_parameter_groups()
        decoder = (
            ()
            if self.contact_decoder is None
            else tuple(self.contact_decoder.parameters())
        )
        return (*actor, *decoder), critic

    def export_actor(self) -> nn.Module:
        """Copy deterministic inference modules without critic or contact decoder.

        The returned module takes the same flattened proprioceptive history as
        training. History acquisition and previous-action bookkeeping stay with
        the deployment environment.
        """
        return nn.Sequential(
            deepcopy(self.actor_obs_normalizer), deepcopy(self.actor)
        ).eval()
