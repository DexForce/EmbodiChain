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

"""Compatibility construction for versioned proprioceptive ActorCritic policies."""

from __future__ import annotations

from collections.abc import Sequence
import torch
from torch import nn

from embodichain.utils import configclass
from embodichain.learning.models import MLP
from .actor_critic import ActorCritic
from .auxiliary import ContactPrediction
from .temporal import (
    ObservationHistory,
    TemporalActor,
    TemporalActorCfg,
    TemporalObservationNormalizer,
    _ExportHistoryEncoder as _HistoryEncoder,
)

__all__ = [
    "ObservationHistory",
    "ProprioceptiveActorCriticCfg",
    "ProprioceptiveActorCritic",
]


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


# Preserve the qualified module identities recorded by schema-v2 checkpoints.
# Network behavior is implemented once in the reusable temporal components.
class _ExportNormalizer(TemporalObservationNormalizer):
    pass


class _ExportHistoryEncoder(_HistoryEncoder):
    pass


class _TemporalActor(TemporalActor):
    _official_encoder_type = _ExportHistoryEncoder

    def __init__(self, cfg: ProprioceptiveActorCriticCfg, action_dim: int) -> None:
        super().__init__(
            TemporalActorCfg(
                frame_dim=cfg.frame_dim,
                history_length=cfg.history_length,
                use_history=cfg.use_history,
                latent_dim=cfg.latent_dim,
                hidden_sizes=cfg.actor_hidden_sizes,
                architecture_version=cfg.architecture_version,
            ),
            action_dim,
        )


class ProprioceptiveActorCritic(ActorCritic):
    """Construct composable modules using the historical policy configuration.

    The public name, configuration order, parameter keys and module identities
    remain stable for old weights and complete training checkpoints. Action
    evaluation, auxiliary training and export are owned by :class:`ActorCritic`.
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
        if any(
            overlaps(group, cfg.contact_observation_group) for group in actor_obs_groups
        ):
            raise ValueError("Contact labels cannot enter actor observations.")
        if cfg.contact_points < 0 or cfg.contact_classes < 2:
            raise ValueError(
                "Contact point count must be non-negative and classes at least two."
            )
        actor = _TemporalActor(cfg, action_dim)
        critic = nn.Sequential(
            *MLP(critic_obs_dim, 1, cfg.critic_hidden_sizes).children()
        )
        tasks = {}
        if cfg.contact_points:
            tasks["contact"] = ContactPrediction(
                input_dim=actor.feature_dims["latent"],
                contact_points=cfg.contact_points,
                contact_classes=cfg.contact_classes,
                hidden_sizes=cfg.decoder_hidden_sizes,
                class_weights=cfg.contact_class_weights,
                observation_group=cfg.contact_observation_group,
            )
        super().__init__(
            obs_dim,
            action_dim,
            device,
            actor,
            critic,
            critic_obs_dim=critic_obs_dim,
            actor_obs_groups=actor_obs_groups,
            critic_obs_groups=critic_obs_groups,
            actor_obs_normalization=actor_obs_normalization,
            critic_obs_normalization=critic_obs_normalization,
            initial_action_std=initial_action_std,
            action_std_range=action_std_range,
            auxiliary_tasks=tasks,
            actor_normalizer=(
                _ExportNormalizer(obs_dim)
                if cfg.architecture_version == "official_v1" and actor_obs_normalization
                else None
            ),
            critic_normalizer=(
                _ExportNormalizer(critic_obs_dim)
                if cfg.architecture_version == "official_v1"
                and critic_obs_normalization
                else None
            ),
        )
        self.cfg = cfg
        if not cfg.contact_points:
            self.contact_decoder = None
            weights = ContactPrediction._class_weight_tensor(
                cfg.contact_class_weights, cfg.contact_classes
            )
            self.register_buffer(
                "contact_class_weights", None if weights is None else weights.to(device)
            )
