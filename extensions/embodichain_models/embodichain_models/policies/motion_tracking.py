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

"""Goal-conditioned motion policy and privileged scalar value model."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from typing import Any

from gymnasium import spaces
import torch
from torch import nn
from tensordict import TensorDict

from embodichain.learning.models.mlp import MLP
from embodichain.learning.models.normalizer import EmpiricalNormalizer
from embodichain.learning.rl.policies import Policy, register_policy
from embodichain.utils import configclass

__all__ = ["MotionTrackingPolicyCfg", "MotionTrackingPolicy"]


@configclass
class MotionTrackingPolicyCfg:
    """Dimensions, normalization and MLPs of a deterministic tracking policy."""

    history_dim: int = 930
    goal_dim: int = 730
    critic_dim: int = 1009
    action_dim: int = 29
    latent_dim: int = 128
    encoder_hidden: tuple[int, ...] = (2048, 1024, 512, 256)
    decoder_hidden: tuple[int, ...] = (2048, 1024, 512, 256)
    critic_hidden: tuple[int, ...] = (256, 128)
    normalization_epsilon: float = 1e-2
    amp_bfloat16: bool = False


class _GoalActor(nn.Module):
    def __init__(self, cfg: MotionTrackingPolicyCfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.obs_norm = EmpiricalNormalizer(
            cfg.history_dim, cfg.normalization_epsilon, variance_epsilon=True, clip=10.0
        )
        self.goal_norm = EmpiricalNormalizer(
            cfg.goal_dim, cfg.normalization_epsilon, variance_epsilon=True, clip=10.0
        )
        self.encoder = MLP(cfg.goal_dim, cfg.latent_dim, cfg.encoder_hidden)
        self.decoder = MLP(
            cfg.latent_dim + cfg.history_dim, cfg.action_dim, cfg.decoder_hidden
        )
        nn.init.orthogonal_(self.decoder[-1].weight, gain=0.01)
        nn.init.zeros_(self.decoder[-1].bias)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        history, goal = observation.split((self.cfg.history_dim, self.cfg.goal_dim), -1)
        history, goal = self.obs_norm(history), self.goal_norm(goal)
        with (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if self.cfg.amp_bfloat16 and observation.is_cuda
            else nullcontext()
        ):
            latent = self.encoder(goal).float().tanh()
            return self.decoder(torch.cat((latent, history), -1)).float()


class MotionTrackingPolicy(Policy):
    """Deterministic goal encoder/decoder with a separately optimized critic.

    Actor observations concatenate oldest-first proprioceptive history and future
    goals. Privileged critic inputs are supplied by the task. Exploration noise
    belongs to environment collection; the policy has no learned action variance.
    Normalization updates are explicit and are checkpointed with network weights.
    """

    actor_obs_groups = ("policy",)
    critic_obs_groups = ("critic",)
    uses_separate_critic_obs = True

    def __init__(
        self, cfg: MotionTrackingPolicyCfg, device: torch.device | str = "cpu"
    ) -> None:
        super().__init__()
        cfg.validate()
        if (
            min(
                cfg.history_dim,
                cfg.goal_dim,
                cfg.critic_dim,
                cfg.action_dim,
                cfg.latent_dim,
            )
            < 1
        ):
            raise ValueError("Tracking model dimensions must be positive.")
        self.cfg = cfg
        self.device = torch.device(device)
        self.obs_dim = self.actor_obs_dim = cfg.history_dim + cfg.goal_dim
        self.critic_obs_dim, self.action_dim = cfg.critic_dim, cfg.action_dim
        self.actor = _GoalActor(cfg)
        self.critic_obs_normalizer = EmpiricalNormalizer(
            cfg.critic_dim, cfg.normalization_epsilon, variance_epsilon=True, clip=10.0
        )
        self.critic = MLP(cfg.critic_dim, 1, cfg.critic_hidden)
        self.to(self.device)

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        obs_space: spaces.Space,
        action_space: spaces.Space,
        device: torch.device,
    ) -> MotionTrackingPolicy:
        """Build from task dimensions and validate observation spaces.

        Args:
            config: Policy block with model dimensions and architecture in ``cfg``.
            obs_space: Named actor and critic observation spaces.
            action_space: Flat continuous action space.
            device: Device for model parameters and normalization buffers.

        Returns:
            Policy compatible with the supplied observation and action spaces.
        """
        if not isinstance(obs_space, spaces.Dict) or not isinstance(
            action_space, spaces.Box
        ):
            raise TypeError(
                "Motion tracking requires Dict observations and Box actions."
            )
        values = dict(config.get("cfg", {}))
        for name in ("encoder_hidden", "decoder_hidden", "critic_hidden"):
            if name in values:
                values[name] = tuple(values[name])
        cfg = MotionTrackingPolicyCfg(**values)
        cfg.validate()
        expected = {
            "policy": cfg.history_dim + cfg.goal_dim,
            "critic": cfg.critic_dim,
        }
        if any(
            key not in obs_space.spaces or obs_space[key].shape != (size,)
            for key, size in expected.items()
        ) or action_space.shape != (cfg.action_dim,):
            raise ValueError(
                "Tracking policy dimensions differ from environment spaces."
            )
        return cls(cfg, device)

    def optimization_parameter_groups(
        self,
    ) -> tuple[tuple[nn.Parameter, ...], tuple[nn.Parameter, ...]]:
        """Return disjoint actor and critic weights."""
        return tuple(self.actor.parameters()), tuple(self.critic.parameters())

    def forward(
        self, tensordict: TensorDict, deterministic: bool = False
    ) -> TensorDict:
        """Write deterministic actions; noise is supplied by the environment."""
        tensordict["action"] = self.actor(tensordict["obs"])
        return tensordict

    def get_differentiable_action(
        self, tensordict: TensorDict, deterministic: bool = False
    ) -> TensorDict:
        """Evaluate the actor with its state and parameter graph intact."""
        return self.forward(tensordict, deterministic)

    def get_value(self, tensordict: TensorDict) -> TensorDict:
        """Evaluate a scalar value from the task's privileged features."""
        value = self.critic_obs_normalizer(tensordict["critic_obs"])
        with (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if self.cfg.amp_bfloat16 and value.is_cuda
            else nullcontext()
        ):
            tensordict["value"] = self.critic(value).float().squeeze(-1)
        return tensordict

    def evaluate_actions(self, tensordict: TensorDict) -> TensorDict:
        """Reject likelihood-based updates for this deterministic policy."""
        raise NotImplementedError("MotionTrackingPolicy uses pathwise APG updates.")

    @torch.no_grad()
    def update_actor_normalization(self, observation: torch.Tensor) -> None:
        """Merge measured history and clean goals before collecting actions."""
        history, goal = observation.reshape(-1, self.obs_dim).split(
            (self.cfg.history_dim, self.cfg.goal_dim), -1
        )
        self.actor.obs_norm.update(history)
        self.actor.goal_norm.update(goal)

    @torch.no_grad()
    def update_critic_normalization(self, observation: torch.Tensor) -> None:
        """Merge detached imagined critic inputs after actor backpropagation."""
        self.critic_obs_normalizer.update(observation.reshape(-1, self.critic_obs_dim))

    def get_extra_state(self) -> dict[str, Any]:
        """Save the model configuration used to interpret network weights."""
        return self.cfg.to_dict()

    def set_extra_state(self, state: dict[str, Any]) -> None:
        """Reject checkpoints with a different model/normalization identity."""
        identity = dict(state)
        current = self.cfg.to_dict()
        identity.pop("amp_bfloat16", None)
        current.pop("amp_bfloat16", None)
        if identity != current:
            raise ValueError("Tracking policy checkpoint configuration differs.")


register_policy("motion_tracking", MotionTrackingPolicy)
