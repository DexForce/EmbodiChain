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

"""Analytic Policy Gradient (APG) algorithm for differentiable environments.

APG performs direct backpropagation through differentiable environment dynamics
to compute exact policy gradients. Unlike PPO/GRPO, it does not use a replay
buffer or minibatch updates. Instead, it rolls out episodes through a
differentiable simulator and backpropagates the reward signal directly through
the environment dynamics into the policy parameters.

Key features:
  - Segment-based gradient truncation to limit backprop chain length
  - Optional critic bootstrapping at segment boundaries
  - Monte Carlo or learned-value bootstrapping modes
  - Running observation normalization for stable training

Reference:
    Analytic Policy Gradients for differentiable environments.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from embodichain.agents.rl.utils import AlgorithmCfg
from embodichain.utils import configclass

from .base import BaseAlgorithm

__all__ = ["APGCfg", "APG", "RunningObsNormalizer"]


@configclass
class APGCfg(AlgorithmCfg):
    """Configuration for the APG algorithm.

    Args:
        num_grad_steps: Number of gradient steps per iteration.
        segment_length: Segment length for value bootstrapping. ``0`` means
            use the full episode length (no segmentation).
        bootstrap: Bootstrap mode for segmented APG. ``"mc"`` uses collected
            future rewards (Monte Carlo), ``"critic"`` uses a learned value
            function.
        critic_coef: Coefficient for the critic loss when using critic
            bootstrap.
        critic_lr: Separate learning rate for the critic. ``None`` means use
            the same learning rate as the actor.
        ent_coef: Entropy bonus coefficient. ``0`` disables entropy
            regularization.
        max_episode_steps: Maximum episode length for the differentiable
            environment.
        anneal_lr: Whether to linearly anneal the learning rate.
    """

    num_grad_steps: int = 8
    segment_length: int = 0
    bootstrap: Literal["mc", "critic"] = "critic"
    critic_coef: float = 0.5
    critic_lr: float | None = None
    ent_coef: float = 0.0
    max_episode_steps: int = 30
    anneal_lr: bool = True


class RunningObsNormalizer:
    """Welford-style running mean/std tracker for observation normalization.

    Maintains online estimates of the observation mean and variance using
    Welford's algorithm, and normalizes observations to zero mean and unit
    variance for stable policy training.

    Args:
        obs_dim: Dimensionality of the observation vector.
        device: Torch device for the statistics tensors.
    """

    def __init__(self, obs_dim: int, device: torch.device | str = "cpu"):
        self.mean = torch.zeros(obs_dim, device=device)
        self.var = torch.ones(obs_dim, device=device)
        self.count: float = 1e-4

    def update(self, obs_batch: torch.Tensor) -> None:
        """Update running statistics with a new batch of observations.

        Args:
            obs_batch: Observation batch of shape ``[batch, obs_dim]``.
        """
        batch_mean = obs_batch.mean(dim=0)
        batch_var = obs_batch.var(dim=0, unbiased=False)
        batch_count = obs_batch.shape[0]
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean = self.mean + delta * batch_count / total
        m2 = (
            self.var * self.count
            + batch_var * batch_count
            + delta**2 * self.count * batch_count / total
        )
        self.var = m2 / total
        self.count = total

    def normalize(self, obs: torch.Tensor) -> torch.Tensor:
        """Normalize observations using running statistics.

        Args:
            obs: Observation tensor to normalize.

        Returns:
            Normalized observation tensor.
        """
        return (obs - self.mean) / (self.var.sqrt() + 1e-8)


class APG(BaseAlgorithm):
    """Analytic Policy Gradient algorithm for differentiable environments.

    APG backpropagates through differentiable environment dynamics to compute
    exact policy gradients. It supports segment-based gradient truncation with
    optional critic or Monte Carlo bootstrapping at segment boundaries.

    Unlike PPO/GRPO, APG does not consume a pre-collected rollout buffer.
    Instead, the :meth:`update` method performs a full gradient step by rolling
    out the environment and backpropagating through the dynamics.

    Args:
        cfg: APG configuration.
        policy: The actor-critic policy (must expose ``actor``, ``critic``,
            ``log_std``, and ``_distribution`` via the standard
            :class:`~embodichain.agents.rl.models.ActorCritic` interface).
    """

    def __init__(self, cfg: APGCfg, policy: nn.Module):
        self.cfg = cfg
        self.policy = policy
        self.device = torch.device(cfg.device)
        self.obs_normalizer: RunningObsNormalizer | None = None

        # Build optimizer with optional separate critic LR
        actor_params = self._get_actor_params()
        param_groups = [{"params": actor_params, "lr": cfg.learning_rate}]

        use_critic = (
            cfg.segment_length > 0
            and cfg.segment_length < cfg.max_episode_steps
            and cfg.bootstrap == "critic"
        )
        self._use_critic = use_critic

        if use_critic:
            critic_lr = (
                cfg.critic_lr if cfg.critic_lr is not None else cfg.learning_rate
            )
            critic_params = list(self._get_critic().parameters())
            param_groups.append({"params": critic_params, "lr": critic_lr})

        self.optimizer = torch.optim.Adam(param_groups)

    def _get_actor_params(self) -> list[nn.Parameter]:
        """Collect actor parameters (actor network + log_std)."""
        raw = getattr(self.policy, "module", self.policy)
        params = list(raw.actor.parameters())
        if hasattr(raw, "log_std"):
            params.append(raw.log_std)
        return params

    def _get_critic(self) -> nn.Module:
        """Return the critic network."""
        raw = getattr(self.policy, "module", self.policy)
        return raw.critic

    def _get_apg_action(self, norm_obs: torch.Tensor) -> torch.Tensor:
        """Sample a differentiable action via reparameterization.

        Args:
            norm_obs: Normalized observation tensor.

        Returns:
            Differentiable action tensor (uses rsample for gradient flow).
        """
        raw = getattr(self.policy, "module", self.policy)
        dist = raw._distribution(norm_obs)
        return dist.rsample()

    def _get_entropy(self, norm_obs: torch.Tensor) -> torch.Tensor:
        """Compute per-dimension entropy summed over action dimensions.

        Args:
            norm_obs: Normalized observation tensor.

        Returns:
            Entropy tensor of shape ``[num_envs]``.
        """
        raw = getattr(self.policy, "module", self.policy)
        dist = raw._distribution(norm_obs)
        return dist.entropy().sum(-1)

    def _get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute value estimate for given observations.

        Args:
            obs: Observation tensor.

        Returns:
            Value tensor of shape ``[num_envs]``.
        """
        raw = getattr(self.policy, "module", self.policy)
        return raw.critic(obs).squeeze(-1)

    def update(self, rollout: TensorDict) -> Dict[str, float]:
        """Perform one APG gradient step.

        .. attention::
            APG does not consume a pre-collected rollout buffer. The
            ``rollout`` parameter carries the differentiable environment and
            current observation state needed for the APG loop. It must contain:

            - ``"env"``: The differentiable environment instance.
            - ``"obs"``: Current observation tensor ``[num_envs, obs_dim]``.
            - ``"num_envs"``: Number of parallel environments.

        Args:
            rollout: TensorDict with env, obs, and num_envs.

        Returns:
            Dictionary of training metrics.
        """
        env = rollout["env"]
        obs: torch.Tensor = rollout["obs"]
        num_envs: int = rollout["num_envs"]

        # Lazy-init observation normalizer
        obs_dim = obs.shape[-1]
        if self.obs_normalizer is None:
            self.obs_normalizer = RunningObsNormalizer(obs_dim, self.device)

        cfg = self.cfg
        effective_seg = (
            cfg.segment_length if cfg.segment_length > 0 else cfg.max_episode_steps
        )
        num_segments = math.ceil(cfg.max_episode_steps / effective_seg)
        use_seg = effective_seg < cfg.max_episode_steps

        self.optimizer.zero_grad()

        policy_loss = torch.tensor(0.0, device=self.device)
        all_obs_for_norm: list[torch.Tensor] = []
        all_entropies: list[torch.Tensor] = []

        # Per-segment storage
        seg_rewards_all: list[list[torch.Tensor]] = []
        seg_norm_obs_end: list[torch.Tensor] = []
        seg_norm_obs_start: list[torch.Tensor] = []
        seg_steps_list: list[int] = []

        # ===== Segmented forward pass =====
        for seg_idx in range(num_segments):
            seg_start = seg_idx * effective_seg
            seg_end = min(seg_start + effective_seg, cfg.max_episode_steps)
            seg_steps = seg_end - seg_start
            seg_steps_list.append(seg_steps)

            seg_norm_obs_start.append(self.obs_normalizer.normalize(obs.detach()))

            segment_rewards: list[torch.Tensor] = []
            for step in range(seg_steps):
                norm_obs = self.obs_normalizer.normalize(obs)
                all_obs_for_norm.append(obs.detach())

                action = self._get_apg_action(norm_obs)

                if cfg.ent_coef > 0:
                    ent = self._get_entropy(norm_obs)
                    all_entropies.append(ent)

                obs, reward, terminated, truncated, infos = env.step(action)
                obs = obs.to(self.device)
                reward = reward.to(self.device)

                segment_rewards.append(reward.view(-1))

            seg_rewards_all.append(segment_rewards)
            seg_norm_obs_end.append(self.obs_normalizer.normalize(obs))

            # Detach at segment boundary to limit gradient chain
            obs = obs.detach()
            if hasattr(env, "detach_state"):
                env.detach_state()

        # Update obs normalizer
        self.obs_normalizer.update(torch.cat(all_obs_for_norm, dim=0))

        # ===== Compute per-segment returns and bootstrap =====
        critic_values_list: list[torch.Tensor] = []
        critic_targets_list: list[torch.Tensor] = []

        # Pre-compute MC future returns (backwards accumulation)
        mc_future: list[torch.Tensor | None] = [None] * num_segments
        if use_seg and cfg.bootstrap == "mc":
            running_future = torch.zeros(num_envs, device=self.device)
            for seg_idx in reversed(range(num_segments)):
                mc_future[seg_idx] = running_future.clone()
                seg_steps = seg_steps_list[seg_idx]
                running_future = running_future * (cfg.gamma**seg_steps)
                for t, r in enumerate(seg_rewards_all[seg_idx]):
                    running_future = running_future + r.detach() * (cfg.gamma**t)

        # Build per-segment losses
        for seg_idx in range(num_segments):
            seg_steps = seg_steps_list[seg_idx]
            seg_rewards_t = torch.stack(seg_rewards_all[seg_idx])
            discounts = cfg.gamma ** torch.arange(
                seg_steps, device=self.device, dtype=torch.float32
            )
            seg_return = (seg_rewards_t * discounts.unsqueeze(1)).sum(dim=0)

            is_last = seg_idx == num_segments - 1
            if not is_last and use_seg:
                if cfg.bootstrap == "mc":
                    bootstrap_value = (cfg.gamma**seg_steps) * mc_future[seg_idx]
                elif cfg.bootstrap == "critic":
                    bootstrap_value = (cfg.gamma**seg_steps) * self._get_value(
                        seg_norm_obs_end[seg_idx]
                    )
                else:
                    bootstrap_value = torch.zeros(num_envs, device=self.device)
            else:
                bootstrap_value = torch.zeros(num_envs, device=self.device)

            policy_loss = policy_loss - (seg_return + bootstrap_value).mean()

            # Collect critic training data
            if use_seg and cfg.bootstrap == "critic":
                critic_pred = self._get_value(seg_norm_obs_start[seg_idx])
                critic_values_list.append(critic_pred)
                critic_targets_list.append((seg_return + bootstrap_value).detach())

        # ===== Critic loss =====
        if use_seg and cfg.bootstrap == "critic" and len(critic_values_list) > 0:
            critic_values_t = torch.cat(critic_values_list)
            critic_targets_t = torch.cat(critic_targets_list)
            critic_loss = F.mse_loss(critic_values_t, critic_targets_t)
        else:
            critic_loss = torch.tensor(0.0, device=self.device)

        # ===== Total loss =====
        loss = policy_loss + cfg.critic_coef * critic_loss

        if cfg.ent_coef > 0 and all_entropies:
            entropy_loss = torch.stack(all_entropies).mean()
            loss = loss - cfg.ent_coef * entropy_loss
        else:
            entropy_loss = torch.tensor(0.0, device=self.device)

        loss.backward()

        nn.utils.clip_grad_norm_(self._get_actor_params(), cfg.max_grad_norm)
        if self._use_critic:
            nn.utils.clip_grad_norm_(self._get_critic().parameters(), cfg.max_grad_norm)
        self.optimizer.step()

        # Store updated obs for caller
        rollout["obs"] = obs

        # Compute total reward for logging
        total_reward = sum(r.sum().item() for seg in seg_rewards_all for r in seg)
        horizon_return = total_reward / max(1, num_envs)

        return {
            "policy_loss": policy_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy": entropy_loss.item() if cfg.ent_coef > 0 else 0.0,
            "total_loss": loss.item(),
            "horizon_return": horizon_return,
        }
