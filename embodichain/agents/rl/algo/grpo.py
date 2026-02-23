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

from copy import deepcopy
from typing import Any, Callable, Dict

import torch

from embodichain.agents.rl.buffer import RolloutBuffer
from embodichain.agents.rl.utils import AlgorithmCfg, flatten_dict_observation
from embodichain.utils import configclass
from .base import BaseAlgorithm


@configclass
class GRPOCfg(AlgorithmCfg):
    """Configuration for GRPO."""

    n_epochs: int = 10
    clip_coef: float = 0.2
    ent_coef: float = 0.0
    kl_coef: float = 0.02
    group_size: int = 4
    eps: float = 1e-8
    # Collect fresh groups every rollout instead of continuing from prior states.
    reset_every_rollout: bool = True
    # If True, do not optimize steps after the first done in each environment
    # during a rollout. This better matches "one completion per prompt".
    truncate_at_first_done: bool = True


class GRPO(BaseAlgorithm):
    """Group Relative Policy Optimization on top of RolloutBuffer."""

    def __init__(self, cfg: GRPOCfg, policy):
        if cfg.group_size < 2:
            raise ValueError(
                f"GRPO requires group_size >= 2 for within-group normalization, got {cfg.group_size}."
            )
        self.cfg = cfg
        self.policy = policy
        self.device = torch.device(cfg.device)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.learning_rate)
        self.buffer: RolloutBuffer | None = None
        # Frozen reference policy for KL regularization.
        self.ref_policy = deepcopy(policy).to(self.device).eval()
        for param in self.ref_policy.parameters():
            param.requires_grad_(False)

    def initialize_buffer(
        self, num_steps: int, num_envs: int, obs_dim: int, action_dim: int
    ) -> None:
        if num_envs % self.cfg.group_size != 0:
            raise ValueError(
                f"GRPO requires num_envs divisible by group_size, got "
                f"num_envs={num_envs}, group_size={self.cfg.group_size}."
            )
        self.buffer = RolloutBuffer(
            num_steps, num_envs, obs_dim, action_dim, self.device
        )

    def _compute_sequence_returns_and_mask(
        self, rewards: torch.Tensor, dones: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute per-env sequence return and valid-step mask.

        Returns:
            seq_returns: shape [N], one scalar sequence return per environment.
            seq_mask: shape [T, N], 1 for valid sequence steps, 0 otherwise.
        """
        t_steps, n_envs = rewards.shape
        seq_mask = torch.ones(
            (t_steps, n_envs), dtype=torch.float32, device=self.device
        )
        seq_returns = torch.zeros(n_envs, dtype=torch.float32, device=self.device)

        alive = torch.ones(n_envs, dtype=torch.float32, device=self.device)
        discount = 1.0
        for t in range(t_steps):
            seq_mask[t] = alive
            seq_returns += alive * rewards[t] * discount
            if self.cfg.truncate_at_first_done:
                alive = alive * (~dones[t]).float()
            discount *= self.cfg.gamma
        return seq_returns, seq_mask

    def _compute_group_advantages(self, seq_returns: torch.Tensor) -> torch.Tensor:
        """Normalize sequence returns within each fixed group."""
        n_envs = seq_returns.shape[0]
        group_size = self.cfg.group_size
        returns_grouped = seq_returns.view(n_envs // group_size, group_size)
        group_mean = returns_grouped.mean(dim=1, keepdim=True)
        # unbiased=False avoids NaN when group size is small.
        group_std = returns_grouped.std(dim=1, keepdim=True, unbiased=False)
        adv = (returns_grouped - group_mean) / (group_std + self.cfg.eps)
        return adv.reshape(-1)

    def collect_rollout(
        self,
        env,
        policy,
        obs: torch.Tensor,
        num_steps: int,
        on_step_callback: Callable | None = None,
    ) -> Dict[str, Any]:
        if self.buffer is None:
            raise RuntimeError(
                "Buffer not initialized. Call initialize_buffer() first."
            )

        policy.train()
        self.buffer.step = 0
        current_obs = obs

        if self.cfg.reset_every_rollout:
            current_obs, _ = env.reset()
            if isinstance(current_obs, dict):
                current_obs = flatten_dict_observation(current_obs)

        for _ in range(num_steps):
            actions, log_prob, _ = policy.get_action(current_obs, deterministic=False)
            action_type = getattr(env, "action_type", "delta_qpos")
            action_dict = {action_type: actions}
            next_obs, reward, terminated, truncated, env_info = env.step(action_dict)
            done = (terminated | truncated).bool()
            reward = reward.float()

            if isinstance(next_obs, dict):
                next_obs = flatten_dict_observation(next_obs)

            # GRPO does not use value function targets; store zeros in value slot.
            value_placeholder = torch.zeros_like(reward)
            self.buffer.add(
                current_obs, actions, reward, done, value_placeholder, log_prob
            )

            if on_step_callback is not None:
                on_step_callback(current_obs, actions, reward, done, env_info, next_obs)
            current_obs = next_obs

        seq_returns, seq_mask = self._compute_sequence_returns_and_mask(
            self.buffer.rewards, self.buffer.dones
        )
        group_adv = self._compute_group_advantages(seq_returns)
        advantages = (
            group_adv.unsqueeze(0).expand(num_steps, -1).contiguous() * seq_mask
        )

        self.buffer.set_extras(
            {
                "advantages": advantages,
                "seq_mask": seq_mask,
                "seq_return": seq_returns.unsqueeze(0).expand(num_steps, -1),
            }
        )
        return {}

    def update(self) -> Dict[str, float]:
        if self.buffer is None:
            raise RuntimeError("Buffer not initialized. Call collect_rollout() first.")

        total_actor_loss = 0.0
        total_entropy = 0.0
        total_kl = 0.0
        total_weight = 0.0

        for _ in range(self.cfg.n_epochs):
            for batch in self.buffer.iterate_minibatches(self.cfg.batch_size):
                obs = batch["obs"]
                actions = batch["actions"]
                old_logprobs = batch["logprobs"]
                advantages = batch["advantages"].detach()
                seq_mask = batch["seq_mask"].float()

                logprobs, entropy, _ = self.policy.evaluate_actions(obs, actions)
                ratio = (logprobs - old_logprobs).exp()
                surr1 = ratio * advantages
                surr2 = (
                    torch.clamp(
                        ratio, 1.0 - self.cfg.clip_coef, 1.0 + self.cfg.clip_coef
                    )
                    * advantages
                )
                actor_num = -(torch.min(surr1, surr2) * seq_mask).sum()
                denom = torch.clamp(seq_mask.sum(), min=1.0)
                actor_loss = actor_num / denom

                entropy_loss = -(entropy * seq_mask).sum() / denom

                with torch.no_grad():
                    ref_logprobs, _, _ = self.ref_policy.evaluate_actions(obs, actions)
                # Positive sample-based KL approximation:
                # exp(log pi_ref - log pi) - (log pi_ref - log pi) - 1
                # ~= D_KL(pi || pi_ref) for small shifts.
                log_ref_over_pi = ref_logprobs - logprobs
                kl_per = torch.exp(log_ref_over_pi) - log_ref_over_pi - 1.0
                kl = (kl_per * seq_mask).sum() / denom

                loss = (
                    actor_loss
                    + self.cfg.kl_coef * kl
                    + self.cfg.ent_coef * entropy_loss
                )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.cfg.max_grad_norm
                )
                self.optimizer.step()

                weight = float(denom.item())
                total_actor_loss += actor_loss.item() * weight
                masked_entropy = (entropy * seq_mask).sum() / denom
                total_entropy += masked_entropy.item() * weight
                total_kl += kl.item() * weight
                total_weight += weight

        return {
            "actor_loss": total_actor_loss / max(1.0, total_weight),
            "entropy": total_entropy / max(1.0, total_weight),
            "approx_ref_kl": total_kl / max(1.0, total_weight),
        }