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

import math
from copy import deepcopy
from typing import Dict, Iterator

import torch
from tensordict import TensorDict

from embodichain.agents.rl.utils import AlgorithmCfg
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
    reset_every_rollout: bool = True
    truncate_at_first_done: bool = True


class GRPO(BaseAlgorithm):
    """Group Relative Policy Optimization on top of TensorDict rollouts."""

    def __init__(self, cfg: GRPOCfg, policy):
        if cfg.group_size < 2:
            raise ValueError(
                f"GRPO requires group_size >= 2 for within-group normalization, got {cfg.group_size}."
            )
        self.cfg = cfg
        self.policy = policy
        self.device = torch.device(cfg.device)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.learning_rate)
        if self.cfg.kl_coef > 0.0:
            self.ref_policy = deepcopy(policy).to(self.device).eval()
            for param in self.ref_policy.parameters():
                param.requires_grad_(False)
        else:
            self.ref_policy = None

    def _compute_step_returns_and_mask(
        self, rewards: torch.Tensor, dones: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute discounted returns and valid-step mask over `[N, T]` rollout."""
        n_envs, t_steps = rewards.shape
        seq_mask = torch.ones(
            (n_envs, t_steps), dtype=torch.float32, device=self.device
        )
        step_returns = torch.zeros(
            (n_envs, t_steps), dtype=torch.float32, device=self.device
        )

        alive = torch.ones(n_envs, dtype=torch.float32, device=self.device)
        for t in range(t_steps):
            seq_mask[:, t] = alive
            if self.cfg.truncate_at_first_done:
                alive = alive * (~dones[:, t]).float()

        running_return = torch.zeros(n_envs, dtype=torch.float32, device=self.device)
        for t in reversed(range(t_steps)):
            running_return = (
                rewards[:, t] + self.cfg.gamma * running_return * (~dones[:, t]).float()
            )
            step_returns[:, t] = running_return

        return step_returns, seq_mask

    def _compute_step_group_advantages(
        self, step_returns: torch.Tensor, seq_mask: torch.Tensor
    ) -> torch.Tensor:
        """Normalize per-step returns within each environment group."""
        n_envs, t_steps = step_returns.shape
        group_size = self.cfg.group_size

        returns_grouped = step_returns.view(n_envs // group_size, group_size, t_steps)
        mask_grouped = seq_mask.view(n_envs // group_size, group_size, t_steps)

        valid_count = mask_grouped.sum(dim=1, keepdim=True)
        valid_count_safe = torch.clamp(valid_count, min=1.0)

        group_mean = (returns_grouped * mask_grouped).sum(
            dim=1, keepdim=True
        ) / valid_count_safe
        diff_sq = ((returns_grouped - group_mean) ** 2) * mask_grouped
        group_var = diff_sq.sum(dim=1, keepdim=True) / valid_count_safe
        group_std = torch.sqrt(group_var)

        advantages = (returns_grouped - group_mean) / (group_std + self.cfg.eps)
        return advantages.view(n_envs, t_steps) * seq_mask

    def update(self, rollout: TensorDict) -> Dict[str, float]:
        rollout = rollout.clone()
        num_envs = rollout.batch_size[0]
        if num_envs % self.cfg.group_size != 0:
            raise ValueError(
                f"GRPO requires num_envs divisible by group_size, got "
                f"num_envs={num_envs}, group_size={self.cfg.group_size}."
            )

        rewards = rollout["next", "reward"].float()
        dones = rollout["next", "done"].bool()
        step_returns, seq_mask = self._compute_step_returns_and_mask(rewards, dones)
        rollout["advantage"] = self._compute_step_group_advantages(
            step_returns, seq_mask
        )
        rollout["seq_mask"] = seq_mask
        rollout["seq_return"] = step_returns

        flat_rollout = rollout.reshape(math.prod(rollout.batch_size))

        total_actor_loss = 0.0
        total_entropy = 0.0
        total_kl = 0.0
        total_weight = 0.0

        for _ in range(self.cfg.n_epochs):
            for batch in self._iterate_minibatches(flat_rollout, self.cfg.batch_size):
                old_logprobs = batch["sample_log_prob"].clone()
                advantages = batch["advantage"].detach()
                seq_mask_batch = batch["seq_mask"].float()

                eval_batch = self.policy.evaluate_actions(batch.clone())
                logprobs = eval_batch["sample_log_prob"]
                entropy = eval_batch["entropy"]
                ratio = (logprobs - old_logprobs).exp()
                surr1 = ratio * advantages
                surr2 = (
                    torch.clamp(
                        ratio, 1.0 - self.cfg.clip_coef, 1.0 + self.cfg.clip_coef
                    )
                    * advantages
                )
                actor_num = -(torch.min(surr1, surr2) * seq_mask_batch).sum()
                denom = torch.clamp(seq_mask_batch.sum(), min=1.0)
                actor_loss = actor_num / denom

                entropy_loss = -(entropy * seq_mask_batch).sum() / denom

                if self.ref_policy is not None:
                    with torch.no_grad():
                        ref_batch = self.ref_policy.evaluate_actions(batch.clone())
                        ref_logprobs = ref_batch["sample_log_prob"]
                    log_ref_over_pi = ref_logprobs - logprobs
                    kl_per = torch.exp(log_ref_over_pi) - log_ref_over_pi - 1.0
                    kl = (kl_per * seq_mask_batch).sum() / denom
                else:
                    kl = torch.tensor(0.0, device=self.device)

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
                masked_entropy = (entropy * seq_mask_batch).sum() / denom
                total_entropy += masked_entropy.item() * weight
                total_kl += kl.item() * weight
                total_weight += weight

        return {
            "actor_loss": total_actor_loss / max(1.0, total_weight),
            "entropy": total_entropy / max(1.0, total_weight),
            "approx_ref_kl": total_kl / max(1.0, total_weight),
        }

    def _iterate_minibatches(
        self, rollout: TensorDict, batch_size: int
    ) -> Iterator[TensorDict]:
        total = rollout.batch_size[0]
        indices = torch.randperm(total, device=self.device)
        for start in range(0, total, batch_size):
            yield rollout[indices[start : start + batch_size]]
