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

import math
from typing import Dict, Iterator

import torch
from tensordict import TensorDict

from embodichain.agents.rl.utils import AlgorithmCfg
from embodichain.utils import configclass
from .common import compute_gae
from .base import BaseAlgorithm


@configclass
class PPOCfg(AlgorithmCfg):
    """Configuration for the PPO algorithm."""

    n_epochs: int = 10
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5


class PPO(BaseAlgorithm):
    """PPO algorithm consuming TensorDict rollouts."""

    def __init__(self, cfg: PPOCfg, policy):
        self.cfg = cfg
        self.policy = policy
        self.device = torch.device(cfg.device)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.learning_rate)
        # no per-rollout aggregation for dense logging

    def update(self, rollout: TensorDict) -> Dict[str, float]:
        """Update the policy using a collected rollout."""
        rollout = rollout.clone()
        compute_gae(rollout, gamma=self.cfg.gamma, gae_lambda=self.cfg.gae_lambda)
        flat_rollout = rollout.reshape(math.prod(rollout.batch_size))

        advantages = flat_rollout["advantage"]
        adv_mean = advantages.mean()
        adv_std = advantages.std().clamp_min(1e-8)

        total_actor_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_steps = 0

        for _ in range(self.cfg.n_epochs):
            for batch in self._iterate_minibatches(flat_rollout, self.cfg.batch_size):
                old_logprobs = batch["sample_log_prob"].clone()
                returns = batch["return"].clone()
                batch_advantages = ((batch["advantage"] - adv_mean) / adv_std).detach()

                eval_batch = self.policy.evaluate_actions(batch)
                logprobs = eval_batch["sample_log_prob"]
                entropy = eval_batch["entropy"]
                values = eval_batch["value"]
                ratio = (logprobs - old_logprobs).exp()
                surr1 = ratio * batch_advantages
                surr2 = (
                    torch.clamp(
                        ratio, 1.0 - self.cfg.clip_coef, 1.0 + self.cfg.clip_coef
                    )
                    * batch_advantages
                )
                actor_loss = -torch.min(surr1, surr2).mean()
                value_loss = torch.nn.functional.mse_loss(values, returns)
                entropy_loss = -entropy.mean()

                loss = (
                    actor_loss
                    + self.cfg.vf_coef * value_loss
                    + self.cfg.ent_coef * entropy_loss
                )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.cfg.max_grad_norm
                )
                self.optimizer.step()

                bs = batch.batch_size[0]
                total_actor_loss += actor_loss.item() * bs
                total_value_loss += value_loss.item() * bs
                total_entropy += (-entropy_loss.item()) * bs
                total_steps += bs

        return {
            "actor_loss": total_actor_loss / max(1, total_steps),
            "value_loss": total_value_loss / max(1, total_steps),
            "entropy": total_entropy / max(1, total_steps),
        }

    def _iterate_minibatches(
        self, rollout: TensorDict, batch_size: int
    ) -> Iterator[TensorDict]:
        total = rollout.batch_size[0]
        indices = torch.randperm(total, device=self.device)
        for start in range(0, total, batch_size):
            yield rollout[indices[start : start + batch_size]]
