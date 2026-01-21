# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
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

import torch
from typing import Dict, Any, Tuple, Callable

from embodichain.agents.rl.utils import AlgorithmCfg, flatten_dict_observation
from embodichain.agents.rl.buffer import RolloutBuffer
from embodichain.utils import configclass
from .base import BaseAlgorithm


@configclass
class PPOCfg(AlgorithmCfg):
    """Configuration for the PPO algorithm."""

    n_epochs: int = 10
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5


class PPO(BaseAlgorithm):
    """PPO algorithm operating via Policy and RolloutBuffer (algo-agnostic design)."""

    def __init__(self, cfg: PPOCfg, policy):
        self.cfg = cfg
        self.policy = policy
        self.device = torch.device(cfg.device)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.learning_rate)
        self.buffer: RolloutBuffer | None = None
        # no per-rollout aggregation for dense logging

    def _compute_gae(
        self, rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Internal method to compute GAE. Only called by collect_rollout."""
        T, N = rewards.shape
        advantages = torch.zeros_like(rewards, device=self.device)
        last_adv = torch.zeros(N, device=self.device)
        for t in reversed(range(T)):
            next_value = values[t + 1] if t < T - 1 else torch.zeros_like(values[0])
            not_done = (~dones[t]).float()
            delta = rewards[t] + self.cfg.gamma * next_value * not_done - values[t]
            last_adv = (
                delta + self.cfg.gamma * self.cfg.gae_lambda * not_done * last_adv
            )
            advantages[t] = last_adv
        returns = advantages + values
        return advantages, returns

    def initialize_buffer(
        self, num_steps: int, num_envs: int, obs_dim: int, action_dim: int
    ):
        """Initialize the rollout buffer. Called by trainer before first rollout."""
        self.buffer = RolloutBuffer(
            num_steps, num_envs, obs_dim, action_dim, self.device
        )

    def collect_rollout(
        self,
        env,
        policy,
        obs: torch.Tensor,
        num_steps: int,
        on_step_callback: Callable | None = None,
    ) -> Dict[str, Any]:
        """Collect a rollout. Algorithm controls the data collection process."""
        if self.buffer is None:
            raise RuntimeError(
                "Buffer not initialized. Call initialize_buffer() first."
            )

        policy.train()
        self.buffer.step = 0
        current_obs = obs

        for t in range(num_steps):
            # Get action from policy
            actions, log_prob, value = policy.get_action(
                current_obs, deterministic=False
            )

            # Step environment
            result = env.step(actions)
            next_obs, reward, terminated, truncated, env_info = result
            done = terminated | truncated
            # Light dtype normalization
            reward = reward.float()
            done = done.bool()

            # Flatten dict observation from ObservationManager if needed
            if isinstance(next_obs, dict):
                next_obs = flatten_dict_observation(next_obs)

            # Add to buffer
            self.buffer.add(current_obs, actions, reward, done, value, log_prob)

            # Dense logging is handled in Trainer.on_step via info; no aggregation here
            # Call callback for statistics and logging
            if on_step_callback is not None:
                on_step_callback(current_obs, actions, reward, done, env_info, next_obs)

            current_obs = next_obs

        # Compute advantages/returns and attach to buffer extras
        adv, ret = self._compute_gae(
            self.buffer.rewards, self.buffer.values, self.buffer.dones
        )
        self.buffer.set_extras({"advantages": adv, "returns": ret})

        # No aggregated logging results; Trainer performs dense per-step logging
        return {}

    def update(self) -> dict:
        """Update the policy using the collected rollout buffer."""
        if self.buffer is None:
            raise RuntimeError("Buffer not initialized. Call collect_rollout() first.")

        # Normalize advantages (optional, common default)
        adv = self.buffer._extras.get("advantages")
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        total_actor_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_steps = 0

        for _ in range(self.cfg.n_epochs):
            for batch in self.buffer.iterate_minibatches(self.cfg.batch_size):
                obs = batch["obs"]
                actions = batch["actions"]
                old_logprobs = batch["logprobs"]
                returns = batch["returns"]
                advantages = (
                    (batch["advantages"] - adv.mean()) / (adv.std() + 1e-8)
                ).detach()

                logprobs, entropy, values = self.policy.evaluate_actions(obs, actions)
                ratio = (logprobs - old_logprobs).exp()
                surr1 = ratio * advantages
                surr2 = (
                    torch.clamp(
                        ratio, 1.0 - self.cfg.clip_coef, 1.0 + self.cfg.clip_coef
                    )
                    * advantages
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

                bs = obs.shape[0]
                total_actor_loss += actor_loss.item() * bs
                total_value_loss += value_loss.item() * bs
                total_entropy += (-entropy_loss.item()) * bs
                total_steps += bs

        return {
            "actor_loss": total_actor_loss / max(1, total_steps),
            "value_loss": total_value_loss / max(1, total_steps),
            "entropy": total_entropy / max(1, total_steps),
        }
