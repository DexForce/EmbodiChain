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

import copy
from typing import Any, Callable, Dict

import torch

from embodichain.agents.rl.buffer import RolloutBuffer
from embodichain.agents.rl.utils import flatten_dict_observation
from embodichain.utils import configclass
from .ppo import PPOCfg
from .base import BaseAlgorithm


@configclass
class GRPOCfg(PPOCfg):
    """Configuration for strict GRPO."""

    group_size: int = 8
    group_norm_eps: float = 1e-8
    use_group_std_norm: bool = True
    kl_coef: float = 0.02
    kl_target: float = 0.0
    state_sync_enabled: bool = True


class GRPO(BaseAlgorithm):
    """GRPO using episodic group-relative returns and frozen reference policy KL.

    Like LLM GRPO: same initial state (prompt) -> G trajectories -> compare G episode returns.
    - Sync only at rollout start (reset), not per-step, to preserve MDP dynamics.
    - Advantage = (episode_return - mean) / std within each group, broadcast to all timesteps.
    Requires num_envs % group_size == 0.
    """

    def __init__(self, cfg: GRPOCfg, policy):
        self.cfg = cfg
        self.policy = policy
        self.device = torch.device(cfg.device)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.learning_rate)
        self.buffer: RolloutBuffer | None = None

        # reference policy is copied once and frozen.
        self.ref_policy = copy.deepcopy(policy)
        self.ref_policy.to(self.device)
        self.ref_policy.eval()
        for p in self.ref_policy.parameters():
            p.requires_grad_(False)

        self.current_kl_coef = float(self.cfg.kl_coef)

    def initialize_buffer(
        self, num_steps: int, num_envs: int, obs_dim: int, action_dim: int
    ):
        G = self.cfg.group_size
        if num_envs % G != 0:
            raise ValueError(
                f"GRPO requires num_envs ({num_envs}) divisible by group_size ({G}). "
                f"Use num_envs in {{64, 32, 16, 8}} or set group_size to a divisor of num_envs."
            )
        self.buffer = RolloutBuffer(
            num_steps, num_envs, obs_dim, action_dim, self.device
        )

    def _compute_discounted_returns(
        self, rewards: torch.Tensor, dones: torch.Tensor
    ) -> torch.Tensor:
        T, N = rewards.shape
        returns = torch.zeros_like(rewards, device=self.device)
        next_return = torch.zeros(N, device=self.device)
        for t in reversed(range(T)):
            not_done = (~dones[t]).float()
            next_return = rewards[t] + self.cfg.gamma * next_return * not_done
            returns[t] = next_return
        return returns

    def _compute_group_advantages(
        self, rewards: torch.Tensor, dones: torch.Tensor
    ) -> torch.Tensor:
        """Compute episodic group-relative advantages (broadcast to all timesteps).

        For each group of G envs (same initial state at reset), compute undiscounted
        episode return per trajectory, then A_i = (R_i - mean(R_group)) / (std + eps).
        Broadcast this scalar advantage to all timesteps of that trajectory.

        Uses dones mask to avoid mixing rewards from multiple episodes when env auto-resets.
        Shifted dones ensure the terminal step reward (e.g., +100 success / -100 crash) is
        included, while rewards from the new episode after reset are zeroed out.
        """
        T, N = rewards.shape
        G = self.cfg.group_size
        eps = float(self.cfg.group_norm_eps)

        if N % G != 0:
            raise ValueError(
                f"GRPO requires num_envs ({N}) to be divisible by group_size ({G}). "
                f"Set num_envs to a multiple of group_size (e.g., 64, 32)."
            )

        shifted_dones = torch.roll(dones, shifts=1, dims=0)
        shifted_dones[0] = False
        mask = (~shifted_dones).float().cumprod(dim=0)
        masked_rewards = rewards * mask
        episode_returns = masked_rewards.sum(dim=0)
        num_groups = N // G
        advantages_per_env = torch.zeros(N, device=self.device, dtype=rewards.dtype)

        for g in range(num_groups):
            start = g * G
            group_returns = episode_returns[start : start + G]
            centered = group_returns - group_returns.mean()
            if self.cfg.use_group_std_norm:
                std = group_returns.std(unbiased=False) + eps
                centered = centered / std
            advantages_per_env[start : start + G] = centered

        return advantages_per_env.unsqueeze(0).expand(T, -1)

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

        if self.cfg.state_sync_enabled and hasattr(env, "sync_group_states"):
            env.sync_group_states(self.cfg.group_size)
            raw_obs = env.get_obs()
            current_obs = (
                flatten_dict_observation(raw_obs)
                if isinstance(raw_obs, dict)
                else raw_obs
            )

        for _ in range(num_steps):
            actions, log_prob, value = policy.get_action(
                current_obs, deterministic=False
            )
            action_type = getattr(env, "action_type", "delta_qpos")
            result = env.step({action_type: actions})
            next_obs, reward, terminated, truncated, env_info = result
            done = (terminated | truncated).bool()
            reward = reward.float()

            if isinstance(next_obs, dict):
                next_obs = flatten_dict_observation(next_obs)

            self.buffer.add(current_obs, actions, reward, done, value, log_prob)
            if on_step_callback is not None:
                on_step_callback(current_obs, actions, reward, done, env_info, next_obs)
            current_obs = next_obs

        returns = self._compute_discounted_returns(
            self.buffer.rewards, self.buffer.dones
        )
        advantages = self._compute_group_advantages(
            self.buffer.rewards, self.buffer.dones
        )
        self.buffer.set_extras({"returns": returns, "advantages": advantages})
        return {}

    def update(self) -> Dict[str, float]:
        if self.buffer is None:
            raise RuntimeError("Buffer not initialized. Call collect_rollout() first.")

        if (
            "advantages" not in self.buffer._extras
            or "returns" not in self.buffer._extras
        ):
            raise RuntimeError(
                "Missing advantages/returns in rollout extras for GRPO update."
            )

        total_policy_loss = 0.0
        total_entropy = 0.0
        total_kl = 0.0
        total_group_adv_mean = 0.0
        total_group_adv_std = 0.0
        total_clip_frac = 0.0
        total_steps = 0

        for _ in range(self.cfg.n_epochs):
            for batch in self.buffer.iterate_minibatches(self.cfg.batch_size):
                obs = batch["obs"]
                actions = batch["actions"]
                old_logprobs = batch["logprobs"]
                advantages = batch["advantages"].detach()

                logprobs, entropy, _ = self.policy.evaluate_actions(obs, actions)
                with torch.no_grad():
                    ref_logprobs, _, _ = self.ref_policy.evaluate_actions(obs, actions)

                ratio = (logprobs - old_logprobs).exp()
                clipped_ratio = torch.clamp(
                    ratio, 1.0 - self.cfg.clip_coef, 1.0 + self.cfg.clip_coef
                )
                surr1 = ratio * advantages
                surr2 = clipped_ratio * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                kl = (logprobs - ref_logprobs).mean()
                entropy_bonus = entropy.mean()
                loss = (
                    policy_loss
                    + self.current_kl_coef * kl
                    - self.cfg.ent_coef * entropy_bonus
                )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.cfg.max_grad_norm
                )
                self.optimizer.step()

                if self.cfg.kl_target > 0.0:
                    if kl.item() > 1.5 * self.cfg.kl_target:
                        self.current_kl_coef *= 1.5
                    elif kl.item() < self.cfg.kl_target / 1.5:
                        self.current_kl_coef /= 1.5
                    self.current_kl_coef = float(
                        max(1e-6, min(10.0, self.current_kl_coef))
                    )

                bs = obs.shape[0]
                clip_frac = (torch.abs(ratio - 1.0) > self.cfg.clip_coef).float().mean()
                total_policy_loss += policy_loss.item() * bs
                total_entropy += entropy_bonus.item() * bs
                total_kl += kl.item() * bs
                total_group_adv_mean += advantages.mean().item() * bs
                total_group_adv_std += advantages.std(unbiased=False).item() * bs
                total_clip_frac += clip_frac.item() * bs
                total_steps += bs

        denom = max(1, total_steps)
        return {
            "policy_loss": total_policy_loss / denom,
            "entropy": total_entropy / denom,
            "kl": total_kl / denom,
            "kl_coef": float(self.current_kl_coef),
            "group_adv_mean": total_group_adv_mean / denom,
            "group_adv_std": total_group_adv_std / denom,
            "group_clip_frac": total_clip_frac / denom,
        }
