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

from typing import Dict

import torch
from tensordict import TensorDict

from embodichain.learning.rl.buffer import transition_view
from embodichain.learning.rl.utils import AlgorithmCfg
from embodichain.utils import configclass
from .common import compute_gae
from .base import BaseAlgorithm

__all__ = ["PPO", "PPOCfg"]


@configclass
class PPOCfg(AlgorithmCfg):
    """Configuration for the PPO algorithm."""

    n_epochs: int = 10
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    use_clipped_value_loss: bool = False
    schedule: str = "fixed"
    desired_kl: float | None = None
    minimum_learning_rate: float = 1e-5
    maximum_learning_rate: float = 1e-2
    num_mini_batches: int | None = None
    normalize_advantage_per_mini_batch: bool = False


class PPO(BaseAlgorithm[TensorDict]):
    """PPO algorithm consuming TensorDict rollouts."""

    def __init__(self, cfg: PPOCfg, policy):
        if cfg.schedule not in {"fixed", "adaptive"}:
            raise ValueError("PPO schedule must be 'fixed' or 'adaptive'.")
        if cfg.schedule == "adaptive" and (
            cfg.desired_kl is None or cfg.desired_kl <= 0.0
        ):
            raise ValueError("Adaptive PPO requires a positive desired_kl.")
        if cfg.minimum_learning_rate <= 0.0:
            raise ValueError("minimum_learning_rate must be positive.")
        if cfg.maximum_learning_rate < cfg.minimum_learning_rate:
            raise ValueError(
                "maximum_learning_rate must be greater than or equal to "
                "minimum_learning_rate."
            )
        if cfg.num_mini_batches is not None and cfg.num_mini_batches <= 0:
            raise ValueError("num_mini_batches must be positive when provided.")
        self.cfg = cfg
        self.policy = policy
        self.device = torch.device(cfg.device)
        self._setup_optimization(cfg, policy.parameters())

    def update(self, rollout: TensorDict) -> Dict[str, float]:
        """Update the policy using a collected rollout."""
        rollout = rollout.clone()
        compute_gae(rollout, gamma=self.cfg.gamma, gae_lambda=self.cfg.gae_lambda)
        flat_rollout = transition_view(rollout, flatten=True)

        if not self.cfg.normalize_advantage_per_mini_batch:
            advantages = flat_rollout["advantage"]
            flat_rollout["advantage"] = (advantages - advantages.mean()) / (
                advantages.std() + 1e-8
            )

        total_actor_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_kl = 0.0
        total_steps = 0

        minibatch_indices = self._minibatch_indices(flat_rollout.batch_size[0])
        for _ in range(self.cfg.n_epochs):
            for indices in minibatch_indices:
                batch = flat_rollout[indices]
                old_logprobs = batch["sample_log_prob"].clone()
                returns = batch["return"].clone()
                batch_advantages = batch["advantage"]
                if self.cfg.normalize_advantage_per_mini_batch:
                    batch_advantages = (batch_advantages - batch_advantages.mean()) / (
                        batch_advantages.std() + 1e-8
                    )
                batch_advantages = batch_advantages.detach()

                policy_module = getattr(self.policy, "module", self.policy)
                eval_batch = policy_module.evaluate_actions(batch)
                logprobs = eval_batch["sample_log_prob"]
                entropy = eval_batch["entropy"]
                values = eval_batch["value"]
                kl = self._update_adaptive_learning_rate(batch, eval_batch)
                ratio = (logprobs - old_logprobs).exp()
                surr1 = ratio * batch_advantages
                surr2 = (
                    torch.clamp(
                        ratio, 1.0 - self.cfg.clip_coef, 1.0 + self.cfg.clip_coef
                    )
                    * batch_advantages
                )
                actor_loss = -torch.min(surr1, surr2).mean()
                value_loss = self._value_loss(values, returns, batch["value"])
                entropy_loss = -entropy.mean()

                loss = (
                    actor_loss
                    + self.cfg.vf_coef * value_loss
                    + self.cfg.ent_coef * entropy_loss
                )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self._clip_gradients()
                self.optimizer.step()

                bs = batch.batch_size[0]
                total_actor_loss += actor_loss.item() * bs
                total_value_loss += value_loss.item() * bs
                total_entropy += (-entropy_loss.item()) * bs
                total_kl += kl * bs
                total_steps += bs

        self._step_scheduler()
        return {
            "actor_loss": total_actor_loss / max(1, total_steps),
            "value_loss": total_value_loss / max(1, total_steps),
            "entropy": total_entropy / max(1, total_steps),
            "kl": total_kl / max(1, total_steps),
            "learning_rate": self.current_learning_rate(),
        }

    def _minibatch_indices(self, total: int) -> tuple[torch.Tensor, ...]:
        """Build one shuffled mini-batch partition reused across PPO epochs."""
        if self.cfg.num_mini_batches is not None:
            mini_batch_size = total // self.cfg.num_mini_batches
            if mini_batch_size <= 0:
                raise ValueError(
                    "num_mini_batches cannot exceed the rollout transition count."
                )
            usable = self.cfg.num_mini_batches * mini_batch_size
            permutation = torch.randperm(usable, device=self.device)
            return tuple(
                permutation[index * mini_batch_size : (index + 1) * mini_batch_size]
                for index in range(self.cfg.num_mini_batches)
            )
        if self.cfg.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        permutation = torch.randperm(total, device=self.device)
        return tuple(
            permutation[start : start + self.cfg.batch_size]
            for start in range(0, total, self.cfg.batch_size)
        )

    def _value_loss(
        self,
        values: torch.Tensor,
        returns: torch.Tensor,
        old_values: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the configured PPO value objective."""
        if not self.cfg.use_clipped_value_loss:
            return torch.nn.functional.mse_loss(values, returns)
        value_delta = (values - old_values).clamp(
            -self.cfg.clip_coef,
            self.cfg.clip_coef,
        )
        clipped_values = old_values + value_delta
        value_losses = (values - returns).square()
        clipped_value_losses = (clipped_values - returns).square()
        return torch.maximum(value_losses, clipped_value_losses).mean()

    @torch.no_grad()
    def _update_adaptive_learning_rate(
        self,
        batch: TensorDict,
        eval_batch: TensorDict,
    ) -> float:
        """Apply exact diagonal-Gaussian KL scheduling and return mean KL."""
        if self.cfg.schedule != "adaptive":
            return 0.0
        desired_kl = self.cfg.desired_kl
        if desired_kl is None:
            raise RuntimeError("adaptive PPO requires desired_kl")
        for name in ("action_mean", "action_std"):
            if name not in batch.keys():
                raise KeyError(f"adaptive PPO requires rollout field '{name}'")
        old_mean = batch["action_mean"]
        old_std = batch["action_std"].clamp_min(1e-8)
        new_mean = eval_batch["action_mean"]
        new_std = eval_batch["action_std"].clamp_min(1e-8)
        kl = torch.sum(
            torch.log(new_std / old_std)
            + (old_std.square() + (old_mean - new_mean).square())
            / (2.0 * new_std.square())
            - 0.5,
            dim=-1,
        ).mean()
        mean_kl = float(kl)
        learning_rate = self.current_learning_rate()
        if mean_kl > desired_kl * 2.0:
            learning_rate = max(
                self.cfg.minimum_learning_rate,
                learning_rate / 1.5,
            )
        elif 0.0 < mean_kl < desired_kl * 0.5:
            learning_rate = min(
                self.cfg.maximum_learning_rate,
                learning_rate * 1.5,
            )
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = learning_rate
        return mean_kl

    def _clip_gradients(self) -> None:
        """Clip actor and critic gradients and reject non-finite updates."""
        policy_module = getattr(self.policy, "module", self.policy)
        parameter_groups = getattr(policy_module, "optimization_parameter_groups", None)
        groups = (
            parameter_groups()
            if parameter_groups is not None
            else (tuple(self.policy.parameters()),)
        )
        try:
            for parameters in groups:
                torch.nn.utils.clip_grad_norm_(
                    parameters,
                    self.cfg.max_grad_norm,
                    error_if_nonfinite=True,
                )
        except RuntimeError as exc:
            raise FloatingPointError(
                "PPO update produced non-finite gradients; optimizer step skipped."
            ) from exc
