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

"""Analytic policy gradients for differentiable environments."""

from __future__ import annotations

from typing import Dict

import torch

from embodichain.learning.rl.collector import DifferentiableRollout
from embodichain.learning.rl.gradients import BatchedGradientNormStats
from embodichain.learning.rl.utils import AlgorithmCfg
from embodichain.utils import configclass

from .base import BaseAlgorithm, RolloutKind

__all__ = [
    "APG",
    "APGCfg",
    "complete_discounted_return",
    "segmented_discounted_return",
]


def segmented_discounted_return(
    rollout: DifferentiableRollout,
    gamma: float,
) -> torch.Tensor:
    """Compute one discounted return per environment within a rollout segment."""
    if rollout.num_steps == 0:
        raise ValueError("Cannot compute returns for an empty rollout.")
    discount = torch.ones_like(rollout.transitions[0].reward)
    returns = torch.zeros_like(discount)
    for transition in rollout.transitions:
        returns = returns + discount * transition.reward
        continuation_discount = discount * gamma
        discount = torch.where(
            transition.done,
            torch.ones_like(discount),
            continuation_discount,
        )
    return returns


def complete_discounted_return(
    rollout: DifferentiableRollout,
    gamma: float,
) -> torch.Tensor:
    """Compute per-environment returns up to the first terminal transition.

    Unlike segmented TBPTT returns, complete-rollout discounting never restarts
    after ``done``. This excludes rewards emitted by any automatic reset during
    the fixed full horizon.

    Args:
        rollout: Complete graph-preserving trajectory.
        gamma: Per-step discount factor.

    Returns:
        One masked discounted return per environment.

    Raises:
        ValueError: If the rollout is empty.
    """
    if rollout.num_steps == 0:
        raise ValueError("Cannot compute returns for an empty rollout.")
    rewards = rollout.rewards
    discounts = torch.as_tensor(gamma, device=rewards.device, dtype=rewards.dtype) ** (
        torch.arange(rollout.num_steps, device=rewards.device, dtype=rewards.dtype)
    )
    return (
        rewards * rollout.alive_mask.to(rewards.dtype) * discounts.unsqueeze(-1)
    ).sum(dim=0)


@configclass
class APGCfg(AlgorithmCfg):
    """Analytic policy-gradient config.

    ``gamma`` applies within each TBPTT segment and restarts after done.
    """

    ent_coef: float = 0.0
    skip_nonfinite_updates: bool = True
    max_grad_norm_before_clip: float = 0.0


class APG(BaseAlgorithm[DifferentiableRollout]):
    """Optimize policy parameters through differentiable rollout rewards."""

    rollout_kind = RolloutKind.DIFFERENTIABLE

    def __init__(self, cfg: APGCfg, policy: torch.nn.Module) -> None:
        if cfg.max_grad_norm_before_clip < 0.0:
            raise ValueError("max_grad_norm_before_clip cannot be negative.")
        self.cfg = cfg
        self.policy = policy
        self.device = torch.device(cfg.device)
        self._setup_optimization(cfg, policy.parameters())
        self._update_active = False
        self._update_valid = True
        self._discount: torch.Tensor | None = None
        self._loss_total = 0.0
        self._objective_total = 0.0
        self._entropy_total = 0.0
        self._num_accumulated_steps = 0
        self._action_gradient_stats: list[BatchedGradientNormStats] = []

    def update(self, rollout: DifferentiableRollout) -> Dict[str, float]:
        """Apply one pathwise-gradient update from a rollout segment."""
        self.begin_update()
        try:
            self.accumulate_segment(rollout)
            return self.finish_update()
        except Exception:
            self.cancel_update()
            raise

    def begin_update(self) -> None:
        if self._update_active:
            raise RuntimeError("An APG optimizer update is already active.")
        self.optimizer.zero_grad(set_to_none=True)
        self._update_active = True
        self._update_valid = True
        self._discount = None
        self._loss_total = 0.0
        self._objective_total = 0.0
        self._entropy_total = 0.0
        self._num_accumulated_steps = 0
        self._action_gradient_stats = []

    def accumulate_segment(self, rollout: DifferentiableRollout) -> None:
        """Accumulate gradients from one TBPTT segment without stepping the optimizer."""
        if not self._update_active:
            raise RuntimeError("Call begin_update() before accumulating a segment.")
        if rollout.num_steps == 0:
            raise ValueError("APG requires a non-empty differentiable rollout.")

        returns, entropy_returns, self._discount = self._discounted_terms(
            rollout,
            initial_discount=self._discount,
        )
        objective = returns.mean()
        entropy = entropy_returns.mean()
        loss = -objective - self.cfg.ent_coef * entropy
        self._accumulate_loss(
            rollout,
            loss=loss,
            objective=objective,
            entropy=entropy,
        )

    def accumulate_complete_rollout(
        self,
        rollout: DifferentiableRollout,
        *,
        objective_scale: float | torch.Tensor = 1.0,
        accumulation_scale: float = 1.0,
    ) -> None:
        """Accumulate one independent full-horizon rollout.

        Args:
            rollout: Complete graph-preserving trajectory.
            objective_scale: Scalar or per-environment return multiplier.
            accumulation_scale: Loss multiplier, normally reciprocal to the
                number of independent rollout microbatches in one update.

        Raises:
            RuntimeError: If no APG update is active.
            ValueError: If the rollout, scaling, or accumulation factor is invalid.
        """
        if not self._update_active:
            raise RuntimeError("Call begin_update() before accumulating a rollout.")
        if rollout.num_steps == 0:
            raise ValueError("APG requires a non-empty differentiable rollout.")
        if accumulation_scale <= 0.0:
            raise ValueError("accumulation_scale must be positive.")

        returns = complete_discounted_return(rollout, self.cfg.gamma)
        scale = torch.as_tensor(
            objective_scale,
            dtype=returns.dtype,
            device=returns.device,
        ).detach()
        if scale.ndim > 1 or (scale.ndim == 1 and scale.shape != returns.shape):
            raise ValueError(
                "objective_scale must be scalar or have one value per environment."
            )
        if not bool(torch.isfinite(scale).all()):
            raise ValueError("objective_scale must contain only finite values.")

        entropy_returns = torch.zeros_like(returns)
        discount = torch.ones_like(returns)
        alive = torch.ones_like(returns, dtype=torch.bool)
        for transition in rollout.transitions:
            if "entropy" in transition.policy_output.keys():
                entropy_returns = entropy_returns + (
                    discount
                    * transition.policy_output["entropy"]
                    * alive.to(discount.dtype)
                )
            alive = alive & ~transition.done
            discount = discount * self.cfg.gamma

        objective = (returns * scale).mean()
        entropy = (entropy_returns * scale).mean()
        loss = (-objective - self.cfg.ent_coef * entropy) * accumulation_scale
        self._accumulate_loss(
            rollout,
            loss=loss,
            objective=objective * accumulation_scale,
            entropy=entropy * accumulation_scale,
        )

    def _accumulate_loss(
        self,
        rollout: DifferentiableRollout,
        *,
        loss: torch.Tensor,
        objective: torch.Tensor,
        entropy: torch.Tensor,
    ) -> None:
        """Validate and backpropagate one contribution to the active update."""
        self._loss_total += float(loss.detach())
        self._objective_total += float(objective.detach())
        self._entropy_total += float(entropy.detach())
        self._num_accumulated_steps += rollout.num_steps
        if rollout.action_gradient_stats is not None:
            self._action_gradient_stats.append(rollout.action_gradient_stats)

        loss_is_finite = bool(torch.isfinite(loss))
        if not loss_is_finite and not self.cfg.skip_nonfinite_updates:
            raise FloatingPointError("APG produced a non-finite loss.")

        # Backward is also the lifecycle boundary for custom differentiable
        # simulator steps. Invoke it even when this accumulation window is
        # already invalid so every rollout can release its retained graph/tape.
        loss.backward()
        parameters = tuple(self.policy.parameters())
        gradients_are_finite = all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in parameters
        )
        if not loss_is_finite or not gradients_are_finite or not self._update_valid:
            if not self.cfg.skip_nonfinite_updates:
                raise FloatingPointError("APG produced a non-finite policy gradient.")
            self._update_valid = False
            self.optimizer.zero_grad(set_to_none=True)
            return

    def finish_update(self) -> Dict[str, float]:
        """Clip gradients and apply one optimizer step."""
        if not self._update_active:
            raise RuntimeError("Call begin_update() before finishing an update.")
        if self._num_accumulated_steps == 0:
            raise RuntimeError("Cannot finish an APG update without any segments.")

        parameters = tuple(self.policy.parameters())
        if not self._update_valid:
            metrics = self._accumulated_metrics(
                grad_norm=float("nan"),
                skipped_update=1.0,
                skipped_excessive_gradient=0.0,
            )
            self._update_active = False
            return metrics
        grad_norm = torch.nn.utils.clip_grad_norm_(
            parameters,
            self.cfg.max_grad_norm,
        )
        excessive_gradient = not bool(torch.isfinite(grad_norm)) or (
            self.cfg.max_grad_norm_before_clip > 0.0
            and float(grad_norm) > self.cfg.max_grad_norm_before_clip
        )
        if excessive_gradient:
            self.optimizer.zero_grad(set_to_none=True)
            metrics = self._accumulated_metrics(
                grad_norm=float(grad_norm.detach()),
                skipped_update=1.0,
                skipped_excessive_gradient=1.0,
            )
            self._update_active = False
            return metrics
        self.optimizer.step()
        self._step_scheduler()
        metrics = self._accumulated_metrics(
            grad_norm=float(grad_norm.detach()),
            skipped_update=0.0,
            skipped_excessive_gradient=0.0,
        )
        self._update_active = False
        return metrics

    def cancel_update(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)
        self._update_active = False
        self._action_gradient_stats = []

    def _discounted_terms(
        self,
        rollout: DifferentiableRollout,
        initial_discount: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        discount = (
            torch.ones_like(rollout.transitions[0].reward)
            if initial_discount is None
            else initial_discount
        )
        returns = torch.zeros_like(discount)
        entropy_returns = torch.zeros_like(discount)
        for transition in rollout.transitions:
            returns = returns + discount * transition.reward
            if "entropy" in transition.policy_output.keys():
                entropy_returns = (
                    entropy_returns + discount * transition.policy_output["entropy"]
                )
            discount = torch.where(
                transition.done,
                torch.ones_like(discount),
                discount * self.cfg.gamma,
            )
        return returns, entropy_returns, discount

    def _accumulated_metrics(
        self,
        *,
        grad_norm: float,
        skipped_update: float,
        skipped_excessive_gradient: float,
    ) -> Dict[str, float]:
        metrics = {
            "loss": self._loss_total,
            "objective": self._objective_total,
            "entropy": self._entropy_total,
            "grad_norm": grad_norm,
            "skipped_update": skipped_update,
            "skipped_excessive_gradient": skipped_excessive_gradient,
            "learning_rate": self.current_learning_rate(),
        }
        if self._action_gradient_stats:
            rows = sum(float(stats.rows) for stats in self._action_gradient_stats)
            finite_rows = sum(
                float(stats.finite_rows) for stats in self._action_gradient_stats
            )
            metrics.update(
                {
                    "action_adjoint_preclip_mean_norm": (
                        sum(
                            float(stats.norm_sum)
                            for stats in self._action_gradient_stats
                        )
                        / finite_rows
                        if finite_rows > 0.0
                        else 0.0
                    ),
                    "action_adjoint_preclip_max_norm": max(
                        float(stats.norm_max) for stats in self._action_gradient_stats
                    ),
                    "action_adjoint_clipped_fraction": (
                        sum(
                            float(stats.clipped_rows)
                            for stats in self._action_gradient_stats
                        )
                        / rows
                        if rows > 0.0
                        else 0.0
                    ),
                    "action_adjoint_nonfinite_fraction": (
                        sum(
                            float(stats.nonfinite_rows)
                            for stats in self._action_gradient_stats
                        )
                        / rows
                        if rows > 0.0
                        else 0.0
                    ),
                }
            )
        return metrics
