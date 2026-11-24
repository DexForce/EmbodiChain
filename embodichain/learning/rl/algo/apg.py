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

import math
from typing import Dict
from contextlib import AbstractContextManager, nullcontext

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
    *,
    terminal_value: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute per-environment returns up to the first terminal transition.

    Unlike segmented TBPTT returns, complete-rollout discounting never restarts
    after ``done``. This excludes rewards emitted by any automatic reset during
    the fixed full horizon.

    Args:
        rollout: Complete graph-preserving trajectory.
        gamma: Per-step discount factor.
        terminal_value: Optional value at the final next state, shaped like
            one reward row. It retains gradients and contributes with discount
            ``gamma**num_steps`` only for rows with no termination or truncation.
            Callers own value estimation and any differentiable soft survival.

    Returns:
        One masked discounted return per environment.

    Raises:
        ValueError: If the rollout is empty or the terminal value has wrong shape.
    """
    if rollout.num_steps == 0:
        raise ValueError("Cannot compute returns for an empty rollout.")
    rewards = rollout.rewards
    discounts = torch.as_tensor(gamma, device=rewards.device, dtype=rewards.dtype) ** (
        torch.arange(rollout.num_steps, device=rewards.device, dtype=rewards.dtype)
    )
    alive_mask = rollout.alive_mask
    returns = (rewards * alive_mask.to(rewards.dtype) * discounts.unsqueeze(-1)).sum(
        dim=0
    )
    if terminal_value is not None:
        if terminal_value.shape != returns.shape:
            raise ValueError("terminal_value must have one value per environment.")
        survives = alive_mask[-1] & ~rollout.transitions[-1].done
        returns = returns + gamma**rollout.num_steps * torch.where(
            survives, terminal_value, torch.zeros_like(terminal_value)
        )
    return returns


@configclass
class APGCfg(AlgorithmCfg):
    """Analytic policy-gradient config.

    ``gamma`` carries across TBPTT segments within an update and restarts after
    done. Complete rollouts stop accumulating rewards after the first done.
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
        self._setup_optimization(cfg, self._optimization_parameters())
        self._update_active = False
        self._update_valid = True
        self._discount: torch.Tensor | None = None
        self._loss_total = 0.0
        self._objective_total = 0.0
        self._entropy_total = 0.0
        self._num_accumulated_steps = 0
        self._action_gradient_stats: list[BatchedGradientNormStats] = []

    def rollout_context(self) -> AbstractContextManager[None]:
        """Return the model-state context used during graph-preserving collection."""
        return nullcontext()

    def _optimization_parameters(self) -> tuple[torch.nn.Parameter, ...]:
        return tuple(self.policy.parameters())

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
        terminal_value: torch.Tensor | None = None,
    ) -> None:
        """Accumulate one independent full-horizon rollout.

        Args:
            rollout: Complete graph-preserving trajectory.
            objective_scale: Scalar or per-environment return multiplier.
            accumulation_scale: Loss multiplier, normally reciprocal to the
                number of independent rollout microbatches in one update.
            terminal_value: Optional graph-connected final-state value. Hard
                terminal/truncated rows do not bootstrap; see
                :func:`complete_discounted_return`. Value fitting and freezing
                estimator weights remain the caller's responsibility.

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

        returns = complete_discounted_return(
            rollout, self.cfg.gamma, terminal_value=terminal_value
        )
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
        scalars = (loss.detach(), objective.detach(), entropy.detach())
        if loss.is_cuda and all(value.device == loss.device for value in scalars):
            # Read reporting scalars at the existing pre-backward safety boundary.
            loss_value, objective_value, entropy_value = (
                torch.stack(scalars).cpu().tolist()
            )
        else:
            loss_value, objective_value, entropy_value = map(float, scalars)
        self._loss_total += loss_value
        self._objective_total += objective_value
        self._entropy_total += entropy_value
        self._num_accumulated_steps += rollout.num_steps
        if rollout.action_gradient_stats is not None:
            self._action_gradient_stats.append(rollout.action_gradient_stats)

        loss_is_finite = math.isfinite(loss_value)
        if not loss_is_finite and not self.cfg.skip_nonfinite_updates:
            raise FloatingPointError("APG produced a non-finite loss.")

        # Backward is also the lifecycle boundary for custom differentiable
        # simulator steps. Invoke it even when this accumulation window is
        # already invalid so every rollout can release its retained graph/tape.
        loss.backward()
        gradients = [
            parameter.grad
            for parameter in self._optimization_parameters()
            if parameter.grad is not None
        ]
        if (
            len(gradients) > 1
            and gradients[0].is_cuda
            and all(
                type(gradient) is torch.Tensor
                and gradient.layout == torch.strided
                and gradient.device == gradients[0].device
                for gradient in gradients
            )
        ):
            # Every segment must reject non-finite gradients before the next one.
            gradients_are_finite = bool(
                torch.stack(
                    [torch.isfinite(gradient).all() for gradient in gradients]
                ).all()
            )
        else:
            gradients_are_finite = all(
                bool(torch.isfinite(gradient).all()) for gradient in gradients
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

        parameters = self._optimization_parameters()
        if not self._update_valid:
            metrics = self._accumulated_metrics(
                grad_norm=float("nan"),
                skipped_update=1.0,
                skipped_excessive_gradient=0.0,
            )
            self._update_active = False
            return metrics
        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(
                parameters,
                self.cfg.max_grad_norm,
            ).detach()
        )
        excessive_gradient = not math.isfinite(grad_norm) or (
            self.cfg.max_grad_norm_before_clip > 0.0
            and grad_norm > self.cfg.max_grad_norm_before_clip
        )
        if excessive_gradient:
            self.optimizer.zero_grad(set_to_none=True)
            metrics = self._accumulated_metrics(
                grad_norm=grad_norm,
                skipped_update=1.0,
                skipped_excessive_gradient=1.0,
            )
            self._update_active = False
            return metrics
        self._after_gradient_clip()
        self.optimizer.step()
        self._step_scheduler()
        metrics = self._accumulated_metrics(
            grad_norm=grad_norm,
            skipped_update=0.0,
            skipped_excessive_gradient=0.0,
        )
        self._update_active = False
        return metrics

    def _after_gradient_clip(self) -> None:
        """Apply algorithm-specific gradient transforms after norm validation."""

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
            fields = [
                (
                    stats.rows,
                    stats.finite_rows,
                    stats.norm_sum,
                    stats.norm_max,
                    stats.clipped_rows,
                    stats.nonfinite_rows,
                )
                for stats in self._action_gradient_stats
            ]
            scalars = [value for values in fields for value in values]
            if scalars[0].is_cuda and all(
                value.device == scalars[0].device for value in scalars
            ):
                # Hooks have finished. Transfer once, retaining the original
                # Python sum/max order and precision across rollout segments.
                values = torch.stack(scalars).cpu().tolist()
                snapshots = [
                    values[index : index + 6] for index in range(0, len(values), 6)
                ]
            else:
                snapshots = [list(map(float, values)) for values in fields]
            rows = sum(values[0] for values in snapshots)
            finite_rows = sum(values[1] for values in snapshots)
            metrics.update(
                {
                    "action_adjoint_preclip_mean_norm": (
                        sum(values[2] for values in snapshots) / finite_rows
                        if finite_rows > 0.0
                        else 0.0
                    ),
                    "action_adjoint_preclip_max_norm": max(
                        values[3] for values in snapshots
                    ),
                    "action_adjoint_clipped_fraction": (
                        sum(values[4] for values in snapshots) / rows
                        if rows > 0.0
                        else 0.0
                    ),
                    "action_adjoint_nonfinite_fraction": (
                        sum(values[5] for values in snapshots) / rows
                        if rows > 0.0
                        else 0.0
                    ),
                }
            )
        return metrics
