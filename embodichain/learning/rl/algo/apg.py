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
from embodichain.learning.rl.utils import AlgorithmCfg
from embodichain.utils import configclass

from .base import BaseAlgorithm

__all__ = ["APG", "APGCfg", "segmented_discounted_return"]


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


@configclass
class APGCfg(AlgorithmCfg):
    """Configuration for analytic policy-gradient updates.

    ``gamma`` is applied within each truncated-backpropagation segment. The
    discount restarts after a terminated or truncated transition.
    """

    ent_coef: float = 0.0
    skip_nonfinite_updates: bool = True


class APG(BaseAlgorithm[DifferentiableRollout]):
    """Optimize policy parameters through differentiable rollout rewards."""

    def __init__(self, cfg: APGCfg, policy: torch.nn.Module) -> None:
        self.cfg = cfg
        self.policy = policy
        self.device = torch.device(cfg.device)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.learning_rate)
        self._update_active = False
        self._update_valid = True
        self._discount: torch.Tensor | None = None
        self._loss_total = 0.0
        self._objective_total = 0.0
        self._entropy_weighted_total = 0.0
        self._num_accumulated_steps = 0

    def update(self, rollout: DifferentiableRollout) -> Dict[str, float]:
        """Apply one pathwise-gradient update from a rollout segment.

        Args:
            rollout: Graph-preserving rollout used for the update.

        Returns:
            Scalar metrics describing the optimizer update.
        """
        self.begin_update()
        try:
            self.accumulate_segment(rollout)
            return self.finish_update()
        except Exception:
            self.cancel_update()
            raise

    def begin_update(self) -> None:
        """Start accumulating segment gradients for one optimizer update."""
        if self._update_active:
            raise RuntimeError("An APG optimizer update is already active.")
        self.optimizer.zero_grad(set_to_none=True)
        self._update_active = True
        self._update_valid = True
        self._discount = None
        self._loss_total = 0.0
        self._objective_total = 0.0
        self._entropy_weighted_total = 0.0
        self._num_accumulated_steps = 0

    def accumulate_segment(self, rollout: DifferentiableRollout) -> None:
        """Accumulate gradients from one TBPTT segment without updating policy.

        Args:
            rollout: Graph-preserving segment from the active update horizon.
        """
        if not self._update_active:
            raise RuntimeError("Call begin_update() before accumulating a segment.")
        if rollout.num_steps == 0:
            raise ValueError("APG requires a non-empty differentiable rollout.")

        returns, self._discount = self._discounted_return(
            rollout,
            initial_discount=self._discount,
        )
        objective = returns.mean()
        entropy = self._mean_entropy(rollout)
        loss = -objective - self.cfg.ent_coef * entropy
        self._loss_total += float(loss.detach())
        self._objective_total += float(objective.detach())
        self._entropy_weighted_total += float(entropy.detach()) * rollout.num_steps
        self._num_accumulated_steps += rollout.num_steps

        if not bool(torch.isfinite(loss)):
            if not self.cfg.skip_nonfinite_updates:
                raise FloatingPointError("APG produced a non-finite loss.")
            self._update_valid = False
            self.optimizer.zero_grad(set_to_none=True)
            return
        if not self._update_valid:
            return

        loss.backward()
        parameters = tuple(self.policy.parameters())
        gradients_are_finite = all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in parameters
        )
        if not gradients_are_finite:
            if not self.cfg.skip_nonfinite_updates:
                raise FloatingPointError("APG produced a non-finite policy gradient.")
            self._update_valid = False
            self.optimizer.zero_grad(set_to_none=True)
            return

    def finish_update(self) -> Dict[str, float]:
        """Clip accumulated gradients and perform one optimizer step.

        Returns:
            Metrics aggregated across all accumulated segments.
        """
        if not self._update_active:
            raise RuntimeError("Call begin_update() before finishing an update.")
        if self._num_accumulated_steps == 0:
            raise RuntimeError("Cannot finish an APG update without any segments.")

        parameters = tuple(self.policy.parameters())
        if not self._update_valid:
            metrics = self._accumulated_metrics(
                grad_norm=float("nan"),
                skipped_update=1.0,
            )
            self._update_active = False
            return metrics
        grad_norm = torch.nn.utils.clip_grad_norm_(
            parameters,
            self.cfg.max_grad_norm,
        )
        self.optimizer.step()
        metrics = self._accumulated_metrics(
            grad_norm=float(grad_norm.detach()),
            skipped_update=0.0,
        )
        self._update_active = False
        return metrics

    def cancel_update(self) -> None:
        """Discard gradients accumulated for the active optimizer update."""
        self.optimizer.zero_grad(set_to_none=True)
        self._update_active = False

    def _discounted_return(
        self,
        rollout: DifferentiableRollout,
        initial_discount: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        discount = (
            torch.ones_like(rollout.transitions[0].reward)
            if initial_discount is None
            else initial_discount
        )
        returns = torch.zeros_like(discount)
        for transition in rollout.transitions:
            returns = returns + discount * transition.reward
            discount = torch.where(
                transition.done,
                torch.ones_like(discount),
                discount * self.cfg.gamma,
            )
        return returns, discount

    def _mean_entropy(self, rollout: DifferentiableRollout) -> torch.Tensor:
        entropies = [
            transition.policy_output["entropy"]
            for transition in rollout.transitions
            if "entropy" in transition.policy_output.keys()
        ]
        if not entropies:
            return rollout.rewards.new_zeros(())
        return torch.stack(entropies).mean()

    def _accumulated_metrics(
        self,
        *,
        grad_norm: float,
        skipped_update: float,
    ) -> Dict[str, float]:
        return {
            "loss": self._loss_total,
            "objective": self._objective_total,
            "entropy": self._entropy_weighted_total / self._num_accumulated_steps,
            "grad_norm": grad_norm,
            "skipped_update": skipped_update,
        }
