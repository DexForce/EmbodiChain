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

    def update(self, rollout: DifferentiableRollout) -> Dict[str, float]:
        """Apply one pathwise-gradient update from a rollout segment."""
        if rollout.num_steps == 0:
            raise ValueError("APG requires a non-empty differentiable rollout.")

        objective = segmented_discounted_return(rollout, self.cfg.gamma).mean()
        entropy = self._mean_entropy(rollout)
        loss = -objective - self.cfg.ent_coef * entropy

        self.optimizer.zero_grad(set_to_none=True)
        if not bool(torch.isfinite(loss)):
            if not self.cfg.skip_nonfinite_updates:
                raise FloatingPointError("APG produced a non-finite loss.")
            return self._skipped_metrics(loss, objective, entropy)

        loss.backward()
        parameters = tuple(self.policy.parameters())
        gradients_are_finite = all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in parameters
        )
        if not gradients_are_finite:
            if not self.cfg.skip_nonfinite_updates:
                raise FloatingPointError("APG produced a non-finite policy gradient.")
            self.optimizer.zero_grad(set_to_none=True)
            return self._skipped_metrics(loss, objective, entropy)

        grad_norm = torch.nn.utils.clip_grad_norm_(
            parameters,
            self.cfg.max_grad_norm,
        )
        self.optimizer.step()
        return {
            "loss": float(loss.detach()),
            "objective": float(objective.detach()),
            "entropy": float(entropy.detach()),
            "grad_norm": float(grad_norm.detach()),
            "skipped_update": 0.0,
        }

    def _mean_entropy(self, rollout: DifferentiableRollout) -> torch.Tensor:
        entropies = [
            transition.policy_output["entropy"]
            for transition in rollout.transitions
            if "entropy" in transition.policy_output.keys()
        ]
        if not entropies:
            return rollout.rewards.new_zeros(())
        return torch.stack(entropies).mean()

    @staticmethod
    def _skipped_metrics(
        loss: torch.Tensor,
        objective: torch.Tensor,
        entropy: torch.Tensor,
    ) -> Dict[str, float]:
        return {
            "loss": float(loss.detach()),
            "objective": float(objective.detach()),
            "entropy": float(entropy.detach()),
            "grad_norm": float("nan"),
            "skipped_update": 1.0,
        }
