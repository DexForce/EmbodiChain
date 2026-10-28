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

"""Bounded candidate-batched cross-entropy planning in LeWM latent space."""

from __future__ import annotations

import torch

from embodichain.utils import configclass

from .model import LeWM

__all__ = ["LeWMCEMCfg", "LeWMCEMPlanner"]


@configclass
class LeWMCEMCfg:
    """Configure cross-entropy model predictive control.

    Args:
        horizon: Number of single-control-step candidate actions.
        num_samples: Candidates drawn per iteration and environment.
        iterations: Number of optimization iterations.
        topk: Number of elite sequences used to fit the next distribution.
        candidate_batch_size: Maximum candidates evaluated together.
        min_std: Minimum sampling standard deviation in raw action units.
    """

    horizon: int = 5
    num_samples: int = 300
    iterations: int = 30
    topk: int = 30
    candidate_batch_size: int = 300
    min_std: float = 0.01


class LeWMCEMPlanner:
    """Optimize bounded raw controller actions against an encoded visual goal.

    Args:
        model: Frozen evaluation-mode LeWM.
        low: Per-component lower action bounds.
        high: Per-component upper action bounds.
        cfg: Candidate and optimization settings.
        seed: Seed for a planner-owned device RNG.
    """

    def __init__(
        self,
        model: LeWM,
        low: torch.Tensor,
        high: torch.Tensor,
        cfg: LeWMCEMCfg | None = None,
        *,
        seed: int = 0,
    ) -> None:
        self.model, self.cfg = model, cfg or LeWMCEMCfg()
        self.cfg.validate()
        c = self.cfg
        if (
            any(
                type(v) is not int or v < 1
                for v in (
                    c.horizon,
                    c.num_samples,
                    c.iterations,
                    c.topk,
                    c.candidate_batch_size,
                )
            )
            or c.topk > c.num_samples
            or not 0 < c.min_std < float("inf")
        ):
            raise ValueError("invalid CEM configuration")
        device = next(model.parameters()).device
        self.low = torch.as_tensor(low, device=device, dtype=torch.float32)
        self.high = torch.as_tensor(high, device=device, dtype=torch.float32)
        if (
            self.low.shape != (model.cfg.action_dim,)
            or self.high.shape != self.low.shape
        ):
            raise ValueError("action bounds do not match the model")
        if (
            not torch.isfinite(self.low).all()
            or not torch.isfinite(self.high).all()
            or not (self.high > self.low).all()
        ):
            raise ValueError("CEM requires finite ordered action bounds")
        self.generator = torch.Generator(device=device).manual_seed(seed)

    @torch.no_grad()
    def plan(
        self, history: torch.Tensor, past_actions: torch.Tensor, goal: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the first action and best final cost for each environment.

        Args:
            history: Encoded frames ``(B, H, D)``.
            past_actions: Raw actions between history frames ``(B, H-1, A)``.
            goal: Encoded goal images ``(B, D)``.

        Returns:
            First raw actions ``(B, A)`` and final costs ``(B,)``.
        """
        if self.model.training:
            raise ValueError("planning requires the model in eval mode")
        b = history.shape[0]
        c = self.cfg
        mean = ((self.low + self.high) * 0.5).expand(b, c.horizon, -1).clone()
        std = ((self.high - self.low) * 0.5).expand_as(mean).clone()
        best_action, best_cost = None, None
        for _ in range(c.iterations):
            noise = torch.randn(
                (b, c.num_samples, c.horizon, self.model.cfg.action_dim),
                generator=self.generator,
                device=mean.device,
            )
            candidates = (mean[:, None] + std[:, None] * noise).clamp(
                self.low, self.high
            )
            costs = torch.cat(
                [
                    self.model.goal_cost(
                        self.model.rollout(
                            history,
                            past_actions,
                            candidates[:, start : start + c.candidate_batch_size],
                        ),
                        goal,
                    )
                    for start in range(0, c.num_samples, c.candidate_batch_size)
                ],
                dim=1,
            )
            values, indices = costs.topk(c.topk, dim=1, largest=False, sorted=True)
            elites = candidates.gather(
                1,
                indices[:, :, None, None].expand(
                    -1, -1, c.horizon, self.model.cfg.action_dim
                ),
            )
            mean = elites.mean(dim=1)
            std = elites.std(dim=1, unbiased=False).clamp_min(c.min_std)
            action, value = elites[:, 0, 0], values[:, 0]
            if best_cost is None:
                best_action, best_cost = action, value
            else:
                keep = value < best_cost
                best_action = torch.where(keep[:, None], action, best_action)
                best_cost = torch.minimum(value, best_cost)
        assert best_action is not None and best_cost is not None
        return best_action, best_cost
