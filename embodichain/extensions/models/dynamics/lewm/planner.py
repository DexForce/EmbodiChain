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
        horizon: Number of model-action blocks in a candidate sequence.
        num_samples: Candidates drawn per iteration and environment.
        iterations: Number of optimization iterations.
        topk: Number of elite sequences used to fit the next distribution.
        candidate_batch_size: Maximum candidates evaluated together.
        min_std: Minimum sampling standard deviation, in the search space's
            units. Zero leaves the elite deviation without a lower bound.
        sampling_distribution: Initial raw-action distribution: ``bounds`` uses
            the action-box midpoint and half-width; ``training`` uses the frozen
            training action mean and standard deviation. Model inputs and
            returned actions remain in raw controller units. ``normalized``
            searches in training-standardized coordinates, starting at zero
            mean and unit standard deviation, as in the official LeWM example.
        action_selection: Return the first action of the best sampled sequence
            (``best``), or of the final elite mean sequence (``mean``). The latter
            recomputes the cost of that mean sequence.
        clip_candidates: Clip candidates to the controller bounds. The official
            example leaves candidates unbounded; the environment owns clipping
            of workspace positions and gripper commands.
        include_mean: Replace candidate zero with the current distribution mean
            at every iteration, as in the official CEM solver.
        std_correction: Degrees of freedom subtracted from the elite count.
            The original LeWM-compatible solver uses one; zero preserves the
            population-deviation behavior of the bounded planner.
    """

    horizon: int = 5
    num_samples: int = 300
    iterations: int = 30
    topk: int = 30
    candidate_batch_size: int = 300
    min_std: float = 0.01
    sampling_distribution: str = "bounds"
    action_selection: str = "best"
    clip_candidates: bool = True
    include_mean: bool = False
    std_correction: int = 0


class LeWMCEMPlanner:
    """Optimize controller actions against an encoded visual goal.

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
            or not 0 <= c.min_std < float("inf")
            or c.sampling_distribution not in {"bounds", "training", "normalized"}
            or c.action_selection not in {"best", "mean"}
            or type(c.clip_candidates) is not bool
            or type(c.include_mean) is not bool
            or type(c.std_correction) is not int
            or c.std_correction not in (0, 1)
            or c.topk <= c.std_correction
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
        if c.sampling_distribution in {"training", "normalized"} and (
            not torch.isfinite(model.action_mean).all()
            or not torch.isfinite(model.action_std).all()
            or not (model.action_std > 0).all()
        ):
            raise ValueError(
                "CEM requires finite training action moments and positive std"
            )
        self.generator = torch.Generator(device=device).manual_seed(seed)

    @torch.no_grad()
    def plan(
        self, history: torch.Tensor, past_actions: torch.Tensor, goal: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the first action and sequence cost for each environment.

        Args:
            history: Encoded frames ``(B, H, D)``.
            past_actions: Raw actions between history frames ``(B, H-1, A)``.
            goal: Encoded goal images ``(B, D)``.

        Returns:
            First raw actions ``(B, A)`` and costs ``(B,)`` of the selected
            complete sequences, using the configured selection rule.
        """
        sequence, cost = self.plan_sequence(history, past_actions, goal)
        return sequence[:, 0], cost

    @torch.no_grad()
    def plan_sequence(
        self, history: torch.Tensor, past_actions: torch.Tensor, goal: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a complete sequence and its terminal goal cost.

        Args:
            history: Encoded observed frames ``(B, H, D)``.
            past_actions: Executed model-action blocks ``(B, H-1, A)``.
            goal: Encoded goal images ``(B, D)``.

        Returns:
            Raw action blocks ``(B, horizon, A)`` and sequence costs ``(B,)``.
            The host unpacks each block into consecutive controller actions.
        """
        if self.model.training:
            raise ValueError("planning requires the model in eval mode")
        b = history.shape[0]
        c = self.cfg
        normalized = c.sampling_distribution == "normalized"
        if normalized:
            initial_mean = torch.zeros_like(self.low)
            initial_std = torch.ones_like(self.low)
        elif c.sampling_distribution == "training":
            initial_mean = self.model.action_mean
            if c.clip_candidates:
                initial_mean = initial_mean.clamp(self.low, self.high)
            initial_std = self.model.action_std
        else:
            initial_mean = (self.low + self.high) * 0.5
            initial_std = (self.high - self.low) * 0.5
        mean = initial_mean.expand(b, c.horizon, -1).clone()
        std = initial_std.expand_as(mean).clone()
        best_action, best_cost = None, None
        for _ in range(c.iterations):
            noise = torch.randn(
                (b, c.num_samples, c.horizon, self.model.cfg.action_dim),
                generator=self.generator,
                device=mean.device,
            )
            candidates = mean[:, None] + std[:, None] * noise
            if c.include_mean:
                candidates[:, 0] = mean
            raw_candidates = (
                candidates * self.model.action_std + self.model.action_mean
                if normalized
                else candidates
            )
            if c.clip_candidates:
                raw_candidates = raw_candidates.clamp(self.low, self.high)
                candidates = (
                    (raw_candidates - self.model.action_mean) / self.model.action_std
                    if normalized
                    else raw_candidates
                )
            costs = torch.cat(
                [
                    self.model.goal_cost(
                        self.model.rollout(
                            history,
                            past_actions,
                            raw_candidates[:, start : start + c.candidate_batch_size],
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
            std = elites.std(dim=1, correction=c.std_correction).clamp_min(c.min_std)
            action, value = elites[:, 0], values[:, 0]
            if best_cost is None:
                best_action, best_cost = action, value
            else:
                keep = value < best_cost
                best_action = torch.where(keep[:, None, None], action, best_action)
                best_cost = torch.minimum(value, best_cost)
        if c.action_selection == "mean":
            if normalized:
                mean = mean * self.model.action_std + self.model.action_mean
            cost = self.model.goal_cost(
                self.model.rollout(history, past_actions, mean[:, None]), goal
            )[:, 0]
            return mean, cost
        assert best_action is not None and best_cost is not None
        if normalized:
            best_action = best_action * self.model.action_std + self.model.action_mean
        return best_action, best_cost
