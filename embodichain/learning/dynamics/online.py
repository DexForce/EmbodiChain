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

"""Count and order measured collection, model warmup and online update cycles."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any
import math

from embodichain.utils import configclass

__all__ = ["OnlineLearningCfg", "OnlineLearningSchedule"]


@configclass
class OnlineLearningCfg:
    """Vector-step update ratio and distinct model-only warmup budget."""

    warmup_updates: int = 500
    updates_per_collect: float = 0.4
    model_updates_per_cycle: int = 10


class OnlineLearningSchedule:
    """Drive supplied update operations synchronously at graph-safe boundaries.

    Collection commits detached physical data. Model updates finish before each
    policy update starts; the policy callback completes its full backward pass
    before returning. Callbacks own tensors, optimizers and their checkpoints.
    This schedule owns progress and fractional update credit, independently of
    environment count and network architecture.
    """

    def __init__(self, cfg: OnlineLearningCfg) -> None:
        if (
            cfg.warmup_updates < 0
            or cfg.model_updates_per_cycle < 0
            or not math.isfinite(cfg.updates_per_collect)
            or cfg.updates_per_collect <= 0
        ):
            raise ValueError("Invalid online learning update budget.")
        self.cfg = cfg
        self.iterations = self.prefill_steps = self.physical_transitions = 0
        self.warmup_updates = self.model_updates = self.policy_updates = 0
        self.credit = 0.0

    def collect(self, callback: Callable[[], int], *, prefill: bool = False) -> None:
        """Commit one vector collection and record its physical transition count."""
        self.physical_transitions += callback()
        self.prefill_steps += int(prefill)

    def warmup(self, update_model: Callable[[], dict[str, float]]) -> dict[str, float]:
        """Perform one pending model-only update without consuming online credit."""
        if self.warmup_updates >= self.cfg.warmup_updates:
            raise RuntimeError("Model warmup is already complete.")
        metrics = update_model()
        self.warmup_updates += 1
        self.model_updates += 1
        return metrics

    def step(
        self,
        collect: Callable[[], int],
        update_model: Callable[[], dict[str, float]],
        update_policy: Callable[[], dict[str, float]],
    ) -> dict[str, float]:
        """Collect once and execute the due model/policy cycles in dependency order."""
        if self.warmup_updates != self.cfg.warmup_updates:
            raise RuntimeError("Complete model warmup before online learning.")
        self.collect(collect)
        self.credit += self.cfg.updates_per_collect
        metrics: dict[str, float] = {}
        samples: dict[str, int] = {}

        def merge(prefix: str, values: Mapping[str, float]) -> None:
            for key, value in values.items():
                name = prefix + key
                metrics[name] = metrics.get(name, 0.0) + float(value)
                samples[name] = samples.get(name, 0) + 1

        while self.credit >= 1.0 - 1e-12:
            for _ in range(self.cfg.model_updates_per_cycle):
                merge("wm/", update_model())
                self.model_updates += 1
            merge("apg/", update_policy())
            self.policy_updates += 1
            self.credit = max(0.0, self.credit - 1.0)
        self.iterations += 1
        return {key: value / samples[key] for key, value in metrics.items()}

    def state_dict(self) -> dict[str, Any]:
        """Serialize update progress; replay contents are independently owned."""
        return {
            "config": self.cfg.to_dict(),
            **{
                key: getattr(self, key)
                for key in (
                    "iterations",
                    "prefill_steps",
                    "physical_transitions",
                    "warmup_updates",
                    "model_updates",
                    "policy_updates",
                    "credit",
                )
            },
        }

    def validate_state_dict(self, state: Mapping[str, Any]) -> None:
        """Check schedule identity and counters before replacing live progress."""
        if (
            state.keys() != self.state_dict().keys()
            or state["config"] != self.cfg.to_dict()
        ):
            raise ValueError("Online schedule checkpoint configuration differs.")
        for key in (
            "iterations",
            "prefill_steps",
            "physical_transitions",
            "warmup_updates",
            "model_updates",
            "policy_updates",
        ):
            if not isinstance(state[key], int) or state[key] < 0:
                raise ValueError(
                    "Online schedule counters must be nonnegative integers."
                )
        if (
            not 0 <= state["credit"] < 1
            or state["warmup_updates"] > self.cfg.warmup_updates
        ):
            raise ValueError("Invalid online schedule credit or warmup progress.")

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore completed update counters and fractional scheduling credit."""
        self.validate_state_dict(state)
        for key, value in state.items():
            if key != "config":
                setattr(self, key, value)
