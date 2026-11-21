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

"""Cooperative call/deadline accounting, without an execution state machine."""

from __future__ import annotations

from collections.abc import Callable
import time
from threading import Event

from .cfg import SceneRobotFeasibilityCfg

__all__ = ["FeasibilityBudget", "FeasibilityBudgetExhausted"]


class FeasibilityBudgetExhausted(RuntimeError):
    """The shared deadline, cancellation or one call quota has been reached."""


class FeasibilityBudget:
    """One monotonic budget shared by preparation, planning and execution.

    Args:
        cfg: Validated search limits.
        clock: Monotonic clock; injectable for deterministic tests.
        cancelled: Optional caller-owned cancellation event.
    """

    def __init__(
        self,
        cfg: SceneRobotFeasibilityCfg,
        *,
        clock: Callable[[], float] = time.monotonic,
        cancelled: Event | None = None,
    ) -> None:
        self.cfg = cfg
        self.clock = clock
        self.started = clock()
        self.deadline = self.started + cfg.max_wall_time_s
        self.search_deadline = self.started + cfg.max_wall_time_s * cfg.search_fraction
        self.cancelled = cancelled
        self.counts = {"candidates": 0, "ik": 0, "planning": 0, "rollouts": 0}
        self.validation = False
        self._blocked = None

    @property
    def remaining_s(self) -> float:
        """Return remaining time in the current search/validation allocation."""
        end = self.deadline if self.validation else self.search_deadline
        return max(0.0, end - self.clock())

    def check(self) -> None:
        """Raise at cooperative boundaries when time or cancellation is exhausted."""
        if self._blocked is not None and self._blocked[0] == self.validation:
            raise FeasibilityBudgetExhausted(self._blocked[1])
        if self.cancelled is not None and self.cancelled.is_set():
            raise FeasibilityBudgetExhausted("cancelled")
        if self.remaining_s <= 0:
            raise FeasibilityBudgetExhausted("wall_time_exhausted")

    def charge(self, kind: str, amount: int = 1) -> None:
        """Reserve call quota before invoking work; failures still consume it.

        Args:
            kind: candidates, ik, planning or rollouts.
            amount: Number of actual requests, including internal retries.
        """
        self.check()
        if type(amount) is not int or amount < 1:
            raise ValueError("budget charges must be positive integers")
        limits = {
            "candidates": self.cfg.max_candidates,
            "ik": self.cfg.max_ik_calls,
            "planning": self.cfg.max_planning_calls,
            "rollouts": self.cfg.max_rollouts,
        }
        # Validation proposals have a separate fixed quota, while expensive
        # calls and the wall deadline remain shared with search.
        if not self.validation and kind != "candidates":
            limits[kind] = max(1, int(limits[kind] * self.cfg.search_fraction))
        if not (kind == "candidates" and self.validation):
            if self.counts[kind] + amount > limits[kind]:
                self._blocked = (self.validation, kind + "_budget_exhausted")
                raise FeasibilityBudgetExhausted(self._blocked[1])
        self.counts[kind] += amount
