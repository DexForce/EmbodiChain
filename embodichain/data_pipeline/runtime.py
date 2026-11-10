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

"""Owner-side flow control for an explicitly started online data engine."""

from __future__ import annotations

import math
import os
import sys
import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from embodichain.utils import configclass

from .configuration import ODSConfiguration
from .datasets.sequence import DataBudgetExhaustedError, DataUnavailableError
from .engine.data import _add_exception_note

if TYPE_CHECKING:
    import torch
    from tensordict import TensorDict

    from .datasets import OnlineDataSource, SequenceSpecCfg
    from .engine import OnlineDataEngine

__all__ = ["ODSFlowControllerCfg", "ODSFlowController"]


@configclass
class ODSFlowControllerCfg:
    """Bound owner-side waits and optional whole-cache production.

    Args:
        data_wait_timeout: Maximum seconds blocked on missing windows or quota.
        generation_timeout: Maximum seconds waiting for a complete refill.
        poll_interval: Maximum sleep/probe duration between health/event checks.
        max_generated_episodes: Optional accepted-episode ceiling. A refill must
            fit a complete cache; unused partial-cache capacity is not produced.
    """

    data_wait_timeout: float = 30.0
    generation_timeout: float = 1800.0
    poll_interval: float = 0.02
    max_generated_episodes: int | None = None

    def __post_init__(self) -> None:
        for name in ("data_wait_timeout", "generation_timeout", "poll_interval"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
        limit = self.max_generated_episodes
        if limit is not None and (type(limit) is not int or limit < 1):
            raise ValueError("max_generated_episodes must be a positive integer")


class ODSFlowController:
    """Apply a measured plan without changing sampling or engine ownership.

    The creating process/thread owns this controller. The application starts
    and stops the engine and owns optimizer/evaluation deadlines. Sources and
    Datasets retain their immediate sampling/error contracts.

    Args:
        engine: Creator-owned engine configured for manual refill.
        cfg: Wait bounds and optional production limit.
        plan: Optional configuration to enforce before each batch. Cache,
            producer count, source allowances and batch size must match the plan.
        progress: Called during polling; may raise for cancellation/deadlines.
        on_events: Receives drained lifecycle records, including after stop.
        on_refill: Receives a new requested generation and its absolute host
            monotonic request time. Coalesced requests are reported once.
        on_interval: Receives wait kind and absolute host monotonic start/end.
    """

    def __init__(
        self,
        engine: OnlineDataEngine,
        cfg: ODSFlowControllerCfg | None = None,
        *,
        plan: ODSConfiguration | None = None,
        progress: Callable[[], object] | None = None,
        on_events: Callable[[list[dict[str, object]]], object] | None = None,
        on_refill: Callable[[int, float], object] | None = None,
        on_interval: Callable[[str, float, float], object] | None = None,
    ) -> None:
        engine._require_owner_process("ODSFlowController")
        self.cfg = cfg if cfg is not None else ODSFlowControllerCfg()
        self.cfg.validate()
        if engine.cfg.refill_policy != "manual":
            raise ValueError("ODSFlowController requires manual refill")
        limit = self.cfg.max_generated_episodes
        if limit is not None and limit < engine.buffer_size:
            raise ValueError(
                "Production limit cannot be smaller than the initial cache"
            )
        if plan is not None or limit is not None:
            producers = engine.cfg.gym_config.get("num_envs")
            if (
                type(producers) is not int
                or producers < 1
                or engine.buffer_size % producers
            ):
                raise ValueError(
                    "Planned or limited flow requires explicit num_envs dividing the cache"
                )
        if plan is not None:
            if not isinstance(plan, ODSConfiguration):
                raise TypeError("plan must be an ODSConfiguration")
            if (
                engine.buffer_size != plan.buffer_size
                or engine.cfg.episode_sample_budget != plan.episode_sample_budget
                or engine.cfg.episode_sample_hard_limit
                != plan.episode_sample_hard_limit
                or producers != plan.producer_envs
            ):
                raise ValueError(
                    "Engine cache, producer count and quotas must match the plan"
                )
            rate = min(
                plan.training_chunks_per_second_target,
                plan.active_chunks_per_second_cap,
            )
            if not math.isfinite(rate) or rate <= 0:
                raise ValueError("Plan training rate must be finite and positive")
        self.engine, self.plan = engine, plan
        self._progress, self._on_events = progress, on_events
        self._on_refill, self._on_interval = on_refill, on_interval
        self._owner = os.getpid(), threading.get_ident()
        self._last_refill = 0
        self._next_batch = 0.0

    def _require_owner(self) -> None:
        if self._owner != (os.getpid(), threading.get_ident()):
            raise RuntimeError("ODS flow control is owner-process/thread only")

    def drain_events(self) -> list[dict[str, object]]:
        """Drain accounting records, also during failure/shutdown diagnostics.

        Returns:
            Detached lifecycle records delivered to the optional event observer.
        """
        self._require_owner()
        events = self.engine.drain_lifecycle_events()
        if events and self._on_events is not None:
            self._on_events(events)
        return events

    def poll(self) -> None:
        """Check caller deadlines, drain lifecycle events and verify worker health."""
        self._require_owner()
        if self._progress is not None:
            self._progress()
        self.drain_events()
        self.engine.check_health()

    def _record(self, kind: str, started: float | None) -> None:
        ended = time.monotonic()
        if self._on_interval is not None:
            primary = sys.exc_info()[1]
            try:
                self._on_interval(kind, ended if started is None else started, ended)
            except BaseException as error:
                if primary is None:
                    raise
                _add_exception_note(
                    primary, f"ODS interval observer also failed: {error}"
                )

    def _sleep(self, deadline: float, message: str) -> None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(message)
        time.sleep(min(self.cfg.poll_interval, remaining))

    def _room(self, state: dict[str, int | None]) -> bool:
        limit = self.cfg.max_generated_episodes
        return (
            limit is None
            or state["successful_episodes"] + self.engine.buffer_size <= limit
        )

    def request_refill(self) -> int | None:
        """Request or coalesce a fill within the accepted-episode ceiling.

        Returns:
            Pending or newly requested generation, or None if no whole cache fits.
        """
        self.poll()
        state = self.engine.get_flow_status()
        pending = state["requested_generation"] > state["completed_generation"]
        if pending:
            return state["requested_generation"]
        if not self._room(state):
            return None
        started = time.monotonic()
        generation = self.engine.request_refill()
        if generation != self._last_refill:
            self._last_refill = generation
            if self._on_refill is not None:
                self._on_refill(generation, started)
        return generation

    def wait_for_generation(
        self, generation: int, *, deadline: float | None = None
    ) -> None:
        """Wait with a total bound, optionally shortened by an absolute deadline.

        Args:
            generation: Target generation returned by a refill request.
            deadline: Optional absolute host monotonic time limiting this wait.

        Raises:
            TimeoutError: If the generation does not complete within the bound.
        """
        self._require_owner()
        if deadline is not None and (
            isinstance(deadline, bool) or not math.isfinite(deadline)
        ):
            raise ValueError("deadline must be a finite host monotonic time")
        limit = time.monotonic() + self.cfg.generation_timeout
        if deadline is not None:
            limit = min(limit, deadline)
        started = time.monotonic()
        try:
            while True:
                self.poll()
                remaining = limit - time.monotonic()
                try:
                    self.engine.wait_for_generation(
                        generation,
                        timeout=max(1e-9, min(self.cfg.poll_interval, remaining)),
                    )
                except TimeoutError:
                    if time.monotonic() >= limit:
                        raise TimeoutError(
                            f"ODS generation {generation} missed its deadline"
                        ) from None
                    continue
                self.drain_events()
                return
        finally:
            self._record("data_wait", started)

    def prepare_batch(self, batch_size: int) -> bool:
        """Refill/pace a planned batch; False means final supply is exhausted.

        Unplanned engines are only polled; their refill schedule stays explicit.
        Quota readiness is a coarse bound. Sampling still validates the exact
        context/filter request and can require a bounded data wait.

        Args:
            batch_size: Positive chunk count, matching the plan when supplied.

        Returns:
            False only when capped final supply cannot fund a complete batch.

        Raises:
            TimeoutError: If missing unlocked quota exceeds the data wait bound.
        """
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        self._require_owner()
        if self.plan is None:
            self.poll()
            return True
        if batch_size != self.plan.batch_size:
            raise ValueError("batch_size must match the measured plan")
        blocked = None
        try:
            while True:
                self.poll()
                state = self.engine.get_flow_status()
                remaining = state["unlocked_remaining_chunks"]
                pending = state["requested_generation"] > state["completed_generation"]
                if not pending and remaining <= self.plan.refill_remaining_chunks:
                    pending = self.request_refill() is not None
                if remaining >= batch_size:
                    break
                if not pending and not self._room(state):
                    return False
                if blocked is None:
                    blocked = time.monotonic()
                self._sleep(
                    blocked + self.cfg.data_wait_timeout,
                    "No unlocked ODS quota before the data deadline",
                )
        finally:
            self._record("quota_wait", blocked)
        started = time.monotonic()
        try:
            while time.monotonic() < self._next_batch:
                self.poll()
                time.sleep(
                    max(
                        0.0,
                        min(
                            self.cfg.poll_interval, self._next_batch - time.monotonic()
                        ),
                    )
                )
            # Slow updates/evaluation do not accumulate catch-up credits.
            self._next_batch = time.monotonic() + batch_size / min(
                self.plan.training_chunks_per_second_target,
                self.plan.active_chunks_per_second_cap,
            )
        finally:
            self._record("usage_throttle", started)
        return True

    def sample_context(
        self,
        source: OnlineDataSource,
        batch_size: int,
        sequence: SequenceSpecCfg,
        *,
        generator: torch.Generator | None = None,
    ) -> TensorDict:
        """Retry only transient live-sampling errors within the data deadline.

        Call :meth:`prepare_batch` first to apply a plan's pacing/supply limit.
        This helper accepts only a live source wrapping the controlled engine.
        Sample resident sources through their own ``sample_context()`` method;
        they are not accepted by this controller.

        Args:
            source: Live source wrapping this controller's engine.
            batch_size: Number of chunks to sample.
            sequence: Requested context and prediction window.
            generator: Optional sampling random generator.

        Returns:
            The source's batch, including masks and provenance.

        Raises:
            TimeoutError: If transient unavailability exceeds the data wait bound.
        """
        self._require_owner()
        if source.engine is not self.engine:
            raise ValueError("source must use the controlled live engine")
        blocked = None
        try:
            while True:
                self.poll()
                try:
                    return source.sample_context(
                        batch_size, sequence, generator=generator
                    )
                except (DataUnavailableError, DataBudgetExhaustedError):
                    if blocked is None:
                        blocked = time.monotonic()
                    self._sleep(
                        blocked + self.cfg.data_wait_timeout,
                        "No eligible ODS context before the data deadline",
                    )
        finally:
            self._record("data_wait", blocked)
