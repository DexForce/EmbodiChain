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

"""Single-slot execution ports and lifecycle orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .contracts import CommitReceipt, ExpertEpisode, ValidationResult
from .coordinator import CandidateCoordinator, CandidateWorkItem

__all__ = [
    "InitialStatePort",
    "MeasuredExecutor",
    "EpisodeSink",
    "SingleSlotOutcome",
    "SingleSlotRunner",
]


class InitialStatePort(Protocol):
    """Restore and verify one candidate's complete initial state."""

    def restore(self, item: CandidateWorkItem) -> object:
        """Restore a candidate and return an opaque prepared binding."""


class MeasuredExecutor(Protocol):
    """Plan-check and execute one prepared candidate."""

    def validate_plan(self, item: CandidateWorkItem) -> ValidationResult:
        """Return host-owned planning/path validation evidence."""

    def execute(
        self,
        item: CandidateWorkItem,
        prepared: object,
        *,
        episode_id: str,
        commit_id: str,
    ) -> ExpertEpisode:
        """Execute one candidate and return measured T/T+1 evidence."""


class EpisodeSink(Protocol):
    """Persist one accepted episode and return its final receipt."""

    def submit(
        self,
        episode: ExpertEpisode,
        *,
        submission_id: int,
    ) -> CommitReceipt:
        """Persist and confirm one episode synchronously or through a bounded sink."""


@dataclass(frozen=True)
class SingleSlotOutcome:
    """Result of one coordinator-driven physical candidate attempt."""

    status: str
    candidate_id: str | None = None
    receipt: CommitReceipt | None = None
    reason: str | None = None


class SingleSlotRunner:
    """Drive one candidate through restore, measured execution, and persistence.

    This runner is deliberately host-neutral. #591 can supply a fixed-scene
    initial-state port and physical executor, while a later sink adapter can
    supply LeRobot persistence. The runner itself owns no simulator, Gym, or
    dataset implementation.
    """

    def __init__(
        self,
        coordinator: CandidateCoordinator,
        restorer: InitialStatePort,
        executor: MeasuredExecutor,
        sink: EpisodeSink,
        *,
        episode_byte_budget: int,
    ) -> None:
        if not isinstance(coordinator, CandidateCoordinator):
            raise TypeError("coordinator must be a CandidateCoordinator")
        for name, value in (
            ("restorer", restorer),
            ("executor", executor),
            ("sink", sink),
        ):
            if value is None:
                raise TypeError(f"{name} must be supplied")
        if type(episode_byte_budget) is not int or episode_byte_budget <= 0:
            raise ValueError("episode_byte_budget must be a positive integer")
        self._coordinator = coordinator
        self._restorer = restorer
        self._executor = executor
        self._sink = sink
        self._episode_byte_budget = episode_byte_budget

    def run_next(self) -> SingleSlotOutcome:
        """Execute the next FIFO candidate, or report an empty queue."""
        item = self._coordinator.take_next()
        if item is None:
            return SingleSlotOutcome("no_candidate")
        candidate_id = item.spec.identity.candidate_id
        session = self._coordinator.session
        try:
            validation = self._executor.validate_plan(item)
            self._coordinator.admit_planned(item, validation)
        except Exception as error:
            session.release(item.spec.identity, reason="planning_failed")
            return SingleSlotOutcome(
                "planning_rejected", candidate_id=candidate_id, reason=str(error)
            )

        ready = session.take_ready(
            item.spec.identity.scene_case_id,
            item.spec.identity.initial_state_id,
            episode_byte_budget=self._episode_byte_budget,
        )
        if ready is None:
            session.release(item.spec.identity, reason="execution_capacity_unavailable")
            return SingleSlotOutcome("capacity_unavailable", candidate_id=candidate_id)

        episode_id, commit_id = session.episode_ids(item.spec.identity)
        prepared = self._restorer.restore(item)
        session.mark_rollout_started(item.spec.identity)
        episode = self._executor.execute(
            item,
            prepared,
            episode_id=episode_id,
            commit_id=commit_id,
        )
        if not session.accept_episode(episode):
            return SingleSlotOutcome("measured_rejected", candidate_id=candidate_id)
        receipt = self._sink.submit(episode, submission_id=0)
        session.apply_receipt(receipt)
        return SingleSlotOutcome(
            "committed" if receipt.confirmed else "write_failed",
            candidate_id=candidate_id,
            receipt=receipt,
            reason=receipt.error or None,
        )

    @property
    def coordinator(self) -> CandidateCoordinator:
        """Return the coordinator owned by this runner."""
        return self._coordinator
