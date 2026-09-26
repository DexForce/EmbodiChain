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

"""Adapter from #670 GenerationSession lifecycle to the G-03 runner.

The host supplies proposal, physical rollout and persistence operations. This
module owns only the translation of measured episode/receipt outcomes into the
benchmark's source-neutral ``GenerationAttempt`` record.
"""

from __future__ import annotations

from typing import Protocol

from embodichain.lab.sim.motion.expansion.contracts import (
    CandidateIdentity,
    CommitReceipt,
    ExpertEpisode,
    ValidationResult,
)
from embodichain.lab.sim.motion.expansion.session import GenerationSession

from .contracts import GenerationAttempt, GenerationCase, PersistenceReceipt

__all__ = ["GenerationSessionHost", "GenerationSessionExecutor"]


class GenerationSessionHost(Protocol):
    """Host operations supplied by the production generation coordinator."""

    def propose(self, case: GenerationCase, attempt_id: int) -> CandidateIdentity:
        """Allocate a candidate identity and enqueue its planned trajectory."""

    def rollout(
        self, case: GenerationCase, identity: CandidateIdentity
    ) -> ExpertEpisode:
        """Execute and validate one physical episode, returning measured evidence."""

    def persist(self, case: GenerationCase, episode: ExpertEpisode) -> CommitReceipt:
        """Submit one episode to the production EpisodeSink and return its receipt."""


def _validation_status(result: ValidationResult) -> str:
    """Map the fail-closed session validation result to benchmark state."""
    return "passed" if result.accepted else "failed"


def _task_status(result: ValidationResult) -> str:
    """Extract the task/effect predicate without treating path checks as task success."""
    checks = {check.check_id: check.status for check in result.checks}
    task = checks.get("task_success")
    if task == "passed":
        return "passed"
    if task in {"failed", "unavailable", "not_run"}:
        return "failed"
    return "not_evaluated"


class GenerationSessionExecutor:
    """Translate a production host's session lifecycle into one benchmark attempt."""

    def __init__(self, session: GenerationSession, host: GenerationSessionHost) -> None:
        self.session = session
        self.host = host

    def __call__(self, case: GenerationCase, attempt_id: int) -> GenerationAttempt:
        """Run one proposed candidate through measured evidence and receipt confirmation."""
        identity = self.host.propose(case, attempt_id)
        self.session.mark_rollout_started(identity)
        episode = self.host.rollout(case, identity)
        if episode.identity != identity:
            raise ValueError("host rollout returned a different candidate identity")
        validation_status = _validation_status(episode.validation)
        task_status = _task_status(episode.validation)
        if not self.session.accept_episode(episode):
            return GenerationAttempt(
                case=case,
                attempt_id=attempt_id,
                execution_status="completed",
                validation_status=validation_status,
                task_status=task_status,
                elapsed_s=float(episode.timestamps[-1].item()),
                failure_reason="measured_validation_failed",
                evidence=("generation_session.validation",),
            )
        receipt = self.host.persist(case, episode)
        self.session.apply_receipt(receipt)
        return GenerationAttempt(
            case=case,
            attempt_id=attempt_id,
            execution_status="completed",
            validation_status=validation_status,
            task_status=task_status,
            receipt=PersistenceReceipt(
                episode_id=receipt.episode_id,
                commit_id=receipt.commit_id,
                submission_id=receipt.submission_id,
                storage_id=receipt.storage_id,
                confirmed=receipt.confirmed,
                error=receipt.error,
            ),
            elapsed_s=float(episode.timestamps[-1].item()),
            failure_reason=None if receipt.confirmed else receipt.error,
            evidence=("generation_session.validation", "episode_sink.receipt"),
        )
