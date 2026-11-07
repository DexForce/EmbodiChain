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

"""Compatible-cohort orchestration using the canonical expansion session.

Hosts execute a whole cohort with one world loop. They own physical preparation,
measured evidence and row masking; the engine never advances a simulation.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import time
from typing import Protocol
from .configured import ExpansionRunCfg

from embodichain.utils import configclass
from .contracts import EpisodeCandidate
from .journal import RecipeAttempt
from .contracts import (
    CandidateIdentity,
    CommitReceipt,
    ExpertEpisode,
    ValidationResult,
    CandidateSpec,
)
from embodichain.lab.sim.motion.expansion.coordinator import CandidateWorkItem
from .session import ExpansionSession

__all__ = [
    "ExpansionJobCfg",
    "ExpansionAssignment",
    "PreparedEpisodeWorkItem",
    "ExpansionWorkItem",
    "ExpansionRecipeSource",
    "PreparingExpansionHost",
    "ExpansionHost",
    "EpisodeSink",
    "ExpansionEngine",
]


@configclass
class ExpansionJobCfg:
    """Bound one engine's compatible-cohort execution.

    Args:
        batch_size: Maximum physical rows in a cohort.
        episode_byte_budget: Reserved numeric evidence bytes per physical row.
        candidate_budget: Maximum number of queued work items.
    """

    batch_size: int = 8
    episode_byte_budget: int = 8 * 1024 * 1024
    candidate_budget: int = 1024

    def __post_init__(self) -> None:
        for name in ("batch_size", "episode_byte_budget", "candidate_budget"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class PreparedEpisodeWorkItem:
    """Prepared lazy recipe sharing the trajectory scheduler's admission protocol.

    Args:
        candidate: Immutable recipe and session-allocated identity.
        compatibility_key: Host-validated World/resource compatibility identity.
        env_id: Physical row prepared by the host; rejected peers never renumber it.
    """

    candidate: EpisodeCandidate
    compatibility_key: str
    env_id: int

    @property
    def spec(self) -> CandidateSpec:
        """Expose the common scheduling identity without inventing a trajectory."""
        return CandidateSpec(
            self.candidate.identity, compatibility_key=self.compatibility_key
        )


ExpansionWorkItem = CandidateWorkItem | PreparedEpisodeWorkItem


@dataclass(frozen=True)
class ExpansionAssignment:
    """One logical candidate assigned to a physical cohort row.

    Args:
        item: Source-owned candidate and template.
        env_id: Physical row index, never part of candidate identity.
        episode_id: Session-assigned episode identity.
        commit_id: Idempotent persistence identity.
    """

    item: ExpansionWorkItem
    env_id: int
    episode_id: str
    commit_id: str


class ExpansionHost(Protocol):
    """Host port for one actual batched world, including incomplete final cohorts.

    Hosts must mask unused rows and return one episode or explicit rejection
    per assignment. Call ``on_started`` only after that row's first command.
    Preparation failures and native exceptions propagate and abort the cohort.
    """

    capacity: int

    def preflight(self, items: tuple[ExpansionWorkItem, ...]) -> None:
        """Check supported sources, backend, scope and budgets without mutation."""
        ...

    def validate_plan(self, item: ExpansionWorkItem) -> ValidationResult:
        """Return independent required planning checks for one candidate."""
        ...

    def execute_batch(
        self,
        assignments: tuple[ExpansionAssignment, ...],
        *,
        on_started: Callable[[CandidateIdentity], None],
    ) -> Sequence[ExpertEpisode | None]:
        """Prepare and execute all assignments in one world loop."""
        ...


class ExpansionRecipeSource(Protocol):
    """Bounded durable recipe-attempt producer, independent of motion source kind."""

    def next_batch(self, capacity: int) -> tuple[RecipeAttempt, ...]:
        """Allocate at most capacity requests; an empty batch ends sampling."""
        ...

    def after_batch(self, snapshot: Mapping[str, object]) -> None:
        """Persist collection progress after the common scheduler drains a cohort."""
        ...


class PreparingExpansionHost(ExpansionHost, Protocol):
    """Host that prepares immutable recipe attempts before engine admission."""

    def prepare_batch(
        self, requests: tuple[RecipeAttempt, ...], session: ExpansionSession
    ) -> tuple[PreparedEpisodeWorkItem, ...]:
        """Return session-proposed items bound to validated physical rows."""
        ...


class EpisodeSink(Protocol):
    """Single-owner writer; ambiguous writes must be reconciled before retry."""

    def submit(self, episode: ExpertEpisode, *, submission_id: int) -> CommitReceipt:
        """Return a final identity-matching receipt for the immutable payload."""
        ...


class ExpansionEngine:
    """Schedule compatible candidates without duplicating session accounting.

    Args:
        cfg: Cohort and payload bounds, copied on construction.
        session: Canonical owner of registered candidates and all counters.
        host: Explicit batched execution adapter.
        sink: Writer returning final commit receipts.

    Candidates must already have been proposed in ``session``. This adapter
    handles planning admission, assignment, measured admission and persistence.
    Use :meth:`from_job` to select the native Gym host and durable file workflow.
    """

    def __init__(
        self,
        cfg: ExpansionJobCfg | ExpansionRunCfg,
        *,
        session: ExpansionSession | None = None,
        host: ExpansionHost | None = None,
        sink: EpisodeSink | None = None,
    ) -> None:
        from .configured import ExpansionRunCfg

        self._configured_cfg = None
        if isinstance(cfg, ExpansionRunCfg):
            if any(port is not None for port in (session, host, sink)):
                raise ValueError("configured jobs assemble their execution ports")
            self._configured_cfg = ExpansionRunCfg(**cfg.to_dict())
            self.cfg = self._configured_cfg
        else:
            cfg.__post_init__()
            if session is None or host is None or sink is None:
                raise ValueError("custom engines require session, host and sink")
            self.cfg = ExpansionJobCfg(**cfg.to_dict())
            if type(host.capacity) is not int or host.capacity < cfg.batch_size:
                raise ValueError("host capacity cannot satisfy requested batch_size")
            if cfg.episode_byte_budget > session.pending_max_bytes:
                raise ValueError("episode_byte_budget exceeds session pending capacity")
        self.session, self.host, self.sink = session, host, sink
        self._queue: list[ExpansionWorkItem] = []
        self._seen: set[str] = set()
        self._pending: dict[str, ExpertEpisode] = {}
        self._admitted: set[str] = set()
        self._submitted: set[str] = set()
        self._staged: set[str] = set()
        self._submission_ids: dict[str, int] = {}
        self._write_failures: set[str] = set()
        self._resolved_receipts: dict[tuple[str, int], CommitReceipt] = {}
        self._failed = False
        self.timings: dict[str, float] = {}

    def _measure(self, name, operation, *args, **kwargs):
        started = time.perf_counter()
        try:
            return operation(*args, **kwargs)
        finally:
            self.timings[name] = (
                self.timings.get(name, 0.0) + time.perf_counter() - started
            )

    def _snapshot(self):
        return {**self.session.snapshot(), "timings": dict(self.timings)}

    @classmethod
    def from_job(cls, cfg: "ExpansionRunCfg") -> "ExpansionEngine":
        """Create the native offline host from a resolved file job.

        Args:
            cfg: Deployment, source, expansion, media and persistence settings.
        Returns:
            Engine supporting preflight, run and episode-boundary resume.
        """
        from .configured import ExpansionRunCfg

        if not isinstance(cfg, ExpansionRunCfg):
            raise TypeError("cfg must be ExpansionRunCfg")
        return cls(cfg)

    def resume(self) -> Mapping[str, object]:
        """Reconcile durable receipts and continue a configured offline job."""
        if self._configured_cfg is None:
            raise ValueError("durable resume requires ExpansionEngine.from_job")
        from .workflow import run_expansion_job

        return run_expansion_job(self._configured_cfg, resume=True)

    def enqueue(self, items: Sequence[ExpansionWorkItem]) -> None:
        """Append a bounded, unique set of candidates after read-only preflight.

        Args:
            items: Candidates proposed through this engine's session.
        """
        if self._configured_cfg is not None:
            raise ValueError("configured jobs own their candidate source")
        owned = tuple(items)
        if any(
            not isinstance(item, (CandidateWorkItem, PreparedEpisodeWorkItem))
            for item in owned
        ):
            raise TypeError(
                "items must contain trajectory or prepared episode work items"
            )
        ids = [item.spec.identity.candidate_id for item in owned]
        if len(ids) != len(set(ids)) or self._seen.intersection(ids):
            raise ValueError("candidate may only be enqueued once")
        if len(self._queue) + len(owned) > self.cfg.candidate_budget:
            raise BufferError("candidate queue budget exceeded")
        proposed = {
            identity: state for identity, state, _ in self.session.snapshot()["audit"]
        }
        if any(proposed.get(item.spec.identity) != "proposed" for item in owned):
            raise ValueError("candidates must be proposed in this session")
        self.host.preflight(owned)
        self._queue.extend(owned)
        self._seen.update(ids)

    def preflight(self) -> tuple[tuple[str, ...], ...] | Mapping[str, object]:
        """Return stable compatible cohorts without reserving or executing rows.

        Returns:
            Candidate IDs grouped by compatibility key in first-seen order.
        """
        if self._configured_cfg is not None:
            from .configured import preflight_expansion_job

            return preflight_expansion_job(self._configured_cfg)
        self.host.preflight(tuple(self._queue))
        groups: dict[str, list[str]] = OrderedDict()
        for item in self._queue:
            groups.setdefault(item.spec.compatibility_key, []).append(
                item.spec.identity.candidate_id
            )
        return tuple(
            tuple(ids[start : start + self.cfg.batch_size])
            for ids in groups.values()
            for start in range(0, len(ids), self.cfg.batch_size)
        )

    def run(self) -> Mapping[str, object]:
        """Drain compatible cohorts until exhausted, stopped or backpressured.

        Returns:
            Canonical session snapshot. Pending writes require reconciliation.
        """
        if self._configured_cfg is not None:
            from .workflow import run_expansion_job

            return run_expansion_job(self._configured_cfg)
        return self._drain()

    def _drain(self, *, one_batch: bool = False) -> Mapping[str, object]:
        if self._failed:
            raise RuntimeError(
                "host failed; construct a new host before further execution"
            )
        self.flush()
        self.preflight()
        while self._queue and not self.session.stop_reason:
            key = self._queue[0].spec.compatibility_key
            selected = [
                item for item in self._queue if item.spec.compatibility_key == key
            ][: self.cfg.batch_size]
            assignments: list[ExpansionAssignment] = []
            try:
                for item in selected:
                    identity = item.spec.identity
                    if identity.candidate_id not in self._admitted:
                        validation = self._measure(
                            "admission", self.host.validate_plan, item
                        )
                        if not validation.accepted:
                            self.session.release(
                                identity, reason="planning_validation_failed"
                            )
                            self._queue.remove(item)
                            continue
                        if isinstance(item, PreparedEpisodeWorkItem):
                            self.session.add_prepared_episode(
                                item.candidate, validation
                            )
                        else:
                            self.session.add_planned(item.to_batch(), validation)
                        self._admitted.add(identity.candidate_id)
                    ready = self.session.take_ready(
                        identity.scene_case_id,
                        identity.initial_state_id,
                        episode_byte_budget=self.cfg.episode_byte_budget,
                        candidate_id=identity.candidate_id,
                    )
                    if ready is None:
                        break
                    episode_id, commit_id = self.session.episode_ids(identity)
                    assignments.append(
                        ExpansionAssignment(
                            item,
                            (
                                item.env_id
                                if isinstance(item, PreparedEpisodeWorkItem)
                                else len(assignments)
                            ),
                            episode_id,
                            commit_id,
                        )
                    )
            except Exception:
                self._failed = True
                for assignment in assignments:
                    self.session.release(
                        assignment.item.spec.identity, reason="cohort_admission_failed"
                    )
                raise
            if not assignments:
                if not any(
                    item.spec.identity.candidate_id in self._admitted
                    for item in selected
                ):
                    continue
                break
            if len({a.env_id for a in assignments}) != len(assignments) or any(
                not 0 <= a.env_id < self.host.capacity for a in assignments
            ):
                for a in assignments:
                    self.session.release(a.item.spec.identity, reason="invalid_slots")
                self._failed = True
                raise ValueError("host preparation produced invalid physical slots")
            started: set[CandidateIdentity] = set()
            expected = {a.item.spec.identity for a in assignments}

            def on_started(identity: CandidateIdentity) -> None:
                if identity not in expected or identity in started:
                    raise ValueError("unexpected or repeated rollout start")
                self.session.mark_rollout_started(identity)
                started.add(identity)

            try:
                episodes = self._measure(
                    "execute",
                    lambda: tuple(
                        self.host.execute_batch(
                            tuple(assignments), on_started=on_started
                        )
                    ),
                )
                if len(episodes) != len(assignments):
                    raise ValueError(
                        "host must return exactly one result per assignment"
                    )
                # Validate the entire host response before committing any row.
                for assignment, episode in zip(assignments, episodes):
                    identity = assignment.item.spec.identity
                    if episode is not None and (
                        not isinstance(episode, ExpertEpisode)
                        or episode.identity != identity
                        or identity not in started
                        or (episode.episode_id, episode.commit_id)
                        != (assignment.episode_id, assignment.commit_id)
                    ):
                        raise ValueError(
                            "host returned mismatched or unstarted evidence"
                        )
            except Exception:
                self._failed = True
                for assignment in assignments:
                    self.session.release(
                        assignment.item.spec.identity, reason="host_failed"
                    )
                raise
            try:
                for assignment, episode in zip(assignments, episodes):
                    self._queue.remove(assignment.item)
                    if episode is None:
                        self.session.release(
                            assignment.item.spec.identity, reason="host_rejected"
                        )
                    else:
                        accepted = self.session.accept_episode(episode)
                        if accepted:
                            self._pending[episode.commit_id] = episode
                        callback = getattr(self.host, "on_acceptance", None)
                        if callback is not None:
                            callback(assignment, episode, accepted)
            except Exception:
                self._failed = True
                states = {
                    identity: state
                    for identity, state, _ in self.session.snapshot()["audit"]
                }
                for assignment in assignments:
                    identity = assignment.item.spec.identity
                    if states[identity] in {"assigned", "running"}:
                        self.session.release(
                            identity, reason="invalid_measured_evidence"
                        )
                raise
            # Stage every accepted row before any write can fail.
            self.flush()
            if one_batch:
                break
        return self._snapshot()

    def run_source(self, source: ExpansionRecipeSource) -> Mapping[str, object]:
        """Drive bounded source batches through the same admission/execution/writer core.

        The source owns durable logical recipe allocation. The host prepares a
        whole cohort and returns session-proposed work items. No native adapter
        implements a second dispatch or commit loop.
        """
        if self._configured_cfg is not None:
            raise ValueError("configured jobs assemble their source internally")
        if not callable(getattr(self.host, "prepare_batch", None)):
            raise TypeError("recipe execution requires a PreparingExpansionHost")
        while not self.session.stop_reason:
            requests = source.next_batch(self.cfg.batch_size)
            if not requests:
                break
            try:
                items = self._measure(
                    "prepare",
                    lambda: tuple(self.host.prepare_batch(requests, self.session)),
                )
                self.enqueue(items)
                self._drain(one_batch=True)
                for item in tuple(self._queue):
                    self.session.release(
                        item.spec.identity, reason="cohort_not_started"
                    )
                    callback = getattr(self.host, "on_deferred", None)
                    if callback is not None:
                        callback(item)
                    self._queue.remove(item)
                source.after_batch(self._snapshot())
            except BaseException:
                self._failed = True
                raise
        return self._snapshot()

    def flush(self) -> None:
        """Submit staged rows, stopping until ambiguous writes are reconciled.

        A row staged behind a failed submission remains available and is written
        after the writer resolves the original receipt. No rollout is repeated.
        """
        if self._configured_cfg is not None:
            raise ValueError("configured jobs reconcile durable writes through resume")
        if self._submitted:
            raise RuntimeError("reconcile pending receipts before submitting more data")
        if self._write_failures:
            raise RuntimeError("retry failed writes before submitting more data")
        for commit_id, episode in tuple(self._pending.items()):
            stage = getattr(self.sink, "stage", None)
            if commit_id not in self._staged and stage is not None:
                self._measure("stage", stage, episode)
                self._staged.add(commit_id)
        for commit_id, episode in tuple(self._pending.items()):
            self._submitted.add(commit_id)
            receipt = self._measure(
                "submit",
                self.sink.submit,
                episode,
                submission_id=self._submission_ids.get(commit_id, 0),
            )
            self.resolve_receipt(receipt)

    def resolve_receipt(self, receipt: CommitReceipt) -> None:
        """Apply a reconciled original-submission receipt without a new rollout.

        Args:
            receipt: Writer-confirmed result, including exact candidate identity.
        """
        if self._configured_cfg is not None:
            raise ValueError("configured jobs reconcile durable writes through resume")
        key = (receipt.commit_id, receipt.submission_id)
        if key in self._resolved_receipts:
            if self._resolved_receipts[key] != receipt:
                raise ValueError("contradictory final receipt")
            return
        if receipt.commit_id not in self._submitted:
            raise ValueError("receipt does not identify a submitted episode")
        self.session.apply_receipt(receipt)
        self._resolved_receipts[key] = receipt
        if receipt.confirmed:
            del self._pending[receipt.commit_id]
        else:
            self._write_failures.add(receipt.commit_id)
        self._submitted.remove(receipt.commit_id)

    def retry_commit(self, commit_id: str) -> None:
        """Retry one explicitly failed frozen payload independently of peer failures."""
        if self._submitted:
            raise RuntimeError("reconcile pending receipts before retrying data")
        if commit_id not in self._write_failures:
            raise ValueError("commit has no resolved failure to retry")
        episode, submission_id = self.session.retry_write(commit_id)
        self._pending[commit_id] = episode
        self._submission_ids[commit_id] = submission_id
        self._write_failures.remove(commit_id)
        self._submitted.add(commit_id)
        receipt = self._measure(
            "submit", self.sink.submit, episode, submission_id=submission_id
        )
        self.resolve_receipt(receipt)

    @property
    def pending_commit_ids(self) -> tuple[str, ...]:
        """Return submissions requiring writer-side completion or reconciliation."""
        if self._configured_cfg is not None:
            raise ValueError(
                "configured jobs expose durable state through inspect_expansion_job"
            )
        return tuple(self._pending)
