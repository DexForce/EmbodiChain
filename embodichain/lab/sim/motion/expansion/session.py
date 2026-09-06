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

"""Host-independent candidate bookkeeping and confirmed collection quotas."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields, replace
import hashlib
import json
import time
from types import MappingProxyType

import torch

from .cfg import TrajectoryGenerationJobCfg
from .contracts import (
    CandidateIdentity,
    CandidateTrajectoryBatch,
    CommitReceipt,
    ExpertEpisode,
    SceneCase,
    ValidationCheck,
    ValidationResult,
)
from .coverage import CoverageIndex, describe_trajectory

__all__ = ["GenerationSession"]


@dataclass
class _Attempt:
    identity: CandidateIdentity
    state: str = "proposed"
    trajectory: CandidateTrajectoryBatch | None = None
    joint_names: tuple[str, ...] = ()
    episode: ExpertEpisode | None = None
    episode_byte_budget: int = 0
    submission_id: int = 0
    reason: str | None = None
    planning_validation: ValidationResult | None = None
    rollout_validation: ValidationResult | None = None


def _digest(*values: object) -> str:
    return hashlib.sha256(
        json.dumps(values, separators=(",", ":")).encode()
    ).hexdigest()


def _tensor_bytes(*values: torch.Tensor | None) -> int:
    return sum(
        value.numel() * value.element_size() for value in values if value is not None
    )


def _json_bytes(value: object) -> int:
    def encode(item: object) -> dict:
        if isinstance(item, Mapping):
            return dict(item)
        return {field.name: getattr(item, field.name) for field in fields(item)}

    return len(json.dumps(value, default=encode, ensure_ascii=False).encode())


def _candidate_bytes(batch: CandidateTrajectoryBatch) -> int:
    return _tensor_bytes(
        batch.positions, batch.dt, batch.valid_length, batch.source_row_indices
    ) + sum(
        _json_bytes((identity, batch.joint_names, phases, factors))
        for identity, phases, factors in zip(
            batch.identities, batch.phases, batch.factors
        )
    )


def _validation_summary(result: ValidationResult) -> ValidationResult:
    """Keep four-state outcomes with bounded text and numeric audit context."""
    if len(result.checks) > 64:
        raise ValueError("Validation supports at most 64 checks per result.")
    return ValidationResult(
        tuple(
            ValidationCheck(
                check.check_id,
                check.status,
                check.detail[:256],
                dict(tuple(check.metrics.items())[:16]),
            )
            for check in result.checks
        )
    )


class GenerationSession:
    """Own one job's bounded candidates, budgets, evidence, and commit history.

    This value-only session never reads an environment or steps a simulator.
    Call ``mark_rollout_started`` only after the first actual command. A sink
    must provide one final outcome per submission and persist ``commit_id``
    idempotently across retries. Contradictory final receipts are rejected.
    The caller serializes operations. Cancellation means stopping new work and
    releasing individual candidates; pending writes still require final receipts.
    Byte limits include tensor data and serialized value metadata, excluding
    Python object allocation overhead. The runner separately bounds host buffers
    and any copies it allocates outside this session.

    Args:
        cfg: Semantically valid job settings, copied on construction.
        clock: Monotonic seconds, injectable for deterministic budget tests.
    """

    def __init__(
        self,
        cfg: TrajectoryGenerationJobCfg,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._cfg = TrajectoryGenerationJobCfg.from_mapping(cfg.to_dict())
        self._clock, self._started_at = clock, clock()
        self._cases: dict[
            tuple[str, str], tuple[SceneCase, torch.Tensor, tuple[str, ...]]
        ] = {}
        self._coverage: dict[str, CoverageIndex] = {}
        self._ordinals: dict[tuple[str, ...], int] = {}
        self._attempts: dict[str, _Attempt] = {}
        self._commits: dict[str, str] = {}
        self._receipts: dict[tuple[str, int], CommitReceipt] = {}
        self._counts = dict.fromkeys(
            (
                "proposed",
                "planned_valid",
                "rollout_attempted",
                "validated_accepted",
                "committed",
            ),
            0,
        )

    def register_case(
        self,
        case: SceneCase,
        joint_limits: torch.Tensor,
        *,
        joint_names: tuple[str, ...],
    ) -> None:
        """Register immutable case conditions and complete joint normalization limits.

        Args:
            case: Scene and initial-state identity with fixed robot conditions.
            joint_limits: Finite increasing joint intervals, shape ``(D_full, 2)``.
            joint_names: Complete unique joint order matching the limit rows.
        """
        names = tuple(joint_names)
        if (
            isinstance(joint_names, str)
            or not names
            or any(not isinstance(name, str) or not name.strip() for name in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError("joint_names must contain unique nonempty names.")
        limits = joint_limits.detach().to(device="cpu", dtype=torch.float64).clone()
        if limits.shape != (len(names), 2):
            raise ValueError("joint_limits must have shape (len(joint_names), 2).")
        if not torch.isfinite(limits).all() or not (limits[:, 1] > limits[:, 0]).all():
            raise ValueError(
                "joint_limits must be finite, strictly increasing intervals."
            )
        key = (case.scene_case_id, case.initial_state_id)
        if key in self._cases:
            previous, previous_limits, previous_names = self._cases[key]
            if (
                previous != case
                or not torch.equal(previous_limits, limits)
                or previous_names != names
            ):
                raise ValueError("A registered case's fixed conditions cannot change.")
            return
        for (case_id, _), (
            previous,
            previous_limits,
            previous_names,
        ) in self._cases.items():
            if case_id == case.scene_case_id and (
                replace(previous, initial_state_id=case.initial_state_id) != case
                or not torch.equal(previous_limits, limits)
                or previous_names != names
            ):
                raise ValueError(
                    "Initial states in one case must share scene and robot conditions."
                )
        self._cases[key] = case, limits, names
        if case.scene_case_id not in self._coverage:
            coverage = self._cfg.augmentation.coverage
            self._coverage[case.scene_case_id] = CoverageIndex(
                geometry_tolerance=coverage.joint_dedup_normalized_tol,
                target_per_geometry=coverage.target_per_cell,
            )

    def propose(
        self,
        case_id: str,
        initial_state_id: str,
        *,
        source_id: str,
        source_revision: str,
        template_id: str,
        operator_id: str,
        geometry_family_id: str | None = None,
        parent_id: str | None = None,
    ) -> tuple[CandidateIdentity, torch.Generator]:
        """Allocate stable lineage and a local CPU RNG independent of execution slots.

        Args:
            case_id: Registered scene-case identifier.
            initial_state_id: Registered initial state within that case.
            source_id: Identifier of the reference trajectory source.
            source_revision: Revision of the source used for this proposal.
            template_id: Identifier of the reference template.
            operator_id: Operator name selecting the deterministic proposal stream.
            geometry_family_id: Shared geometry lineage, or ``None`` for a new family.
            parent_id: Existing candidate in the same case and initial state, if any.

        Returns:
            The new candidate identity and its seeded local CPU generator.
        """
        self._cases[(case_id, initial_state_id)]
        if (
            self.stop_reason
            or self._counts["proposed"] >= self._cfg.collection.max_proposals
        ):
            raise RuntimeError(
                "Proposal budget is exhausted or the session has stopped."
            )
        if not isinstance(operator_id, str) or not operator_id.strip():
            raise ValueError("operator_id must be a nonempty string.")
        stream = (
            case_id,
            initial_state_id,
            source_id,
            source_revision,
            template_id,
            operator_id,
        )
        ordinal = self._ordinals.get(stream, 0)
        digest = _digest(self._cfg.augmentation.seed, *stream, ordinal)
        identity = CandidateIdentity(
            case_id,
            initial_state_id,
            "candidate_" + digest,
            geometry_family_id or "geometry_" + digest,
            source_id,
            source_revision,
            template_id,
            parent_id=parent_id,
        )
        if parent_id is not None:
            parent = self._attempts[parent_id].identity
            if (parent.scene_case_id, parent.initial_state_id) != (
                case_id,
                initial_state_id,
            ):
                raise ValueError(
                    "A candidate parent must belong to the same case and initial state."
                )
        self._ordinals[stream] = ordinal + 1
        self._attempts[identity.candidate_id] = _Attempt(identity)
        self._counts["proposed"] += 1
        return identity, torch.Generator(device="cpu").manual_seed(
            int(digest[:16], 16) % 2**63
        )

    def _attempt(self, identity: CandidateIdentity, *states: str) -> _Attempt:
        attempt = self._attempts[identity.candidate_id]
        if attempt.identity != identity or (states and attempt.state not in states):
            raise ValueError(
                "Candidate identity or lifecycle state does not match this operation."
            )
        return attempt

    def _validation_passes(
        self, validation: ValidationResult, *, rollout: bool
    ) -> bool:
        required = (
            {"path_collision"} if self._cfg.validation.require_path_collision else set()
        )
        if rollout and self._cfg.validation.require_task_success:
            required.add("task_success")
        return validation.accepted and required <= {
            check.check_id for check in validation.checks
        }

    def add_planned(
        self, batch: CandidateTrajectoryBatch, validation: ValidationResult
    ) -> None:
        """Atomically admit validated candidate rows within ready count and byte limits.

        The caller owns physical planning checks. Nonempty, passing checks must
        include ``path_collision`` when required by this job.

        Args:
            batch: Planned rows matching candidates previously proposed by this session.
            validation: One planning result per row, including required collision checks.
        """
        attempts = [
            self._attempt(identity, "proposed") for identity in batch.identities
        ]
        for attempt in attempts:
            attempt.planning_validation = _validation_summary(validation)
        if not self._validation_passes(validation, rollout=False):
            for attempt in attempts:
                attempt.reason = "planning_validation_failed"
            raise ValueError(
                "Planned candidates require passing mandatory planning checks."
            )
        for identity in batch.identities:
            names = self._cases[(identity.scene_case_id, identity.initial_state_id)][2]
            if batch.joint_names != names:
                raise ValueError(
                    "Candidate joint layout does not match the registered limits."
                )
        ready = [
            attempt.trajectory
            for attempt in self._attempts.values()
            if attempt.state == "ready"
        ]
        if (
            len(ready) + len(batch.identities)
            > self._cfg.execution.ready_high_watermark
            or sum(map(_candidate_bytes, ready)) + _candidate_bytes(batch)
            > self._cfg.execution.ready_max_bytes
        ):
            raise BufferError("The ready candidate pool is full.")
        rows = [batch.row(index) for index in range(len(batch.identities))]
        for row in rows:
            attempt = self._attempt(row.identities[0])
            attempt.reason = None
            attempt.state, attempt.trajectory, attempt.joint_names = (
                "ready",
                row,
                row.joint_names,
            )
        self._counts["planned_valid"] += len(rows)

    def _reserved(self) -> int:
        return sum(
            attempt.state in {"assigned", "running", "pending_write", "committed"}
            for attempt in self._attempts.values()
        )

    def take_ready(
        self,
        case_id: str,
        initial_state_id: str,
        *,
        episode_byte_budget: int | None = None,
    ) -> CandidateTrajectoryBatch | None:
        """Reserve a row, rollout attempt, and pending payload capacity before execution.

        ``episode_byte_budget`` is a trusted upper bound on serialized metadata
        plus tensor data in the eventual episode. ``None`` reserves the entire
        pending byte limit, conservatively allowing only one outstanding episode.
        Smaller explicit bounds permit concurrent rows; exceeding a bound rejects
        the episode. The runner owns separate limits on host-side allocations.

        Args:
            case_id: Registered scene-case identifier to execute next.
            initial_state_id: Registered initial state whose ready rows may be selected.
            episode_byte_budget: Maximum episode payload bytes to reserve for this row.

        Returns:
            One assigned candidate row, or ``None`` when no matching row or capacity
            is available or the session has stopped.
        """
        self._cases[(case_id, initial_state_id)]
        byte_limit = self._cfg.persistence.pending_max_bytes
        budget = byte_limit if episode_byte_budget is None else episode_byte_budget
        if type(budget) is not int or not 0 < budget <= byte_limit:
            raise ValueError(
                "episode_byte_budget must be a positive integer within pending_max_bytes."
            )
        pending_count, pending_bytes = self._pending_reservation()
        assigned = sum(
            attempt.state == "assigned" for attempt in self._attempts.values()
        )
        if (
            self.stop_reason
            or self._reserved() >= self._cfg.collection.target_committed_episodes
            or self._counts["rollout_attempted"] + assigned
            >= self._cfg.collection.max_rollout_attempts
            or pending_count >= self._cfg.persistence.pending_episode_limit
            or pending_bytes + budget > byte_limit
        ):
            return None
        for attempt in self._attempts.values():
            identity = attempt.identity
            if attempt.state == "ready" and (
                identity.scene_case_id,
                identity.initial_state_id,
            ) == (case_id, initial_state_id):
                attempt.state = "assigned"
                attempt.episode_byte_budget = budget
                trajectory, attempt.trajectory = attempt.trajectory, None
                return trajectory
        return None

    def mark_rollout_started(self, identity: CandidateIdentity) -> None:
        """Count an assigned candidate once its first command actually ran.

        Args:
            identity: Assigned candidate whose first command has been executed.
        """
        attempt = self._attempt(identity, "assigned")
        attempt.state, attempt.trajectory = "running", None
        self._counts["rollout_attempted"] += 1

    def release(
        self, identity: CandidateIdentity, *, reason: str = "discarded"
    ) -> None:
        """Discard uncommitted work while preserving its failure reason and checks.

        Args:
            identity: Candidate to release before a write is pending or confirmed.
            reason: Nonempty rejection or cancellation reason retained in diagnostics.
        """
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("release reason must be a nonempty string.")
        attempt = self._attempt(
            identity, "proposed", "ready", "assigned", "running", "write_failed"
        )
        attempt.state, attempt.trajectory, attempt.episode = "released", None, None
        attempt.episode_byte_budget = 0
        attempt.reason = reason

    def episode_ids(self, identity: CandidateIdentity) -> tuple[str, str]:
        """Return the deterministic episode and idempotent commit IDs for an attempt.

        Args:
            identity: Candidate identity registered in this session.

        Returns:
            The episode ID and commit ID, stable across persistence retries.
        """
        self._attempt(identity)
        digest = _digest(identity.candidate_id, identity.attempt_id)
        return "episode_" + digest, "commit_" + digest

    def _episode_bytes(self, episode: ExpertEpisode) -> int:
        return _tensor_bytes(
            episode.actions, episode.timestamps, *episode.observations.values()
        ) + _json_bytes(
            (
                episode.identity,
                episode.episode_id,
                episode.commit_id,
                episode.action_representation,
                tuple(episode.observations),
                episode.validation,
                episode.phases,
                episode.metadata,
            )
        )

    def _pending_reservation(self) -> tuple[int, int]:
        count, size = 0, 0
        for attempt in self._attempts.values():
            if attempt.episode is not None:
                count += 1
                size += self._episode_bytes(attempt.episode)
            elif attempt.state in {"assigned", "running"}:
                count += 1
                size += attempt.episode_byte_budget
        return count, size

    def _reserve_coverage(self, episode: ExpertEpisode) -> bool:
        identity = episode.identity
        limits = self._cases[(identity.scene_case_id, identity.initial_state_id)][1]
        descriptor = describe_trajectory(
            episode.observations["joint_positions"],
            episode.timestamps,
            limits,
            phases=episode.phases,
            samples_per_phase=self._cfg.augmentation.coverage.geometry_samples_per_phase,
        )
        return self._coverage[identity.scene_case_id].reserve(
            episode.commit_id, identity.geometry_family_id, descriptor
        )

    def accept_episode(self, episode: ExpertEpisode) -> bool:
        """Reserve passed measured evidence for writing; rejected evidence releases its slot.

        Joint positions, observation timestamps, and actual phase intervals
        supply coverage. Planned targets are never used as measured evidence.

        Args:
            episode: Actual commands, observations, and validation for a running candidate.

        Returns:
            ``True`` when the episode is reserved for persistence; ``False`` when
            validation or coverage rejects it. No storage write occurs here.
        """
        attempt = self._attempt(episode.identity, "running")
        if (episode.episode_id, episode.commit_id) != self.episode_ids(
            episode.identity
        ):
            raise ValueError("Episode and commit IDs must match the candidate attempt.")
        attempt.rollout_validation = _validation_summary(episode.validation)
        if episode.action_representation != "qpos" or episode.actions.shape[1] != len(
            attempt.joint_names
        ):
            raise ValueError(
                "Episode actions must use full-joint qpos labels in candidate joint order."
            )
        if not self._validation_passes(episode.validation, rollout=True):
            self.release(episode.identity, reason="rollout_validation_failed")
            return False
        positions = episode.observations.get("joint_positions")
        if (
            positions is None
            or positions.ndim != 2
            or positions.shape[1] != len(attempt.joint_names)
        ):
            raise ValueError(
                "Evidence requires actual joint_positions in the candidate joint order."
            )
        if self._episode_bytes(episode) > attempt.episode_byte_budget:
            self.release(episode.identity, reason="episode_byte_budget_exceeded")
            raise BufferError("Payload exceeds its reserved episode byte budget.")
        owned = replace(episode)
        if not self._reserve_coverage(owned):
            self.release(episode.identity, reason="coverage_rejected")
            return False
        attempt.state, attempt.episode = "pending_write", owned
        attempt.episode_byte_budget = 0
        self._commits[episode.commit_id] = episode.identity.candidate_id
        self._counts["validated_accepted"] += 1
        return True

    def apply_receipt(self, receipt: CommitReceipt) -> bool:
        """Apply a final submission result to its original identity, idempotently.

        A repeated receipt is ignored. Different outcomes for one submission
        are invalid even after retry; the sink must resolve ambiguous writes
        before reporting a final failure and releasing its reservation.

        Args:
            receipt: Final outcome for an outstanding episode write submission.

        Returns:
            ``True`` when a new success or failure receipt is applied, or ``False``
            for an identical previously applied receipt.
        """
        attempt = self._attempts[self._commits[receipt.commit_id]]
        identity = attempt.identity
        expected = (
            *self.episode_ids(identity),
            identity.candidate_id,
            identity.attempt_id,
            identity.scene_case_id,
        )
        if (
            receipt.episode_id,
            receipt.commit_id,
            receipt.candidate_id,
            receipt.attempt_id,
            receipt.scene_case_id,
        ) != expected:
            raise ValueError("Receipt identity does not match its original candidate.")
        key = (receipt.commit_id, receipt.submission_id)
        if key in self._receipts:
            if self._receipts[key] != receipt:
                raise ValueError(
                    "Contradictory final receipts for one write submission."
                )
            return False
        if (
            attempt.state != "pending_write"
            or receipt.submission_id != attempt.submission_id
        ):
            raise ValueError("Receipt does not identify a pending write submission.")
        coverage = self._coverage[identity.scene_case_id]
        if receipt.confirmed:
            coverage.confirm(receipt.commit_id)
            attempt.state, attempt.episode = "committed", None
            self._counts["committed"] += 1
        else:
            coverage.release(receipt.commit_id)
            attempt.state = "write_failed"
            attempt.reason = receipt.error or "write_failed"
        self._receipts[key] = receipt
        return True

    def retry_write(self, commit_id: str) -> tuple[ExpertEpisode, int]:
        """Re-reserve a failed payload and return a copy plus its next submission ID.

        A failed write frees collection and coverage quota. Retry therefore
        fails if another candidate has occupied either quota in the meantime.
        It never adds a proposal, rollout attempt, or accepted-episode count.

        Args:
            commit_id: Stable commit identifier of an episode whose last write failed.

        Returns:
            An owned episode copy and the incremented submission ID for the retry.
        """
        attempt = self._attempts[self._commits[commit_id]]
        if attempt.state != "write_failed":
            raise ValueError("Only a final failed write can be retried.")
        if (
            self._reserved() >= self._cfg.collection.target_committed_episodes
            or self._clock() - self._started_at >= self._cfg.collection.max_wall_time_s
        ):
            raise RuntimeError(
                "No collection quota or time remains for this write retry."
            )
        if not self._reserve_coverage(attempt.episode):
            raise RuntimeError(
                "Coverage quota is no longer available for this write retry."
            )
        attempt.state = "pending_write"
        attempt.reason = None
        attempt.submission_id += 1
        return replace(attempt.episode), attempt.submission_id

    @property
    def stop_reason(self) -> str | None:
        """Report exhausted collection budgets while allowing outstanding results to finish.

        Returns:
            An exhausted budget identifier, or ``None`` while new work is permitted.
        """
        collection = self._cfg.collection
        if self._counts["committed"] >= collection.target_committed_episodes:
            return "target_reached"
        if self._clock() - self._started_at >= collection.max_wall_time_s:
            return "wall_time_exhausted"
        active = any(
            attempt.state in {"assigned", "running", "pending_write"}
            for attempt in self._attempts.values()
        )
        if (
            not active
            and self._counts["rollout_attempted"] >= collection.max_rollout_attempts
        ):
            return "rollout_budget_exhausted"
        if (
            not active
            and self._counts["proposed"] >= collection.max_proposals
            and not any(
                attempt.state in {"proposed", "ready"}
                for attempt in self._attempts.values()
            )
        ):
            return "proposal_budget_exhausted"
        return None

    def snapshot(self) -> Mapping[str, object]:
        """Return read-only counters, case coverage, and lightweight lifecycle audit.

        Diagnostics retain each check's status, first 256 detail characters,
        and first 16 numeric metrics. Results support at most 64 checks.
        Pending reservations cover assigned/running upper bounds and retained
        payload bytes, including failed writes that remain available for retry.

        Returns:
            Immutable mappings of counters, reservations, coverage, candidate audit
            entries, diagnostics, and the current stop reason.
        """
        counts = dict(self._counts)
        pending_count, pending_bytes = self._pending_reservation()
        for state in ("ready", "assigned", "running", "pending_write", "write_failed"):
            counts[state] = sum(
                attempt.state == state for attempt in self._attempts.values()
            )
        return MappingProxyType(
            {
                "counts": MappingProxyType(counts),
                "pending_reserved_episodes": pending_count,
                "pending_reserved_bytes": pending_bytes,
                "coverage": MappingProxyType(
                    {key: value.geometry_count for key, value in self._coverage.items()}
                ),
                "audit": tuple(
                    (attempt.identity, attempt.state, attempt.submission_id)
                    for attempt in self._attempts.values()
                ),
                "diagnostics": MappingProxyType(
                    {
                        candidate_id: MappingProxyType(
                            {
                                "reason": attempt.reason,
                                "planning_validation": attempt.planning_validation,
                                "rollout_validation": attempt.rollout_validation,
                            }
                        )
                        for candidate_id, attempt in self._attempts.items()
                    }
                ),
                "stop_reason": self.stop_reason,
            }
        )
