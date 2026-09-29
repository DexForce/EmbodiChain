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

"""Source-neutral G-03 cases, attempt outcomes and persistence receipts."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from types import MappingProxyType
from typing import Any, Literal, Mapping

__all__ = ["GenerationAttempt", "GenerationCase", "PersistenceReceipt"]


def _text(value: object, name: str) -> str:
    """Validate one stable nonempty identifier."""
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be a nonempty string without outer whitespace")
    return value


def _json_mapping(value: Mapping[str, object], name: str) -> Mapping[str, object]:
    """Snapshot finite JSON metadata without importing a project serializer."""
    try:
        json.dumps(dict(value), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain finite JSON values") from exc
    return MappingProxyType(dict(value))


@dataclass(frozen=True)
class GenerationCase:
    """One fixed expert-generation case resolved before attempts start."""

    experiment_id: str
    case_id: str
    source_kind: Literal[
        "handwritten", "motion_generator", "atomic_action", "task_program"
    ]
    source_id: str
    source_revision: str
    scene_case_id: str
    initial_state_id: str
    seed: int
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "experiment_id",
            "case_id",
            "source_id",
            "source_revision",
            "scene_case_id",
            "initial_state_id",
        ):
            _text(getattr(self, name), name)
        if self.source_kind not in {
            "handwritten",
            "motion_generator",
            "atomic_action",
            "task_program",
        }:
            raise ValueError(f"unsupported source_kind: {self.source_kind!r}")
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer")
        object.__setattr__(self, "metadata", _json_mapping(self.metadata, "metadata"))

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible case manifest row."""
        return {
            "experiment_id": self.experiment_id,
            "case_id": self.case_id,
            "source_kind": self.source_kind,
            "source_id": self.source_id,
            "source_revision": self.source_revision,
            "scene_case_id": self.scene_case_id,
            "initial_state_id": self.initial_state_id,
            "seed": self.seed,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class PersistenceReceipt:
    """Final sink outcome for one episode write submission.

    ``commit_id`` identifies the logical episode across retries and
    ``submission_id`` identifies one concrete write attempt. A confirmed receipt
    is the only persistence evidence that can produce an accepted episode.
    """

    episode_id: str
    commit_id: str
    submission_id: int
    storage_id: str
    confirmed: bool
    error: str = ""

    def __post_init__(self) -> None:
        for name in ("episode_id", "commit_id", "storage_id"):
            _text(getattr(self, name), name)
        if type(self.submission_id) is not int or self.submission_id < 0:
            raise ValueError("submission_id must be a nonnegative integer")
        if type(self.confirmed) is not bool:
            raise ValueError("confirmed must be a boolean")
        if not isinstance(self.error, str):
            raise ValueError("error must be a string")
        if self.confirmed and self.error:
            raise ValueError("confirmed receipts cannot contain an error")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible receipt."""
        return {
            "episode_id": self.episode_id,
            "commit_id": self.commit_id,
            "submission_id": self.submission_id,
            "storage_id": self.storage_id,
            "confirmed": self.confirmed,
            "error": self.error,
        }


@dataclass(frozen=True)
class GenerationAttempt:
    """One bounded physical-generation attempt with independent stage states."""

    case: GenerationCase
    attempt_id: int
    execution_status: Literal[
        "not_run", "completed", "failed", "timeout", "unsupported"
    ]
    validation_status: Literal["not_run", "passed", "failed", "unsupported"]
    task_status: Literal["not_evaluated", "passed", "failed", "unsupported"]
    receipt: PersistenceReceipt | None = None
    elapsed_s: float = 0.0
    metrics: Mapping[str, object] = field(default_factory=dict)
    failure_reason: str | None = None
    evidence: tuple[str, ...] = ()
    data_status: Literal[
        "not_evaluated", "accepted", "rejected", "not_persisted", "duplicate"
    ] = "not_evaluated"

    def __post_init__(self) -> None:
        if not isinstance(self.case, GenerationCase):
            raise ValueError("case must be a GenerationCase")
        if type(self.attempt_id) is not int or self.attempt_id < 0:
            raise ValueError("attempt_id must be a nonnegative integer")
        if self.execution_status not in {
            "not_run",
            "completed",
            "failed",
            "timeout",
            "unsupported",
        }:
            raise ValueError("unknown execution_status")
        if self.validation_status not in {"not_run", "passed", "failed", "unsupported"}:
            raise ValueError("unknown validation_status")
        if self.task_status not in {
            "not_evaluated",
            "passed",
            "failed",
            "unsupported",
        }:
            raise ValueError("unknown task_status")
        if not math.isfinite(self.elapsed_s) or self.elapsed_s < 0:
            raise ValueError("elapsed_s must be finite and nonnegative")
        if self.data_status not in {
            "not_evaluated",
            "accepted",
            "rejected",
            "not_persisted",
            "duplicate",
        }:
            raise ValueError("unknown data_status")
        object.__setattr__(self, "metrics", _json_mapping(self.metrics, "metrics"))
        if any(not isinstance(item, str) for item in self.evidence):
            raise ValueError("evidence must contain strings")
        if self.receipt is not None and not isinstance(
            self.receipt, PersistenceReceipt
        ):
            raise ValueError("receipt must be a PersistenceReceipt or None")

    @property
    def execution_completed(self) -> bool:
        """Return whether the attempt reached a completed execution stage."""
        return self.execution_status == "completed"

    @property
    def validation_passed(self) -> bool:
        """Return whether measured validation passed."""
        return self.validation_status == "passed"

    @property
    def task_passed(self) -> bool:
        """Return whether the task/effect predicate passed."""
        return self.task_status == "passed"

    def with_data_status(self, status: str) -> "GenerationAttempt":
        """Return an owned copy with runner-assigned persistence state."""
        return GenerationAttempt(
            case=self.case,
            attempt_id=self.attempt_id,
            execution_status=self.execution_status,
            validation_status=self.validation_status,
            task_status=self.task_status,
            receipt=self.receipt,
            elapsed_s=self.elapsed_s,
            metrics=self.metrics,
            failure_reason=self.failure_reason,
            evidence=self.evidence,
            data_status=status,
        )

    def to_dict(self) -> dict[str, object]:
        """Return a raw attempt record suitable for ``raw.jsonl``."""
        return {
            "record_id": f"{self.case.case_id}/attempt-{self.attempt_id}",
            "experiment_id": self.case.experiment_id,
            "case_id": self.case.case_id,
            "attempt_id": self.attempt_id,
            "source_kind": self.case.source_kind,
            "source_id": self.case.source_id,
            "source_revision": self.case.source_revision,
            "scene_case_id": self.case.scene_case_id,
            "initial_state_id": self.case.initial_state_id,
            "seed": self.case.seed,
            "execution_status": self.execution_status,
            "validation_status": self.validation_status,
            "task_status": self.task_status,
            "data_status": self.data_status,
            "elapsed_s": self.elapsed_s,
            "metrics": dict(self.metrics),
            "failure_reason": self.failure_reason,
            "evidence": list(self.evidence),
            "receipt": None if self.receipt is None else self.receipt.to_dict(),
        }
