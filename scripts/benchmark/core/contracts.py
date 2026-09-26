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

"""Domain-neutral benchmark definitions and evidence records.

The records in this module are deliberately independent of NumPy, Torch and
simulator packages. Domain workers may add fields to their JSON payloads, while
these records provide the common identity, lifecycle and accounting contract.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
import json
import math
import os
from types import MappingProxyType
from typing import Any, Literal, Mapping, Sequence

__all__ = [
    "AggregateMetric",
    "Budget",
    "DataStatus",
    "ExecutionStatus",
    "ExperimentDefinition",
    "MetricDefinition",
    "QualityStatus",
    "RawObservation",
    "RunRecord",
    "TaskStatus",
]

ExecutionStatus = Literal["not_run", "completed", "failed", "timeout", "interrupted"]
QualityStatus = Literal["qualified", "not_qualified", "unsupported", "missing"]
TaskStatus = Literal["not_evaluated", "passed", "failed", "unsupported"]
DataStatus = Literal[
    "not_evaluated", "accepted", "rejected", "not_persisted", "duplicate"
]


def _json_value(value: object) -> object:
    """Convert immutable record values into JSON-compatible containers."""
    if is_dataclass(value) and not isinstance(value, type):
        return _json_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    return value


def _validate_json(value: object, *, name: str) -> None:
    """Reject non-finite values before they enter a frozen definition."""
    try:
        json.dumps(_json_value(value), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite JSON data") from exc


@dataclass(frozen=True)
class Budget:
    """Fixed limits for runs, attempts and wall-clock execution."""

    max_runs: int | None = None
    max_attempts: int | None = None
    wall_time_s: float | None = None

    def __post_init__(self) -> None:
        for name in ("max_runs", "max_attempts"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"{name} must be a positive integer or None")
        if self.wall_time_s is not None and (
            isinstance(self.wall_time_s, bool)
            or not math.isfinite(self.wall_time_s)
            or self.wall_time_s <= 0
        ):
            raise ValueError("wall_time_s must be positive and finite or None")

    def to_dict(self) -> dict[str, object]:
        """Return the budget with explicit nullable limits."""
        return {
            "max_runs": self.max_runs,
            "max_attempts": self.max_attempts,
            "wall_time_s": self.wall_time_s,
        }


@dataclass(frozen=True)
class MetricDefinition:
    """Definition of one raw metric and its counting population."""

    key: str
    unit: str
    population: str
    definition_version: str = "1"
    denominator: str = "completed_runs"

    def __post_init__(self) -> None:
        for name in ("key", "unit", "population", "definition_version", "denominator"):
            if (
                not isinstance(getattr(self, name), str)
                or not getattr(self, name).strip()
            ):
                raise ValueError(f"{name} must be a nonempty string")

    def to_dict(self) -> dict[str, str]:
        """Return a serializable metric definition."""
        return {
            "key": self.key,
            "unit": self.unit,
            "population": self.population,
            "definition_version": self.definition_version,
            "denominator": self.denominator,
        }


@dataclass(frozen=True)
class ExperimentDefinition:
    """Frozen experiment identity, parameter matrix, quality and budget rules."""

    experiment_id: str
    definition_version: str = "1"
    parameter_matrix: Mapping[str, Sequence[object]] = field(default_factory=dict)
    budget: Budget = field(default_factory=Budget)
    quality_protocol: Mapping[str, object] = field(default_factory=dict)
    comparison_invariants: tuple[str, ...] = ()
    comparison_relationships: tuple[Mapping[str, object], ...] = ()
    metric_definitions: tuple[MetricDefinition, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.experiment_id, str) or not self.experiment_id.strip():
            raise ValueError("experiment_id must be a nonempty string")
        if (
            not isinstance(self.definition_version, str)
            or not self.definition_version.strip()
        ):
            raise ValueError("definition_version must be a nonempty string")
        normalized: dict[str, tuple[object, ...]] = {}
        for key, values in self.parameter_matrix.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("parameter matrix keys must be nonempty strings")
            values_tuple = tuple(values)
            if not values_tuple:
                raise ValueError(f"parameter matrix {key!r} must not be empty")
            _validate_json(values_tuple, name=f"parameter_matrix[{key!r}]")
            normalized[key] = values_tuple
        _validate_json(self.quality_protocol, name="quality_protocol")
        for path in self.comparison_invariants:
            if not isinstance(path, str) or not path.strip():
                raise ValueError("comparison invariants must be nonempty strings")
        _validate_json(self.comparison_relationships, name="comparison_relationships")
        object.__setattr__(self, "parameter_matrix", MappingProxyType(normalized))
        object.__setattr__(
            self,
            "quality_protocol",
            MappingProxyType(dict(self.quality_protocol)),
        )

    def to_dict(self) -> dict[str, object]:
        """Return the definition used to freeze an experiment directory."""
        return {
            "schema_version": 1,
            "experiment_id": self.experiment_id,
            "definition_version": self.definition_version,
            "parameter_matrix": {
                key: [_json_value(item) for item in values]
                for key, values in self.parameter_matrix.items()
            },
            "budget": self.budget.to_dict(),
            "quality_protocol": _json_value(self.quality_protocol),
            "comparison_invariants": list(self.comparison_invariants),
            "comparison_relationships": _json_value(self.comparison_relationships),
            "metric_definitions": [
                metric.to_dict() for metric in self.metric_definitions
            ],
        }


@dataclass(frozen=True)
class RunRecord:
    """Common run lifecycle record with independent quality and data states."""

    experiment_id: str
    run_id: str
    case_id: str
    backend: str
    repeat: int
    status: ExecutionStatus = "not_run"
    attempt: int = 0
    quality_status: QualityStatus = "missing"
    task_status: TaskStatus = "not_evaluated"
    data_status: DataStatus = "not_evaluated"
    hardware_id: str | None = None
    software: Mapping[str, object] = field(default_factory=dict)
    asset_sha256: str | None = None
    config_sha256: str | None = None
    metrics: Mapping[str, object] = field(default_factory=dict)
    failure_reason: str | None = None
    evidence: tuple[str, ...] = ()
    missing_metrics: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not all(
            isinstance(getattr(self, name), str) and getattr(self, name).strip()
            for name in ("experiment_id", "run_id", "case_id", "backend")
        ):
            raise ValueError("run identity fields must be nonempty strings")
        if type(self.repeat) is not int or self.repeat < 0:
            raise ValueError("repeat must be a nonnegative integer")
        if type(self.attempt) is not int or self.attempt < 0:
            raise ValueError("attempt must be a nonnegative integer")
        if self.status not in {
            "not_run",
            "completed",
            "failed",
            "timeout",
            "interrupted",
        }:
            raise ValueError(f"unknown execution status: {self.status!r}")
        if self.quality_status not in {
            "qualified",
            "not_qualified",
            "unsupported",
            "missing",
        }:
            raise ValueError(f"unknown quality status: {self.quality_status!r}")
        if self.task_status not in {"not_evaluated", "passed", "failed", "unsupported"}:
            raise ValueError(f"unknown task status: {self.task_status!r}")
        if self.data_status not in {
            "not_evaluated",
            "accepted",
            "rejected",
            "not_persisted",
            "duplicate",
        }:
            raise ValueError(f"unknown data status: {self.data_status!r}")
        _validate_json(self.software, name="software")
        _validate_json(self.metrics, name="metrics")
        _validate_json(self.missing_metrics, name="missing_metrics")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible run record."""
        return {
            "schema_version": 1,
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "case_id": self.case_id,
            "backend": self.backend,
            "repeat": self.repeat,
            "attempt": self.attempt,
            "status": self.status,
            "execution_status": self.status,
            "quality_status": self.quality_status,
            "task_status": self.task_status,
            "data_status": self.data_status,
            "hardware_id": self.hardware_id,
            "software": _json_value(self.software),
            "asset_sha256": self.asset_sha256,
            "config_sha256": self.config_sha256,
            "metrics": _json_value(self.metrics),
            "failure_reason": self.failure_reason,
            "evidence": list(self.evidence),
            "missing_metrics": _json_value(self.missing_metrics),
        }


@dataclass(frozen=True)
class RawObservation:
    """One raw stage, continuous sample or attempt observation."""

    record_id: str
    run_id: str
    case_id: str
    stage: str
    index: int
    elapsed_s: float | None = None
    status: str = "completed"
    counters: Mapping[str, object] = field(default_factory=dict)
    metrics: Mapping[str, object] = field(default_factory=dict)
    failure_reason: str | None = None
    evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not all(
            isinstance(getattr(self, name), str) and getattr(self, name).strip()
            for name in ("record_id", "run_id", "case_id", "stage")
        ):
            raise ValueError("observation identity fields must be nonempty strings")
        if type(self.index) is not int or self.index < 0:
            raise ValueError("index must be a nonnegative integer")
        if self.elapsed_s is not None and (
            not isinstance(self.elapsed_s, (int, float))
            or isinstance(self.elapsed_s, bool)
            or not math.isfinite(float(self.elapsed_s))
            or self.elapsed_s < 0
        ):
            raise ValueError("elapsed_s must be finite and nonnegative or None")
        _validate_json(self.counters, name="counters")
        _validate_json(self.metrics, name="metrics")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible observation."""
        return {
            "schema_version": 1,
            "record_id": self.record_id,
            "run_id": self.run_id,
            "case_id": self.case_id,
            "stage": self.stage,
            "index": self.index,
            "elapsed_s": self.elapsed_s,
            "status": self.status,
            "counters": _json_value(self.counters),
            "metrics": _json_value(self.metrics),
            "failure_reason": self.failure_reason,
            "evidence": list(self.evidence),
        }


@dataclass(frozen=True)
class AggregateMetric:
    """Offline aggregate with denominator, uncertainty and source run IDs."""

    key: str
    definition_version: str
    unit: str
    population: str
    aggregation: str
    denominator: int
    valid_count: int
    missing_count: int
    value: float | None
    uncertainty: Mapping[str, object] | None
    source_runs: tuple[str, ...]
    missing_reasons: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.denominator < 0 or self.valid_count < 0 or self.missing_count < 0:
            raise ValueError("aggregate counts must be nonnegative")
        if self.valid_count + self.missing_count != self.denominator:
            raise ValueError("aggregate counts must conserve the denominator")
        if self.value is not None and not math.isfinite(float(self.value)):
            raise ValueError("aggregate value must be finite or None")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible aggregate metric."""
        return {
            "key": self.key,
            "definition_version": self.definition_version,
            "unit": self.unit,
            "population": self.population,
            "aggregation": self.aggregation,
            "denominator": self.denominator,
            "valid_count": self.valid_count,
            "missing_count": self.missing_count,
            "value": self.value,
            "uncertainty": _json_value(self.uncertainty),
            "source_runs": list(self.source_runs),
            "missing_reasons": _json_value(self.missing_reasons),
        }
