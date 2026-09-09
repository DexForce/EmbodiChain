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

"""Tracking metrics and hard qualification gates."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from .schema import QualificationThresholds


@dataclass(frozen=True)
class TrackingMetrics:
    """Serializable tracking evidence produced from one evaluator run.

    Attributes:
        joint_names: Ordered joint names matching observation columns.
        sample_count: Number of time samples in the evaluation.
        aggregate_rmse: Root-mean-square error over all joints and samples.
        aggregate_p95: 95th percentile of absolute tracking error.
        per_joint_rmse: RMSE indexed by joint name.
        per_joint_p95: Absolute-error P95 indexed by joint name.
        per_control_group_rmse: RMSE indexed by application control group.
        per_control_group_p95: Absolute-error P95 by control group.
        worst_joint_rmse: Largest per-joint RMSE.
        worst_joint: Joint associated with ``worst_joint_rmse``.
        cvar95: Mean absolute error over the worst five-percent tail.
        overshoot: Optional application-defined step-response overshoot.
        settling_time_seconds: Optional application-defined settling time.
        saturation_fraction: Fraction of effort samples at their limits.
        velocity_saturation_fraction: Fraction of velocity samples at limits.
        joint_limit_violation: Maximum observed position-limit violation.
        requested_control_hz: Application-requested control frequency.
        actual_control_hz: Frequency represented by the physics schedule.
        control_frequency_relative_error: Relative requested/actual mismatch.
        target_qvel_write_count: Observed target-velocity API write count, or
            ``None`` when the evaluator supplied no instrumentation evidence.
        stable: Whether all required observations remained finite and stable.
    """

    joint_names: tuple[str, ...]
    sample_count: int
    aggregate_rmse: float
    aggregate_p95: float
    per_joint_rmse: dict[str, float]
    per_joint_p95: dict[str, float]
    per_control_group_rmse: dict[str, float]
    per_control_group_p95: dict[str, float]
    worst_joint_rmse: float
    worst_joint: str
    cvar95: float
    overshoot: float | None
    settling_time_seconds: float | None
    saturation_fraction: float | None
    velocity_saturation_fraction: float | None
    joint_limit_violation: float | None
    requested_control_hz: float
    actual_control_hz: float
    control_frequency_relative_error: float
    target_qvel_write_count: int | None
    stable: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation.

        Returns:
            Metrics payload with non-finite values represented as strings.
        """
        return {
            "joint_names": list(self.joint_names),
            "sample_count": self.sample_count,
            "aggregate_rmse": _json_number(self.aggregate_rmse),
            "aggregate_p95": _json_number(self.aggregate_p95),
            "per_joint_rmse": _json_number_mapping(self.per_joint_rmse),
            "per_joint_p95": _json_number_mapping(self.per_joint_p95),
            "per_control_group_rmse": _json_number_mapping(self.per_control_group_rmse),
            "per_control_group_p95": _json_number_mapping(self.per_control_group_p95),
            "worst_joint_rmse": _json_number(self.worst_joint_rmse),
            "worst_joint": self.worst_joint,
            "cvar95": _json_number(self.cvar95),
            "overshoot": _json_number(self.overshoot),
            "settling_time_seconds": _json_number(self.settling_time_seconds),
            "saturation_fraction": _json_number(self.saturation_fraction),
            "velocity_saturation_fraction": _json_number(
                self.velocity_saturation_fraction
            ),
            "joint_limit_violation": _json_number(self.joint_limit_violation),
            "requested_control_hz": self.requested_control_hz,
            "actual_control_hz": self.actual_control_hz,
            "control_frequency_relative_error": _json_number(
                self.control_frequency_relative_error
            ),
            "target_qvel_write_count": self.target_qvel_write_count,
            "stable": self.stable,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TrackingMetrics:
        """Restore metrics from a cache or report dictionary.

        Args:
            data: Serialized metrics payload produced by :meth:`to_dict`.

        Returns:
            Reconstructed immutable metrics.

        Raises:
            KeyError: If a required metric is absent.
            TypeError: If instrumentation fields have incompatible types.
            ValueError: If numeric fields cannot be parsed.
        """
        return cls(
            joint_names=tuple(str(item) for item in data["joint_names"]),
            sample_count=int(data["sample_count"]),
            aggregate_rmse=float(data["aggregate_rmse"]),
            aggregate_p95=float(data["aggregate_p95"]),
            per_joint_rmse={
                str(key): float(value)
                for key, value in dict(data["per_joint_rmse"]).items()
            },
            per_joint_p95={
                str(key): float(value)
                for key, value in dict(data["per_joint_p95"]).items()
            },
            per_control_group_rmse={
                str(key): float(value)
                for key, value in dict(data.get("per_control_group_rmse", {})).items()
            },
            per_control_group_p95={
                str(key): float(value)
                for key, value in dict(data.get("per_control_group_p95", {})).items()
            },
            worst_joint_rmse=float(data["worst_joint_rmse"]),
            worst_joint=str(data["worst_joint"]),
            cvar95=float(data["cvar95"]),
            overshoot=_optional_float(data.get("overshoot")),
            settling_time_seconds=_optional_float(data.get("settling_time_seconds")),
            saturation_fraction=_optional_float(data.get("saturation_fraction")),
            velocity_saturation_fraction=_optional_float(
                data.get("velocity_saturation_fraction")
            ),
            joint_limit_violation=_optional_float(data.get("joint_limit_violation")),
            requested_control_hz=float(data["requested_control_hz"]),
            actual_control_hz=float(data["actual_control_hz"]),
            control_frequency_relative_error=float(
                data["control_frequency_relative_error"]
            ),
            target_qvel_write_count=_optional_integer(
                data.get("target_qvel_write_count")
            ),
            stable=bool(data["stable"]),
        )


@dataclass(frozen=True)
class QualificationGate:
    """One auditable hard-gate decision.

    Attributes:
        name: Stable metric or invariant name.
        passed: Whether the observed value satisfies the gate.
        observed: Value used for the decision, or ``None`` when unavailable.
        expected: Configured upper bound or exact expected value.
        entity: Optional joint or control-group associated with the value.
    """

    name: str
    passed: bool
    observed: float | int | bool | None
    expected: float | int | bool
    entity: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation.

        Returns:
            Gate decision and associated observed and expected values.
        """
        return {
            "name": self.name,
            "passed": self.passed,
            "observed": _json_number(self.observed),
            "expected": self.expected,
            "entity": self.entity,
        }


@dataclass(frozen=True)
class QualificationResult:
    """Aggregate qualification status with every evaluated gate.

    Attributes:
        status: ``pass`` only when every configured gate passes; otherwise
            ``fail``.
        gates: Complete ordered gate evidence.
    """

    status: str
    gates: tuple[QualificationGate, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation.

        Returns:
            Aggregate status and all individual gate decisions.
        """
        return {
            "status": self.status,
            "gates": [gate.to_dict() for gate in self.gates],
        }


def compute_tracking_metrics(raw: Mapping[str, Any]) -> TrackingMetrics:
    """Compute aggregate and per-joint metrics from evaluator observations.

    The evaluator owns application execution but not metric definitions. This
    central calculation keeps candidate comparisons and qualification stable.

    Args:
        raw: Evaluator observations containing joint names, target and actual
            position matrices, requested and actual control frequencies, and
            optional group, limit, effort, velocity, and stability evidence.

    Returns:
        Validated aggregate, per-joint, group, tail, and safety metrics.

    Raises:
        KeyError: If a required observation is absent.
        TypeError: If an observation has an incompatible type.
        ValueError: If observation shapes, names, limits, or frequencies are
            inconsistent.
    """
    raw_joint_names = raw["joint_names"]
    if not isinstance(raw_joint_names, list) or not all(
        isinstance(name, str) and name for name in raw_joint_names
    ):
        raise TypeError("joint_names must be a list of non-empty strings")
    joint_names = tuple(raw_joint_names)
    target = np.asarray(raw["target_qpos"], dtype=np.float64)
    actual = np.asarray(raw["actual_qpos"], dtype=np.float64)
    if target.ndim != 2 or actual.ndim != 2:
        raise ValueError("target_qpos and actual_qpos must be two-dimensional")
    if target.shape != actual.shape:
        raise ValueError("target_qpos and actual_qpos must have identical shapes")
    if target.shape[0] == 0 or target.shape[1] == 0:
        raise ValueError("tracking observations cannot be empty")
    if target.shape[1] != len(joint_names):
        raise ValueError("joint_names must match the observation joint dimension")
    if len(joint_names) != len(set(joint_names)):
        raise ValueError("joint_names must be unique")

    with np.errstate(over="ignore", invalid="ignore"):
        error = actual - target
    finite = bool(np.isfinite(error).all())
    safe_error = np.where(np.isfinite(error), error, np.inf)
    absolute_error = np.abs(safe_error)
    per_joint_rmse_values = np.asarray(
        [_rmse(safe_error[:, index]) for index in range(safe_error.shape[1])]
    )
    per_joint_p95_values = np.asarray(
        [_p95(absolute_error[:, index]) for index in range(absolute_error.shape[1])]
    )
    per_joint_rmse = {
        name: float(value) for name, value in zip(joint_names, per_joint_rmse_values)
    }
    per_joint_p95 = {
        name: float(value) for name, value in zip(joint_names, per_joint_p95_values)
    }
    group_rmse, group_p95 = _compute_control_group_metrics(raw, joint_names, safe_error)
    worst_index = int(np.argmax(per_joint_rmse_values))
    flattened = np.sort(absolute_error.reshape(-1))
    tail_count = max(1, int(math.ceil(flattened.size * 0.05)))

    requested_hz = float(raw["requested_control_hz"])
    actual_hz = float(raw["actual_control_hz"])
    if not math.isfinite(requested_hz) or requested_hz <= 0.0:
        raise ValueError("requested_control_hz must be finite and greater than zero")
    if not math.isfinite(actual_hz) or actual_hz <= 0.0:
        raise ValueError("actual_control_hz must be finite and greater than zero")

    saturation_fraction, effort_finite = _compute_saturation_fraction(
        raw,
        actual.shape,
        value_name="effort",
        limit_name="effort_limits",
    )
    velocity_saturation_fraction, velocity_finite = _compute_saturation_fraction(
        raw,
        actual.shape,
        value_name="qvel",
        limit_name="qvel_limits",
    )
    joint_limit_violation = _compute_joint_limit_violation(raw, actual)
    overshoot, overshoot_finite = _optional_nonnegative_metric(raw, "overshoot")
    settling_time, settling_finite = _optional_nonnegative_metric(
        raw, "settling_time_seconds"
    )
    raw_stable = raw.get("stable", True)
    if not isinstance(raw_stable, bool):
        raise TypeError("stable must be a boolean when provided")
    raw_qvel_writes = raw.get("target_qvel_write_count")
    if raw_qvel_writes is not None:
        if isinstance(raw_qvel_writes, bool) or not isinstance(
            raw_qvel_writes, (int, np.integer)
        ):
            raise TypeError("target_qvel_write_count must be an integer")
        if raw_qvel_writes < 0:
            raise ValueError("target_qvel_write_count cannot be negative")
    stable = (
        raw_stable
        and finite
        and effort_finite
        and velocity_finite
        and overshoot_finite
        and settling_finite
    )
    return TrackingMetrics(
        joint_names=joint_names,
        sample_count=target.shape[0],
        aggregate_rmse=_rmse(safe_error),
        aggregate_p95=_p95(absolute_error),
        per_joint_rmse=per_joint_rmse,
        per_joint_p95=per_joint_p95,
        per_control_group_rmse=group_rmse,
        per_control_group_p95=group_p95,
        worst_joint_rmse=float(per_joint_rmse_values[worst_index]),
        worst_joint=joint_names[worst_index],
        cvar95=_mean_absolute(flattened[-tail_count:]),
        overshoot=overshoot,
        settling_time_seconds=settling_time,
        saturation_fraction=saturation_fraction,
        velocity_saturation_fraction=velocity_saturation_fraction,
        joint_limit_violation=joint_limit_violation,
        requested_control_hz=requested_hz,
        actual_control_hz=actual_hz,
        control_frequency_relative_error=abs(actual_hz - requested_hz) / requested_hz,
        target_qvel_write_count=(
            None if raw_qvel_writes is None else int(raw_qvel_writes)
        ),
        stable=stable,
    )


def qualify(
    metrics: TrackingMetrics, thresholds: QualificationThresholds
) -> QualificationResult:
    """Apply configured aggregate, per-joint, timing, and stability gates.

    Args:
        metrics: Held-out application tracking and safety evidence.
        thresholds: Hard upper bounds and exact expected invariants.

    Returns:
        Overall pass/fail result with every configured gate preserved.
    """
    gates: list[QualificationGate] = []
    _append_upper_gate(
        gates, "aggregate_rmse", metrics.aggregate_rmse, thresholds.aggregate_rmse_max
    )
    _append_upper_gate(
        gates, "aggregate_p95", metrics.aggregate_p95, thresholds.aggregate_p95_max
    )
    _append_per_joint_gate(
        gates, "per_joint_rmse", metrics.per_joint_rmse, thresholds.per_joint_rmse_max
    )
    _append_per_joint_gate(
        gates, "per_joint_p95", metrics.per_joint_p95, thresholds.per_joint_p95_max
    )
    _append_group_gate(
        gates,
        "per_control_group_rmse",
        metrics.per_control_group_rmse,
        thresholds.per_control_group_rmse_max,
    )
    _append_group_gate(
        gates,
        "per_control_group_p95",
        metrics.per_control_group_p95,
        thresholds.per_control_group_p95_max,
    )
    _append_upper_gate(
        gates,
        "worst_joint_rmse",
        metrics.worst_joint_rmse,
        thresholds.worst_joint_rmse_max,
        entity=metrics.worst_joint,
    )
    _append_upper_gate(gates, "cvar95", metrics.cvar95, thresholds.cvar95_max)
    _append_upper_gate(gates, "overshoot", metrics.overshoot, thresholds.overshoot_max)
    _append_upper_gate(
        gates,
        "settling_time_seconds",
        metrics.settling_time_seconds,
        thresholds.settling_time_seconds_max,
    )
    _append_upper_gate(
        gates,
        "saturation_fraction",
        metrics.saturation_fraction,
        thresholds.saturation_fraction_max,
    )
    _append_upper_gate(
        gates,
        "velocity_saturation_fraction",
        metrics.velocity_saturation_fraction,
        thresholds.velocity_saturation_fraction_max,
    )
    _append_upper_gate(
        gates,
        "joint_limit_violation",
        metrics.joint_limit_violation,
        thresholds.joint_limit_violation_max,
    )
    _append_upper_gate(
        gates,
        "control_frequency_relative_error",
        metrics.control_frequency_relative_error,
        thresholds.control_frequency_relative_error_max,
    )
    if thresholds.expected_target_qvel_write_count is not None:
        expected = thresholds.expected_target_qvel_write_count
        gates.append(
            QualificationGate(
                "target_qvel_write_count",
                metrics.target_qvel_write_count is not None
                and metrics.target_qvel_write_count == expected,
                metrics.target_qvel_write_count,
                expected,
            )
        )
    if thresholds.require_stable:
        gates.append(QualificationGate("stable", metrics.stable, metrics.stable, True))
    return QualificationResult(
        status="pass" if all(gate.passed for gate in gates) else "fail",
        gates=tuple(gates),
    )


def _compute_saturation_fraction(
    raw: Mapping[str, Any],
    observation_shape: tuple[int, ...],
    *,
    value_name: str,
    limit_name: str,
) -> tuple[float | None, bool]:
    has_values = value_name in raw
    has_limits = limit_name in raw
    if has_values != has_limits:
        raise ValueError(f"{value_name} and {limit_name} must be provided together")
    if not has_values:
        return None, True
    values = np.asarray(raw[value_name], dtype=np.float64)
    limits = np.asarray(raw[limit_name], dtype=np.float64)
    if values.shape != observation_shape:
        raise ValueError(f"{value_name} must match the qpos observation shape")
    try:
        limits = np.broadcast_to(limits, values.shape)
    except ValueError as error:
        raise ValueError(f"{limit_name} cannot be broadcast to {value_name}") from error
    if np.any(~np.isfinite(limits)) or np.any(limits <= 0.0):
        raise ValueError(f"{limit_name} must be finite and greater than zero")
    if not np.isfinite(values).all():
        return math.inf, False
    return float(np.mean(np.abs(values) >= limits)), True


def _optional_nonnegative_metric(
    raw: Mapping[str, Any], name: str
) -> tuple[float | None, bool]:
    if name not in raw:
        return None, True
    value = float(raw[name])
    if math.isnan(value) or value < 0.0:
        return math.inf, False
    return value, math.isfinite(value)


def _compute_control_group_metrics(
    raw: Mapping[str, Any],
    joint_names: tuple[str, ...],
    error: np.ndarray,
) -> tuple[dict[str, float], dict[str, float]]:
    raw_groups = raw.get("control_groups", {})
    if raw_groups is None:
        return {}, {}
    if not isinstance(raw_groups, Mapping):
        raise TypeError("control_groups must be a mapping of group names to joints")
    indices_by_name = {name: index for index, name in enumerate(joint_names)}
    rmse: dict[str, float] = {}
    p95: dict[str, float] = {}
    for raw_name, raw_members in raw_groups.items():
        if not isinstance(raw_name, str) or not raw_name:
            raise TypeError("control group names must be non-empty strings")
        name = raw_name
        if isinstance(raw_members, (str, bytes)) or not isinstance(raw_members, list):
            raise TypeError(f"control group {name!r} must contain a list of joints")
        if not all(isinstance(member, str) and member for member in raw_members):
            raise TypeError(
                f"control group {name!r} must contain non-empty joint names"
            )
        if len(raw_members) != len(set(raw_members)):
            raise ValueError(f"control group {name!r} cannot repeat a joint")
        unknown = [member for member in raw_members if member not in indices_by_name]
        if unknown:
            raise ValueError(
                f"control group {name!r} contains unknown joints: {', '.join(unknown)}"
            )
        if not raw_members:
            raise ValueError(f"control group {name!r} cannot be empty")
        group_error = error[:, [indices_by_name[member] for member in raw_members]]
        rmse[name] = _rmse(group_error)
        p95[name] = _p95(np.abs(group_error))
    return rmse, p95


def _compute_joint_limit_violation(
    raw: Mapping[str, Any], actual: np.ndarray
) -> float | None:
    has_lower = "qpos_lower" in raw
    has_upper = "qpos_upper" in raw
    if has_lower != has_upper:
        raise ValueError("qpos_lower and qpos_upper must be provided together")
    if not has_lower:
        return None
    lower = np.asarray(raw["qpos_lower"], dtype=np.float64)
    upper = np.asarray(raw["qpos_upper"], dtype=np.float64)
    try:
        lower = np.broadcast_to(lower, actual.shape)
        upper = np.broadcast_to(upper, actual.shape)
    except ValueError as error:
        raise ValueError("qpos limits cannot be broadcast to actual_qpos") from error
    if np.isnan(lower).any() or np.isnan(upper).any():
        raise ValueError("qpos limits cannot contain NaN")
    if np.any(lower > upper):
        raise ValueError("qpos_lower cannot exceed qpos_upper")
    violation = np.maximum(np.maximum(lower - actual, actual - upper), 0.0)
    return float(np.max(violation))


def _append_upper_gate(
    gates: list[QualificationGate],
    name: str,
    observed: float | None,
    threshold: float | None,
    *,
    entity: str | None = None,
) -> None:
    if threshold is None:
        return
    passed = observed is not None and math.isfinite(observed) and observed <= threshold
    gates.append(QualificationGate(name, passed, observed, threshold, entity))


def _append_per_joint_gate(
    gates: list[QualificationGate],
    name: str,
    observed: Mapping[str, float],
    threshold: float | None,
) -> None:
    if threshold is None:
        return
    entity, worst = max(observed.items(), key=lambda item: item[1])
    passed = math.isfinite(worst) and worst <= threshold
    gates.append(QualificationGate(name, passed, worst, threshold, entity))


def _append_group_gate(
    gates: list[QualificationGate],
    name: str,
    observed: Mapping[str, float],
    threshold: float | None,
) -> None:
    if threshold is None:
        return
    if not observed:
        gates.append(QualificationGate(name, False, None, threshold))
        return
    entity, worst = max(observed.items(), key=lambda item: item[1])
    passed = math.isfinite(worst) and worst <= threshold
    gates.append(QualificationGate(name, passed, worst, threshold, entity))


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_integer(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("cached target_qvel_write_count must be an integer")
    return value


def _p95(values: np.ndarray) -> float:
    if not np.isfinite(values).all():
        return math.inf
    return float(np.percentile(values, 95.0))


def _rmse(values: np.ndarray) -> float:
    maximum = float(np.max(np.abs(values)))
    if not math.isfinite(maximum):
        return math.inf
    if maximum == 0.0:
        return 0.0
    scaled = values / maximum
    return maximum * float(np.sqrt(np.mean(np.square(scaled))))


def _mean_absolute(values: np.ndarray) -> float:
    maximum = float(np.max(np.abs(values)))
    if not math.isfinite(maximum):
        return math.inf
    if maximum == 0.0:
        return 0.0
    return maximum * float(np.mean(np.abs(values) / maximum))


def _json_number(value: Any) -> Any:
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        if math.isnan(float(value)):
            return "nan"
        return "inf" if float(value) > 0.0 else "-inf"
    return value


def _json_number_mapping(values: Mapping[str, float]) -> dict[str, float | str]:
    return {name: _json_number(value) for name, value in values.items()}


__all__ = [
    "QualificationGate",
    "QualificationResult",
    "TrackingMetrics",
    "compute_tracking_metrics",
    "qualify",
]
