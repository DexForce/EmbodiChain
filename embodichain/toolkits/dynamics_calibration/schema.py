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

"""Configuration and result contracts for dynamics calibration."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import MISSING, dataclass
from numbers import Integral
from pathlib import Path
from typing import Any

import yaml

from embodichain.utils import configclass

_DRIVE_FIELDS = {
    "armature",
    "damping",
    "friction",
    "max_effort",
    "max_velocity",
    "stiffness",
}


@dataclass(frozen=True)
class ControlSchedule:
    """An application control period resolved onto physics updates.

    Attributes:
        physics_steps_per_control: Integral physics updates per control sample.
        requested_hz: Application-requested control frequency.
        actual_hz: Frequency produced by the integral physics schedule.
    """

    physics_steps_per_control: int
    requested_hz: float
    actual_hz: float

    @property
    def relative_error(self) -> float:
        """Return the relative difference between actual and requested rates.

        Returns:
            Absolute frequency error divided by ``requested_hz``.
        """
        return abs(self.actual_hz - self.requested_hz) / self.requested_hz

    def to_dict(self) -> dict[str, int | float]:
        """Return a JSON-serializable representation.

        Returns:
            Integral schedule and requested, actual, and relative frequencies.
        """
        return {
            "physics_steps_per_control": self.physics_steps_per_control,
            "requested_hz": self.requested_hz,
            "actual_hz": self.actual_hz,
            "relative_error": self.relative_error,
        }


def resolve_control_schedule(
    physics_dt: float,
    requested_hz: float,
    *,
    allow_approximate: bool = False,
    tolerance: float = 1.0e-9,
) -> ControlSchedule:
    """Resolve a control frequency without silently changing its timing.

    Args:
        physics_dt: Duration of one physics update in seconds.
        requested_hz: Requested application control frequency.
        allow_approximate: Permit the nearest integral number of physics steps.
        tolerance: Relative frequency error accepted as exact.

    Raises:
        TypeError: If ``allow_approximate`` is not a boolean.
        ValueError: If values are invalid or the requested frequency is not exact.

    Returns:
        Exact or explicitly permitted approximate integral control schedule.
    """
    if not isinstance(allow_approximate, bool):
        raise TypeError("allow_approximate must be a boolean")
    physics_dt = float(physics_dt)
    requested_hz = float(requested_hz)
    tolerance = float(tolerance)
    if not math.isfinite(physics_dt) or physics_dt <= 0.0:
        raise ValueError("physics_dt must be finite and greater than zero")
    if not math.isfinite(requested_hz) or requested_hz <= 0.0:
        raise ValueError("requested_hz must be finite and greater than zero")
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and non-negative")

    ideal_steps = 1.0 / (physics_dt * requested_hz)
    lower_steps = max(1, math.floor(ideal_steps))
    upper_steps = max(1, math.ceil(ideal_steps))
    physics_steps = min(
        {lower_steps, upper_steps},
        key=lambda steps: (
            abs((1.0 / (physics_dt * steps)) - requested_hz) / requested_hz,
            steps,
        ),
    )
    actual_hz = 1.0 / (physics_dt * physics_steps)
    relative_error = abs(actual_hz - requested_hz) / requested_hz
    if relative_error > tolerance and not allow_approximate:
        raise ValueError(
            f"{requested_hz:g} Hz cannot be represented exactly by physics_dt "
            f"{physics_dt:g}; nearest is {actual_hz:g} Hz "
            f"({physics_steps} physics steps)"
        )
    return ControlSchedule(physics_steps, requested_hz, actual_hz)


@configclass
class EvaluatorConfig:
    """Configuration for one isolated application evaluator.

    Attributes:
        target: ``module:function`` or ``/path/to/file.py:function`` callable.
        timeout_seconds: Maximum wall time for each worker process.
        payload: Application-owned strict-JSON evaluator configuration.
    """

    target: str = MISSING
    timeout_seconds: float = 300.0
    payload: dict[str, Any] = {}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, base_dir: Path) -> EvaluatorConfig:
        """Parse and validate evaluator configuration.

        Args:
            data: Evaluator mapping from the calibration configuration.
            base_dir: Directory used to resolve evaluator file paths.

        Returns:
            Validated evaluator configuration with absolute file targets.

        Raises:
            FileNotFoundError: If an explicit evaluator file does not exist.
            TypeError: If the payload is not a mapping.
            ValueError: If the target or timeout is invalid.
        """
        _reject_unknown(data, {"target", "timeout_seconds", "payload"}, "evaluator")
        target = str(data.get("target", "")).strip()
        if not target or ":" not in target:
            raise ValueError("evaluator.target must use 'module:function' syntax")
        module_or_path, attribute = target.rsplit(":", maxsplit=1)
        if not module_or_path or not attribute:
            raise ValueError("evaluator.target must use 'module:function' syntax")
        possible_path = Path(module_or_path).expanduser()
        if possible_path.suffix == ".py":
            if not possible_path.is_absolute():
                possible_path = base_dir / possible_path
            if not possible_path.is_file():
                raise FileNotFoundError(
                    f"evaluator module does not exist: {possible_path.resolve()}"
                )
            target = f"{possible_path.resolve()}:{attribute}"

        timeout_seconds = float(data.get("timeout_seconds", 300.0))
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0.0:
            raise ValueError("evaluator.timeout_seconds must be greater than zero")
        payload = data.get("payload", {})
        if not isinstance(payload, Mapping):
            raise TypeError("evaluator.payload must be a mapping")
        return cls(
            target=target,
            timeout_seconds=timeout_seconds,
            payload=dict(payload),
        )


@configclass
class DriveParameterSpec:
    """One bounded effective drive parameter exposed to the search.

    Attributes:
        name: Unique candidate-coordinate name.
        field: RobotCfg ``drive_pros`` field to tune.
        selector: Exact name, regular expression, or control-part selector.
        lower: Inclusive lower search bound.
        upper: Inclusive upper search bound.
        initial: Baseline value included as the first candidate.
        scale: ``linear`` or ``log`` sampling scale.
    """

    name: str = MISSING
    field: str = MISSING
    selector: str = MISSING
    lower: float = MISSING
    upper: float = MISSING
    initial: float = MISSING
    scale: str = "linear"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DriveParameterSpec:
        """Parse and validate one drive parameter.

        Args:
            data: Parameter name, target, bounds, initial value, and scale.

        Returns:
            Validated bounded parameter specification.

        Raises:
            KeyError: If a required bound is absent.
            TypeError: If a numeric field cannot be converted.
            ValueError: If names, bounds, initial value, field, or scale are
                invalid.
        """
        _reject_unknown(
            data,
            {"name", "field", "selector", "lower", "upper", "initial", "scale"},
            "parameter",
        )
        name = str(data.get("name", "")).strip()
        field = str(data.get("field", "")).strip()
        selector = str(data.get("selector", "")).strip()
        if not name:
            raise ValueError("parameter.name cannot be empty")
        if field not in _DRIVE_FIELDS:
            raise ValueError(
                f"parameter {name!r} field must be one of {sorted(_DRIVE_FIELDS)}"
            )
        if not selector:
            raise ValueError(f"parameter {name!r} selector cannot be empty")
        lower = float(data["lower"])
        upper = float(data["upper"])
        initial = float(data.get("initial", (lower + upper) / 2.0))
        if not all(math.isfinite(value) for value in (lower, upper, initial)):
            raise ValueError(f"parameter {name!r} bounds must be finite")
        if lower >= upper:
            raise ValueError(f"parameter {name!r} lower must be less than upper")
        if not lower <= initial <= upper:
            raise ValueError(f"parameter {name!r} initial must lie within its bounds")
        scale = str(data.get("scale", "linear"))
        if scale not in {"linear", "log"}:
            raise ValueError(f"parameter {name!r} scale must be 'linear' or 'log'")
        if scale == "log" and lower <= 0.0:
            raise ValueError(f"log-scaled parameter {name!r} must have positive bounds")
        return cls(name, field, selector, lower, upper, initial, scale)


@configclass
class QualificationThresholds:
    """Hard gates used to admit a candidate after held-out evaluation.

    Attributes:
        aggregate_rmse_max: Maximum flattened RMSE.
        aggregate_p95_max: Maximum flattened absolute-error P95.
        per_joint_rmse_max: Maximum RMSE for every individual joint.
        per_joint_p95_max: Maximum absolute-error P95 for every joint.
        per_control_group_rmse_max: Maximum RMSE for every control group.
        per_control_group_p95_max: Maximum absolute-error P95 per group.
        worst_joint_rmse_max: Maximum worst-joint RMSE.
        cvar95_max: Maximum mean absolute error in the worst five-percent tail.
        overshoot_max: Maximum application-defined overshoot observation.
        settling_time_seconds_max: Maximum application-defined settling time.
        saturation_fraction_max: Maximum effort saturation fraction.
        velocity_saturation_fraction_max: Maximum velocity saturation fraction.
        joint_limit_violation_max: Maximum position-limit violation.
        control_frequency_relative_error_max: Maximum requested/actual control
            frequency mismatch.
        expected_target_qvel_write_count: Exact expected target-velocity API
            write count, or ``None`` to disable this instrumentation gate.
        require_stable: Require finite, stable evaluator evidence.
    """

    aggregate_rmse_max: float | None = None
    aggregate_p95_max: float | None = None
    per_joint_rmse_max: float | None = None
    per_joint_p95_max: float | None = None
    per_control_group_rmse_max: float | None = None
    per_control_group_p95_max: float | None = None
    worst_joint_rmse_max: float | None = None
    cvar95_max: float | None = None
    overshoot_max: float | None = None
    settling_time_seconds_max: float | None = None
    saturation_fraction_max: float | None = None
    velocity_saturation_fraction_max: float | None = None
    joint_limit_violation_max: float | None = None
    control_frequency_relative_error_max: float | None = 0.0
    expected_target_qvel_write_count: int | None = 0
    require_stable: bool = True

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> QualificationThresholds:
        """Parse and validate qualification thresholds.

        Args:
            data: Hard-gate names and non-negative bounds.

        Returns:
            Validated qualification policy.

        Raises:
            TypeError: If boolean or count fields have incompatible types.
            ValueError: If a threshold is unknown, negative, or non-finite.
        """
        allowed = set(cls.__annotations__)
        _reject_unknown(data, allowed, "qualification")
        values = dict(data)
        for key in allowed - {"require_stable", "expected_target_qvel_write_count"}:
            if key in values and values[key] is not None:
                values[key] = float(values[key])
                if not math.isfinite(values[key]) or values[key] < 0.0:
                    raise ValueError(
                        f"qualification.{key} must be finite and non-negative"
                    )
        if values.get("expected_target_qvel_write_count") is not None:
            values["expected_target_qvel_write_count"] = _parse_integer(
                values["expected_target_qvel_write_count"],
                "qualification.expected_target_qvel_write_count",
            )
            if values["expected_target_qvel_write_count"] < 0:
                raise ValueError(
                    "qualification.expected_target_qvel_write_count cannot be negative"
                )
        if "require_stable" in values:
            values["require_stable"] = _parse_boolean(
                values["require_stable"], "qualification.require_stable"
            )
        return cls(**values)


@configclass
class CalibrationConfig:
    """Complete effective-drive calibration configuration.

    Attributes:
        schema_version: Configuration schema version; V1 is currently supported.
        assets: Absolute robot asset paths.
        backend: Application-selected physics backend identifier.
        device: Evaluator device identifier.
        physics_dt: Physics update duration in seconds.
        control_frequency_hz: Requested application control frequency.
        allow_approximate_control_frequency: Permit an explicitly recorded
            approximate integral control schedule.
        seed: Deterministic candidate-design seed.
        candidate_count: Number of candidates to evaluate.
        evaluator: Isolated application evaluator configuration.
        parameters: Bounded effective drive parameters to search.
        qualification: Hard gates for held-out admission.
    """

    assets: list[str] = MISSING
    evaluator: EvaluatorConfig = MISSING
    parameters: list[DriveParameterSpec] = MISSING
    schema_version: int = 1
    backend: str = "default"
    device: str = "cpu"
    physics_dt: float = 1.0 / 240.0
    control_frequency_hz: float = 60.0
    allow_approximate_control_frequency: bool = False
    seed: int = 0
    candidate_count: int = 9
    qualification: QualificationThresholds = QualificationThresholds()

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        base_dir: str | Path = ".",
    ) -> CalibrationConfig:
        """Parse a calibration configuration and resolve local paths.

        Args:
            data: Complete V1 calibration mapping.
            base_dir: Directory used to resolve relative asset and evaluator
                paths.

        Returns:
            Validated configuration with absolute asset and evaluator paths.

        Raises:
            FileNotFoundError: If a configured asset or evaluator file is absent.
            TypeError: If nested configuration values have incompatible types.
            ValueError: If fields, parameters, bounds, or timing are invalid.
        """
        allowed = set(cls.__annotations__)
        _reject_unknown(data, allowed, "calibration")
        schema_version = _parse_integer(data.get("schema_version", 1), "schema_version")
        if schema_version != 1:
            raise ValueError(f"unsupported calibration schema_version {schema_version}")
        resolved_base = Path(base_dir).expanduser().resolve()
        raw_assets = data.get("assets", [])
        if not isinstance(raw_assets, list) or not raw_assets:
            raise ValueError("assets must be a non-empty list")
        assets: list[str] = []
        for raw_path in raw_assets:
            asset = Path(str(raw_path)).expanduser()
            if not asset.is_absolute():
                asset = resolved_base / asset
            asset = asset.resolve()
            if not asset.is_file():
                raise FileNotFoundError(f"calibration asset does not exist: {asset}")
            assets.append(str(asset))

        physics_dt = float(data.get("physics_dt", 1.0 / 240.0))
        control_frequency_hz = float(data.get("control_frequency_hz", 60.0))
        allow_approximate = _parse_boolean(
            data.get("allow_approximate_control_frequency", False),
            "allow_approximate_control_frequency",
        )
        resolve_control_schedule(
            physics_dt,
            control_frequency_hz,
            allow_approximate=allow_approximate,
        )

        candidate_count = _parse_integer(
            data.get("candidate_count", 9), "candidate_count"
        )
        if candidate_count < 1:
            raise ValueError("candidate_count must be at least one")
        raw_evaluator = data.get("evaluator")
        if not isinstance(raw_evaluator, Mapping):
            raise TypeError("evaluator must be a mapping")
        raw_parameters = data.get("parameters")
        if not isinstance(raw_parameters, list) or not raw_parameters:
            raise ValueError("parameters must be a non-empty list")
        parameters = []
        for index, item in enumerate(raw_parameters):
            if not isinstance(item, Mapping):
                raise TypeError(f"parameters[{index}] must be a mapping")
            parameters.append(DriveParameterSpec.from_dict(item))
        names = [parameter.name for parameter in parameters]
        if len(names) != len(set(names)):
            raise ValueError("parameter names must be unique")
        targets = [(parameter.field, parameter.selector) for parameter in parameters]
        if len(targets) != len(set(targets)):
            raise ValueError("each drive field/selector pair may be tuned only once")
        raw_qualification = data.get("qualification", {})
        if not isinstance(raw_qualification, Mapping):
            raise TypeError("qualification must be a mapping")

        backend = str(data.get("backend", "default")).strip()
        if not backend:
            raise ValueError("backend cannot be empty")
        device = str(data.get("device", "cpu")).strip()
        if not device:
            raise ValueError("device cannot be empty")
        return cls(
            schema_version=schema_version,
            assets=assets,
            backend=backend,
            device=device,
            physics_dt=physics_dt,
            control_frequency_hz=control_frequency_hz,
            allow_approximate_control_frequency=allow_approximate,
            seed=_parse_integer(data.get("seed", 0), "seed"),
            candidate_count=candidate_count,
            evaluator=EvaluatorConfig.from_dict(raw_evaluator, base_dir=resolved_base),
            parameters=parameters,
            qualification=QualificationThresholds.from_dict(raw_qualification),
        )

    def asset_records(self) -> list[dict[str, str]]:
        """Return immutable asset identities used by overlays and cache keys.

        Returns:
            Ordered absolute asset paths and their SHA-256 digests.
        """
        return [
            {
                "path": path,
                "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            }
            for path in self.assets
        ]

    def evaluation_context(self, phase: str) -> dict[str, Any]:
        """Build the factual, serializable context passed to an evaluator.

        Args:
            phase: ``training`` for search or ``qualification`` for held-out
                admission.

        Returns:
            Strict-JSON-compatible asset, runtime, timing, seed, and payload
            context.

        Raises:
            ValueError: If ``phase`` is unsupported or the configured control
                schedule is not representable.
        """
        if phase not in {"training", "qualification"}:
            raise ValueError("evaluation phase must be 'training' or 'qualification'")
        schedule = resolve_control_schedule(
            self.physics_dt,
            self.control_frequency_hz,
            allow_approximate=self.allow_approximate_control_frequency,
        )
        return {
            "schema_version": self.schema_version,
            "phase": phase,
            "assets": self.asset_records(),
            "backend": self.backend,
            "device": self.device,
            "physics_dt": self.physics_dt,
            "requested_control_hz": schedule.requested_hz,
            "actual_control_hz": schedule.actual_hz,
            "physics_steps_per_control": schedule.physics_steps_per_control,
            "seed": self.seed,
            "payload": dict(self.evaluator.payload),
        }


def _reject_unknown(data: Mapping[str, Any], allowed: set[str], location: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"unknown {location} fields: {', '.join(unknown)}")


def _parse_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    return int(value)


def _parse_boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")
    return value


def load_calibration_config(path: str | Path) -> CalibrationConfig:
    """Load a YAML or JSON calibration configuration from disk.

    Args:
        path: Configuration file path.

    Returns:
        Validated configuration with paths resolved relative to the file.

    Raises:
        FileNotFoundError: If the configuration or a referenced local file is
            absent.
        TypeError: If nested configuration values have incompatible types.
        ValueError: If the document is not a valid V1 calibration mapping.
    """
    source = Path(path).expanduser().resolve()
    loaded = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):
        raise ValueError("calibration configuration must contain a mapping")
    return CalibrationConfig.from_dict(loaded, base_dir=source.parent)


__all__ = [
    "CalibrationConfig",
    "ControlSchedule",
    "DriveParameterSpec",
    "EvaluatorConfig",
    "QualificationThresholds",
    "load_calibration_config",
    "resolve_control_schedule",
]
