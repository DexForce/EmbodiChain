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

"""Deterministic bounded search for effective drive parameters."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .evaluator import run_candidate
from .metrics import TrackingMetrics
from .overlay import build_drive_overlay
from .schema import CalibrationConfig, DriveParameterSpec


@dataclass(frozen=True)
class TuningTrial:
    """One evaluated candidate and its scalar ranking objective.

    Attributes:
        candidate: Parameter-name to evaluated-value mapping.
        objective: Scalar robust tracking objective used for ranking.
        cache_hit: Whether this trial reused cached evaluator evidence.
        metrics: Centralized metrics for the training trajectory.
        evaluator_metadata: Application evidence attached by the evaluator.
    """

    candidate: dict[str, float]
    objective: float
    cache_hit: bool
    metrics: TrackingMetrics
    evaluator_metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation.

        Returns:
            Candidate, objective, metrics, and evaluator provenance.
        """
        return {
            "candidate": dict(self.candidate),
            "objective": _json_objective(self.objective),
            "cache_hit": self.cache_hit,
            "metrics": self.metrics.to_dict(),
            "evaluator_metadata": dict(self.evaluator_metadata),
        }


@dataclass(frozen=True)
class TuningResult:
    """Best overlay plus complete candidate-search evidence.

    Attributes:
        best_candidate: Parameter mapping with the lowest objective.
        overlay: Non-destructive drive overlay for the best candidate.
        best_objective: Lowest observed training objective.
        baseline_objective: Objective of the configured initial candidate.
        trials: Every evaluated candidate in deterministic order.
    """

    best_candidate: dict[str, float]
    overlay: dict[str, Any]
    best_objective: float
    baseline_objective: float
    trials: tuple[TuningTrial, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation.

        Returns:
            Best candidate, overlay, objectives, and all trial evidence.
        """
        return {
            "best_candidate": dict(self.best_candidate),
            "overlay": self.overlay,
            "best_objective": _json_objective(self.best_objective),
            "baseline_objective": _json_objective(self.baseline_objective),
            "trials": [trial.to_dict() for trial in self.trials],
        }


def tune_drive(config: CalibrationConfig, *, cache_dir: str | Path) -> TuningResult:
    """Evaluate a reproducible candidate design and return the best overlay.

    Args:
        config: Validated calibration search and evaluator configuration.
        cache_dir: Directory for content-addressed candidate results.

    Returns:
        Best candidate and complete deterministic search evidence.

    Raises:
        EvaluationError: If any isolated candidate evaluation fails.
    """
    candidates = _generate_candidates(
        config.parameters, config.candidate_count, config.seed
    )
    trials: list[TuningTrial] = []
    context = config.evaluation_context("training")
    for candidate in candidates:
        overlay = build_drive_overlay(config, candidate)
        evaluation = run_candidate(
            config.evaluator,
            overlay,
            context,
            cache_dir=cache_dir,
        )
        objective = _tracking_objective(
            evaluation.metrics, candidate, config.parameters
        )
        trials.append(
            TuningTrial(
                candidate,
                objective,
                evaluation.cache_hit,
                evaluation.metrics,
                evaluation.metadata,
            )
        )
    best = min(trials, key=lambda trial: trial.objective)
    return TuningResult(
        best_candidate=dict(best.candidate),
        overlay=build_drive_overlay(config, best.candidate),
        best_objective=best.objective,
        baseline_objective=trials[0].objective,
        trials=tuple(trials),
    )


def _generate_candidates(
    parameters: list[DriveParameterSpec], count: int, seed: int
) -> list[dict[str, float]]:
    templates = [
        {parameter.name: parameter.initial for parameter in parameters},
        {parameter.name: parameter.lower for parameter in parameters},
        {parameter.name: parameter.upper for parameter in parameters},
        {parameter.name: _midpoint(parameter) for parameter in parameters},
    ]
    candidates: list[dict[str, float]] = []
    seen: set[tuple[tuple[str, float], ...]] = set()

    def append(candidate: dict[str, float]) -> None:
        key = tuple(sorted(candidate.items()))
        if key not in seen and len(candidates) < count:
            seen.add(key)
            candidates.append(candidate)

    for template in templates:
        append(template)
    generator = random.Random(seed)
    while len(candidates) < count:
        candidate = {
            parameter.name: _sample(parameter, generator) for parameter in parameters
        }
        append(candidate)
    return candidates


def _midpoint(parameter: DriveParameterSpec) -> float:
    if parameter.scale == "log":
        return math.sqrt(parameter.lower * parameter.upper)
    return (parameter.lower + parameter.upper) / 2.0


def _sample(parameter: DriveParameterSpec, generator: random.Random) -> float:
    unit = generator.random()
    if parameter.scale == "log":
        low = math.log(parameter.lower)
        high = math.log(parameter.upper)
        return math.exp(low + unit * (high - low))
    return parameter.lower + unit * (parameter.upper - parameter.lower)


def _tracking_objective(
    metrics: TrackingMetrics,
    candidate: dict[str, float],
    parameters: list[DriveParameterSpec],
) -> float:
    if not metrics.stable:
        return math.inf
    objective = (
        metrics.aggregate_rmse
        + metrics.worst_joint_rmse
        + 0.25 * metrics.aggregate_p95
        + 0.25 * metrics.cvar95
    )
    if metrics.saturation_fraction is not None:
        objective += metrics.saturation_fraction
    if metrics.velocity_saturation_fraction is not None:
        objective += metrics.velocity_saturation_fraction
    if metrics.overshoot is not None:
        objective += metrics.overshoot
    if metrics.joint_limit_violation is not None:
        objective += metrics.joint_limit_violation
    prior_deviation = sum(
        (
            (candidate[parameter.name] - parameter.initial)
            / (parameter.upper - parameter.lower)
        )
        ** 2
        for parameter in parameters
    ) / len(parameters)
    objective += 0.01 * prior_deviation
    return objective


def _json_objective(value: float) -> float | str:
    if math.isfinite(value):
        return value
    if math.isnan(value):
        return "nan"
    return "inf" if value > 0.0 else "-inf"


__all__ = ["TuningResult", "TuningTrial", "tune_drive"]
