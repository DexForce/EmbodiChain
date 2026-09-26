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

"""Simulator-independent contracts shared by Atomic Skill benchmark owners."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol

import torch

if TYPE_CHECKING:
    from .models import BenchmarkCase, CaseOutcome

__all__ = [
    "EvaluationDomain",
    "ExecutionObservation",
    "PhysicalEvaluation",
    "PhysicalStageEvaluator",
    "StageOutcome",
    "validate_physical_evaluations",
]


@dataclass(frozen=True)
class EvaluationDomain:
    """Versioned evaluation population; classification does not generate cases."""

    id: str
    version: str
    kind: Literal["nominal", "robustness", "held_out"]

    def __post_init__(self) -> None:
        for name in ("id", "version"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"domain.{name} must be a non-empty string.")
        if self.kind not in {"nominal", "robustness", "held_out"}:
            raise ValueError("domain.kind must be nominal, robustness, or held_out.")


@dataclass(frozen=True)
class StageOutcome:
    """One physical stage; unreached and inapplicable stages are not failures.

    Measurements retain numeric values and units in their keys. Evidence must
    come from execution observations, never projected planning effects.
    """

    stage_id: str
    status: Literal["passed", "failed", "not_reached", "not_applicable"]
    failure_code: str | None = None
    measurements: dict[str, float | bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.stage_id, str) or not self.stage_id.strip():
            raise ValueError("stage_id must be a non-empty string.")
        if self.status not in {"passed", "failed", "not_reached", "not_applicable"}:
            raise ValueError(f"Unknown stage status: {self.status!r}.")
        if self.status == "failed":
            if not isinstance(self.failure_code, str) or not self.failure_code.strip():
                raise ValueError("A failed stage requires a failure_code.")
        elif self.failure_code is not None:
            raise ValueError("Only a failed stage may carry a failure_code.")
        for name, value in self.measurements.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Measurement names must be non-empty strings.")
            if not isinstance(value, (int, float, bool)) or not math.isfinite(value):
                raise ValueError("Stage measurements must be finite numeric values.")


@dataclass(frozen=True)
class PhysicalEvaluation:
    """Ordered stage results for one environment, independent of plan success."""

    env_index: int
    stages: tuple[StageOutcome, ...]

    def __post_init__(self) -> None:
        if type(self.env_index) is not int or self.env_index < 0:
            raise ValueError("env_index must be a non-negative integer.")
        if not self.stages:
            raise ValueError("Physical evaluation requires at least one stage.")
        names = [stage.stage_id for stage in self.stages]
        if len(names) != len(set(names)):
            raise ValueError("Physical stage ids must be unique.")
        stopped = False
        for stage in self.stages:
            if stopped and stage.status in {"passed", "failed"}:
                raise ValueError(
                    "A reached stage cannot follow a failed or unreached stage."
                )
            stopped = stopped or stage.status in {"failed", "not_reached"}

    @property
    def task_success(self) -> bool:
        """Return whether at least one applicable stage exists and all passed."""
        applicable = [
            stage for stage in self.stages if stage.status != "not_applicable"
        ]
        return bool(applicable) and all(
            stage.status == "passed" for stage in applicable
        )

    @property
    def failure_stage(self) -> str | None:
        """Return the first failed or unreached stage, if any."""
        return next(
            (
                stage.stage_id
                for stage in self.stages
                if stage.status in {"failed", "not_reached"}
            ),
            None,
        )

    @property
    def failure_code(self) -> str | None:
        """Return the measured failure or an explicit incomplete-evaluation code."""
        if self.task_success:
            return None
        return next(
            (stage.failure_code for stage in self.stages if stage.status == "failed"),
            "physical_stages_incomplete",
        )


@dataclass(frozen=True)
class ExecutionObservation:
    """Measured execution summary shared with physical evaluators.

    Optional values are unavailable evidence, not successful effects. Segment
    sampling and skill-specific contact evidence can extend this contract in
    the physical-evaluation follow-up; no projected context is carried here.
    """

    execution_success: torch.Tensor
    final_tcp_pose: torch.Tensor
    joint_tracking_rmse_rad: torch.Tensor
    execution_time_ms: float
    task_completion_time_s: float
    object_lift_delta_m: torch.Tensor | None = None
    final_arm_qpos: torch.Tensor | None = None
    final_object_pose: torch.Tensor | None = None
    articulation_joint_initial: torch.Tensor | None = None
    articulation_joint_final: torch.Tensor | None = None
    articulation_joint_peak_signed_delta: torch.Tensor | None = None


class PhysicalStageEvaluator(Protocol):
    """Evaluate measured scenes without access to a compiled/projected plan."""

    def evaluate(
        self,
        case: BenchmarkCase,
        observation: ExecutionObservation,
        motion_outcomes: tuple[CaseOutcome, ...],
    ) -> tuple[PhysicalEvaluation, ...]:
        """Return physical results for every environment in the case.

        Args:
            case: Frozen planner-independent task inputs.
            observation: Measured replay evidence.
            motion_outcomes: Motion validation outcomes in environment order.

        Returns:
            One result per environment with a common ordered stage schema.
        """
        ...


def validate_physical_evaluations(
    results: tuple[PhysicalEvaluation, ...],
    *,
    batch_size: int,
) -> tuple[PhysicalEvaluation, ...]:
    """Validate batch coverage and stage order, then order results by environment.

    Args:
        results: Evaluator outputs in any environment order.
        batch_size: Number of environments that must be represented exactly once.

    Returns:
        Results ordered by environment index.

    Raises:
        ValueError: Rows are missing/duplicated or stage schemas differ.
    """
    if batch_size < 1 or len(results) != batch_size:
        raise ValueError(
            "Physical evaluator must return every environment exactly once."
        )
    ordered = tuple(sorted(results, key=lambda result: result.env_index))
    if [result.env_index for result in ordered] != list(range(batch_size)):
        raise ValueError(
            "Physical evaluator returned invalid or duplicate environment indices."
        )
    schema = tuple(stage.stage_id for stage in ordered[0].stages)
    if any(
        tuple(stage.stage_id for stage in result.stages) != schema for result in ordered
    ):
        raise ValueError("Physical stage order must match across environment rows.")
    return ordered
