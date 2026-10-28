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
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, Protocol

import torch

if TYPE_CHECKING:
    from .models import BenchmarkCase, CaseOutcome

__all__ = [
    "CapabilityDecision",
    "EvaluationDomain",
    "ExecutionObservation",
    "PhysicalEvaluation",
    "PhysicalStageEvaluator",
    "ReplayObservationTrace",
    "ReplayObservationWindow",
    "StageOutcome",
    "validate_physical_evaluations",
]


_REPLAY_CHANNELS = frozenset(
    {
        "tcp_pose",
        "object_pose",
        "scene_object_pose",
        "articulation_joint",
        "gripper_qpos",
    }
)


@dataclass(frozen=True)
class ReplayObservationWindow:
    """One compiled-coordinate window requested by a physical evaluator.

    ``start`` and ``stop`` are half-open waypoint indices. A terminal-settle
    request uses ``start == stop`` at the trajectory boundary together with
    ``include_settle=True``. Providers resolve semantic segment names while
    they still have the compiled trajectory; evaluators receive only the
    measured trace identified by ``observation_id``.
    """

    observation_id: str
    start: int
    stop: int
    channels: frozenset[str]
    include_settle: bool = False
    scene_object_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.observation_id, str) or not self.observation_id.strip():
            raise ValueError("observation_id must be a non-empty string.")
        if type(self.start) is not int or type(self.stop) is not int:
            raise TypeError("Replay observation bounds must be integers.")
        if self.start < 0 or self.stop < self.start:
            raise ValueError(
                "Replay observation bounds must be ordered and non-negative."
            )
        channels = frozenset(self.channels)
        if not channels or not channels <= _REPLAY_CHANNELS:
            raise ValueError(
                "Replay observation channels must be a non-empty subset of "
                f"{sorted(_REPLAY_CHANNELS)}."
            )
        if self.start == self.stop and not self.include_settle:
            raise ValueError("An empty waypoint window must include terminal settle.")
        scene_object_ids = tuple(self.scene_object_ids)
        if any(
            not isinstance(object_id, str) or not object_id.strip()
            for object_id in scene_object_ids
        ):
            raise ValueError("scene_object_ids must contain non-empty strings.")
        if len(scene_object_ids) != len(set(scene_object_ids)):
            raise ValueError("scene_object_ids must be unique.")
        if ("scene_object_pose" in channels) != bool(scene_object_ids):
            raise ValueError(
                "scene_object_pose windows must declare scene_object_ids, and "
                "scene_object_ids require the scene_object_pose channel."
            )
        object.__setattr__(self, "channels", channels)
        object.__setattr__(self, "scene_object_ids", scene_object_ids)


@dataclass(frozen=True)
class ReplayObservationTrace:
    """Raw synchronized simulator samples for one requested replay window.

    Every tensor starts with ``(sample, environment)``. Optional channels are
    absent evidence; gripper tensors additionally retain their joint axis.
    """

    tcp_pose: torch.Tensor | None = None
    object_pose: torch.Tensor | None = None
    articulation_joint: torch.Tensor | None = None
    gripper_qpos: Mapping[str, torch.Tensor] = field(default_factory=dict)
    scene_object_poses: Mapping[str, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        tensors = [
            value
            for value in (self.tcp_pose, self.object_pose, self.articulation_joint)
            if value is not None
        ]
        grippers = dict(self.gripper_qpos)
        scene_objects = dict(self.scene_object_poses)
        tensors.extend(grippers.values())
        tensors.extend(scene_objects.values())
        if not tensors:
            raise ValueError(
                "A replay observation trace requires at least one channel."
            )
        if any(not isinstance(value, torch.Tensor) for value in tensors):
            raise TypeError("Replay observation channels must be torch tensors.")
        shape = tensors[0].shape[:2]
        if len(shape) != 2 or shape[0] < 1 or shape[1] < 1:
            raise ValueError(
                "Replay observation channels must start with non-empty sample and "
                "environment dimensions."
            )
        if any(value.dim() < 2 or value.shape[:2] != shape for value in tensors):
            raise ValueError(
                "Replay observation channels must share sample and environment axes."
            )
        if self.tcp_pose is not None and self.tcp_pose.shape[2:] != (4, 4):
            raise ValueError("tcp_pose trace must have shape (S, B, 4, 4).")
        if self.object_pose is not None and self.object_pose.shape[2:] != (4, 4):
            raise ValueError("object_pose trace must have shape (S, B, 4, 4).")
        if self.articulation_joint is not None and self.articulation_joint.dim() != 2:
            raise ValueError("articulation_joint trace must have shape (S, B).")
        if any(not part for part in grippers):
            raise ValueError("Gripper trace keys must be non-empty strings.")
        if any(not object_id for object_id in scene_objects):
            raise ValueError("Scene-object trace keys must be non-empty strings.")
        if any(value.shape[2:] != (4, 4) for value in scene_objects.values()):
            raise ValueError("scene_object_poses traces must have shape (S, B, 4, 4).")
        object.__setattr__(
            self, "tcp_pose", None if self.tcp_pose is None else self.tcp_pose.clone()
        )
        object.__setattr__(
            self,
            "object_pose",
            None if self.object_pose is None else self.object_pose.clone(),
        )
        object.__setattr__(
            self,
            "articulation_joint",
            (
                None
                if self.articulation_joint is None
                else self.articulation_joint.clone()
            ),
        )
        object.__setattr__(
            self,
            "gripper_qpos",
            MappingProxyType({part: value.clone() for part, value in grippers.items()}),
        )
        object.__setattr__(
            self,
            "scene_object_poses",
            MappingProxyType(
                {object_id: value.clone() for object_id, value in scene_objects.items()}
            ),
        )


@dataclass(frozen=True)
class EvaluationDomain:
    """Versioned evaluation population; classification does not generate cases."""

    id: str
    version: str
    kind: Literal["nominal", "robustness", "held_out"]
    required_capabilities: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        for name in ("id", "version"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"domain.{name} must be a non-empty string.")
        if self.kind not in {"nominal", "robustness", "held_out"}:
            raise ValueError("domain.kind must be nominal, robustness, or held_out.")
        for capability in self.required_capabilities:
            if not isinstance(capability, str) or not capability.strip():
                raise ValueError(
                    "domain.required_capabilities must contain non-empty strings."
                )


@dataclass(frozen=True)
class CapabilityDecision:
    """Independent capability preflight result for one benchmark population."""

    status: Literal["supported", "unsupported", "invalid"]
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.status not in {"supported", "unsupported", "invalid"}:
            raise ValueError(f"Unknown capability status: {self.status!r}.")
        if self.status == "supported" and self.reason is not None:
            raise ValueError("A supported capability decision cannot have a reason.")
        if self.status != "supported" and not self.reason:
            raise ValueError(
                "Unsupported or invalid capability decisions require a reason."
            )


@dataclass(frozen=True)
class StageOutcome:
    """One physical stage; unreached and inapplicable stages are not failures.

    Measurements retain numeric values and units in their keys. Evidence must
    come from execution observations, never projected planning effects.
    """

    stage_id: str
    status: Literal["passed", "failed", "not_reached", "not_applicable", "unavailable"]
    failure_code: str | None = None
    measurements: dict[str, float | bool] = field(default_factory=dict)
    evidence_source: str | None = None
    observation_window: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.stage_id, str) or not self.stage_id.strip():
            raise ValueError("stage_id must be a non-empty string.")
        if self.status not in {
            "passed",
            "failed",
            "not_reached",
            "not_applicable",
            "unavailable",
        }:
            raise ValueError(f"Unknown stage status: {self.status!r}.")
        if self.status in {"failed", "unavailable"}:
            if not isinstance(self.failure_code, str) or not self.failure_code.strip():
                raise ValueError(
                    "A failed or unavailable stage requires a failure_code."
                )
        elif self.failure_code is not None:
            raise ValueError(
                "Only a failed or unavailable stage may carry a failure_code."
            )
        for name, value in self.measurements.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Measurement names must be non-empty strings.")
            if not isinstance(value, (int, float, bool)) or not math.isfinite(value):
                raise ValueError("Stage measurements must be finite numeric values.")
        for value in (self.evidence_source, self.observation_window):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError("Evidence provenance must be a non-empty string.")


@dataclass(frozen=True)
class PhysicalEvaluation:
    """Ordered stage results for one environment, independent of plan success."""

    env_index: int
    stages: tuple[StageOutcome, ...]
    protocol_version: str | None = None

    def __post_init__(self) -> None:
        if type(self.env_index) is not int or self.env_index < 0:
            raise ValueError("env_index must be a non-negative integer.")
        if not self.stages:
            raise ValueError("Physical evaluation requires at least one stage.")
        if self.protocol_version is not None and (
            not isinstance(self.protocol_version, str)
            or not self.protocol_version.strip()
        ):
            raise ValueError("protocol_version must be a non-empty string.")
        names = [stage.stage_id for stage in self.stages]
        if len(names) != len(set(names)):
            raise ValueError("Physical stage ids must be unique.")
        stopped = False
        for stage in self.stages:
            if stopped and stage.status in {"passed", "failed", "unavailable"}:
                raise ValueError(
                    "A reached stage cannot follow a failed or unreached stage."
                )
            stopped = stopped or stage.status in {
                "failed",
                "not_reached",
                "unavailable",
            }

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
                if stage.status in {"failed", "not_reached", "unavailable"}
            ),
            None,
        )

    @property
    def failure_code(self) -> str | None:
        """Return the measured failure or an explicit incomplete-evaluation code."""
        if self.task_success:
            return None
        return next(
            (
                stage.failure_code
                for stage in self.stages
                if stage.status in {"failed", "unavailable"}
            ),
            "physical_stages_incomplete",
        )


@dataclass(frozen=True)
class ExecutionObservation:
    """Measured execution summary shared with physical evaluators.

    Optional values and missing replay channels are unavailable evidence, not
    successful effects. Provider-declared traces carry raw segment samples;
    no compiled trajectory or projected task context is carried here.
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
    articulation_joint_effect_end: torch.Tensor | None = None
    object_rotation_peak_signed_rad: torch.Tensor | None = None
    scene_object_final_poses: Mapping[str, torch.Tensor] = field(default_factory=dict)
    final_gripper_qpos: Mapping[str, torch.Tensor] = field(default_factory=dict)
    gripper_open_qpos: Mapping[str, torch.Tensor] = field(default_factory=dict)
    gripper_grasp_qpos: Mapping[str, torch.Tensor] = field(default_factory=dict)
    replay_traces: Mapping[str, ReplayObservationTrace] = field(default_factory=dict)
    replay_diagnostics: Mapping[str, object] = field(default_factory=dict)
    """Optional bounded JSON-native measurements persisted with the trial."""


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
    if len({result.protocol_version for result in ordered}) != 1:
        raise ValueError(
            "Physical protocol version must match across environment rows."
        )
    return ordered
