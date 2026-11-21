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

"""Immutable task-conditioned search inputs, candidates, and measured evidence."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cached_property
from types import MappingProxyType
from typing import Literal

from embodichain.lab._scene_config.feasibility import (
    HeldObject,
    PhaseTarget,
    PoseSearchVariable,
    TaskFeasibilityRequest,
    TaskOperationMode,
    TaskPhase,
)
from .._validation import _identifier
from ..expansion import SceneVariant
from ._values import digest, finite, names, owned_json, poses

__all__ = [
    "PoseSearchVariable",
    "PhaseTarget",
    "HeldObject",
    "TaskPhase",
    "TaskOperationMode",
    "TaskFeasibilityRequest",
    "RobotDeploymentCandidate",
    "JointSceneCandidate",
    "FeasibilityCheck",
    "CandidateEvaluation",
]


@dataclass(frozen=True)
class RobotDeploymentCandidate:
    """Explicit physical robot root poses, retaining group mounting relations."""

    poses: Mapping[str, tuple]

    def __post_init__(self) -> None:
        object.__setattr__(self, "poses", poses(self.poses))

    @cached_property
    def deployment_id(self) -> str:
        """Return a content identity independent of physical environment slots."""
        return digest(self.poses)


@dataclass(frozen=True)
class JointSceneCandidate:
    """Owned joint scene/deployment proposal, not a success certificate."""

    request_id: str
    scene: SceneVariant
    deployment: RobotDeploymentCandidate
    mode_id: str
    parameters: tuple[float, ...]

    def __post_init__(self) -> None:
        _identifier(self.request_id, "request_id")
        _identifier(self.mode_id, "mode_id")
        if not isinstance(self.scene, SceneVariant) or not isinstance(
            self.deployment, RobotDeploymentCandidate
        ):
            raise TypeError("candidate requires typed scene and deployment")
        object.__setattr__(self, "parameters", finite(self.parameters, "parameters"))

    @cached_property
    def candidate_id(self) -> str:
        """Return an identity including scene, mounting, mode and source ordinal."""
        return digest(self)


@dataclass(frozen=True)
class FeasibilityCheck:
    """One stage outcome with explicit unsupported and budget-limited evidence."""

    stage: str
    status: Literal[
        "passed",
        "failed",
        "unverified",
        "unsupported",
        "no_solution",
        "budget_exhausted",
    ]
    reason: str = ""
    phase_id: str | None = None
    affected_variables: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _identifier(self.stage, "stage")
        if self.status not in (
            "passed",
            "failed",
            "unverified",
            "unsupported",
            "no_solution",
            "budget_exhausted",
        ):
            raise ValueError("invalid feasibility status")
        if not isinstance(self.reason, str):
            raise TypeError("reason must be text")
        object.__setattr__(
            self,
            "affected_variables",
            names(self.affected_variables, "affected variables"),
        )


@dataclass(frozen=True)
class CandidateEvaluation:
    """Independent preparation, planning and actual task execution evidence.

    Passing check names alone cannot replace normal program completion and
    measured task success. ``committed`` is set only by a confirmed host receipt.
    """

    candidate_id: str
    checks: tuple[FeasibilityCheck, ...]
    program_completed: bool = False
    measured_success: bool = False
    initial_state_id: str | None = None
    clearance_m: float = 0.0
    execution_cost_s: float = 0.0
    committed: bool = False
    evidence: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _identifier(self.candidate_id, "candidate_id")
        checks = tuple(self.checks)
        if any(not isinstance(v, FeasibilityCheck) for v in checks):
            raise TypeError("checks require FeasibilityCheck values")
        names([v.stage for v in checks], "check stages")
        object.__setattr__(self, "checks", checks)
        for attr in ("program_completed", "measured_success", "committed"):
            if type(getattr(self, attr)) is not bool:
                raise TypeError(f"{attr} must be boolean")
        clearance, cost = finite(
            (self.clearance_m, self.execution_cost_s), "quality metrics"
        )
        object.__setattr__(self, "clearance_m", clearance)
        object.__setattr__(self, "execution_cost_s", cost)
        if self.clearance_m < 0 or self.execution_cost_s < 0:
            raise ValueError("quality metrics must be nonnegative")
        if self.initial_state_id is not None:
            _identifier(self.initial_state_id, "initial_state_id")
        object.__setattr__(self, "evidence", owned_json(self.evidence))
        if self.committed and not self.accepted:
            raise ValueError("a rejected execution cannot be committed")

    @property
    def accepted(self) -> bool:
        """Return exact-scene measured success with every mandatory stage passed."""
        required = {"semantics", "geometry", "kinematics", "motion", "execution"}
        return (
            required <= {c.stage for c in self.checks}
            and all(c.status == "passed" for c in self.checks)
            and self.program_completed
            and self.measured_success
            and self.initial_state_id is not None
        )
