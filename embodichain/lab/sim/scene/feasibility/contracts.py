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
import math
from typing import Literal

from .._validation import _identifier, _rigid_pose
from ..expansion import ScenePoseChange, SceneVariant
from ._values import count, digest, finite, names, owned_json, poses

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

_IDENTITY = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)


@dataclass(frozen=True)
class PoseSearchVariable:
    """One XY/yaw displacement applied to a scene group or robot installation.

    Args:
        name: Unique variable name.
        kind: ``scene`` or ``robot``; robot members are physical robot UIDs.
        reference_pose: Group-to-arena Z-up transform, including mounting height.
        lower: Lower XY metre / yaw radian displacement bounds.
        upper: Upper displacement bounds; equal endpoints fix a parameter.
        members: Member UID to member-to-group transforms. All members move
            together, preserving dual-arm mounting and support-group relations.
    """

    name: str
    kind: Literal["scene", "robot"]
    reference_pose: tuple
    lower: tuple[float, float, float]
    upper: tuple[float, float, float]
    members: Mapping[str, tuple]

    def __post_init__(self) -> None:
        _identifier(self.name, "variable name")
        if self.kind not in ("scene", "robot"):
            raise ValueError("variable kind must be scene or robot")
        object.__setattr__(self, "reference_pose", _rigid_pose(self.reference_pose))
        lo, hi = finite(self.lower, "lower"), finite(self.upper, "upper")
        if len(lo) != 3 or len(hi) != 3 or any(a > b for a, b in zip(lo, hi)):
            raise ValueError("bounds require three ordered XY/yaw coordinates")
        if hi[2] - lo[2] > 2 * math.pi + 1e-10:
            raise ValueError("yaw interval must span at most one turn")
        object.__setattr__(self, "lower", lo)
        object.__setattr__(self, "upper", hi)
        object.__setattr__(self, "members", poses(self.members))
        if not self.members:
            raise ValueError("variables require explicit member UIDs")


@dataclass(frozen=True)
class PhaseTarget:
    """One simultaneous TCP target in an explicit arena or entity frame.

    Args:
        control_part: Explicit solver name, preserving custom names.
        pose: TCP-to-frame rigid transform.
        frame_uid: Entity frame UID, or ``None`` for the arena frame.
    """

    control_part: str
    pose: tuple
    frame_uid: str | None = None

    def __post_init__(self) -> None:
        _identifier(self.control_part, "control_part")
        object.__setattr__(self, "pose", _rigid_pose(self.pose))
        if self.frame_uid is not None:
            _identifier(self.frame_uid, "frame_uid")


@dataclass(frozen=True)
class HeldObject:
    """Rigid attachment during travel; never a physical-success assertion.

    Args:
        entity_uid: Held object UID.
        control_part: Owner TCP/control part.
        tcp_from_object: Object-to-TCP transform used by collision FK.
    """

    entity_uid: str
    control_part: str
    tcp_from_object: tuple = _IDENTITY

    def __post_init__(self) -> None:
        _identifier(self.entity_uid, "held entity")
        _identifier(self.control_part, "held owner")
        object.__setattr__(self, "tcp_from_object", _rigid_pose(self.tcp_from_object))


@dataclass(frozen=True)
class TaskPhase:
    """A synchronized stage and its explicit planning-state transition.

    Args:
        phase_id: Stable stage identity.
        targets: Simultaneous targets, at most one per control part.
        dependencies: Predecessors; input order must be a topological order.
        held_objects: Attachments during incoming motion, one owner per object.
        allowed_contacts: Physical collision IDs allowed to touch in this stage.
        resulting_poses: Projected object states after this stage, used only for
            subsequent planning; runtime must still read fresh physical state.
        joint_targets: Explicit joint-name commands, e.g. open/closed fingers.
    """

    phase_id: str
    targets: tuple[PhaseTarget, ...] = ()
    dependencies: tuple[str, ...] = ()
    held_objects: tuple[HeldObject, ...] = ()
    allowed_contacts: tuple[tuple[str, str], ...] = ()
    resulting_poses: tuple[ScenePoseChange, ...] = ()
    joint_targets: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        _identifier(self.phase_id, "phase_id")
        for attr, cls in (
            ("targets", PhaseTarget),
            ("held_objects", HeldObject),
            ("resulting_poses", ScenePoseChange),
        ):
            values = tuple(getattr(self, attr))
            if any(not isinstance(v, cls) for v in values):
                raise TypeError(f"{attr} contains an invalid contract")
            object.__setattr__(self, attr, values)
        names([v.control_part for v in self.targets], "phase target parts")
        names([v.entity_uid for v in self.held_objects], "held objects")
        names([v.entity_uid for v in self.resulting_poses], "resulting poses")
        object.__setattr__(
            self, "dependencies", names(self.dependencies, "dependencies")
        )
        pairs = tuple(
            tuple(sorted(names(pair, "contact pair"))) for pair in self.allowed_contacts
        )
        if any(len(p) != 2 for p in pairs) or len(set(pairs)) != len(pairs):
            raise ValueError("allowed contacts require unique distinct pairs")
        object.__setattr__(self, "allowed_contacts", pairs)
        joints = tuple(
            (_identifier(k, "joint target"), finite((v,), "joint target")[0])
            for k, v in self.joint_targets
        )
        names([k for k, _ in joints], "joint targets")
        object.__setattr__(self, "joint_targets", joints)


@dataclass(frozen=True)
class TaskOperationMode:
    """One permitted grasp/assignment/handover branch of the original task."""

    mode_id: str
    phases: tuple[TaskPhase, ...]

    def __post_init__(self) -> None:
        _identifier(self.mode_id, "mode_id")
        values = tuple(self.phases)
        if not values or any(not isinstance(v, TaskPhase) for v in values):
            raise ValueError("modes require explicit task phases")
        names([v.phase_id for v in values], "phase IDs")
        seen = set()
        for phase in values:
            if not set(phase.dependencies) <= seen:
                raise ValueError("phases must be in dependency order without cycles")
            seen.add(phase.phase_id)
        object.__setattr__(self, "phases", values)


@dataclass(frozen=True)
class TaskFeasibilityRequest:
    """Versioned fixed semantics, robot, scene, and allowed joint search domain.

    Args:
        task_id: Semantic identity including original roles/goals/invariants.
        robot_profile_id: Robot, tool and calibrated configuration identity.
        policy_id: Planner/runtime policy version.
        validator_id: Measured acceptance version.
        parent_scene_id: Base scene revision/content identity.
        entity_poses: Authored arena poses, including every variable member.
        variables: Permitted scene and mounting changes.
        modes: Allowed operation modes, preserving resource constraints.
    """

    task_id: str
    robot_profile_id: str
    policy_id: str
    validator_id: str
    parent_scene_id: str
    entity_poses: Mapping[str, tuple]
    variables: tuple[PoseSearchVariable, ...]
    modes: tuple[TaskOperationMode, ...]

    def __post_init__(self) -> None:
        for key in (
            "task_id",
            "robot_profile_id",
            "policy_id",
            "validator_id",
            "parent_scene_id",
        ):
            _identifier(getattr(self, key), key)
        object.__setattr__(self, "entity_poses", poses(self.entity_poses))
        variables, modes = tuple(self.variables), tuple(self.modes)
        if any(not isinstance(v, PoseSearchVariable) for v in variables):
            raise TypeError("variables require PoseSearchVariable values")
        if not modes or any(not isinstance(v, TaskOperationMode) for v in modes):
            raise ValueError("request requires explicit operation modes")
        names([v.name for v in variables], "variable names")
        members = [k for v in variables for k in v.members]
        names(members, "variable members")
        if not set(members) <= self.entity_poses.keys():
            raise ValueError("variable members must exist in the authored scene")
        names([v.mode_id for v in modes], "mode IDs")
        for mode in modes:
            for phase in mode.phases:
                referenced = {
                    t.frame_uid for t in phase.targets if t.frame_uid is not None
                }
                referenced |= {h.entity_uid for h in phase.held_objects}
                referenced |= {p.entity_uid for p in phase.resulting_poses}
                if not referenced <= self.entity_poses.keys():
                    raise ValueError("phase refers to an undeclared entity")
        object.__setattr__(self, "variables", variables)
        object.__setattr__(self, "modes", modes)

    @cached_property
    def request_id(self) -> str:
        """Return the content identity of all fixed conditions and permissions."""
        return digest(self)


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
