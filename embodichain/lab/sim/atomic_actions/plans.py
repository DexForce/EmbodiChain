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

"""Side-effect-free plans produced by atomic actions."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import torch

from embodichain.lab.sim.planners.utils import normalize_success_mask

from .bindings import JointPositionTarget
from .effects import StateDelta
from .policies import RecoveryPolicy
from .runtime_commands import JointPositionPayload, TimedCommandSequence
from .state import PlanningContext


def _validate_optional_trajectory_field(
    value: torch.Tensor | None,
    positions: torch.Tensor,
    name: str,
) -> None:
    """Validate an optional velocity or acceleration tensor."""
    if value is None:
        return
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor or None.")
    if value.shape != positions.shape:
        raise ValueError(f"{name} must match positions shape.")
    if value.device != positions.device:
        raise ValueError(f"{name} must share the positions device.")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values.")


@dataclass(frozen=True, slots=True, eq=False)
class TimedTrajectory:
    """Full-robot joint trajectory with per-environment timing metadata."""

    positions: torch.Tensor
    velocities: torch.Tensor | None
    accelerations: torch.Tensor | None
    dt: torch.Tensor
    """Per-waypoint arrival intervals; the first sample normally has zero dt."""
    duration: torch.Tensor
    env_ids: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.positions, torch.Tensor):
            raise TypeError("positions must be a torch.Tensor.")
        if self.positions.dim() != 3:
            raise ValueError("positions must have shape (batch, waypoint, robot_dof).")
        if self.positions.shape[0] == 0 or self.positions.shape[2] == 0:
            raise ValueError(
                "positions batch and robot_dof dimensions must be non-zero."
            )
        if not torch.isfinite(self.positions).all():
            raise ValueError("positions must contain only finite values.")
        _validate_optional_trajectory_field(
            self.velocities, self.positions, "velocities"
        )
        _validate_optional_trajectory_field(
            self.accelerations, self.positions, "accelerations"
        )
        batch_size, waypoint_count, _ = self.positions.shape
        if not isinstance(self.dt, torch.Tensor) or self.dt.shape != (
            batch_size,
            waypoint_count,
        ):
            raise ValueError(f"dt must have shape ({batch_size}, {waypoint_count}).")
        if self.dt.device != self.positions.device:
            raise ValueError("dt must share the positions device.")
        if not torch.isfinite(self.dt).all() or (self.dt < 0).any():
            raise ValueError("dt must contain finite non-negative values.")
        if not isinstance(self.duration, torch.Tensor) or self.duration.shape != (
            batch_size,
        ):
            raise ValueError(f"duration must have shape ({batch_size},).")
        if self.duration.device != self.positions.device:
            raise ValueError("duration must share the positions device.")
        if not torch.isfinite(self.duration).all() or (self.duration < 0).any():
            raise ValueError("duration must contain finite non-negative values.")
        if not torch.allclose(
            self.duration,
            self.dt.sum(dim=1),
            rtol=1e-4,
            atol=1e-6,
        ):
            raise ValueError("duration must equal the sum of dt for each environment.")
        if not isinstance(self.env_ids, torch.Tensor):
            raise TypeError("env_ids must be a torch.Tensor.")
        if self.env_ids.dtype != torch.long or self.env_ids.shape != (batch_size,):
            raise ValueError(f"env_ids must be int64 with shape ({batch_size},).")
        if self.env_ids.device != self.positions.device:
            raise ValueError("env_ids must share the positions device.")
        if torch.unique(self.env_ids).numel() != batch_size:
            raise ValueError("env_ids must contain unique values.")
        object.__setattr__(self, "positions", self.positions.detach().clone())
        object.__setattr__(
            self,
            "velocities",
            None if self.velocities is None else self.velocities.detach().clone(),
        )
        object.__setattr__(
            self,
            "accelerations",
            (
                None
                if self.accelerations is None
                else self.accelerations.detach().clone()
            ),
        )
        object.__setattr__(self, "dt", self.dt.detach().clone())
        object.__setattr__(self, "duration", self.duration.detach().clone())
        object.__setattr__(self, "env_ids", self.env_ids.detach().clone())

    @property
    def batch_size(self) -> int:
        """Number of environment rows."""
        return int(self.positions.shape[0])

    @property
    def waypoint_count(self) -> int:
        """Number of trajectory samples."""
        return int(self.positions.shape[1])

    @property
    def robot_dof(self) -> int:
        """Number of full-robot command columns."""
        return int(self.positions.shape[2])

    def snapshot(self) -> TimedTrajectory:
        """Return an independently owned copy of this trajectory.

        Returns:
            A trajectory whose tensor storage can be mutated without changing
            the source trajectory.
        """
        return TimedTrajectory(
            positions=self.positions.clone(),
            velocities=(None if self.velocities is None else self.velocities.clone()),
            accelerations=(
                None if self.accelerations is None else self.accelerations.clone()
            ),
            dt=self.dt.clone(),
            duration=self.duration.clone(),
            env_ids=self.env_ids.clone(),
        )

    @classmethod
    def from_positions(
        cls,
        positions: torch.Tensor,
        *,
        env_ids: torch.Tensor,
        control_dt: float,
        velocities: torch.Tensor | None = None,
        accelerations: torch.Tensor | None = None,
        dt: torch.Tensor | None = None,
        duration: torch.Tensor | float | None = None,
    ) -> TimedTrajectory:
        """Build a timed trajectory and synthesize missing timing metadata.

        Args:
            positions: Full-robot positions, shape ``(B, N, D)``.
            env_ids: Environment identifiers, shape ``(B,)``.
            control_dt: Fallback interval used when ``dt`` is absent.
            velocities: Optional joint velocities.
            accelerations: Optional joint accelerations.
            dt: Optional per-sample time deltas.
            duration: Optional duration used to synthesize or validate ``dt``.

        Returns:
            Validated timed trajectory.
        """
        if control_dt <= 0.0:
            raise ValueError("control_dt must be greater than zero.")
        if not isinstance(positions, torch.Tensor) or positions.dim() != 3:
            raise ValueError("positions must have shape (B, N, D).")
        batch_size, waypoint_count, _ = positions.shape
        if dt is None:
            dt = torch.zeros(
                (batch_size, waypoint_count),
                dtype=torch.float32,
                device=positions.device,
            )
            if waypoint_count > 1:
                if duration is None:
                    dt[:, 1:] = control_dt
                else:
                    duration_tensor = torch.as_tensor(
                        duration, dtype=torch.float32, device=positions.device
                    )
                    if duration_tensor.dim() == 0:
                        duration_tensor = duration_tensor.expand(batch_size)
                    if duration_tensor.shape != (batch_size,):
                        raise ValueError(f"duration must have shape ({batch_size},).")
                    dt[:, 1:] = duration_tensor[:, None] / (waypoint_count - 1)
        else:
            dt = dt.to(device=positions.device, dtype=torch.float32)
        computed_duration = dt.sum(dim=1)
        if duration is not None:
            duration_tensor = torch.as_tensor(
                duration, dtype=torch.float32, device=positions.device
            )
            if duration_tensor.dim() == 0:
                duration_tensor = duration_tensor.expand(batch_size)
            if duration_tensor.shape != (batch_size,):
                raise ValueError(f"duration must have shape ({batch_size},).")
            if not torch.allclose(
                computed_duration, duration_tensor, rtol=1e-4, atol=1e-6
            ):
                raise ValueError("duration does not match the supplied dt.")
        return cls(
            positions=positions,
            velocities=velocities,
            accelerations=accelerations,
            dt=dt,
            duration=computed_duration,
            env_ids=env_ids,
        )

    @classmethod
    def empty(
        cls,
        *,
        batch_size: int,
        robot_dof: int,
        device: torch.device | str,
        env_ids: torch.Tensor,
    ) -> TimedTrajectory:
        """Create an empty trajectory with explicit batch and DoF dimensions."""
        resolved = torch.device(device)
        return cls(
            positions=torch.empty(
                (batch_size, 0, robot_dof), dtype=torch.float32, device=resolved
            ),
            velocities=None,
            accelerations=None,
            dt=torch.empty((batch_size, 0), dtype=torch.float32, device=resolved),
            duration=torch.zeros(batch_size, dtype=torch.float32, device=resolved),
            env_ids=env_ids,
        )

    def hold_rows(
        self,
        active_mask: torch.Tensor,
        hold_qpos: torch.Tensor,
    ) -> TimedTrajectory:
        """Replace inactive rows with a fixed hold command.

        Args:
            active_mask: Rows allowed to execute this trajectory.
            hold_qpos: Hold positions, shape ``(B, D)``.

        Returns:
            New trajectory with inactive rows frozen and derivatives zeroed.
        """
        if active_mask.dtype != torch.bool or active_mask.shape != (self.batch_size,):
            raise ValueError("active_mask must be bool with shape (batch_size,).")
        if hold_qpos.shape != (self.batch_size, self.robot_dof):
            raise ValueError("hold_qpos must have shape (batch_size, robot_dof).")
        active_mask = active_mask.to(self.positions.device)
        held = (
            hold_qpos.to(
                device=self.positions.device,
                dtype=self.positions.dtype,
            )
            .unsqueeze(1)
            .expand_as(self.positions)
        )
        positions = torch.where(active_mask[:, None, None], self.positions, held)

        def mask_derivative(value: torch.Tensor | None) -> torch.Tensor | None:
            if value is None:
                return None
            return torch.where(
                active_mask[:, None, None], value, torch.zeros_like(value)
            )

        return TimedTrajectory(
            positions=positions,
            velocities=mask_derivative(self.velocities),
            accelerations=mask_derivative(self.accelerations),
            dt=self.dt,
            duration=self.duration,
            env_ids=self.env_ids,
        )

    @classmethod
    def concatenate(
        cls,
        trajectories: Sequence[TimedTrajectory],
        *,
        empty_like: PlanningContext | None = None,
    ) -> TimedTrajectory:
        """Concatenate trajectories along their waypoint dimension.

        Args:
            trajectories: Compatible trajectories in execution order.
            empty_like: Context used only when ``trajectories`` is empty.

        Returns:
            Concatenated full-robot trajectory.
        """
        if not trajectories:
            if empty_like is None:
                raise ValueError("empty_like is required for an empty concatenation.")
            return cls.empty(
                batch_size=empty_like.batch_size,
                robot_dof=empty_like.robot.robot_dof,
                device=empty_like.robot.qpos.device,
                env_ids=empty_like.env_ids,
            )
        first = trajectories[0]
        for trajectory in trajectories[1:]:
            if trajectory.batch_size != first.batch_size:
                raise ValueError("All trajectories must share a batch size.")
            if trajectory.robot_dof != first.robot_dof:
                raise ValueError("All trajectories must share robot_dof.")
            if not torch.equal(trajectory.env_ids, first.env_ids):
                raise ValueError("All trajectories must share env_ids.")
            if trajectory.positions.device != first.positions.device:
                raise ValueError("All trajectories must share a device.")

        def concatenate_optional(name: str) -> torch.Tensor | None:
            values = [getattr(item, name) for item in trajectories]
            if any(value is None for value in values):
                return None
            return torch.cat(values, dim=1)  # type: ignore[arg-type]

        dt = torch.cat([item.dt for item in trajectories], dim=1)
        return cls(
            positions=torch.cat([item.positions for item in trajectories], dim=1),
            velocities=concatenate_optional("velocities"),
            accelerations=concatenate_optional("accelerations"),
            dt=dt,
            duration=dt.sum(dim=1),
            env_ids=first.env_ids,
        )


@dataclass(frozen=True, slots=True)
class PlannerDiagnostics:
    """Planner metadata retained for debugging and recovery decisions."""

    backend: str
    messages: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.backend, str) or not self.backend:
            raise ValueError("PlannerDiagnostics.backend must be non-empty.")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("PlannerDiagnostics.metadata must be a mapping.")
        messages = tuple(self.messages)
        if not all(type(message) is str for message in messages):
            raise TypeError("PlannerDiagnostics.messages must contain strings.")
        object.__setattr__(self, "messages", messages)
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(deepcopy(dict(self.metadata))),
        )


class ExecutionFeedbackMode(str, Enum):
    """Feedback contract used to decide whether an action reached its target."""

    JOINT_POSITION = "joint_position"
    TIMED = "timed"


@dataclass(frozen=True, slots=True)
class EffectVerificationRequirement:
    """Explicit physical-effect verification independent of symbolic state.

    Presence of this value on an :class:`ActionPlan` forces a terminal effect
    boundary even when the plan declares no :class:`StateDelta`. The open
    ``kind`` identifier lets an external runtime select an appropriate
    verifier without placing backend-specific callbacks in the core plan.

    Args:
        kind: Stable, non-empty discriminator for the physical effect.
    """

    kind: str

    def __post_init__(self) -> None:
        if (
            type(self.kind) is not str
            or not self.kind
            or self.kind != self.kind.strip()
        ):
            raise ValueError(
                "kind must be a non-empty string without outer whitespace."
            )

    def snapshot(self) -> EffectVerificationRequirement:
        """Return an independently owned requirement value."""
        return EffectVerificationRequirement(kind=self.kind)


@dataclass(frozen=True, slots=True)
class TrajectorySegment:
    """Named half-open waypoint range inside an action trajectory.

    Segments describe semantic structure for inspection, visualization, and
    execution tracing. They do not create independent planning or recovery
    boundaries; recovery continues to operate on the enclosing action plan.
    """

    name: str
    start: int
    stop: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("TrajectorySegment.name must be non-empty.")
        if isinstance(self.start, bool) or not isinstance(self.start, int):
            raise TypeError("TrajectorySegment.start must be an integer.")
        if isinstance(self.stop, bool) or not isinstance(self.stop, int):
            raise TypeError("TrajectorySegment.stop must be an integer.")
        if self.start < 0:
            raise ValueError("TrajectorySegment.start must be non-negative.")
        if self.stop <= self.start:
            raise ValueError("TrajectorySegment.stop must be greater than start.")

    @property
    def waypoint_count(self) -> int:
        """Number of waypoints in this segment."""
        return self.stop - self.start

    def contains(self, waypoint_index: int) -> bool:
        """Return whether ``waypoint_index`` belongs to this segment."""
        return self.start <= waypoint_index < self.stop


@dataclass(frozen=True, slots=True, eq=False)
class ActionPlan:
    """Scene-bound planning result for one grounded atomic action invocation.

    An action owns one timed command sequence and one recovery boundary. Named
    :class:`TrajectorySegment` values describe semantic structure within that
    sequence without implying independent planning or recovery boundaries.

    Attributes:
        scene_dependency_monitor_until: Optional exclusive waypoint-index upper
            bounds for individual ``scene_dependencies``. An entity is monitored
            while the current waypoint index is smaller than its bound; ``0``
            disables monitoring immediately, while an omitted entity remains
            monitored for the action's full execution. Once the bound is reached,
            all pose changes for that entity are ignored, regardless of whether
            they were caused by the action or by an external disturbance.
    """

    skill_id: str
    plan_success: torch.Tensor
    commands: TimedCommandSequence
    recovery_policy: RecoveryPolicy
    planned_scene_version: int
    planned_collision_world_revision: tuple[int, ...]
    diagnostics: PlannerDiagnostics
    feedback_mode: ExecutionFeedbackMode = ExecutionFeedbackMode.TIMED
    joint_trajectory: TimedTrajectory | None = None
    segments: tuple[TrajectorySegment, ...] = ()
    scene_dependencies: tuple[str, ...] = ()
    scene_dependency_monitor_until: Mapping[str, int] = field(default_factory=dict)
    collision_world_sensitive: bool = False
    replannable: bool = True
    expected_effects: StateDelta = field(default_factory=StateDelta)
    effect_verification: EffectVerificationRequirement | None = None
    invocation_id: str | None = None
    invocation_revision: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.skill_id, str) or not self.skill_id:
            raise ValueError("ActionPlan.skill_id must be non-empty.")
        if (
            not isinstance(self.invocation_revision, int)
            or self.invocation_revision < 0
        ):
            raise ValueError("invocation_revision must be a non-negative integer.")
        if not isinstance(self.plan_success, torch.Tensor):
            raise TypeError("plan_success must be a torch.Tensor.")
        if self.plan_success.dtype != torch.bool or self.plan_success.dim() != 1:
            raise ValueError("plan_success must be a 1D bool tensor.")
        if not isinstance(self.commands, TimedCommandSequence):
            raise TypeError("commands must be a TimedCommandSequence.")
        if self.commands.batch_size != self.plan_success.shape[0]:
            raise ValueError("plan_success batch must match the command sequence.")
        if self.commands.device != self.plan_success.device:
            raise ValueError("plan_success and commands must share a device.")
        if not isinstance(self.feedback_mode, ExecutionFeedbackMode):
            raise TypeError("feedback_mode must be an ExecutionFeedbackMode.")
        expected_target_types: dict[tuple[str, str], type[object]] | None = None
        expected_target_fingerprints: dict[tuple[str, str], object] | None = None
        for frame_index, frame in enumerate(self.commands.frames):
            target_types = {
                command.destination_key: type(command.target)
                for command in frame.commands
            }
            target_fingerprints = {
                command.destination_key: command.target.address_fingerprint
                for command in frame.commands
            }
            if expected_target_types is None:
                expected_target_types = target_types
                expected_target_fingerprints = target_fingerprints
                continue
            if target_types.keys() != expected_target_types.keys():
                raise ValueError(
                    "ActionPlan command frames must preserve the same destination "
                    f"set; frame {frame_index} differs from frame 0."
                )
            mismatched_types = sorted(
                destination
                for destination, target_type in target_types.items()
                if target_type is not expected_target_types[destination]
            )
            if mismatched_types:
                raise ValueError(
                    "ActionPlan command frames must preserve the exact target type "
                    f"for each destination; frame {frame_index} differs at "
                    f"{mismatched_types}."
                )
            assert expected_target_fingerprints is not None
            mismatched_fingerprints = sorted(
                destination
                for destination, fingerprint in target_fingerprints.items()
                if fingerprint != expected_target_fingerprints[destination]
            )
            if mismatched_fingerprints:
                raise ValueError(
                    "ActionPlan command frames must preserve the target address "
                    f"fingerprint for each destination; frame {frame_index} "
                    f"differs at {mismatched_fingerprints}."
                )
        if self.joint_trajectory is not None:
            if not isinstance(self.joint_trajectory, TimedTrajectory):
                raise TypeError("joint_trajectory must be a TimedTrajectory or None.")
            if self.joint_trajectory.batch_size != self.commands.batch_size:
                raise ValueError(
                    "joint_trajectory batch must match the command sequence."
                )
            if self.joint_trajectory.waypoint_count != self.commands.frame_count:
                raise ValueError(
                    "joint_trajectory waypoints must match command sequence frames."
                )
            if not torch.equal(self.joint_trajectory.env_ids, self.commands.env_ids):
                raise ValueError(
                    "joint_trajectory env_ids must match the command sequence."
                )
            if self.joint_trajectory.positions.device != self.commands.device:
                raise ValueError("joint_trajectory and commands must share a device.")
        if (
            self.feedback_mode is ExecutionFeedbackMode.JOINT_POSITION
            and self.joint_trajectory is None
        ):
            raise ValueError(
                "joint_position feedback requires an owned joint_trajectory."
            )
        if self.feedback_mode is ExecutionFeedbackMode.JOINT_POSITION:
            if bool(self.plan_success.any().item()) and self.commands.frame_count == 0:
                raise ValueError(
                    "joint_position feedback requires command frames when any "
                    "environment planned successfully."
                )
            assert self.joint_trajectory is not None
            expected_destinations: dict[tuple[str, str], tuple[int, ...]] | None = None
            for frame_index, frame in enumerate(self.commands.frames):
                if not frame.commands:
                    raise ValueError(
                        "joint_position feedback requires at least one endpoint "
                        f"command in frame {frame_index}."
                    )
                if any(
                    not isinstance(command.target, JointPositionTarget)
                    or not isinstance(command.payload, JointPositionPayload)
                    for command in frame.commands
                ):
                    raise ValueError(
                        "joint_position feedback accepts only JointPositionTarget "
                        "and JointPositionPayload commands."
                    )
                for command in frame.commands:
                    target = command.target
                    payload = command.payload
                    assert isinstance(target, JointPositionTarget)
                    assert isinstance(payload, JointPositionPayload)
                    if any(
                        joint_id >= self.joint_trajectory.robot_dof
                        for joint_id in target.joint_ids
                    ):
                        raise ValueError(
                            f"Joint target {command.destination_key} contains joint "
                            "IDs outside joint_trajectory robot_dof "
                            f"{self.joint_trajectory.robot_dof}."
                        )
                    joint_ids = list(target.joint_ids)
                    expected_positions = self.joint_trajectory.positions[
                        :, frame_index, joint_ids
                    ]
                    if (
                        payload.positions.dtype != expected_positions.dtype
                        or not torch.equal(payload.positions, expected_positions)
                    ):
                        raise ValueError(
                            f"Joint payload positions for {command.destination_key} "
                            "must exactly match the corresponding joint_trajectory "
                            f"slice at frame {frame_index}."
                        )
                    trajectory_velocities = self.joint_trajectory.velocities
                    if (payload.velocities is None) != (trajectory_velocities is None):
                        raise ValueError(
                            f"Joint payload velocities for {command.destination_key} "
                            "must have the same presence as joint_trajectory "
                            "velocities."
                        )
                    if (
                        payload.velocities is not None
                        and trajectory_velocities is not None
                    ):
                        expected_velocities = trajectory_velocities[
                            :, frame_index, joint_ids
                        ]
                        if (
                            payload.velocities.dtype != expected_velocities.dtype
                            or not torch.equal(
                                payload.velocities,
                                expected_velocities,
                            )
                        ):
                            raise ValueError(
                                "Joint payload velocities for "
                                f"{command.destination_key} must exactly match the "
                                "corresponding joint_trajectory slice at frame "
                                f"{frame_index}."
                            )
                destinations = {
                    command.destination_key: command.target.joint_ids
                    for command in frame.commands
                    if isinstance(command.target, JointPositionTarget)
                }
                if expected_destinations is None:
                    expected_destinations = destinations
                elif destinations != expected_destinations:
                    raise ValueError(
                        "joint_position feedback requires a stable joint endpoint "
                        "set across every command frame."
                    )
        if not isinstance(self.recovery_policy, RecoveryPolicy):
            raise TypeError("recovery_policy must be a RecoveryPolicy.")
        if self.planned_scene_version < 0:
            raise ValueError("planned_scene_version must be non-negative.")
        revisions = tuple(self.planned_collision_world_revision)
        if len(revisions) != self.commands.batch_size:
            raise ValueError(
                "planned_collision_world_revision must contain one value per "
                "command-sequence environment."
            )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in revisions
        ):
            raise ValueError(
                "planned_collision_world_revision must contain non-negative "
                "integers."
            )
        if not isinstance(self.diagnostics, PlannerDiagnostics):
            raise TypeError("diagnostics must be PlannerDiagnostics.")
        dependencies = tuple(self.scene_dependencies)
        if len(set(dependencies)) != len(dependencies) or not all(
            isinstance(entity_id, str) and entity_id for entity_id in dependencies
        ):
            raise ValueError(
                "scene_dependencies must contain unique non-empty entity ids."
            )
        waypoint_count = self.commands.frame_count
        monitor_until = dict(self.scene_dependency_monitor_until)
        if not set(monitor_until).issubset(dependencies):
            raise ValueError(
                "scene_dependency_monitor_until keys must be scene dependencies."
            )
        for entity_id, waypoint_index in monitor_until.items():
            if (
                type(entity_id) is not str
                or not entity_id
                or type(waypoint_index) is not int
                or not 0 <= waypoint_index <= waypoint_count
            ):
                raise ValueError(
                    "scene_dependency_monitor_until must map non-empty entity IDs "
                    "to waypoint indices within the command sequence."
                )
        if not isinstance(self.collision_world_sensitive, bool):
            raise TypeError("collision_world_sensitive must be a bool.")
        if not isinstance(self.replannable, bool):
            raise TypeError("replannable must be a bool.")
        if not isinstance(self.expected_effects, StateDelta):
            raise TypeError("expected_effects must be a StateDelta.")
        if (
            self.effect_verification is not None
            and type(self.effect_verification) is not EffectVerificationRequirement
        ):
            raise TypeError(
                "effect_verification must be exactly "
                "EffectVerificationRequirement or None."
            )
        segments = tuple(self.segments)
        if not all(isinstance(segment, TrajectorySegment) for segment in segments):
            raise TypeError("segments must contain only TrajectorySegment values.")
        if not segments and waypoint_count > 0:
            segments = (TrajectorySegment(self.skill_id, 0, waypoint_count),)
        names = [segment.name for segment in segments]
        if len(set(names)) != len(names):
            raise ValueError("ActionPlan segment names must be unique.")
        if waypoint_count == 0:
            if segments:
                raise ValueError("An empty command sequence cannot contain segments.")
        elif (
            not segments
            or segments[0].start != 0
            or segments[-1].stop != waypoint_count
            or any(
                previous.stop != current.start
                for previous, current in zip(segments, segments[1:])
            )
        ):
            raise ValueError(
                "ActionPlan segments must cover the command sequence exactly without "
                "gaps or overlaps."
            )
        object.__setattr__(self, "plan_success", self.plan_success.clone())
        object.__setattr__(self, "commands", self.commands.snapshot())
        object.__setattr__(
            self,
            "joint_trajectory",
            (
                None
                if self.joint_trajectory is None
                else self.joint_trajectory.snapshot()
            ),
        )
        object.__setattr__(self, "planned_collision_world_revision", revisions)
        object.__setattr__(
            self,
            "diagnostics",
            PlannerDiagnostics(
                backend=self.diagnostics.backend,
                messages=self.diagnostics.messages,
                metadata=self.diagnostics.metadata,
            ),
        )
        object.__setattr__(self, "scene_dependencies", dependencies)
        object.__setattr__(
            self,
            "scene_dependency_monitor_until",
            MappingProxyType(monitor_until),
        )
        object.__setattr__(self, "segments", segments)
        object.__setattr__(
            self,
            "effect_verification",
            (
                None
                if self.effect_verification is None
                else self.effect_verification.snapshot()
            ),
        )

    @property
    def success_all(self) -> bool:
        """Whether every environment row planned successfully."""
        return bool(self.plan_success.all().item())

    def snapshot(self) -> ActionPlan:
        """Return an independently owned inspection snapshot of this plan.

        Runtime tracing and visualization need access to the exact plan that
        reached an execution boundary without being able to mutate the live
        session.  Reconstructing the value through the public constructor also
        re-applies every plan invariant and snapshots all tensor-owning nested
        contracts.

        Returns:
            A validated plan with independently owned tensor storage.
        """
        return ActionPlan(
            skill_id=self.skill_id,
            plan_success=self.plan_success,
            commands=self.commands,
            recovery_policy=self.recovery_policy,
            planned_scene_version=self.planned_scene_version,
            planned_collision_world_revision=self.planned_collision_world_revision,
            diagnostics=self.diagnostics,
            feedback_mode=self.feedback_mode,
            joint_trajectory=self.joint_trajectory,
            segments=self.segments,
            scene_dependencies=self.scene_dependencies,
            scene_dependency_monitor_until=self.scene_dependency_monitor_until,
            collision_world_sensitive=self.collision_world_sensitive,
            replannable=self.replannable,
            expected_effects=self.expected_effects,
            effect_verification=self.effect_verification,
            invocation_id=self.invocation_id,
            invocation_revision=self.invocation_revision,
        )

    @property
    def requires_effect_verification(self) -> bool:
        """Whether execution must verify a terminal physical effect."""
        return (
            self.effect_verification is not None or not self.expected_effects.is_empty
        )

    def segment(self, name: str) -> TrajectorySegment:
        """Return a named trajectory segment.

        Args:
            name: Exact segment name.

        Returns:
            Matching segment metadata.

        Raises:
            KeyError: If the plan has no segment with that name.
        """
        for segment in self.segments:
            if segment.name == name:
                return segment
        raise KeyError(f"Action plan {self.skill_id!r} has no segment {name!r}.")

    def segment_at(self, waypoint_index: int) -> TrajectorySegment:
        """Return the segment containing a global action waypoint index."""
        if waypoint_index < 0 or waypoint_index >= self.commands.frame_count:
            raise IndexError(
                f"waypoint_index {waypoint_index} is outside the action command "
                "sequence."
            )
        for segment in self.segments:
            if segment.contains(waypoint_index):
                return segment
        raise RuntimeError("Validated action plan has no segment for waypoint index.")


@dataclass(frozen=True, slots=True, eq=False)
class CompiledTrajectory:
    """Offline compilation result for a sequence of action invocations."""

    plan_success: torch.Tensor
    trajectory: TimedTrajectory
    action_plans: tuple[ActionPlan, ...]
    projected_context: PlanningContext

    def __post_init__(self) -> None:
        if self.plan_success.dtype != torch.bool or self.plan_success.shape != (
            self.trajectory.batch_size,
        ):
            raise ValueError("Compiled plan_success must be bool with shape (batch,).")
        if self.plan_success.device != self.trajectory.positions.device:
            raise ValueError(
                "Compiled plan_success and trajectory must share a device."
            )
        object.__setattr__(self, "plan_success", self.plan_success.clone())
        object.__setattr__(self, "action_plans", tuple(self.action_plans))

    def action_waypoint_offset(self, action_index: int) -> int:
        """Return the global waypoint offset of one compiled action."""
        if action_index < 0 or action_index >= len(self.action_plans):
            raise IndexError(
                f"action_index {action_index} is outside the compiled sequence."
            )
        return sum(
            (
                0
                if plan.joint_trajectory is None
                else plan.joint_trajectory.waypoint_count
            )
            for plan in self.action_plans[:action_index]
        )

    def segment(self, action_index: int, name: str) -> TrajectorySegment:
        """Return action segment metadata shifted into compiled coordinates."""
        offset = self.action_waypoint_offset(action_index)
        local = self.action_plans[action_index].segment(name)
        return TrajectorySegment(
            name=local.name,
            start=offset + local.start,
            stop=offset + local.stop,
        )


__all__ = [
    "ActionPlan",
    "CompiledTrajectory",
    "EffectVerificationRequirement",
    "ExecutionFeedbackMode",
    "PlannerDiagnostics",
    "TimedTrajectory",
    "TrajectorySegment",
    "normalize_success_mask",
]
