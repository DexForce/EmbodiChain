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

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import torch

from .effects import StateDelta
from .goals import ActionGoal
from .policies import RecoveryPolicy
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
        object.__setattr__(self, "env_ids", self.env_ids.clone())

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
            hold_qpos.to(self.positions.device).unsqueeze(1).expand_as(self.positions)
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


class CompletionConditionKind(str, Enum):
    """Built-in phase completion-condition categories."""

    TRAJECTORY_COMPLETE = "trajectory_complete"
    JOINT_GOAL_REACHED = "joint_goal_reached"
    EEF_GOAL_REACHED = "eef_goal_reached"
    EFFECT_VERIFIED = "effect_verified"


@dataclass(frozen=True, slots=True)
class CompletionCondition:
    """Declarative phase completion condition."""

    kind: CompletionConditionKind = CompletionConditionKind.TRAJECTORY_COMPLETE
    tolerance: float | None = None

    def __post_init__(self) -> None:
        if self.tolerance is not None and self.tolerance <= 0.0:
            raise ValueError("Completion-condition tolerance must be positive.")


@dataclass(frozen=True, slots=True)
class PlannerDiagnostics:
    """Planner metadata retained for debugging and recovery decisions."""

    backend: str
    messages: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.backend, str) or not self.backend:
            raise ValueError("PlannerDiagnostics.backend must be non-empty.")
        object.__setattr__(self, "messages", tuple(self.messages))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class PhaseSpec:
    """Semantic and runtime contract for one sequential action phase."""

    name: str
    goal: ActionGoal
    replannable: bool
    completion_condition: CompletionCondition
    recovery_policy: RecoveryPolicy
    scene_dependencies: tuple[str, ...] = ()
    """Scene entities whose motion can invalidate this phase plan."""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("PhaseSpec.name must be non-empty.")
        dependencies = tuple(self.scene_dependencies)
        if len(set(dependencies)) != len(dependencies) or not all(
            isinstance(entity_id, str) and entity_id for entity_id in dependencies
        ):
            raise ValueError(
                "scene_dependencies must contain unique non-empty entity ids."
            )
        object.__setattr__(self, "scene_dependencies", dependencies)


@dataclass(frozen=True, slots=True)
class PlannedPhase:
    """One scene-bound phase trajectory and its diagnostics."""

    spec: PhaseSpec
    trajectory: TimedTrajectory
    planned_scene_version: int
    diagnostics: PlannerDiagnostics

    def __post_init__(self) -> None:
        if self.planned_scene_version < 0:
            raise ValueError("planned_scene_version must be non-negative.")


@dataclass(frozen=True, slots=True, eq=False)
class ActionPlan:
    """Planning result for one grounded atomic action invocation."""

    skill_id: str
    plan_success: torch.Tensor
    phases: tuple[PlannedPhase, ...]
    expected_effects: StateDelta = field(default_factory=StateDelta)
    invocation_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.skill_id, str) or not self.skill_id:
            raise ValueError("ActionPlan.skill_id must be non-empty.")
        if not isinstance(self.plan_success, torch.Tensor):
            raise TypeError("plan_success must be a torch.Tensor.")
        if self.plan_success.dtype != torch.bool or self.plan_success.dim() != 1:
            raise ValueError("plan_success must be a 1D bool tensor.")
        if not self.phases:
            raise ValueError("ActionPlan must contain at least one phase.")
        phases = tuple(self.phases)
        first = phases[0].trajectory
        if first.batch_size != self.plan_success.shape[0]:
            raise ValueError("plan_success batch must match phase trajectories.")
        if first.positions.device != self.plan_success.device:
            raise ValueError("plan_success and phase trajectories must share a device.")
        for phase in phases[1:]:
            trajectory = phase.trajectory
            if trajectory.batch_size != first.batch_size:
                raise ValueError("All action phases must share a batch size.")
            if trajectory.robot_dof != first.robot_dof:
                raise ValueError("All action phases must share robot_dof.")
            if not torch.equal(trajectory.env_ids, first.env_ids):
                raise ValueError("All action phases must share env_ids.")
        object.__setattr__(self, "plan_success", self.plan_success.clone())
        object.__setattr__(self, "phases", phases)

    @property
    def trajectory(self) -> TimedTrajectory:
        """Concatenate the sequential phase trajectories."""
        return TimedTrajectory.concatenate(
            tuple(phase.trajectory for phase in self.phases)
        )

    @property
    def success_all(self) -> bool:
        """Whether every environment row planned successfully."""
        return bool(self.plan_success.all().item())


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


__all__ = [
    "ActionPlan",
    "CompiledTrajectory",
    "CompletionCondition",
    "CompletionConditionKind",
    "PhaseSpec",
    "PlannedPhase",
    "PlannerDiagnostics",
    "TimedTrajectory",
]
