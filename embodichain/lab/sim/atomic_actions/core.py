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

"""Core semantic objects and planning contract for atomic actions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Generic, TYPE_CHECKING

import torch

from embodichain.lab.sim.common import BatchEntity
from embodichain.utils import configclass

from .affordance import Affordance
from .effects import StateDelta
from .goals import collect_scene_dependencies
from .invocation import ActionInvocation, GoalT
from .plans import (
    ActionPlan,
    CompletionCondition,
    CompletionConditionKind,
    PhaseSpec,
    PlannedPhase,
    PlannerDiagnostics,
    TimedTrajectory,
)

if TYPE_CHECKING:
    from embodichain.lab.sim.planners import MotionGenerator

    from .state import PlanningContext


def resolve_runtime_device(device: torch.device | str) -> torch.device:
    """Resolve an indexless CUDA device to the active concrete GPU index.

    Args:
        device: PyTorch device or device string.

    Returns:
        Concrete runtime device.
    """
    resolved = torch.device(device)
    if resolved.type == "cuda" and resolved.index is None:
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    return resolved


@dataclass
class ObjectSemantics:
    """Semantic and geometric information about an interaction object."""

    affordance: Affordance
    """Affordance data describing supported interactions."""

    geometry: dict[str, Any]
    """Non-affordance geometric metadata."""

    properties: dict[str, Any] = field(default_factory=dict)
    """Physical properties such as mass and friction."""

    label: str = "none"
    """Semantic object category."""

    entity: BatchEntity | None = None
    """Optional simulation entity used by deterministic grounding."""

    def __post_init__(self) -> None:
        if not isinstance(self.affordance, Affordance):
            raise TypeError("affordance must be an Affordance instance.")
        if not isinstance(self.geometry, dict):
            raise TypeError("geometry must be a dict.")
        if not isinstance(self.properties, dict):
            raise TypeError("properties must be a dict.")
        if not isinstance(self.label, str) or not self.label:
            raise ValueError("label must be a non-empty string.")
        self.affordance.object_label = self.label


@dataclass(frozen=True, slots=True)
class SkillDescriptor:
    """Machine-readable metadata for one registered atomic skill."""

    skill_id: str
    goal_type: type[Any] | tuple[type[Any], ...]
    manipulator_roles: tuple[str, ...] = ()
    end_effector_roles: tuple[str, ...] = ()
    agent_visible: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.skill_id, str) or not self.skill_id:
            raise ValueError("SkillDescriptor.skill_id must be non-empty.")
        goal_types = (
            self.goal_type if isinstance(self.goal_type, tuple) else (self.goal_type,)
        )
        if not goal_types or not all(isinstance(item, type) for item in goal_types):
            raise TypeError("SkillDescriptor.goal_type must contain concrete types.")
        for field_name in ("manipulator_roles", "end_effector_roles"):
            roles = tuple(getattr(self, field_name))
            if len(set(roles)) != len(roles) or not all(
                isinstance(role, str) and role for role in roles
            ):
                raise ValueError(f"{field_name} must contain unique non-empty roles.")
            object.__setattr__(self, field_name, roles)


@configclass
class ActionCfg:
    """Base configuration for implementation-owned skill behavior."""

    name: str = "default"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a non-empty string.")


class AtomicAction(Generic[GoalT], ABC):
    """Side-effect-free planner for one semantically meaningful robot skill."""

    skill_id: ClassVar[str]
    """Stable registry identifier for this skill."""

    GoalType: ClassVar[type[Any] | tuple[type[Any], ...]]
    """Concrete goal dataclass or dataclasses accepted by this skill."""

    manipulator_roles: ClassVar[tuple[str, ...]] = ("primary",)
    """Required semantic manipulator roles."""

    end_effector_roles: ClassVar[tuple[str, ...]] = ()
    """Required semantic end-effector roles."""

    agent_visible: ClassVar[bool] = True
    """Whether an Action Agent should expose this skill by default."""

    def __init__(
        self,
        motion_generator: MotionGenerator,
        cfg: ActionCfg | None = None,
    ) -> None:
        self.motion_generator = motion_generator
        self.cfg = cfg if cfg is not None else ActionCfg()
        self.robot = motion_generator.robot
        self.device = resolve_runtime_device(self.robot.device)

    @classmethod
    def descriptor(cls) -> SkillDescriptor:
        """Return stable metadata used by registries and Action Agent adapters."""
        return SkillDescriptor(
            skill_id=cls.skill_id,
            goal_type=cls.GoalType,
            manipulator_roles=cls.manipulator_roles,
            end_effector_roles=cls.end_effector_roles,
            agent_visible=cls.agent_visible,
        )

    def require_goal(self, invocation: ActionInvocation[GoalT]) -> GoalT:
        """Validate an invocation and return its concrete goal.

        Args:
            invocation: Grounded invocation to validate.

        Returns:
            Invocation goal narrowed to this action's declared type.

        Raises:
            ValueError: If the stable skill identifier does not match.
            TypeError: If the goal type is incompatible.
            KeyError: If a required binding role is missing.
        """
        if invocation.skill_id != self.skill_id:
            raise ValueError(
                f"Invocation skill_id {invocation.skill_id!r} does not match "
                f"{self.skill_id!r}."
            )
        if not isinstance(invocation.goal, self.GoalType):
            expected = (
                " | ".join(item.__name__ for item in self.GoalType)
                if isinstance(self.GoalType, tuple)
                else self.GoalType.__name__
            )
            raise TypeError(
                f"Skill {self.skill_id!r} expects goal {expected}, got "
                f"{type(invocation.goal).__name__}."
            )
        for role in self.manipulator_roles:
            invocation.binding.manipulator(role)
        for role in self.end_effector_roles:
            invocation.binding.end_effector(role)
        required_planner = invocation.motion_policy.planner
        configured_planner = getattr(
            getattr(self.motion_generator, "planner", None), "cfg", None
        )
        configured_planner_name = getattr(configured_planner, "planner_type", None)
        if required_planner is not None and required_planner != configured_planner_name:
            raise ValueError(
                f"Motion policy requires planner {required_planner!r}, but this "
                f"action uses {configured_planner_name!r}."
            )
        return invocation.goal

    def build_plan(
        self,
        invocation: ActionInvocation[GoalT],
        context: PlanningContext,
        *,
        success: bool | torch.Tensor,
        trajectory: TimedTrajectory | torch.Tensor,
        expected_effects: StateDelta | None = None,
        phase_name: str | None = None,
        replannable: bool = True,
        completion_kind: CompletionConditionKind = (
            CompletionConditionKind.TRAJECTORY_COMPLETE
        ),
        completion_tolerance: float | None = None,
        diagnostics: PlannerDiagnostics | None = None,
    ) -> ActionPlan:
        """Build a validated single-phase plan for a primitive implementation.

        Args:
            invocation: Grounded invocation being planned.
            context: Planning input used for the plan.
            success: Per-environment planning success or scalar planner result.
            trajectory: Full-robot timed trajectory or position tensor.
            expected_effects: Symbolic effects to verify after execution.
            phase_name: Optional phase name; defaults to the action config name.
            replannable: Whether the execution runtime may replan this phase.
            completion_kind: Completion condition category.
            completion_tolerance: Optional numerical completion tolerance.
            diagnostics: Optional retained planner diagnostics.

        Returns:
            Side-effect-free action plan.
        """
        self.require_goal(invocation)
        if isinstance(success, bool):
            success_mask = torch.full(
                (context.batch_size,),
                success,
                dtype=torch.bool,
                device=self.device,
            )
        elif isinstance(success, torch.Tensor):
            success_mask = success.to(device=self.device)
            if success_mask.dtype != torch.bool:
                raise TypeError("Planning success must have dtype torch.bool.")
            if success_mask.dim() == 0 or success_mask.shape == (1,):
                success_mask = success_mask.reshape(1).expand(context.batch_size)
            if success_mask.shape != (context.batch_size,):
                raise ValueError(
                    "Planning success must have shape "
                    f"({context.batch_size},), got {tuple(success_mask.shape)}."
                )
            success_mask = success_mask.clone()
        else:
            raise TypeError("Planning success must be bool or torch.Tensor.")

        if isinstance(trajectory, torch.Tensor):
            timed = TimedTrajectory.from_positions(
                trajectory,
                env_ids=context.env_ids,
                control_dt=invocation.motion_policy.control_dt,
            )
        elif isinstance(trajectory, TimedTrajectory):
            timed = trajectory
        else:
            raise TypeError("trajectory must be TimedTrajectory or torch.Tensor.")
        if timed.batch_size != context.batch_size:
            raise ValueError("Trajectory and planning context batch sizes must match.")
        if timed.robot_dof != context.robot.robot_dof:
            raise ValueError("Trajectory robot_dof must match the planning context.")

        if diagnostics is None:
            backend = getattr(
                getattr(getattr(self.motion_generator, "planner", None), "cfg", None),
                "planner_type",
                invocation.motion_policy.motion_source,
            )
            diagnostics = PlannerDiagnostics(backend=str(backend))
        phase = PlannedPhase(
            spec=PhaseSpec(
                name=phase_name or self.cfg.name,
                goal=invocation.goal,
                replannable=replannable,
                completion_condition=CompletionCondition(
                    kind=completion_kind,
                    tolerance=completion_tolerance,
                ),
                recovery_policy=invocation.recovery_policy,
                scene_dependencies=collect_scene_dependencies(invocation.goal),
            ),
            trajectory=timed,
            planned_scene_version=context.scene.version,
            diagnostics=diagnostics,
        )
        return ActionPlan(
            skill_id=self.skill_id,
            plan_success=success_mask,
            phases=(phase,),
            expected_effects=expected_effects or StateDelta(),
            invocation_id=invocation.invocation_id,
        )

    def failed_plan(
        self,
        invocation: ActionInvocation[GoalT],
        context: PlanningContext,
        *,
        message: str | None = None,
    ) -> ActionPlan:
        """Build a failed empty plan without changing task state.

        Args:
            invocation: Grounded invocation that failed to plan.
            context: Planning input used for the attempt.
            message: Optional diagnostic message.

        Returns:
            Failed action plan with an empty phase trajectory.
        """
        backend = getattr(
            getattr(getattr(self.motion_generator, "planner", None), "cfg", None),
            "planner_type",
            invocation.motion_policy.motion_source,
        )
        return self.build_plan(
            invocation,
            context,
            success=torch.zeros(
                context.batch_size, dtype=torch.bool, device=self.device
            ),
            trajectory=TimedTrajectory.empty(
                batch_size=context.batch_size,
                robot_dof=context.robot.robot_dof,
                device=self.device,
                env_ids=context.env_ids,
            ),
            replannable=True,
            diagnostics=PlannerDiagnostics(
                backend=str(backend), messages=(() if message is None else (message,))
            ),
        )

    @abstractmethod
    def plan(
        self,
        invocation: ActionInvocation[GoalT],
        context: PlanningContext,
    ) -> ActionPlan:
        """Plan one invocation without stepping simulation or committing state.

        Args:
            invocation: Fully typed and embodiment-bound action request.
            context: Latest observed robot, task, and scene state.

        Returns:
            Scene-bound action plan with expected, uncommitted effects.
        """


__all__ = [
    "ActionCfg",
    "AtomicAction",
    "ObjectSemantics",
    "SkillDescriptor",
    "resolve_runtime_device",
]
